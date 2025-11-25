from __future__ import annotations

import json
import os
import re
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Mapping
import multiprocessing

from astraia.cli import ensure_env_keys, ping_llm_provider
from astraia.config import OptimizationConfig
from astraia.optimization import run_optimization
from astraia.tracking import RunMetadata, create_run, list_runs as tracking_list_runs
from astraia.tracking import load_run as tracking_load_run
from astraia.tracking import update_run_status

from .config_service import validate_config, get_config_root


@dataclass
class RunOptions:
    """Common run overrides coming from the GUI."""

    max_trials: int | None = None
    sampler: str | None = None
    llm_enabled: bool | None = None


@dataclass
class DryRunResult:
    config_path: Path
    run_id: str
    config: dict[str, Any]


@dataclass
class RunLaunch:
    run_id: str
    run_dir: Path
    meta_path: Path
    status: str


@dataclass
class JobRecord:
    run_id: str
    process: multiprocessing.Process
    status_queue: multiprocessing.queues.Queue[Any]
    runs_root: Path
    started_at: datetime
    cancel_requested: bool = False
    watcher: threading.Thread | None = None


_MP_CTX = multiprocessing.get_context("spawn")
_JOB_TABLE: dict[str, JobRecord] = {}
_JOB_LOCK = threading.Lock()


def dry_run_config(
    config_path: str,
    *,
    run_id: str | None = None,
    options: RunOptions | None = None,
    ping_llm: bool = True,
    runs_root: Path | None = None,
) -> DryRunResult:
    """Validate a config and perform LLM connectivity checks without executing trials."""

    resolved, prepared_model, chosen_run_id, prepared_config = _prepare_config(
        config_path,
        requested_run_id=run_id,
        options=options,
        runs_root=runs_root,
    )
    _ensure_env_and_ping(prepared_config, ping_llm=ping_llm)
    return DryRunResult(config_path=resolved, run_id=chosen_run_id, config=prepared_config)


def start_run(
    config_path: str,
    *,
    run_id: str | None = None,
    options: RunOptions | None = None,
    runs_root: Path | None = None,
    perform_dry_run: bool = True,
    ping_llm: bool = True,
) -> RunLaunch:
    """Create run artifacts and launch the optimization loop asynchronously."""

    resolved, prepared_model, chosen_run_id, prepared_config = _prepare_config(
        config_path,
        requested_run_id=run_id,
        options=options,
        runs_root=runs_root,
    )

    if perform_dry_run:
        _ensure_env_and_ping(prepared_config, ping_llm=ping_llm)

    handle = create_run(
        prepared_model,
        config_source=resolved,
        runs_root=_resolve_runs_root(runs_root),
    )

    run_id = handle.run_id
    run_dir = handle.artifacts.run_dir

    update_run_status(
        run_id,
        "running",
        runs_root=_resolve_runs_root(runs_root),
        note="Run started via GUI backend",
    )

    config_payload = handle.config.model_dump(mode="python")
    _launch_worker(run_id, config_payload, runs_root=_resolve_runs_root(runs_root))

    return RunLaunch(
        run_id=run_id,
        run_dir=run_dir,
        meta_path=handle.artifacts.meta_path,
        status="running",
    )


def list_runs(status: str | None = None, *, runs_root: Path | None = None) -> list[RunMetadata]:
    """Return run metadata filtered by status if provided."""

    filters: dict[str, Any] | None = None
    if status:
        filters = {"status": status}
    return tracking_list_runs(filters, runs_root=_resolve_runs_root(runs_root))


def load_run_detail(run_id: str, *, runs_root: Path | None = None) -> tuple[RunMetadata, dict[str, Any] | None]:
    """Load run metadata and resolved configuration."""

    metadata = tracking_load_run(run_id, runs_root=_resolve_runs_root(runs_root))
    config = _load_config_artifact(metadata)
    return metadata, config


def request_cancel(run_id: str) -> bool:
    """Send SIGINT to an active run."""

    record = _get_job(run_id)
    if record is None:
        raise ValueError(f"No active job for run_id '{run_id}'")

    if not record.process.is_alive():
        _remove_job(run_id)
        return False

    record.cancel_requested = True
    update_run_status(
        run_id,
        "cancelling",
        runs_root=record.runs_root,
        note="Cancellation requested",
    )
    record.process.send_signal(signal.SIGINT)
    return True


def active_job_info(run_id: str) -> dict[str, Any] | None:
    """Return lightweight information about an active job, if any."""

    record = _get_job(run_id)
    if record is None:
        return None
    return {
        "pid": record.process.pid,
        "state": "running" if record.process.is_alive() else "stopped",
        "cancel_requested": record.cancel_requested,
        "started_at": record.started_at.isoformat(),
    }


def _prepare_config(
    config_path: str,
    *,
    requested_run_id: str | None,
    options: RunOptions | None,
    runs_root: Path | None,
) -> tuple[Path, OptimizationConfig, str, dict[str, Any]]:
    """Load, override, and validate a configuration for execution."""

    resolved, model = validate_config(config_path, root=get_config_root())
    runs_root = _resolve_runs_root(runs_root)
    run_id = _resolve_run_id(requested_run_id, model, runs_root)
    updated_model = _apply_overrides(model, options, run_id, runs_root)
    config = updated_model.model_dump(mode="python")
    return resolved, updated_model, run_id, config


def _apply_overrides(
    model: OptimizationConfig,
    options: RunOptions | None,
    run_id: str,
    runs_root: Path,
) -> OptimizationConfig:
    overrides = options or RunOptions()
    data = model.model_dump(mode="python")

    artifacts = dict(data.get("artifacts") or {})
    artifacts["run_root"] = str(runs_root / run_id)
    data["artifacts"] = artifacts

    metadata_section = dict(data.get("metadata") or {})
    metadata_section.setdefault("name", run_id)
    data["metadata"] = metadata_section

    if overrides.max_trials is not None:
        stopping = dict(data.get("stopping") or {})
        stopping["max_trials"] = overrides.max_trials
        data["stopping"] = stopping

        search = dict(data.get("search") or {})
        search["n_trials"] = overrides.max_trials
        data["search"] = search

    if overrides.sampler:
        search = dict(data.get("search") or {})
        search["sampler"] = overrides.sampler
        data["search"] = search

    if overrides.llm_enabled is False:
        data["llm"] = None
        llm_guidance = data.get("llm_guidance")
        if isinstance(llm_guidance, Mapping):
            updated = dict(llm_guidance)
            updated["enabled"] = False
            data["llm_guidance"] = updated
        llm_critic = data.get("llm_critic")
        if isinstance(llm_critic, Mapping):
            updated = dict(llm_critic)
            updated["enabled"] = False
            data["llm_critic"] = updated

    return OptimizationConfig.model_validate(data)


def _ensure_env_and_ping(config: Mapping[str, Any], *, ping_llm: bool) -> None:
    env_file = Path(os.environ.get("ASTRAIA_ENV_FILE", ".env"))
    ensure_env_keys(config.get("llm"), env_path=env_file)
    if ping_llm:
        ping_llm_provider(config.get("llm"))


def _launch_worker(run_id: str, config: Mapping[str, Any], *, runs_root: Path) -> None:
    queue: multiprocessing.queues.Queue[Any] = _MP_CTX.Queue()
    process = _MP_CTX.Process(
        target=_run_worker,
        args=(config, run_id, str(runs_root), queue),
        daemon=True,
    )
    process.start()

    record = JobRecord(
        run_id=run_id,
        process=process,
        status_queue=queue,
        runs_root=runs_root,
        started_at=datetime.now(timezone.utc),
    )
    _register_job(record)

    watcher = threading.Thread(
        target=_watch_job,
        args=(record,),
        name=f"run-watch-{run_id}",
        daemon=True,
    )
    record.watcher = watcher
    watcher.start()


def _watch_job(record: JobRecord) -> None:
    final_status: str | None = None
    payload: dict[str, Any] = {}

    try:
        while record.process.is_alive():
            try:
                message = record.status_queue.get(timeout=0.5)
            except Empty:
                continue
            final_status = message.get("status") or final_status
            payload = message.get("payload") or payload

        while True:
            try:
                message = record.status_queue.get_nowait()
            except Empty:
                break
            final_status = message.get("status") or final_status
            payload = message.get("payload") or payload
    finally:
        record.process.join(timeout=0)
        if final_status is None:
            final_status = "failed"
            payload = payload or {}
            if record.cancel_requested:
                payload.setdefault("note", "Run cancelled by user.")
            else:
                payload.setdefault("note", "Run terminated without status payload.")

        update_run_status(
            record.run_id,
            final_status,
            runs_root=record.runs_root,
            **_json_ready(payload),
        )
        _remove_job(record.run_id)


def _run_worker(config: Mapping[str, Any], run_id: str, runs_root: str, status_queue: multiprocessing.queues.Queue[Any]) -> None:
    try:
        result = run_optimization(config)
        payload: dict[str, Any] = {
            "best_value": result.best_value,
            "best_metrics": result.best_metrics,
            "trials_completed": result.trials_completed,
            "best_params": result.best_params,
        }
        if result.total_cost is not None:
            payload["total_cost"] = result.total_cost
        if result.hypervolume is not None:
            payload["hypervolume"] = result.hypervolume
        if result.early_stopped_reason:
            payload["note"] = result.early_stopped_reason
        status_queue.put({"status": "completed", "payload": _json_ready(payload)})
    except KeyboardInterrupt:
        status_queue.put({"status": "failed", "payload": {"note": "Run interrupted (SIGINT)."}})
    except Exception as exc:  # pragma: no cover - runtime protection
        status_queue.put({"status": "failed", "payload": {"error": str(exc)}})
        raise


def _resolve_run_id(requested: str | None, model: OptimizationConfig, runs_root: Path) -> str:
    base_name: str
    if requested:
        base_name = requested
    else:
        base_name = model.metadata.name or ""
        if not base_name.strip():
            base_name = Path(model.metadata.model_dump().get("name") or "").name
        if not base_name.strip():
            base_name = "run"

    slug = _slugify(base_name)
    if not slug:
        slug = "run"

    candidate = slug
    counter = 2
    run_dir = runs_root / candidate
    while run_dir.exists():
        if requested:
            raise FileExistsError(f"Run '{candidate}' already exists under {runs_root}")
        candidate = f"{slug}-{counter:02d}"
        run_dir = runs_root / candidate
        counter += 1

    return candidate


def _slugify(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned)
    return cleaned.strip("-")


def _resolve_runs_root(runs_root: Path | None) -> Path:
    return runs_root if runs_root is not None else Path("runs")


def _register_job(record: JobRecord) -> None:
    with _JOB_LOCK:
        _JOB_TABLE[record.run_id] = record


def _get_job(run_id: str) -> JobRecord | None:
    with _JOB_LOCK:
        return _JOB_TABLE.get(run_id)


def _remove_job(run_id: str) -> None:
    with _JOB_LOCK:
        _JOB_TABLE.pop(run_id, None)


def _load_config_artifact(metadata: RunMetadata) -> dict[str, Any] | None:
    config_path = metadata.artifact_path("config_resolved")
    if config_path and config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None
    return None


def _json_ready(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass

    if isinstance(value, Mapping):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]

    try:
        return float(value)
    except Exception:
        return str(value)


__all__ = [
    "RunOptions",
    "DryRunResult",
    "RunLaunch",
    "dry_run_config",
    "start_run",
    "list_runs",
    "load_run_detail",
    "request_cancel",
    "active_job_info",
]
