from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import threading
import time
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
    seed: int | None = None


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
class LlmComparisonLaunch:
    comparison_id: str
    seed: int | None
    llm_enabled: RunLaunch
    llm_disabled: RunLaunch
    summary_path: Path | None = None


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


def start_llm_comparison(
    config_path: str,
    *,
    run_id: str | None = None,
    options: RunOptions | None = None,
    runs_root: Path | None = None,
    perform_dry_run: bool = True,
    ping_llm: bool = True,
) -> LlmComparisonLaunch:
    """Launch paired LLM-on / LLM-off runs for apples-to-apples comparisons."""

    base_options = options or RunOptions()
    validated_path, validated_model = validate_config(config_path, root=get_config_root())
    if base_options.llm_enabled is False:
        raise ValueError("llm_comparison requires LLM features to remain enabled.")
    if not _llm_available(validated_model):
        raise ValueError("llm_comparison requested but config has no LLM settings.")

    seed = _select_shared_seed(validated_model, base_options, requested_run_id=run_id)
    llm_options = RunOptions(
        max_trials=base_options.max_trials,
        sampler=base_options.sampler,
        llm_enabled=base_options.llm_enabled if base_options.llm_enabled is not False else None,
        seed=seed,
    )
    llm_launch = start_run(
        config_path,
        run_id=run_id,
        options=llm_options,
        runs_root=runs_root,
        perform_dry_run=perform_dry_run,
        ping_llm=ping_llm,
    )

    baseline_run_id = _derive_baseline_run_id(llm_launch.run_id, _resolve_runs_root(runs_root))
    baseline_options = RunOptions(
        max_trials=base_options.max_trials,
        sampler=base_options.sampler,
        llm_enabled=False,
        seed=seed,
    )
    baseline_launch = start_run(
        config_path,
        run_id=baseline_run_id,
        options=baseline_options,
        runs_root=runs_root,
        perform_dry_run=perform_dry_run,
        ping_llm=False,
    )

    comparison_id = f"{llm_launch.run_id}-llm-comparison"
    comparison_path = _init_comparison_record(
        comparison_id,
        llm_launch,
        baseline_launch,
        validated_path,
        seed=seed,
        runs_root=_resolve_runs_root(runs_root),
    )

    _annotate_run_comparison(
        llm_launch.meta_path,
        comparison_id=comparison_id,
        paired_run_id=baseline_launch.run_id,
        seed=seed,
        role="llm_enabled",
        comparison_path=comparison_path,
    )
    _annotate_run_comparison(
        baseline_launch.meta_path,
        comparison_id=comparison_id,
        paired_run_id=llm_launch.run_id,
        seed=seed,
        role="llm_disabled",
        comparison_path=comparison_path,
    )

    _launch_comparison_monitor(
        comparison_id,
        llm_launch.run_id,
        baseline_launch.run_id,
        comparison_path,
        runs_root=_resolve_runs_root(runs_root),
        seed=seed,
    )

    return LlmComparisonLaunch(
        comparison_id=comparison_id,
        seed=seed,
        llm_enabled=llm_launch,
        llm_disabled=baseline_launch,
        summary_path=comparison_path,
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

    if overrides.seed is not None:
        data["seed"] = overrides.seed

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

        artifacts_payload = None
        if isinstance(payload, Mapping):
            artifacts_payload = payload.get("artifacts")
            if artifacts_payload is not None:
                payload = dict(payload)
                payload.pop("artifacts", None)

        update_run_status(
            record.run_id,
            final_status,
            runs_root=record.runs_root,
            artifacts=_json_ready(artifacts_payload) if artifacts_payload else None,
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
        if result.llm_trials is not None:
            payload["llm_trials"] = result.llm_trials
        if result.llm_accept_rate is not None:
            payload["llm_accept_rate"] = result.llm_accept_rate
        if result.early_stopped_reason:
            payload["note"] = result.early_stopped_reason
        artifacts: dict[str, Any] = {}
        if result.summary_path is not None:
            artifacts["summary"] = str(result.summary_path)
        if artifacts:
            payload["artifacts"] = artifacts

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


def _llm_available(model: OptimizationConfig) -> bool:
    if model.llm is not None:
        return True
    if model.llm_guidance is not None and model.llm_guidance.enabled:
        return True
    if model.llm_critic is not None and model.llm_critic.enabled:
        return True
    if (
        model.planner is not None
        and model.planner.enabled
        and getattr(model.planner, "backend", "rule") == "llm"
    ):
        return True
    return False


def _select_shared_seed(
    model: OptimizationConfig,
    options: RunOptions,
    *,
    requested_run_id: str | None,
) -> int | None:
    if options.seed is not None:
        return options.seed
    if model.seed is not None:
        return model.seed
    base = requested_run_id or model.metadata.name or "run"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    # keep value in 32bit-ish range for reproducibility across libs
    return int(digest[:8], 16)


def _derive_baseline_run_id(base_run_id: str, runs_root: Path) -> str:
    suffix = "no-llm"
    candidate = f"{base_run_id}-{suffix}"
    counter = 2
    while (runs_root / candidate).exists():
        candidate = f"{base_run_id}-{suffix}-{counter:02d}"
        counter += 1
    return candidate


def _init_comparison_record(
    comparison_id: str,
    llm_launch: RunLaunch,
    baseline_launch: RunLaunch,
    config_path: Path,
    *,
    seed: int | None,
    runs_root: Path,
) -> Path:
    record = {
        "id": comparison_id,
        "type": "llm_vs_no_llm",
        "config_path": str(config_path),
        "shared_seed": seed,
        "runs": {
            "llm_enabled": llm_launch.run_id,
            "llm_disabled": baseline_launch.run_id,
        },
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    compare_dir = runs_root / "comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = compare_dir / f"{comparison_id}.json"
    comparison_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return comparison_path


def _annotate_run_comparison(
    meta_path: Path,
    *,
    comparison_id: str,
    paired_run_id: str,
    seed: int | None,
    role: str,
    comparison_path: Path,
) -> None:
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return

    comparison_data = data.get("comparison")
    if isinstance(comparison_data, Mapping):
        merged = dict(comparison_data)
    else:
        merged = {}

    merged.update(
        {
            "group": comparison_id,
            "paired_run_id": paired_run_id,
            "role": role,
            "shared_seed": seed,
            "record_path": str(comparison_path),
        }
    )
    data["comparison"] = merged

    artifacts = data.get("artifacts")
    artifacts_map = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    artifacts_map.setdefault("llm_comparison", str(comparison_path))
    data["artifacts"] = artifacts_map

    meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _launch_comparison_monitor(
    comparison_id: str,
    llm_run_id: str,
    baseline_run_id: str,
    comparison_path: Path,
    *,
    runs_root: Path,
    seed: int | None,
) -> None:
    watcher = threading.Thread(
        target=_monitor_comparison_group,
        args=(comparison_id, llm_run_id, baseline_run_id, comparison_path, runs_root, seed),
        name=f"llm-compare-{comparison_id}",
        daemon=True,
    )
    watcher.start()


def _monitor_comparison_group(
    comparison_id: str,
    llm_run_id: str,
    baseline_run_id: str,
    comparison_path: Path,
    runs_root: Path,
    seed: int | None,
) -> None:
    while True:
        try:
            llm_meta = tracking_load_run(llm_run_id, runs_root=runs_root)
            baseline_meta = tracking_load_run(baseline_run_id, runs_root=runs_root)
        except FileNotFoundError:
            time.sleep(0.5)
            continue

        if _is_active_status(llm_meta.status) or _is_active_status(baseline_meta.status):
            time.sleep(1.0)
            continue

        summary = _build_comparison_summary(llm_meta, baseline_meta, seed)
        record = _load_comparison_record(comparison_path)
        record.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
            }
        )
        comparison_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        _annotate_run_comparison(
            llm_meta.meta_path,
            comparison_id=comparison_id,
            paired_run_id=baseline_run_id,
            seed=seed,
            role="llm_enabled",
            comparison_path=comparison_path,
        )
        _annotate_run_comparison(
            baseline_meta.meta_path,
            comparison_id=comparison_id,
            paired_run_id=llm_run_id,
            seed=seed,
            role="llm_disabled",
            comparison_path=comparison_path,
        )
        break


def _is_active_status(status: str | None) -> bool:
    if status is None:
        return True
    lowered = status.lower()
    return lowered in {"running", "cancelling", "created"}


def _load_comparison_record(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_comparison_summary(
    llm_meta: RunMetadata,
    baseline_meta: RunMetadata,
    seed: int | None,
) -> dict[str, Any]:
    llm_payload = _extract_outcome(llm_meta)
    baseline_payload = _extract_outcome(baseline_meta)
    deltas = _compute_metric_deltas(llm_payload, baseline_payload)
    return {
        "shared_seed": seed,
        "llm_enabled": llm_payload,
        "llm_disabled": baseline_payload,
        "deltas": deltas,
    }


def _extract_outcome(metadata: RunMetadata) -> dict[str, Any]:
    payload = metadata.status_payload if isinstance(metadata.status_payload, Mapping) else {}
    artifacts = metadata.artifact_paths
    return {
        "run_id": metadata.run_id,
        "status": metadata.status,
        "best_value": payload.get("best_value"),
        "best_metrics": payload.get("best_metrics"),
        "trials_completed": payload.get("trials_completed"),
        "total_cost": payload.get("total_cost"),
        "note": payload.get("note"),
        "log_path": str(artifacts.get("log")) if "log" in artifacts else None,
        "report_path": str(artifacts.get("report")) if "report" in artifacts else None,
    }


def _compute_metric_deltas(
    llm_payload: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
) -> dict[str, Any]:
    llm_metrics = (
        dict(llm_payload.get("best_metrics")) if isinstance(llm_payload.get("best_metrics"), Mapping) else {}
    )
    baseline_metrics = (
        dict(baseline_payload.get("best_metrics"))
        if isinstance(baseline_payload.get("best_metrics"), Mapping)
        else {}
    )
    deltas: dict[str, Any] = {}
    for name in set(llm_metrics).intersection(baseline_metrics):
        lhs = _safe_float(llm_metrics.get(name))
        rhs = _safe_float(baseline_metrics.get(name))
        if lhs is None or rhs is None:
            continue
        deltas[name] = lhs - rhs

    llm_best = _safe_float(llm_payload.get("best_value"))
    baseline_best = _safe_float(baseline_payload.get("best_value"))
    if llm_best is not None and baseline_best is not None:
        deltas["best_value"] = llm_best - baseline_best
    return deltas


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


__all__ = [
    "RunOptions",
    "DryRunResult",
    "RunLaunch",
    "LlmComparisonLaunch",
    "dry_run_config",
    "start_run",
    "start_llm_comparison",
    "list_runs",
    "load_run_detail",
    "request_cancel",
    "active_job_info",
]
