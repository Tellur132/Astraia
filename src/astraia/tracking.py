"""High level run tracking helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import OptimizationConfig
from .run_management import RunArtifacts, prepare_run_environment


@dataclass(frozen=True)
class RunHandle:
    """Handle returned when a new run is created."""

    config: OptimizationConfig
    artifacts: RunArtifacts

    @property
    def run_id(self) -> str:
        return self.artifacts.run_id

    @property
    def run_dir(self) -> Path:
        return self.artifacts.run_dir


@dataclass(frozen=True)
class RunMetadata:
    """Structured view of run metadata stored on disk."""

    run_id: str
    run_dir: Path
    meta_path: Path
    created_at: datetime | None
    status: str | None
    status_updated_at: datetime | None
    status_payload: Mapping[str, Any]
    metadata: Mapping[str, Any]
    seed: int | None
    report: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    source: Mapping[str, Any]
    raw: Mapping[str, Any]

    def artifact_path(self, name: str) -> Path | None:
        value = self.artifacts.get(name)
        if isinstance(value, str):
            return Path(value)
        return None

    @property
    def artifact_paths(self) -> Mapping[str, Path]:
        return {
            key: Path(value)
            for key, value in self.artifacts.items()
            if isinstance(value, str)
        }


def create_run(
    config: OptimizationConfig,
    *,
    config_source: Path | None = None,
    runs_root: Path | None = None,
) -> RunHandle:
    """Create run artifacts and persist baseline metadata."""

    final_config, artifacts = prepare_run_environment(
        config,
        config_source=config_source,
        runs_root=runs_root,
    )
    final_model = OptimizationConfig.model_validate(final_config)

    # ensure metadata contains an initial status stanza
    _ensure_status(artifacts.meta_path)

    return RunHandle(config=final_model, artifacts=artifacts)


def update_run_status(
    run_id: str | Path,
    status: str,
    *,
    runs_root: Path | None = None,
    **payload: Any,
) -> RunMetadata:
    """Update the run status information stored in ``meta.json``."""

    metadata = load_run(run_id, runs_root=runs_root)
    meta_data = dict(metadata.raw)

    new_status = dict(payload)
    new_status["state"] = status
    new_status["updated_at"] = datetime.now(timezone.utc).isoformat()
    meta_data["status"] = new_status

    metadata.meta_path.write_text(
        json.dumps(meta_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return load_run(metadata.meta_path.parent, runs_root=runs_root)


def list_runs(
    filters: Mapping[str, Any] | None = None,
    *,
    runs_root: Path | None = None,
) -> list[RunMetadata]:
    """Enumerate run metadata objects, optionally filtering results."""

    base_dir = _resolve_runs_root(runs_root)
    if not base_dir.exists():
        return []

    results: list[RunMetadata] = []
    for meta_path in sorted(base_dir.glob("*/meta.json")):
        metadata = _read_metadata(meta_path)
        if _matches_filters(metadata, filters):
            results.append(metadata)
    return results


def load_run(
    run_id: str | Path,
    *,
    runs_root: Path | None = None,
) -> RunMetadata:
    """Load metadata for a specific run."""

    candidate = Path(run_id)
    meta_path = candidate / "meta.json"
    if candidate.is_dir() and meta_path.exists():
        return _read_metadata(meta_path)

    base_dir = _resolve_runs_root(runs_root)
    meta_path = base_dir / str(run_id) / "meta.json"
    if meta_path.exists():
        return _read_metadata(meta_path)

    for alternative in base_dir.glob("*/meta.json"):
        if alternative.parent.name == str(run_id):
            return _read_metadata(alternative)

    raise FileNotFoundError(f"No run metadata found for '{run_id}'")


def _ensure_status(meta_path: Path) -> None:
    data = _read_metadata(meta_path).raw if meta_path.exists() else None
    if data and "status" in data:
        return

    if meta_path.exists():
        meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta_data = {}

    status = meta_data.get("status")
    if not isinstance(status, Mapping):
        created_at = meta_data.get("created_at")
        if not isinstance(created_at, str):
            created_at = datetime.now(timezone.utc).isoformat()
        meta_data["status"] = {
            "state": "created",
            "updated_at": created_at,
        }
        meta_path.write_text(
            json.dumps(meta_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _matches_filters(metadata: RunMetadata, filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return True

    for key, expected in filters.items():
        if key == "run_id":
            if metadata.run_id != expected:
                return False
            continue
        if key == "status":
            if metadata.status != expected:
                return False
            continue
        value = _lookup(metadata.raw, key)
        if value != expected:
            return False
    return True


def _lookup(payload: Mapping[str, Any], dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    current: Any = payload
    for part in parts:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _read_metadata(meta_path: Path) -> RunMetadata:
    raw_data = json.loads(meta_path.read_text(encoding="utf-8"))

    created_at = _parse_datetime(raw_data.get("created_at"))
    status_raw = raw_data.get("status")
    if isinstance(status_raw, Mapping):
        status_payload = dict(status_raw)
        status = status_payload.get("state")
        status_updated_at = _parse_datetime(status_payload.get("updated_at"))
    else:
        status_payload = {}
        status = None
        status_updated_at = None

    metadata_value = raw_data.get("metadata")
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}

    report_value = raw_data.get("report")
    report = dict(report_value) if isinstance(report_value, Mapping) else {}

    artifacts_value = raw_data.get("artifacts")
    artifacts = dict(artifacts_value) if isinstance(artifacts_value, Mapping) else {}

    source_value = raw_data.get("source")
    source = dict(source_value) if isinstance(source_value, Mapping) else {}

    seed = raw_data.get("seed")
    if not isinstance(seed, int):
        seed = None

    run_id = raw_data.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = meta_path.parent.name

    run_dir_value = raw_data.get("run_dir")
    run_dir = Path(run_dir_value) if isinstance(run_dir_value, str) else meta_path.parent

    return RunMetadata(
        run_id=run_id,
        run_dir=run_dir,
        meta_path=meta_path,
        created_at=created_at,
        status=status,
        status_updated_at=status_updated_at,
        status_payload=status_payload,
        metadata=metadata,
        seed=seed,
        report=report,
        artifacts=artifacts,
        source=source,
        raw=raw_data,
    )


def _resolve_runs_root(runs_root: Path | None) -> Path:
    if runs_root is not None:
        return Path(runs_root)
    return Path("runs")


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


__all__ = [
    "RunHandle",
    "RunMetadata",
    "create_run",
    "list_runs",
    "load_run",
    "update_run_status",
]

