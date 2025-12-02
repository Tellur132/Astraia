"""Utilities for managing run directories and experiment metadata."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, MutableMapping

import yaml

from .config import OptimizationConfig


@dataclass(frozen=True)
class RunArtifacts:
    """Paths produced for a concrete experiment run."""

    run_id: str
    run_dir: Path
    config_original: Path
    config_resolved: Path
    log_path: Path
    report_path: Path
    meta_path: Path
    llm_usage_path: Path | None = None
    llm_trace_path: Path | None = None


def prepare_run_environment(
    config_model: OptimizationConfig,
    *,
    config_source: Path | None = None,
    runs_root: Path | None = None,
) -> tuple[Dict[str, Any], RunArtifacts]:
    """Prepare a run directory and normalise artifact paths for execution."""

    config_dict: MutableMapping[str, Any] = config_model.model_dump(mode="python")
    artifacts_cfg = dict(config_dict.get("artifacts") or {})
    config_dict["artifacts"] = artifacts_cfg

    metadata = config_dict.get("metadata") or {}
    base_name = _slugify(str(metadata.get("name", "run")))
    base_runs_dir = runs_root or Path("runs")

    run_root_value = artifacts_cfg.get("run_root")
    if run_root_value:
        run_dir = Path(run_root_value)
    else:
        run_id, run_dir = _allocate_run_directory(base_runs_dir, base_name)
        artifacts_cfg["run_root"] = str(run_dir)
        base_name = run_id

    if not run_root_value:
        # Ensure the chosen run_id reflects the new directory name
        base_name = run_dir.name or base_name

    log_path = run_dir / "log.csv"
    artifacts_cfg["log_file"] = str(log_path)

    report_cfg = dict(config_dict.get("report") or {})
    report_cfg["output_dir"] = str(run_dir)
    report_cfg["filename"] = report_cfg.get("filename") or "report.md"
    config_dict["report"] = report_cfg

    llm_cfg = config_dict.get("llm")
    if isinstance(llm_cfg, MutableMapping):
        llm_cfg["usage_log"] = str(run_dir / "llm_usage.csv")
        llm_cfg["trace_log"] = str(run_dir / "llm_messages.jsonl")

    # Revalidate to capture any automatic adjustments from the schema
    final_model = OptimizationConfig.model_validate(config_dict)
    final_config = final_model.model_dump(mode="python")

    run_dir = Path(final_config["artifacts"]["run_root"])
    log_path = Path(final_config["artifacts"]["log_file"])

    report_section = final_config.get("report", {})
    report_filename = report_section.get("filename") or "report.md"
    report_path = Path(report_section.get("output_dir", run_dir)) / report_filename

    llm_usage_path = None
    llm_trace_path = None
    final_llm_cfg = final_config.get("llm")
    if isinstance(final_llm_cfg, MutableMapping):
        usage_value = final_llm_cfg.get("usage_log")
        if usage_value:
            llm_usage_path = Path(usage_value)
        trace_value = final_llm_cfg.get("trace_log")
        if trace_value:
            llm_trace_path = Path(trace_value)

    artifacts = _write_run_metadata(
        final_model,
        config_source=config_source,
        run_dir=run_dir,
        log_path=log_path,
        report_path=report_path,
        llm_usage_path=llm_usage_path,
        llm_trace_path=llm_trace_path,
    )

    return final_config, artifacts


def _slugify(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned)
    cleaned = cleaned.strip("-")
    return cleaned or "run"


def _allocate_run_directory(base_dir: Path, base_name: str) -> tuple[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_name or "run"
    run_dir = base_dir / candidate
    counter = 2
    while run_dir.exists():
        candidate = f"{base_name}-{counter:02d}"
        run_dir = base_dir / candidate
        counter += 1
    return candidate, run_dir


def _current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _write_run_metadata(
    model: OptimizationConfig,
    *,
    config_source: Path | None,
    run_dir: Path,
    log_path: Path,
    report_path: Path,
    llm_usage_path: Path | None,
    llm_trace_path: Path | None,
) -> RunArtifacts:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if llm_usage_path is not None:
        llm_usage_path.parent.mkdir(parents=True, exist_ok=True)
        llm_usage_path.touch(exist_ok=True)
    if llm_trace_path is not None:
        llm_trace_path.parent.mkdir(parents=True, exist_ok=True)
        llm_trace_path.touch(exist_ok=True)

    original_path = run_dir / "config_original.yaml"
    if config_source and config_source.exists():
        shutil.copyfile(config_source, original_path)
    else:
        with original_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                model.model_dump(mode="python"),
                fh,
                allow_unicode=True,
                sort_keys=False,
            )

    resolved_path = run_dir / "config_resolved.json"
    with resolved_path.open("w", encoding="utf-8") as fh:
        json.dump(model.model_dump(mode="json"), fh, indent=2, ensure_ascii=False)

    created_at = datetime.now(timezone.utc)

    meta: Dict[str, Any] = {
        "run_id": run_dir.name or run_dir.as_posix(),
        "run_dir": str(run_dir),
        "created_at": created_at.isoformat(),
        "status": {
            "state": "created",
            "updated_at": created_at.isoformat(),
        },
        "metadata": model.metadata.model_dump(mode="json"),
        "seed": model.seed,
        "report": {"metrics": model.report.metrics},
        "artifacts": {
            "config_original": str(original_path),
            "config_resolved": str(resolved_path),
            "log": str(log_path),
            "report": str(report_path),
        },
        "source": {
            "config_path": str(config_source) if config_source else None,
            "git_commit": _current_git_commit(),
        },
    }

    if llm_usage_path is not None:
        meta["artifacts"]["llm_usage"] = str(llm_usage_path)
    if llm_trace_path is not None:
        meta["artifacts"]["llm_trace"] = str(llm_trace_path)

    meta_path = run_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    return RunArtifacts(
        run_id=run_dir.name or run_dir.as_posix(),
        run_dir=run_dir,
        config_original=original_path,
        config_resolved=resolved_path,
        log_path=log_path,
        report_path=report_path,
        meta_path=meta_path,
        llm_usage_path=llm_usage_path,
        llm_trace_path=llm_trace_path,
    )


__all__ = ["RunArtifacts", "prepare_run_environment"]
