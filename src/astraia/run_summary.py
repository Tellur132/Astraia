"""Utilities for summarising run results recorded in CSV logs."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict

try:  # pragma: no cover - exercised indirectly in environments with pandas
    import pandas as pd
except ImportError:  # pragma: no cover - fallback path for offline tests
    pd = None  # type: ignore[assignment]

from .tracking import RunMetadata

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import pandas as _pd

MetricReducer = Callable[[Sequence[float], str], Any]


def read_log_dataframe(path: Path) -> "_pd.DataFrame":
    """Read the Optuna trial log into a pandas ``DataFrame``."""

    if pd is None:
        raise RuntimeError(
            "pandas is not installed; install the optional dependency to enable DataFrame outputs."
        )
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:  # type: ignore[attr-defined]
        return pd.DataFrame()


def summarize_run_results(
    metadata: RunMetadata,
    *,
    config: Mapping[str, Any] | None = None,
    metrics: Sequence[str] | None = None,
    statistic_names: Sequence[str] | None = None,
    allow_missing_log: bool = True,
) -> Dict[str, Any]:
    """Build a compact dictionary with summary statistics for a run."""

    log_path = metadata.artifact_path("log") or metadata.run_dir / "log.csv"
    try:
        columns, records = _load_log_records(log_path)
    except FileNotFoundError:
        if not allow_missing_log:
            raise
        columns, records = [], []

    metric_columns = [col for col in columns if col.startswith("metric_")]
    available_metrics = [col.replace("metric_", "", 1) for col in metric_columns]
    if metrics:
        selected_metrics = [name for name in metrics if name in available_metrics]
    else:
        selected_metrics = available_metrics

    stats = _resolve_statistics(statistic_names)
    directions = _resolve_metric_directions(config, selected_metrics)

    metric_summary: Dict[str, Dict[str, Any]] = OrderedDict()
    for metric in selected_metrics:
        column = f"metric_{metric}"
        if column not in columns:
            continue
        values = [
            record[column]
            for record in records
            if column in record and not _is_missing(record[column])
        ]
        summary: Dict[str, Any] = OrderedDict()
        direction = directions.get(metric, "minimize")
        for label, reducer in stats:
            summary[label] = _apply_reducer(values, reducer, direction)
        metric_summary[metric] = summary

    n_trials = len(records)
    valid_trials = _count_valid_trials(records, metric_columns)

    return {
        "run_id": metadata.run_id,
        "log_path": str(log_path),
        "metrics": metric_summary,
        "n_trials": n_trials,
        "n_valid_trials": valid_trials,
        "early_stop_reason": _extract_early_stop_reason(metadata),
    }


def _resolve_statistics(
    statistic_names: Sequence[str] | None,
) -> Sequence[tuple[str, MetricReducer]]:
    registry: Dict[str, MetricReducer] = {
        "best": _best_value,
        "median": _median_value,
        "mean": _mean_value,
    }
    if statistic_names:
        resolved: list[tuple[str, MetricReducer]] = []
        for name in statistic_names:
            key = name.lower()
            if key not in registry:
                raise ValueError(
                    f"Unknown statistic '{name}'. Available options: {', '.join(sorted(registry))}."
                )
            resolved.append((name, registry[key]))
        return resolved
    return [(name, registry[name]) for name in ("best", "median", "mean")]


def _resolve_metric_directions(
    config: Mapping[str, Any] | None, metric_names: Sequence[str]
) -> Dict[str, str]:
    directions = {name: "minimize" for name in metric_names}
    if not config:
        return directions
    search = config.get("search")
    if not isinstance(search, Mapping):
        return directions

    metric_list = _coerce_string_list(search.get("metrics"))
    if not metric_list:
        metric_list = _coerce_string_list(search.get("metric"))

    direction_list = _coerce_string_list(search.get("directions"))
    if not direction_list:
        direction_list = _coerce_string_list(search.get("direction"))

    if metric_list and not direction_list:
        direction_list = ["minimize"] * len(metric_list)
    elif len(direction_list) == 1 and len(metric_list) > 1:
        direction_list = direction_list * len(metric_list)

    for metric, direction in zip(metric_list, direction_list, strict=False):
        directions[str(metric)] = str(direction).lower()
    return directions


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value]
    return []


def _apply_reducer(values: Sequence[float], reducer: MetricReducer, direction: str) -> Any:
    if not values:
        return None
    return reducer(values, direction)


def _best_value(values: Sequence[float], direction: str) -> Any:
    if direction == "maximize":
        return max(values)
    return min(values)


def _median_value(values: Sequence[float], direction: str) -> Any:  # noqa: ARG001
    return statistics.median(values)


def _mean_value(values: Sequence[float], direction: str) -> Any:  # noqa: ARG001
    return statistics.fmean(values)


def _extract_early_stop_reason(metadata: RunMetadata) -> str | None:
    payload = metadata.status_payload if isinstance(metadata.status_payload, Mapping) else {}
    for key in ("early_stop_reason", "early_stop", "reason", "note"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _load_log_records(path: Path) -> tuple[list[str], list[Dict[str, Any]]]:
    if pd is not None:
        df = read_log_dataframe(path)
        columns = list(df.columns)
        records = df.to_dict(orient="records")  # type: ignore[no-any-return]
        return columns, records  # type: ignore[return-value]
    return _read_csv_records(path)


def _read_csv_records(path: Path) -> tuple[list[str], list[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        records: list[Dict[str, Any]] = []
        for row in reader:
            parsed = {key: _coerce_value(value) for key, value in row.items()}
            records.append(parsed)
    return columns, records


def _coerce_value(value: Any) -> Any:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _count_valid_trials(records: Sequence[Mapping[str, Any]], metric_columns: Sequence[str]) -> int:
    if not metric_columns:
        return len(records)
    valid = 0
    for record in records:
        if any(not _is_missing(record.get(column)) for column in metric_columns):
            valid += 1
    return valid


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


__all__ = [
    "read_log_dataframe",
    "summarize_run_results",
    "aggregate_llm_usage",
    "write_run_summary",
]


def aggregate_llm_usage(path: Path | None) -> dict[str, int]:
    """Aggregate LLM usage statistics from a CSV log.

    Returns a dictionary with call counts and token totals. Missing files or
    unparsable rows are treated as zero so that summary generation never fails
    when LLM logging is absent.
    """

    if path is None or not path.exists():
        return {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    calls = 0
    prompt = 0
    completion = 0
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            calls += 1
            prompt += _safe_int(row.get("prompt_tokens"))
            completion += _safe_int(row.get("completion_tokens"))
            total_entry = _safe_int(row.get("total_tokens"))
            if total_entry is not None:
                total += total_entry
            else:
                total += _safe_int(row.get("prompt_tokens")) + _safe_int(
                    row.get("completion_tokens")
                )

    return {
        "llm_calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def write_run_summary(
    *,
    run_id: str,
    run_dir: Path,
    log_path: Path,
    report_path: Path | None,
    trials_completed: int,
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, Any],
    best_value: Any,
    pareto_front: Sequence[Mapping[str, Any]] | None,
    hypervolume: float | None,
    llm_usage_path: Path | None,
    llm_trials: int | None,
    seed: int | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Persist a compact, JSON-friendly summary for downstream comparisons."""

    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = _build_metadata_stub(
        run_id=run_id,
        run_dir=run_dir,
        log_path=log_path,
        report_path=report_path,
        seed=seed,
        config=config,
    )

    metric_overview = summarize_run_results(
        metadata,
        config=config,
        allow_missing_log=True,
    )
    metrics_section = metric_overview.get("metrics", {})
    best_energy_gap = _extract_best_metric(metrics_section, "energy_gap")
    depth_best = _extract_best_metric(metrics_section, "depth")

    llm_usage = aggregate_llm_usage(llm_usage_path)
    llm_trial_count = llm_trials if llm_trials is not None else 0
    llm_accept_rate = (
        llm_trial_count / trials_completed if trials_completed > 0 else None
    )

    pareto_count = len(pareto_front or [])

    summary: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "report_path": str(report_path) if report_path is not None else None,
        "llm_usage_path": str(llm_usage_path) if llm_usage_path else None,
        "trials_completed": trials_completed,
        "best_value": best_value,
        "best_params": _json_safe(best_params),
        "best_metrics": _json_safe(best_metrics),
        "pareto_count": pareto_count,
        "hypervolume": hypervolume,
        "best_energy_gap": best_energy_gap,
        "depth_best": depth_best,
        "llm_trials": llm_trial_count,
        "llm_accept_rate": llm_accept_rate,
        "llm_calls": llm_usage["llm_calls"],
        "tokens": llm_usage["total_tokens"],
        "tokens_prompt": llm_usage["prompt_tokens"],
        "tokens_completion": llm_usage["completion_tokens"],
        "metric_overview": metric_overview,
        "seed": seed,
    }

    if pareto_front is not None:
        summary["pareto_front"] = _json_safe(list(pareto_front))

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _maybe_attach_summary_artifact(run_dir, summary_path)

    return summary, summary_path


def _maybe_attach_summary_artifact(run_dir: Path, summary_path: Path) -> None:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    artifacts = data.get("artifacts")
    artifacts_map = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    artifacts_map["summary"] = str(summary_path)
    data["artifacts"] = artifacts_map
    meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _extract_best_metric(metrics_section: Mapping[str, Any], name: str) -> Any:
    entry = metrics_section.get(name)
    if entry is None and name.startswith("metric_"):
        entry = metrics_section.get(name.replace("metric_", "", 1))
    if entry is None and not name.startswith("metric_"):
        entry = metrics_section.get(f"metric_{name}")
    if isinstance(entry, Mapping):
        return entry.get("best")
    return None


def _build_metadata_stub(
    *,
    run_id: str,
    run_dir: Path,
    log_path: Path,
    report_path: Path | None,
    seed: int | None,
    config: Mapping[str, Any] | None,
) -> RunMetadata:
    artifacts = {"log": str(log_path)}
    if report_path is not None:
        artifacts["report"] = str(report_path)
    metadata_payload = dict(config.get("metadata", {})) if isinstance(config, Mapping) else {}
    report_payload = dict(config.get("report", {})) if isinstance(config, Mapping) else {}

    return RunMetadata(
        run_id=run_id,
        run_dir=run_dir,
        meta_path=run_dir / "meta.json",
        created_at=None,
        status=None,
        status_updated_at=None,
        status_payload={},
        metadata=metadata_payload,
        seed=seed if isinstance(seed, int) else None,
        report=report_payload,
        artifacts=artifacts,
        source={},
        raw={
            "run_id": run_id,
            "run_dir": str(run_dir),
            "artifacts": artifacts,
            "metadata": metadata_payload,
            "seed": seed,
            "report": report_payload,
        },
    )


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except Exception:  # noqa: BLE001 - fallback
        return repr(value)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0
