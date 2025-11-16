"""Utilities for summarising run results recorded in CSV logs."""
from __future__ import annotations

import csv
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
    metric_field = search.get("metric")
    direction_field = search.get("direction")

    if isinstance(metric_field, str):
        metric_list = [metric_field]
    elif isinstance(metric_field, Sequence):
        metric_list = [str(item) for item in metric_field]
    else:
        metric_list = []

    if isinstance(direction_field, str):
        direction_list = [direction_field]
    elif isinstance(direction_field, Sequence):
        direction_list = [str(item) for item in direction_field]
    else:
        direction_list = []

    for metric, direction in zip(metric_list, direction_list, strict=False):  # type: ignore[arg-type]
        directions[str(metric)] = str(direction).lower()
    return directions


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


__all__ = ["read_log_dataframe", "summarize_run_results"]

