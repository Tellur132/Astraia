"""Lightweight strategy catalog builder for reusing past run knowledge."""
from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .run_summary import summarize_run_results
from .tracking import RunMetadata, list_runs

DEFAULT_CATALOG_PATH = Path("strategy_catalog.jsonl")
DEFAULT_FEWSHOT_FILES: dict[str, str] = {
    "qaoa_maxcut": "planner_prompts/qaoa_maxcut_success.md",
    "zdt3": "planner_prompts/zdt3_success.md",
}


@dataclass
class StrategyEntry:
    """Structured view of a catalog entry used for prompts."""

    problem_type: str
    run_id: str
    created_at: str
    trials_completed: int
    best_params: dict[str, Any]
    best_metrics: dict[str, Any]
    progress: dict[str, Any]
    hypervolume: float | None
    llm_guidance: bool
    planner: str | None
    sampler: str | None
    notes: str
    source: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "problem_type": self.problem_type,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "trials_completed": self.trials_completed,
            "best_params": self.best_params,
            "best_metrics": self.best_metrics,
            "progress": self.progress,
            "hypervolume": self.hypervolume,
            "llm_guidance": self.llm_guidance,
            "planner": self.planner,
            "sampler": self.sampler,
            "notes": self.notes,
            "source": self.source,
        }


def resolve_catalog_path(artifacts_cfg: Mapping[str, Any] | None = None) -> Path:
    """Resolve where to store the shared catalog."""

    env_path = os.environ.get("ASTRAIA_STRATEGY_CATALOG")
    if env_path:
        return Path(env_path)
    if artifacts_cfg and isinstance(artifacts_cfg, Mapping):
        custom = artifacts_cfg.get("strategy_catalog")
        if custom:
            return Path(str(custom))
    return DEFAULT_CATALOG_PATH


def infer_problem_type(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Heuristically classify the problem type for catalog grouping."""

    evaluator = config.get("evaluator") or {}
    search_cfg = config.get("search") or {}
    metadata = config.get("metadata") or {}
    name = str(metadata.get("name", "")).lower()
    description = str(metadata.get("description", "")).lower()
    module = str(evaluator.get("module", "")).lower()
    multi_objective = bool(search_cfg.get("multi_objective")) or len(
        search_cfg.get("metrics") or []
    ) > 1
    problem_type = "generic"
    if "zdt3" in module or "zdt3" in name or "zdt3" in description:
        problem_type = "zdt3"
    elif "qaoa" in module or "maxcut" in name or "maxcut" in description:
        problem_type = "qaoa_maxcut"
    elif "qgan" in module:
        problem_type = "qgan_kl"
    elif "branin" in module:
        problem_type = "branin"

    tags = {
        "multi_objective": multi_objective,
        "sampler": str(search_cfg.get("sampler")) if search_cfg else None,
        "metrics": search_cfg.get("metrics") or search_cfg.get("metric"),
    }
    if "num_qubits" in evaluator:
        tags["num_qubits"] = evaluator.get("num_qubits")
    if "edges" in evaluator:
        tags["edges"] = evaluator.get("edges")
    return problem_type, tags


def record_strategy_entry(
    *,
    problem_type: str,
    config: Mapping[str, Any],
    summary: Mapping[str, Any] | None,
    log_path: Path,
    catalog_path: Path,
    created_at: datetime | None = None,
) -> StrategyEntry | None:
    """Append a single strategy entry to the catalog."""

    created = created_at or datetime.now(timezone.utc)
    search_cfg = config.get("search") or {}
    llm_guidance_cfg = config.get("llm_guidance") or {}
    planner_cfg = config.get("planner") or {}

    trials_completed = int(summary.get("trials_completed") or 0) if summary else 0
    best_params = _json_safe(summary.get("best_params", {})) if summary else {}
    best_metrics = _json_safe(summary.get("best_metrics", {})) if summary else {}
    hypervolume = summary.get("hypervolume") if summary else None
    progress = _compute_progress(
        log_path,
        metric_names=_metric_names(config),
        directions=_direction_names(config),
    )
    notes = _build_notes(progress, hypervolume)
    run_id = str(config.get("metadata", {}).get("name") or log_path.parent.name)

    entry = StrategyEntry(
        problem_type=problem_type,
        run_id=run_id,
        created_at=created.isoformat(),
        trials_completed=trials_completed,
        best_params=best_params if isinstance(best_params, dict) else {},
        best_metrics=best_metrics if isinstance(best_metrics, dict) else {},
        progress=progress,
        hypervolume=hypervolume if isinstance(hypervolume, (int, float)) else None,
        llm_guidance=bool(llm_guidance_cfg.get("enabled")),
        planner=str(planner_cfg.get("backend")) if planner_cfg else None,
        sampler=str(search_cfg.get("sampler")) if search_cfg else None,
        notes=notes,
        source={
            "log_path": str(log_path),
            "run_dir": str(log_path.parent),
        },
    )
    _persist_entry(entry, catalog_path)
    return entry


def collect_runs_to_catalog(
    *,
    runs_root: Path,
    catalog_path: Path,
    problem_filter: str | None = None,
) -> list[StrategyEntry]:
    """Harvest existing runs into the catalog for reuse."""

    collected: list[StrategyEntry] = []
    existing = {entry["run_id"] for entry in _load_catalog(catalog_path)}
    for metadata in list_runs(runs_root=runs_root):
        if metadata.run_id in existing:
            continue
        config = _load_run_config(metadata)
        problem_type, _ = infer_problem_type(config)
        if problem_filter is not None and problem_type != problem_filter:
            continue
        summary = _load_run_summary(metadata)
        log_path = metadata.artifact_path("log") or metadata.run_dir / "log.csv"
        if not log_path.exists():
            continue
        entry = record_strategy_entry(
            problem_type=problem_type,
            config=config,
            summary=summary,
            log_path=log_path,
            catalog_path=catalog_path,
            created_at=metadata.created_at,
        )
        if entry is not None:
            collected.append(entry)
    return collected


def load_strategy_notes(
    problem_type: str,
    *,
    catalog_path: Path,
    top_k: int = 3,
) -> list[str]:
    """Return short bullet-style notes for prompt injection."""

    entries = _load_catalog(catalog_path)
    scoped = [entry for entry in entries if entry.get("problem_type") == problem_type]
    scored = sorted(scoped, key=_entry_score)[:top_k]
    notes: list[str] = []
    for entry in scored:
        hints = []
        progress = entry.get("progress") or {}
        milestones = progress.get("milestones") or progress.get("hypervolume_milestones") or {}
        quick = milestones.get("0.5") or milestones.get("50%") or milestones.get("0.8")
        if quick is not None:
            hints.append(f"early_gain_trial={quick}")
        hv = entry.get("hypervolume")
        if hv is not None:
            hints.append(f"hv~{_format_float(hv)}")
        best_params = entry.get("best_params") or {}
        dominant_params = _dominant_params(best_params)
        if dominant_params:
            hints.append("keys=" + ",".join(dominant_params))
        llm_label = "llm_on" if entry.get("llm_guidance") else "llm_off"
        planner = entry.get("planner") or "rule"
        note_body = " ".join(hints) if hints else entry.get("notes", "")
        notes.append(
            f"[{entry.get('run_id')}] {note_body} (planner={planner}, {llm_label})"
        )
    return notes


def load_fewshot_examples(problem_type: str) -> list[str]:
    """Load curated few-shot exemplars for a problem family."""

    path_value = DEFAULT_FEWSHOT_FILES.get(problem_type)
    if not path_value:
        return []
    root = Path(__file__).resolve().parents[2]
    path = (root / path_value) if not Path(path_value).is_absolute() else Path(path_value)
    if not path.exists():
        return []
    blocks = path.read_text(encoding="utf-8").split("\n---\n")
    return [block.strip() for block in blocks if block.strip()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_catalog(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                entries.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return entries


def _persist_entry(entry: StrategyEntry, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_catalog(path)
    replaced = False
    for idx, payload in enumerate(existing):
        if payload.get("run_id") == entry.run_id:
            existing[idx] = entry.to_json()
            replaced = True
            break
    if replaced:
        serialized = "\n".join(
            json.dumps(payload, ensure_ascii=False) for payload in existing
        )
        path.write_text(serialized + ("\n" if serialized else ""), encoding="utf-8")
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")


def _load_run_config(metadata: RunMetadata) -> dict[str, Any]:
    config_path = metadata.artifact_path("config_resolved")
    if config_path and config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    original = metadata.artifact_path("config_original")
    if original and original.exists():
        raw_text = original.read_text(encoding="utf-8")
        for loader in (json.loads, yaml.safe_load):
            try:
                loaded = loader(raw_text)
                if isinstance(loaded, Mapping):
                    return dict(loaded)
            except Exception:
                continue
    return dict(metadata.raw)


def _load_run_summary(metadata: RunMetadata) -> dict[str, Any]:
    summary_path = metadata.artifact_path("summary")
    if summary_path and summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    config = _load_run_config(metadata)
    try:
        return summarize_run_results(metadata, config=config, allow_missing_log=True)
    except Exception:
        return {}


def _compute_progress(
    log_path: Path,
    *,
    metric_names: Sequence[str],
    directions: Sequence[str],
) -> dict[str, Any]:
    if not log_path.exists() or not metric_names:
        return {}
    records = _read_csv(log_path)
    if not records:
        return {}

    primary_direction = directions[0] if directions else "minimize"
    if len(metric_names) == 1:
        return _progress_single_objective(records, metric_names[0], primary_direction)
    return _progress_multi_objective(records, metric_names, directions)


def _progress_single_objective(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    direction: str,
) -> dict[str, Any]:
    values: list[tuple[int, float]] = []
    for record in records:
        value = _extract_metric(record, metric)
        trial = _coerce_int(record.get("trial"), default=len(values))
        if value is None or not math.isfinite(value):
            continue
        values.append((trial, value))

    if not values:
        return {}

    baseline = values[0][1]
    best_value = baseline
    milestones: dict[str, int] = {}
    improvement = 0.0
    for trial, value in values:
        if _is_better(value, best_value, direction):
            best_value = value
            improvement = abs(baseline - best_value)
        milestones.setdefault("first_change", None)
        if milestones["first_change"] is None and not math.isclose(value, baseline):
            milestones["first_change"] = trial

    thresholds = [0.25, 0.5, 0.8]
    best_so_far = baseline
    for trial, value in values:
        if _is_better(value, best_so_far, direction):
            best_so_far = value
        for frac in thresholds:
            label = f"{int(frac * 100)}%"
            if milestones.get(label) is not None:
                continue
            if improvement <= 0:
                continue
            target = _target_value(baseline, improvement, frac, direction)
            if _is_better(best_so_far, target, direction):
                milestones[label] = trial

    return {
        "baseline": baseline,
        "best_value": best_value,
        "milestones": milestones,
    }


def _progress_multi_objective(
    records: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    directions: Sequence[str],
) -> dict[str, Any]:
    points: list[list[float]] = []
    trials: list[int] = []
    for record in records:
        values: list[float] = []
        missing = False
        for name in metric_names:
            value = _extract_metric(record, name)
            if value is None or not math.isfinite(value):
                missing = True
                break
            values.append(value)
        if missing:
            continue
        points.append(values)
        trials.append(_coerce_int(record.get("trial"), default=len(trials)))

    if not points:
        return {}

    hv_series: list[tuple[int, float | None]] = []
    for idx in range(len(points)):
        subset = points[: idx + 1]
        pareto = _pareto_front(subset, directions)
        hv = _approximate_hypervolume(
            pareto_points=pareto or subset,
            all_points=subset,
            direction_names=directions,
            seed=42,
            samples=1500,
        )
        hv_series.append((trials[idx], hv))

    hv_final = hv_series[-1][1]
    milestones: dict[str, int] = {}
    if hv_final and hv_final > 0:
        for frac in (0.25, 0.5, 0.8):
            label = f"{int(frac * 100)}%"
            threshold = hv_final * frac
            for trial, hv in hv_series:
                if hv is not None and hv >= threshold:
                    milestones[label] = trial
                    break

    return {
        "hypervolume": hv_final,
        "hypervolume_milestones": milestones,
    }


def _pareto_front(points: Sequence[Sequence[float]], directions: Sequence[str]) -> list[list[float]]:
    front: list[list[float]] = []
    for idx, candidate in enumerate(points):
        dominated = False
        dominated_indices: list[int] = []
        for jdx, other in enumerate(front):
            if _dominates(other, candidate, directions):
                dominated = True
                break
            if _dominates(candidate, other, directions):
                dominated_indices.append(jdx)
        if dominated:
            continue
        for jdx in reversed(dominated_indices):
            front.pop(jdx)
        front.append(list(candidate))
    return front


def _approximate_hypervolume(
    *,
    pareto_points: Sequence[Sequence[float]],
    all_points: Sequence[Sequence[float]],
    direction_names: Sequence[str],
    seed: int | None,
    samples: int = 1000,
) -> float | None:
    if not pareto_points:
        return None
    transformed_pareto = [
        _transform(point, direction_names) for point in pareto_points
    ]
    transformed_all = [_transform(point, direction_names) for point in all_points]
    dims = len(direction_names)
    lower = [min(point[idx] for point in transformed_pareto) for idx in range(dims)]
    upper = [max(point[idx] for point in transformed_all) for idx in range(dims)]
    for idx in range(dims):
        if math.isclose(lower[idx], upper[idx], rel_tol=1e-12, abs_tol=1e-12):
            upper[idx] = lower[idx] + 1.0
        margin = max(abs(upper[idx]) * 0.1, 1e-6)
        upper[idx] += margin

    rng = random.Random(seed)
    dominated = 0
    for _ in range(samples):
        sample = [rng.uniform(lower[idx], upper[idx]) for idx in range(dims)]
        for point in transformed_pareto:
            if all(sample[i] >= point[i] for i in range(dims)):
                dominated += 1
                break
    volume = 1.0
    for idx in range(dims):
        span = upper[idx] - lower[idx]
        if span <= 0:
            return None
        volume *= span
    return volume * (dominated / samples)


def _transform(values: Sequence[float], directions: Sequence[str]) -> list[float]:
    return [
        value if directions[idx] == "minimize" else -value
        for idx, value in enumerate(values)
    ]


def _dominates(candidate: Sequence[float], other: Sequence[float], directions: Sequence[str]) -> bool:
    better = False
    for idx, cand_value in enumerate(candidate):
        direction = directions[idx] if idx < len(directions) else "minimize"
        other_value = other[idx] if idx < len(other) else cand_value
        if direction == "minimize":
            if cand_value > other_value:
                return False
            better = better or cand_value < other_value
        else:
            if cand_value < other_value:
                return False
            better = better or cand_value > other_value
    return better


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _extract_metric(record: Mapping[str, Any], name: str) -> float | None:
    for key in (f"metric_{name}", name):
        value = record.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _metric_names(config: Mapping[str, Any]) -> list[str]:
    search_cfg = config.get("search") or {}
    metrics = search_cfg.get("metrics")
    if metrics:
        return [entry["name"] if isinstance(entry, Mapping) else str(entry) for entry in metrics]
    metric = search_cfg.get("metric")
    if metric is None:
        return []
    if isinstance(metric, list):
        return [str(entry) for entry in metric]
    return [str(metric)]


def _direction_names(config: Mapping[str, Any]) -> list[str]:
    search_cfg = config.get("search") or {}
    directions = search_cfg.get("directions")
    if directions:
        return [str(entry).lower() for entry in directions]
    direction = search_cfg.get("direction")
    if direction is None:
        return []
    if isinstance(direction, list):
        return [str(entry).lower() for entry in direction]
    return [str(direction).lower()]


def _target_value(baseline: float, improvement: float, fraction: float, direction: str) -> float:
    if direction == "maximize":
        return baseline + improvement * fraction
    return baseline - improvement * fraction


def _is_better(candidate: float, best: float, direction: str) -> bool:
    return candidate > best if direction == "maximize" else candidate < best


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return {}


def _format_float(value: float) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if abs(numeric) >= 1e4 or (abs(numeric) > 0 and abs(numeric) < 1e-3):
        return f"{numeric:.2e}"
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _entry_score(entry: Mapping[str, Any]) -> float:
    progress = entry.get("progress") or {}
    milestones = progress.get("milestones") or progress.get("hypervolume_milestones") or {}
    for key in ("50%", "0.5", "80%", "0.8", "25%", "0.25"):
        value = milestones.get(key)
        if value is not None:
            return float(value)
    trials = entry.get("trials_completed")
    try:
        return float(trials)
    except Exception:
        return float("inf")


def _build_notes(progress: Mapping[str, Any], hypervolume: Any) -> str:
    pieces: list[str] = []
    milestones = progress.get("milestones") or progress.get("hypervolume_milestones") or {}
    if milestones:
        quick = []
        for key in ("25%", "50%", "80%"):
            trial = milestones.get(key)
            if trial is not None:
                quick.append(f"{key}@{trial}")
        if quick:
            pieces.append("milestones=" + ",".join(quick))
    if hypervolume is not None:
        pieces.append(f"hv={_format_float(hypervolume)}")
    return "; ".join(pieces)


def _dominant_params(best_params: Mapping[str, Any]) -> list[str]:
    if not isinstance(best_params, Mapping):
        return []
    keys = [name for name in best_params.keys() if not str(name).startswith("metric_")]
    return list(keys)[:3]


__all__ = [
    "StrategyEntry",
    "collect_runs_to_catalog",
    "infer_problem_type",
    "load_fewshot_examples",
    "load_strategy_notes",
    "record_strategy_entry",
    "resolve_catalog_path",
]
