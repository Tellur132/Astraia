"""Core optimization loop implementation for the Anemoi MVP."""
from __future__ import annotations

import csv
import importlib
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import optuna
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.trial import TrialState

from .evaluators import BaseEvaluator, EvaluatorResult, MetricValue
from .llm_guidance import create_proposal_generator
from .llm_critic import generate_llm_critique
from .llm_interfaces import LLMObjective, LLMRepresentativePoint, LLMRunContext
from .meta_search import (
    SearchSettings,
    apply_meta_adjustment,
    create_meta_search_adjuster,
)
from .planner import create_planner_agent
from .run_summary import write_run_summary


@dataclass
class OptimizationResult:
    """Container for summarising the optimization run."""

    trials_completed: int
    best_params: Dict[str, Any]
    best_metrics: Dict[str, MetricValue]
    best_value: float
    early_stopped_reason: str | None = None
    pareto_front: List[Dict[str, Any]] | None = None
    hypervolume: float | None = None
    total_cost: float | None = None
    llm_trials: int | None = None
    llm_accept_rate: float | None = None
    summary_path: Path | None = None
    summary: Dict[str, Any] | None = None


def _seed_global_generators(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


class LLMTrialScheduler:
    """Gate LLM proposal usage for init-only or mixed schedules."""

    def __init__(self, cfg: Mapping[str, Any] | None, *, seed: int | None) -> None:
        mode = str(cfg.get("mode", "full") if cfg is not None else "full").lower()
        self._mode = mode if mode in {"full", "init_only", "mixed"} else "full"
        self._init_trials = int(cfg.get("init_trials", 5)) if cfg else 5
        self._mix_ratio = float(cfg.get("mix_ratio", 0.5)) if cfg else 0.5
        max_trials = cfg.get("max_llm_trials") if cfg else None
        self._max_llm_trials = int(max_trials) if max_trials is not None else None
        self._rng = random.Random(seed)
        self._llm_trials: int = 0

    @property
    def llm_trials(self) -> int:
        return self._llm_trials

    def allow_llm(self, trials_completed: int) -> bool:
        if self._max_llm_trials is not None and self._llm_trials >= self._max_llm_trials:
            return False
        if self._mode == "init_only":
            return trials_completed < self._init_trials
        if self._mode == "mixed":
            if self._mix_ratio <= 0.0:
                return False
            if self._mix_ratio >= 1.0:
                return True
            return self._rng.random() < self._mix_ratio
        return self._mode != "off"

    def record_trial(self, used_llm: bool) -> None:
        if used_llm:
            self._llm_trials += 1


def run_optimization(config: Mapping[str, Any]) -> OptimizationResult:
    """Execute the optimization loop using the provided configuration."""

    ensure_directories(config)
    seed = config.get("seed")
    _seed_global_generators(seed)
    evaluator = load_evaluator(config["evaluator"])
    search_cfg: Dict[str, Any] = dict(config["search"])
    library = str(search_cfg.get("library", "")).lower()
    if library != "optuna":
        raise ValueError(f"Unsupported search library: {library or 'unknown'}")
    search_space = {name: dict(spec) for name, spec in config["search_space"].items()}
    metric_names, direction_names = _collect_search_objectives(search_cfg)
    primary_metric = metric_names[0]

    sampler = build_sampler(search_cfg, seed)
    study = create_study(search_cfg, sampler, direction_names, seed=seed)

    stopping_cfg = config.get("stopping", {})
    max_trials = int(stopping_cfg.get("max_trials", search_cfg["n_trials"]))
    max_time_minutes = stopping_cfg.get("max_time_minutes")
    patience_value = stopping_cfg.get("no_improve_patience")
    cost_metric = stopping_cfg.get("cost_metric")
    max_total_cost = stopping_cfg.get("max_total_cost")
    target_trials = int(search_cfg.get("n_trials", max_trials))
    settings = SearchSettings(
        sampler=str(search_cfg.get("sampler", "tpe")).lower(),
        max_trials=max_trials,
        trial_budget=min(target_trials, max_trials),
        patience=int(patience_value) if patience_value is not None else None,
    )

    candidate_llm_override = _resolve_candidate_llm_cfg(config.get("planner"))

    proposal_generator = create_proposal_generator(
        config.get("llm_guidance"),
        config.get("llm"),
        search_space,
        seed=seed,
        preferred_llm_cfg=candidate_llm_override,
    )
    llm_scheduler = (
        LLMTrialScheduler(config.get("llm_guidance"), seed=seed)
        if proposal_generator is not None
        else None
    )
    pending_proposals: deque[Dict[str, Any]] = deque()

    planner_agent = create_planner_agent(
        config.get("planner"),
        config.get("llm"),
        search_space=search_space,
    )

    log_file = Path(config.get("artifacts", {}).get("log_file", "runs/log.csv"))

    trials_completed = 0
    early_stop_reason: str | None = None
    best_value: float | None = None
    best_metrics: Dict[str, MetricValue] = {}
    best_params: Dict[str, Any] = {}
    total_cost = 0.0 if cost_metric is not None else None
    llm_trials_used = 0

    study_directions = study.directions
    primary_direction = study_directions[0]

    llm_context = _build_llm_context(
        metric_names=metric_names,
        directions=direction_names,
        best_params=best_params,
        best_metrics=best_metrics,
        trials_completed=0,
    )

    with TrialLogger(
        log_file,
        search_space.keys(),
        config["report"]["metrics"],
    ) as writer:
        start_ts = time.time()
        no_improve_counter = 0

        meta_adjuster = create_meta_search_adjuster(
            config.get("meta_search"),
            config.get("llm"),
            direction=primary_direction,
            metric_name=primary_metric,
            metric_names=metric_names,
            directions=study_directions,
            search_space=search_space,
            seed=seed,
        )

        while trials_completed < settings.max_trials:
            if max_time_minutes is not None:
                elapsed = (time.time() - start_ts) / 60
                if elapsed >= float(max_time_minutes):
                    early_stop_reason = "max_time_minutes reached"
                    break

            llm_allowed = proposal_generator is not None
            if llm_allowed and llm_scheduler is not None:
                llm_allowed = llm_scheduler.allow_llm(trials_completed)

            if llm_allowed and proposal_generator is not None:
                proposal_generator.update_context(llm_context)
                while not pending_proposals:
                    remaining = max_trials - trials_completed
                    if remaining <= 0:
                        break
                    if planner_agent is not None:
                        strategy = planner_agent.generate_strategy(llm_context)
                        proposal_generator.update_strategy(strategy.to_payload())
                    else:
                        proposal_generator.update_strategy(None)
                    batch = proposal_generator.propose_batch(remaining)
                    if not batch:
                        break
                    pending_proposals.extend(batch)
                if pending_proposals:
                    study.enqueue_trial(pending_proposals.popleft())
            else:
                pending_proposals.clear()

            trial = study.ask()
            fixed_params = getattr(trial, "_fixed_params", {}) or {}
            is_llm_trial = bool(fixed_params)
            params = sample_params(trial, search_space)
            metrics, objective_values = _execute_trial(
                trial,
                evaluator,
                params=params,
                metric_names=metric_names,
                study_directions=study_directions,
                seed=seed,
            )
            primary_value = objective_values[0]

            if len(objective_values) == 1:
                study.tell(trial, primary_value)
            else:
                study.tell(trial, objective_values)
            trials_completed += 1

            writer.log(trial.number, params, metrics)

            primary_improved = update_best(
                primary_direction,
                primary_value,
                best_value,
            )
            pareto_improved = False
            if len(objective_values) > 1:
                pareto_improved = trial.number in {t.number for t in study.best_trials}
            improved = primary_improved or pareto_improved

            if primary_improved:
                best_value = primary_value
                best_metrics = dict(metrics)
                best_params = dict(params)

            if improved:
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if cost_metric is not None:
                cost_value = metrics.get(cost_metric)
                if cost_value is None:
                    raise RuntimeError(
                        f"Evaluator did not return cost metric '{cost_metric}'."
                    )
                try:
                    numeric_cost = float(cost_value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"Cost metric '{cost_metric}' must be numeric, got {cost_value!r}."
                    ) from exc
                assert total_cost is not None
                total_cost += numeric_cost
                if max_total_cost is not None and total_cost >= float(max_total_cost):
                    early_stop_reason = "cost_budget_exhausted"
                    break

            if meta_adjuster is not None:
                adjustment = meta_adjuster.register_trial(
                    trial_number=trial.number,
                    value=primary_value,
                    improved=improved,
                    params=params,
                    metrics=metrics,
                    best_value=best_value,
                    best_params=best_params,
                    trials_completed=trials_completed,
                    settings=settings,
                )
                if adjustment is not None:
                    messages = apply_meta_adjustment(
                        adjustment,
                        study=study,
                        search_cfg=search_cfg,
                        search_space=search_space,
                        best_params=best_params,
                        settings=settings,
                        trials_completed=trials_completed,
                        seed=seed,
                        sampler_builder=build_sampler,
                        proposal_generator=proposal_generator,
                    )
                    if messages:
                        print(f"[meta:{adjustment.source}] " + " | ".join(messages))

            llm_context = _build_llm_context(
                metric_names=metric_names,
                directions=direction_names,
                best_params=best_params,
                best_metrics=best_metrics,
                trials_completed=trials_completed,
            )

            if (
                settings.patience is not None
                and no_improve_counter >= int(settings.patience)
            ):
                early_stop_reason = "no_improve_patience reached"
                break

            if trials_completed >= settings.trial_budget:
                break

            if trials_completed >= settings.max_trials:
                break

    if trials_completed == 0:
        raise RuntimeError("Optimization did not record any trials.")

    if best_value is None:
        if study.best_trials:
            best_value = float(study.best_trials[0].values[0])
        else:
            candidates = [
                float(trial.values[0])
                for trial in study.get_trials(deepcopy=False)
                if trial.values
            ]
            if candidates:
                if primary_direction == optuna.study.StudyDirection.MINIMIZE:
                    best_value = min(candidates)
                else:
                    best_value = max(candidates)
            else:
                best_value = float("nan")

    report_path, pareto_records, hypervolume = build_report(
        config,
        best_params,
        best_metrics,
        trials_completed,
        early_stop_reason,
        metric_names=metric_names,
        direction_names=direction_names,
        study=study,
        total_cost=total_cost,
        cost_metric=cost_metric,
        seed=seed,
    )

    llm_accept_rate = (
        llm_trials_used / trials_completed if trials_completed > 0 else None
    )

    summary_payload: Dict[str, Any] | None = None
    summary_path: Path | None = None
    llm_usage_path: Path | None = None
    llm_cfg = config.get("llm")
    if isinstance(llm_cfg, Mapping):
        usage_value = llm_cfg.get("usage_log")
        if usage_value:
            llm_usage_path = Path(usage_value)

    artifacts_cfg = config.get("artifacts") if isinstance(config, Mapping) else {}
    run_root_value = None
    if isinstance(artifacts_cfg, Mapping):
        run_root_value = artifacts_cfg.get("run_root")
    run_root = Path(run_root_value) if run_root_value else log_file.parent
    try:
        summary_payload, summary_path = write_run_summary(
            run_id=str(config.get("metadata", {}).get("name") or run_root.name),
            run_dir=run_root,
            log_path=log_file,
            report_path=report_path,
            trials_completed=trials_completed,
            best_params=best_params,
            best_metrics=best_metrics,
            best_value=best_value,
            pareto_front=pareto_records,
            hypervolume=hypervolume,
            llm_usage_path=llm_usage_path,
            llm_trials=llm_trials_used,
            seed=seed if isinstance(seed, int) else None,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort summary
        print(f"[warning] Failed to write run summary: {exc}")

    return OptimizationResult(
        trials_completed=trials_completed,
        best_params=best_params,
        best_metrics=best_metrics,
        best_value=best_value,
        early_stopped_reason=early_stop_reason,
        pareto_front=pareto_records,
        hypervolume=hypervolume,
        total_cost=total_cost,
        llm_trials=llm_trials_used,
        llm_accept_rate=llm_accept_rate,
        summary_path=summary_path,
        summary=summary_payload,
    )


def _execute_trial(
    trial: optuna.trial.Trial,
    evaluator: Callable[[Dict[str, Any], int | None], EvaluatorResult],
    *,
    params: Mapping[str, Any],
    metric_names: Sequence[str],
    study_directions: Sequence[optuna.study.StudyDirection],
    seed: int | None,
    ) -> tuple[Dict[str, MetricValue], Tuple[float, ...]]:
    metrics = dict(evaluator(params, seed))

    missing_metrics = [name for name in metric_names if name not in metrics]
    if missing_metrics:
        for name in list(missing_metrics):
            prefixed = f"metric_{name}"
            if prefixed in metrics:
                metrics[name] = metrics[prefixed]
        missing_metrics = [name for name in metric_names if name not in metrics]
    if missing_metrics:
        missing = ", ".join(missing_metrics)
        raise RuntimeError(f"Evaluator did not return required metrics: {missing}.")

    objective_values, contains_non_finite = _extract_objective_values(
        metrics,
        metric_names,
    )
    trial_failed = _trial_failed(metrics) or contains_non_finite
    if trial_failed:
        objective_values = _failure_penalty_values(study_directions)

            tuple_values = tuple(objective_values)
            if len(tuple_values) == 1:
                for step, value in enumerate(tuple_values):
                    trial.report(value, step=step)

    trial.set_user_attr("metrics", dict(metrics))
    # Placeholders for future Pareto analytics.
            trial.set_user_attr("dominates", False)
            trial.set_user_attr("pareto_rank", None)

            if is_llm_trial:
                llm_trials_used += 1
                if llm_scheduler is not None:
                    llm_scheduler.record_trial(True)

    return dict(metrics), tuple_values


def _collect_search_objectives(search_cfg: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    metrics_value = search_cfg.get("metrics")
    if metrics_value is None:
        metrics_value = search_cfg.get("metric")
    if metrics_value is None:
        raise ValueError("search.metric must be provided")

    directions_value = search_cfg.get("directions")
    if directions_value is None:
        directions_value = search_cfg.get("direction", "minimize")

    metric_entries: List[Any]
    if isinstance(metrics_value, Sequence) and not isinstance(metrics_value, (str, bytes, bytearray)):
        if not metrics_value:
            raise ValueError("search.metrics must not be empty")
        metric_entries = list(metrics_value)
    else:
        metric_entries = [metrics_value]

    metric_names: List[str] = []
    directions: List[str] = []
    for idx, entry in enumerate(metric_entries):
        metric_name, explicit_direction = _parse_metric_entry(entry)
        metric_names.append(metric_name)
        directions.append(
            _resolve_direction_value(
                explicit_direction,
                default_source=directions_value,
                idx=idx,
                total=len(metric_entries),
            )
        )

    return metric_names, directions


def _parse_metric_entry(entry: Any) -> tuple[str, str | None]:
    if isinstance(entry, Mapping):
        metric_name = str(entry.get("name", "")).strip()
        if not metric_name:
            raise ValueError("search.metrics entries must include a non-empty 'name'")
        direction = entry.get("direction")
        return metric_name, str(direction) if direction is not None else None

    metric_name = str(entry).strip()
    if not metric_name:
        raise ValueError("search.metric entries must be non-empty strings")
    return metric_name, None


def _resolve_direction_value(
    explicit: str | None,
    *,
    default_source: Any,
    idx: int,
    total: int,
) -> str:
    direction_value: Any
    if explicit is not None:
        direction_value = explicit
    elif isinstance(default_source, Sequence) and not isinstance(
        default_source, (str, bytes, bytearray)
    ):
        if not default_source:
            raise ValueError("search.directions must not be empty when provided")
        if len(default_source) == 1:
            direction_value = default_source[0]
        elif idx < len(default_source):
            direction_value = default_source[idx]
        else:
            raise ValueError("search.directions must match the number of metrics")
    else:
        direction_value = default_source

    direction = str(direction_value).lower().strip()
    if direction not in {"minimize", "maximize"}:
        raise ValueError("search.direction entries must be 'minimize' or 'maximize'")
    return direction


def _prepare_pareto_outputs(
    *,
    study: optuna.study.Study,
    metric_names: Sequence[str],
    direction_names: Sequence[str],
    report_dir: Path,
    experiment_name: str,
    seed: int | None,
) -> Dict[str, Any] | None:
    if len(metric_names) <= 1:
        return None

    pareto_trials = list(study.best_trials)
    if not pareto_trials:
        return None

    records: List[Dict[str, Any]] = []
    objective_count = len(metric_names)
    for trial in pareto_trials:
        if not trial.values or len(trial.values) < objective_count:
            continue
        values = {
            metric: trial.values[idx]
            for idx, metric in enumerate(metric_names)
        }
        record: Dict[str, Any] = {
            "trial": trial.number,
            "values": values,
            "params": dict(trial.params),
        }
        metrics_attr = getattr(trial, "user_attrs", {})
        metrics_payload = metrics_attr.get("metrics") if isinstance(metrics_attr, Mapping) else None
        if isinstance(metrics_payload, Mapping):
            record["metrics"] = dict(metrics_payload)
        records.append(record)

    if not records:
        return None

    records.sort(key=lambda item: item["trial"])

    csv_path = report_dir / f"{experiment_name}_pareto.csv"
    param_names = _collect_param_names(records)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "trial",
            *metric_names,
            *[f"param_{name}" for name in param_names],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {"trial": record["trial"]}
            for metric in metric_names:
                row[metric] = record["values"].get(metric)
            for name in param_names:
                row[f"param_{name}"] = record["params"].get(name)
            writer.writerow(row)

    scatter_path = _render_pareto_scatter(
        records=records,
        metric_names=metric_names,
        direction_names=direction_names,
        report_dir=report_dir,
        experiment_name=experiment_name,
    )

    pareto_points = [
        [record["values"][metric] for metric in metric_names]
        for record in records
    ]
    all_points = _collect_objective_points(study, objective_count)
    hypervolume = _approximate_hypervolume(
        pareto_points=pareto_points,
        all_points=all_points,
        direction_names=direction_names,
        seed=seed,
    )

    representatives = _select_pareto_representatives(
        records=records,
        metric_names=metric_names,
        direction_names=direction_names,
    )

    return {
        "records": records,
        "csv_path": csv_path,
        "scatter_path": scatter_path,
        "hypervolume": hypervolume,
        "representatives": representatives,
        "param_names": param_names,
    }


def _collect_param_names(records: Sequence[Mapping[str, Any]]) -> List[str]:
    names: set[str] = set()
    for record in records:
        params = record.get("params")
        if isinstance(params, Mapping):
            names.update(params.keys())
    return sorted(names)


def _select_pareto_representatives(
    *,
    records: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    direction_names: Sequence[str],
) -> List[Dict[str, Any]]:
    representatives: List[Dict[str, Any]] = []
    metric_stats = _collect_metric_stats(records, metric_names)

    weighted = _select_weighted_solution(
        records=records,
        metric_names=metric_names,
        direction_names=direction_names,
        metric_stats=metric_stats,
    )
    if weighted is not None:
        _append_representative(
            representatives,
            record=weighted,
            label="Balanced (weighted sum)",
        )

    for metric, direction in zip(metric_names, direction_names):
        record = _select_extreme_solution(records, metric, direction)
        if record is None:
            continue
        label = f"Best {metric} ({direction})"
        _append_representative(representatives, record=record, label=label)

    return representatives


def _collect_metric_stats(
    records: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
) -> Dict[str, tuple[float | None, float | None]]:
    stats: Dict[str, tuple[float | None, float | None]] = {}
    for metric in metric_names:
        values = []
        for record in records:
            metric_value = record.get("values", {}).get(metric)
            numeric = _safe_float(metric_value)
            if numeric is not None:
                values.append(numeric)
        if values:
            stats[metric] = (min(values), max(values))
        else:
            stats[metric] = (None, None)
    return stats


def _select_weighted_solution(
    *,
    records: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    direction_names: Sequence[str],
    metric_stats: Mapping[str, tuple[float | None, float | None]],
) -> Mapping[str, Any] | None:
    best_score = float("inf")
    best_record: Mapping[str, Any] | None = None
    for record in records:
        score = 0.0
        invalid = False
        for metric, direction in zip(metric_names, direction_names):
            raw_value = record.get("values", {}).get(metric)
            normalised = _normalise_metric_value(
                raw_value,
                stats=metric_stats.get(metric, (None, None)),
                direction=direction,
            )
            if normalised is None:
                invalid = True
                break
            score += normalised
        if invalid:
            continue
        if score < best_score:
            best_score = score
            best_record = record
    return best_record


def _normalise_metric_value(
    value: Any,
    *,
    stats: tuple[float | None, float | None],
    direction: str,
) -> float | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    lower, upper = stats
    if lower is None or upper is None:
        return 0.0
    if math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-12):
        return 0.0
    span = upper - lower
    if span == 0:
        return 0.0
    if direction == "minimize":
        return (numeric - lower) / span
    return (upper - numeric) / span


def _select_extreme_solution(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    direction: str,
) -> Mapping[str, Any] | None:
    best_record: Mapping[str, Any] | None = None
    best_value: float | None = None
    for record in records:
        candidate = _safe_float(record.get("values", {}).get(metric))
        if candidate is None:
            continue
        sort_value = candidate if direction == "minimize" else -candidate
        if best_value is None or sort_value < best_value:
            best_value = sort_value
            best_record = record
    return best_record


def _append_representative(
    collection: List[Dict[str, Any]],
    *,
    record: Mapping[str, Any],
    label: str,
) -> None:
    for existing in collection:
        if existing.get("trial") == record.get("trial"):
            existing_label = existing.get("label") or ""
            if label not in existing_label:
                existing["label"] = (
                    f"{existing_label}, {label}" if existing_label else label
                )
            return
    collection.append(
        {
            "label": label,
            "trial": record.get("trial"),
            "values": record.get("values", {}),
            "params": record.get("params", {}),
        }
    )


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


class DifferentialEvolutionSampler(optuna.samplers.BaseSampler):
    """Lightweight differential evolution sampler for continuous spaces."""

    def __init__(
        self,
        *,
        weight: float = 0.8,
        crossover: float = 0.7,
        seed: int | None = None,
    ) -> None:
        self._weight = float(weight)
        self._crossover = float(crossover)
        self._random = random.Random(seed)
        self._fallback = optuna.samplers.RandomSampler(seed=seed)

    def reseed_rng(self, seed: int | None) -> None:
        self._random.seed(seed)
        self._fallback.reseed_rng(seed)

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Mapping[str, BaseDistribution],
    ) -> Dict[str, float]:
        return {}

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> float:
        if not isinstance(param_distribution, FloatDistribution):
            return self._fallback.sample_independent(
                study, trial, param_name, param_distribution
            )

        population = self._collect_population(study, param_name)
        if len(population) < 3:
            return self._fallback.sample_independent(
                study, trial, param_name, param_distribution
            )

        a, b, c = self._random.sample(population, 3)
        base = a
        candidate = base + self._weight * (b - c)
        low = float(param_distribution.low)
        high = float(param_distribution.high)

        if not math.isfinite(candidate):
            candidate = base

        candidate = min(max(candidate, low), high)

        if self._random.random() > self._crossover:
            candidate = base

        step = param_distribution.step
        if step is not None and step > 0:
            step_f = float(step)
            candidate = round((candidate - low) / step_f) * step_f + low
            candidate = min(max(candidate, low), high)

        if param_distribution.log:
            candidate = max(candidate, low)

        return float(candidate)

    def _collect_population(self, study: optuna.study.Study, param_name: str) -> List[float]:
        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        population: List[float] = []
        for trial in trials:
            value = trial.params.get(param_name)
            if value is None:
                continue
            try:
                population.append(float(value))
            except (TypeError, ValueError):
                continue
        return population


def _render_pareto_scatter(
    *,
    records: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    direction_names: Sequence[str],
    report_dir: Path,
    experiment_name: str,
) -> Path | None:
    if len(metric_names) != 2:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return None

    x_metric, y_metric = metric_names
    x_values = [record["values"].get(x_metric) for record in records]
    y_values = [record["values"].get(y_metric) for record in records]
    if not x_values or not y_values:
        return None

    figure, axis = plt.subplots()
    axis.scatter(x_values, y_values, c="tab:blue", s=60)
    axis.set_xlabel(f"{x_metric} ({direction_names[0]})")
    axis.set_ylabel(f"{y_metric} ({direction_names[1]})")
    axis.set_title("Pareto Front")
    axis.grid(True, linestyle="--", alpha=0.3)

    path = report_dir / f"{experiment_name}_pareto.png"
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _collect_objective_points(
    study: optuna.study.Study,
    objective_count: int,
) -> List[List[float]]:
    points: List[List[float]] = []
    for trial in study.get_trials(deepcopy=False):
        if not trial.values or len(trial.values) < objective_count:
            continue
        values = list(trial.values[:objective_count])
        if not all(math.isfinite(value) for value in values):
            continue
        points.append(values)
    return points


def _approximate_hypervolume(
    *,
    pareto_points: Sequence[Sequence[float]],
    all_points: Sequence[Sequence[float]],
    direction_names: Sequence[str],
    seed: int | None,
    samples: int = 5000,
) -> float | None:
    if not pareto_points:
        return None

    transformed_pareto = [
        _transform_objectives(point, direction_names)
        for point in pareto_points
    ]
    transformed_all = [
        _transform_objectives(point, direction_names)
        for point in all_points
    ]
    if not transformed_all:
        transformed_all = list(transformed_pareto)

    dims = len(direction_names)
    lower_bounds = [
        min(point[idx] for point in transformed_pareto) for idx in range(dims)
    ]
    upper_bounds = [
        max(point[idx] for point in transformed_all) for idx in range(dims)
    ]

    for idx in range(dims):
        if not math.isfinite(lower_bounds[idx]) or not math.isfinite(upper_bounds[idx]):
            return None
        if math.isclose(lower_bounds[idx], upper_bounds[idx], rel_tol=1e-9, abs_tol=1e-9):
            upper_bounds[idx] = lower_bounds[idx] + 1.0
        margin = max(abs(upper_bounds[idx]) * 0.1, 1e-6)
        upper_bounds[idx] += margin

    rng = random.Random(seed)
    dominated = 0
    for _ in range(samples):
        sample = [rng.uniform(lower_bounds[idx], upper_bounds[idx]) for idx in range(dims)]
        for point in transformed_pareto:
            if all(sample[idx] >= point[idx] for idx in range(dims)):
                dominated += 1
                break

    volume = 1.0
    for idx in range(dims):
        span = upper_bounds[idx] - lower_bounds[idx]
        if span <= 0:
            return None
        volume *= span

    return volume * dominated / samples


def _resolve_candidate_llm_cfg(
    planner_cfg: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(planner_cfg, Mapping):
        return None
    roles = planner_cfg.get("roles")
    if not isinstance(roles, Mapping):
        return None
    candidate_entry = roles.get("candidate")
    if not isinstance(candidate_entry, Mapping):
        return None
    candidate_llm = candidate_entry.get("llm")
    if isinstance(candidate_llm, Mapping):
        return candidate_llm
    return None


def _build_llm_context(
    *,
    metric_names: Sequence[str],
    directions: Sequence[str],
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, MetricValue],
    trials_completed: int,
) -> LLMRunContext:
    objectives: list[LLMObjective] = []
    for idx, name in enumerate(metric_names):
        direction = directions[idx] if idx < len(directions) else None
        objectives.append(LLMObjective(name=name, direction=direction))
    if not objectives:
        objectives.append(LLMObjective(name="primary_objective"))

    current_best: list[LLMRepresentativePoint] = []
    if best_metrics:
        values = {
            metric: best_metrics.get(metric)
            for metric in metric_names
            if metric in best_metrics
        }
        current_best.append(
            LLMRepresentativePoint(
                label="best",
                trial=None,
                values=values,
                params=dict(best_params),
                metrics=dict(best_metrics),
            )
        )

    return LLMRunContext(
        objectives=objectives,
        current_best=current_best,
        trials_completed=trials_completed or None,
    )


def _trial_failed(metrics: Mapping[str, MetricValue]) -> bool:
    status = metrics.get("status")
    if isinstance(status, str) and status.lower() != "ok":
        return True
    timed_out = metrics.get("timed_out")
    if isinstance(timed_out, bool):
        return timed_out
    return bool(timed_out)


def _extract_objective_values(
    metrics: Mapping[str, MetricValue],
    metric_names: Sequence[str],
) -> tuple[list[float], bool]:
    values: list[float] = []
    contains_non_finite = False
    for name in metric_names:
        value = float(metrics[name])
        if not math.isfinite(value):
            contains_non_finite = True
        values.append(value)
    return values, contains_non_finite


def _failure_penalty_values(
    directions: Sequence[optuna.study.StudyDirection],
) -> list[float]:
    penalties: list[float] = []
    for direction in directions:
        penalty = (
            float("inf")
            if direction == optuna.study.StudyDirection.MINIMIZE
            else float("-inf")
        )
        penalties.append(penalty)
    if not penalties:
        penalties.append(float("inf"))
    return penalties


def _transform_objectives(
    values: Sequence[float],
    direction_names: Sequence[str],
) -> List[float]:
    return [
        value if direction == "minimize" else -value
        for value, direction in zip(values, direction_names)
    ]


def _format_metric_value(value: Any) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(numeric)
    if abs(numeric) >= 1e4 or (abs(numeric) > 0 and abs(numeric) < 1e-3):
        return f"{numeric:.3e}"
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _format_param_summary(
    params: Mapping[str, Any],
    *,
    preferred_order: Sequence[str],
    limit: int = 3,
) -> str:
    if not params:
        return "—"
    ordered = []
    seen = set()
    for name in preferred_order:
        if name in params and name not in seen:
            ordered.append((name, params[name]))
            seen.add(name)
        if len(ordered) >= limit:
            break
    if len(ordered) < limit:
        for name in sorted(params.keys()):
            if name in seen:
                continue
            ordered.append((name, params[name]))
            if len(ordered) >= limit:
                break
    return ", ".join(f"{name}={_format_metric_value(value)}" for name, value in ordered)


def ensure_directories(config: Mapping[str, Any]) -> None:
    artifacts = config.get("artifacts", {})
    report = config.get("report", {})

    log_path = Path(artifacts.get("log_file", "runs/log.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    report_dir = Path(report.get("output_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    run_root = artifacts.get("run_root")
    if run_root:
        Path(run_root).mkdir(parents=True, exist_ok=True)

    llm_cfg = config.get("llm")
    if isinstance(llm_cfg, Mapping):
        usage_log = llm_cfg.get("usage_log")
        if usage_log:
            Path(usage_log).parent.mkdir(parents=True, exist_ok=True)


def load_evaluator(
    config: Mapping[str, Any]
) -> Callable[[Dict[str, Any], int | None], EvaluatorResult]:
    module = importlib.import_module(config["module"])
    target = getattr(module, config["callable"])

    evaluator_obj: Any
    if isinstance(target, BaseEvaluator):
        evaluator_obj = target
    elif isinstance(target, type) and issubclass(target, BaseEvaluator):
        evaluator_obj = target()  # type: ignore[call-arg]
    elif callable(target):
        signature = inspect.signature(target)
        if len(signature.parameters) <= 1:
            evaluator_obj = target(config)
        else:
            evaluator_obj = target
    else:
        raise TypeError(
            "Evaluator callable must be a function, factory, or BaseEvaluator instance."
        )

    if isinstance(evaluator_obj, BaseEvaluator):
        return evaluator_obj.evaluate
    if callable(evaluator_obj):
        return evaluator_obj
    raise TypeError("Evaluator factory did not return a callable or BaseEvaluator instance.")


def build_sampler(search_cfg: Mapping[str, Any], seed: int | None) -> optuna.samplers.BaseSampler:
    sampler_name = search_cfg.get("sampler", "tpe").lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "nsga2":
        return optuna.samplers.NSGAIISampler(seed=seed)
    if sampler_name == "nsgaiii":
        return optuna.samplers.NSGAIIISampler(seed=seed)
    if sampler_name == "motpe":
        sampler_cls = getattr(optuna.samplers, "MOTPESampler", None)
        if sampler_cls is None:
            raise ValueError(
                "Optuna installation does not provide MOTPESampler; upgrade Optuna to use 'motpe'."
            )
        return sampler_cls(seed=seed)  # type: ignore[call-arg]
    if sampler_name == "moead":
        sampler_cls = getattr(optuna.samplers, "MOEADSampler", None)
        if sampler_cls is None:
            raise ValueError(
                "Optuna installation does not provide MOEADSampler; upgrade Optuna to use 'moead'."
            )
        return sampler_cls(seed=seed)  # type: ignore[call-arg]
    if sampler_name == "mocma":
        sampler_cls = getattr(optuna.samplers, "MOCMASampler", None)
        if sampler_cls is None:
            raise ValueError(
                "Optuna installation does not provide MOCMASampler; upgrade Optuna to use 'mocma'."
            )
        return sampler_cls(seed=seed)  # type: ignore[call-arg]
    if sampler_name == "nevergrad":
        try:
            from optuna_integration.nevergrad import NevergradSampler  # type: ignore
        except ImportError:
            try:
                from optuna.integration import NevergradSampler  # type: ignore
            except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dep
                raise ValueError(
                    "sampler 'nevergrad' requires optuna-integration[nevergrad] to be installed"
                ) from exc
        return NevergradSampler(seed=seed)
    if sampler_name == "de":
        return DifferentialEvolutionSampler(seed=seed)
    raise ValueError(f"Unsupported sampler: {sampler_name}")


def create_study(
    search_cfg: Mapping[str, Any],
    sampler: optuna.samplers.BaseSampler,
    direction_names: Sequence[str],
    *,
    seed: int | None,
) -> optuna.study.Study:
    if len(direction_names) == 1:
        return optuna.create_study(
            study_name=search_cfg.get("study_name"),
            direction=direction_names[0],
            sampler=sampler,
        )
    sampler_params = search_cfg.get("sampler_params")
    nsga_sampler = _build_nsga_sampler(sampler_params, seed)
    return optuna.create_study(
        study_name=search_cfg.get("study_name"),
        directions=list(direction_names),
        sampler=nsga_sampler,
    )


def _build_nsga_sampler(
    sampler_params: Any,
    seed: int | None,
) -> optuna.samplers.NSGAIISampler:
    allowed_keys = {
        "population_size",
        "mutation_prob",
        "mutation_eta",
        "crossover_prob",
        "crossover_eta",
        "swapping_prob",
    }
    kwargs: Dict[str, Any] = {}
    if isinstance(sampler_params, Mapping):
        for key in allowed_keys:
            if key in sampler_params:
                kwargs[key] = sampler_params[key]
    if seed is not None and "seed" not in kwargs:
        kwargs["seed"] = seed
    return optuna.samplers.NSGAIISampler(**kwargs)


def sample_params(trial: optuna.trial.Trial, space: Mapping[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    # Layered QAOA angles: sample n_layers first so deeper gamma/beta are conditional.
    n_layers_value = None
    n_layers_spec = space.get("n_layers")
    if isinstance(n_layers_spec, Mapping) and n_layers_spec.get("type") == "int":
        step = n_layers_spec.get("step")
        int_step = int(step) if step is not None else 1
        params["n_layers"] = trial.suggest_int(
            "n_layers",
            int(n_layers_spec["low"]),
            int(n_layers_spec["high"]),
            step=int_step,
        )
        n_layers_value = params["n_layers"]

    for name, spec in space.items():
        if name == "n_layers":
            continue

        layer_index = _layer_index(name)
        if n_layers_value is not None and layer_index is not None and layer_index >= n_layers_value:
            params[name] = 0.0  # keep CSV aligned but avoid exploring unused angles
            continue

        param_type = spec.get("type")
        if param_type == "float":
            step = spec.get("step")
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
                step=None if step is None else float(step),
            )
        elif param_type == "int":
            step = spec.get("step")
            int_step = int(step) if step is not None else 1
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int_step,
            )
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif param_type == "llm_only":
            params[name] = _suggest_llm_only(trial, name, spec)
        else:
            raise ValueError(f"Unsupported parameter type for '{name}': {param_type}")
    return params


def _layer_index(name: str) -> int | None:
    """Extract gamma/beta layer index from names like gamma_2 or beta_3."""
    if "_" not in name:
        return None
    prefix, _, suffix = name.partition("_")
    if prefix not in {"gamma", "beta"}:
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def _suggest_llm_only(trial: optuna.trial.Trial, name: str, spec: Mapping[str, Any]) -> Any:
    fixed_value = None
    fixed_params = getattr(trial, "_fixed_params", {})
    if isinstance(fixed_params, Mapping):
        fixed_value = fixed_params.get(name)

    if fixed_value is None:
        fallback = spec.get("default")
        if fallback is None:
            fallback = f"llm_value_{random.randint(0, 1_000_000)}"
        fixed_value = fallback

    return trial.suggest_categorical(name, [fixed_value])


def update_best(
    direction: optuna.study.StudyDirection,
    current_value: float,
    best_value: float | None,
) -> bool:
    if best_value is None:
        return True
    if direction == optuna.study.StudyDirection.MINIMIZE:
        return current_value < best_value
    return current_value > best_value


class TrialLogger:
    """Utility to append trial information to a CSV log file."""

    def __init__(
        self,
        path: Path,
        param_names: Iterable[str],
        metric_names: Iterable[str],
    ) -> None:
        self.path = path
        self.param_names = list(param_names)
        self.metric_names = [str(name) for name in metric_names]
        self.metric_fields = [
            (
                name,
                name if name.startswith("metric_") else f"metric_{name}",
            )
            for name in self.metric_names
        ]

        write_header = not path.exists()
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=[
                "trial",
                *[f"param_{name}" for name in self.param_names],
                *[field for _, field in self.metric_fields],
            ],
        )
        if write_header:
            self._writer.writeheader()

    def log(
        self,
        trial_number: int,
        params: Mapping[str, Any],
        metrics: Mapping[str, MetricValue],
    ) -> None:
        row: Dict[str, Any] = {"trial": trial_number}
        for name in self.param_names:
            row[f"param_{name}"] = params.get(name)
        for name, field in self.metric_fields:
            row[field] = self._resolve_metric_value(metrics, name)
        self._writer.writerow(row)
        self._fh.flush()

    @staticmethod
    def _resolve_metric_value(
        metrics: Mapping[str, MetricValue],
        name: str,
    ) -> MetricValue | None:
        if name in metrics:
            return metrics[name]

        if name.startswith("metric_"):
            bare_name = name.removeprefix("metric_")
            if bare_name in metrics:
                return metrics[bare_name]
        else:
            prefixed = f"metric_{name}"
            if prefixed in metrics:
                return metrics[prefixed]

        return None

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TrialLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001 - guard for interpreter shutdown
            pass


def build_report(
    config: Mapping[str, Any],
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, float],
    trials_completed: int,
    early_stop_reason: str | None,
    *,
    metric_names: Sequence[str],
    direction_names: Sequence[str],
    study: optuna.study.Study,
    total_cost: float | None,
    cost_metric: str | None,
    seed: int | None,
) -> tuple[Path, List[Dict[str, Any]] | None, float | None]:
    report_cfg = config.get("report", {})
    report_dir = Path(report_cfg.get("output_dir", "reports"))
    filename = report_cfg.get("filename") or f"{config['metadata']['name']}.md"
    report_path = report_dir / filename
    log_path = Path(config.get("artifacts", {}).get("log_file", "runs/log.csv"))
    objective_summary = ", ".join(
        f"{metric} ({direction})"
        for metric, direction in zip(metric_names, direction_names)
    )

    lines: List[str] = [
        f"# Experiment Report — {config['metadata']['name']}",
        "",
        f"Description: {config['metadata'].get('description', '')}",
        "",
        f"Trials executed: {trials_completed}",
        f"Objectives: {objective_summary}",
        "",
        "## Best Parameters",
    ]
    lines.extend([f"- **{name}**: {value}" for name, value in best_params.items()])
    lines.append("")
    lines.append("## Best Metrics")
    lines.extend([f"- **{name}**: {value}" for name, value in best_metrics.items()])
    if total_cost is not None and cost_metric is not None:
        lines.extend(["", f"Total {cost_metric}: {_format_metric_value(total_cost)}"])
    if early_stop_reason:
        lines.extend(["", f"_Early stopping_: {early_stop_reason}"])

    pareto_summary = _prepare_pareto_outputs(
        study=study,
        metric_names=metric_names,
        direction_names=direction_names,
        report_dir=report_dir,
        experiment_name=config["metadata"]["name"],
        seed=seed,
    )

    if pareto_summary is not None:
        lines.extend(["", "## Pareto Front", ""])

        representatives = pareto_summary.get("representatives") or []
        param_names = pareto_summary.get("param_names") or []
        if representatives:
            lines.extend(["### Representative Solutions", ""])
            rep_header = (
                "| Representative | Trial | "
                + " | ".join(metric_names)
                + " | Key params |"
            )
            rep_separator = "|" + " --- |" * (len(metric_names) + 3)
            lines.extend([rep_header, rep_separator])
            for entry in representatives:
                values = entry.get("values", {})
                params = entry.get("params", {})
                row = [
                    str(entry.get("label", "")),
                    str(entry.get("trial", "—")),
                ]
                row.extend(
                    _format_metric_value(values.get(metric)) for metric in metric_names
                )
                row.append(
                    _format_param_summary(
                        params,
                        preferred_order=param_names,
                        limit=3,
                    )
                )
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

        lines.append("### All Pareto Trials")
        header = "| Trial | " + " | ".join(metric_names) + " |"
        separator = "|" + " --- |" * (len(metric_names) + 1)
        lines.extend(["", header, separator])
        for record in pareto_summary["records"]:
            values = record["values"]
            row = [str(record["trial"])]
            row.extend(_format_metric_value(values.get(metric)) for metric in metric_names)
            lines.append("| " + " | ".join(row) + " |")

        lines.extend(
            [
                "",
                f"Pareto CSV: `{pareto_summary['csv_path'].name}` (objectives + params)",
            ]
        )
        scatter_path = pareto_summary.get("scatter_path")
        if scatter_path is not None:
            lines.append(f"![Pareto scatter plot]({scatter_path.name})")
        hypervolume = pareto_summary.get("hypervolume")
        if hypervolume is not None:
            lines.append("")
            lines.append(
                f"Approximate hypervolume: {_format_metric_value(hypervolume)}"
            )
    else:
        hypervolume = None

    critic_section = generate_llm_critique(
        config=config,
        metadata=config.get("metadata", {}),
        metric_names=metric_names,
        direction_names=direction_names,
        best_params=best_params,
        best_metrics=best_metrics,
        trials_completed=trials_completed,
        early_stop_reason=early_stop_reason,
        log_path=log_path,
        pareto_summary=pareto_summary,
    )
    if critic_section:
        lines.extend(["", "## LLM考察", ""])
        lines.extend(critic_section)

    report_path.write_text("\n".join(lines), encoding="utf-8")
    pareto_records = (
        [dict(record) for record in pareto_summary["records"]]
        if pareto_summary is not None
        else None
    )
    return report_path, pareto_records, hypervolume
