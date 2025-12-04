"""Core optimization loop implementation for the Anemoi MVP."""
from __future__ import annotations

import csv
import importlib
import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import optuna
from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.trial import TrialState

from .evaluators import BaseEvaluator, EvaluatorResult, MetricValue
from .llm_guidance import create_proposal_generator, fingerprint_proposal
from .llm_critic import generate_llm_critique
from .llm_interfaces import LLMObjective, LLMRepresentativePoint, LLMRunContext
from .meta_search import (
    SearchSettings,
    apply_meta_adjustment,
    create_meta_search_adjuster,
)
from .planner import create_planner_agent
from .run_summary import write_run_summary
from .strategy_catalog import (
    collect_runs_to_catalog,
    infer_problem_type,
    load_fewshot_examples,
    load_strategy_notes,
    record_strategy_entry,
    resolve_catalog_path,
)


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
        self._mix_ratio_floor = (
            float(cfg.get("mix_ratio_floor", 0.1)) if cfg else 0.1
        )
        self._mix_ratio_decay = float(cfg.get("mix_ratio_decay", 0.5)) if cfg else 0.5
        self._current_mix_ratio = max(self._mix_ratio_floor, self._mix_ratio)
        max_trials = cfg.get("max_llm_trials") if cfg else None
        self._max_llm_trials = int(max_trials) if max_trials is not None else None
        self._rng = random.Random(seed)
        self._llm_trials: int = 0

    @property
    def llm_trials(self) -> int:
        return self._llm_trials

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def current_mix_ratio(self) -> float:
        return self._current_mix_ratio

    @property
    def max_llm_trials(self) -> int | None:
        return self._max_llm_trials

    def has_capacity(self) -> bool:
        """Return whether additional LLM trials are allowed under the configured cap."""

        return self._max_llm_trials is None or self._llm_trials < self._max_llm_trials

    def allow_llm(self, trials_completed: int) -> bool:
        if self._max_llm_trials is not None and self._llm_trials >= self._max_llm_trials:
            return False
        if self._mode == "init_only":
            return trials_completed < self._init_trials
        if self._mode == "mixed":
            if self._current_mix_ratio <= 0.0:
                return False
            if self._current_mix_ratio >= 1.0:
                return True
            return self._rng.random() < self._current_mix_ratio
        return self._mode != "off"

    def record_trial(self, used_llm: bool) -> None:
        if used_llm:
            self._llm_trials += 1

    def update_mix_ratio(self, new_ratio: float) -> float:
        """Clamp and update the live LLM mix ratio."""

        clamped = min(1.0, max(self._mix_ratio_floor, float(new_ratio)))
        self._current_mix_ratio = clamped
        return self._current_mix_ratio

    def decay_mix_ratio(self) -> float:
        """Reduce the mix ratio using the configured decay factor."""

        target = self._current_mix_ratio * self._mix_ratio_decay
        return self.update_mix_ratio(target)


@dataclass
class LLMUsageDecision:
    """Outcome of an adaptive LLM usage update."""

    ratio: float | None = None
    force_llm: bool = False
    reason: str | None = None
    hypervolume: float | None = None


class LLMUsageOptimizer:
    """Bandit-like controller that adapts LLM usage probability and triggers."""

    def __init__(
        self,
        cfg: Mapping[str, Any] | None,
        *,
        direction_names: Sequence[str],
        no_improve_patience: int | None,
        seed: int | None,
    ) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("adaptive_usage", True))
        self._floor = max(0.0, float(cfg.get("mix_ratio_floor", 0.1)))
        ceiling_value = cfg.get("adaptive_max_ratio", 0.8)
        self._ceiling = min(1.0, max(self._floor, float(ceiling_value)))
        self._decay = max(0.0, min(1.0, float(cfg.get("mix_ratio_decay", 0.5))))
        prior = float(cfg.get("adaptive_usage_prior", 0.2))
        self._bandit_prior = prior if prior > 0 else 0.1
        self._stagnation_boost = max(0.0, float(cfg.get("stagnation_boost", 0.2)))
        cooldown = int(cfg.get("adaptive_cooldown_trials", 2))
        self._cooldown_trials = cooldown if cooldown >= 0 else 0
        stagnation_value = cfg.get("stagnation_trials")
        if stagnation_value is None:
            derived = int(math.ceil((no_improve_patience or 8) * 0.6))
            stagnation_value = max(4, derived)
        self._stagnation_trials = int(stagnation_value) if stagnation_value else None
        pareto_patience = int(cfg.get("pareto_stagnation_trials", 4))
        self._pareto_patience = pareto_patience if pareto_patience > 0 else None
        hv_window = int(cfg.get("hv_plateau_window", 3))
        hv_window = hv_window if hv_window > 0 else 3
        hv_interval = int(cfg.get("hv_guard_interval", 6))
        hv_interval = hv_interval if hv_interval > 0 else 6
        self._hv_patience = hv_interval * hv_window
        self._hv_tol = max(0.0, float(cfg.get("hv_plateau_rel_tol", 0.01)))
        self._hv_samples = int(cfg.get("hv_guard_samples", 1200) or 1200)
        self._hv_interval = hv_interval
        self._seed = seed if isinstance(seed, int) else None
        self._rng = random.Random(seed)
        self._directions = list(direction_names)

        start_ratio = cfg.get("mix_ratio", self._floor)
        self._current_ratio = min(self._ceiling, max(self._floor, float(start_ratio)))
        self._llm_success = 0
        self._llm_trials = 0
        self._baseline_success = 0
        self._baseline_trials = 0
        self._last_force: int | None = None
        self._hv_best: float | None = None
        self._hv_best_trial: int | None = None
        self._multiobjective = len(direction_names) > 1

    @property
    def current_ratio(self) -> float:
        return self._current_ratio

    def apply_external_ratio(self, ratio: float) -> float:
        """Synchronise the internal ratio with an external adjustment."""

        clamped = min(self._ceiling, max(self._floor, float(ratio)))
        self._current_ratio = clamped
        return self._current_ratio

    def update_after_trial(
        self,
        *,
        study: optuna.study.Study | None,
        trials_completed: int,
        is_llm_trial: bool,
        improved: bool,
        pareto_improved: bool,
        no_improve_counter: int,
        pareto_no_improve_counter: int | None,
        hypervolume: float | None = None,
    ) -> LLMUsageDecision:
        if not self.enabled:
            return LLMUsageDecision()

        ratio_before = self._current_ratio
        if is_llm_trial:
            self._llm_trials += 1
            if improved:
                self._llm_success += 1
                self._boost_ratio()
            else:
                self._decay_ratio()
        else:
            self._baseline_trials += 1
            if improved:
                self._baseline_success += 1
                self._decay_ratio()

        hv_value = self._maybe_track_hypervolume(
            study=study,
            trials_completed=trials_completed,
            hypervolume=hypervolume,
        )
        reason = self._stagnation_reason(
            trials_completed=trials_completed,
            no_improve_counter=no_improve_counter,
            pareto_no_improve_counter=pareto_no_improve_counter,
        )
        force_llm = False
        if reason is not None:
            force_llm = True
            self._last_force = trials_completed
            boosted = max(self._current_ratio, self._floor + self._stagnation_boost)
            self._current_ratio = min(self._ceiling, boosted)

        ratio_changed = not math.isclose(self._current_ratio, ratio_before, rel_tol=1e-9)
        return LLMUsageDecision(
            ratio=self._current_ratio if ratio_changed else None,
            force_llm=force_llm,
            reason=reason,
            hypervolume=hv_value,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decay_ratio(self) -> None:
        decayed = self._current_ratio * self._decay
        if decayed < self._current_ratio:
            self._current_ratio = max(self._floor, decayed)

    def _boost_ratio(self) -> None:
        target = self._bandit_target()
        if target > self._current_ratio:
            self._current_ratio = min(self._ceiling, target)

    def _bandit_target(self) -> float:
        llm_alpha = 1.0 + self._bandit_prior + self._llm_success
        llm_beta = 1.0 + self._bandit_prior + (self._llm_trials - self._llm_success)
        base_alpha = 1.0 + self._bandit_prior + self._baseline_success
        base_beta = 1.0 + self._bandit_prior + (
            self._baseline_trials - self._baseline_success
        )
        llm_score = self._rng.betavariate(llm_alpha, llm_beta)
        base_score = self._rng.betavariate(base_alpha, base_beta)
        if llm_score <= base_score:
            return self._floor
        advantage = llm_score - base_score
        span = self._ceiling - self._floor
        return max(self._floor, min(self._ceiling, self._floor + advantage * span))

    def _maybe_track_hypervolume(
        self,
        *,
        study: optuna.study.Study | None,
        trials_completed: int,
        hypervolume: float | None,
    ) -> float | None:
        hv_value = hypervolume
        if (
            hv_value is None
            and self._multiobjective
            and self._hv_interval > 0
            and trials_completed % self._hv_interval == 0
        ):
            hv_value = self._compute_hypervolume(study)

        if hv_value is None:
            return None

        if not math.isfinite(hv_value):
            return None

        if self._hv_best is None or hv_value > self._hv_best * (1.0 + self._hv_tol):
            self._hv_best = hv_value
            self._hv_best_trial = trials_completed
        return hv_value

    def _compute_hypervolume(self, study: optuna.study.Study | None) -> float | None:
        if study is None:
            return None
        trials = study.get_trials(deepcopy=False)
        if not trials:
            return None
        objective_count = len(self._directions) if self._directions else len(trials[0].values or [])
        if objective_count < 2:
            return None
        points = _collect_objective_points(study, objective_count)
        if not points:
            return None
        directions = self._directions
        if len(directions) > objective_count:
            directions = directions[:objective_count]
        if len(directions) < objective_count:
            directions = list(directions) + ["minimize"] * (objective_count - len(directions))
        pareto = _pareto_front_from_points(points, directions)
        return _approximate_hypervolume(
            pareto_points=pareto or points,
            all_points=points,
            direction_names=directions,
            seed=self._seed,
            samples=self._hv_samples,
        )

    def _stagnation_reason(
        self,
        *,
        trials_completed: int,
        no_improve_counter: int,
        pareto_no_improve_counter: int | None,
    ) -> str | None:
        if (
            self._last_force is not None
            and self._cooldown_trials > 0
            and trials_completed - self._last_force <= self._cooldown_trials
        ):
            return None

        if self._stagnation_trials is not None and no_improve_counter >= int(
            self._stagnation_trials
        ):
            return "best_plateau"

        if (
            self._pareto_patience is not None
            and pareto_no_improve_counter is not None
            and pareto_no_improve_counter >= self._pareto_patience
        ):
            return "pareto_plateau"

        if (
            self._hv_patience is not None
            and self._hv_best_trial is not None
            and trials_completed - self._hv_best_trial >= self._hv_patience
        ):
            return "hypervolume_plateau"

        return None


class DiversityGuard:
    """Enforce stratified coverage and minimum diversity for a structural parameter."""

    def __init__(
        self,
        cfg: Mapping[str, Any] | None,
        search_space: Mapping[str, Mapping[str, Any]],
        *,
        trial_budget: int,
    ) -> None:
        param_default = "n_layers"
        param_name = str(cfg.get("param", param_default) if cfg else param_default).strip()
        self.param_name = param_name or param_default
        default_enabled = self.param_name in search_space
        self.enabled = bool(cfg.get("enabled", default_enabled)) if cfg else default_enabled
        self._values = self._enumerate_values(search_space.get(self.param_name))
        self._trial_budget = max(1, int(trial_budget))
        self._window = max(1, int(cfg.get("window", 8) if cfg else 8))
        self._min_unique = max(1, int(cfg.get("min_unique", 2) if cfg else 2))
        fraction_value = float(cfg.get("stratified_fraction", 0.35) if cfg else 0.35)
        self._stratified_fraction = max(0.0, min(1.0, fraction_value))
        min_pareto = int(cfg.get("min_pareto_points", 3) if cfg else 3)
        self._min_pareto_points = min_pareto if min_pareto > 0 else None
        self._target_total = (
            min(trial_budget, int(math.ceil(trial_budget * self._stratified_fraction)))
            if self._stratified_fraction > 0
            else 0
        )
        self._recent: deque[int] = deque(maxlen=self._window)
        self._pending = 0
        self._completed = 0
        self._enqueued = 0
        self._rescue_pending = False
        self._cycle_index = 0

        if not self._values:
            self.enabled = False

    @property
    def min_pareto_points(self) -> int | None:
        return self._min_pareto_points if self.enabled else None

    def plan_enqueues(self, trials_completed: int) -> list[tuple[Dict[str, Any], str]]:
        """Return stratified or rescue payloads to enqueue."""

        if not self.enabled or not self._values:
            return []

        plans: list[tuple[Dict[str, Any], str]] = []
        if trials_completed + self._pending >= self._trial_budget:
            return plans
        target_so_far = self._target_for_progress(trials_completed + 1)
        while (
            self._target_total > 0
            and self._completed + self._pending < target_so_far
            and self._enqueued < self._target_total
            and trials_completed + self._pending < self._trial_budget
        ):
            payload = {self.param_name: self._next_cycle_value()}
            plans.append((payload, "stratified"))
            self._pending += 1
            self._enqueued += 1

        rescue_value = self._rescue_value()
        if rescue_value is not None and not self._rescue_pending:
            if trials_completed + self._pending >= self._trial_budget:
                return plans
            plans.append(({self.param_name: rescue_value}, "diversity_guard"))
            self._pending += 1
            self._rescue_pending = True

        return plans

    def record_trial(self, params: Mapping[str, Any], source: str | None) -> None:
        if not self.enabled:
            return

        if self.param_name in params:
            try:
                value = int(params[self.param_name])
            except (TypeError, ValueError):
                value = None
            if value is not None:
                self._recent.append(value)

        if source in {"stratified", "diversity_guard"}:
            if self._pending > 0:
                self._pending -= 1
            self._completed += 1
            if source == "diversity_guard":
                self._rescue_pending = False

    def should_pause_llm(self) -> bool:
        if not self.enabled or not self._values:
            return False
        if len(self._recent) < self._window:
            return False
        return len(set(self._recent)) < self._min_unique

    def _target_for_progress(self, upcoming_trial: int) -> int:
        if self._target_total <= 0:
            return 0
        target = math.ceil(upcoming_trial * self._stratified_fraction)
        return min(self._target_total, target)

    def _next_cycle_value(self) -> int:
        value = self._values[self._cycle_index % len(self._values)]
        self._cycle_index = (self._cycle_index + 1) % len(self._values)
        return value

    def _rescue_value(self) -> int | None:
        if len(self._recent) < self._window:
            return None
        if len(set(self._recent)) >= self._min_unique:
            return None
        counts = Counter(self._recent)
        missing = [value for value in self._values if value not in counts]
        if missing:
            return missing[0]
        return min(self._values, key=lambda value: counts.get(value, 0))

    @staticmethod
    def _enumerate_values(spec: Mapping[str, Any] | None) -> list[int]:
        if not isinstance(spec, Mapping):
            return []
        if str(spec.get("type", "")).lower() != "int":
            return []
        try:
            low = int(spec.get("low", 0))
            high = int(spec.get("high", low))
            step = int(spec.get("step", 1) or 1)
        except (TypeError, ValueError):
            return []
        if step <= 0 or high < low:
            return []
        values: list[int] = []
        current = low
        limit = 200
        while current <= high and len(values) < limit:
            values.append(current)
            current += step
        if values and values[-1] != high and len(values) < limit:
            values.append(high)
        return sorted(set(values))


class HypervolumeMixGuard:
    """Guardrail that backs off LLM usage when diversity metrics regress."""

    def __init__(
        self,
        cfg: Mapping[str, Any] | None,
        *,
        enabled: bool,
        min_pareto_points: int | None,
        seed: int | None,
    ) -> None:
        self.enabled = enabled
        self._interval = int(cfg.get("hv_guard_interval", 6) if cfg else 6)
        self._margin = float(cfg.get("hv_guard_margin", 0.0) if cfg else 0.0)
        self._samples = int(cfg.get("hv_guard_samples", 1200) if cfg else 1200)
        decay_value = float(cfg.get("mix_ratio_decay", 0.5) if cfg else 0.5)
        floor_value = float(cfg.get("mix_ratio_floor", 0.1) if cfg else 0.1)
        self._decay = min(1.0, max(0.0, decay_value))
        self._floor = min(1.0, max(0.0, floor_value))
        self._last_checked = 0
        self._seed = seed if isinstance(seed, int) else None
        self._min_pareto_points = min_pareto_points

        if self._interval <= 0:
            self._interval = 6
        if self._samples <= 0:
            self._samples = 1200

    def maybe_adjust(
        self,
        *,
        study: optuna.study.Study,
        direction_names: Sequence[str],
        trials_completed: int,
        current_mix_ratio: float | None,
    ) -> tuple[float, str | None, float | None, float | None] | None:
        if (
            not self.enabled
            or current_mix_ratio is None
            or trials_completed - self._last_checked < self._interval
        ):
            return None

        self._last_checked = trials_completed
        objective_count = len(direction_names)
        if objective_count < 2:
            return None

        all_points = _collect_objective_points(study, objective_count)
        baseline_points = _collect_objective_points(
            study,
            objective_count,
            exclude_sources={"llm"},
        )
        if not all_points:
            return None

        pareto_all = _pareto_front_from_points(all_points, direction_names)
        pareto_baseline = _pareto_front_from_points(baseline_points, direction_names)

        hv_all = _approximate_hypervolume(
            pareto_points=pareto_all or all_points,
            all_points=all_points,
            direction_names=direction_names,
            seed=self._seed,
            samples=self._samples,
        )
        hv_baseline = _approximate_hypervolume(
            pareto_points=pareto_baseline or baseline_points,
            all_points=baseline_points,
            direction_names=direction_names,
            seed=self._seed,
            samples=self._samples,
        )

        reason: str | None = None
        if (
            self._min_pareto_points is not None
            and len(pareto_all) < int(self._min_pareto_points)
        ):
            reason = "pareto_collapse"

        margin = max(self._margin, 0.0)
        if hv_baseline is not None and hv_all is not None and hv_all + margin < hv_baseline:
            reason = "hypervolume_drop"

        if reason is None:
            return None

        new_ratio = max(self._floor, current_mix_ratio * self._decay)
        if new_ratio < current_mix_ratio:
            return new_ratio, reason, hv_all, hv_baseline
        return None


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

    problem_type, problem_tags = infer_problem_type(config)
    artifacts_cfg = config.get("artifacts") or {}
    catalog_path = resolve_catalog_path(artifacts_cfg)
    run_root_value = artifacts_cfg.get("run_root")
    runs_root = Path(run_root_value).parent if run_root_value else Path("runs")
    try:
        collect_runs_to_catalog(
            runs_root=runs_root,
            catalog_path=catalog_path,
            problem_filter=problem_type,
        )
    except Exception as exc:
        print(f"[warning] Failed to refresh strategy catalog: {exc}")
    strategy_notes = load_strategy_notes(
        problem_type,
        catalog_path=catalog_path,
        top_k=3,
    )
    fewshot_examples = load_fewshot_examples(problem_type)
    context_note_parts = [f"problem_type={problem_type}"]
    for key, value in (problem_tags or {}).items():
        if value is None:
            continue
        context_note_parts.append(f"{key}={value}")
    context_note = "; ".join(context_note_parts)

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
    llm_trace_logger = (
        proposal_generator.trace_logger if proposal_generator is not None else None
    )
    pending_proposals: deque[tuple[Dict[str, Any], str]] = deque()
    proposal_lineage: Dict[str, Dict[str, Any]] = {}

    planner_agent = create_planner_agent(
        config.get("planner"),
        config.get("llm"),
        search_space=search_space,
        knowledge_hints=strategy_notes,
        fewshot_examples=fewshot_examples,
    )
    diversity_guard = DiversityGuard(
        config.get("diversity_guard"),
        search_space,
        trial_budget=settings.trial_budget,
    )
    hypervolume_guard = HypervolumeMixGuard(
        config.get("llm_guidance"),
        enabled=len(direction_names) > 1,
        min_pareto_points=diversity_guard.min_pareto_points,
        seed=seed if isinstance(seed, int) else None,
    )
    usage_optimizer = (
        LLMUsageOptimizer(
            config.get("llm_guidance"),
            direction_names=direction_names,
            no_improve_patience=settings.patience,
            seed=seed if isinstance(seed, int) else None,
        )
        if proposal_generator is not None
        else None
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
        history_notes=strategy_notes,
        notes=context_note,
    )
    force_llm_next = False
    pareto_no_improve_counter: int | None = 0 if len(direction_names) > 1 else None

    with TrialLogger(
        log_file,
        search_space.keys(),
        config["report"]["metrics"],
    ) as writer:
        start_ts = time.time()
        no_improve_counter = 0
        llm_paused_for_diversity = False

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

            forced_llm = force_llm_next
            force_llm_next = False

            diversity_plans = diversity_guard.plan_enqueues(trials_completed)
            for payload, source in diversity_plans:
                fingerprint = fingerprint_proposal(payload)
                proposal_lineage[fingerprint] = {"source": source, "llm": False}
                study.enqueue_trial(payload)
                if llm_trace_logger is not None:
                    llm_trace_logger.log_event(
                        kind="proposal_enqueued",
                        stage="diversity_guard",
                        trace_id=None,
                        data={
                            "fingerprint": fingerprint,
                            "proposal": payload,
                            "source": source,
                        },
                    )

            diversity_block = diversity_guard.should_pause_llm()
            if diversity_block and proposal_generator is not None and not llm_paused_for_diversity:
                print("[diversity] Pausing LLM proposals to restore structural coverage.")
            llm_paused_for_diversity = diversity_block

            llm_allowed = proposal_generator is not None and not diversity_block
            if (
                llm_allowed
                and llm_scheduler is not None
                and usage_optimizer is not None
                and llm_scheduler.mode == "mixed"
            ):
                llm_scheduler.update_mix_ratio(usage_optimizer.current_ratio)
            scheduler_allows = True
            if llm_allowed and llm_scheduler is not None:
                if forced_llm:
                    scheduler_allows = llm_scheduler.has_capacity()
                else:
                    scheduler_allows = llm_scheduler.allow_llm(trials_completed)
            llm_allowed = llm_allowed and scheduler_allows
            forced_llm = forced_llm and llm_allowed

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
                    for proposal in batch:
                        fingerprint = fingerprint_proposal(proposal)
                        meta = proposal_generator.describe_proposal(fingerprint) or {}
                        if "llm" not in meta:
                            meta["llm"] = True
                        proposal_lineage[fingerprint] = meta
                        pending_proposals.append((proposal, fingerprint))
                        if llm_trace_logger is not None:
                            llm_trace_logger.log_event(
                                kind="proposal_buffered",
                                stage="llm_guidance",
                                trace_id=meta.get("trace_id"),
                                data={
                                    "fingerprint": fingerprint,
                                    "source": meta.get("source"),
                                },
                            )
                if pending_proposals:
                    proposal, fingerprint = pending_proposals.popleft()
                    study.enqueue_trial(proposal)
                    if llm_trace_logger is not None:
                        meta = proposal_lineage.get(fingerprint, {})
                        llm_trace_logger.log_event(
                            kind="proposal_enqueued",
                            stage="llm_guidance",
                            trace_id=meta.get("trace_id"),
                            data={
                                "fingerprint": fingerprint,
                                "proposal": proposal,
                                "source": meta.get("source"),
                            },
                        )
            else:
                pending_proposals.clear()

            trial = study.ask()
            fixed_params = getattr(trial, "_fixed_params", {}) or {}
            proposal_fingerprint = (
                fingerprint_proposal(fixed_params) if fixed_params else None
            )
            lineage_meta = (
                proposal_lineage.get(proposal_fingerprint or "", {})
                if proposal_fingerprint
                else {}
            )
            is_llm_trial = bool(lineage_meta.get("llm", bool(fixed_params)))
            source_label = lineage_meta.get("source") if lineage_meta else None
            if source_label is None:
                source_label = "llm" if is_llm_trial else "sampler"
            trial.set_user_attr("proposal_source", source_label)
            trial.set_user_attr("llm_trial", bool(is_llm_trial))
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
            if is_llm_trial:
                llm_trials_used += 1
                if llm_scheduler is not None:
                    llm_scheduler.record_trial(True)
                if llm_trace_logger is not None:
                    meta = proposal_lineage.get(proposal_fingerprint or "", {})
                    llm_trace_logger.log_event(
                        kind="llm_trial_result",
                        stage="llm_guidance",
                        trace_id=meta.get("trace_id"),
                        data={
                            "fingerprint": proposal_fingerprint,
                            "trial": trial.number,
                            "params": fixed_params,
                            "metrics": metrics,
                            "values": list(objective_values),
                            "source": meta.get("source"),
                        },
                    )

            writer.log(trial.number, params, metrics)
            diversity_guard.record_trial(params, source_label)

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

            if pareto_no_improve_counter is not None:
                if pareto_improved:
                    pareto_no_improve_counter = 0
                else:
                    pareto_no_improve_counter += 1

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

            usage_decision = None
            if usage_optimizer is not None:
                usage_decision = usage_optimizer.update_after_trial(
                    study=study,
                    trials_completed=trials_completed,
                    is_llm_trial=is_llm_trial,
                    improved=improved,
                    pareto_improved=pareto_improved,
                    no_improve_counter=no_improve_counter,
                    pareto_no_improve_counter=pareto_no_improve_counter,
                )
                if (
                    usage_decision.ratio is not None
                    and llm_scheduler is not None
                    and llm_scheduler.mode == "mixed"
                ):
                    prev_ratio = llm_scheduler.current_mix_ratio
                    updated_ratio = llm_scheduler.update_mix_ratio(usage_decision.ratio)
                    if not math.isclose(updated_ratio, prev_ratio, rel_tol=1e-9):
                        note = usage_decision.reason or "bandit_gain"
                        print(
                            f"[llm-usage] mix {prev_ratio:.2f}->{updated_ratio:.2f} ({note})"
                        )
                if usage_decision is not None and usage_decision.force_llm:
                    force_llm_next = True
                    if usage_decision.reason:
                        print(f"[llm-usage] Triggering LLM due to {usage_decision.reason}")

            llm_context = _build_llm_context(
                metric_names=metric_names,
                directions=direction_names,
                best_params=best_params,
                best_metrics=best_metrics,
                trials_completed=trials_completed,
                history_notes=strategy_notes,
                notes=context_note,
            )

            if (
                hypervolume_guard.enabled
                and llm_scheduler is not None
                and llm_scheduler.mode == "mixed"
            ):
                hv_adjustment = hypervolume_guard.maybe_adjust(
                    study=study,
                    direction_names=direction_names,
                    trials_completed=trials_completed,
                    current_mix_ratio=llm_scheduler.current_mix_ratio,
                )
                if hv_adjustment is not None:
                    new_ratio, reason, hv_all, hv_baseline = hv_adjustment
                    previous_ratio = llm_scheduler.current_mix_ratio
                    updated_ratio = llm_scheduler.update_mix_ratio(new_ratio)
                    if usage_optimizer is not None:
                        usage_optimizer.apply_external_ratio(updated_ratio)
                    if updated_ratio < previous_ratio:
                        message = (
                            f"[diversity] {reason or 'hv_guard'}: "
                            f"LLM mix {previous_ratio:.2f}->{updated_ratio:.2f}"
                        )
                        if hv_all is not None and hv_baseline is not None:
                            message += (
                                f" (hv_all={_format_metric_value(hv_all)}, "
                                f"hv_no_llm={_format_metric_value(hv_baseline)})"
                            )
                        print(message)

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

    try:
        record_strategy_entry(
            problem_type=problem_type,
            config=config,
            summary=summary_payload or {},
            log_path=log_file,
            catalog_path=catalog_path,
        )
    except Exception as exc:  # noqa: BLE001 - catalog best-effort
        print(f"[warning] Failed to update strategy catalog: {exc}")

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
    if trial_failed and len(tuple_values) == 1:
        for step, value in enumerate(tuple_values):
            trial.report(value, step=step)

    trial.set_user_attr("metrics", dict(metrics))
    # Placeholders for future Pareto analytics.
    trial.set_user_attr("dominates", False)
    trial.set_user_attr("pareto_rank", None)

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
    *,
    include_sources: set[str] | None = None,
    exclude_sources: set[str] | None = None,
) -> List[List[float]]:
    points: List[List[float]] = []
    for trial in study.get_trials(deepcopy=False):
        if not trial.values or len(trial.values) < objective_count:
            continue
        if include_sources is not None or exclude_sources is not None:
            source = None
            attrs = getattr(trial, "user_attrs", {})
            if isinstance(attrs, Mapping):
                source = attrs.get("proposal_source")
            if include_sources is not None and source not in include_sources:
                continue
            if exclude_sources is not None and source in exclude_sources:
                continue
        values = list(trial.values[:objective_count])
        if not all(math.isfinite(value) for value in values):
            continue
        points.append(values)
    return points


def _pareto_front_from_points(
    points: Sequence[Sequence[float]],
    direction_names: Sequence[str],
) -> List[List[float]]:
    if not points:
        return []
    transformed = [_transform_objectives(point, direction_names) for point in points]
    front: list[List[float]] = []
    for idx, candidate in enumerate(transformed):
        dominated = False
        for jdx, other in enumerate(transformed):
            if idx == jdx:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            front.append(points[idx])
    return front


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
    history_notes: Sequence[str] | None = None,
    notes: str | None = None,
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
        history_notes=list(history_notes) if history_notes else [],
        notes=notes,
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


def _dominates(candidate: Sequence[float], other: Sequence[float]) -> bool:
    if len(candidate) != len(other):
        return False
    not_worse = all(c <= o for c, o in zip(candidate, other))
    strictly_better = any(c < o for c, o in zip(candidate, other))
    return not_worse and strictly_better


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
