"""Meta-level search strategy adjustments driven by LLM feedback."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from uuid import uuid4

import optuna

from .llm_guidance import create_llm_provider
from .llm_interfaces import (
    LLMHistoryMetric,
    LLMObjective,
    LLMRepresentativePoint,
    LLMRunContext,
)
from .pareto_summary import ParetoSummaryGenerator
from .llm_providers import (
    LLMExchangeLogger,
    LLMResult,
    Prompt,
    PromptMessage,
    ToolDefinition,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .llm_guidance import LLMProposalGenerator


_SUPPORTED_META_SAMPLERS = {
    "tpe",
    "random",
    "nsga2",
    "nsgaiii",
    "motpe",
    "moead",
    "mocma",
    "nevergrad",
    "de",
}


@dataclass
class TrialRecord:
    """Lightweight summary of a completed trial."""

    number: int
    value: float
    improved: bool
    params: Mapping[str, Any]
    metrics: Mapping[str, Any]


@dataclass
class SearchSettings:
    """Mutable container reflecting the live search configuration."""

    sampler: str
    max_trials: int
    trial_budget: int
    patience: int | None


@dataclass
class MetaAdjustment:
    """Parsed adjustment directives produced by the meta controller."""

    sampler: str | None = None
    rescale: Dict[str, float] = field(default_factory=dict)
    trial_budget: int | None = None
    max_trials: int | None = None
    patience: int | None = None
    guidance_directives: List[str] = field(default_factory=list)
    search_space_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: str | None = None
    source: str = "heuristic"


@dataclass
class PolicyCondition:
    """Declarative condition describing when to trigger a policy."""

    no_improve: int | None = None
    metric_below: Tuple[str, float] | None = None
    metric_above: Tuple[str, float] | None = None
    samplers: Set[str] | None = None
    min_trials: int | None = None


@dataclass
class PolicyRule:
    """Declarative policy describing when and how to adjust the search."""

    name: str | None
    condition: PolicyCondition
    sampler: str | None
    rescale: Dict[str, float]
    trial_budget: int | None
    max_trials: int | None
    patience: int | None
    notes: str | None
    cooldown_trials: int
    trigger_once: bool
    last_triggered: Optional[int] = None

    def can_trigger(self, *, current_trial: int) -> bool:
        if self.trigger_once and self.last_triggered is not None:
            return False
        if self.cooldown_trials > 0 and self.last_triggered is not None:
            if current_trial - self.last_triggered < self.cooldown_trials:
                return False
        return True

    def record_trigger(self, *, current_trial: int) -> None:
        self.last_triggered = current_trial

    def make_adjustment(self) -> MetaAdjustment:
        notes = self.notes
        if not notes and self.name:
            notes = self.name
        return MetaAdjustment(
            sampler=self.sampler,
            rescale=dict(self.rescale),
            trial_budget=self.trial_budget,
            max_trials=self.max_trials,
            patience=self.patience,
            notes=notes,
            source="policy",
        )


class MetaSearchAdjuster:
    """Periodically summarise progress and request strategy updates."""

    _SYSTEM_PROMPT = (
        "あなたはハイパーパラメータ探索を監督するメタ最適化コントローラです。"
        "提供されたサマリを読み、次バッチの探索方針をJSONで提示してください。"
        "追加の説明文は notes に含め、レスポンス全体は有効なJSONにしてください。"
    )

    def __init__(
        self,
        *,
        interval: int,
        summary_window: int,
        direction: optuna.study.StudyDirection,
        metric_name: str,
        search_space: Mapping[str, Mapping[str, Any]],
        llm_cfg: Mapping[str, Any] | None,
        seed: int | None,
        policies: Sequence[Mapping[str, Any]] | None = None,
        objectives: Sequence[str] | None = None,
        objective_directions: Sequence[optuna.study.StudyDirection] | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be positive")
        if summary_window <= 0:
            raise ValueError("summary_window must be positive")

        self._interval = interval
        self._summary_window = summary_window
        self._history: Deque[TrialRecord] = deque(maxlen=max(summary_window * 2, interval * 2))
        self._since_last = 0
        self._direction = direction
        self._metric_name = metric_name
        self._metric_name_lc = metric_name.lower()
        self._search_space = {name: dict(spec) for name, spec in search_space.items()}
        self._provider, self._usage_logger, self._trace_logger = create_llm_provider(llm_cfg)
        self._seed = seed
        self._policies = self._build_policies(policies or [])
        if objectives is None:
            objectives = [metric_name]
        if objective_directions is None:
            objective_directions = [direction]
        if len(objectives) != len(objective_directions):
            raise ValueError("objectives and objective_directions length mismatch")
        self._objective_names = list(objectives)
        self._objective_directions = list(objective_directions)
        self._objective_direction_map = {
            name.lower(): objective_directions[idx]
            for idx, name in enumerate(self._objective_names)
        }
        if len(self._objective_names) >= 2:
            self._pareto_summary = ParetoSummaryGenerator(
                self._objective_names, self._objective_directions
            )
        else:
            self._pareto_summary = None

    def register_trial(
        self,
        *,
        trial_number: int,
        value: float,
        improved: bool,
        params: Mapping[str, Any],
        metrics: Mapping[str, Any],
        best_value: float | None,
        best_params: Mapping[str, Any],
            trials_completed: int,
            settings: SearchSettings,
    ) -> MetaAdjustment | None:
        """Record the latest trial and, if due, return an adjustment suggestion."""

        self._history.append(
            TrialRecord(
                number=trial_number,
                value=value,
                improved=improved,
                params=dict(params),
                metrics=dict(metrics),
            )
        )
        self._since_last += 1

        if self._since_last < self._interval:
            return None

        self._since_last = 0

        policy_adjustment = self._evaluate_policies(
            best_value=best_value,
            trials_completed=trials_completed,
            settings=settings,
            latest_metrics=metrics,
        )
        if policy_adjustment is not None:
            return policy_adjustment

        if self._provider is not None:
            adjustment = self._request_plan_from_llm(
                best_value=best_value,
                best_params=best_params,
                trials_completed=trials_completed,
                settings=settings,
            )
            if adjustment is not None:
                adjustment.source = "llm"
                return adjustment

        return self._heuristic_adjustment(
            best_value=best_value,
            best_params=best_params,
            settings=settings,
        )

    # ------------------------------------------------------------------
    # Plan generation helpers
    # ------------------------------------------------------------------
    def _build_policies(self, policies: Sequence[Mapping[str, Any]]) -> List[PolicyRule]:
        rules: List[PolicyRule] = []
        for idx, payload in enumerate(policies):
            try:
                rule = self._parse_policy(payload, index=idx)
            except Exception:
                continue
            if rule is not None:
                rules.append(rule)
        return rules

    def _parse_policy(
        self, payload: Mapping[str, Any], *, index: int
    ) -> PolicyRule | None:
        when = payload.get("when")
        then = payload.get("then")
        if not isinstance(when, Mapping) or not isinstance(then, Mapping):
            return None

        no_improve = self._coerce_int(when.get("no_improve"))
        metric_below = self._parse_metric_condition(when.get("metric_below"))
        metric_above = self._parse_metric_condition(when.get("metric_above"))
        samplers = self._parse_sampler_set(when.get("sampler"))
        min_trials = self._coerce_int(when.get("min_trials"))

        condition = PolicyCondition(
            no_improve=no_improve,
            metric_below=metric_below,
            metric_above=metric_above,
            samplers=samplers,
            min_trials=min_trials,
        )

        rescale_payload = then.get("rescale") if isinstance(then.get("rescale"), Mapping) else None
        rescale = {str(k): float(v) for k, v in (rescale_payload or {}).items()}
        sampler = then.get("sampler")
        sampler_name = str(sampler).lower() if isinstance(sampler, str) else None
        if sampler_name is not None and sampler_name not in _SUPPORTED_META_SAMPLERS:
            sampler_name = None

        trial_budget = self._coerce_int(then.get("trial_budget"))
        max_trials = self._coerce_int(then.get("max_trials"))
        patience = self._coerce_int(then.get("patience"))
        notes = then.get("notes")
        note_text = str(notes).strip() if isinstance(notes, str) else None
        if note_text == "":
            note_text = None

        cooldown = self._coerce_int(payload.get("cooldown_trials")) or 0
        trigger_once = bool(payload.get("trigger_once", False))
        name = payload.get("name")
        name_text = str(name).strip() if isinstance(name, str) and name.strip() else None

        return PolicyRule(
            name=name_text,
            condition=condition,
            sampler=sampler_name,
            rescale=rescale,
            trial_budget=trial_budget,
            max_trials=max_trials,
            patience=patience,
            notes=note_text,
            cooldown_trials=max(0, cooldown),
            trigger_once=trigger_once,
        )

    def _parse_metric_condition(
        self, payload: Mapping[str, Any] | None
    ) -> Tuple[str, float] | None:
        if not isinstance(payload, Mapping):
            return None
        metric = payload.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            return None
        try:
            value = float(payload.get("value"))
        except (TypeError, ValueError):
            return None
        return (metric.strip().lower(), value)

    def _parse_sampler_set(self, payload: Any) -> Set[str] | None:
        if payload is None:
            return None
        samplers: Set[str] = set()
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            for item in payload:
                if isinstance(item, str) and item.strip():
                    sampler_name = item.strip().lower()
                    if sampler_name in _SUPPORTED_META_SAMPLERS:
                        samplers.add(sampler_name)
        elif isinstance(payload, str) and payload.strip():
            sampler_name = payload.strip().lower()
            if sampler_name in _SUPPORTED_META_SAMPLERS:
                samplers.add(sampler_name)
        return samplers or None

    def _coerce_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _evaluate_policies(
        self,
        *,
        best_value: float | None,
        trials_completed: int,
        settings: SearchSettings,
        latest_metrics: Mapping[str, Any],
    ) -> MetaAdjustment | None:
        if not self._policies:
            return None

        streak = self._count_no_improve()
        metric_summary = self._collect_metric_summary()
        latest_numeric = self._normalise_metrics(latest_metrics)

        for rule in self._policies:
            if not rule.can_trigger(current_trial=trials_completed):
                continue

            cond = rule.condition
            if cond.min_trials is not None and trials_completed < cond.min_trials:
                continue
            if cond.no_improve is not None and streak < cond.no_improve:
                continue
            if cond.samplers and settings.sampler not in cond.samplers:
                continue

            if cond.metric_below is not None:
                metric_name, threshold = cond.metric_below
                value = self._resolve_metric_value(
                    metric_name,
                    metric_summary,
                    latest_numeric,
                    best_value=best_value,
                    prefer="min",
                )
                if value is None or value >= threshold:
                    continue

            if cond.metric_above is not None:
                metric_name, threshold = cond.metric_above
                value = self._resolve_metric_value(
                    metric_name,
                    metric_summary,
                    latest_numeric,
                    best_value=best_value,
                    prefer="max",
                )
                if value is None or value <= threshold:
                    continue

            rule.record_trigger(current_trial=trials_completed)
            return rule.make_adjustment()

        return None

    def _count_no_improve(self) -> int:
        streak = 0
        for record in reversed(self._history):
            if record.improved:
                break
            streak += 1
        return streak

    def _collect_metric_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for record in self._history:
            for name, value in record.metrics.items():
                if not isinstance(name, str):
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                key = name.lower()
                stats = summary.setdefault(
                    key,
                    {
                        "latest": numeric,
                        "min": numeric,
                        "max": numeric,
                        "total": 0.0,
                        "count": 0,
                    },
                )
                stats["latest"] = numeric
                stats["min"] = min(stats["min"], numeric)
                stats["max"] = max(stats["max"], numeric)
                stats["total"] += numeric
                stats["count"] += 1
        for stats in summary.values():
            total = stats.pop("total", None)
            count = stats.get("count")
            if total is not None and count:
                stats["mean"] = total / count
        return summary

    def _normalise_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name, value in metrics.items():
            if not isinstance(name, str):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            values[name.lower()] = numeric
        return values

    def _resolve_metric_value(
        self,
        metric_name: str,
        metric_summary: Mapping[str, Mapping[str, float]],
        latest_metrics: Mapping[str, float],
        *,
        best_value: float | None,
        prefer: str,
    ) -> float | None:
        metric_key = metric_name.lower()
        if metric_key == self._metric_name_lc and best_value is not None:
            if prefer == "min" and self._direction == optuna.study.StudyDirection.MINIMIZE:
                return float(best_value)
            if prefer == "max" and self._direction == optuna.study.StudyDirection.MAXIMIZE:
                return float(best_value)

        stats = metric_summary.get(metric_key)
        if stats is not None:
            if prefer == "min":
                return stats.get("min")
            if prefer == "max":
                return stats.get("max")

        return latest_metrics.get(metric_key)

    def _request_plan_from_llm(
        self,
        *,
        best_value: float | None,
        best_params: Mapping[str, Any],
        trials_completed: int,
        settings: SearchSettings,
    ) -> MetaAdjustment | None:
        if self._provider is None:
            return None

        schema = self._adjustment_schema(trials_completed, settings)
        prompt = self._build_prompt(
            best_value,
            best_params,
            trials_completed,
            settings,
            schema=schema,
        )
        tool = ToolDefinition(
            name="meta_search_plan",
            description="Return adjustments to the search strategy using the provided schema.",
            parameters=schema,
        )
        params = {"temperature": 0.1, "json_mode": False}
        result: LLMResult | None = None
        error: str | None = None
        trace_id = f"meta-{uuid4().hex[:8]}"
        start = time.perf_counter()
        latency_ms: float | None = None
        try:
            result = self._provider.generate(
                prompt,
                temperature=0.1,
                json_mode=False,
                system=self._SYSTEM_PROMPT,
                tool=tool,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            error = f"{exc.__class__.__name__}: {exc}"
            self._log_exchange(
                prompt=prompt,
                system=self._SYSTEM_PROMPT,
                tool=tool,
                params=params,
                result=None,
                error=error,
                stage="meta_search",
                trace_id=trace_id,
                latency_ms=latency_ms,
            )
            return None

        self._log_usage(result)
        parsed = self._parse_plan(result.content, trials_completed, settings)
        parse_info: Dict[str, Any] = {
            "status": "ok" if parsed is not None else "error",
        }
        self._log_exchange(
            prompt=prompt,
            system=self._SYSTEM_PROMPT,
            tool=tool,
            params=params,
            result=result,
            stage="meta_search",
            trace_id=trace_id,
            latency_ms=latency_ms,
            parse=parse_info,
            decision={
                "trials_completed": trials_completed,
                "settings": settings.__dict__,
            },
        )
        return parsed

    def _heuristic_adjustment(
        self,
        *,
        best_value: float | None,
        best_params: Mapping[str, Any],
        settings: SearchSettings,
    ) -> MetaAdjustment | None:
        if not self._history:
            return None

        recent = list(self._history)[-self._interval :]
        if not any(record.improved for record in recent):
            if settings.sampler == "tpe":
                return MetaAdjustment(
                    sampler="random",
                    notes="改善が停滞したためランダムサンプラーに切替",
                    source="heuristic",
                )
            shrink_targets = self._find_numeric_params(best_params)
            if shrink_targets:
                return MetaAdjustment(
                    rescale={name: 0.5 for name in shrink_targets},
                    notes="改善が停滞したためベスト付近へ探索範囲を縮小",
                    source="heuristic",
                )
            if settings.patience is not None and settings.patience > 1:
                return MetaAdjustment(
                    patience=max(1, settings.patience - 1),
                    notes="改善が停滞したため早期停止を厳格化",
                    source="heuristic",
                )
        else:
            shrink_targets = self._find_numeric_params(best_params)
            if shrink_targets:
                return MetaAdjustment(
                    rescale={name: 0.7 for name in shrink_targets},
                    notes="改善が得られたため探索範囲を緩やかに縮小",
                    source="heuristic",
                )
        return None

    def _find_numeric_params(self, best_params: Mapping[str, Any]) -> Sequence[str]:
        numeric: list[str] = []
        for name, spec in self._search_space.items():
            if name not in best_params:
                continue
            param_type = spec.get("type")
            if param_type in {"float", "int"}:
                numeric.append(name)
        return numeric

    def _build_prompt(
        self,
        best_value: float | None,
        best_params: Mapping[str, Any],
        trials_completed: int,
        settings: SearchSettings,
        *,
        schema: Mapping[str, Any],
    ) -> Prompt:
        context = self._build_run_context(
            best_value=best_value,
            best_params=best_params,
            settings=settings,
            trials_completed=trials_completed,
        )
        context_json = context.to_json(indent=2)
        quantum_summary = self._build_quantum_summary()
        example_payload = {
            "sampler": settings.sampler,
            "rescale": {"param_name": 0.5},
            "trial_budget": settings.trial_budget,
            "max_trials": settings.max_trials,
            "patience": settings.patience,
            "guidance": {
                "directives": ["次の20トライアルではV型アンサッツを試す"],
                "search_space": {"max_gate": {"high": 18}},
            },
            "notes": "Focus on promising regions while keeping exploration.",
        }
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

        lines = [
            "以下は共通フォーマットで提供される最適化状態の JSON です。",
            context_json,
            "---",
            "量子回路向けの直近評価サマリ（fidelity/depth/gate_count, 失敗率）も参考にしてください。",
        ]
        if quantum_summary is not None:
            lines.extend(
                [
                    f"直近 {quantum_summary['window']} 試行の量子評価サマリ:",
                    json.dumps(quantum_summary, ensure_ascii=False, indent=2),
                ]
            )
        lines.extend(
            [
                "回路構造の変更方針、ゲート数を増減すべきか、チェーン/全結合など別アーキテクチャを試すべきかを日本語で短く述べてください。",
                "そのうえで、次の探索方針を JSON で返してください。guidance.search_space には特定パラメタの上限変更など一時的な制約を記述できます。",
                "guidance.directives には「次の20トライアルでは○○型アンサッツ」「CNOTの配置規則を変更」といった指示を短く列挙してください。",
                f"目的: {self._metric_name} を {self._direction.name.lower()}",
                f"次の {self._interval} 試行に向けた探索方針を指示してください。",
                "必ず JSON オブジェクトのみを出力し、キーは sampler, rescale, trial_budget, max_trials, patience, guidance, notes を使用してください。",
                "rescale はパラメタ名をキー、0.05〜1.0 の縮小係数を値とするオブジェクトです。不要な場合は空オブジェクトにしてください。",
                "guidance.search_space は {\"param\": {\"low\": .., \"high\": ..}} のように一時的な範囲変更を指定します。",
                "trial_budget は総試行数上限、max_trials は絶対上限、patience は改善なし許容回数を指定します。",
                "応答は meta_search_plan 関数の引数として Schema に完全準拠する JSON のみを返してください。",
                "Schema:",
                schema_text,
                "例:",
                json.dumps(example_payload, ensure_ascii=False, indent=2),
            ]
        )
        content = "\n".join(lines)
        return Prompt(messages=[PromptMessage(role="user", content=content)])

    def _adjustment_schema(
        self, trials_completed: int, settings: SearchSettings
    ) -> Dict[str, Any]:
        minimum_trials = max(0, trials_completed)
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sampler": {
                    "type": "string",
                    "enum": sorted(_SUPPORTED_META_SAMPLERS),
                },
                "rescale": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "number",
                        "minimum": 0.05,
                        "maximum": 1.0,
                    },
                },
                "trial_budget": {
                    "type": "integer",
                    "minimum": minimum_trials,
                },
                "max_trials": {
                    "type": "integer",
                    "minimum": minimum_trials,
                },
                "patience": {
                    "anyOf": [
                        {"type": "integer", "minimum": 0},
                        {"type": "null"},
                    ]
                },
                "guidance": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "directives": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "search_space": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "low": {"type": "number"},
                                    "high": {"type": "number"},
                                    "step": {"type": "number"},
                                    "choices": {"type": "array", "items": {}},
                                    "default": {},
                                },
                            },
                        },
                    },
                },
                "notes": {"type": "string"},
            },
        }

    def _build_run_context(
        self,
        *,
        best_value: float | None,
        best_params: Mapping[str, Any],
        settings: SearchSettings,
        trials_completed: int,
    ) -> LLMRunContext:
        objectives = [
            LLMObjective(name=name, direction=direction.name.lower())
            for name, direction in zip(self._objective_names, self._objective_directions)
        ]
        current_best: List[LLMRepresentativePoint] = []
        history_notes: List[str] = []
        if self._pareto_summary is not None:
            pareto_points, pareto_notes = self._pareto_summary.summarise(
                list(self._history)
            )
            if pareto_points:
                current_best.extend(pareto_points)
            history_notes.extend(pareto_notes)
        if best_value is not None or best_params:
            values: Dict[str, Any] = {}
            if best_value is not None:
                values[self._metric_name] = best_value
            current_best.append(
                LLMRepresentativePoint(
                    label="Best primary objective",
                    trial=self._best_trial_number(best_value),
                    values=values,
                    params=dict(best_params),
                )
            )

        metric_summary = self._collect_metric_summary()
        history_window = min(len(self._history), self._summary_window)
        history_summary: List[LLMHistoryMetric] = []
        for name, stats in metric_summary.items():
            direction = None
            mapped_direction = self._objective_direction_map.get(name)
            if mapped_direction is not None:
                direction = mapped_direction.name.lower()
            count_value = stats.get("count")
            history_summary.append(
                LLMHistoryMetric(
                    name=name,
                    direction=direction,
                    window=history_window or None,
                    latest=stats.get("latest"),
                    minimum=stats.get("min"),
                    maximum=stats.get("max"),
                    mean=stats.get("mean"),
                    count=int(count_value) if count_value else None,
                )
            )

        remaining_budget = max(0, settings.trial_budget - trials_completed)
        patience_text = (
            "disabled" if settings.patience is None else str(settings.patience)
        )
        notes = (
            f"sampler={settings.sampler}, remaining_budget={remaining_budget}, patience={patience_text}"
        )

        return LLMRunContext(
            objectives=objectives,
            current_best=current_best,
            history_summary=history_summary,
            history_notes=history_notes,
            trials_completed=trials_completed,
            notes=notes,
        )

    def _best_trial_number(self, best_value: float | None) -> int | None:
        if best_value is None:
            return None
        for record in self._history:
            if math.isclose(record.value, best_value, rel_tol=1e-9, abs_tol=1e-9):
                return record.number
        return None

    def _build_quantum_summary(self) -> Dict[str, Any] | None:
        """Summarise recent trials for circuit-specific signals."""

        if not self._history:
            return None

        window = min(len(self._history), self._summary_window)
        recent = list(self._history)[-window:]

        metric_keys = {
            "metric_fidelity": "fidelity",
            "metric_depth": "depth",
            "metric_gate_count": "gate_count",
            "metric_t_gate_count": "t_gate_count",
            "metric_error_probability": "error_probability",
        }

        metric_stats: Dict[str, Dict[str, float]] = {}
        for key, label in metric_keys.items():
            stats = self._summarise_metric(recent, metric_name=key)
            if stats:
                metric_stats[label] = stats

        failure_count = sum(1 for record in recent if self._trial_failed(record.metrics))

        if not metric_stats and failure_count == 0:
            return None

        summary: Dict[str, Any] = {"window": window, "metrics": metric_stats}
        summary["failures"] = {
            "count": failure_count,
            "rate": failure_count / float(window),
        }
        return summary

    def _summarise_metric(
        self, records: Sequence[TrialRecord], *, metric_name: str
    ) -> Dict[str, float]:
        values: list[float] = []
        for record in records:
            metrics = self._normalise_metrics(record.metrics)
            if metric_name not in metrics:
                continue
            try:
                numeric = float(metrics[metric_name])
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric) or math.isinf(numeric):
                continue
            values.append(numeric)
        if not values:
            return {}
        return {
            "latest": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    def _trial_failed(self, metrics: Mapping[str, Any]) -> bool:
        status = metrics.get("status")
        if isinstance(status, str) and status.lower() != "ok":
            return True
        timed_out = metrics.get("timed_out")
        if isinstance(timed_out, bool):
            return timed_out
        return bool(timed_out)

    def _parse_plan(
        self,
        payload: str,
        trials_completed: int,
        settings: SearchSettings,
    ) -> MetaAdjustment | None:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, MutableMapping):
            return None

        allowed_keys = {
            "sampler",
            "rescale",
            "trial_budget",
            "max_trials",
            "patience",
            "guidance",
            "notes",
        }
        if any(key not in allowed_keys for key in data.keys()):
            return None

        adjustment = MetaAdjustment()

        sampler = data.get("sampler")
        if isinstance(sampler, str):
            sampler_lc = sampler.lower()
            if sampler_lc in _SUPPORTED_META_SAMPLERS:
                adjustment.sampler = sampler_lc

        rescale = data.get("rescale")
        if isinstance(rescale, Mapping):
            for name, value in rescale.items():
                try:
                    factor = float(value)
                except (TypeError, ValueError):
                    continue
                if not (0.05 <= factor <= 1.0):
                    continue
                adjustment.rescale[name] = factor

        trial_budget = data.get("trial_budget")
        if isinstance(trial_budget, (int, float)):
            trial_budget_int = int(trial_budget)
            if trial_budget_int >= trials_completed:
                adjustment.trial_budget = trial_budget_int

        max_trials = data.get("max_trials")
        if isinstance(max_trials, (int, float)):
            max_trials_int = int(max_trials)
            if max_trials_int >= trials_completed:
                adjustment.max_trials = max_trials_int

        patience = data.get("patience")
        if patience is None:
            adjustment.patience = None
        elif isinstance(patience, (int, float)):
            patience_int = int(patience)
            if patience_int >= 0:
                adjustment.patience = patience_int

        guidance = data.get("guidance")
        if isinstance(guidance, Mapping):
            directives_payload = guidance.get("directives")
            if isinstance(directives_payload, Sequence) and not isinstance(
                directives_payload, (str, bytes, bytearray)
            ):
                directives: list[str] = []
                for entry in directives_payload:
                    text = str(entry).strip()
                    if text:
                        directives.append(text)
                adjustment.guidance_directives = directives

            overrides_payload = guidance.get("search_space")
            if isinstance(overrides_payload, Mapping):
                for name, override in overrides_payload.items():
                    if not isinstance(override, Mapping):
                        continue
                    cleaned: Dict[str, Any] = {}
                    for key in ("low", "high", "step", "choices", "default"):
                        if key in override:
                            cleaned[key] = override[key]
                    if cleaned:
                        adjustment.search_space_overrides[str(name)] = cleaned

        notes = data.get("notes")
        if isinstance(notes, str) and notes.strip():
            adjustment.notes = notes.strip()

        if (
            adjustment.sampler
            or adjustment.rescale
            or adjustment.trial_budget is not None
            or adjustment.max_trials is not None
            or adjustment.patience is not None
            or adjustment.guidance_directives
            or adjustment.search_space_overrides
            or adjustment.notes
        ):
            return adjustment
        return None

    def _log_usage(self, result: LLMResult) -> None:
        if self._usage_logger is not None:
            self._usage_logger.log(result.usage)

    def _log_exchange(
        self,
        *,
        prompt: Prompt,
        system: str | None,
        tool: ToolDefinition | None,
        params: Mapping[str, Any],
        result: LLMResult | None,
        error: str | None = None,
        stage: str | None = None,
        trace_id: str | None = None,
        latency_ms: float | None = None,
        parse: Mapping[str, Any] | None = None,
        decision: Mapping[str, Any] | None = None,
    ) -> None:
        if self._trace_logger is None:
            return
        self._trace_logger.log(
            prompt=prompt,
            system=system,
            tool=tool,
            params=params,
            result=result,
            error=error,
            stage=stage,
            trace_id=trace_id,
            latency_ms=latency_ms,
            parse=parse,
            decision=decision,
        )


def create_meta_search_adjuster(
    meta_cfg: Mapping[str, Any] | None,
    llm_cfg: Mapping[str, Any] | None,
    *,
    direction: optuna.study.StudyDirection,
    metric_name: str,
    metric_names: Sequence[str],
    directions: Sequence[optuna.study.StudyDirection],
    search_space: Mapping[str, Mapping[str, Any]],
    seed: int | None,
) -> MetaSearchAdjuster | None:
    """Factory helper mirroring :func:`create_proposal_generator`."""

    if meta_cfg is None or not meta_cfg.get("enabled"):
        return None

    interval = int(meta_cfg.get("interval", 10))
    summary_window = int(meta_cfg.get("summary_trials", interval))

    return MetaSearchAdjuster(
        interval=max(1, interval),
        summary_window=max(1, summary_window),
        direction=direction,
        metric_name=metric_name,
        search_space=search_space,
        llm_cfg=llm_cfg,
        seed=seed,
        policies=meta_cfg.get("policies"),
        objectives=list(metric_names),
        objective_directions=list(directions),
    )


def apply_meta_adjustment(
    adjustment: MetaAdjustment,
    *,
    study: optuna.study.Study,
    search_cfg: MutableMapping[str, Any],
    search_space: MutableMapping[str, MutableMapping[str, Any]],
    best_params: Mapping[str, Any],
    settings: SearchSettings,
    trials_completed: int,
    seed: int | None,
    sampler_builder: Callable[[Mapping[str, Any], int | None], optuna.samplers.BaseSampler],
    proposal_generator: "LLMProposalGenerator" | None = None,
) -> list[str]:
    """Apply an adjustment and return human-readable change logs."""

    messages: list[str] = []

    def _apply_override(spec: MutableMapping[str, Any], payload: Mapping[str, Any]) -> str | None:
        param_type = spec.get("type")
        if param_type == "float":
            try:
                low = float(payload.get("low", spec.get("low", 0.0)))
                high = float(payload.get("high", spec.get("high", 0.0)))
            except (TypeError, ValueError):
                return None
            if high <= low:
                return None
            spec["low"] = low
            spec["high"] = high
            step_value = payload.get("step")
            if step_value is not None:
                try:
                    step = float(step_value)
                except (TypeError, ValueError):
                    step = None
                if step is not None and step > 0:
                    spec["step"] = step
            return f"range=[{low}, {high}]"
        if param_type == "int":
            try:
                low = int(payload.get("low", spec.get("low", 0)))
                high = int(payload.get("high", spec.get("high", 0)))
            except (TypeError, ValueError):
                return None
            if high <= low:
                return None
            spec["low"] = low
            spec["high"] = high
            step_value = payload.get("step")
            if step_value is not None:
                try:
                    step = int(step_value)
                except (TypeError, ValueError):
                    step = None
                if step is not None and step > 0:
                    spec["step"] = step
            return f"range=[{low}, {high}]"
        if param_type == "categorical":
            choices = payload.get("choices")
            if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes, bytearray)):
                new_choices = [item for item in choices if item is not None]
                if new_choices:
                    spec["choices"] = list(new_choices)
                    return f"choices={len(new_choices)}"
        if param_type == "llm_only":
            default = payload.get("default")
            if isinstance(default, str) and default.strip():
                spec["default"] = default.strip()
                return "default updated"
        return None

    if adjustment.sampler and adjustment.sampler != settings.sampler:
        new_cfg = dict(search_cfg)
        new_cfg["sampler"] = adjustment.sampler
        try:
            sampler = sampler_builder(new_cfg, seed)
        except Exception:
            pass
        else:
            study.sampler = sampler
            settings.sampler = adjustment.sampler
            search_cfg["sampler"] = adjustment.sampler
            messages.append(f"sampler -> {adjustment.sampler}")

    if adjustment.rescale:
        rescaled: list[str] = []
        for name, factor in adjustment.rescale.items():
            spec = search_space.get(name)
            if spec is None:
                continue
            if name not in best_params:
                continue
            try:
                factor_value = float(factor)
            except (TypeError, ValueError):
                continue
            if not (0.05 <= factor_value <= 1.0):
                continue
            param_type = spec.get("type")
            if param_type == "float":
                low = float(spec.get("low", 0.0))
                high = float(spec.get("high", 0.0))
                if high <= low:
                    continue
                center = float(best_params[name])
                width = high - low
                new_width = max(width * factor_value, 1e-9)
                half = new_width / 2
                new_low = max(low, center - half)
                new_high = min(high, center + half)
                if new_high - new_low < 1e-9:
                    continue
                spec["low"] = new_low
                spec["high"] = new_high
                rescaled.append(f"{name}×{factor_value:.2f}")
            elif param_type == "int":
                low = int(spec.get("low", 0))
                high = int(spec.get("high", 0))
                if high <= low:
                    continue
                center = int(best_params[name])
                span = max(1, int(round((high - low) * factor_value)))
                new_low = max(low, center - span // 2)
                new_high = min(high, new_low + span)
                if new_high <= new_low:
                    new_high = min(high, max(new_low + 1, center + 1))
                spec["low"] = int(new_low)
                spec["high"] = int(new_high)
                rescaled.append(f"{name}×{factor_value:.2f}")
        if rescaled:
            messages.append("rescale: " + ", ".join(rescaled))

    if adjustment.search_space_overrides:
        overrides_applied: list[str] = []
        for name, payload in adjustment.search_space_overrides.items():
            spec = search_space.get(name)
            if spec is None:
                continue
            updated = _apply_override(spec, payload)
            if updated:
                overrides_applied.append(f"{name}: {updated}")
        if overrides_applied:
            messages.append("search_space: " + ", ".join(overrides_applied))
        if proposal_generator is not None:
            generator_logs = proposal_generator.apply_search_space_overrides(
                adjustment.search_space_overrides
            )
            if generator_logs:
                messages.append("llm_guidance: " + ", ".join(generator_logs))

    if adjustment.guidance_directives:
        if proposal_generator is not None:
            proposal_generator.apply_meta_directives(adjustment.guidance_directives)
        messages.append("guidance updated")

    if adjustment.max_trials is not None:
        new_max = max(trials_completed, int(adjustment.max_trials))
        if new_max != settings.max_trials:
            settings.max_trials = new_max
            if settings.trial_budget > settings.max_trials:
                settings.trial_budget = settings.max_trials
            messages.append(f"max_trials -> {settings.max_trials}")

    if adjustment.trial_budget is not None:
        new_budget = max(trials_completed, int(adjustment.trial_budget))
        new_budget = min(new_budget, settings.max_trials)
        if new_budget != settings.trial_budget:
            settings.trial_budget = new_budget
            messages.append(f"trial_budget -> {settings.trial_budget}")

    if adjustment.patience is not None:
        new_patience = None if adjustment.patience <= 0 else int(adjustment.patience)
        if new_patience != settings.patience:
            settings.patience = new_patience
            label = "disabled" if new_patience is None else str(new_patience)
            messages.append(f"patience -> {label}")

    if adjustment.notes:
        messages.append(adjustment.notes)

    return messages


__all__ = [
    "MetaAdjustment",
    "MetaSearchAdjuster",
    "SearchSettings",
    "apply_meta_adjustment",
    "create_meta_search_adjuster",
]
