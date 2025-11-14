"""Meta-level search strategy adjustments driven by LLM feedback."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from typing import Any, Callable, Deque, Dict, Mapping, MutableMapping, Sequence

import optuna

from .llm_guidance import create_llm_provider
from .llm_providers import LLMResult, Prompt, PromptMessage, ToolDefinition


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
    notes: str | None = None
    source: str = "heuristic"


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
        self._search_space = {name: dict(spec) for name, spec in search_space.items()}
        self._provider, self._usage_logger = create_llm_provider(llm_cfg)
        self._seed = seed

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
        try:
            result = self._provider.generate(
                prompt,
                temperature=0.1,
                json_mode=False,
                system=self._SYSTEM_PROMPT,
                tool=tool,
            )
        except Exception:
            return None

        self._log_usage(result)
        return self._parse_plan(result.content, trials_completed, settings)

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
        summary = self._summarise_history(best_value, best_params, settings, trials_completed)
        example_payload = {
            "sampler": settings.sampler,
            "rescale": {"param_name": 0.5},
            "trial_budget": settings.trial_budget,
            "max_trials": settings.max_trials,
            "patience": settings.patience,
            "notes": "Focus on promising regions while keeping exploration.",
        }
        schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

        lines = [
            "以下は最新の試行サマリです。",
            "---",
            summary,
            "---",
            f"目的: {self._metric_name} を {self._direction.name.lower()}",
            f"次の {self._interval} 試行に向けた探索方針を指示してください。",
            "必ず JSON オブジェクトのみを出力し、キーは sampler, rescale, trial_budget, max_trials, patience, notes を使用してください。",
            "rescale はパラメタ名をキー、0.05〜1.0 の縮小係数を値とするオブジェクトです。不要な場合は空オブジェクトにしてください。",
            "trial_budget は総試行数上限、max_trials は絶対上限、patience は改善なし許容回数を指定します。",
            "応答は meta_search_plan 関数の引数として Schema に完全準拠する JSON のみを返してください。",
            "Schema:",
            schema_text,
            "例:",
            json.dumps(example_payload, ensure_ascii=False, indent=2),
        ]

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
                    "enum": ["tpe", "random"],
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
                "notes": {"type": "string"},
            },
        }

    def _summarise_history(
        self,
        best_value: float | None,
        best_params: Mapping[str, Any],
        settings: SearchSettings,
        trials_completed: int,
    ) -> str:
        lines: list[str] = []
        lines.append(f"現在のサンプラー: {settings.sampler}")
        lines.append(f"残り試行予算: {max(0, settings.trial_budget - trials_completed)}")
        if settings.patience is None:
            lines.append("早期停止: 無効")
        else:
            lines.append(f"早期停止許容 (patience): {settings.patience}")
        if best_value is not None:
            lines.append(f"ベスト値: {best_value}")
        if best_params:
            formatted = ", ".join(f"{k}={v}" for k, v in best_params.items())
            lines.append(f"ベストパラメタ: {formatted}")

        lines.append("直近試行:")
        recent = list(self._history)[-self._summary_window :]
        for record in recent:
            metric_val = record.metrics.get(self._metric_name)
            metric_repr = metric_val if metric_val is not None else record.value
            flag = "*" if record.improved else "-"
            lines.append(
                f"  {flag} Trial {record.number}: value={record.value} metric={metric_repr} params={self._short_params(record.params)}"
            )
        return "\n".join(lines)

    def _short_params(self, params: Mapping[str, Any]) -> str:
        items = list(params.items())
        if len(items) <= 3:
            return ", ".join(f"{k}={v}" for k, v in items)
        head = ", ".join(f"{k}={v}" for k, v in items[:3])
        return f"{head}, ..."

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
            "notes",
        }
        if any(key not in allowed_keys for key in data.keys()):
            return None

        adjustment = MetaAdjustment()

        sampler = data.get("sampler")
        if isinstance(sampler, str):
            sampler_lc = sampler.lower()
            if sampler_lc in {"tpe", "random"}:
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

        notes = data.get("notes")
        if isinstance(notes, str) and notes.strip():
            adjustment.notes = notes.strip()

        if (
            adjustment.sampler
            or adjustment.rescale
            or adjustment.trial_budget is not None
            or adjustment.max_trials is not None
            or adjustment.patience is not None
            or adjustment.notes
        ):
            return adjustment
        return None

    def _log_usage(self, result: LLMResult) -> None:
        if self._usage_logger is not None:
            self._usage_logger.log(result.usage)


def create_meta_search_adjuster(
    meta_cfg: Mapping[str, Any] | None,
    llm_cfg: Mapping[str, Any] | None,
    *,
    direction: optuna.study.StudyDirection,
    metric_name: str,
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
) -> list[str]:
    """Apply an adjustment and return human-readable change logs."""

    messages: list[str] = []

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
