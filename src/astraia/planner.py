"""Planner agent implementations coordinating strategy directives."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence
from uuid import uuid4

from .llm_guidance import create_llm_provider
from .llm_interfaces import LLMRunContext
from .llm_providers import (
    LLMExchangeLogger,
    LLMResult,
    LLMUsageLogger,
    Prompt,
    PromptMessage,
    ToolDefinition,
)


@dataclass
class PlannerStrategy:
    """Structured description of guidance for downstream candidate generation."""

    objectives: list[str]
    emphasis: str | None
    parameter_focus: Dict[str, Any]
    batch_size_hint: int | None = None
    notes: str | None = None
    raw: Mapping[str, Any] | None = None
    source: str = "rule"

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "objectives": list(self.objectives),
            "parameter_focus": dict(self.parameter_focus),
            "source": self.source,
        }
        if self.emphasis:
            payload["emphasis"] = self.emphasis
        if self.batch_size_hint is not None:
            payload["batch_size_hint"] = int(self.batch_size_hint)
        if self.notes:
            payload["notes"] = self.notes
        if self.raw is not None:
            payload["raw"] = dict(self.raw)
        return payload


class BasePlanner:
    """Shared interface for planner implementations."""

    def generate_strategy(self, context: LLMRunContext) -> PlannerStrategy:
        raise NotImplementedError


class RuleBasedPlanner(BasePlanner):
    """Fallback heuristic planner used when LLMs are unavailable."""

    def __init__(self, search_space: Mapping[str, Mapping[str, Any]]):
        self._search_space = {name: dict(spec) for name, spec in search_space.items()}

    def generate_strategy(self, context: LLMRunContext) -> PlannerStrategy:
        objectives = [entry.name for entry in context.objectives] or ["primary_objective"]
        emphasis = objectives[0]
        parameter_focus: Dict[str, Any] = {}
        for name, spec in self._search_space.items():
            focus: Dict[str, Any] = {"mode": "explore"}
            param_type = str(spec.get("type", ""))
            if param_type in {"float", "int"}:
                focus["range_hint"] = [spec.get("low"), spec.get("high")]
            elif param_type == "categorical":
                focus["choices"] = list(spec.get("choices", []))
            parameter_focus[name] = focus
        notes = f"Emphasize {emphasis} while covering the full search window."
        raw = {
            "strategy": "default",
            "trials_completed": context.trials_completed,
        }
        return PlannerStrategy(
            objectives=objectives,
            emphasis=emphasis,
            parameter_focus=parameter_focus,
            notes=notes,
            raw=raw,
            source="rule",
        )


class LLMPlanner(BasePlanner):
    """LLM-driven planner that emits structured directives per batch."""

    _SYSTEM_PROMPT = (
        "You are the planner for an optimization loop. "
        "Summarize the next-batch strategy as valid JSON matching the schema."
    )

    def __init__(
        self,
        *,
        search_space: Mapping[str, Mapping[str, Any]],
        prompt_template: str,
        provider: Any,
        usage_logger: LLMUsageLogger | None,
        trace_logger: LLMExchangeLogger | None,
        fallback: RuleBasedPlanner,
        extra_directives: str | None = None,
        knowledge_hints: Sequence[str] | None = None,
        fewshot_examples: Sequence[str] | None = None,
    ) -> None:
        self._search_space = {name: dict(spec) for name, spec in search_space.items()}
        self._prompt_template = prompt_template
        self._provider = provider
        self._usage_logger = usage_logger
        self._trace_logger = trace_logger
        self._fallback = fallback
        self._extra_directives = extra_directives
        self._tool_schema: ToolDefinition | None = None
        self._knowledge_hints = list(knowledge_hints) if knowledge_hints else []
        self._fewshot_examples = list(fewshot_examples) if fewshot_examples else []

    def generate_strategy(self, context: LLMRunContext) -> PlannerStrategy:
        if self._provider is None:
            return self._fallback.generate_strategy(context)

        prompt = self._build_prompt(context)
        tool = self._strategy_tool()
        params = {"temperature": 0.2, "json_mode": False}
        result: LLMResult | None = None
        error: str | None = None
        trace_id = f"planner-{uuid4().hex[:8]}"
        start = time.perf_counter()
        latency_ms: float | None = None
        try:
            result = self._provider.generate(
                prompt,
                temperature=0.2,
                system=self._SYSTEM_PROMPT,
                tool=tool,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
        except RuntimeError as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            error = f"{exc.__class__.__name__}: {exc}"
            self._log_exchange(
                prompt=prompt,
                system=self._SYSTEM_PROMPT,
                tool=tool,
                params=params,
                result=None,
                error=error,
                stage="planner",
                trace_id=trace_id,
                latency_ms=latency_ms,
            )
            return self._fallback.generate_strategy(context)
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
                stage="planner",
                trace_id=trace_id,
                latency_ms=latency_ms,
            )
            raise

        self._log_usage(result)
        strategy = self._parse_result(result)
        parse_info = {"status": "ok" if strategy is not None else "error"}
        self._log_exchange(
            prompt=prompt,
            system=self._SYSTEM_PROMPT,
            tool=tool,
            params=params,
            result=result,
            stage="planner",
            trace_id=trace_id,
            latency_ms=latency_ms,
            parse=parse_info,
        )
        if strategy is None:
            return self._fallback.generate_strategy(context)
        return strategy

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
        )

    def _build_prompt(self, context: LLMRunContext) -> Prompt:
        lines = [self._prompt_template.strip(), ""]
        lines.append("[Shared optimization context]")
        lines.append(context.to_json(indent=2))
        lines.append("")
        if self._extra_directives:
            lines.append("[Additional directives]")
            lines.append(self._extra_directives.strip())
            lines.append("")
        if self._knowledge_hints:
            lines.append("[Reusable strategies from past runs]")
            lines.extend(f"- {hint}" for hint in self._knowledge_hints)
            lines.append("")
        if self._fewshot_examples:
            lines.append("[Few-shot examples that worked]")
            for idx, example in enumerate(self._fewshot_examples, start=1):
                lines.append(f"Example {idx}:")
                lines.append(example)
                lines.append("")
        lines.append("Return JSON with fields objectives/emphasis/parameter_focus/batch_size_hint/notes.")
        content = "\n".join(lines)
        return Prompt(messages=[PromptMessage(role="user", content=content)])

    def _strategy_tool(self) -> ToolDefinition:
        if self._tool_schema is None:
            parameter_properties: Dict[str, Any] = {}
            for name, spec in self._search_space.items():
                descriptor: Dict[str, Any] = {"type": "object", "additionalProperties": True}
                descriptor["properties"] = {
                    "mode": {"type": "string"},
                    "range_hint": {"type": "array", "items": {"type": "number"}},
                    "choices": {"type": "array", "items": {}},
                }
                parameter_properties[name] = descriptor
            schema = {
                "type": "object",
                "properties": {
                    "objectives": {"type": "array", "items": {"type": "string"}},
                    "emphasis": {"type": ["string", "null"]},
                    "parameter_focus": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": parameter_properties,
                    },
                    "batch_size_hint": {"type": "integer"},
                    "notes": {"type": "string"},
                },
                "required": ["objectives", "parameter_focus"],
            }
            self._tool_schema = ToolDefinition(
                name="plan_next_batch",
                description="Return JSON strategy directives.",
                parameters=schema,
            )
        return self._tool_schema

    def _parse_result(self, result: LLMResult) -> PlannerStrategy | None:
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, MutableMapping):
            return None

        objectives = [str(entry) for entry in payload.get("objectives", []) if entry]
        if not objectives:
            return None
        emphasis = payload.get("emphasis")
        if emphasis is not None:
            emphasis = str(emphasis)
        raw_focus = payload.get("parameter_focus")
        if not isinstance(raw_focus, Mapping):
            return None
        parameter_focus: Dict[str, Any] = {
            str(name): value for name, value in raw_focus.items()
        }
        batch_size_hint = payload.get("batch_size_hint")
        if batch_size_hint is not None:
            try:
                batch_size_hint = int(batch_size_hint)
            except (TypeError, ValueError):
                batch_size_hint = None
        notes = payload.get("notes")
        if notes is not None:
            notes = str(notes)
        return PlannerStrategy(
            objectives=objectives,
            emphasis=emphasis,
            parameter_focus=parameter_focus,
            batch_size_hint=batch_size_hint,
            notes=notes,
            raw=payload,
            source="llm",
        )


def create_planner_agent(
    planner_cfg: Mapping[str, Any] | None,
    llm_cfg: Mapping[str, Any] | None,
    *,
    search_space: Mapping[str, Mapping[str, Any]],
    knowledge_hints: Sequence[str] | None = None,
    fewshot_examples: Sequence[str] | None = None,
) -> BasePlanner | None:
    """Instantiate the configured planner agent if enabled."""

    if planner_cfg is None or not planner_cfg.get("enabled"):
        return None

    backend = str(planner_cfg.get("backend", "rule")).lower()
    fallback = RuleBasedPlanner(search_space)

    if backend != "llm":
        return fallback

    role_name = str(planner_cfg.get("role") or "planner")
    role_entry = _lookup_role(planner_cfg.get("roles"), role_name)
    prompt_template = None
    if role_entry is not None:
        prompt_template = role_entry.get("prompt_template")
    if not prompt_template:
        prompt_template = planner_cfg.get("prompt_template")
    if not prompt_template:
        return fallback

    extra_directives = _load_optional_text(planner_cfg.get("config_path"))
    role_llm_cfg = role_entry.get("llm") if role_entry else None
    effective_llm_cfg = role_llm_cfg or llm_cfg
    provider, usage_logger, trace_logger = create_llm_provider(effective_llm_cfg)
    if provider is None:
        return fallback

    template_text = _load_template_text(prompt_template)
    return LLMPlanner(
        search_space=search_space,
        prompt_template=template_text,
        provider=provider,
        usage_logger=usage_logger,
        trace_logger=trace_logger,
        fallback=fallback,
        extra_directives=extra_directives,
        knowledge_hints=knowledge_hints,
        fewshot_examples=fewshot_examples,
    )


def _lookup_role(
    roles_cfg: Any, role_name: str
) -> Mapping[str, Any] | None:  # pragma: no cover - tiny helper
    if isinstance(roles_cfg, Mapping):
        entry = roles_cfg.get(role_name)
        if isinstance(entry, Mapping):
            return entry
    return None


def _load_template_text(value: str) -> str:
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


def _load_optional_text(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


__all__ = ["PlannerStrategy", "create_planner_agent"]
