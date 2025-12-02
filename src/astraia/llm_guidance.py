"""LLM-assisted proposal generation with caching and fallbacks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence
import warnings

from .llm_interfaces import LLMObjective, LLMRunContext
from .llm_providers import (
    LLMResult,
    LLMUsage,
    LLMUsageLogger,
    LLMExchangeLogger,
    Prompt,
    PromptMessage,
    ProviderUnavailableError,
    GeminiProvider,
    OpenAIProvider,
    ToolDefinition,
)


def create_llm_provider(
    llm_cfg: Mapping[str, Any] | None,
    *,
    strict: bool = False,
) -> tuple[Any | None, LLMUsageLogger | None, LLMExchangeLogger | None]:
    """Instantiate an LLM provider and optional usage logger."""

    if llm_cfg is None:
        return None, None, None

    usage_logger: LLMUsageLogger | None = None
    usage_log_path = llm_cfg.get("usage_log")
    if usage_log_path:
        usage_logger = LLMUsageLogger(usage_log_path)

    provider_name = str(llm_cfg.get("provider", "")).lower()
    model_name = llm_cfg.get("model")

    trace_logger: LLMExchangeLogger | None = None
    trace_log_path = llm_cfg.get("trace_log")
    if trace_log_path:
        trace_logger = LLMExchangeLogger(
            trace_log_path,
            provider=provider_name or None,
            model=str(model_name) if model_name else None,
        )

    if not provider_name or not model_name:
        return None, usage_logger, trace_logger

    provider: Any | None = None
    try:
        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY") or None
            organization = os.environ.get("OPENAI_ORG_ID") or None
            provider = OpenAIProvider(
                model=str(model_name),
                api_key=api_key,
                organization=organization,
            )
        elif provider_name == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") or None
            provider = GeminiProvider(model=str(model_name), api_key=api_key)
        else:
            message = f"Unknown LLM provider '{provider_name}', falling back to no provider."
            if strict:
                raise ValueError(message)
            warnings.warn(
                message,
                RuntimeWarning,
                stacklevel=2,
            )
    except ProviderUnavailableError as exc:
        if strict:
            raise
        warnings.warn(
            f"LLM provider '{provider_name}' unavailable ({exc}); using fallback heuristics.",
            RuntimeWarning,
            stacklevel=2,
        )
        provider = None

    if provider is None:
        return None, usage_logger

    guard = _build_budget_guard(llm_cfg)
    if guard is not None and guard.has_limits:
        provider = _BudgetedProvider(provider, guard)

    return provider, usage_logger, trace_logger


class LLMBudgetGuard:
    """Track aggregate usage and enforce coarse runtime limits."""

    def __init__(
        self,
        *,
        max_calls: int | None,
        max_tokens: int | None,
        budget_usd: float | None,
        prompt_cost_per_1k: float | None,
        completion_cost_per_1k: float | None,
    ) -> None:
        self._max_calls = max_calls if max_calls and max_calls > 0 else None
        self._max_tokens = max_tokens if max_tokens and max_tokens > 0 else None
        self._budget_usd = budget_usd if budget_usd and budget_usd > 0 else None
        self._prompt_cost = (
            prompt_cost_per_1k if prompt_cost_per_1k and prompt_cost_per_1k > 0 else None
        )
        self._completion_cost = (
            completion_cost_per_1k
            if completion_cost_per_1k and completion_cost_per_1k > 0
            else None
        )

        self._calls_made = 0
        self._tokens_used = 0
        self._budget_spent = 0.0
        self._exhausted = False
        self._exhaustion_reason: str | None = None

    @property
    def has_limits(self) -> bool:
        return any(
            value is not None
            for value in (self._max_calls, self._max_tokens, self._budget_usd)
        )

    def start_call(self) -> None:
        if not self.has_limits:
            return
        if self._exhausted:
            raise RuntimeError(self._exhaustion_reason or "LLM budget exhausted")
        if self._max_calls is not None and self._calls_made >= self._max_calls:
            self._exhausted = True
            self._exhaustion_reason = "LLM call limit reached"
            raise RuntimeError(self._exhaustion_reason)
        self._calls_made += 1

    def cancel_call(self) -> None:
        if not self.has_limits:
            return
        if self._calls_made > 0:
            self._calls_made -= 1

    def finalize_call(self, usage: LLMUsage | None) -> None:
        if not self.has_limits:
            return

        if usage is not None:
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            total_tokens = usage.total_tokens
            if total_tokens is None:
                total_tokens = prompt_tokens + completion_tokens
            if total_tokens:
                self._tokens_used += int(total_tokens)
            if self._budget_usd is not None:
                cost = 0.0
                if self._prompt_cost is not None and prompt_tokens:
                    cost += (prompt_tokens / 1000.0) * self._prompt_cost
                if self._completion_cost is not None and completion_tokens:
                    cost += (completion_tokens / 1000.0) * self._completion_cost
                self._budget_spent += cost

        self._check_exhaustion()

    def _check_exhaustion(self) -> None:
        if self._max_tokens is not None and self._tokens_used >= self._max_tokens:
            self._exhausted = True
            if self._exhaustion_reason is None:
                self._exhaustion_reason = "LLM token budget exhausted"
        if (
            self._budget_usd is not None
            and self._prompt_cost is not None
            and self._completion_cost is not None
            and self._budget_spent >= self._budget_usd
        ):
            self._exhausted = True
            if self._exhaustion_reason is None:
                self._exhaustion_reason = "LLM cost budget exhausted"


class _BudgetedProvider:
    """Proxy provider that enforces :class:`LLMBudgetGuard` limits."""

    def __init__(self, provider: Any, guard: LLMBudgetGuard) -> None:
        self._provider = provider
        self._guard = guard

    def generate(self, prompt: Prompt, *, tool: ToolDefinition | None = None, **kwargs: Any) -> LLMResult:
        self._guard.start_call()
        try:
            result = self._provider.generate(prompt, tool=tool, **kwargs)
        except Exception:
            self._guard.cancel_call()
            raise
        self._guard.finalize_call(result.usage)
        return result

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self._provider, name)


def _build_budget_guard(cfg: Mapping[str, Any]) -> LLMBudgetGuard | None:
    max_calls = cfg.get("max_calls")
    max_tokens = cfg.get("max_tokens_per_run")
    budget = cfg.get("budget_usd")
    prompt_cost = cfg.get("prompt_cost_per_1k")
    completion_cost = cfg.get("completion_cost_per_1k")

    guard = LLMBudgetGuard(
        max_calls=int(max_calls) if isinstance(max_calls, (int, float)) else None,
        max_tokens=int(max_tokens) if isinstance(max_tokens, (int, float)) else None,
        budget_usd=float(budget) if isinstance(budget, (int, float)) else None,
        prompt_cost_per_1k=float(prompt_cost)
        if isinstance(prompt_cost, (int, float))
        else None,
        completion_cost_per_1k=float(completion_cost)
        if isinstance(completion_cost, (int, float))
        else None,
    )

    return guard if guard.has_limits else None


@dataclass
class PromptCache:
    """Lightweight in-memory cache for parsed proposal lists."""

    _storage: MutableMapping[str, List[Dict[str, Any]]]

    def __init__(self) -> None:
        self._storage = {}

    def get(self, key: str) -> List[Dict[str, Any]] | None:
        cached = self._storage.get(key)
        if cached is None:
            return None
        return [dict(item) for item in cached]

    def set(self, key: str, proposals: Sequence[Mapping[str, Any]]) -> None:
        self._storage[key] = [dict(item) for item in proposals]


class LLMProposalGenerator:
    """Generate parameter proposals via an LLM with strict validation."""

    _SYSTEM_PROMPT = (
        "You propose candidate hyper-parameters for an optimization algorithm. "
        "Follow the constraints exactly and respond with strict JSON."
    )

    def __init__(
        self,
        *,
        search_space: Mapping[str, Mapping[str, Any]],
        problem_summary: str,
        objective: str,
        batch_size: int,
        base_temperature: float,
        min_temperature: float,
        max_retries: int,
        provider: Any | None,
        usage_logger: LLMUsageLogger | None,
        trace_logger: LLMExchangeLogger | None,
        cache: PromptCache,
        seed: int | None,
    ) -> None:
        self._search_space = {name: dict(spec) for name, spec in search_space.items()}
        self._problem_summary = problem_summary
        self._objective = objective
        self._batch_size = batch_size
        self._base_temperature = base_temperature
        self._min_temperature = min_temperature
        self._max_retries = max_retries
        self._provider = provider
        self._usage_logger = usage_logger
        self._trace_logger = trace_logger
        self._cache = cache
        self._rng = random.Random(seed)
        self._schema_cache: dict[int, Dict[str, Any]] = {}
        self._tool_cache: dict[int, ToolDefinition] = {}
        self._seen_hashes: set[str] = set()
        self._context: LLMRunContext | None = None
        self._strategy_payload: Dict[str, Any] | None = None
        self._meta_directives: list[str] = []

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def propose_batch(self, remaining: int | None = None) -> List[Dict[str, Any]]:
        if remaining is None:
            count = self._batch_size
        else:
            remaining = int(remaining)
            if remaining <= 0:
                return []
            count = min(self._batch_size, remaining)

        if self._provider is None:
            return self._random_unique_batch(count)

        cache_key = self._cache_key(count)
        cached = self._cache.get(cache_key)
        if cached is not None:
            unique = self._ensure_unique_batch(cached, expected=count)
            if unique is not None:
                return unique

        prompt = self._build_prompt(count)
        temperature = self._base_temperature
        attempts = 0

        while attempts <= self._max_retries:
            tool = self._proposal_tool(count)
            params: Dict[str, Any] = {"temperature": temperature, "json_mode": False}
            result: LLMResult | None = None
            error: str | None = None
            try:
                result = self._provider.generate(
                    prompt,
                    temperature=temperature,
                    json_mode=False,
                    system=self._SYSTEM_PROMPT,
                    tool=tool,
                )
            except RuntimeError as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                self._log_exchange(
                    prompt=prompt,
                    system=self._SYSTEM_PROMPT,
                    tool=tool,
                    params=params,
                    result=None,
                    error=error,
                )
                break
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                self._log_exchange(
                    prompt=prompt,
                    system=self._SYSTEM_PROMPT,
                    tool=tool,
                    params=params,
                    result=None,
                    error=error,
                )
                raise
            self._log_usage(result)
            self._log_exchange(
                prompt=prompt,
                system=self._SYSTEM_PROMPT,
                tool=tool,
                params=params,
                result=result,
            )
            proposals = self._parse_and_validate(result, expected=count)
            if proposals is not None:
                unique = self._ensure_unique_batch(proposals, expected=count)
                if unique is not None:
                    self._cache.set(cache_key, unique)
                    return unique
            attempts += 1
            if attempts > self._max_retries:
                break
            temperature = max(self._min_temperature, temperature * 0.5)

        return self._random_unique_batch(count)

    def update_context(self, context: LLMRunContext | None) -> None:
        """Attach the latest shared optimization context for prompt construction."""

        self._context = context

    def update_strategy(self, strategy: Mapping[str, Any] | None) -> None:
        """Attach the latest planner directives for prompt construction."""

        self._strategy_payload = dict(strategy) if strategy is not None else None

    def apply_meta_directives(self, directives: Iterable[str]) -> None:
        """Persist meta-search directives to be surfaced in prompts."""

        cleaned: list[str] = []
        for directive in directives:
            text = str(directive).strip()
            if text:
                cleaned.append(text)
        self._meta_directives = cleaned

    def apply_search_space_overrides(
        self, overrides: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        """Update the internal search space with temporary constraints."""

        updates: list[str] = []
        for name, payload in overrides.items():
            spec = self._search_space.get(name)
            if spec is None:
                continue
            if not isinstance(payload, Mapping):
                continue
            param_type = spec.get("type")
            if param_type == "float":
                updated = self._apply_numeric_override(spec, payload, is_int=False)
            elif param_type == "int":
                updated = self._apply_numeric_override(spec, payload, is_int=True)
            elif param_type == "categorical":
                updated = self._apply_choice_override(spec, payload)
            elif param_type == "llm_only":
                updated = self._apply_llm_only_override(spec, payload)
            else:
                continue
            if updated:
                updates.append(f"{name}: {updated}")

        if updates:
            self._schema_cache.clear()
            self._tool_cache.clear()
            self._seen_hashes.clear()
        return updates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
        )

    def _cache_key(self, count: int) -> str:
        signature = json.dumps(self._search_space, sort_keys=True, ensure_ascii=False)
        context_hash = self._current_context().fingerprint()
        strategy_hash = self._strategy_hash()
        meta_hash = self._meta_hash()
        return (
            f"{count}::{self._problem_summary}::{self._objective}::{signature}::{context_hash}::{strategy_hash}::{meta_hash}"
        )

    def _build_prompt(self, count: int) -> Prompt:
        context_json = self._current_context().to_json(indent=2)
        lines = [
            "最適化状態 (objectives/current_best/history_summary) の共通JSON:",
            context_json,
            "",
        ]
        if self._strategy_payload is not None:
            strategy_json = json.dumps(
                self._strategy_payload, ensure_ascii=False, indent=2
            )
            lines.extend([
                "プランナーからのバッチ指示:",
                strategy_json,
                "",
            ])
        if self._meta_directives:
            lines.extend([
                "メタ探索コントローラからの直近指示:",
                "- " + "\n- ".join(self._meta_directives),
                "",
            ])
        lines.extend(
            [
                "問題概要:",
                self._problem_summary,
                "",
                "目的:",
                self._objective,
                "",
                "探索空間の制約:",
            ]
        )
        for name, spec in self._search_space.items():
            lines.append(self._describe_parameter(name, spec))

        example = {
            "proposals": [
                {name: self._example_value(spec) for name, spec in self._search_space.items()}
                for _ in range(count)
            ]
        }

        schema = json.dumps(self._proposal_schema(count), ensure_ascii=False, indent=2)

        instructions = (
            f"候補は必ず {count} 件。" "JSON オブジェクト {\"proposals\"} に配列で返すこと。"
            "各パラメタはキーに対応する数値/文字列のみを含め、追加情報を含めないこと。"
            "整数は整数値で、浮動小数点は数値で、カテゴリは指定された選択肢のみを使用すること。"
        )

        lines.extend([
            "",
            instructions,
            "",
            "戻り値は propose_candidates 関数の引数として Schema に完全一致する JSON オブジェクト1件のみを出力すること。",
            "Schema:",
            schema,
            "",
            "出力例 (構造のみ、値はサンプル):",
            json.dumps(example, ensure_ascii=False, indent=2),
        ])

        content = "\n".join(lines)
        return Prompt(messages=[PromptMessage(role="user", content=content)])

    def _strategy_hash(self) -> str:
        if self._strategy_payload is None:
            return "no-strategy"
        canonical = json.dumps(self._strategy_payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _meta_hash(self) -> str:
        if not self._meta_directives:
            return "no-meta-directives"
        canonical = json.dumps(sorted(self._meta_directives), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _current_context(self) -> LLMRunContext:
        if self._context is not None:
            return self._context
        return LLMRunContext(
            objectives=[
                LLMObjective(
                    name=self._objective or "primary_objective",
                    direction=None,
                )
            ]
        )

    def _proposal_tool(self, count: int) -> ToolDefinition:
        tool = self._tool_cache.get(count)
        if tool is None:
            schema = self._proposal_schema(count)
            tool = ToolDefinition(
                name="propose_candidates",
                description="Return hyper-parameter proposals that satisfy the schema.",
                parameters=schema,
            )
            self._tool_cache[count] = tool
        return tool

    def _proposal_schema(self, count: int) -> Dict[str, Any]:
        schema = self._schema_cache.get(count)
        if schema is not None:
            return schema

        param_properties: Dict[str, Any] = {}
        required: List[str] = []
        for name, spec in self._search_space.items():
            param_properties[name] = self._parameter_schema(spec)
            required.append(name)

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "proposals": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "properties": param_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["proposals"],
        }
        self._schema_cache[count] = schema
        return schema

    def _parameter_schema(self, spec: Mapping[str, Any]) -> Dict[str, Any]:
        param_type = spec.get("type")
        if param_type == "float":
            low = float(spec.get("low"))
            high = float(spec.get("high"))
            schema: Dict[str, Any] = {"type": "number", "minimum": low, "maximum": high}
            step = spec.get("step")
            if step is not None:
                schema["multipleOf"] = float(step)
            return schema
        if param_type == "int":
            low = int(spec.get("low"))
            high = int(spec.get("high"))
            schema = {"type": "integer", "minimum": low, "maximum": high}
            step = spec.get("step")
            if step is not None:
                schema["multipleOf"] = max(1, int(step))
            return schema
        if param_type == "categorical":
            choices = list(spec.get("choices", []))
            return {"enum": choices}
        if param_type == "llm_only":
            return {"type": "string"}
        return {}

    def _describe_parameter(self, name: str, spec: Mapping[str, Any]) -> str:
        param_type = spec.get("type", "")
        if param_type == "float":
            low = spec.get("low")
            high = spec.get("high")
            step = spec.get("step")
            parts = [f"- {name}: float [{low}, {high}]"]
            if step is not None:
                parts.append(f" step={step}")
            if spec.get("log"):
                parts.append(" log-scale")
            return "".join(parts)
        if param_type == "int":
            low = spec.get("low")
            high = spec.get("high")
            step = spec.get("step") or 1
            return f"- {name}: int [{low}, {high}] step={step}"
        if param_type == "categorical":
            choices = ", ".join(map(str, spec.get("choices", [])))
            return f"- {name}: choice from {{{choices}}}"
        if param_type == "llm_only":
            return f"- {name}: free-form string filled by LLM"
        return f"- {name}: unknown specification"

    def _example_value(self, spec: Mapping[str, Any]) -> Any:
        param_type = spec.get("type")
        if param_type == "float":
            low = float(spec.get("low", 0.0))
            high = float(spec.get("high", low))
            return round((low + high) / 2, 6)
        if param_type == "int":
            low = int(spec.get("low", 0))
            high = int(spec.get("high", low))
            return (low + high) // 2
        if param_type == "categorical":
            choices = spec.get("choices", [])
            return choices[0] if choices else None
        if param_type == "llm_only":
            default = spec.get("default")
            if default is not None:
                return default
            return "example_value"
        return None

    def _apply_numeric_override(
        self, spec: MutableMapping[str, Any], payload: Mapping[str, Any], *, is_int: bool
    ) -> str | None:
        try:
            base_low = spec.get("low") if spec.get("low") is not None else 0
            base_high = spec.get("high") if spec.get("high") is not None else 0
            new_low = base_low if "low" not in payload else (int(payload["low"]) if is_int else float(payload["low"]))
            new_high = base_high if "high" not in payload else (int(payload["high"]) if is_int else float(payload["high"]))
        except (TypeError, ValueError):
            return None

        if new_high is None or new_low is None:
            return None
        if float(new_high) <= float(new_low):
            return None

        spec["low"] = int(new_low) if is_int else float(new_low)
        spec["high"] = int(new_high) if is_int else float(new_high)

        if "step" in payload:
            try:
                step_value = int(payload["step"]) if is_int else float(payload["step"])
            except (TypeError, ValueError):
                step_value = None
            if step_value is not None and step_value > 0:
                spec["step"] = step_value

        return f"range=[{spec['low']}, {spec['high']}]"

    def _apply_choice_override(
        self, spec: MutableMapping[str, Any], payload: Mapping[str, Any]
    ) -> str | None:
        choices = payload.get("choices")
        if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes, bytearray)):
            new_choices = [item for item in choices if item is not None]
            if new_choices:
                spec["choices"] = list(new_choices)
                return f"choices={len(new_choices)}"
        return None

    def _apply_llm_only_override(
        self, spec: MutableMapping[str, Any], payload: Mapping[str, Any]
    ) -> str | None:
        default = payload.get("default")
        if isinstance(default, str) and default.strip():
            spec["default"] = default.strip()
            return "default updated"
        return None

    def _parse_and_validate(
        self, result: LLMResult, *, expected: int
    ) -> List[Dict[str, Any]] | None:
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, Mapping):
            proposals = payload.get("proposals")
        else:
            proposals = payload

        if not isinstance(proposals, Sequence):
            return None
        if len(proposals) != expected:
            return None

        validated: List[Dict[str, Any]] = []
        try:
            for item in proposals:
                if not isinstance(item, Mapping):
                    return None
                validated.append(self._validate_single(item))
        except ValueError:
            return None

        return validated

    def _validate_single(self, proposal: Mapping[str, Any]) -> Dict[str, Any]:
        validated: Dict[str, Any] = {}
        for name, spec in self._search_space.items():
            if name not in proposal:
                raise ValueError(f"Missing parameter: {name}")
            value = proposal[name]
            p_type = spec.get("type")
            if p_type == "float":
                validated[name] = self._validate_float(value, spec)
            elif p_type == "int":
                validated[name] = self._validate_int(value, spec)
            elif p_type == "categorical":
                validated[name] = self._validate_choice(value, spec)
            elif p_type == "llm_only":
                validated[name] = self._validate_llm_only(value, spec)
            else:
                raise ValueError(f"Unsupported parameter type: {p_type}")
        return validated

    def _validate_float(self, value: Any, spec: Mapping[str, Any]) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("float conversion failed") from None

        low = float(spec.get("low"))
        high = float(spec.get("high"))
        if not (low <= number <= high):
            raise ValueError("float out of bounds")

        step = spec.get("step")
        if step is not None:
            step = float(step)
            if step <= 0:
                raise ValueError("invalid float step")
            steps = round((number - low) / step)
            snapped = low + steps * step
            if not math.isclose(number, snapped, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("float step mismatch")
            number = snapped

        return float(number)

    def _validate_int(self, value: Any, spec: Mapping[str, Any]) -> int:
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid int parameter")
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError("int conversion failed") from None

        low = int(spec.get("low"))
        high = int(spec.get("high"))
        if not (low <= number <= high):
            raise ValueError("int out of bounds")

        step = spec.get("step")
        if step is None:
            step = 1
        else:
            step = int(step)
            if step <= 0:
                raise ValueError("invalid int step")

        if (number - low) % step != 0:
            raise ValueError("int step mismatch")

        return number

    def _validate_choice(self, value: Any, spec: Mapping[str, Any]) -> Any:
        choices = list(spec.get("choices", []))
        if value not in choices:
            raise ValueError("choice not in options")
        return value

    def _validate_llm_only(self, value: Any, spec: Mapping[str, Any]) -> str:  # noqa: ARG002
        text = str(value)
        if not text:
            raise ValueError("llm_only parameter must be non-empty")
        return text

    def _random_proposal(self) -> Dict[str, Any]:
        proposal: Dict[str, Any] = {}
        for name, spec in self._search_space.items():
            p_type = spec.get("type")
            if p_type == "float":
                proposal[name] = self._random_float(spec)
            elif p_type == "int":
                proposal[name] = self._random_int(spec)
            elif p_type == "categorical":
                proposal[name] = self._random_choice(spec)
            elif p_type == "llm_only":
                proposal[name] = self._random_llm_text(spec)
            else:
                raise ValueError(f"Unsupported parameter type: {p_type}")
        return proposal

    def _random_float(self, spec: Mapping[str, Any]) -> float:
        low = float(spec.get("low"))
        high = float(spec.get("high"))
        step = spec.get("step")
        if step is None:
            return self._rng.uniform(low, high)
        step = float(step)
        if step <= 0:
            raise ValueError("invalid float step")
        steps = int(math.floor((high - low) / step))
        if steps <= 0:
            return low
        index = self._rng.randint(0, steps)
        value = low + index * step
        if value > high:
            value = high
        return float(value)

    def _random_int(self, spec: Mapping[str, Any]) -> int:
        low = int(spec.get("low"))
        high = int(spec.get("high"))
        step = spec.get("step")
        if step is None:
            step = 1
        else:
            step = int(step)
            if step <= 0:
                raise ValueError("invalid int step")
        count = ((high - low) // step)
        index = self._rng.randint(0, count)
        return low + index * step

    def _random_choice(self, spec: Mapping[str, Any]) -> Any:
        choices = list(spec.get("choices", []))
        if not choices:
            raise ValueError("categorical parameter requires choices")
        return self._rng.choice(choices)

    def _random_llm_text(self, spec: Mapping[str, Any]) -> str:
        default = spec.get("default")
        if isinstance(default, str) and default:
            return default
        return f"llm_candidate_{self._rng.randint(0, 1_000_000)}"

    def _ensure_unique_batch(
        self, proposals: Iterable[Mapping[str, Any]], *, expected: int
    ) -> List[Dict[str, Any]] | None:
        unique: List[Dict[str, Any]] = []
        for proposal in proposals:
            proposal_dict = dict(proposal)
            if self._register_proposal(proposal_dict):
                unique.append(proposal_dict)
            if len(unique) >= expected:
                return unique[:expected]

        needed = expected - len(unique)
        if needed <= 0:
            return unique[:expected]

        filler = self._random_unique_batch(needed)
        unique.extend(filler[:needed])
        if len(unique) >= expected:
            return unique[:expected]
        return None

    def _random_unique_batch(self, count: int) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []
        attempts = 0
        attempt_limit = max(50, count * 20)
        while len(proposals) < count and attempts < attempt_limit:
            candidate = self._random_proposal()
            attempts += 1
            if self._register_proposal(candidate):
                proposals.append(candidate)
        if len(proposals) < count:
            while len(proposals) < count:
                candidate = self._random_proposal()
                proposals.append(candidate)
        return proposals

    def _register_proposal(self, proposal: Mapping[str, Any]) -> bool:
        fingerprint = self._hash_proposal(proposal)
        if fingerprint in self._seen_hashes:
            return False
        self._seen_hashes.add(fingerprint)
        return True

    def _hash_proposal(self, proposal: Mapping[str, Any]) -> str:
        canonical = json.dumps(dict(proposal), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_PROMPT_CACHE = PromptCache()


def create_proposal_generator(
    guidance_cfg: Mapping[str, Any] | None,
    llm_cfg: Mapping[str, Any] | None,
    search_space: Mapping[str, Mapping[str, Any]],
    *,
    seed: int | None,
    preferred_llm_cfg: Mapping[str, Any] | None = None,
) -> LLMProposalGenerator | None:
    """Create an :class:`LLMProposalGenerator` if guidance is enabled."""

    if guidance_cfg is None or not guidance_cfg.get("enabled"):
        return None

    problem_summary = str(guidance_cfg.get("problem_summary", "")).strip()
    objective = str(guidance_cfg.get("objective", "")).strip()
    batch_size = int(guidance_cfg.get("n_proposals", 1))
    max_retries = int(guidance_cfg.get("max_retries", 2))
    base_temperature = float(guidance_cfg.get("base_temperature", 0.7))
    min_temperature = float(guidance_cfg.get("min_temperature", 0.1))

    effective_llm_cfg = preferred_llm_cfg or llm_cfg
    provider, usage_logger, trace_logger = create_llm_provider(effective_llm_cfg)

    return LLMProposalGenerator(
        search_space=search_space,
        problem_summary=problem_summary,
        objective=objective,
        batch_size=batch_size,
        base_temperature=base_temperature,
        min_temperature=min_temperature,
        max_retries=max_retries,
        provider=provider,
        usage_logger=usage_logger,
        trace_logger=trace_logger,
        cache=_PROMPT_CACHE,
        seed=seed,
    )


__all__ = [
    "LLMProposalGenerator",
    "PromptCache",
    "create_llm_provider",
    "create_proposal_generator",
]
