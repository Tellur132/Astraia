"""LLM-assisted proposal generation with caching and fallbacks."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence
import warnings

from .llm_providers import (
    LLMResult,
    LLMUsageLogger,
    Prompt,
    PromptMessage,
    ProviderUnavailableError,
    GeminiProvider,
    OpenAIProvider,
)


def create_llm_provider(
    llm_cfg: Mapping[str, Any] | None,
) -> tuple[Any | None, LLMUsageLogger | None]:
    """Instantiate an LLM provider and optional usage logger."""

    if llm_cfg is None:
        return None, None

    usage_logger: LLMUsageLogger | None = None
    usage_log_path = llm_cfg.get("usage_log")
    if usage_log_path:
        usage_logger = LLMUsageLogger(usage_log_path)

    provider_name = str(llm_cfg.get("provider", "")).lower()
    model_name = llm_cfg.get("model")
    if not provider_name or not model_name:
        return None, usage_logger

    provider: Any | None = None
    try:
        if provider_name == "openai":
            provider = OpenAIProvider(model=model_name)
        elif provider_name == "gemini":
            provider = GeminiProvider(model=model_name)
        else:
            warnings.warn(
                f"Unknown LLM provider '{provider_name}', falling back to no provider.",
                RuntimeWarning,
                stacklevel=2,
            )
    except ProviderUnavailableError as exc:
        warnings.warn(
            f"LLM provider '{provider_name}' unavailable ({exc}); using fallback heuristics.",
            RuntimeWarning,
            stacklevel=2,
        )
        provider = None

    return provider, usage_logger


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
        self._cache = cache
        self._rng = random.Random(seed)

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
            return [self._random_proposal() for _ in range(count)]

        cache_key = self._cache_key(count)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(count)
        temperature = self._base_temperature
        attempts = 0

        while attempts <= self._max_retries:
            result = self._provider.generate(
                prompt,
                temperature=temperature,
                json_mode=True,
                system=self._SYSTEM_PROMPT,
            )
            self._log_usage(result)
            proposals = self._parse_and_validate(result, expected=count)
            if proposals is not None:
                self._cache.set(cache_key, proposals)
                return proposals
            attempts += 1
            if attempts > self._max_retries:
                break
            temperature = max(self._min_temperature, temperature * 0.5)

        proposals = [self._random_proposal() for _ in range(count)]
        return proposals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_usage(self, result: LLMResult) -> None:
        if self._usage_logger is not None:
            self._usage_logger.log(result.usage)

    def _cache_key(self, count: int) -> str:
        signature = json.dumps(self._search_space, sort_keys=True, ensure_ascii=False)
        return f"{count}::{self._problem_summary}::{self._objective}::{signature}"

    def _build_prompt(self, count: int) -> Prompt:
        lines = [
            "問題概要:",
            self._problem_summary,
            "",
            "目的:",
            self._objective,
            "",
            "探索空間の制約:",
        ]
        for name, spec in self._search_space.items():
            lines.append(self._describe_parameter(name, spec))

        example = {
            "proposals": [
                {name: self._example_value(spec) for name, spec in self._search_space.items()}
                for _ in range(count)
            ]
        }

        instructions = (
            f"候補は必ず {count} 件。" "JSON オブジェクト {\"proposals\"} に配列で返すこと。"
            "各パラメタはキーに対応する数値/文字列のみを含め、追加情報を含めないこと。"
            "整数は整数値で、浮動小数点は数値で、カテゴリは指定された選択肢のみを使用すること。"
        )

        lines.extend([
            "",
            instructions,
            "",
            "出力例 (構造のみ、値はサンプル):",
            json.dumps(example, ensure_ascii=False, indent=2),
        ])

        content = "\n".join(lines)
        return Prompt(messages=[PromptMessage(role="user", content=content)])

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


_PROMPT_CACHE = PromptCache()


def create_proposal_generator(
    guidance_cfg: Mapping[str, Any] | None,
    llm_cfg: Mapping[str, Any] | None,
    search_space: Mapping[str, Mapping[str, Any]],
    *,
    seed: int | None,
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

    provider, usage_logger = create_llm_provider(llm_cfg)

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
        cache=_PROMPT_CACHE,
        seed=seed,
    )


__all__ = [
    "LLMProposalGenerator",
    "PromptCache",
    "create_llm_provider",
    "create_proposal_generator",
]
