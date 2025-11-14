"""Google Gemini adapter."""
from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from . import LLMResult, LLMUsage, Prompt, ProviderUnavailableError, ToolDefinition

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - executed when dependency missing
    genai = None  # type: ignore[assignment]


class GeminiProvider:
    """Adapter that wraps the google-generativeai SDK."""

    def __init__(self, model: str, *, api_key: str | None = None):
        if genai is None:  # pragma: no cover - environment dependent
            raise ProviderUnavailableError("google-generativeai package is not installed")

        self._model_name = model
        if api_key:
            genai.configure(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: Prompt,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
        stop: Sequence[str] | None = None,
        tool: ToolDefinition | None = None,
    ) -> LLMResult:
        if genai is None:  # pragma: no cover - environment dependent
            raise ProviderUnavailableError("google-generativeai package is not installed")

        generation_config: Dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = float(temperature)
        if top_p is not None:
            generation_config["top_p"] = float(top_p)
        if stop:
            generation_config["stop_sequences"] = list(stop)
        if json_mode or tool is not None:
            generation_config["response_mime_type"] = "application/json"
        if tool is not None:
            generation_config["response_schema"] = tool.parameters

        messages = prompt.to_chat_messages()
        # Gemini expects the system instruction separately.
        tool_config = None
        if tool is not None:
            tool_config = [{
                "function_declarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                ]
            }]
        model = genai.GenerativeModel(
            self._model_name,
            system_instruction=system,
            tools=tool_config,
        )
        response = model.generate_content(messages, generation_config=generation_config)

        text, tool_name = _extract_text(response, expected_tool=tool)
        usage = _extract_usage(response, provider="gemini", model=self._model_name)

        return LLMResult(content=text, usage=usage, raw_response=response, tool_name=tool_name)

    def ping(self) -> None:
        """Perform a lightweight connectivity check against the Gemini API."""

        if genai is None:  # pragma: no cover - environment dependent
            raise ProviderUnavailableError("google-generativeai package is not installed")

        try:
            genai.get_model(self._model_name)
        except Exception as exc:  # pragma: no cover - depends on network
            raise RuntimeError("Gemini ping failed") from exc


def _extract_text(response: Any, *, expected_tool: ToolDefinition | None) -> tuple[str, str | None]:
    if expected_tool is not None:
        arguments = _extract_tool_arguments(response, tool_name=expected_tool.name)
        if arguments is not None:
            return arguments, expected_tool.name

    text = getattr(response, "text", "") or ""
    return str(text), None


def _extract_usage(response: Any, *, provider: str, model: str) -> LLMUsage | None:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None

    prompt_tokens = getattr(metadata, "prompt_token_count", None)
    completion_tokens = getattr(metadata, "candidates_token_count", None)
    total_tokens = getattr(metadata, "total_token_count", None)
    request_id = getattr(response, "response_id", None)

    return LLMUsage(
        provider=provider,
        model=model,
        request_id=request_id,
        prompt_tokens=_safe_int(prompt_tokens),
        completion_tokens=_safe_int(completion_tokens),
        total_tokens=_safe_int(total_tokens),
    )


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _extract_tool_arguments(response: Any, *, tool_name: str) -> str | None:
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, Sequence):
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if isinstance(parts, Sequence):
                for part in parts:
                    function_call = getattr(part, "function_call", None)
                    if function_call is None:
                        continue
                    name = getattr(function_call, "name", None)
                    if name and str(name) != tool_name:
                        continue
                    args = getattr(function_call, "args", None)
                    if args is not None:
                        return json.dumps(args)
    return None
