"""OpenAI Responses API adapter."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from . import LLMResult, LLMUsage, Prompt, ProviderUnavailableError, ToolDefinition

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - executed when dependency missing
    OpenAI = None  # type: ignore[assignment]


class OpenAIProvider:
    """Adapter around the OpenAI Responses API."""

    def __init__(self, model: str, *, api_key: str | None = None, organization: str | None = None):
        if OpenAI is None:  # pragma: no cover - environment dependent
            raise ProviderUnavailableError("openai package is not installed")

        client_args: Dict[str, Any] = {}
        if api_key:
            client_args["api_key"] = api_key
        if organization:
            client_args["organization"] = organization

        self._client = OpenAI(**client_args)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

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
        if OpenAI is None:  # pragma: no cover - environment dependent
            raise ProviderUnavailableError("openai package is not installed")

        messages = prompt.to_chat_messages(system=system)
        response_kwargs: Dict[str, Any] = {
            "model": self._model,
            "input": messages,
        }
        if temperature is not None:
            response_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            response_kwargs["top_p"] = float(top_p)
        if json_mode:
            response_kwargs["text"] = {
                "format": {
                    "type": "json_object",
                }
            }
        if tool is not None:
            response_kwargs["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            ]
            response_kwargs["tool_choice"] = {
                "type": "tool",
                "name": tool.name,
            }
        if stop:
            response_kwargs["stop"] = list(stop)

        response = self._client.responses.create(**response_kwargs)

        content, tool_name = _extract_text(response, expected_tool=tool)
        usage = _extract_usage(response, provider="openai", model=self._model)

        return LLMResult(content=content, usage=usage, raw_response=response, tool_name=tool_name)

    def ping(self) -> None:
        """Issue a lightweight request to verify API connectivity."""

        if OpenAI is None:  # pragma: no cover - environment dependent
            raise ProviderUnavailableError("openai package is not installed")

        try:
            self._client.models.list()
        except Exception as exc:  # pragma: no cover - depends on network
            raise RuntimeError("OpenAI ping failed") from exc


def _extract_text(response: Any, *, expected_tool: ToolDefinition | None) -> tuple[str, str | None]:
    """Best-effort extraction of text content from an OpenAI response."""

    if expected_tool is not None:
        arguments = _extract_tool_arguments(response, tool_name=expected_tool.name)
        if arguments is not None:
            return arguments, expected_tool.name

    if hasattr(response, "output_text") and response.output_text is not None:
        return str(response.output_text), None

    output = getattr(response, "output", None)
    if isinstance(output, Sequence):
        pieces: List[str] = []
        for block in output:
            contents = getattr(block, "content", None)
            if isinstance(contents, Sequence):
                for item in contents:
                    text = getattr(item, "text", None)
                    if text:
                        pieces.append(str(text))
        if pieces:
            return "".join(pieces), None

    return "", None


def _extract_tool_arguments(response: Any, *, tool_name: str) -> str | None:
    output = getattr(response, "output", None)
    if isinstance(output, Sequence):
        for block in output:
            content_items = getattr(block, "content", None)
            if isinstance(content_items, Sequence):
                for item in content_items:
                    tool_calls = getattr(item, "tool_calls", None)
                    if isinstance(tool_calls, Sequence):
                        for call in tool_calls:
                            function = getattr(call, "function", None)
                            if function is None:
                                continue
                            name = getattr(function, "name", None)
                            if name and str(name) != tool_name:
                                continue
                            arguments = getattr(function, "arguments", None)
                            if arguments is not None:
                                return str(arguments)
    # Some SDK versions expose tool calls directly on the top level choices
    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence):
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None:
                continue
            tool_calls = getattr(message, "tool_calls", None)
            if isinstance(tool_calls, Sequence):
                for call in tool_calls:
                    function = getattr(call, "function", None)
                    if function is None:
                        continue
                    name = getattr(function, "name", None)
                    if name and str(name) != tool_name:
                        continue
                    arguments = getattr(function, "arguments", None)
                    if arguments is not None:
                        return str(arguments)
    return None


def _extract_usage(response: Any, *, provider: str, model: str) -> LLMUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    request_id = getattr(response, "id", None) or getattr(
        response, "response_id", None)

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
