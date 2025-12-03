"""Shared interfaces and utilities for optional LLM providers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, List, Mapping, Sequence
import csv
import json

__all__ = [
    "PromptMessage",
    "Prompt",
    "LLMUsage",
    "LLMResult",
    "ToolDefinition",
    "ProviderUnavailableError",
    "LLMUsageLogger",
    "LLMExchangeLogger",
    "OpenAIProvider",
    "GeminiProvider",
]


@dataclass(frozen=True)
class PromptMessage:
    """Single chat-style message for prompting an LLM."""

    role: str
    content: str

    def __post_init__(self) -> None:  # pragma: no cover - trivial guard
        if not self.role or not self.role.strip():
            raise ValueError("PromptMessage.role must be a non-empty string")
        if not self.content:
            raise ValueError("PromptMessage.content must be provided")


@dataclass(frozen=True)
class Prompt:
    """Ordered collection of prompt messages."""

    messages: Sequence[PromptMessage]

    def __post_init__(self) -> None:  # pragma: no cover - trivial guard
        if not isinstance(self.messages, Sequence) or not self.messages:
            raise ValueError("Prompt must contain at least one message")

    @classmethod
    def from_text(cls, content: str, *, role: str = "user") -> "Prompt":
        """Create a prompt from a single text block."""

        return cls(messages=[PromptMessage(role=role, content=content)])

    def to_chat_messages(self, *, system: str | None = None) -> List[dict[str, str]]:
        """Convert the prompt to an OpenAI/Gemini compatible message list."""

        converted: List[dict[str, str]] = []
        if system:
            converted.append({"role": "system", "content": system})
        for message in self.messages:
            converted.append({"role": message.role, "content": message.content})
        return converted


@dataclass(frozen=True)
class LLMUsage:
    """Structured usage metadata returned by providers."""

    provider: str
    model: str
    request_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def as_csv_row(self) -> dict[str, str | int | None]:
        return {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMResult:
    """Normalised result object returned by adapters."""

    content: str
    usage: LLMUsage | None = None
    raw_response: object | None = None
    tool_name: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """Definition of a structured tool/function call for an LLM."""

    name: str
    description: str
    parameters: Mapping[str, Any]


class ProviderUnavailableError(RuntimeError):
    """Raised when an optional provider dependency is missing."""


class LLMUsageLogger:
    """Append-only CSV logger for tracking LLM usage across runs."""

    _fieldnames = [
        "timestamp",
        "provider",
        "model",
        "request_id",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]

    def __init__(self, path: Path | str):
        self._path = Path(path)

    def log(self, usage: LLMUsage | None) -> None:
        """Persist a usage record if usage information is available."""

        if usage is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self._path.exists()

        with self._path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(usage.as_csv_row())


def ensure_usage_log(path: Path | str) -> None:
    """Create the parent directory for a usage log path."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)


class LLMExchangeLogger:
    """Append-only JSONL logger capturing prompts and responses."""

    def __init__(
        self,
        path: Path | str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._provider = provider
        self._model = model

    def log(
        self,
        *,
        prompt: Prompt,
        system: str | None,
        tool: ToolDefinition | None,
        params: Mapping[str, Any] | None,
        result: LLMResult | None,
        error: str | None = None,
        stage: str | None = None,
        trace_id: str | None = None,
        latency_ms: float | None = None,
        parse: Mapping[str, Any] | None = None,
        validation: Mapping[str, Any] | None = None,
        decision: Mapping[str, Any] | None = None,
        extras: Mapping[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Persist the given prompt/response pair as a JSON line."""

        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": provider or self._provider,
            "model": model or self._model,
            "stage": stage,
            "trace_id": trace_id,
            "request": {
                "system": system,
                "messages": [self._serialise_message(msg) for msg in prompt.messages],
                "tool": self._serialise_tool(tool),
                "params": self._clean_params(params),
            },
        }
        if latency_ms is not None:
            record["latency_ms"] = float(latency_ms)
        if parse is not None:
            record["parse"] = self._safe_value(parse)
        if validation is not None:
            record["validation"] = self._safe_value(validation)
        if decision is not None:
            record["decision"] = self._safe_value(decision)
        if extras is not None:
            record["extras"] = self._safe_value(extras)
        if result is not None:
            record["response"] = {
                "content": result.content,
                "tool_name": result.tool_name,
                "usage": result.usage.as_csv_row() if result.usage else None,
                "raw_response": self._safe_value(result.raw_response),
            }
        if error is not None:
            record["error"] = error

        self._write_record(record)

    def log_event(
        self,
        *,
        kind: str,
        stage: str | None = None,
        data: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Persist a non-call audit event to the JSONL log."""

        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": self._provider,
            "model": self._model,
            "stage": stage,
            "trace_id": trace_id,
            "kind": kind,
            "event": self._safe_value(data) if data is not None else None,
        }
        self._write_record(record)

    def _write_record(self, record: Mapping[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    def _serialise_message(self, message: PromptMessage) -> dict[str, str]:
        return {"role": message.role, "content": message.content}

    def _serialise_tool(self, tool: ToolDefinition | None) -> Mapping[str, Any] | None:
        if tool is None:
            return None
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": self._safe_value(tool.parameters),
        }

    def _clean_params(self, params: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if params is None:
            return None
        cleaned: dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            cleaned[key] = self._safe_value(value)
        return cleaned

    def _safe_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(k): self._safe_value(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._safe_value(item) for item in value]
        try:
            json.dumps(value)  # type: ignore[arg-type]
            return value
        except Exception:  # pragma: no cover - defensive fallback
            return repr(value)


from .gemini import GeminiProvider  # noqa: E402,F401
from .openai import OpenAIProvider  # noqa: E402,F401
