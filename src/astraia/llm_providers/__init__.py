"""Shared interfaces and utilities for optional LLM providers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Sequence
import csv

__all__ = [
    "PromptMessage",
    "Prompt",
    "LLMUsage",
    "LLMResult",
    "ProviderUnavailableError",
    "LLMUsageLogger",
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


from .gemini import GeminiProvider  # noqa: E402,F401
from .openai import OpenAIProvider  # noqa: E402,F401
