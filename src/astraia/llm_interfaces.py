"""Shared dataclasses for LLM-facing optimization context payloads."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Mapping


@dataclass
class LLMObjective:
    """Describe an optimization objective exposed to the LLM."""

    name: str
    direction: str | None = None
    weight_hint: float | None = None
    description: str | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": self.name}
        if self.direction is not None:
            payload["direction"] = self.direction
        if self.weight_hint is not None:
            payload["weight_hint"] = self.weight_hint
        if self.description:
            payload["description"] = self.description
        return payload


@dataclass
class LLMRepresentativePoint:
    """Representative solution (e.g. Pareto front sample) shared with the LLM."""

    label: str
    trial: int | None
    values: Mapping[str, Any]
    params: Mapping[str, Any]
    metrics: Mapping[str, Any] | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "label": self.label,
            "trial": self.trial,
            "values": dict(self.values),
            "params": dict(self.params),
        }
        if self.metrics is not None:
            payload["metrics"] = dict(self.metrics)
        return payload


@dataclass
class LLMHistoryMetric:
    """Simple statistics describing the recent behaviour of an objective."""

    name: str
    direction: str | None = None
    window: int | None = None
    latest: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    count: int | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": self.name}
        if self.direction is not None:
            payload["direction"] = self.direction
        if self.window is not None:
            payload["window"] = self.window
        if self.latest is not None:
            payload["latest"] = self.latest
        if self.minimum is not None:
            payload["min"] = self.minimum
        if self.maximum is not None:
            payload["max"] = self.maximum
        if self.mean is not None:
            payload["mean"] = self.mean
        if self.count is not None:
            payload["count"] = self.count
        return payload


@dataclass
class LLMRunContext:
    """Common payload shared across all LLM helpers."""

    objectives: List[LLMObjective]
    current_best: List[LLMRepresentativePoint] = field(default_factory=list)
    history_summary: List[LLMHistoryMetric] = field(default_factory=list)
    trials_completed: int | None = None
    notes: str | None = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "objectives": [entry.to_payload() for entry in self.objectives],
            "current_best": [entry.to_payload() for entry in self.current_best],
            "history_summary": [entry.to_payload() for entry in self.history_summary],
        }
        if self.trials_completed is not None:
            payload["trials_completed"] = self.trials_completed
        if self.notes:
            payload["notes"] = self.notes
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_payload(), ensure_ascii=False, indent=indent)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "LLMHistoryMetric",
    "LLMObjective",
    "LLMRepresentativePoint",
    "LLMRunContext",
]

