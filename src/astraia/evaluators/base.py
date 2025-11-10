"""Base interfaces and utilities for evaluator plugins."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping


class BaseEvaluator(ABC):
    """Common interface for all evaluator implementations.

    Evaluators receive a dictionary of trial parameters and return a dictionary of
    computed metrics. The primary metric is defined by the optimization
    configuration, but evaluators are free to emit additional diagnostics.
    """

    @abstractmethod
    def evaluate(self, params: Mapping[str, Any], seed: int | None = None) -> Dict[str, float]:
        """Compute metrics for the provided parameter set."""

    def __call__(self, params: Mapping[str, Any], seed: int | None = None) -> Dict[str, float]:
        return self.evaluate(params, seed)

