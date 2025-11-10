"""Base interfaces and utilities for evaluator plugins."""
from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Dict, Mapping, MutableMapping, Sequence


MetricValue = float | int | str | bool | None
"""Supported value types in an evaluator result payload."""


EvaluatorResult = Dict[str, MetricValue]
"""Canonical mapping type returned by evaluators."""


class BaseEvaluator(ABC):
    """Common interface for all evaluator implementations.

    Evaluators receive a dictionary of trial parameters and return a dictionary of
    computed metrics. The primary metric is defined by the optimization
    configuration, but evaluators are free to emit additional diagnostics.

    Concrete subclasses must implement :meth:`_evaluate_impl` and return a mapping
    that includes the required keys ``kl``, ``depth``, ``shots``, and ``params``.
    The base class converts the mapping into a normalized :class:`EvaluatorResult`
    and fills in default values for optional control fields such as ``status`` and
    ``timed_out``.
    """

    #: Keys that must always be included in an evaluator payload.
    REQUIRED_METRICS: Sequence[str] = ("kl", "depth", "shots", "params")

    #: Accepted status labels for standardized evaluator results.
    VALID_STATUSES: Sequence[str] = ("ok", "error", "timeout")

    def evaluate(
        self,
        params: Mapping[str, Any],
        seed: int | None = None,
    ) -> EvaluatorResult:
        """Compute metrics for the provided parameter set.

        Subclasses should implement :meth:`_evaluate_impl` and return a
        :class:`dict`-like object. The base implementation will validate and
        normalize the payload before returning it to callers.
        """

        raw_payload = self._evaluate_impl(params, seed)
        return self._finalize_result(raw_payload)

    @abstractmethod
    def _evaluate_impl(
        self,
        params: Mapping[str, Any],
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        """Return the raw evaluator payload prior to normalization."""

    def __call__(self, params: Mapping[str, Any], seed: int | None = None) -> EvaluatorResult:
        return self.evaluate(params, seed)

    def _finalize_result(self, payload: Mapping[str, Any]) -> EvaluatorResult:
        """Validate and normalize the raw evaluator payload.

        Parameters
        ----------
        payload:
            Raw mapping returned by :meth:`_evaluate_impl`.

        Returns
        -------
        EvaluatorResult
            Normalized mapping with the required metrics and default control
            fields.
        """

        normalized: MutableMapping[str, MetricValue]
        normalized = dict(payload)

        missing = [key for key in self.REQUIRED_METRICS if key not in normalized]
        if missing:
            raise ValueError(
                "Evaluator payload is missing required metrics: " + ", ".join(missing)
            )

        for metric in self.REQUIRED_METRICS:
            normalized[metric] = float(normalized[metric])  # type: ignore[arg-type]

        kl_value = float(normalized["kl"])
        if math.isnan(kl_value):
            raise ValueError("Evaluator payload produced NaN for 'kl'.")

        status = normalized.get("status")
        if status is None:
            normalized["status"] = "ok"
        elif isinstance(status, str):
            if status not in self.VALID_STATUSES:
                raise ValueError(f"Unsupported evaluator status: {status!r}")
        else:
            raise TypeError("Evaluator status must be a string when provided.")

        for flag in ("timed_out", "terminated_early"):
            if flag not in normalized:
                normalized[flag] = False
            else:
                normalized[flag] = bool(normalized[flag])

        if "elapsed_seconds" in normalized:
            normalized["elapsed_seconds"] = float(normalized["elapsed_seconds"])

        if "reason" in normalized and normalized["reason"] is not None:
            normalized["reason"] = str(normalized["reason"])

        return dict(normalized)

