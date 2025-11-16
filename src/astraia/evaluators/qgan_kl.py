"""Deterministic qGAN KL evaluator suitable for early prototyping."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .base import BaseEvaluator, EvaluatorResult, GracefulNaNPolicy


@dataclass(slots=True)
class QGANKLEvaluator(BaseEvaluator):
    """Light-weight analytic evaluator used before integrating real simulators."""

    backend: str = "pennylane"
    shots: int = 256
    timeout_seconds: float | None = None

    _TARGET_MEAN: float = 0.0
    _TARGET_VAR: float = 1.0

    def _evaluate_impl(
        self,
        params: Mapping[str, Any],
        seed: int | None = None,
    ) -> EvaluatorResult:
        start = time.perf_counter()

        depth = max(1, int(params.get("depth", 1)))
        theta = float(params.get("theta", 0.0))
        shots = max(1, int(params.get("shots", self.shots)))

        generator_var = 0.4 + 0.25 * depth
        generator_var = max(generator_var, 1e-3)

        variance_ratio = self._TARGET_VAR / generator_var
        mean_diff = theta - self._TARGET_MEAN

        raw_kl = 0.5 * (
            variance_ratio
            + (mean_diff**2) / generator_var
            - 1.0
            + math.log(max(generator_var, 1e-12) / self._TARGET_VAR)
        )
        if math.isnan(raw_kl) or math.isinf(raw_kl):
            elapsed = time.perf_counter() - start
            return self._failure_result(
                depth=depth,
                shots=shots,
                param_count=len(params),
                elapsed=elapsed,
                reason="nan_detected" if math.isnan(raw_kl) else "kl_overflow",
            )

        rng = random.Random()
        derived_seed = self._seed_from_params(params, seed)
        rng.seed(derived_seed)
        noise_scale = 0.015 * (1 + 1 / max(shots, 1))
        noisy_kl = raw_kl + rng.gauss(0.0, noise_scale)
        kl = max(0.0, noisy_kl)

        if math.isnan(noisy_kl) or math.isnan(kl) or math.isinf(noisy_kl) or math.isinf(kl):
            elapsed = time.perf_counter() - start
            return self._failure_result(
                depth=depth,
                shots=shots,
                param_count=len(params),
                elapsed=elapsed,
                reason="nan_detected",
            )

        elapsed = time.perf_counter() - start
        timed_out = False
        status = "ok"
        reason: str | None = None

        if math.isnan(kl):
            return self._failure_result(
                depth=depth,
                shots=shots,
                param_count=len(params),
                elapsed=elapsed,
                reason="nan_detected",
            )

        if self.timeout_seconds is not None and elapsed > self.timeout_seconds:
            status = "timeout"
            timed_out = True
            reason = "timeout_exceeded"
            kl = float("inf")

        result: EvaluatorResult = {
            "kl": float(kl),
            "depth": float(depth),
            "shots": float(shots),
            "params": float(len(params)),
            "status": status,
            "timed_out": timed_out,
            "terminated_early": False,
            "elapsed_seconds": elapsed,
        }
        if reason is not None:
            result["reason"] = reason
        return result

    @staticmethod
    def _seed_from_params(params: Mapping[str, Any], base_seed: int | None) -> int:
        items = tuple(sorted(params.items()))
        combined = hash(items)
        if base_seed is not None:
            combined ^= int(base_seed)
        return combined & 0xFFFFFFFF

    def _failure_result(
        self,
        *,
        depth: int,
        shots: int,
        param_count: int,
        elapsed: float,
        reason: str,
    ) -> EvaluatorResult:
        return {
            "kl": float("inf"),
            "depth": float(depth),
            "shots": float(shots),
            "params": float(param_count),
            "status": "error",
            "reason": reason,
            "timed_out": False,
            "terminated_early": False,
            "elapsed_seconds": elapsed,
        }

    @property
    def graceful_nan_policy(self) -> GracefulNaNPolicy:
        return GracefulNaNPolicy.MARK_FAILURE


def create_evaluator(config: Mapping[str, Any]) -> QGANKLEvaluator:
    """Factory used by the configuration loader to instantiate the evaluator."""

    backend = str(config.get("backend", "pennylane"))
    shots = int(config.get("shots", 256))
    timeout = config.get("timeout_seconds")
    timeout_seconds = float(timeout) if timeout is not None else None
    return QGANKLEvaluator(backend=backend, shots=shots, timeout_seconds=timeout_seconds)
