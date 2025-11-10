"""Deterministic qGAN KL evaluator suitable for early prototyping."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .base import BaseEvaluator


@dataclass(slots=True)
class QGANKLEvaluator(BaseEvaluator):
    """Light-weight analytic evaluator used before integrating real simulators."""

    backend: str = "pennylane"
    shots: int = 256

    _TARGET_MEAN: float = 0.0
    _TARGET_VAR: float = 1.0

    def evaluate(self, params: Mapping[str, Any], seed: int | None = None) -> Dict[str, float]:
        depth = max(1, int(params.get("depth", 1)))
        theta = float(params.get("theta", 0.0))

        generator_var = 0.4 + 0.25 * depth
        generator_var = max(generator_var, 1e-3)

        variance_ratio = self._TARGET_VAR / generator_var
        mean_diff = theta - self._TARGET_MEAN

        kl = 0.5 * (
            variance_ratio
            + (mean_diff**2) / generator_var
            - 1.0
            + math.log(max(generator_var, 1e-12) / self._TARGET_VAR)
        )

        rng = random.Random()
        derived_seed = self._seed_from_params(params, seed)
        rng.seed(derived_seed)
        noise_scale = 0.015 * (1 + 1 / max(self.shots, 1))
        kl = max(0.0, kl + rng.gauss(0.0, noise_scale))

        param_count = float(len(params))
        metrics: Dict[str, float] = {
            "kl": kl,
            "depth": float(depth),
            "shots": float(self.shots),
            "params": param_count,
        }
        return metrics

    @staticmethod
    def _seed_from_params(params: Mapping[str, Any], base_seed: int | None) -> int:
        items = tuple(sorted(params.items()))
        combined = hash(items)
        if base_seed is not None:
            combined ^= int(base_seed)
        return combined & 0xFFFFFFFF


def create_evaluator(config: Mapping[str, Any]) -> QGANKLEvaluator:
    """Factory used by the configuration loader to instantiate the evaluator."""

    backend = str(config.get("backend", "pennylane"))
    shots = int(config.get("shots", 256))
    return QGANKLEvaluator(backend=backend, shots=shots)
