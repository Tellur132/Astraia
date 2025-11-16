"""Continuous multi-objective benchmark evaluators."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

EvaluatorPayload = Dict[str, float | int | bool | None]


@dataclass(slots=True)
class ZDT3Evaluator:
    """Deterministic implementation of the classic ZDT3 benchmark."""

    variables: Sequence[str] = ("x1", "x2", "x3", "x4", "x5")
    noise_std: float = 0.0

    def __call__(self, params: Mapping[str, Any], seed: int | None = None) -> EvaluatorPayload:
        vector = [float(params[name]) for name in self.variables]
        if len(vector) < 2:
            raise ValueError("ZDT3 requires at least two decision variables")

        start = time.perf_counter()
        f1 = vector[0]
        tail = vector[1:]
        tail_mean = sum(tail) / max(len(tail), 1)
        g = 1.0 + 9.0 * tail_mean
        ratio = f1 / g if g != 0 else 0.0
        ratio = min(max(ratio, 0.0), 1.0)
        f2 = g * (1.0 - math.sqrt(ratio) - ratio * math.sin(10.0 * math.pi * f1))

        if self.noise_std > 0:
            rng = random.Random()
            rng.seed(self._seed(params, seed))
            f1 += rng.gauss(0.0, self.noise_std)
            f2 += rng.gauss(0.0, self.noise_std)

        elapsed = time.perf_counter() - start
        return {
            "f1": float(f1),
            "f2": float(f2),
            "g": float(g),
            "status": "ok",
            "timed_out": False,
            "terminated_early": False,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _seed(params: Mapping[str, Any], base_seed: int | None) -> int:
        combined = hash(tuple(sorted(params.items())))
        if base_seed is not None:
            combined ^= int(base_seed)
        return combined & 0xFFFFFFFF


def create_zdt3_evaluator(config: Mapping[str, Any]) -> ZDT3Evaluator:
    """Factory helper used by YAML configs."""

    variables_cfg = config.get("variables")
    if variables_cfg is None:
        variables = ("x1", "x2", "x3", "x4", "x5")
    elif isinstance(variables_cfg, Sequence) and not isinstance(variables_cfg, (bytes, bytearray, str)):
        variables = tuple(str(entry) for entry in variables_cfg)
        if len(variables) < 2:
            raise ValueError("zdt3 evaluator requires at least two variable names")
    else:
        raise TypeError("evaluator.variables must be a sequence of names")

    noise_std = float(config.get("noise_std", 0.0))
    return ZDT3Evaluator(variables=variables, noise_std=noise_std)
