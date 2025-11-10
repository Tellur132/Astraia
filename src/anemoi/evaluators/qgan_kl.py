"""Placeholder evaluator for qGAN KL divergence."""
from __future__ import annotations

from typing import Any, Dict


def evaluate(params: Dict[str, Any], seed: int | None = None) -> Dict[str, float]:
    """Return dummy metrics for the initial MVP step.

    Parameters
    ----------
    params:
        Dictionary of parameter names to values.
    seed:
        Optional random seed placeholder.

    Returns
    -------
    dict
        Dummy metrics shaped like the future real evaluator output.
    """

    # This is intentionally a stub; future steps will implement the real evaluation.
    kl = float(len(params)) * 0.1
    depth = float(params.get("depth", 1))
    shots = 256.0
    param_count = float(len(params))

    return {
        "kl": kl,
        "depth": depth,
        "shots": shots,
        "params": param_count,
    }
