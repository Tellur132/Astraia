from __future__ import annotations

import unittest

from astraia.config import OptimizationConfig, ValidationError


def make_base_config() -> dict:
    return {
        "metadata": {
            "name": "experiment",
            "description": "example",
        },
        "seed": 123,
        "search": {
            "library": "optuna",
            "sampler": "tpe",
            "n_trials": 4,
            "direction": "minimize",
            "metric": "kl",
        },
        "stopping": {
            "max_trials": 4,
            "max_time_minutes": 5,
            "no_improve_patience": 2,
        },
        "search_space": {
            "theta": {
                "type": "float",
                "low": -1.0,
                "high": 1.0,
            }
        },
        "evaluator": {
            "module": "astraia.evaluators.qgan_kl",
            "callable": "create_evaluator",
        },
        "report": {
            "metrics": ["kl"],
        },
    }


class OptimizationConfigValidationTests(unittest.TestCase):
    def test_valid_configuration_passes(self) -> None:
        config = OptimizationConfig.model_validate(make_base_config())
        self.assertEqual(config.search.metric, "kl")

    def test_metric_must_be_in_report_metrics(self) -> None:
        data = make_base_config()
        data["report"]["metrics"] = ["depth"]
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_search_space_requires_valid_ranges(self) -> None:
        data = make_base_config()
        data["search_space"]["theta"]["low"] = 10
        data["search_space"]["theta"]["high"] = 1
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_missing_required_field_fails(self) -> None:
        data = make_base_config()
        del data["evaluator"]
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
