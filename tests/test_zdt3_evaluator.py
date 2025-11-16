from __future__ import annotations

import unittest

from astraia.evaluators.zdt3 import ZDT3Evaluator, create_zdt3_evaluator


class ZDT3EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"x1": 0.3, "x2": 0.7, "x3": 0.2, "x4": 0.5, "x5": 0.1}

    def test_returns_expected_metrics(self) -> None:
        evaluator = ZDT3Evaluator()
        metrics = evaluator(self.params, seed=0)
        self.assertIn("f1", metrics)
        self.assertIn("f2", metrics)
        self.assertIn("g", metrics)
        self.assertEqual(metrics["status"], "ok")
        self.assertGreaterEqual(metrics["g"], 1.0)

    def test_factory_accepts_custom_variables(self) -> None:
        config = {"variables": ["a", "b"]}
        evaluator = create_zdt3_evaluator(config)
        metrics = evaluator({"a": 0.2, "b": 0.8}, seed=1)
        self.assertAlmostEqual(metrics["f1"], 0.2)
        self.assertIn("f2", metrics)

    def test_noise_std_is_deterministic_with_seed(self) -> None:
        evaluator = ZDT3Evaluator(noise_std=0.01)
        first = evaluator(self.params, seed=123)
        second = evaluator(self.params, seed=123)
        self.assertEqual(first["f1"], second["f1"])
        self.assertEqual(first["f2"], second["f2"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
