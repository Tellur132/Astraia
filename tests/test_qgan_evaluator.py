"""Tests for the analytic qGAN KL evaluator."""
from __future__ import annotations

import unittest

from astraia.evaluators.qgan_kl import QGANKLEvaluator, create_evaluator
from astraia.optimization import load_evaluator


class QGANKLEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"theta": 0.25, "depth": 2}
        self.config = {
            "module": "astraia.evaluators.qgan_kl",
            "callable": "create_evaluator",
            "backend": "pennylane",
            "shots": 128,
        }

    def test_metrics_structure(self) -> None:
        evaluator = create_evaluator(self.config)
        metrics = evaluator.evaluate(self.params, seed=123)
        self.assertIn("kl", metrics)
        self.assertIn("depth", metrics)
        self.assertIn("shots", metrics)
        self.assertIn("params", metrics)
        self.assertGreaterEqual(metrics["kl"], 0.0)
        self.assertEqual(metrics["depth"], float(self.params["depth"]))
        self.assertEqual(metrics["shots"], float(self.config["shots"]))

    def test_deterministic_with_seed(self) -> None:
        evaluator = QGANKLEvaluator(backend="pennylane", shots=256)
        first = evaluator.evaluate(self.params, seed=777)
        second = evaluator.evaluate(self.params, seed=777)
        self.assertEqual(first, second)

    def test_seed_changes_metrics(self) -> None:
        evaluator = QGANKLEvaluator(backend="pennylane", shots=256)
        first = evaluator.evaluate(self.params, seed=1)
        second = evaluator.evaluate(self.params, seed=2)
        self.assertNotEqual(first, second)

    def test_loader_returns_callable(self) -> None:
        evaluator_fn = load_evaluator(self.config)
        metrics = evaluator_fn(self.params, seed=42)
        self.assertIn("kl", metrics)
        self.assertIsInstance(metrics["shots"], float)


if __name__ == "__main__":
    unittest.main()
