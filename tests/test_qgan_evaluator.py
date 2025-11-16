"""Tests for the analytic qGAN KL evaluator."""
from __future__ import annotations

import math
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
        self.assertIn("status", metrics)
        self.assertIn("elapsed_seconds", metrics)
        self.assertIn("timed_out", metrics)
        self.assertIn("terminated_early", metrics)
        self.assertGreaterEqual(metrics["kl"], 0.0)
        self.assertEqual(metrics["depth"], float(self.params["depth"]))
        self.assertEqual(metrics["shots"], float(self.config["shots"]))
        self.assertEqual(metrics["params"], float(len(self.params)))
        self.assertEqual(metrics["status"], "ok")
        self.assertFalse(metrics["timed_out"])
        self.assertFalse(metrics["terminated_early"])
        self.assertGreaterEqual(metrics["elapsed_seconds"], 0.0)

    def test_deterministic_with_seed(self) -> None:
        evaluator = QGANKLEvaluator(backend="pennylane", shots=256)
        first = evaluator.evaluate(self.params, seed=777)
        second = evaluator.evaluate(self.params, seed=777)
        for key in first:
            if key == "elapsed_seconds":
                continue
            self.assertEqual(first[key], second[key])

    def test_seed_changes_metrics(self) -> None:
        evaluator = QGANKLEvaluator(backend="pennylane", shots=256)
        first = evaluator.evaluate(self.params, seed=1)
        second = evaluator.evaluate(self.params, seed=2)
        self.assertNotEqual(first, second)

    def test_nan_failure_returns_error_payload(self) -> None:
        evaluator = QGANKLEvaluator(backend="pennylane", shots=256)
        params = {"theta": float("nan"), "depth": 1}
        metrics = evaluator.evaluate(params, seed=0)
        self.assertEqual(metrics["status"], "error")
        self.assertEqual(metrics["reason"], "nan_detected")
        self.assertFalse(metrics["timed_out"])
        self.assertTrue(math.isinf(metrics["kl"]))

    def test_timeout_status_is_reported(self) -> None:
        evaluator = QGANKLEvaluator(backend="pennylane", shots=256, timeout_seconds=0.0)
        metrics = evaluator.evaluate(self.params, seed=99)
        self.assertEqual(metrics["status"], "timeout")
        self.assertTrue(metrics["timed_out"])
        self.assertEqual(metrics.get("reason"), "timeout_exceeded")
        self.assertTrue(math.isinf(metrics["kl"]))

    def test_shots_parameter_overrides_default(self) -> None:
        evaluator = QGANKLEvaluator(backend="pennylane", shots=512)
        params = dict(self.params)
        params["shots"] = 64
        metrics = evaluator.evaluate(params, seed=5)
        self.assertEqual(metrics["shots"], 64.0)
        reference = evaluator.evaluate(self.params, seed=5)
        self.assertEqual(reference["shots"], 512.0)
        self.assertNotEqual(metrics["kl"], reference["kl"])  # ノイズスケールが変わる

    def test_loader_returns_callable(self) -> None:
        evaluator_fn = load_evaluator(self.config)
        metrics = evaluator_fn(self.params, seed=42)
        self.assertIn("kl", metrics)
        self.assertIsInstance(metrics["shots"], float)
        self.assertEqual(metrics["status"], "ok")


if __name__ == "__main__":
    unittest.main()
