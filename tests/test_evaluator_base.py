"""Unit tests covering BaseEvaluator infrastructure features."""
from __future__ import annotations

import math
import time
import unittest
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency in tests
    import numpy as np
except Exception:  # noqa: BLE001
    np = None  # type: ignore[assignment]

from astraia.evaluators.base import BaseEvaluator, GracefulNaNPolicy


class _BaseTestEvaluator(BaseEvaluator):
    """Minimal evaluator returning a fixed payload for testing."""

    def _evaluate_impl(
        self,
        params: Mapping[str, Any],
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        return {
            "kl": 0.1,
            "depth": float(len(params) or 1),
            "shots": 32.0,
            "params": float(len(params)),
            "status": "ok",
            "timed_out": False,
            "terminated_early": False,
        }


class BaseEvaluatorExecutionTests(unittest.TestCase):
    def test_timeout_returns_structured_payload(self) -> None:
        class SlowEvaluator(_BaseTestEvaluator):
            def _evaluate_impl(
                self,
                params: Mapping[str, Any],
                seed: int | None = None,
            ) -> Mapping[str, Any]:
                time.sleep(0.2)
                return super()._evaluate_impl(params, seed)

        evaluator = SlowEvaluator()
        metrics = evaluator.evaluate({"alpha": 1}, trial_timeout_sec=0.01)
        self.assertEqual(metrics["status"], "timeout")
        self.assertTrue(metrics["timed_out"])
        self.assertTrue(math.isinf(metrics["kl"]))
        self.assertGreaterEqual(metrics["depth"], 0.0)

    def test_max_retries_allows_recovery(self) -> None:
        class FlakyEvaluator(_BaseTestEvaluator):
            def __init__(self) -> None:
                self._attempts = 0

            def _evaluate_impl(
                self,
                params: Mapping[str, Any],
                seed: int | None = None,
            ) -> Mapping[str, Any]:
                self._attempts += 1
                if self._attempts == 1:
                    raise RuntimeError("transient")
                return super()._evaluate_impl(params, seed)

        evaluator = FlakyEvaluator()
        metrics = evaluator.evaluate({"beta": 2}, max_retries=1)
        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(evaluator._attempts, 2)

    def test_exception_after_retries_returns_failure_payload(self) -> None:
        class AlwaysFailEvaluator(_BaseTestEvaluator):
            def _evaluate_impl(
                self,
                params: Mapping[str, Any],
                seed: int | None = None,
            ) -> Mapping[str, Any]:
                raise RuntimeError("permanent")

        evaluator = AlwaysFailEvaluator()
        metrics = evaluator.evaluate({"gamma": 3}, max_retries=2)
        self.assertEqual(metrics["status"], "error")
        self.assertEqual(metrics["reason"], "exception:RuntimeError")
        self.assertTrue(math.isinf(metrics["kl"]))


class BaseEvaluatorNanPolicyTests(unittest.TestCase):
    def test_mark_failure_policy(self) -> None:
        class NanEvaluator(_BaseTestEvaluator):
            def _evaluate_impl(
                self,
                params: Mapping[str, Any],
                seed: int | None = None,
            ) -> Mapping[str, Any]:
                payload = super()._evaluate_impl(params, seed)
                payload = dict(payload)
                payload["kl"] = float("nan")
                return payload

        evaluator = NanEvaluator()
        metrics = evaluator.evaluate({"x": 1}, graceful_nan_policy=GracefulNaNPolicy.MARK_FAILURE)
        self.assertEqual(metrics["status"], "error")
        self.assertIn("invalid_metric:kl", metrics["reason"])
        self.assertTrue(math.isinf(metrics["kl"]))

    def test_coerce_policy_sets_infinite_metric(self) -> None:
        class NanEvaluator(_BaseTestEvaluator):
            def _evaluate_impl(
                self,
                params: Mapping[str, Any],
                seed: int | None = None,
            ) -> Mapping[str, Any]:
                payload = super()._evaluate_impl(params, seed)
                payload = dict(payload)
                payload["kl"] = float("nan")
                return payload

        evaluator = NanEvaluator()
        metrics = evaluator.evaluate({"x": 1}, graceful_nan_policy=GracefulNaNPolicy.COERCE_TO_INF)
        self.assertEqual(metrics["status"], "error")
        self.assertTrue(math.isinf(metrics["kl"]))
        self.assertIn("invalid_metric:kl", metrics["reason"])


@unittest.skipIf(np is None, "numpy is required for RNG propagation tests")
class BaseEvaluatorSeedTests(unittest.TestCase):
    def test_seed_propagation_to_numpy(self) -> None:
        class NumpyEvaluator(_BaseTestEvaluator):
            def _evaluate_impl(
                self,
                params: Mapping[str, Any],
                seed: int | None = None,
            ) -> Mapping[str, Any]:
                value = float(np.random.random())
                payload = super()._evaluate_impl(params, seed)
                payload = dict(payload)
                payload["kl"] = value
                return payload

        evaluator = NumpyEvaluator()
        first = evaluator.evaluate({"x": 1}, seed=1234)
        second = evaluator.evaluate({"x": 1}, seed=1234)
        self.assertEqual(first["kl"], second["kl"])

    def test_numpy_state_restored_after_evaluation(self) -> None:
        class NumpyEvaluator(_BaseTestEvaluator):
            def _evaluate_impl(
                self,
                params: Mapping[str, Any],
                seed: int | None = None,
            ) -> Mapping[str, Any]:
                np.random.random()
                return super()._evaluate_impl(params, seed)

        np.random.seed(2024)
        state_before = np.random.get_state()
        evaluator = NumpyEvaluator()
        evaluator.evaluate({"z": 5}, seed=999)
        state_after = np.random.get_state()
        for before, after in zip(state_before, state_after):
            if isinstance(before, (tuple, list)):
                self.assertEqual(before, after)
            elif hasattr(before, "shape"):
                self.assertTrue(np.array_equal(before, after))
            else:
                self.assertEqual(before, after)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
