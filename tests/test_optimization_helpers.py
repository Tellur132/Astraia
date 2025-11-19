"""Unit tests for optimization helper utilities."""
from __future__ import annotations

import math
import unittest

import optuna

from astraia import optimization


class OptimizationHelperTests(unittest.TestCase):
    def test_trial_failed_detects_non_ok_status(self) -> None:
        metrics = {"status": "error", "timed_out": False}
        self.assertTrue(optimization._trial_failed(metrics))

    def test_trial_failed_detects_timeout_flag(self) -> None:
        metrics = {"status": "ok", "timed_out": True}
        self.assertTrue(optimization._trial_failed(metrics))

    def test_extract_objective_values_marks_non_finite(self) -> None:
        metrics = {"a": 1.5, "b": float("nan")}
        values, contains_non_finite = optimization._extract_objective_values(
            metrics,
            ["a", "b"],
        )
        self.assertEqual(values[0], 1.5)
        self.assertTrue(contains_non_finite)

    def test_failure_penalty_values_respects_directions(self) -> None:
        penalties = optimization._failure_penalty_values(
            [
                optuna.study.StudyDirection.MINIMIZE,
                optuna.study.StudyDirection.MAXIMIZE,
            ]
        )
        self.assertTrue(math.isinf(penalties[0]) and penalties[0] > 0)
        self.assertTrue(math.isinf(penalties[1]) and penalties[1] < 0)

    def test_llm_only_parameter_respects_enqueued_value(self) -> None:
        search_space = {"code": {"type": "llm_only", "default": "placeholder"}}
        study = optuna.create_study()
        study.enqueue_trial({"code": "from LLM"})
        trial = study.ask()

        params = optimization.sample_params(trial, search_space)

        self.assertEqual(params["code"], "from LLM")


if __name__ == "__main__":
    unittest.main()
