"""Unit tests for optimization helper utilities."""
from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

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


class TrialLoggerTests(unittest.TestCase):
    def test_logger_handles_prefixed_and_unprefixed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "log.csv"
            metrics = {
                "metric_fidelity": 0.87,
                "energy": -0.5,
            }
            params = {"width": 2}

            with optimization.TrialLogger(
                log_path,
                params.keys(),
                ["fidelity", "metric_energy"],
            ) as logger:
                logger.log(0, params, metrics)

            with log_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn("metric_fidelity", row)
        self.assertIn("metric_energy", row)
        self.assertNotIn("metric_metric_energy", row)
        self.assertAlmostEqual(float(row["metric_fidelity"]), 0.87)
        self.assertAlmostEqual(float(row["metric_energy"]), -0.5)


if __name__ == "__main__":
    unittest.main()
