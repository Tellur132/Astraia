"""Unit tests for optimization helper utilities."""
from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import optuna

from astraia import optimization


class AdaptiveLLMUsageTests(unittest.TestCase):
    def test_bandit_boosts_ratio_on_llm_improvement(self) -> None:
        cfg = {
            "enabled": True,
            "mix_ratio": 0.1,
            "mix_ratio_floor": 0.05,
            "adaptive_max_ratio": 0.6,
            "adaptive_usage_prior": 0.2,
            "mix_ratio_decay": 0.5,
        }
        controller = optimization.LLMUsageOptimizer(
            cfg,
            direction_names=["minimize"],
            no_improve_patience=10,
            seed=2,
        )

        first = controller.update_after_trial(
            study=None,
            trials_completed=1,
            is_llm_trial=True,
            improved=True,
            pareto_improved=False,
            no_improve_counter=0,
            pareto_no_improve_counter=None,
        )
        self.assertIsNotNone(first.ratio)
        self.assertGreater(controller.current_ratio, 0.1)

        second = controller.update_after_trial(
            study=None,
            trials_completed=2,
            is_llm_trial=False,
            improved=True,
            pareto_improved=False,
            no_improve_counter=0,
            pareto_no_improve_counter=None,
        )
        self.assertIsNotNone(second.ratio)
        first_ratio = first.ratio if first.ratio is not None else controller.current_ratio
        self.assertLessEqual(controller.current_ratio, first_ratio)

    def test_stagnation_triggers_forced_llm(self) -> None:
        cfg = {
            "enabled": True,
            "mix_ratio": 0.05,
            "mix_ratio_floor": 0.05,
            "adaptive_max_ratio": 0.4,
            "stagnation_trials": 2,
            "stagnation_boost": 0.15,
            "adaptive_cooldown_trials": 0,
        }
        controller = optimization.LLMUsageOptimizer(
            cfg,
            direction_names=["minimize", "minimize"],
            no_improve_patience=None,
            seed=3,
        )

        first = controller.update_after_trial(
            study=None,
            trials_completed=1,
            is_llm_trial=False,
            improved=False,
            pareto_improved=False,
            no_improve_counter=1,
            pareto_no_improve_counter=1,
        )
        self.assertFalse(first.force_llm)

        second = controller.update_after_trial(
            study=None,
            trials_completed=2,
            is_llm_trial=False,
            improved=False,
            pareto_improved=False,
            no_improve_counter=2,
            pareto_no_improve_counter=2,
        )
        self.assertTrue(second.force_llm)
        self.assertGreaterEqual(controller.current_ratio, 0.2)


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

    def test_qaoa_layers_gate_unused_angles(self) -> None:
        search_space = {
            "n_layers": {"type": "int", "low": 1, "high": 3},
            "gamma_0": {"type": "float", "low": 0.0, "high": 1.0},
            "beta_0": {"type": "float", "low": 0.0, "high": 1.0},
            "gamma_1": {"type": "float", "low": 0.0, "high": 1.0},
            "beta_1": {"type": "float", "low": 0.0, "high": 1.0},
        }
        study = optuna.create_study()
        study.enqueue_trial({"n_layers": 1})
        trial = study.ask()

        params = optimization.sample_params(trial, search_space)

        self.assertEqual(params["n_layers"], 1)
        self.assertIn("gamma_0", params)
        self.assertIn("beta_0", params)
        self.assertEqual(params["gamma_1"], 0.0)
        self.assertEqual(params["beta_1"], 0.0)


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
