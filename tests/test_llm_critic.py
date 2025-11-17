from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import optuna

from astraia.config import OptimizationConfig
from astraia.optimization import build_report


def _make_config() -> dict:
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


def _make_multi_config(log_path: Path, report_dir: Path) -> dict:
    data = {
        "metadata": {"name": "multi", "description": "multi-objective"},
        "seed": 7,
        "search": {
            "library": "optuna",
            "sampler": "tpe",
            "n_trials": 4,
            "multi_objective": True,
            "metrics": ["kl", "depth"],
            "directions": ["minimize", "minimize"],
        },
        "stopping": {"max_trials": 4},
        "search_space": {
            "lr": {"type": "float", "low": 0.0, "high": 1.0},
            "depth": {"type": "float", "low": 1.0, "high": 10.0},
        },
        "evaluator": {
            "module": "astraia.evaluators.qgan_kl",
            "callable": "create_evaluator",
        },
        "report": {
            "metrics": ["kl", "depth"],
            "output_dir": str(report_dir),
        },
        "artifacts": {"log_file": str(log_path)},
    }
    return data


class LLMCriticReportTests(unittest.TestCase):
    def test_build_report_includes_llm_section_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_path = tmp_path / "run_log.csv"
            report_dir = tmp_path / "reports"
            report_dir.mkdir()

            log_path.write_text(
                "trial,param_theta,metric_kl\n"
                "0,0.1,0.5\n"
                "1,0.2,NaN\n"
                "2,0.3,0.51\n"
                "3,0.4,0.51\n",
                encoding="utf-8",
            )

            data = _make_config()
            data["metadata"]["name"] = "demo"
            data["metadata"]["description"] = "demo run"
            data["llm"] = {"provider": "dummy", "model": "stub"}
            data["llm_critic"] = {"enabled": True}
            data["artifacts"] = {"log_file": str(log_path)}
            data["report"]["output_dir"] = str(report_dir)

            config = OptimizationConfig.model_validate(data).model_dump(mode="python")

            study = optuna.create_study(direction="minimize")

            report_path, _, _ = build_report(
                config,
                best_params={"theta": 0.3},
                best_metrics={"kl": 0.49},
                trials_completed=4,
                early_stop_reason=None,
                metric_names=["kl"],
                direction_names=["minimize"],
                study=study,
                total_cost=None,
                cost_metric=None,
                seed=123,
            )

            content = report_path.read_text(encoding="utf-8")
            self.assertIn("## LLM考察", content)
            self.assertIn("NaN", content)
            self.assertIn("制約調整", content)

    def test_multi_objective_fallback_includes_tradeoff_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_path = tmp_path / "run_log.csv"
            report_dir = tmp_path / "reports"
            report_dir.mkdir()

            log_path.write_text(
                "trial,param_lr,param_depth,metric_kl,metric_depth\n"
                "0,0.1,5,0.42,6\n"
                "1,0.2,6,0.35,7.5\n"
                "2,0.3,7,0.3,8.2\n"
                "3,0.4,8,0.32,9.1\n",
                encoding="utf-8",
            )

            data = _make_multi_config(log_path, report_dir)
            data["llm"] = {"provider": "dummy", "model": "stub"}
            data["llm_critic"] = {"enabled": True}

            config = OptimizationConfig.model_validate(data).model_dump(mode="python")

            study = optuna.create_study(directions=("minimize", "minimize"))
            distributions = {
                "lr": optuna.distributions.FloatDistribution(0.0, 1.0),
                "depth": optuna.distributions.FloatDistribution(1.0, 10.0),
            }
            trials = [
                ([0.42, 6.0], {"lr": 0.1, "depth": 5.0}),
                ([0.35, 7.5], {"lr": 0.2, "depth": 6.0}),
                ([0.3, 8.2], {"lr": 0.25, "depth": 7.0}),
            ]
            for values, params in trials:
                study.add_trial(
                    optuna.trial.create_trial(
                        params=params,
                        distributions=distributions,
                        values=values,
                        state=optuna.trial.TrialState.COMPLETE,
                    )
                )

            report_path, _, _ = build_report(
                config,
                best_params=trials[0][1],
                best_metrics={"kl": 0.42, "depth": 6.0},
                trials_completed=4,
                early_stop_reason=None,
                metric_names=["kl", "depth"],
                direction_names=["minimize", "minimize"],
                study=study,
                total_cost=None,
                cost_metric=None,
                seed=7,
            )

            content = report_path.read_text(encoding="utf-8")
            self.assertIn("## LLM考察", content)
            self.assertIn("ボトルネック候補", content)
            self.assertIn("計算資源で伸ばせる目的", content)
            self.assertIn("次回実験で調整したい点", content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

