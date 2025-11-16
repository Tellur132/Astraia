from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import optuna
from optuna.trial import TrialState

from astraia.config import OptimizationConfig
from astraia.optimization import build_report


class ParetoReportTests(unittest.TestCase):
    def _make_config(self, report_dir: Path) -> dict:
        data = {
            "metadata": {
                "name": "multi", 
                "description": "multi-objective run",
            },
            "seed": 42,
            "search": {
                "library": "optuna",
                "sampler": "tpe",
                "n_trials": 6,
                "multi_objective": True,
                "metrics": ["kl", "depth"],
                "directions": ["minimize", "minimize"],
            },
            "stopping": {
                "max_trials": 6,
            },
            "search_space": {
                "lr": {"type": "float", "low": 0.0, "high": 1.0},
                "depth": {"type": "float", "low": 1.0, "high": 10.0},
                "alpha": {"type": "float", "low": 0.0, "high": 1.0},
            },
            "evaluator": {
                "module": "astraia.evaluators.qgan_kl",
                "callable": "create_evaluator",
            },
            "report": {
                "metrics": ["kl", "depth"],
                "output_dir": str(report_dir),
            },
            "artifacts": {},
        }
        return OptimizationConfig.model_validate(data).model_dump(mode="python")

    def test_report_includes_representative_summary_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            report_dir = base / "reports"
            report_dir.mkdir()

            config = self._make_config(report_dir)

            study = optuna.create_study(directions=("minimize", "minimize"))
            distributions = {
                "lr": optuna.distributions.FloatDistribution(0.0, 1.0),
                "depth": optuna.distributions.FloatDistribution(1.0, 10.0),
                "alpha": optuna.distributions.FloatDistribution(0.0, 1.0),
            }
            trial_payloads = [
                ([0.25, 6.0], {"lr": 0.2, "depth": 6.0, "alpha": 0.6}),
                ([0.18, 8.0], {"lr": 0.35, "depth": 8.0, "alpha": 0.1}),
                ([0.35, 4.0], {"lr": 0.1, "depth": 4.0, "alpha": 0.4}),
            ]
            for values, params in trial_payloads:
                study.add_trial(
                    optuna.trial.create_trial(
                        params=params,
                        distributions=distributions,
                        values=values,
                        state=TrialState.COMPLETE,
                    )
                )

            report_path, pareto_records, _ = build_report(
                config,
                best_params=trial_payloads[0][1],
                best_metrics={"kl": 0.25, "depth": 6.0},
                trials_completed=3,
                early_stop_reason=None,
                metric_names=["kl", "depth"],
                direction_names=["minimize", "minimize"],
                study=study,
                total_cost=None,
                cost_metric=None,
                seed=42,
            )

            content = report_path.read_text(encoding="utf-8")
            self.assertIn("### Representative Solutions", content)
            self.assertIn("Balanced (weighted sum)", content)
            self.assertIn("Best kl (minimize)", content)
            self.assertIn("Best depth (minimize)", content)

            csv_path = report_dir / "multi_pareto.csv"
            self.assertTrue(csv_path.exists())
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("param_lr", csv_text)
            self.assertIn("param_depth", csv_text)

            self.assertIsNotNone(pareto_records)
            assert pareto_records is not None
            self.assertTrue(all("params" in record for record in pareto_records))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
