from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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

            report_path = build_report(
                config,
                best_params={"theta": 0.3},
                best_metrics={"kl": 0.49},
                trials_completed=4,
                early_stop_reason=None,
            )

            content = report_path.read_text(encoding="utf-8")
            self.assertIn("## LLM考察", content)
            self.assertIn("NaN", content)
            self.assertIn("制約調整", content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

