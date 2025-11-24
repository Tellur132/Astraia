from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

from astraia import cli
from astraia.optimization import run_optimization


class MultiObjectiveConfigSmokeTests(TestCase):
    """Ensure the bundled multi-objective configs run end-to-end."""

    CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "multiobj"

    def _run_config(self, filename: str) -> None:
        config_path = self.CONFIG_ROOT / filename
        config = cli.load_config(config_path)
        config_dict = config.model_dump(mode="python")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_root = tmp_path / "runs" / config_dict["metadata"]["name"]
            log_file = run_root / "log.csv"
            report_dir = tmp_path / "reports"

            artifacts = dict(config_dict.get("artifacts") or {})
            artifacts["run_root"] = str(run_root)
            artifacts["log_file"] = str(log_file)
            config_dict["artifacts"] = artifacts

            report_cfg = dict(config_dict.get("report") or {})
            report_cfg["output_dir"] = str(report_dir)
            config_dict["report"] = report_cfg

            result = run_optimization(config_dict)

        self.assertGreaterEqual(result.trials_completed, 1)
        self.assertIsNotNone(result.pareto_front)
        self.assertGreaterEqual(len(result.pareto_front or []), 1)

    def test_qgan_kl_depth_config_runs(self) -> None:
        self._run_config("qgan_kl_depth.yaml")

    def test_zdt3_config_runs(self) -> None:
        self._run_config("zdt3.yaml")

    def test_qft_fidelity_depth_config_runs(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "quantum" / "qft_fidelity_depth.yaml"
        config = cli.load_config(config_path)
        config_dict = config.model_dump(mode="python")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_root = tmp_path / "runs" / config_dict["metadata"]["name"]
            log_file = run_root / "log.csv"
            report_dir = tmp_path / "reports"

            artifacts = dict(config_dict.get("artifacts") or {})
            artifacts["run_root"] = str(run_root)
            artifacts["log_file"] = str(log_file)
            config_dict["artifacts"] = artifacts

            report_cfg = dict(config_dict.get("report") or {})
            report_cfg["output_dir"] = str(report_dir)
            config_dict["report"] = report_cfg

            result = run_optimization(config_dict)

        self.assertGreaterEqual(result.trials_completed, 1)
        self.assertIsNotNone(result.pareto_front)
        self.assertGreaterEqual(len(result.pareto_front or []), 1)
