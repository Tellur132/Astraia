from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from astraia import cli
from astraia.config import OptimizationConfig
from astraia.tracking import create_run


def _make_config(name: str) -> OptimizationConfig:
    cfg = {
        "metadata": {"name": name, "description": f"{name} description"},
        "seed": 1,
        "search": {
            "library": "optuna",
            "sampler": "tpe",
            "n_trials": 2,
            "direction": "minimize",
            "metric": "kl",
        },
        "stopping": {
            "max_trials": 2,
            "max_time_minutes": 1,
            "no_improve_patience": 1,
        },
        "search_space": {
            "theta": {"type": "float", "low": -1.0, "high": 1.0},
        },
        "evaluator": {
            "module": "astraia.evaluators.qgan_kl",
            "callable": "create_evaluator",
        },
        "report": {"metrics": ["kl"]},
    }
    return OptimizationConfig.model_validate(cfg)


class RunsCliTests(TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self._tmp.name) / "runs"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _invoke(self, *args: str) -> str:
        stdout = io.StringIO()
        with patch.object(sys, "argv", ["astraia", *args]):
            with contextlib.redirect_stdout(stdout):
                cli.main()
        return stdout.getvalue()

    def _create_run(self, name: str) -> str:
        handle = create_run(_make_config(name), runs_root=self.runs_root)
        return handle.run_id

    def test_runs_status_and_list_output(self) -> None:
        run_id = self._create_run("alpha")

        self._invoke(
            "runs",
            "status",
            "--run-id",
            run_id,
            "--state",
            "completed",
            "--best-value",
            "0.123",
            "--metric",
            "kl=0.123",
            "--runs-root",
            str(self.runs_root),
        )

        table_output = self._invoke(
            "runs",
            "list",
            "--runs-root",
            str(self.runs_root),
        )

        self.assertIn(run_id, table_output)
        self.assertIn("0.123", table_output)

        json_output = self._invoke(
            "runs",
            "list",
            "--runs-root",
            str(self.runs_root),
            "--json",
        )
        parsed = json.loads(json_output)
        self.assertEqual(parsed[0]["run_id"], run_id)
        self.assertEqual(parsed[0]["status"], "completed")

    def test_runs_show_as_json(self) -> None:
        run_id = self._create_run("beta")
        payload = self._invoke(
            "runs",
            "show",
            "--run-id",
            run_id,
            "--runs-root",
            str(self.runs_root),
            "--as-json",
        )
        data = json.loads(payload)
        self.assertIn("meta", data)
        self.assertEqual(data["meta"]["run_id"], run_id)
        self.assertIn("config_resolved", data)

    def test_runs_delete_command_removes_directory(self) -> None:
        run_id = self._create_run("gamma")
        run_dir = self.runs_root / run_id
        self.assertTrue(run_dir.exists())

        self._invoke(
            "runs",
            "delete",
            "--run-id",
            run_id,
            "--runs-root",
            str(self.runs_root),
            "--yes",
        )

        self.assertFalse(run_dir.exists())

    def test_runs_diff_command_highlights_changes(self) -> None:
        first = self._create_run("delta")
        second = self._create_run("epsilon")

        output = self._invoke(
            "runs",
            "diff",
            "--run-id",
            first,
            "--run-id",
            second,
            "--runs-root",
            str(self.runs_root),
        )

        self.assertIn("metadata.name", output)
        self.assertIn("delta", output)
        self.assertIn("epsilon", output)

    def test_runs_diff_command_reports_identical_configs(self) -> None:
        first = self._create_run("theta")
        second = self._create_run("theta")

        output = self._invoke(
            "runs",
            "diff",
            "--run-id",
            first,
            "--run-id",
            second,
            "--runs-root",
            str(self.runs_root),
        )

        self.assertIn("No configuration differences found", output)
