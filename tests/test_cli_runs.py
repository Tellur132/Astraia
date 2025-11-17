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
from astraia.tracking import create_run, load_run


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

    @staticmethod
    def _write_log(path: Path, values: list[float]) -> None:
        lines = ["trial,param_theta,metric_kl"]
        for idx, value in enumerate(values):
            lines.append(f"{idx},0.0,{value}")
        path.write_text("\n".join(lines), encoding="utf-8")

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

    def test_runs_status_supports_tags_and_pareto_summary(self) -> None:
        run_id = self._create_run("alpha-pareto")
        pareto_path = Path(self._tmp.name) / "pareto.json"
        pareto_payload = {
            "objectives": ["kl", "depth"],
            "points": [
                {"kl": 0.1, "depth": 0.9},
                {"kl": 0.2, "depth": 0.8},
            ],
        }
        pareto_path.write_text(json.dumps(pareto_payload), encoding="utf-8")

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
            "--tag",
            "multi_objective=true",
            "--tag",
            'objectives=["kl","depth"]',
            "--pareto-summary",
            str(pareto_path),
            "--runs-root",
            str(self.runs_root),
        )

        metadata = load_run(run_id, runs_root=self.runs_root)
        tags = metadata.status_payload.get("tags") or {}
        self.assertTrue(tags.get("multi_objective"))
        self.assertEqual(tags.get("objectives"), ["kl", "depth"])

        comparison_path = metadata.run_dir / "comparison_summary.json"
        self.assertTrue(comparison_path.exists())
        comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        self.assertEqual(comparison_payload["best_value"], 0.123)
        self.assertEqual(comparison_payload["pareto_summary"], pareto_payload)

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

    def test_runs_compare_command_outputs_table_and_json(self) -> None:
        handle_a = create_run(_make_config("iota"), runs_root=self.runs_root)
        handle_b = create_run(_make_config("kappa"), runs_root=self.runs_root)
        self._write_log(handle_a.artifacts.log_path, [0.5, 0.2])
        self._write_log(handle_b.artifacts.log_path, [0.4, 0.1])

        output = self._invoke(
            "runs",
            "compare",
            "--runs-root",
            str(self.runs_root),
            "--runs",
            handle_a.run_id,
            handle_b.run_id,
        )
        self.assertIn(handle_a.run_id, output)
        self.assertIn(handle_b.run_id, output)
        self.assertIn("0.2", output)

        json_payload = self._invoke(
            "runs",
            "compare",
            "--runs-root",
            str(self.runs_root),
            "--runs",
            handle_a.run_id,
            handle_b.run_id,
            "--json",
        )
        parsed = json.loads(json_payload)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["run_id"], handle_a.run_id)
        self.assertIn("metrics", parsed[0])

    def test_summarize_config_includes_objective_details(self) -> None:
        config = _make_config("summary").model_dump(mode="python")
        search = config["search"]
        search["multi_objective"] = True
        search["metrics"] = ["kl", "depth"]
        search["directions"] = ["minimize", "maximize"]
        search.pop("metric", None)
        search.pop("direction", None)

        summary = cli.summarize_config(config)
        self.assertIn("Multi-objective", summary)
        self.assertIn("minimize: kl", summary)
        self.assertIn("maximize: depth", summary)
