import contextlib
import io
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from astraia import cli
from astraia.config import OptimizationConfig
from astraia.tracking import create_run


def _make_config(name: str) -> OptimizationConfig:
    metrics = ["loss", "accuracy"]
    directions = ["minimize", "maximize"]
    search: dict[str, object] = {
        "library": "optuna",
        "sampler": "tpe",
        "n_trials": 4,
        "metrics": metrics,
        "directions": directions,
        "multi_objective": True,
    }
    cfg = {
        "metadata": {"name": name, "description": f"{name} description"},
        "seed": 1,
        "search": search,
        "stopping": {
            "max_trials": 4,
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
        "report": {"metrics": metrics},
    }
    return OptimizationConfig.model_validate(cfg)


class VisualizeCliTests(TestCase):
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
    def _write_log(path: Path, rows: list[tuple[float, float]]) -> None:
        lines = ["trial,param_theta,metric_loss,metric_accuracy"]
        for idx, (loss, acc) in enumerate(rows):
            lines.append(f"{idx},0.0,{loss},{acc}")
        path.write_text("\n".join(lines), encoding="utf-8")

    def test_visualize_command_generates_images(self) -> None:
        run_id = self._create_run("viz")
        log_path = self.runs_root / run_id / "log.csv"
        self._write_log(
            log_path,
            [
                (0.6, 0.2),
                (0.4, 0.3),
                (0.35, 0.32),
                (0.5, 0.35),
            ],
        )

        history_output = Path(self._tmp.name) / "history.png"
        self._invoke(
            "visualize",
            "--run-id",
            run_id,
            "--runs-root",
            str(self.runs_root),
            "--type",
            "history",
            "--output",
            str(history_output),
        )
        assert history_output.exists()

        pareto_output = Path(self._tmp.name) / "pareto.png"
        self._invoke(
            "visualize",
            "--run-id",
            run_id,
            "--runs-root",
            str(self.runs_root),
            "--type",
            "pareto",
            "--output",
            str(pareto_output),
        )
        assert pareto_output.exists()
