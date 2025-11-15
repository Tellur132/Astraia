import json
import tempfile
import unittest
from pathlib import Path

from astraia.config import OptimizationConfig
from astraia.tracking import (
    RunHandle,
    RunMetadata,
    create_run,
    list_runs,
    load_run,
    update_run_status,
)


def _make_config() -> dict:
    return {
        "metadata": {"name": "Example Run", "description": "demo"},
        "seed": 7,
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
            "theta": {"type": "float", "low": -1.0, "high": 1.0},
        },
        "evaluator": {
            "module": "astraia.evaluators.qgan_kl",
            "callable": "create_evaluator",
        },
        "report": {"metrics": ["kl"]},
    }


class TrackingModuleTests(unittest.TestCase):
    def test_create_update_and_list_runs(self) -> None:
        config = OptimizationConfig.model_validate(_make_config())

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "runs"
            handle = create_run(config, runs_root=runs_root)

            self.assertIsInstance(handle, RunHandle)
            self.assertTrue(handle.run_dir.exists())
            self.assertEqual(handle.run_id, handle.artifacts.run_id)

            loaded = load_run(handle.run_id, runs_root=runs_root)
            self.assertIsInstance(loaded, RunMetadata)
            self.assertEqual(loaded.status, "created")
            self.assertEqual(loaded.status_payload.get("state"), "created")
            self.assertEqual(
                loaded.artifact_path("config_original"),
                handle.artifacts.config_original,
            )

            refreshed = update_run_status(
                handle.run_id,
                "running",
                runs_root=runs_root,
                progress=0.5,
            )

            self.assertEqual(refreshed.status, "running")
            self.assertEqual(refreshed.status_payload.get("progress"), 0.5)
            self.assertIsNotNone(refreshed.status_updated_at)

            meta_raw = json.loads(refreshed.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta_raw["status"]["state"], "running")
            self.assertEqual(meta_raw["status"]["progress"], 0.5)

            all_runs = list_runs(runs_root=runs_root)
            self.assertEqual([run.run_id for run in all_runs], [handle.run_id])

            filtered = list_runs({"status": "running"}, runs_root=runs_root)
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].run_id, handle.run_id)

            by_name = list_runs({"metadata.name": "Example Run"}, runs_root=runs_root)
            self.assertEqual(len(by_name), 1)
            self.assertEqual(by_name[0].run_id, handle.run_id)

            with self.assertRaises(FileNotFoundError):
                load_run("missing", runs_root=runs_root)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

