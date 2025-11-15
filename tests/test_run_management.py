import json
import tempfile
import unittest
from pathlib import Path

import yaml

from astraia.config import OptimizationConfig
from astraia.run_management import RunArtifacts, prepare_run_environment


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


class PrepareRunEnvironmentTests(unittest.TestCase):
    def test_creates_standard_run_layout(self) -> None:
        base_config = OptimizationConfig.model_validate(_make_config())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text(yaml.safe_dump(_make_config()), encoding="utf-8")

            final_config, artifacts = prepare_run_environment(
                base_config,
                config_source=config_path,
                runs_root=tmp_path / "runs",
            )

            self.assertIsInstance(artifacts, RunArtifacts)
            self.assertTrue(artifacts.run_dir.exists())
            self.assertTrue(artifacts.config_original.exists())
            self.assertTrue(artifacts.config_resolved.exists())
            self.assertTrue(artifacts.meta_path.exists())

            expected_log = artifacts.run_dir / "log.csv"
            expected_report = artifacts.run_dir / "report.md"

            self.assertEqual(Path(final_config["artifacts"]["log_file"]), expected_log)
            self.assertEqual(Path(final_config["report"]["output_dir"]), artifacts.run_dir)
            self.assertEqual(final_config["report"]["filename"], "report.md")

            meta = json.loads(artifacts.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["run_id"], artifacts.run_dir.name)
            self.assertEqual(meta["artifacts"]["log"], str(expected_log))
            self.assertEqual(meta["artifacts"]["report"], str(expected_report))
            self.assertEqual(meta["source"]["config_path"], str(config_path))
            self.assertNotIn("llm_usage", meta["artifacts"])

    def test_respects_existing_run_root_and_adds_llm_usage(self) -> None:
        data = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run_dir = tmp_path / "custom" / "demo"
            data["artifacts"] = {"run_root": str(run_dir)}
            data["llm"] = {"provider": "openai", "model": "gpt-4o"}

            base_config = OptimizationConfig.model_validate(data)

            final_config, artifacts = prepare_run_environment(
                base_config,
                config_source=None,
                runs_root=tmp_path / "runs",
            )

            self.assertEqual(Path(final_config["artifacts"]["run_root"]), run_dir)
            self.assertTrue(artifacts.run_dir.exists())
            self.assertIsNotNone(artifacts.llm_usage_path)
            assert artifacts.llm_usage_path is not None
            self.assertTrue(artifacts.llm_usage_path.exists())
            self.assertEqual(
                final_config["llm"]["usage_log"],
                str(artifacts.llm_usage_path),
            )

            meta = json.loads(artifacts.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["artifacts"]["llm_usage"], str(artifacts.llm_usage_path))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
