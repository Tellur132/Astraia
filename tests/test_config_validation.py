from __future__ import annotations

import unittest

from astraia.config import OptimizationConfig, ValidationError


def make_base_config() -> dict:
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


class OptimizationConfigValidationTests(unittest.TestCase):
    def test_valid_configuration_passes(self) -> None:
        config = OptimizationConfig.model_validate(make_base_config())
        self.assertEqual(config.search.metric, "kl")
        self.assertEqual(config.search.metric_names, ["kl"])

    def test_metric_must_be_in_report_metrics(self) -> None:
        data = make_base_config()
        data["report"]["metrics"] = ["depth"]
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_multi_objective_configuration(self) -> None:
        data = make_base_config()
        data["search"]["multi_objective"] = True
        data["search"].pop("metric")
        data["search"]["metrics"] = ["kl", "depth"]
        data["search"]["directions"] = ["minimize", "minimize"]
        data["report"]["metrics"] = ["kl", "depth", "shots"]
        config = OptimizationConfig.model_validate(data)
        self.assertTrue(config.search.multi_objective)
        self.assertEqual(config.search.metric_names, ["kl", "depth"])
        self.assertEqual(config.search.direction_names, ["minimize", "minimize"])

    def test_metric_and_direction_length_must_match(self) -> None:
        data = make_base_config()
        data["search"]["multi_objective"] = True
        data["search"].pop("metric")
        data["search"]["metrics"] = ["kl", "depth"]
        data["search"]["directions"] = ["minimize"]
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_multi_objective_flag_requires_multiple_metrics(self) -> None:
        data = make_base_config()
        data["search"]["multi_objective"] = True
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_multiple_metrics_require_multi_objective_flag(self) -> None:
        data = make_base_config()
        data["search"]["metrics"] = ["kl", "depth"]
        data["search"].pop("metric")
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_cost_budget_requires_metric(self) -> None:
        data = make_base_config()
        data["stopping"]["max_total_cost"] = 100
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_search_space_requires_valid_ranges(self) -> None:
        data = make_base_config()
        data["search_space"]["theta"]["low"] = 10
        data["search_space"]["theta"]["high"] = 1
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_missing_required_field_fails(self) -> None:
        data = make_base_config()
        del data["evaluator"]
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_planner_backend_validation(self) -> None:
        data = make_base_config()
        data["planner"] = {"backend": "invalid", "enabled": True}
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_llm_planner_requires_prompt_and_role(self) -> None:
        data = make_base_config()
        data["planner"] = {"backend": "llm", "enabled": True, "prompt_template": ""}
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_llm_usage_log_defaults_to_run_root(self) -> None:
        data = make_base_config()
        data["artifacts"] = {"run_root": "runs/demo"}
        data["llm"] = {"provider": "openai", "model": "gpt-4o"}
        config = OptimizationConfig.model_validate(data)
        assert config.llm is not None
        self.assertEqual(config.llm.usage_log, "runs/demo/llm_usage.csv")

    def test_llm_guidance_requires_llm_when_enabled(self) -> None:
        data = make_base_config()
        data["llm_guidance"] = {
            "enabled": True,
            "problem_summary": "demo",
            "objective": "minimize",
        }
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_llm_guidance_requires_problem_and_objective(self) -> None:
        data = make_base_config()
        data["llm"] = {"provider": "openai", "model": "gpt-4o"}
        data["llm_guidance"] = {
            "enabled": True,
            "problem_summary": "",
            "objective": "",
        }
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_llm_guidance_accepts_disabled_without_llm(self) -> None:
        data = make_base_config()
        data["llm_guidance"] = {
            "enabled": False,
            "problem_summary": "demo",
            "objective": "minimize",
            "n_proposals": 2,
        }
        config = OptimizationConfig.model_validate(data)
        self.assertIsNone(config.llm)

    def test_meta_search_validation(self) -> None:
        data = make_base_config()
        data["meta_search"] = {"enabled": True, "interval": 0, "summary_trials": 5}
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

        data["meta_search"] = {"enabled": True, "interval": 5, "summary_trials": 0}
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

        data["meta_search"] = {"enabled": True, "interval": 3, "summary_trials": 2}
        config = OptimizationConfig.model_validate(data)
        assert config.meta_search is not None
        self.assertTrue(config.meta_search.enabled)

    def test_meta_search_policies_validation(self) -> None:
        data = make_base_config()
        data["meta_search"] = {
            "enabled": True,
            "interval": 4,
            "summary_trials": 2,
            "policies": [
                {
                    "name": "stagnation-switch",
                    "when": {"no_improve": 2},
                    "then": {"sampler": "nsga2"},
                }
            ],
        }
        config = OptimizationConfig.model_validate(data)
        assert config.meta_search is not None
        assert config.meta_search.policies is not None
        self.assertEqual(len(config.meta_search.policies), 1)

        data["meta_search"]["policies"] = [
            {"when": {"no_improve": 1}, "then": {}},
        ]
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

    def test_llm_critic_requires_llm_when_enabled(self) -> None:
        data = make_base_config()
        data["llm_critic"] = {"enabled": True}
        with self.assertRaises(ValidationError):
            OptimizationConfig.model_validate(data)

        data["llm"] = {"provider": "dummy", "model": "stub"}
        config = OptimizationConfig.model_validate(data)
        assert config.llm_critic is not None
        self.assertTrue(config.llm_critic.enabled)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
