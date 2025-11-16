import json
from pathlib import Path

import pytest

from astraia.config import OptimizationConfig
from astraia.run_summary import summarize_run_results
from astraia.tracking import create_run, load_run, update_run_status


def _make_config(name: str, *, direction: str = "minimize") -> OptimizationConfig:
    cfg = {
        "metadata": {"name": name, "description": f"{name} description"},
        "seed": 1,
        "search": {
            "library": "optuna",
            "sampler": "tpe",
            "n_trials": 3,
            "direction": direction,
            "metric": "score",
        },
        "stopping": {"max_trials": 3},
        "search_space": {
            "theta": {"type": "float", "low": -1.0, "high": 1.0},
        },
        "evaluator": {
            "module": "astraia.evaluators.qgan_kl",
            "callable": "create_evaluator",
        },
        "report": {"metrics": ["score"]},
    }
    return OptimizationConfig.model_validate(cfg)


def _write_log(path: Path, values: list[float]) -> None:
    lines = ["trial,param_theta,metric_score"]
    for idx, value in enumerate(values):
        lines.append(f"{idx},0.0,{value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_summarize_run_results_returns_expected_metrics(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    handle = create_run(_make_config("summary"), runs_root=runs_root)
    _write_log(handle.artifacts.log_path, [0.4, 0.1, 0.2])
    update_run_status(handle.run_id, "completed", runs_root=runs_root, note="max_trials")

    metadata = load_run(handle.run_id, runs_root=runs_root)
    config = json.loads(handle.artifacts.config_resolved.read_text(encoding="utf-8"))

    summary = summarize_run_results(
        metadata,
        config=config,
        metrics=["score"],
        statistic_names=["best", "mean"],
    )

    metric_summary = summary["metrics"]["score"]
    assert metric_summary["best"] == 0.1
    assert metric_summary["mean"] == pytest.approx(0.2333333333333333)
    assert summary["n_trials"] == 3
    assert summary["n_valid_trials"] == 3
    assert summary["early_stop_reason"] == "max_trials"
