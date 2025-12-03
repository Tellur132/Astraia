import json
from pathlib import Path

import pytest

from astraia.config import OptimizationConfig
from astraia.run_summary import summarize_run_results, write_run_summary
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


def _write_llm_usage(path: Path) -> None:
    lines = [
        "timestamp,provider,model,request_id,prompt_tokens,completion_tokens,total_tokens",
        "2024-01-01T00:00:00Z,openai,gpt-5,id-a,10,5,15",
        "2024-01-01T00:00:01Z,openai,gpt-5,id-b,4,6,10",
    ]
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


def test_write_run_summary_emits_llm_metrics(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    log_path = run_dir / "log.csv"
    log_lines = [
        "trial,param_theta,metric_energy_gap,metric_depth",
        "0,0.1,0.3,5",
        "1,0.2,0.1,3",
    ]
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    usage_path = run_dir / "llm_usage.csv"
    _write_llm_usage(usage_path)

    summary, summary_path = write_run_summary(
        run_id="demo",
        run_dir=run_dir,
        log_path=log_path,
        report_path=run_dir / "report.md",
        trials_completed=2,
        best_params={"theta": 0.2},
        best_metrics={"metric_energy_gap": 0.1, "metric_depth": 3},
        best_value=0.1,
        pareto_front=[{"values": {"energy_gap": 0.1}}],
        hypervolume=1.23,
        llm_usage_path=usage_path,
        llm_trials=1,
        seed=99,
        config={"report": {"metrics": ["energy_gap", "depth"]}},
    )

    assert summary_path.exists()
    assert summary["pareto_count"] == 1
    assert summary["llm_calls"] == 2
    assert summary["tokens"] == 25
    assert summary["llm_trials"] == 1
    assert summary["llm_accept_rate"] == 0.5
    assert summary["best_energy_gap"] == 0.1
    assert summary["depth_best"] == 3
