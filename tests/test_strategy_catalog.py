from __future__ import annotations

from pathlib import Path

from astraia.strategy_catalog import (
    infer_problem_type,
    load_fewshot_examples,
    load_strategy_notes,
    record_strategy_entry,
)


def _make_config(problem_type: str = "qaoa_maxcut") -> dict:
    evaluator_module = "astraia.evaluators.qaoa" if problem_type == "qaoa_maxcut" else "astraia.evaluators.zdt3"
    metrics = ["energy"] if problem_type == "qaoa_maxcut" else ["f1", "f2"]
    directions = "minimize" if len(metrics) == 1 else ["minimize", "minimize"]
    search_cfg: dict[str, object] = {
        "library": "optuna",
        "sampler": "tpe",
        "n_trials": 3,
        "metric": metrics[0] if len(metrics) == 1 else None,
        "metrics": metrics if len(metrics) > 1 else None,
        "direction": directions if isinstance(directions, str) else None,
        "directions": directions if isinstance(directions, list) else None,
        "multi_objective": len(metrics) > 1,
    }
    return {
        "metadata": {"name": "demo", "description": f"{problem_type} demo"},
        "search": search_cfg,
        "stopping": {"max_trials": 3},
        "search_space": {"x": {"type": "float", "low": 0.0, "high": 1.0}},
        "evaluator": {"module": evaluator_module, "callable": "create"},
        "report": {"metrics": metrics},
    }


def test_record_strategy_entry_and_notes(tmp_path: Path) -> None:
    log_path = tmp_path / "log.csv"
    log_path.write_text(
        "\n".join(
            [
                "trial,param_x,metric_energy",
                "0,0.1,1.2",
                "1,0.2,0.9",
                "2,0.3,0.8",
            ]
        ),
        encoding="utf-8",
    )

    config = _make_config("qaoa_maxcut")
    summary = {
        "trials_completed": 3,
        "best_params": {"x": 0.3},
        "best_metrics": {"metric_energy": 0.8},
    }
    catalog_path = tmp_path / "catalog.jsonl"

    entry = record_strategy_entry(
        problem_type="qaoa_maxcut",
        config=config,
        summary=summary,
        log_path=log_path,
        catalog_path=catalog_path,
    )

    assert entry is not None
    assert catalog_path.exists()
    notes = load_strategy_notes("qaoa_maxcut", catalog_path=catalog_path, top_k=1)
    assert notes and "demo" in notes[0]


def test_infer_problem_type_and_fewshot_examples() -> None:
    config = _make_config("qaoa_maxcut")
    problem_type, _ = infer_problem_type(config)
    assert problem_type == "qaoa_maxcut"
    examples = load_fewshot_examples(problem_type)
    assert examples, "few-shot examples should be available for qaoa_maxcut"
