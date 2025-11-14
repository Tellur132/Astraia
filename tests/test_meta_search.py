import json
from collections import deque

import optuna

from astraia.meta_search import (
    MetaAdjustment,
    MetaSearchAdjuster,
    SearchSettings,
    apply_meta_adjustment,
)
from astraia.optimization import build_sampler
from astraia.llm_providers import LLMResult


class StubProvider:
    def __init__(self, payloads):
        self._payloads = deque(payloads)

    def generate(
        self,
        prompt,
        *,
        temperature=None,
        json_mode=False,
        system=None,
        tool=None,
    ):  # noqa: ANN001
        if not self._payloads:
            raise RuntimeError("no payload queued")
        return LLMResult(content=self._payloads.popleft(), usage=None, raw_response=None)


def make_search_space() -> dict[str, dict[str, object]]:
    return {
        "alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "depth": {"type": "int", "low": 1, "high": 5, "step": 1},
    }


def test_heuristic_switches_sampler_when_stalled() -> None:
    adjuster = MetaSearchAdjuster(
        interval=2,
        summary_window=2,
        direction=optuna.study.StudyDirection.MINIMIZE,
        metric_name="loss",
        search_space=make_search_space(),
        llm_cfg=None,
        seed=123,
    )
    settings = SearchSettings(sampler="tpe", max_trials=10, trial_budget=10, patience=4)

    best_value: float | None = None
    best_params: dict[str, float] = {}
    adjustment = None
    values = [1.0, 1.0, 1.1, 1.2]
    for idx, value in enumerate(values):
        improved = best_value is None or value < best_value
        if improved:
            best_value = value
            best_params = {"alpha": value}
        adjustment = adjuster.register_trial(
            trial_number=idx,
            value=value,
            improved=improved,
            params={"alpha": value},
            metrics={"loss": value},
            best_value=best_value,
            best_params=best_params,
            trials_completed=idx + 1,
            settings=settings,
        )
    assert adjustment is not None
    assert adjustment.sampler == "random"


def test_llm_plan_is_parsed() -> None:
    adjuster = MetaSearchAdjuster(
        interval=1,
        summary_window=1,
        direction=optuna.study.StudyDirection.MINIMIZE,
        metric_name="score",
        search_space=make_search_space(),
        llm_cfg=None,
        seed=42,
    )
    payload = json.dumps(
        {
            "sampler": "tpe",
            "rescale": {"alpha": 0.4},
            "trial_budget": 7,
            "max_trials": 9,
            "patience": 3,
            "notes": "集中探索",
        },
        ensure_ascii=False,
    )
    adjuster._provider = StubProvider([payload])  # type: ignore[attr-defined]
    adjuster._usage_logger = None  # type: ignore[attr-defined]

    settings = SearchSettings(sampler="random", max_trials=12, trial_budget=12, patience=5)
    adjustment = adjuster.register_trial(
        trial_number=0,
        value=0.5,
        improved=True,
        params={"alpha": 0.5},
        metrics={"score": 0.5},
        best_value=0.5,
        best_params={"alpha": 0.5},
        trials_completed=1,
        settings=settings,
    )
    assert adjustment is not None
    assert adjustment.sampler == "tpe"
    assert adjustment.rescale == {"alpha": 0.4}
    assert adjustment.trial_budget == 7
    assert adjustment.max_trials == 9
    assert adjustment.patience == 3
    assert adjustment.notes == "集中探索"


def test_apply_meta_adjustment_updates_settings() -> None:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=10))
    search_cfg: dict[str, object] = {"sampler": "tpe"}
    search_space = {"alpha": {"type": "float", "low": 0.0, "high": 1.0}}
    settings = SearchSettings(sampler="tpe", max_trials=10, trial_budget=10, patience=4)

    adjustment = MetaAdjustment(
        sampler="random",
        rescale={"alpha": 0.5},
        trial_budget=8,
        patience=2,
    )

    messages = apply_meta_adjustment(
        adjustment,
        study=study,
        search_cfg=search_cfg,
        search_space=search_space,
        best_params={"alpha": 0.6},
        settings=settings,
        trials_completed=4,
        seed=11,
        sampler_builder=build_sampler,
    )

    assert any(msg.startswith("sampler ->") for msg in messages)
    assert settings.sampler == "random"
    assert isinstance(study.sampler, optuna.samplers.RandomSampler)
    assert settings.trial_budget == 8
    assert settings.patience == 2
    assert search_space["alpha"]["low"] > 0.0
    assert search_space["alpha"]["high"] < 1.0
