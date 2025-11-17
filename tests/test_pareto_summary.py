import optuna

from astraia.meta_search import TrialRecord
from astraia.pareto_summary import ParetoSummaryGenerator


def make_record(number: int, kl: float, depth: float) -> TrialRecord:
    return TrialRecord(
        number=number,
        value=kl,
        improved=False,
        params={"depth": depth},
        metrics={"kl": kl, "depth": depth},
    )


def test_pareto_summary_returns_representatives_and_notes() -> None:
    generator = ParetoSummaryGenerator(
        ["kl", "depth"],
        [optuna.study.StudyDirection.MINIMIZE, optuna.study.StudyDirection.MINIMIZE],
        max_representatives=3,
    )
    history = [
        make_record(0, 0.3, 5),
        make_record(1, 0.2, 6),
        make_record(2, 0.18, 8),
        make_record(3, 0.15, 10),
        make_record(4, 0.12, 13),
    ]

    representatives, notes = generator.summarise(history)

    assert representatives, "expected Pareto representatives to be generated"
    assert any("kl" in note for note in notes), "summary should mention primary metric"
    assert any("depth" in note for note in notes), "trade-off note should mention secondary metric"
