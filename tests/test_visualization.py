from pathlib import Path

from astraia.visualization import ObjectiveSpec, plot_history, plot_pareto_front


def _write_log(path: Path, header: str, rows: list[str]) -> None:
    contents = [header, *rows]
    path.write_text("\n".join(contents), encoding="utf-8")


def test_plot_pareto_front_creates_image(tmp_path: Path) -> None:
    log_path = tmp_path / "log.csv"
    _write_log(
        log_path,
        "trial,metric_loss,metric_accuracy",
        [
            "0,0.6,0.2",
            "1,0.4,0.3",
            "2,0.3,0.25",
            "3,0.5,0.35",
        ],
    )

    objectives = [
        ObjectiveSpec("loss", "minimize"),
        ObjectiveSpec("accuracy", "maximize"),
    ]
    output = tmp_path / "pareto.png"
    result = plot_pareto_front(log_path, objectives, output_path=output)

    assert result == output
    assert output.exists()


def test_plot_history_tracks_best_values(tmp_path: Path) -> None:
    log_path = tmp_path / "log.csv"
    _write_log(
        log_path,
        "trial,metric_loss,metric_accuracy",
        [
            "0,0.6,0.2",
            "1,0.4,0.25",
            "2,0.5,0.3",
            "3,0.35,0.32",
        ],
    )
    objectives = [
        ObjectiveSpec("loss", "minimize"),
        ObjectiveSpec("accuracy", "maximize"),
    ]
    output = tmp_path / "history.png"
    result = plot_history(log_path, objectives, output_path=output)

    assert result == output
    assert output.exists()
