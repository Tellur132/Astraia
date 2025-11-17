"""Utility functions for generating static plots from optimization logs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

# Force a non-interactive backend to support headless environments (tests/CI).
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ObjectiveSpec:
    """Description of a single optimization objective."""

    name: str
    direction: str = "minimize"

    @property
    def column(self) -> str:
        return f"metric_{self.name}"

    @property
    def label(self) -> str:
        direction = "min" if self.direction == "minimize" else "max"
        return f"{self.name} ({direction})"


class VisualizationError(RuntimeError):
    """Raised when a visualization cannot be generated."""


def plot_pareto_front(
    log_path: Path | str,
    objectives: Sequence[ObjectiveSpec],
    *,
    title: str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Render a scatter plot of the Pareto front for two objectives."""

    if len(objectives) != 2:
        raise VisualizationError("Pareto front plots require exactly two objectives.")

    log_path = Path(log_path)
    df = _read_log(log_path)

    columns = [objective.column for objective in objectives]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise VisualizationError(
            f"Log file {log_path} does not contain required metric columns: {', '.join(missing)}"
        )

    subset = df[columns].dropna()
    if subset.empty:
        raise VisualizationError("Log file does not contain any valid rows for the requested metrics.")

    values = subset.to_numpy(dtype=float)
    mask = _pareto_mask(values, [obj.direction for obj in objectives])
    pareto_points = values[mask]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(values[:, 0], values[:, 1], alpha=0.5, label="Trials", color="#5DA5DA")
    if len(pareto_points):
        ax.scatter(
            pareto_points[:, 0],
            pareto_points[:, 1],
            color="#F15854",
            label="Pareto front",
        )
    ax.set_xlabel(objectives[0].label)
    ax.set_ylabel(objectives[1].label)
    ax.set_title(title or "Pareto front")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    output = _resolve_output_path(log_path, output_path, suffix="pareto")
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_history(
    log_path: Path | str,
    objectives: Sequence[ObjectiveSpec],
    *,
    title: str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Plot the running best value for each objective over time."""

    if not objectives:
        raise VisualizationError("At least one objective must be provided to plot history.")

    log_path = Path(log_path)
    df = _read_log(log_path)
    if "trial" in df.columns:
        df = df.sort_values("trial")
    else:
        df = df.reset_index(drop=False).rename(columns={"index": "trial"})

    fig, ax = plt.subplots(figsize=(6, 4))
    x_values = df["trial"].to_list()

    for objective in objectives:
        if objective.column not in df.columns:
            raise VisualizationError(
                f"Log file {log_path} does not contain required metric column: {objective.column}"
            )
        series = pd.to_numeric(df[objective.column], errors="coerce")
        running_best = _running_best(series, objective.direction)
        ax.plot(x_values, running_best, label=objective.label)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best value")
    ax.set_title(title or "Best value history")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    output = _resolve_output_path(log_path, output_path, suffix="history")
    fig.savefig(output)
    plt.close(fig)
    return output


def _read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise VisualizationError(f"Log file not found: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:  # type: ignore[attr-defined]
        raise VisualizationError(f"Log file is empty: {path}") from exc
    return df


def _pareto_mask(values: Iterable[Sequence[float]], directions: Sequence[str]) -> list[bool]:
    data = list(values)
    mask = [True] * len(data)
    for idx, point in enumerate(data):
        for jdx, other in enumerate(data):
            if idx == jdx:
                continue
            if _dominates(other, point, directions):
                mask[idx] = False
                break
    return mask


def _dominates(candidate: Sequence[float], target: Sequence[float], directions: Sequence[str]) -> bool:
    better_or_equal = True
    strictly_better = False
    for value, other, direction in zip(candidate, target, directions, strict=False):
        if direction == "maximize":
            if value < other:
                better_or_equal = False
                break
            if value > other:
                strictly_better = True
        else:  # minimize
            if value > other:
                better_or_equal = False
                break
            if value < other:
                strictly_better = True
    return better_or_equal and strictly_better


def _running_best(series: pd.Series, direction: str) -> list[float]:
    values: list[float] = []
    best: float | None = None
    maximize = direction == "maximize"
    for value in series:
        if pd.isna(value):
            values.append(float("nan") if best is None else best)
            continue
        if best is None:
            best = float(value)
        else:
            best = max(best, float(value)) if maximize else min(best, float(value))
        values.append(best)
    return values


def _resolve_output_path(log_path: Path, output_path: Path | str | None, *, suffix: str) -> Path:
    if output_path is not None:
        return Path(output_path)
    directory = log_path.parent
    stem = log_path.stem
    return directory / f"{stem}_{suffix}.png"


__all__ = ["ObjectiveSpec", "VisualizationError", "plot_history", "plot_pareto_front"]
