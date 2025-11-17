"""Summarise Pareto fronts for multi-objective guidance."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import optuna

from .llm_interfaces import LLMRepresentativePoint


@dataclass
class _ParetoPoint:
    trial: int
    values: Tuple[float, ...]
    metrics: Dict[str, float]
    params: Mapping[str, Any]


class ParetoSummaryGenerator:
    """Generate representative Pareto points and textual trade-off summaries."""

    def __init__(
        self,
        metric_names: Sequence[str],
        directions: Sequence[optuna.study.StudyDirection],
        *,
        max_representatives: int = 5,
    ) -> None:
        if len(metric_names) != len(directions):
            raise ValueError("metric_names and directions must have the same length")
        if not metric_names:
            raise ValueError("metric_names must not be empty")
        self._metric_names = list(metric_names)
        self._metric_keys = [name.lower() for name in metric_names]
        self._directions = list(directions)
        self._max_representatives = max(2, max_representatives)
        self._primary_direction = directions[0]

    def summarise(
        self, history: Sequence["TrialRecord"]
    ) -> Tuple[List[LLMRepresentativePoint], List[str]]:
        points = self._extract_points(history)
        if len(points) < 2:
            return [], []
        front = self._pareto_front(points)
        representatives = self._select_representatives(front)
        notes = self._describe_front(front)
        return representatives, notes

    # ------------------------------------------------------------------
    # Point extraction
    # ------------------------------------------------------------------
    def _extract_points(self, history: Sequence["TrialRecord"]) -> List[_ParetoPoint]:
        points: List[_ParetoPoint] = []
        for record in history:
            values: List[float] = []
            metrics: Dict[str, float] = {}
            normalised = self._normalise_metrics(record.metrics)
            missing = False
            for name, key in zip(self._metric_names, self._metric_keys):
                value = normalised.get(key)
                if value is None or not math.isfinite(value):
                    missing = True
                    break
                values.append(value)
                metrics[name] = value
            if missing:
                continue
            points.append(
                _ParetoPoint(
                    trial=record.number,
                    values=tuple(values),
                    metrics=metrics,
                    params=record.params,
                )
            )
        return points

    def _normalise_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, float]:
        normalised: Dict[str, float] = {}
        for name, value in metrics.items():
            if not isinstance(name, str):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            normalised[name.lower()] = numeric
        return normalised

    # ------------------------------------------------------------------
    # Pareto calculations
    # ------------------------------------------------------------------
    def _pareto_front(self, points: Sequence[_ParetoPoint]) -> List[_ParetoPoint]:
        front: List[_ParetoPoint] = []
        for point in points:
            dominated = False
            dominated_indices: List[int] = []
            for idx, existing in enumerate(front):
                if self._dominates(existing.values, point.values):
                    dominated = True
                    break
                if self._dominates(point.values, existing.values):
                    dominated_indices.append(idx)
            if dominated:
                continue
            for idx in reversed(dominated_indices):
                front.pop(idx)
            front.append(point)
        sort_multiplier = 1.0
        if self._primary_direction == optuna.study.StudyDirection.MAXIMIZE:
            sort_multiplier = -1.0
        front.sort(key=lambda p: sort_multiplier * p.values[0])
        return front

    def _dominates(self, candidate: Sequence[float], other: Sequence[float]) -> bool:
        better = False
        for idx, (cand_value, other_value) in enumerate(zip(candidate, other)):
            direction = self._directions[idx]
            if direction == optuna.study.StudyDirection.MINIMIZE:
                if cand_value > other_value:
                    return False
                if cand_value < other_value:
                    better = True
            else:
                if cand_value < other_value:
                    return False
                if cand_value > other_value:
                    better = True
        return better

    # ------------------------------------------------------------------
    # Representation and description
    # ------------------------------------------------------------------
    def _select_representatives(
        self, front: Sequence[_ParetoPoint]
    ) -> List[LLMRepresentativePoint]:
        if not front:
            return []
        if len(front) <= self._max_representatives:
            selection = list(front)
        else:
            selection = []
            step = (len(front) - 1) / (self._max_representatives - 1)
            picked = set()
            for idx in range(self._max_representatives):
                position = min(len(front) - 1, round(idx * step))
                if position not in picked:
                    picked.add(position)
                    selection.append(front[position])
        representatives: List[LLMRepresentativePoint] = []
        total = len(selection)
        for idx, point in enumerate(selection, start=1):
            label = self._build_label(idx, total)
            values = {name: point.metrics[name] for name in self._metric_names}
            representatives.append(
                LLMRepresentativePoint(
                    label=label,
                    trial=point.trial,
                    values=values,
                    params=dict(point.params),
                    metrics=dict(point.metrics),
                )
            )
        return representatives

    def _build_label(self, position: int, total: int) -> str:
        if position == 1:
            if self._primary_direction == optuna.study.StudyDirection.MINIMIZE:
                return "Pareto edge (primary-low)"
            return "Pareto edge (primary-high)"
        if position == total:
            if self._primary_direction == optuna.study.StudyDirection.MINIMIZE:
                return "Pareto edge (primary-high)"
            return "Pareto edge (primary-low)"
        return f"Pareto sample {position}/{total}"

    def _describe_front(self, front: Sequence[_ParetoPoint]) -> List[str]:
        notes: List[str] = []
        primary_name = self._metric_names[0]
        primary_values = [point.metrics[primary_name] for point in front]
        primary_range = self._format_range(primary_values)
        notes.append(
            f"パレート点 {len(front)} 件: {primary_name} は {primary_range} の範囲に広がる"
        )
        notes.extend(self._tradeoff_notes(front, primary_values))
        notes.extend(self._density_notes(primary_values))
        return notes

    def _tradeoff_notes(
        self, front: Sequence[_ParetoPoint], primary_values: Sequence[float]
    ) -> List[str]:
        notes: List[str] = []
        if len(front) < 2:
            return notes
        primary_improve = self._align_direction(primary_values, self._directions[0])
        for idx in range(1, len(self._metric_names)):
            other_name = self._metric_names[idx]
            other_values = [point.metrics[other_name] for point in front]
            note = self._describe_tradeoff(
                other_name,
                primary_improve,
                self._align_direction(other_values, self._directions[idx]),
                other_values,
                self._directions[idx],
            )
            if note:
                notes.append(note)
        return notes

    def _describe_tradeoff(
        self,
        name: str,
        primary_aligned: Sequence[float],
        other_aligned: Sequence[float],
        original_values: Sequence[float],
        direction: optuna.study.StudyDirection,
    ) -> str:
        if len(original_values) < 2:
            return ""
        span = self._format_range(original_values)
        if self._is_nearly_constant(original_values):
            return f"{name} は {span} でほぼ一定"
        corr = self._pearson(primary_aligned, other_aligned)
        primary_action = self._improvement_action(self._directions[0])
        other_action = self._improvement_action(direction)
        other_decline = self._decline_action(direction)
        if corr <= -0.25:
            return f"{name} は {span} で推移し、{primary_action}ほど {other_decline}傾向"
        if corr >= 0.25:
            return f"{name} は {span} で推移し、{primary_action}と同時に {other_action}傾向"
        return f"{name} は {span} で推移し、{primary_action}との相関は弱い"

    def _density_notes(self, primary_values: Sequence[float]) -> List[str]:
        if len(primary_values) < 4:
            return []
        if self._is_nearly_constant(primary_values):
            return []
        bins = min(4, len(primary_values))
        min_value = min(primary_values)
        max_value = max(primary_values)
        width = (max_value - min_value) / bins
        notes: List[str] = []
        for idx in range(bins):
            start = min_value + idx * width
            end = start + width if idx < bins - 1 else max_value
            count = self._count_in_range(primary_values, start, end, inclusive=idx == bins - 1)
            threshold = max(1, len(primary_values) // (bins * 2))
            if count == 0:
                notes.append(
                    f"{self._metric_names[0]} {self._format_interval(start, end)} にパレート点が存在しない"
                )
            elif count <= threshold:
                notes.append(
                    f"{self._metric_names[0]} {self._format_interval(start, end)} には点が少なく探索余地が大きい"
                )
        return notes

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _format_range(self, values: Sequence[float]) -> str:
        if not values:
            return "(no data)"
        low = min(values)
        high = max(values)
        if self._is_nearly_constant([low, high]):
            return self._format_value(low)
        return f"{self._format_value(low)}〜{self._format_value(high)}"

    def _format_interval(self, start: float, end: float) -> str:
        if math.isclose(start, end, rel_tol=1e-9, abs_tol=1e-12):
            return self._format_value(start)
        return f"{self._format_value(start)}〜{self._format_value(end)}"

    def _format_value(self, value: float) -> str:
        if not math.isfinite(value):
            return "nan"
        return f"{value:.4g}"

    def _is_nearly_constant(self, values: Sequence[float]) -> bool:
        if not values:
            return True
        return math.isclose(min(values), max(values), rel_tol=1e-9, abs_tol=1e-12)

    def _align_direction(
        self, values: Sequence[float], direction: optuna.study.StudyDirection
    ) -> List[float]:
        if direction == optuna.study.StudyDirection.MINIMIZE:
            return list(values)
        return [-value for value in values]

    def _pearson(self, xs: Sequence[float], ys: Sequence[float]) -> float:
        n = len(xs)
        if n != len(ys) or n < 2:
            return 0.0
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x <= 1e-12 or var_y <= 1e-12:
            return 0.0
        return cov / math.sqrt(var_x * var_y)

    def _count_in_range(
        self,
        values: Sequence[float],
        start: float,
        end: float,
        *,
        inclusive: bool,
    ) -> int:
        if inclusive:
            return sum(1 for value in values if start <= value <= end)
        return sum(1 for value in values if start <= value < end)

    def _improvement_action(self, direction: optuna.study.StudyDirection) -> str:
        return "下げる" if direction == optuna.study.StudyDirection.MINIMIZE else "上げる"

    def _decline_action(self, direction: optuna.study.StudyDirection) -> str:
        return "上がる" if direction == optuna.study.StudyDirection.MINIMIZE else "下がる"


__all__ = ["ParetoSummaryGenerator"]
