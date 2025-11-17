"""Generate LLM-backed diagnostic reports for failed optimization behaviour."""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .llm_guidance import create_llm_provider
from .llm_interfaces import (
    LLMHistoryMetric,
    LLMObjective,
    LLMRepresentativePoint,
    LLMRunContext,
)
from .llm_providers import Prompt, ToolDefinition


_SYSTEM_PROMPT = (
    "あなたはハイパーパラメータ探索の批評家です。ログの失敗シグナルを分析し、"
    "制約・前処理・ノイズモデルの改善案をMarkdownでまとめてください。"
    "レスポンスは『失敗ケース診断』と『改善提案』の2セクションのみで構成してください。"
)


@dataclass
class FailureSignals:
    """Structured summary of failure symptoms detected in trial history."""

    total_observed: int
    nan_count: int
    inf_count: int
    longest_no_improve: int
    trailing_no_improve: int
    improvements: int
    best_value: float | None
    last_value: float | None
    last_finite_value: float | None


@dataclass
class StructuredCritique:
    """Parsed LLM response describing diagnostic insights."""

    diagnosis: List[str]
    recommendations: List[str]
    bottlenecks: List[str] = field(default_factory=list)
    improvement_opportunities: List[str] = field(default_factory=list)
    next_experiments: List[str] = field(default_factory=list)


def generate_llm_critique(
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    metric_names: Sequence[str],
    direction_names: Sequence[str],
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, Any],
    trials_completed: int,
    early_stop_reason: str | None,
    log_path: Path,
    pareto_summary: Mapping[str, Any] | None = None,
) -> List[str] | None:
    """Return an optional Markdown section with LLM (or heuristic) critique."""

    critic_cfg = config.get("llm_critic")
    if not isinstance(critic_cfg, Mapping) or not critic_cfg.get("enabled"):
        return None

    llm_cfg = config.get("llm") if isinstance(config, Mapping) else None
    max_history = int(critic_cfg.get("max_history", 200))
    prompt_preamble = critic_cfg.get("prompt_preamble")

    metric_names = list(metric_names)
    if not metric_names:
        return None
    direction_map = _normalise_directions(metric_names, direction_names)
    primary_metric = metric_names[0]
    primary_direction = direction_map.get(primary_metric, "minimize")

    histories = _load_metric_histories(log_path, metric_names, limit=max_history)
    series = histories.get(primary_metric, [])
    signals = _summarise_failures(series, primary_direction)

    pareto_representatives = _convert_pareto_representatives(
        pareto_summary.get("representatives") if pareto_summary else None
    )
    pareto_records = (
        pareto_summary.get("records") if isinstance(pareto_summary, Mapping) else None
    )
    pareto_insights = _summarise_pareto_insights(
        pareto_records,
        metric_names=metric_names,
        direction_map=direction_map,
    )
    timeline_notes = _format_metric_timelines(histories)
    multi_objective = len(metric_names) > 1

    fallback = _format_fallback(
        signals,
        trials_completed,
        early_stop_reason,
        metric_names=metric_names,
        direction_map=direction_map,
        multi_objective=multi_objective,
        pareto_insights=pareto_insights,
        timeline_notes=timeline_notes,
    )

    provider, usage_logger = create_llm_provider(llm_cfg)
    if provider is None:
        return fallback

    schema = _critique_schema(multi_objective=multi_objective)
    prompt = _build_prompt(
        metadata=metadata,
        metric_names=metric_names,
        direction_map=direction_map,
        best_params=best_params,
        best_metrics=best_metrics,
        trials_completed=trials_completed,
        early_stop_reason=early_stop_reason,
        signals=signals,
        histories=histories,
        timeline_notes=timeline_notes,
        pareto_representatives=pareto_representatives,
        pareto_insights=pareto_insights,
        multi_objective=multi_objective,
        prompt_preamble=prompt_preamble,
        schema=schema,
    )

    try:
        result = provider.generate(
            prompt,
            temperature=0.2,
            system=_SYSTEM_PROMPT,
            json_mode=False,
            tool=ToolDefinition(
                name="render_diagnostic_report",
                description="Summarise failure diagnosis and actionable recommendations.",
                parameters=schema,
            ),
        )
    except Exception:
        return fallback

    if usage_logger is not None:
        usage_logger.log(result.usage)

    parsed = _parse_structured_critique(result.content, multi_objective=multi_objective)
    if parsed is None:
        return fallback

    return _format_structured_sections(parsed)


def _normalise_directions(
    metric_names: Sequence[str], direction_names: Sequence[str]
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    fallback = str(direction_names[0]).lower() if direction_names else "minimize"
    for idx, name in enumerate(metric_names):
        if idx < len(direction_names):
            direction = str(direction_names[idx]).lower()
        else:
            direction = fallback
        mapping[name] = direction if direction in {"minimize", "maximize"} else fallback
    return mapping


def _load_metric_histories(
    path: Path, metric_names: Sequence[str], *, limit: int
) -> Dict[str, List[float]]:
    histories: Dict[str, List[float]] = {name: [] for name in metric_names}
    if limit <= 0 or not path.exists() or not metric_names:
        return histories

    columns = {name: f"metric_{name}" for name in metric_names}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for name, column in columns.items():
                raw = row.get(column)
                if raw is None:
                    continue
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                histories[name].append(value)

    for name in metric_names:
        series = histories.get(name) or []
        if len(series) > limit:
            histories[name] = series[-limit:]
    return histories


def _summarise_failures(series: Iterable[float], direction: str) -> FailureSignals:
    values = list(series)
    nan_count = sum(1 for value in values if math.isnan(value))
    inf_count = sum(1 for value in values if math.isinf(value))
    best_value: float | None = None
    improvements = 0
    longest_no_improve = 0
    trailing_no_improve = 0
    current_no_improve = 0
    compare = _make_comparator(direction)

    for value in values:
        if not math.isfinite(value):
            current_no_improve += 1
            longest_no_improve = max(longest_no_improve, current_no_improve)
            continue

        if best_value is None or compare(value, best_value):
            best_value = value
            improvements += 1
            longest_no_improve = max(longest_no_improve, current_no_improve)
            current_no_improve = 0
        else:
            current_no_improve += 1
            longest_no_improve = max(longest_no_improve, current_no_improve)

    trailing_no_improve = current_no_improve
    last_value = values[-1] if values else None
    last_finite = next((v for v in reversed(values) if math.isfinite(v)), None)

    return FailureSignals(
        total_observed=len(values),
        nan_count=nan_count,
        inf_count=inf_count,
        longest_no_improve=longest_no_improve,
        trailing_no_improve=trailing_no_improve,
        improvements=improvements,
        best_value=best_value,
        last_value=last_value,
        last_finite_value=last_finite,
    )


def _convert_pareto_representatives(
    entries: Sequence[Mapping[str, Any]] | None,
) -> List[LLMRepresentativePoint]:
    if not entries:
        return []
    representatives: List[LLMRepresentativePoint] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        label = str(entry.get("label", "Pareto sample"))
        trial = entry.get("trial")
        values = entry.get("values")
        params = entry.get("params")
        if not isinstance(values, Mapping) or not isinstance(params, Mapping):
            continue
        metrics = entry.get("metrics")
        metrics_payload = dict(metrics) if isinstance(metrics, Mapping) else None
        representatives.append(
            LLMRepresentativePoint(
                label=label,
                trial=trial if isinstance(trial, int) else None,
                values=dict(values),
                params=dict(params),
                metrics=metrics_payload,
            )
        )
    return representatives


def _summarise_pareto_insights(
    records: Sequence[Mapping[str, Any]] | None,
    *,
    metric_names: Sequence[str],
    direction_map: Mapping[str, str],
) -> Dict[str, List[str]]:
    summary: List[str] = []
    tradeoffs: List[str] = []
    density: List[str] = []
    if not records or len(metric_names) < 2:
        return {"summary": summary, "tradeoffs": tradeoffs, "density": density}

    series_map: Dict[str, List[float]] = {name: [] for name in metric_names}
    for record in records:
        values = record.get("values")
        if not isinstance(values, Mapping):
            continue
        row: List[float] = []
        for name in metric_names:
            numeric = _safe_float(values.get(name))
            if numeric is None:
                row = []
                break
            row.append(numeric)
        if not row:
            continue
        for name, value in zip(metric_names, row, strict=False):
            series_map[name].append(value)

    primary_name = metric_names[0]
    primary_values = series_map.get(primary_name, [])
    if primary_values:
        summary.append(
            f"パレート点 {len(primary_values)} 件: {primary_name} は {_format_range(primary_values)} に分布"
        )

    if len(primary_values) >= 2:
        primary_dir = direction_map.get(primary_name, "minimize")
        primary_aligned = _align_direction(primary_values, primary_dir)
        for name in metric_names[1:]:
            other_values = series_map.get(name, [])
            if len(other_values) < 2:
                continue
            other_dir = direction_map.get(name, "minimize")
            note = _describe_tradeoff(
                name=name,
                primary_aligned=primary_aligned,
                other_aligned=_align_direction(other_values, other_dir),
                original_values=other_values,
                primary_direction=primary_dir,
                other_direction=other_dir,
            )
            if note:
                tradeoffs.append(note)
        density.extend(
            _density_notes(
                primary_values,
                primary_name=primary_name,
            )
        )

    return {"summary": summary, "tradeoffs": tradeoffs, "density": density}


def _format_metric_timelines(histories: Mapping[str, Sequence[float]]) -> Dict[str, str]:
    notes: Dict[str, str] = {}
    for name, series in histories.items():
        if not series:
            notes[name] = "(no data)"
            continue
        tail = series[-8:]
        notes[name] = ", ".join(f"{value:.4g}" for value in tail)
    return notes


def _make_comparator(direction: str):
    norm_direction = direction.lower()
    if norm_direction == "maximize":
        return lambda value, best: value > best + 1e-12
    return lambda value, best: value < best - 1e-12


def _format_fallback(
    signals: FailureSignals,
    trials_completed: int,
    early_stop_reason: str | None,
    *,
    metric_names: Sequence[str],
    direction_map: Mapping[str, str],
    multi_objective: bool,
    pareto_insights: Mapping[str, Sequence[str]],
    timeline_notes: Mapping[str, str],
) -> List[str]:
    issues: List[str] = []
    if signals.total_observed == 0:
        issues.append("試行ログが見つからず、自動診断は参考情報のみとなります。")
    else:
        if signals.nan_count:
            issues.append(
                f"主指標で NaN が {signals.nan_count} 件観測され、数値不安定が疑われます。"
            )
        if signals.inf_count:
            issues.append(
                f"主指標で Inf が {signals.inf_count} 件検出され、勾配爆発や探索制約逸脱が示唆されます。"
            )
        if signals.improvements <= 1:
            issues.append("改善回数がほとんどなく、収束が停滞しています。")
        if signals.longest_no_improve >= 3:
            issues.append(
                f"最大 {signals.longest_no_improve} 試行連続で改善がなく、探索が膠着しています。"
            )
        if not issues:
            issues.append("大きな異常は検出されませんが、最終ログの傾向を継続監視してください。")

    if early_stop_reason:
        issues.append(f"早期停止理由: {early_stop_reason}")

    if multi_objective:
        summary_notes = pareto_insights.get("summary") or []
        if summary_notes:
            issues.extend(summary_notes)

    suggestions = _build_suggestions(signals)

    bottlenecks: List[str] = []
    improvement: List[str] = []
    next_experiments: List[str] = []

    primary_metric = metric_names[0] if metric_names else "primary"
    primary_timeline = timeline_notes.get(primary_metric, "(no data)")

    if multi_objective:
        bottlenecks = list(pareto_insights.get("tradeoffs") or [])
        if not bottlenecks:
            bottlenecks = [
                f"{primary_metric} ({direction_map.get(primary_metric, 'minimize')}) が停滞し、他目的の改善余地を塞いでいます。"
            ]
        improvement = list(pareto_insights.get("density") or [])
        if not improvement:
            improvement = [
                f"{primary_metric} のタイムライン {primary_timeline} から、探索していない範囲にパレート点を追加する余地があります。"
            ]
        next_experiments.extend(
            [
                "サンプラー: 多目的最適化向け NSGA-II や MOTPE に切り替え、トレードオフ領域の被覆率を高める。",
                f"探索空間: {improvement[0]} を手掛かりに、疎な目的範囲へ境界や prior を広げる。",
                f"再評価: タイムライン {primary_timeline} を踏まえ、計算資源追加時は再サンプリングやバッチ評価を増やす。",
            ]
        )
    else:
        bottlenecks = [
            f"{primary_metric} ({direction_map.get(primary_metric, 'minimize')}) が停滞し、改善回数 {signals.improvements} に留まっています。"
        ]
        improvement = [
            f"最近の推移 {primary_timeline} を確認し、ノイズを下げれば追加改善の余地があります。"
        ]
        next_experiments.extend(
            [
                "サンプラー: 停滞が続く場合は提案分布を広げる (例: TPE→CMA-ES) ことで局所解を回避。",
                "探索空間: ベスト周辺でパラメータ範囲を微調整し、再評価回数を増やして安定性を確認。",
            ]
        )

    sections = StructuredCritique(
        diagnosis=issues,
        recommendations=suggestions,
        bottlenecks=bottlenecks,
        improvement_opportunities=improvement,
        next_experiments=next_experiments,
    )
    return _format_structured_sections(sections)


def _build_suggestions(signals: FailureSignals) -> List[str]:
    suggestions: List[str] = []

    if signals.nan_count or signals.inf_count:
        suggestions.append(
            "制約調整: 勾配クリッピングやパラメータ範囲の再設定で数値爆発を抑制してください。"
        )
        suggestions.append(
            "前処理: 入力特徴量の正規化と欠損値補完を徹底し、NaN 発生源を除去します。"
        )
    else:
        suggestions.append(
            "制約調整: ベスト近傍の探索範囲を動的に絞りつつ、探索上限を再検討してください。"
        )
        suggestions.append(
            "前処理: 特徴量の分散スケーリングや外れ値除去で停滞を解消します。"
        )

    if signals.trailing_no_improve >= 3:
        suggestions.append(
            "ノイズモデル: 観測ノイズ分散の推定を見直し、リサンプリングやアンサンブルで頑健性を高めます。"
        )
    else:
        suggestions.append(
            "ノイズモデル: 軽微な停滞でも感度が落ちないよう、ノイズ推定と評価リピート数を定期的に再評価してください。"
        )

    return suggestions


def _critique_schema(*, multi_objective: bool) -> Dict[str, Any]:
    properties: Dict[str, Any] = {
        "diagnosis": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    }
    required = ["diagnosis", "recommendations"]
    if multi_objective:
        for key in ("bottlenecks", "improvement_opportunities", "next_experiments"):
            properties[key] = {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            }
            required.append(key)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _parse_structured_critique(
    payload: str, *, multi_objective: bool
) -> StructuredCritique | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, Mapping):
        return None
    allowed = {"diagnosis", "recommendations"}
    if multi_objective:
        allowed.update({"bottlenecks", "improvement_opportunities", "next_experiments"})
    if any(key not in allowed for key in data.keys()):
        return None

    diagnosis = _normalise_string_list(data.get("diagnosis"))
    recommendations = _normalise_string_list(data.get("recommendations"))
    if not diagnosis or not recommendations:
        return None
    bottlenecks = _normalise_string_list(data.get("bottlenecks"))
    improvement = _normalise_string_list(data.get("improvement_opportunities"))
    next_experiments = _normalise_string_list(data.get("next_experiments"))
    if multi_objective:
        if not bottlenecks or not improvement or not next_experiments:
            return None
    return StructuredCritique(
        diagnosis=diagnosis,
        recommendations=recommendations,
        bottlenecks=bottlenecks,
        improvement_opportunities=improvement,
        next_experiments=next_experiments,
    )


def _normalise_string_list(value: Any) -> List[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return []
    results: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            results.append(text)
    return results


def _build_llm_context(
    *,
    metric_names: Sequence[str],
    direction_map: Mapping[str, str],
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, Any],
    trials_completed: int,
    early_stop_reason: str | None,
    signals: FailureSignals,
    histories: Mapping[str, Sequence[float]],
    pareto_representatives: Sequence[LLMRepresentativePoint],
    pareto_insights: Mapping[str, Sequence[str]],
) -> LLMRunContext:
    objectives = [
        LLMObjective(name=name, direction=direction_map.get(name))
        for name in metric_names
    ]
    current_best = [
        LLMRepresentativePoint(
            label="Best observed metrics",
            trial=None,
            values=dict(best_metrics),
            params=dict(best_params),
        )
    ]
    if pareto_representatives:
        current_best.extend(pareto_representatives)

    history_summary: List[LLMHistoryMetric] = []
    for name in metric_names:
        series = histories.get(name, [])
        stats = _series_stats(series)
        history_summary.append(
            LLMHistoryMetric(
                name=name,
                direction=direction_map.get(name),
                window=stats.get("window"),
                latest=series[-1] if series else None,
                minimum=stats.get("min"),
                maximum=stats.get("max"),
                mean=stats.get("mean"),
                count=stats.get("count"),
            )
        )

    history_notes: List[str] = []
    for key in ("summary", "tradeoffs", "density"):
        values = pareto_insights.get(key) if isinstance(pareto_insights, Mapping) else None
        if values:
            history_notes.extend(values)

    notes: List[str] = [
        f"objectives={len(metric_names)}",
        f"nan={signals.nan_count}",
        f"inf={signals.inf_count}",
        f"stagnation={signals.longest_no_improve}/{signals.trailing_no_improve}",
    ]
    if early_stop_reason:
        notes.append(f"early_stop={early_stop_reason}")
    notes.append(f"improvements={signals.improvements}")

    return LLMRunContext(
        objectives=objectives,
        current_best=current_best,
        history_summary=history_summary,
        history_notes=history_notes,
        trials_completed=trials_completed,
        notes="; ".join(notes),
    )


def _series_stats(series: Sequence[float]) -> Dict[str, float | int | None]:
    window = len(series)
    finite = [value for value in series if math.isfinite(value)]
    if not finite:
        return {"min": None, "max": None, "mean": None, "count": window or None, "window": window or None}
    total = sum(finite)
    count = len(finite)
    return {
        "min": min(finite),
        "max": max(finite),
        "mean": total / count if count else None,
        "count": count,
        "window": window or None,
    }


def _describe_tradeoff(
    *,
    name: str,
    primary_aligned: Sequence[float],
    other_aligned: Sequence[float],
    original_values: Sequence[float],
    primary_direction: str,
    other_direction: str,
) -> str:
    if len(original_values) < 2:
        return ""
    span = _format_range(original_values)
    if _is_nearly_constant(original_values):
        return f"{name} は {span} でほぼ一定"
    corr = _pearson(primary_aligned, other_aligned)
    primary_action = _improvement_action(primary_direction)
    other_action = _improvement_action(other_direction)
    other_decline = _decline_action(other_direction)
    if corr <= -0.25:
        return f"{name} は {span} で推移し、{primary_action}ほど {other_decline}傾向"
    if corr >= 0.25:
        return f"{name} は {span} で推移し、{primary_action}と同時に {other_action}傾向"
    return f"{name} は {span} で推移し、{primary_action}との相関は弱い"


def _density_notes(primary_values: Sequence[float], *, primary_name: str) -> List[str]:
    if len(primary_values) < 4 or _is_nearly_constant(primary_values):
        return []
    bins = min(4, len(primary_values))
    min_value = min(primary_values)
    max_value = max(primary_values)
    width = (max_value - min_value) / bins if bins else 0.0
    notes: List[str] = []
    for idx in range(bins):
        start = min_value + idx * width
        end = start + width if idx < bins - 1 else max_value
        count = _count_in_range(primary_values, start, end, inclusive=idx == bins - 1)
        threshold = max(1, len(primary_values) // (bins * 2))
        if count == 0:
            notes.append(
                f"{primary_name} {_format_interval(start, end)} にパレート点が存在しない"
            )
        elif count <= threshold:
            notes.append(
                f"{primary_name} {_format_interval(start, end)} には点が少なく探索余地が大きい"
            )
    return notes


def _format_range(values: Sequence[float]) -> str:
    if not values:
        return "(no data)"
    low = min(values)
    high = max(values)
    if _is_nearly_constant([low, high]):
        return _format_value(low)
    return f"{_format_value(low)}〜{_format_value(high)}"


def _format_interval(start: float, end: float) -> str:
    if math.isclose(start, end, rel_tol=1e-9, abs_tol=1e-12):
        return _format_value(start)
    return f"{_format_value(start)}〜{_format_value(end)}"


def _format_value(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4g}"


def _is_nearly_constant(values: Sequence[float]) -> bool:
    if not values:
        return True
    return math.isclose(min(values), max(values), rel_tol=1e-9, abs_tol=1e-12)


def _align_direction(values: Sequence[float], direction: str) -> List[float]:
    if direction == "maximize":
        return [-value for value in values]
    return list(values)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _count_in_range(
    values: Sequence[float], start: float, end: float, *, inclusive: bool
) -> int:
    if inclusive:
        return sum(1 for value in values if start <= value <= end)
    return sum(1 for value in values if start <= value < end)


def _improvement_action(direction: str) -> str:
    return "下げる" if direction == "minimize" else "上げる"


def _decline_action(direction: str) -> str:
    return "上がる" if direction == "minimize" else "下がる"


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _format_structured_sections(sections: StructuredCritique) -> List[str]:
    lines: List[str] = ["### 失敗ケース診断", ""]
    lines.extend(f"- {item}" for item in sections.diagnosis)
    if sections.bottlenecks:
        lines.extend(["", "#### ボトルネック候補", ""])
        lines.extend(f"- {item}" for item in sections.bottlenecks)
    if sections.improvement_opportunities:
        lines.extend(["", "#### 計算資源で伸ばせる目的", ""])
        lines.extend(f"- {item}" for item in sections.improvement_opportunities)
    lines.extend(["", "### 改善提案", ""])
    lines.extend(f"- {item}" for item in sections.recommendations)
    if sections.next_experiments:
        lines.extend(["", "#### 次回実験で調整したい点", ""])
        lines.extend(f"- {item}" for item in sections.next_experiments)
    return lines


def _build_prompt(
    *,
    metadata: Mapping[str, Any],
    metric_names: Sequence[str],
    direction_map: Mapping[str, str],
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, Any],
    trials_completed: int,
    early_stop_reason: str | None,
    signals: FailureSignals,
    histories: Mapping[str, Sequence[float]],
    timeline_notes: Mapping[str, str],
    pareto_representatives: Sequence[LLMRepresentativePoint],
    pareto_insights: Mapping[str, Sequence[str]],
    multi_objective: bool,
    prompt_preamble: str | None,
    schema: Mapping[str, Any],
) -> Prompt:
    name = metadata.get("name", "unknown")
    description = metadata.get("description", "")

    context = _build_llm_context(
        metric_names=metric_names,
        direction_map=direction_map,
        best_params=best_params,
        best_metrics=best_metrics,
        trials_completed=trials_completed,
        early_stop_reason=early_stop_reason,
        signals=signals,
        histories=histories,
        pareto_representatives=pareto_representatives,
        pareto_insights=pareto_insights,
    )
    context_json = context.to_json(indent=2)

    summary_lines = [
        f"試行数: {trials_completed}",
        f"観測済みログ: {signals.total_observed}",
        f"NaN 件数: {signals.nan_count}",
        f"Inf 件数: {signals.inf_count}",
        f"改善回数: {signals.improvements}",
        f"最大停滞長: {signals.longest_no_improve}",
        f"末尾停滞長: {signals.trailing_no_improve}",
        f"最新値: {signals.last_value}",
        f"最新有限値: {signals.last_finite_value}",
        f"ベスト値: {signals.best_value}",
    ]

    if early_stop_reason:
        summary_lines.append(f"早期停止理由: {early_stop_reason}")

    best_metric_lines = [f"{key}: {value}" for key, value in best_metrics.items()]
    best_param_lines = [f"{key}: {value}" for key, value in best_params.items()]

    timeline_lines = [
        f"{name} ({direction_map.get(name, 'minimize')}): {timeline_notes.get(name, '(no data)')}"
        for name in metric_names
    ]

    pareto_summary_lines: List[str] = []
    if multi_objective:
        for key in ("summary", "tradeoffs", "density"):
            values = pareto_insights.get(key)
            if values:
                pareto_summary_lines.append(f"[{key}]")
                pareto_summary_lines.extend(f"- {text}" for text in values)

    representatives_payload = (
        [entry.to_payload() for entry in pareto_representatives]
        if pareto_representatives
        else []
    )

    preamble = f"{prompt_preamble}\n" if prompt_preamble else ""

    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

    objective_summary = ", ".join(
        f"{name} ({direction_map.get(name, 'minimize')})" for name in metric_names
    )

    text = (
        f"{preamble}実験名: {name}\n"
        f"説明: {description}\n"
        f"目的一覧: {objective_summary}\n"
        "最適化状態JSON:\n"
        + context_json
        + "\n---\n"
        "失敗シグナル概要:\n"
        + "\n".join(f"- {line}" for line in summary_lines)
        + "\n---\n"
        "目的別タイムライン:\n"
        + "\n".join(f"- {line}" for line in timeline_lines)
        + "\n---\n"
        "ベスト指標:\n"
        + "\n".join(f"- {line}" for line in best_metric_lines)
        + "\nベストパラメータ:\n"
        + "\n".join(f"- {line}" for line in best_param_lines)
    )

    if multi_objective:
        text += "\n---\n多目的インサイト:\n"
        if pareto_summary_lines:
            text += "\n".join(pareto_summary_lines)
        else:
            text += "(Pareto summary unavailable)"
        if representatives_payload:
            text += "\nPareto代表JSON:\n"
            text += json.dumps(representatives_payload, ensure_ascii=False, indent=2)

    requirements = [
        "- 失敗ケース診断ではNaN頻発や勾配爆発、収束停滞などの要因を特定すること。",
        "- 改善提案では制約・前処理・ノイズモデルの観点で具体的なアクションを列挙すること。",
        "- 応答は render_diagnostic_report 関数の引数として Schema に従う JSON オブジェクトのみを返すこと。",
    ]
    if multi_objective:
        requirements.extend(
            [
                "- 多目的レポートではボトルネックになっている目的を特定し、理由をパレート情報やタイムラインから説明すること。",
                "- 計算資源を増やした場合に伸ばせる余地がある目的を列挙し、探索すべき領域やトレードオフを示すこと。",
                "- 次回の実験で調整すべき探索空間・サンプラー・再評価戦略などを必ず提案すること。",
            ]
        )

    text += (
        "\n---\n"
        "要求:\n"
        + "\n".join(requirements)
        + "\n"
        "Schema:\n"
        + schema_text
    )

    return Prompt.from_text(text)


__all__ = ["generate_llm_critique"]

