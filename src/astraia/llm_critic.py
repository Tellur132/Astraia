"""Generate LLM-backed diagnostic reports for failed optimization behaviour."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .llm_guidance import create_llm_provider
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


def generate_llm_critique(
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    primary_metric: str,
    direction: str,
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, Any],
    trials_completed: int,
    early_stop_reason: str | None,
    log_path: Path,
) -> List[str] | None:
    """Return an optional Markdown section with LLM (or heuristic) critique."""

    critic_cfg = config.get("llm_critic")
    if not isinstance(critic_cfg, Mapping) or not critic_cfg.get("enabled"):
        return None

    llm_cfg = config.get("llm") if isinstance(config, Mapping) else None
    max_history = int(critic_cfg.get("max_history", 200))
    prompt_preamble = critic_cfg.get("prompt_preamble")

    series = _load_metric_history(log_path, primary_metric, limit=max_history)
    signals = _summarise_failures(series, direction)

    fallback = _format_fallback(signals, trials_completed, early_stop_reason)

    provider, usage_logger = create_llm_provider(llm_cfg)
    if provider is None:
        return fallback

    schema = _critique_schema()
    prompt = _build_prompt(
        metadata=metadata,
        primary_metric=primary_metric,
        direction=direction,
        best_params=best_params,
        best_metrics=best_metrics,
        trials_completed=trials_completed,
        early_stop_reason=early_stop_reason,
        signals=signals,
        series=series,
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

    parsed = _parse_structured_critique(result.content)
    if parsed is None:
        return fallback

    diagnosis, recommendations = parsed
    return _format_structured_sections(diagnosis, recommendations)


def _load_metric_history(path: Path, metric_name: str, *, limit: int) -> List[float]:
    if limit <= 0:
        return []

    if not path.exists():
        return []

    column = f"metric_{metric_name}"
    samples: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
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
            samples.append(value)

    if len(samples) > limit:
        samples = samples[-limit:]

    return samples


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


def _make_comparator(direction: str):
    norm_direction = direction.lower()
    if norm_direction == "maximize":
        return lambda value, best: value > best + 1e-12
    return lambda value, best: value < best - 1e-12


def _format_fallback(
    signals: FailureSignals,
    trials_completed: int,
    early_stop_reason: str | None,
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

    suggestions = _build_suggestions(signals)

    lines: List[str] = ["### 失敗ケース診断", ""]
    lines.extend([f"- {item}" for item in issues])
    lines.extend(["", "### 改善提案", ""])
    lines.extend([f"- {item}" for item in suggestions])
    return lines


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


def _critique_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
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
        },
        "required": ["diagnosis", "recommendations"],
    }


def _parse_structured_critique(payload: str) -> tuple[List[str], List[str]] | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, Mapping):
        return None
    allowed = {"diagnosis", "recommendations"}
    if any(key not in allowed for key in data.keys()):
        return None

    diagnosis = _normalise_string_list(data.get("diagnosis"))
    recommendations = _normalise_string_list(data.get("recommendations"))
    if not diagnosis or not recommendations:
        return None
    return diagnosis, recommendations


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


def _format_structured_sections(
    diagnosis: List[str], recommendations: List[str]
) -> List[str]:
    lines: List[str] = ["### 失敗ケース診断", ""]
    lines.extend(f"- {item}" for item in diagnosis)
    lines.extend(["", "### 改善提案", ""])
    lines.extend(f"- {item}" for item in recommendations)
    return lines


def _build_prompt(
    *,
    metadata: Mapping[str, Any],
    primary_metric: str,
    direction: str,
    best_params: Mapping[str, Any],
    best_metrics: Mapping[str, Any],
    trials_completed: int,
    early_stop_reason: str | None,
    signals: FailureSignals,
    series: List[float],
    prompt_preamble: str | None,
    schema: Mapping[str, Any],
) -> Prompt:
    name = metadata.get("name", "unknown")
    description = metadata.get("description", "")
    recent_values = ", ".join(f"{value:.4g}" for value in series[-8:]) if series else "(no data)"

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
        f"最近の主指標履歴: {recent_values}",
    ]

    if early_stop_reason:
        summary_lines.append(f"早期停止理由: {early_stop_reason}")

    best_metric_lines = [f"{key}: {value}" for key, value in best_metrics.items()]
    best_param_lines = [f"{key}: {value}" for key, value in best_params.items()]

    preamble = f"{prompt_preamble}\n" if prompt_preamble else ""

    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)

    text = (
        f"{preamble}実験名: {name}\n"
        f"説明: {description}\n"
        f"主指標: {primary_metric} ({direction})\n"
        "---\n"
        "失敗シグナル概要:\n"
        + "\n".join(f"- {line}" for line in summary_lines)
        + "\n---\n"
        "ベスト指標:\n"
        + "\n".join(f"- {line}" for line in best_metric_lines)
        + "\nベストパラメータ:\n"
        + "\n".join(f"- {line}" for line in best_param_lines)
        + "\n---\n"
        "要求:\n"
        "- 失敗ケース診断ではNaN頻発や勾配爆発、収束停滞などの要因を特定すること。\n"
        "- 改善提案では制約・前処理・ノイズモデルの観点で具体的なアクションを列挙すること。\n"
        "- 応答は render_diagnostic_report 関数の引数として Schema に従う JSON オブジェクトのみを返すこと。\n"
        "Schema:\n"
        + schema_text
    )

    return Prompt.from_text(text)


__all__ = ["generate_llm_critique"]

