"""LLM が真似しやすい evaluator テンプレート。"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .base import EvaluatorResult


@dataclass(slots=True)
class SimpleWaveEvaluator:
    """サイン波ベースのおもちゃ目的関数。

    - `params` は YAML の `search_space` に合わせた dict を受け取ります。
    - `seed` は乱数の再現に使える任意の整数です（Optuna から自動で渡されます）。
    - 戻り値は `score` を含む dict。レポートに載せたい値をキーに追加します。
    """

    response_noise: float = 0.0

    def __call__(self, params: Mapping[str, Any], seed: int | None = None) -> EvaluatorResult:
        start = time.perf_counter()

        # 1. YAML で宣言したパラメタを取り出し、数値化しておく。
        amplitude = float(params.get("amplitude", 1.0))
        frequency = float(params.get("frequency", 1.0))
        phase = float(params.get("phase", 0.0))
        depth = int(params.get("depth", 1))

        # 2. 目的関数を決める。ここでは「サイン波の揺らぎ + 深さのペナルティ」という簡単な式。
        oscillation = math.sin(frequency * amplitude + phase)
        smoothness_penalty = 0.02 * depth
        deterministic_score = (oscillation**2) + smoothness_penalty + abs(amplitude - 1.5) * 0.1

        # 3. 再現可能なノイズを足す。seed がなければ depth を混ぜておくと変化が出る。
        rng = random.Random(seed if seed is not None else depth * 7919)
        noisy_score = deterministic_score + rng.gauss(0.0, self.response_noise)

        elapsed = time.perf_counter() - start

        # 4. dict を返す。score 以外のキーはレポート表示やデバッグ用に自由追加。
        return {
            "score": float(noisy_score),
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": phase,
            "depth": float(depth),
            "status": "ok",
            "elapsed_seconds": elapsed,
        }


def create_evaluator(config: Mapping[str, Any]) -> SimpleWaveEvaluator:
    """YAML の evaluator セクションから渡された値でテンプレートを組み立てる。"""

    response_noise = float(config.get("response_noise", 0.0))
    return SimpleWaveEvaluator(response_noise=response_noise)


__all__ = ["SimpleWaveEvaluator", "create_evaluator"]
