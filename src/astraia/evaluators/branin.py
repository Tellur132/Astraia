# src/astraia/evaluators/branin.py
# ここを変更: パスはプロジェクト構成に合わせて

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
import math
import time


def branin_objective(
    params: Mapping[str, float],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """2次元 Branin 関数の評価関数。

    Astraia 側からは evaluator(params, seed) という形で呼ばれる前提。
    """
    # --- パラメータ取得 ---
    # ここを変更: パラメータ名は YAML の search_space と揃える
    x1 = float(params["x1"])
    x2 = float(params["x2"])

    start = time.perf_counter()

    # --- Branin の定数（標準形）---
    # f(x1, x2) = a (x2 - b x1^2 + c x1 - r)^2 + s(1 − t) cos(x1) + s
    a = 1.0
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)

    # --- Branin 関数値 ---
    y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + \
        s * (1.0 - t) * math.cos(x1) + s

    elapsed = time.perf_counter() - start

    # Astraia は dict でメトリクスを受け取る想定（qgan_kl と同じフォーム）
    return {
        "f": float(y),          # ← search.metric で使う主指標
        "x1": x1,
        "x2": x2,
        "status": "ok",
        "elapsed_seconds": elapsed,
    }
