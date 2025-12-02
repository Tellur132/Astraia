# Quantum evaluator の詳細ガイド

量子 evaluator は LLM が提案した回路や手書きの OpenQASM を安全に評価するための仕組みです。`CircuitFidelityEvaluator` や QFT/QAOA 用のカスタム evaluator は共通して `BaseEvaluator` を継承し、失敗時は構造化されたペイロードで探索を継続できるようになっています。

## CircuitFidelityEvaluator の使い方

`CircuitFidelityEvaluator` は以下の 2 つのモードをサポートします。

- **ターゲットユニタリ**: `target_unitary`（正方行列）を与えると、候補回路とのプロセス忠実度を計算します。
- **真理値表**: `truth_table` で入力ビット列→出力ビット列を列挙すると、指定した射影確率の平均を忠実度として返します。

設定例（OpenQASM で生成した 2 量子ビットの SWAP を再現）:

```yaml
evaluator:
  module: astraia.evaluators.circuit_fidelity
  callable: create_circuit_fidelity_evaluator
  truth_table:
    "00": "00"
    "01": "10"
    "10": "01"
    "11": "11"
```

`metric_fidelity` / `metric_depth` / `metric_gate_count` に加え、成功時は `metric_valid=1.0`、失敗時は `metric_valid=0.0` として返却します。パース・評価に失敗した場合でも `status="error"` と `reason` が埋め込まれ、Optuna の探索は継続します。

## QFT/MaxCut のサンプル evaluator

- **QFT 合成** (`configs/quantum/qft_fidelity_depth.yaml`): QFT 近似の忠実度と回路深さを多目的最適化するテンプレートです。`noise_model` を与えると `metric_error_probability` も計算されます。
- **QAOA (MaxCut)** (`configs/quantum/qaoa_small.yaml`): 小規模グラフに対する QAOA のエネルギー（`metric_energy`）と忠実度（`metric_fidelity`）を返します。コストハミルトニアンは `H_C = Σ(0.5 Z_i Z_j - 0.5 I)` で、値が小さいほど（より負になるほど）良いカットです。同じ `H_C` から厳密エネルギー `metric_energy_exact` とギャップ `metric_energy_gap`、最適ビット列の成功確率 `metric_success_prob_opt`（ノイズありは `_noisy` / `_delta` も）を返します。`n_layers` をパラメタに含めることでレイヤー数を探索できます。

いずれも `pip install -e .[quantum]` で依存関係を満たした後、`astraia --config <file>` で即座に実行できます。`--summarize` / `--as-json` は設定検証のみを行うため、LLM なしで動作確認したい場合に便利です。

## NISQ ノイズシミュレーションとの比較

`noise_simulation` ブロックを設定すると、Aer の密度行列シミュレータを使って「ノイズあり」状態を併せて計算し、理想状態とのギャップを記録できます。

ノイズシミュレーションには `qiskit-aer` が必要です。`pip install -e .[quantum]` を使うと Qiskit 本体と合わせて `qiskit-aer` も導入されるため、追加インストールなしで利用できます。

```yaml
evaluator:
  module: astraia.evaluators.qaoa
  callable: create_qaoa_evaluator
  noise_simulation:
    enabled: true        # false でノイズ計算をスキップ
    label: nisq_mock     # ログ用の任意ラベル
    single_qubit_depolarizing: 0.001
    two_qubit_depolarizing: 0.01
    readout_error: 0.02
```

QFT テンプレートでは `metric_fidelity_noisy` / `metric_fidelity_delta`、QAOA テンプレートではさらに `metric_energy_noisy` / `metric_energy_delta` が得られます。`report.metrics` にこれらを追加すると `log.csv` やレポートで理想・ノイズありを並べて確認できます。

## レポートと可視化

量子 evaluator の実行結果は `runs/<run_id>/` に保存されます。

- `log.csv`: 忠実度や深さなどのメトリクスが試行ごとに記録されます。
- `report.md`: ベストトライアルの回路（メトリクス付き）と Pareto front の概要、LLM クリティックの診断を含みます。
- `log_pareto.png` / `log_history.png`: `astraia visualize` コマンドで自動生成される PNG。忠実度と深さのトレードオフや収束速度を確認できます。

`runs compare --metric fidelity --stat best` で複数実行の忠実度を並べたり、`visualize --type pareto --metric fidelity --metric depth` で Pareto front を描画することで、量子回路の設計・改善を定量的に追跡できます。
