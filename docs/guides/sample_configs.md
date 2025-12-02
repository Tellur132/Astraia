# サンプル設定とベンチマーク

このページではリポジトリに同梱された主な YAML 設定と用途を一覧します。どれも `astraia --config <file>` で即実行でき、`--summarize` / `--as-json` で事前確認できます。

## qGAN KL（単目的）
- `configs/qgan_kl.yaml`: MVP 用の最小構成サンプル。LLM ガイダンスやメタ探索を有効にして挙動を確認できます。

## 多目的ベンチマーク
- `configs/multiobj/qgan_kl_depth.yaml`: KL 最小化と回路深さ（`depth`）のトレードオフを最適化。NSGA-II サンプラーで Pareto front を生成。
- `configs/multiobj/zdt3.yaml`: 連続関数ベンチマーク ZDT3 を evaluator 化した純粋な 2 目的探索。LLM なしで動作し、散布図を `visualize --type pareto` で確認可能。
- `configs/multiobj/zdt3_llm.yaml`: ZDT3 をベースに LLM ガイダンス設定を加えたバリエーション。LLM あり/なし比較に使用。

## 量子回路関連
- `configs/quantum/qft_fidelity_depth.yaml`: QFT 回路の忠実度と深さを同時に最適化するテンプレート。`noise_simulation` を有効にして NISQ ノイズを比較できます。
- `configs/quantum/qaoa_small.yaml`: 小規模 MaxCut の QAOA チューニング例。`metric_energy` を最小化し、ノイズあり/なしの差分を記録。
- `configs/quantum/qaoa_llm_guided.yaml`: LLM ガイダンス付きの QAOA 実行例。`llm_guidance` の挙動を量子設定で試す際に利用。
- 量子 evaluator の詳細は [Quantum evaluator の詳細ガイド](quantum_evaluator.md) を参照。

## LLM evaluator / 設定テンプレート
- `configs/examples/llm_template.yaml`: コメント付きの設定テンプレート。`llm_guidance` / `meta_search` / `llm_critic` の各ブロックをまとめて試せます。
- `src/astraia/evaluators/llm_template.py`: `SimpleWaveEvaluator` のファクトリ例。LLM が読みやすいように段階をコメントで区切り済み。
- 解説は [LLM 向け Evaluator / Config サンプル](llm_evaluator_config_examples.md) を参照。
