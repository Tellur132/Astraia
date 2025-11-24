# LLM ガイド付き QAOA 回路設計 実行手順

## 1. 前提
- Python 3.11 以上と仮想環境（推奨）。
- 依存関係: 量子 + LLM の extra をまとめて入れます。

```bash
pip install -e .[llm,quantum]
```

- LLM API キーを `.env` に設定（例: `OPENAI_API_KEY=...` または `GEMINI_API_KEY=...`）。

## 2. 設定ファイルの概要
- 使用ファイル: `configs/quantum/qaoa_llm_guided.yaml`
- ターゲット: 4 量子ビットのリング (0-1-2-3-0) で MaxCut。QAOA の n_layers は 1〜4。
- 探索: NSGA-II 多目的（energy 最小化・depth 最小化）、最大 36 トライアル。
- LLM: OpenAI `gpt-4.1-mini` を想定。`llm_guidance` で角度提案、`planner` でバッチ戦略、`llm_critic` でログ診断を実行。
- プロンプト: `planner_prompts/quantum_circuit_planner.md` に加え、追記事項を `planner_prompts/qaoa_planner_directives.md` で指定。
- ノイズ: Aer の粗い NISQ ノイズを有効化（`noise_simulation`）。

## 3. 実行前チェック
```bash
# YAML 構造とメトリクス整合性を確認
astraia --config configs/quantum/qaoa_llm_guided.yaml --summarize

# .env のキーと LLM 疎通を事前確認
astraia --config configs/quantum/qaoa_llm_guided.yaml --dry-run
```

## 4. 実行コマンド
```bash
astraia --config configs/quantum/qaoa_llm_guided.yaml
```

- LLM コストを抑えたい場合は `llm.max_calls` や `max_tokens_per_run` を下げる。
- Gemini を使う場合は `llm.provider: gemini` とモデル名を変更し、`GEMINI_API_KEY` を `.env` に入れる。

## 5. 結果の確認
- `runs/qaoa_llm_guided/` に成果物が保存されます。
  - `log.csv`: トライアルごとのパラメタとメトリクス (`metric_energy`, `metric_depth`, ほかノイズ指標)。
  - `report.md`: ベスト試行・Pareto 概要・LLM クリティックの所見。
  - `llm_usage.csv`: プランナー/ガイダンス/クリティックのコール履歴。
  - `config_resolved.json`: 実行時の検証済み設定スナップショット。
- 可視化（任意）:

```bash
astraia visualize --run-id qaoa_llm_guided --type pareto --metric energy --metric depth
astraia visualize --run-id qaoa_llm_guided --type history --metric energy
```

## 6. よく使う調整ポイント
- グラフを変える: `evaluator.edges` を置き換え、必要なら `num_qubits` と `search_space` の角度数を合わせる。
- レイヤー上限を変える: `search_space.n_layers.high` と gamma/beta の数を揃え、必要なら `max_trials` を増やす。
- ノイズなしで試す: `noise_simulation.enabled: false`。
- コスト削減: `llm_critic.enabled: false` にしてクリティックを止める、`n_trials` を減らす。
