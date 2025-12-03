# 実行で生成される成果物

このページでは `astraia` 実行後に `runs/<run_id>/` に保存されるファイルと役割をまとめます。

| ファイル | 内容 |
| --- | --- |
| `config_original.yaml` / `config_resolved.json` | 実行時点の設定（元ファイルのコピーと検証済み JSON）。 |
| `log.csv` | 各トライアルのパラメタ・メトリクス。`astraia visualize` や `runs compare` が参照。 |
| `report.md` | ベスト試行、Pareto front、LLM クリティック、ハイパーボリュームの Markdown レポート。 |
| `summary.json` | Run 全体のサマリ。`pareto_count` / `hypervolume` / `best_energy_gap` / `depth_best` / `llm_calls` / `tokens` / `llm_accept_rate` などの比較指標を集計。 |
| `llm_usage.csv` | LLM 呼び出しログ（LLM 設定が存在する場合）。 |
| `llm_messages.jsonl` | プロンプト/レスポンス/パース結果/採否/レイテンシを含む LLM 監査ログと、提案→enqueue→トライアル結果までのトレース。 |
| `comparison_summary.json` | `runs compare` が利用する統計キャッシュ。`runs status --pareto-summary` で Pareto サマリを追記可能。 |
| `meta.json` | `runs list` / `runs show` が参照するメタデータ。Git コミットやシード、アーティファクトの場所を記録。 |

`astraia visualize` を実行すると `log_history.png` や `log_pareto.png` が同ディレクトリに生成されます。`--output` を指定すれば任意のパスに保存できます。
