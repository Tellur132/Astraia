# 実行で生成される成果物

このページでは `astraia` 実行後に `runs/<run_id>/` に保存されるファイルと役割をまとめます。成果物ルートは CLI の `--runs-root` で変更できます。

| ファイル | 内容 |
| --- | --- |
| `config_original.yaml` / `config_resolved.json` | 実行時点の設定（元ファイルのコピーと検証済み JSON）。 |
| `log.csv` | 各トライアルのパラメタ・メトリクス。`astraia visualize` や `runs compare` が参照。 |
| `report.md` | ベスト試行、Pareto front、LLM クリティック、ハイパーボリュームの Markdown レポート。 |
| `summary.json` | Run 全体のサマリ。`pareto_count` / `hypervolume` / `llm_calls` / `tokens` / `llm_accept_rate` などの比較指標とベスト解を集計。`runs ab-template` でもここを参照します。 |
| `llm_usage.csv` | LLM 呼び出しログ（LLM 設定が存在する場合）。 |
| `llm_messages.jsonl` | プロンプト/レスポンス/パース結果/採否/レイテンシを含む LLM 監査ログと、提案→enqueue→トライアル結果までのトレース。 |
| `comparison_summary.json` | `runs compare` が利用する統計キャッシュ。`runs status --pareto-summary` で Pareto サマリを追記可能。 |
| `meta.json` | `runs list` / `runs show` が参照するメタデータ。Git コミットやシード、アーティファクトの場所を記録。`summary.json` を生成すると `artifacts.summary` にも追記されます。 |
| `visualization_history.png` / `visualization_pareto.png` | `astraia visualize` で生成される既定の画像。`--output` を指定すれば任意のパスに保存できます。 |

`runs/` 直下には `comparisons/<id>.json`（LLM 有効/無効の比較レコード）などが作成されることもあります。run ごとに閉じた成果物は上表のとおり `runs/<run_id>/` に集約されます。
