# CLI リファレンス

このページでわかること:
- `astraia` CLI の引数とサブコマンドの一覧、主なデフォルト値

対象読者:
- コマンドの全体像やオプションを手早く調べたい人

## 基本オプション

| オプション | 説明 |
| --- | --- |
| `--config PATH` | 最適化設定ファイル（既定: `configs/qgan_kl.yaml`）。 |
| `--summarize` | 実行せず設定サマリのみ出力。 |
| `--as-json` | バリデーション済み設定を JSON で表示。 |
| `--planner {none,rule,llm}` | 設定ファイル内のプランナー指定を一時的に上書き。`none` でプランナー無効。 |
| `--planner-config PATH` | プランナー固有設定（プロンプトなど）を差し替え。`--planner llm` と組み合わせて使用。 |
| `--dry-run` | `.env` の秘密鍵検証と LLM 疎通テストのみ実施。`--summarize` / `--as-json` とは併用不可。 |
| `visualize` | ログ CSV からベスト値履歴 or Pareto front を描画。詳細は下記。 |

## `runs` サブコマンド

すべてのサブコマンドで `--runs-root`（既定: `runs`）を付けると成果物ルートを差し替えられます。

| コマンド | 説明 | 例 |
| --- | --- | --- |
| `astraia runs list` | 既存の実行を一覧表示。`--status`、`--filter key=value`（`metadata.name=...` など）、`--sort`（`created/run_id/name/status/best_value`）、`--reverse`、`--limit`、`--json` で絞り込み。 | `astraia runs list --status completed --limit 10 --sort best_value --reverse` |
| `astraia runs show --run-id <id>` | 指定実行のメタデータ・成果物・解決済み設定を表示。`--as-json` も可。 | `astraia runs show --run-id qgan_kl_minimal` |
| `astraia runs delete --run-id <id>` | 成果物ディレクトリを削除。`--dry-run` で予行演習、`--yes` で確認スキップ。 | `astraia runs delete --run-id old_run --yes` |
| `astraia runs status --run-id <id>` | 任意の状態メモを付与。`--state`（必須）に加えて `--best-value`、`--metric name=value`、`--payload key=value`、`--tag key=value` を付与。`--pareto-summary <json>` で Pareto 集計を添付可能。 | `astraia runs status --run-id demo --state completed --best-value 0.12 --note "収束" --pareto-summary pareto.json` |
| `astraia runs diff --run-id <a> --run-id <b>` | 検証済み設定 (`config_resolved.json`) の差分を表示。複数 run を指定可能。 | `astraia runs diff --run-id baseline --run-id tuned` |
| `astraia runs compare --runs <id...>` | ログ CSV を集計し、メトリクス別のベスト/中央値/平均を比較。`--metric` や `--stat` で列を制御し、`--json` で機械可読に。 | `astraia runs compare --runs qgan_kl_minimal another_run --metric kl --stat best` |
| `astraia runs ab-template --config <file>` | 同じ seed で no-LLM / init-only / mixed / full の 4 パターンを連続実行し、結果を表で出力。`--seed` で固定、`--init-trials` `--mix-ratio` で LLM 配分を調整。 | `astraia runs ab-template --config configs/quantum/qaoa_llm_guided.yaml --seed 1234 --mix-ratio 0.4` |

## `visualize` サブコマンド

| オプション | 説明 |
| --- | --- |
| `--run-id <id>` | 可視化対象の run_id（必須）。`--runs-root` で場所を変えられる。 |
| `--type {history, pareto}` | ベスト値履歴または Pareto front を描画。Pareto の場合はメトリクスを 2 つ指定。 |
| `--metric NAME` | 描画するメトリクスを列挙。未指定なら config → `log.csv` の順で自動検出。 |
| `--output PATH` | 保存先を明示。省略時は `runs/<id>/visualization_history.png` / `visualization_pareto.png` が生成。 |
| `--title TEXT` | グラフタイトルを上書き。 |
