# CLI リファレンス

このページは `astraia` CLI の主なオプションとサブコマンドを一覧します。詳細な使い方や例は他のガイドを参照してください。

## 基本オプション

| オプション | 説明 |
| --- | --- |
| `--config PATH` | 最適化設定ファイル（既定: `configs/qgan_kl.yaml`）。 |
| `--summarize` | 実行せず設定サマリのみ出力。 |
| `--as-json` | バリデーション済み設定を JSON で表示。 |
| `--planner {none,rule,llm}` | 設定ファイル内のプランナー指定を一時的に上書き。 |
| `--planner-config PATH` | プランナー固有設定（プロンプトなど）を差し替え。 |
| `--dry-run` | `.env` の秘密鍵検証と LLM 疎通テストのみ実施。 |
| `visualize` | ログ CSV からベスト値履歴 or Pareto front を描画。 |

## `runs` サブコマンド

| コマンド | 説明 | 例 |
| --- | --- | --- |
| `astraia runs list` | 既存の実行を一覧表示。`--status`、`--filter key=value`、`--json`、`--limit` 等で絞り込み可能。 | `astraia runs list --status completed --limit 10` |
| `astraia runs show --run-id <id>` | 指定実行のメタデータ・成果物・解決済み設定を表示。`--as-json` も可。 | `astraia runs show --run-id qgan_kl_minimal` |
| `astraia runs delete --run-id <id>` | 成果物ディレクトリを削除。`--dry-run` や `--yes` で挙動制御。 | `astraia runs delete --run-id old_run --yes` |
| `astraia runs status --run-id <id>` | 任意の状態メモを付与。`--state`（必須）に加えて `--best-value`、`--metric name=value`、`--payload key=value`、`--tag key=value` を付与。 | `astraia runs status --run-id demo --state completed --best-value 0.12 --note "収束"` |
| `astraia runs diff --run-id <a> --run-id <b>` | 検証済み設定 (`config_resolved.json`) の差分を表示。 | `astraia runs diff --run-id baseline --run-id tuned` |
| `astraia runs compare --runs <id...>` | ログ CSV を集計し、メトリクス別のベスト/中央値/平均を比較。`--metric` や `--stat` で列を制御し、`--json` で機械可読に。 | `astraia runs compare --runs qgan_kl_minimal another_run --metric kl --stat best` |

## `visualize` サブコマンド

| オプション | 説明 |
| --- | --- |
| `--type {history, pareto}` | ベスト値履歴または Pareto front を描画。Pareto の場合はメトリクスを 2 つ指定。 |
| `--metric NAME` | 描画するメトリクスを列挙。多目的探索時は `--metric kl --metric depth` のように複数指定。 |
| `--output PATH` | 保存先を明示的に指定。省略時は `runs/<id>/log_history.png` などが生成。 |
| `--title TEXT` | グラフタイトルを上書き。 |
