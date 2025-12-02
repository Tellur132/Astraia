# 設定ファイルの構成

このページでは `src/astraia/config.py` の `OptimizationConfig` が検証する主なセクションをまとめます。各設定は YAML で記述し、CLI から `--summarize` / `--as-json` で確認できます。

## セクション一覧
- **metadata**: 実験名や説明。`runs list` などで表示される。
- **seed**: 乱数シード。Optuna と LLM まわりで共有される。
- **search**: Optuna の方向（`minimize`/`maximize`）、対象メトリクス、サンプラーなどの設定。
- **stopping**: 停止条件 (`max_trials`, `max_time_minutes`, `no_improve_patience`)。
- **search_space**: `float` / `int` / `categorical` など型別に探索範囲を定義。
- **evaluator**: `module` と `callable` を指定し、`BaseEvaluator` 互換の評価器をロード。
- **report**: Markdown レポートや Pareto CSV/PNG の出力先と表示メトリクス。
- **artifacts**: ルートディレクトリ (`run_root`)、ログファイル (`log_file`)、追加アーティファクトの出力パス。
- **planner**: ルール/LLM バックエンド（`backend`）と固有設定パス (`config_path`)。
- **llm**: `provider`（`openai` or `gemini`）、`model`、`usage_log` などの LLM 設定。
- **llm_guidance**: LLM を使った候補生成の有効化フラグ、バッチサイズ、プロンプト設定。
- **meta_search**: トライアル要約頻度、利用する LLM またはヒューリスティックの種類。Pareto summary の代表点や trade-off テキストもここで生成。
- **llm_critic**: 実行後レポートを生成する LLM/ヒューリスティックの設定。

## バリデーションのポイント
- `search.metric(s)` と `report.metrics` の整合性、探索空間の境界チェックを実施。
- `astraia --summarize` / `--as-json` で検証エラーを事前に把握し、GUI バックエンドからも同じ検証ロジックを利用可能。
- LLM なしで試したい場合は設定内で `llm_guidance.enabled=false` などにするか、CLI で `--planner none` を指定します。
