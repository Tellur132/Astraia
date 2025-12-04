# 設定ファイルの構成

このページでは `src/astraia/config.py` の `OptimizationConfig` が検証する主なセクションをまとめます。各設定は YAML で記述し、CLI から `--summarize` / `--as-json` で確認できます。

## セクション一覧
- **metadata**: 実験名や説明。`runs list` などで表示され、`run_id` の初期値にも使われる。
- **seed**: 乱数シード。Optuna・LLM・NumPy・量子ノイズシミュレーションで共有される。
- **search**: Optuna の方向（`minimize`/`maximize`）、対象メトリクス、サンプラーなどの設定。複数メトリクスを定義すると自動で multi-objective とみなされる。
- **stopping**: 停止条件 (`max_trials`, `max_time_minutes`, `no_improve_patience`, `max_total_cost` + `cost_metric`)。
- **search_space**: `float` / `int` / `categorical` / `llm_only` で探索範囲を定義。`llm_only` は LLM が提案しない限り既定値を使うパラメタ。
- **evaluator**: `module` と `callable` を指定し、`BaseEvaluator` 互換の評価器をロード。
- **report**: Markdown レポートや Pareto CSV/PNG の出力先と表示メトリクス。`report.metrics` に含まれないメトリクスはログに出てもサマリに表示されない。
- **artifacts**: ルートディレクトリ (`run_root`)、ログファイル (`log_file`) などのパス。未設定でも CLI が `runs/<run_id>/` に自動で割り当てる。
- **planner**: ルール/LLM バックエンド（`backend`）。LLM バックエンドでは `role`/`prompt_template` または `roles.candidate.llm` などの役割単位設定を指定する。
- **llm**: `provider`（`openai` or `gemini`）、`model`、使用量ログ（`usage_log`）、トレースログ（`trace_log`）、コスト上限などの LLM 設定。
- **llm_guidance**: LLM を使った候補生成の有効化フラグとガイド設定。`mode` で `full` / `init_only` / `mixed` を切替え、`mix_ratio` や `max_llm_trials`、ハイパーボリューム保護 (`hv_guard_*`) などで投入頻度を制御できる。
- **meta_search**: 試行履歴をまとめ、ポリシーや LLM の提案でサンプラー変更・探索空間スケール調整を行う設定。Pareto summary の代表点や trade-off テキスト生成も含む。
- **llm_critic**: ログを見て失敗傾向を診断し、改善案を Markdown で出す LLM/ヒューリスティックの設定。`max_history` やカスタム冒頭文 (`prompt_preamble`) を指定可能。
- **diversity_guard**: 直近の試行で多様性が失われたときに特定パラメタのユニーク数を保つためのガード。`window` / `min_unique` / `stratified_fraction` などでしきい値を設定。

## バリデーションのポイント
- `search.metric(s)` と `report.metrics` の整合性、探索空間の境界チェックを実施。
- `astraia --summarize` / `--as-json` で検証エラーを事前に把握し、GUI バックエンドからも同じ検証ロジックを利用可能。
- LLM なしで試したい場合は設定内で `llm_guidance.enabled=false` などにするか、CLI で `--planner none` を指定します。
