# Astraia MVP Skeleton

「Astraia」は、YAML 設定で定義した評価器・探索空間を読み込んで Optuna ベースの探索ループを自動生成し、CSV/Markdown/LLM レポートを含む成果物を残す LLM 駆動最適化フレームワークです。本リポジトリには MVP（最小実行可能プロダクト）のソースコードと、誰でもすぐに試せる qGAN KL 最適化のサンプルが含まれています。

## 主な機能

- `configs/` にある YAML 設定を `OptimizationConfig`（`src/astraia/config.py`）で検証しながら読み込み。
- Optuna による探索・停止条件制御・ベストトライアル追跡を自動化。
- CSV ログ、Markdown レポート、LLM 使用量ログを `runs/` 以下に保存。
- LLM ガイダンス / メタ探索 / LLM クリティックで探索候補生成・戦略調整・診断を強化。
- `astraia runs` サブコマンドで実験管理（一覧、詳細、削除、ステータス更新）が可能。

## セットアップ手順

1. Python 3.11 以上と pip を用意します。
2. このリポジトリをクローンし、任意の仮想環境を有効化します。
3. 依存関係をインストールします。LLM を使用しない場合はベースのみ、OpenAI/Gemini を使う場合は extras を付けます。

   ```bash
   pip install -e .           # ベースのみ
   pip install -e .[openai]   # OpenAI 連携
   pip install -e .[gemini]   # Gemini 連携
   pip install -e .[llm]      # 両方まとめて
   ```

4. LLM API キーを `.env` に追記します（例: `OPENAI_API_KEY=sk-...`）。テンプレートは `.env.example` をコピーしてください。
5. `astraia --dry-run` を実行すると `.env` の必須キー検証と LLM プロバイダ疎通チェックを行えます。

## クイックスタート

サンプル設定（`configs/qgan_kl.yaml`）を用いて最適化を行う場合は次のコマンドを実行します。

```bash
astraia --config configs/qgan_kl.yaml
# または python -m astraia.cli --config configs/qgan_kl.yaml
```

実行後は以下の成果物が作成されます。

- `runs/qgan_kl_minimal/log.csv`: 各トライアルのパラメタとメトリクス
- `reports/qgan_kl_minimal.md`: ベストトライアルをまとめた Markdown レポート
- `runs/qgan_kl_minimal/llm_usage.csv`: LLM 呼び出しログ（LLM 有効時のみ）

CLI は常に設定ファイルを Pydantic で検証し、`search.metric` と `report.metrics` の整合性や探索空間の境界チェック（`low < high` など）を実施します。実行前に内容を確認したい場合は以下が便利です。

```bash
# 設定サマリを確認
astraia --config configs/qgan_kl.yaml --summarize

# バリデーション後の設定を JSON で取得
astraia --config configs/qgan_kl.yaml --as-json

# プランナーを一時的に上書き
astraia --config configs/qgan_kl.yaml --planner llm \
  --planner-config planner_prompts/qgan_kl_minimal.txt
```

## CLI リファレンス

### 基本オプション

| オプション | 説明 |
| --- | --- |
| `--config PATH` | 最適化設定ファイル（既定: `configs/qgan_kl.yaml`） |
| `--summarize` | 実行せずに設定サマリのみ出力 |
| `--as-json` | バリデーション済み設定を JSON で表示 |
| `--planner {none,rule,llm}` | 設定ファイルのプランナー指定を一時的に上書き |
| `--planner-config PATH` | プランナー固有設定（プロンプトなど）を差し替え |
| `--dry-run` | `.env` の秘密鍵検証と LLM 疎通テストのみ実施 |

### `runs` サブコマンド

`astraia runs` は `runs/` ディレクトリをメタデータ付きで管理します。全サブコマンドは `--runs-root` でルートを変更できます。

| コマンド | 説明 | 例 |
| --- | --- | --- |
| `astraia runs list` | 既存の実行を一覧表示。`--status`、`--filter key=value`、`--json`、`--limit` 等で絞り込み可能。 | `astraia runs list --status completed --limit 10` |
| `astraia runs show --run-id <id>` | 指定した実行のメタデータ・成果物・解決済み設定を表示。`--as-json` も可。 | `astraia runs show --run-id qgan_kl_minimal` |
| `astraia runs delete --run-id <id>` | 成果物ディレクトリを削除。`--dry-run` や `--yes` で挙動制御。 | `astraia runs delete --run-id old_run --yes` |
| `astraia runs status --run-id <id>` | 任意の状態メモを付与。`--state`（必須）に加えて `--best-value`、`--metric name=value`、`--payload key=value` で指標を更新。 | `astraia runs status --run-id demo --state archived --best-value 0.12` |

## 設定ファイルの構成

`src/astraia/config.py` の `OptimizationConfig` が YAML を厳密に検証します。主なセクションは以下の通りです。

- `metadata`: 実験名や説明。`runs list` などで表示されます。
- `seed`: 乱数シード（Optuna、LLM ガイダンス、メタ探索で共有）。
- `search`: Optuna 設定。ライブラリやサンプラー、最適化方向（`minimize`/`maximize`）、対象メトリクスを定義。
- `stopping`: 停止条件 (`max_trials`, `max_time_minutes`, `no_improve_patience`)。
- `search_space`: `float` / `int` / `categorical` など型別に探索範囲を定義。
- `evaluator`: `module` と `callable` を指定し、`BaseEvaluator` 互換の評価器をロード。
- `report`: Markdown レポートの保存先や表示するメトリクス。
- `artifacts`: ルートディレクトリ (`run_root`)、ログファイル (`log_file`)、追加アーティファクトの出力パス。
- `planner`: ルール / LLM バックエンド（`backend`）と固有設定パス (`config_path`)。
- `llm`: `provider`（`openai` or `gemini`）、`model`、`usage_log` などの LLM 設定。
- `llm_guidance`: LLM を使った候補生成の有効化フラグ、バッチサイズ、プロンプト設定。
- `meta_search`: トライアル要約頻度、利用する LLM またはヒューリスティックの種類。
- `llm_critic`: 実行後レポートを生成する LLM/ヒューリスティックの設定。

## 評価モジュール

- すべての評価器は `BaseEvaluator`（`src/astraia/evaluators/base.py`）を継承し、`evaluate(params, seed)` を実装します。
- qGAN KL のサンプルは `QGANKLEvaluator`（`src/astraia/evaluators/qgan_kl.py`）として実装済みで、`evaluator.callable: create_evaluator` によってファクトリを呼び出します。
- 評価結果は `kl`（主指標）や `depth`, `shots`, `params` などを含む辞書で返却され、`search.metric` に一致するキーを Optuna へ報告します。
- `BaseEvaluator` は `trial_timeout_sec` / `max_retries` / `graceful_nan_policy` を解釈し、例外・NaN/Inf・タイムアウト発生時でも構造化された失敗ペイロードで探索を継続します。
- 乱数シードは Python/NumPy/PyTorch へ一括で設定され、一時ディレクトリで副作用を隔離します。

## LLM 連携

- `llm_guidance`: LLM で探索候補をバッチ生成し、失敗時はキャッシュや乱択でフォールバックします。
- `meta_search`: トライアル履歴を要約して LLM またはヒューリスティックに渡し、サンプラー切り替え・探索範囲縮小・早期停止を指示できます。
- `llm_critic`: 実行ログを解析し、NaN/Inf や停滞を Markdown で報告。`usage_log` または `artifacts.run_root` 配下に `llm_usage.csv` を追記します。
- `.env` の秘密鍵読み込みは `ensure_env_keys`（`astraia.cli` 内）が行い、未設定の場合は明示的なエラーで終了します。
- `--dry-run` オプションは設定検証 → `.env` チェック → LLM SDK の `ping` 呼び出し（`create_llm_provider` 経由）までを実行し、ネットワーク疎通や API 権限を事前確認できます。

## ディレクトリ構成

| パス | 内容 |
| --- | --- |
| `configs/` | サンプル設定。`qgan_kl.yaml` がデフォルト実験です。 |
| `planner_prompts/` | LLM プランナー向けのプロンプトテンプレート。`--planner-config` で指定。 |
| `src/astraia/` | CLI、設定検証、Optuna ループ、LLM 関連モジュールの実装。 |
| `tests/` | `unittest` ベースのテストスイート。環境差分を減らすため `PYTHONPATH=src` で起動してください。 |
| `runs/` | 実行ごとの成果物とメタデータ。`astraia runs` で操作します。 |
| `reports/` | Markdown レポート出力先（`report.output_dir` で変更可）。 |

## 現在の進捗状況

- qGAN KL 最適化の最小構成設定（`configs/qgan_kl.yaml`）を整備。
- CLI から設定読み込み / バリデーション / サマリ表示 / JSON 出力 / 探索実行 / 実行管理サブコマンドに対応。
- Optuna ベースの探索ループと CSV ログ・Markdown レポート生成を実装。ベスト値や早期停止理由も記録。
- Pydantic スキーマで探索空間や LLM 依存関係を検証。
- LLM ガイダンス、メタ探索、LLM クリティック、使用量ロガーを実装し、依存未インストール時は安全にフォールバック。
- `tests/` 以下に設定検証・LLM 補助機能・qGAN 評価器のユニットテストを整備。

## 今後の計画

1. **LLM プランナー/メタ戦略の強化**: プロンプトテンプレートの改善や新規サンプラー対応。
2. **評価モジュールの拡充**: 量子回路以外のベンチマーク、ノイズモデル、多目的指標への拡張。
3. **レポート可視化の強化**: 図表生成や MLflow 等のトラッキングサービス連携。
4. **オーケストレーション改善**: 実行再開、ドライラン拡張、ストリーミング出力などの CLI 機能追加。

## TODO リスト

- [x] Optuna + 評価器による最小探索ループ
- [x] CSV ログ / Markdown レポート出力
- [x] LLM ガイダンス / メタ探索 / クリティック / 使用量ログ
- [x] CLI オプションと `runs` サブコマンド
- [ ] CI ワークフロー（lint/test）の自動実行
- [ ] 追加ベンチマーク設定と評価器

## テスト

標準ライブラリの `unittest` を利用したテストスイートを用意しています。

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
