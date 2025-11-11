# Astraia MVP Skeleton

このリポジトリは、LLM 駆動最適化フレームワーク「Astraia」の最小実装（MVP）を収録しています。YAML で定義された最適化設定を読み込み、
Optuna ベースの探索ループを実行し、結果を CSV ログおよび Markdown レポートとして保存します。必要に応じて LLM を用いた探索候補提案
（LLM Guidance）、進行中の探索戦略調整（Meta Search）、失敗シグナルの診断レポート（LLM Critic）も有効化できます。

## セットアップ

Python 3.11 以上を推奨します。依存ライブラリは `PyYAML`、`Optuna`、`Pydantic` を使用します。
（オフライン環境では最小互換レイヤーが自動ロードされますが、本番では公式実装のインストールを推奨します。）
OpenAI / Gemini などの LLM を併用する場合はオプショナル依存を extras として指定できます。

```bash
pip install -e .[openai]
# または
pip install -e .[gemini]
pip install -e .[llm]  # 両方まとめて

# LLM を利用しない場合はベースのみ
pip install -e .
```

LLM プロバイダの API キーは `.env` に記載し、`OPENAI_API_KEY` や `GEMINI_API_KEY` を環境変数として読み込ませてください。
雛形として `.env.example` を用意しています。

## 使い方

以下のコマンドを実行すると、`configs/qgan_kl.yaml` を読み込み、探索・評価・レポート生成を含む最小構成の最適化ループが走ります。

```bash
astraia --config configs/qgan_kl.yaml
# または
python -m astraia.cli --config configs/qgan_kl.yaml
```

実行後は以下が生成されます。

- `runs/qgan_kl_minimal/log.csv`: 各トライアルのパラメタとメトリクスを記録
- `reports/qgan_kl_minimal.md`: ベストトライアルの概要レポート
- （LLM を有効化した場合）`runs/qgan_kl_minimal/llm_usage.csv`: LLM 呼び出しの使用量ログ

CLI には確認用のオプションも用意しています。YAML は Pydantic モデルにマップされ、必須フィールドの存在だけでなく型・範囲・依存関係
まで検証されます（例: `search.metric` が `report.metrics` に含まれているか、探索空間パラメタで `low < high` が守られているか 等）。

```bash
# 設定のサマリのみ表示
python -m astraia.cli --config configs/qgan_kl.yaml --summarize

# YAML の内容を JSON で出力
python -m astraia.cli --config configs/qgan_kl.yaml --as-json

# プランナーのバックエンドを一時的に差し替え
python -m astraia.cli --config configs/qgan_kl.yaml --planner llm \
  --planner-config planner_prompts/qgan_kl_minimal.txt
```

## CLI オプション一覧

| オプション | 説明 |
| --- | --- |
| `--config PATH` | 最適化設定ファイル（既定: `configs/qgan_kl.yaml`） |
| `--summarize` | 実行せずに設定サマリのみ出力 |
| `--as-json` | バリデーション済み設定を JSON で表示 |
| `--planner {none,rule,llm}` | 設定ファイルのプランナー指定を一時的に上書き |
| `--planner-config PATH` | プランナー固有設定のパスを注入（例: プロンプトファイル） |

## 設定ファイルの構成

`OptimizationConfig` (`src/astraia/config.py`) が YAML を検証します。主なセクションは以下のとおりです。

- `metadata`: 実験名と説明。
- `seed`: 乱数シード。LLM ガイダンスやメタ探索の乱択にも利用されます。
- `search`: Optuna 設定。サンプラーは `tpe` または `random` を指定できます。
- `stopping`: ループ停止条件 (`max_trials`, `max_time_minutes`, `no_improve_patience`)。
- `search_space`: 各パラメタの探索範囲を `float` / `int` / `categorical` から選択して定義します。
- `evaluator`: `module` と `callable` を指定して評価器をロードします。戻り値には主指標（例: `kl`）とレポート対象のメトリクスを含めます。
- `report`: 出力ディレクトリ、ファイル名、表示したいメトリクスを指定します。`search.metric` に設定した主指標を必ず含めてください。
- `artifacts`: ログ出力先 (`log_file`) と任意のルート (`run_root`) を指定します。`run_root` を設定すると LLM 使用量ログも同ディレクトリに生成されます。
- `planner`: ルールベースまたは LLM バックエンドを選択できます。CLI オプションで一時的に無効化 (`none`) も可能です。
- `llm`: LLM プロバイダ（`openai` / `gemini`）とモデル名、任意の使用量ログ出力先を設定します。未指定の場合でも `artifacts.run_root` があれば自動で `llm_usage.csv` を割り当てます。
- `llm_guidance`: LLM に探索候補を生成させる機能です。未設定または `enabled: false` の場合は乱択候補にフォールバックします。
- `meta_search`: 一定間隔で探索状況を要約し、LLM もしくはヒューリスティックでサンプラー切替や探索範囲縮小などの指示を返します。
- `llm_critic`: 実行後のログを分析し、失敗シグナルと改善提案を Markdown セクションとしてレポートに追加します。LLM が利用できない場合はヒューリスティックの診断を返します。

## 評価モジュールの構造

- 全ての評価器は `BaseEvaluator` (`src/astraia/evaluators/base.py`) を継承するか、互換の `evaluate(params, seed)` を提供するファクトリから生成します。
- qGAN KL 用の最小実装は `QGANKLEvaluator` (`src/astraia/evaluators/qgan_kl.py`) として提供され、YAML で `evaluator.callable: create_evaluator` を指定するとインスタンス化されます。
- 評価結果の辞書には主指標 (`kl`) とレポート対象メトリクス（例: `depth`, `params`, `shots`）を含めます。Optuna への報告値は `search.metric` で指定したキーの値になります。

## LLM 機能

- `llm_guidance`: 提案候補をバッチ生成します。LLM 利用が失敗した場合でもキャッシュと乱択フォールバックで探索を継続します。
- `meta_search`: トライアル履歴を集約し、LLM またはヒューリスティックで探索戦略を更新します。サンプラーの切り替え、探索空間の縮小、早期停止条件の調整に対応しています。
- `llm_critic`: ログから NaN/Inf や改善停滞を検出し、LLM が有効な場合は Markdown レポートを生成します。使用量は `llm_usage.csv` に追記されます。
- LLM プロバイダは OpenAI Responses API と Google Gemini に対応し、未インストール時は `ProviderUnavailableError` を検知してヒューリスティックにフォールバックします。

## 現在の進捗状況

- qGAN KL 最適化の最小構成となる設定ファイル（`configs/qgan_kl.yaml`）を整備。
- CLI (`astraia` / `python -m astraia.cli`) から設定読み込み、バリデーション、サマリ表示、JSON 出力、探索実行に対応。
- Optuna を用いた探索ループを実装し、CSV ログ・Markdown レポートを生成。ベスト指標・早期停止理由も記録。
- Pydantic ベースの設定スキーマを導入し、探索空間・指標の整合性や LLM 機能の依存関係を検証。
- LLM ガイダンス、メタ探索アジャスタ、LLM 批評レポート、LLM 使用量ロガーを追加し、依存未インストール時は安全にフォールバック。
- `tests/` 以下に設定検証・LLM 補助機能・qGAN 評価器のユニットテストを整備。

## 今後のプログラム計画

1. **LLM プランナー/メタ戦略の強化**: プロンプトテンプレートとヒューリスティックのチューニング、追加サンプラー対応。
2. **評価モジュールの拡充**: 量子回路以外のベンチマークやノイズモデルを追加し、制約評価や多目的指標に対応。
3. **レポートの可視化強化**: 図表生成や外部トラッキングサービス（MLflow など）との連携を検討。
4. **オーケストレーション改善**: CLI からのドライラン / 再開機能、実行中メタ情報のストリーミング出力を追加。

## TODO リスト

- [x] Optuna と評価器を接続した最小探索ループの実装
- [x] CSV ログと Markdown レポート出力の整備
- [x] LLM ガイダンス / メタ探索 / 批評モジュールと Usage ログ機構の導入
- [x] CLI オプション（サマリ / JSON 出力 / プランナー上書き）の実装
- [ ] CI ワークフローでの自動検証（lint / test）の導入
- [ ] 追加ベンチマーク向け設定ファイルと評価器の実装

## テスト

標準ライブラリの `unittest` を利用したテストスイートを用意しています。

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
