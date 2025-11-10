# Astraia MVP Skeleton

このリポジトリは、LLM駆動最適化フレームワーク「Astraia」のMVPに向けた初期セットアップです。計画書で示された最初のステップに従い、最小実験サイズの qGAN KL 最適化設定ファイルを確定し、Optuna を用いた自動探索ループまで実行できる CLI を提供しています。

## セットアップ

Python 3.11 以上を推奨します。依存ライブラリは `PyYAML`、`Optuna`、`Pydantic` を使用します。
（オフライン環境では最小互換レイヤーが自動ロードされますが、本番では公式実装のインストールを推奨します。）

```bash
pip install -e .
```

## 使い方

以下のコマンドを実行すると、`configs/qgan_kl.yaml` を読み込み、探索・評価・レポート生成を含む最小構成の最適化ループが走ります。

```bash
python -m astraia.cli --config configs/qgan_kl.yaml
```

実行後は以下が生成されます。

- `runs/qgan_kl_minimal/log.csv`: 各トライアルのパラメタとメトリクスを記録
- `reports/qgan_kl_minimal.md`: ベストトライアルの概要レポート

CLI には確認用のオプションも用意しています。YAML は Pydantic モデルにマップされ、
必須フィールドの存在だけでなく、型・範囲・依存関係まで厳密に検証されます（例: `search.metric`
が `report.metrics` に含まれているか、各サーチパラメータの `low < high` が守られているか など）。

```bash
# 設定のサマリのみ表示
python -m astraia.cli --config configs/qgan_kl.yaml --summarize

# YAML の内容を JSON で出力
python -m astraia.cli --config configs/qgan_kl.yaml --as-json
```

設定ファイルは後続の開発で最適化ループの各コンポーネントに接続するための基礎となります。

## 評価モジュールの構造

- すべての評価器は `BaseEvaluator` (`src/astraia/evaluators/base.py`) を継承するか、同等の `evaluate(params, seed)` メソッドを持つファクトリ
  から生成します。
- qGAN KL 用の最小実装は `QGANKLEvaluator` (`src/astraia/evaluators/qgan_kl.py`) として提供しており、YAML からは
  `evaluator.callable: create_evaluator` を指定することでインスタンス化されます。
- 評価結果は `kl`, `depth`, `shots`, `params` の各メトリクスを返し、`search.metric` に設定した主目的（ここでは `kl`）を
  最適化ループが参照します。

### 評価結果 I/F の正式仕様

- すべての評価器は `kl`, `depth`, `shots`, `params` の 4 つのメトリクスを **必ず** `float` 値で返します。追加の診断値は
  任意に同じ辞書へ含めて構いません。
- コントロールフィールドとして以下が規約化されています。
  - `status`: `"ok" | "error" | "timeout"`。省略時は `"ok"` に正規化されます。
  - `timed_out`: タイムアウトが発生した場合に `True`。既定値は `False`。
  - `terminated_early`: 評価器側で早期終了した場合に `True`。既定値は `False`。
  - `elapsed_seconds`: 評価に要した実測時間（秒）。
  - `reason`: 異常終了やタイムアウト時の理由を示す文字列（任意）。
- 異常検知時は主要メトリクスに安全側の値を設定しつつ `status` と `reason` を付与します。例:

  ```json
  {
    "kl": Infinity,
    "depth": 1.0,
    "shots": 256.0,
    "params": 2.0,
    "status": "error",
    "reason": "nan_detected"
  }
  ```

- `timeout_seconds` を設定した評価器は、計測時間が上限を超えた場合に `status: "timeout"` と `reason: "timeout_exceeded"`
  を付与し、`kl` を `Infinity` へフォールバックします。早期終了を意図的に行った場合は `terminated_early: True` を明示してください。

## 現在の進捗状況

- qGAN KL 最適化の最小構成となる設定ファイル（`configs/qgan_kl.yaml`）を確定。
- CLI (`python -m astraia.cli`) から設定ファイルの読み込み・バリデーション・要約表示・JSON 出力に対応。
- Optuna を用いた最小構成の自動探索ループを実装し、CSV ログと Markdown レポートを生成。
- Pydantic ベースの設定スキーマを導入し、必須フィールドの存在に加えて範囲や依存関係まで厳密に検証できるようにした。
- `src/astraia/evaluators` に共通インターフェースと qGAN KL 用アナリティック評価器を追加し、コンフィグで差し替え可能にした。

## 今後のプログラム計画

1. **最適化ループの実装**: 探索ライブラリ（例: Optuna）と評価器を接続し、設定ファイルをもとに自動探索が実行できるようにする。✅ 完了
2. **評価モジュールの拡充**: `src/astraia/evaluators` 以下に qGAN 固有の評価ロジックや、将来的な拡張に備えた共通インターフェースを追加。
3. **レポート生成の自動化**: 実験結果を標準化した形式で保存し、再現性と可観測性を高めるためのテンプレートを整備。
4. **CLI 機能の強化**: 実行モードの切り替え（例: ドライラン / 本番）、進捗表示、ロギング設定などを追加。

## TODO リスト

- [x] 設定ファイルに記載された探索ライブラリと実装コードの接続
- [x] 評価器プラグインの実装およびテストケースの整備（簡易アナリティック版）
- [x] 実行ログ・メトリクスのファイル出力サポート
- [ ] CI ワークフローでの自動検証（lint/test）の導入
- [x] README に最適化ループの使用例と想定される入力/出力サンプルを追加

## テスト

ローカルで動作確認する場合は、標準ライブラリの `unittest` を利用したテストスイートを実行できます。

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
