# Anemoi MVP Skeleton

このリポジトリは、LLM駆動最適化フレームワーク「Anemoi」のMVPに向けた初期セットアップです。計画書で示された最初のステップに従い、最小実験サイズの qGAN KL 最適化設定ファイルを確定し、Optuna を用いた自動探索ループまで実行できる CLI を提供しています。

## セットアップ

Python 3.11 以上を推奨します。依存ライブラリは `PyYAML` と `Optuna` を使用します。

```bash
uv pip install -r pyproject.toml  # または pip install -e .
```

## 使い方

以下のコマンドを実行すると、`configs/qgan_kl.yaml` を読み込み、探索・評価・レポート生成を含む最小構成の最適化ループが走ります。

```bash
python -m anemoi.cli --config configs/qgan_kl.yaml
```

実行後は以下が生成されます。

- `runs/qgan_kl_minimal/log.csv`: 各トライアルのパラメタとメトリクスを記録
- `reports/qgan_kl_minimal.md`: ベストトライアルの概要レポート

CLI には確認用のオプションも用意しています。

```bash
# 設定のサマリのみ表示
python -m anemoi.cli --config configs/qgan_kl.yaml --summarize

# YAML の内容を JSON で出力
python -m anemoi.cli --config configs/qgan_kl.yaml --as-json
```

設定ファイルは後続の開発で最適化ループの各コンポーネントに接続するための基礎となります。

## 現在の進捗状況

- qGAN KL 最適化の最小構成となる設定ファイル（`configs/qgan_kl.yaml`）を確定。
- CLI (`python -m anemoi.cli`) から設定ファイルの読み込み・バリデーション・要約表示・JSON 出力に対応。
- Optuna を用いた最小構成の自動探索ループを実装し、CSV ログと Markdown レポートを生成。
- MVP 段階で要求される必須フィールド（メタデータ・探索設定・停止条件・レポート設定・評価器）の存在チェックを実装。

## 今後のプログラム計画

1. **最適化ループの実装**: 探索ライブラリ（例: Optuna）と評価器を接続し、設定ファイルをもとに自動探索が実行できるようにする。✅ 完了
2. **評価モジュールの拡充**: `src/anemoi/evaluators` 以下に qGAN 固有の評価ロジックや、将来的な拡張に備えた共通インターフェースを追加。
3. **レポート生成の自動化**: 実験結果を標準化した形式で保存し、再現性と可観測性を高めるためのテンプレートを整備。
4. **CLI 機能の強化**: 実行モードの切り替え（例: ドライラン / 本番）、進捗表示、ロギング設定などを追加。

## TODO リスト

- [x] 設定ファイルに記載された探索ライブラリと実装コードの接続
- [ ] 評価器プラグインの実装およびテストケースの整備
- [x] 実行ログ・メトリクスのファイル出力サポート
- [ ] CI ワークフローでの自動検証（lint/test）の導入
- [x] README に最適化ループの使用例と想定される入力/出力サンプルを追加
