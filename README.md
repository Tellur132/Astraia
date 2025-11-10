# Anemoi MVP Skeleton

このリポジトリは、LLM駆動最適化フレームワーク「Anemoi」のMVPに向けた初期セットアップです。計画書で示された最初のステップに従い、最小実験サイズのqGAN KL最適化設定ファイルを確定し、それを読み込むシンプルなCLIを用意しています。

## セットアップ

Python 3.11 以上を推奨します。依存ライブラリは `PyYAML` のみです。

```bash
uv pip install -r pyproject.toml  # または pip install -e .
```

## 使い方

以下のコマンドを実行すると、`configs/qgan_kl.yaml` を読み込み、設定内容の概要を表示します。

```bash
python -m anemoi.cli --config configs/qgan_kl.yaml
```

設定ファイルは後続の開発で最適化ループの各コンポーネントに接続するための基礎となります。
