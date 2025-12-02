# Astraia 概要

このページでは Astraia の現在の位置付けとコア機能を一枚にまとめます。詳細な手順や設定例はそれぞれの専用ページを参照してください。

## 目的と現在地
- YAML 設定を検証し、Optuna ベースの探索ループと成果物生成を自動化する「LLM 駆動最適化フレームワーク」の MVP。
- CLI は安定化済みで、`configs/` からそのまま qGAN / ZDT / 量子回路のサンプルを実行可能。
- LLM ガイダンス・メタ探索・LLM クリティックはオン/オフを切り替えられる実験的機能として提供。
- FastAPI ベースの実験管理 GUI バックエンド（React フロント付き）は検証中。ローカル用途を想定。

## 主な特徴
- 厳密な設定バリデーション: `src/astraia/config.py` の `OptimizationConfig` でメトリクス整合性や探索空間境界をチェック。
- 自動化された探索ループ: 停止条件・Pareto front・ハイパーボリュームを追跡し、ベスト試行を記録。
- 豊富な成果物: CSV/Markdown/PNG/JSON に加え、Git スナップショットと LLM 使用量ログを `runs/<run_id>/` に保存。
- LLM 連携: プランナー上書き、LLM ガイダンスによる候補生成、メタ要約、LLM クリティックでレポートを補強。
- 実行管理ユーティリティ: `astraia runs list|show|compare|diff|status` で履歴を操作し、`visualize` で履歴や Pareto front を即時描画。

## コンポーネント
- CLI (`astraia`): 設定検証、探索実行、成果物閲覧、実行履歴管理のエントリポイント。
- Evaluator / Config 群: qGAN、ZDT3、多目的 qGAN、量子回路評価などのテンプレートを同梱。
- Docs (`docs/`): 1 ページ 1 項目で手順・計画を整理。各ガイドは `docs/index.md` から辿れます。
- GUI バックエンド（実験的）: FastAPI + React。設定閲覧、dry-run、実行開始/キャンセル、LLM あり/なし比較を HTTP 経由で操作。
