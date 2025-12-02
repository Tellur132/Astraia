# Astraia マニュアル

Astraia のドキュメントを GitHub Pages（`docs/` 直下）でまとめたトップページです。Diátaxis に倣い、手順（How-to）/ 仕様（Reference）/ 背景（Explanations）/ 意思決定ログ（Decisions）に分けています。編集ルールは [manual_usage.md](manual_usage.md) を参照してください。

## セクション一覧（Diátaxis）

### How-to（具体的な手順）
- [セットアップとインストール](howto/setup_install.md) — 依存インストール、API キー設定、dry-run の確認手順。
- [CLI クイックスタート](howto/cli_quickstart.md) — 設定確認から実行、成果物閲覧までのひと通りの流れ。
- [Astraia 実験管理 GUI バックエンド利用ガイド](howto/gui_backend_usage.md) — FastAPI バックエンドの起動手順と主要エンドポイント、React フロントの開発方法。
- [LLM あり/なし同時比較モードの使い方](howto/llm_comparison_mode.md) — LLM と古典ベースラインを同じ seed で走らせ、差分サマリを確認する手順。
- [LLM ガイド付き QAOA 回路設計 実行手順](howto/qaoa_llm_runbook.md) — 依存インストール、実行コマンド、結果の見方。

### Reference（仕様・パラメータの一覧）
- [CLI リファレンス](reference/cli_reference.md) — 主要オプションと `runs` / `visualize` サブコマンドの一覧。
- [設定ファイルの構成](reference/configuration_structure.md) — `OptimizationConfig` が検証するセクションまとめ。
- [実行で生成される成果物](reference/run_artifacts.md) — `runs/<run_id>/` に保存されるファイルの役割。
- [サンプル設定とベンチマーク](reference/sample_configs.md) — qGAN/ZDT/量子回路など同梱 YAML の用途一覧。
- [LLM 向け Evaluator / Config サンプル](reference/llm_evaluator_config_examples.md) — LLM が参照しやすい evaluator テンプレートと YAML 設定例。
- [Quantum evaluator の詳細ガイド](reference/quantum_evaluator.md) — CircuitFidelityEvaluator などの使い方とノイズシミュレーションの設定方法。

### Explanations（背景・コンセプト）
- [Astraia 概要](explanations/overview.md) — フレームワークの狙い、現状の機能、主要コンポーネント。

### Decisions（計画・意思決定）
- [Astraia 実験管理 GUI 計画 v2](decisions/experiment_gui_plan_v2.md) — MVP のゴール、アーキテクチャ決定事項、ロードマップ。
- [実験管理・設定 GUI TODO / プラン](decisions/experiment_gui_todo.md) — 想定ユーザー、成功指標、UX/技術の TODO 一覧。
- [LLM駆動最適化フレームワーク — 現実路線プロジェクト計画書 v1.0](decisions/llm_optimization_framework_plan_v1.md) — フレームワーク全体の背景、スコープ、ロードマップ。
- [量子回路自動設計拡張 計画](decisions/quantum_circuit_autodesign_expansion_plan.md) — 量子回路自動設計の段階的ゴールとサブパッケージ構成案。

### Meta
- [マニュアルの使い方と運用ルール](manual_usage.md) — ドキュメントの書き方・探し方。
