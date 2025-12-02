# Astraia マニュアル

Astraia のドキュメントを GitHub Pages（`docs/` 直下）でまとめたトップページです。開発者・運用者が知りたい情報へ最短距離で辿れるよう、ガイドと計画資料を整理しています。編集ルールは [manual_usage.md](manual_usage.md) を参照してください。

## セクション一覧

### 概要・セットアップ
- [Astraia 概要](guides/overview.md) — フレームワークの狙い、現状の機能、主要コンポーネント。
- [セットアップとインストール](guides/setup_install.md) — 依存インストール、API キー設定、dry-run の確認手順。
- [マニュアルの使い方と運用ルール](manual_usage.md) — ドキュメントの書き方・探し方。

### CLI 利用
- [CLI クイックスタート](guides/cli_quickstart.md) — 設定確認から実行、成果物閲覧までのひと通りの流れ。
- [CLI リファレンス](guides/cli_reference.md) — 主要オプションと `runs` / `visualize` サブコマンドの一覧。
- [設定ファイルの構成](guides/configuration_structure.md) — `OptimizationConfig` が検証するセクションまとめ。
- [実行で生成される成果物](guides/run_artifacts.md) — `runs/<run_id>/` に保存されるファイルの役割。
- [サンプル設定とベンチマーク](guides/sample_configs.md) — qGAN/ZDT/量子回路など同梱 YAML の用途一覧。

### 量子/LLM ガイド・実行手順
- [LLM 向け Evaluator / Config サンプル](guides/llm_evaluator_config_examples.md) — LLM が参照しやすい evaluator テンプレートと YAML 設定例。
- [LLM ガイド付き QAOA 回路設計 実行手順](guides/qaoa_llm_runbook.md) — 依存インストール、実行コマンド、結果の見方。
- [Quantum evaluator の詳細ガイド](guides/quantum_evaluator.md) — CircuitFidelityEvaluator などの使い方とノイズシミュレーションの設定方法。

### GUI / 実験管理
- [Astraia 実験管理 GUI バックエンド利用ガイド](guides/gui_backend_usage.md) — FastAPI バックエンドの起動手順と主要エンドポイント、React フロントの開発方法。
- [Astraia 実験管理 GUI 計画 v2](plans/experiment_gui_plan_v2.md) — MVP のゴール、アーキテクチャ決定事項、ロードマップ。
- [実験管理・設定 GUI TODO / プラン](plans/experiment_gui_todo.md) — 想定ユーザー、成功指標、UX/技術の TODO 一覧。

### プロジェクト計画・背景
- [LLM駆動最適化フレームワーク — 現実路線プロジェクト計画書 v1.0](plans/llm_optimization_framework_plan_v1.md) — フレームワーク全体の背景、スコープ、ロードマップ。
- [量子回路自動設計拡張 計画](plans/quantum_circuit_autodesign_expansion_plan.md) — 量子回路自動設計の段階的ゴールとサブパッケージ構成案。

### 補足
- リポジトリ全体のセットアップとクイックスタートは `README.md` よりも各ガイド（上記リンク）を参照してください。
- Docs への新規追加・更新時は [manual_usage.md](manual_usage.md) を必ず確認し、`docs/index.md` の該当セクションにリンクを追加してください。
