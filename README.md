# Astraia MVP

Astraia は YAML 設定から Optuna ベースの探索ループを自動生成し、CSV / Markdown / PNG / LLM ログなどの成果物を残す LLM 駆動最適化フレームワークです。CLI は安定化済み、GUI バックエンドは実験的に提供しています。

## 特徴（概要）
- 設定バリデーション付きの探索ループ生成と成果物保存（`runs/<run_id>/`）。
- LLM ガイダンス / メタ探索 / LLM クリティックをオン/オフ切替可能。
- `runs` サブコマンドで実行履歴を管理し、`visualize` で履歴・Pareto front を描画。
- qGAN / ZDT / 量子回路などのテンプレート evaluator と YAML 設定を同梱。

## クイックスタート
1. 依存インストール
   ```bash
   pip install -e .
   # LLM を使うなら extras を追加: pip install -e .[openai] など
   ```
2. API キーを `.env` に設定し、疎通を確認
   ```bash
   cp .env.example .env
   echo "OPENAI_API_KEY=sk-..." >> .env
   astraia --dry-run
   ```
3. サンプルを実行
   ```bash
   astraia --config configs/qgan_kl.yaml
   astraia visualize --run-id qgan_kl_minimal --type history
   ```

## ドキュメント
- 全体の目次: `docs/index.md`
- 概要: `docs/guides/overview.md`
- セットアップ: `docs/guides/setup_install.md`
- CLI 手順: `docs/guides/cli_quickstart.md`, `docs/guides/cli_reference.md`
- サンプル設定と成果物: `docs/guides/sample_configs.md`, `docs/guides/run_artifacts.md`
- GUI バックエンド（実験的）: `docs/guides/gui_backend_usage.md`

## 状態メモ
- CLI と同梱 evaluator / config は MVP 相当の機能を提供中。
- LLM ガイダンスやメタ探索はオプション扱いで、環境に応じて無効化可能。
- FastAPI + React の GUI はローカル利用を想定した検証段階です。
