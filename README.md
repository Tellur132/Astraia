# Astraia MVP

Astraia は YAML で書いた設定から Optuna ベースの探索ループを自動生成し、CSV / Markdown / PNG / LLM ログなどの成果物を残す「LLM 駆動最適化フレームワーク」です。CLI は安定化済み、FastAPI + React の GUI バックエンドは実験的に提供しています。

> ドキュメントまとめ: `docs/index.md`（GitHub Pages ミラー: https://tellur132.github.io/Astraia/ ）

## できること
- 設定バリデーション付きの探索ループをそのまま実行し、`runs/<run_id>/` に成果物を保存。
- LLM ガイダンス / メタ探索 / LLM クリティック / プランナー（rule or LLM）をオン/オフ切替。
- `runs list/show/diff/compare/status/ab-template` で履歴管理、`visualize` で履歴・Pareto front を即時描画。
- qGAN / ZDT / 量子回路などの evaluator テンプレートと YAML サンプルを同梱。

## クイックスタート（最短パス）
1. 依存を入れる（LLM を使うなら extras を追加）
   ```bash
   pip install -e .
   pip install -e .[openai]   # OpenAI を使う場合
   ```
2. `.env` に API キーを書き、疎通を確認
   ```bash
   cp .env.example .env
   echo "OPENAI_API_KEY=sk-..." >> .env
   astraia --dry-run                 # config 検証 + LLM ping
   ```
3. サンプルを走らせて結果を眺める
   ```bash
   astraia --config configs/qgan_kl.yaml           # 単目的 qGAN
   astraia visualize --run-id qgan_kl_minimal      # history PNG を runs/<id>/ に生成
   astraia runs list                               # 実行履歴を一覧
   ```

## もう少し細かい使い方（例）
- LLM なしで確認したい: `astraia --config <file> --planner none` または config の `llm_guidance.enabled=false`。
- LLM あり/なし・init-only・mixed・full を一気に比較したい: `astraia runs ab-template --config <file> --seed 1234`。
- 複数 run の設定差分を見たい: `astraia runs diff --run-id <a> --run-id <b>`。
- マルチ目的の Pareto front を見たい: `astraia visualize --run-id <id> --type pareto --metric kl --metric depth`。

詳細な手順やトラブルシュートは `docs/howto/cli_quickstart.md` と `docs/howto/cli_usage_cookbook.md` を参照してください。

## 状態メモ
- CLI と同梱 evaluator / config は MVP 相当の機能を提供中。
- LLM ガイダンスやメタ探索はオプション扱いで、環境に応じて無効化可能。
- FastAPI + React の GUI はローカル利用を想定した検証段階です。
