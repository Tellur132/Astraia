# CLI クイックスタート

このページでわかること:
- `astraia` CLI を初めて触る人が、設定確認 → 実行 → 成果物確認までを迷わず進める手順
- 実行後にどこを見ればよいか、トラブルになりやすいポイント

対象読者:
- Python/Optuna は触ったことがあるが Astraia は初めて、という人

## 前提
- Python 3.11 以上、`pip install -e .` で依存を導入済み（LLM を使うなら `pip install -e .[openai]` などの extras も）
- `.env` に必要な LLM API キーを入れてある（LLM を使わない設定なら空でも可）
- リポジトリ直下でコマンドを実行する想定

## 手順（最短コース）
1. **設定を確認する**
   ```bash
   astraia --config configs/qgan_kl.yaml --summarize
   astraia --config configs/qgan_kl.yaml --as-json   # 検証済み JSON 全体を確認
   ```
   - ここでエラーが出たら YAML の構造やメトリクス名を修正します。
2. **ドライランで環境チェック**
   ```bash
   astraia --config configs/qgan_kl.yaml --dry-run
   ```
   - `.env` の必須キー不足や LLM SDK の疎通エラーがわかります。
3. **探索を実行する**
   ```bash
   astraia --config configs/qgan_kl.yaml
   # python -m astraia.cli --config ... でも同じ
   ```
   - run_id は設定の `metadata.name` をスラッグ化したもの（例: `qgan_kl_minimal`）。重複すると `-02` などの連番が付きます。
4. **成果物を確認する**
   - 出力先: `runs/<run_id>/`（例: `runs/qgan_kl_minimal/`）
   - 主なファイル: `log.csv`, `report.md`, `summary.json`, `llm_usage.csv`, `llm_messages.jsonl`, `config_*`
   - グラフ生成（履歴 / Pareto）:
     ```bash
     astraia visualize --run-id qgan_kl_minimal --type history
     astraia visualize --run-id qgan_kl_minimal --type pareto --metric kl --metric depth
     # 既定で runs/<run_id>/visualization_history.png / visualization_pareto.png を作成
     ```
5. **実行履歴を眺める/比較する**
   ```bash
   astraia runs list                          # run_id/開始時刻/ベスト値/状態を一覧
   astraia runs show --run-id qgan_kl_minimal # メタ情報と resolved config を表示
   astraia runs compare --runs qgan_kl_minimal another_run --metric kl --stat best
   astraia runs diff --run-id qgan_kl_minimal --run-id another_run
   ```

## 追加シナリオ例
- **LLM を無効化して baseline を取りたい**: `astraia --config <file> --planner none` か、設定で `llm_guidance.enabled=false` / `llm_critic.enabled=false` にする。
- **LLM あり/なし・init-only・mixed・full をまとめて走らせたい**: `astraia runs ab-template --config <file> --seed 1234`。
- **成果物を別ディレクトリに置きたい**: すべての `runs` サブコマンドと `visualize` に `--runs-root <dir>` を付けると `runs/` の代わりにそのディレクトリを使う。

## 検証（成功判定）
- `runs/<run_id>/log.csv` に trial が記録され、`report.md` / `summary.json` が生成されている。
- `astraia visualize` 後に `visualization_history.png` が `runs/<run_id>/` に存在する。
- `astraia runs list` に該当 run_id が表示され、ステータスが `completed` になっている。

## トラブルシュート
- 症状: `--dry-run` で「Missing required secrets」エラーが出る
  - 対処: `.env` に `OPENAI_API_KEY`（または Gemini 用キー）を書き、`ASTRAIA_ENV_FILE` を使っている場合はパスを確認する。
- 症状: `visualize` で「metric_xxx が無い」エラー
  - 対処: `--metric` に `log.csv` に存在する列名を指定する。マルチ目的時は 2 つ以上を指定する。
- 症状: ベスト値が `nan` になる
  - 対処: evaluator が返す値を確認し、無効な trial をスキップするようにする。`llm_guidance` の `max_llm_trials` で LLM 投入回数を絞るのも有効。

## 関連リンク
- `../index.md`
- `../reference/cli_reference.md`
- `../reference/run_artifacts.md`
- `./cli_usage_cookbook.md`
