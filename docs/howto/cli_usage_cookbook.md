# CLI 使い方カタログ（初心者向け）

このページでわかること:
- Astraia CLI を初めて触る人向けに、典型シナリオ別の手順をまとめて確認できる
- LLM あり/なし、マルチ目的、履歴管理など複数のニーズに合わせた操作例

対象読者:
- 「まず何か 1 本回したい」「複数パターンを比べたい」など、具体的なゴールがある初学者

## 前提
- Python 3.11+、`pip install -e .` で依存導入済み（LLM を使うなら `pip install -e .[openai]` など）
- `.env` に必要な API キーを設定済み（LLM を使わないシナリオでは不要）
- リポジトリ直下でコマンドを実行する想定。成果物ルートは既定で `runs/`

---

## シナリオ 1: LLM を使わずにまず動かす（baseline 確認）
1. 設定を確認
   ```bash
   astraia --config configs/qgan_kl.yaml --summarize
   ```
2. LLM を無効にして実行（`--planner none` でガイダンスを切る）
   ```bash
   astraia --config configs/qgan_kl.yaml --planner none
   ```
3. 成果物を確認
   - `runs/qgan_kl_minimal/log.csv` に trial が入っている
   - `report.md` / `summary.json` が生成されている
4. 簡単なメモを残す（任意）
   ```bash
   astraia runs status --run-id qgan_kl_minimal --state completed --note "LLM 無効 baseline"
   ```

検証: `astraia runs list` に `qgan_kl_minimal` が `completed` として表示される。

---

## シナリオ 2: マルチ目的で Pareto front を描く
1. 多目的設定を選ぶ（例: ZDT3）
   ```bash
   astraia --config configs/multiobj/zdt3.yaml --summarize
   ```
2. 実行する
   ```bash
   astraia --config configs/multiobj/zdt3.yaml
   ```
3. Pareto front を画像化
   ```bash
   astraia visualize --run-id multiobj_zdt3 --type pareto --metric f1 --metric f2
   # 既定で runs/multiobj_zdt3/visualization_pareto.png を生成
   ```
4. ベスト値履歴も併せて確認
   ```bash
   astraia visualize --run-id multiobj_zdt3 --type history
   ```

検証: `runs/multiobj_zdt3/visualization_pareto.png` に 2 目的の散布図が描かれている。

---

## シナリオ 3: LLM あり/なし・init-only・mixed・full をまとめて回す
1. LLM 設定入りの YAML を選ぶ（例: `configs/quantum/qaoa_llm_guided.yaml`）
2. `.env` に LLM キーを入れ、`--dry-run` で疎通を確認
   ```bash
   astraia --config configs/quantum/qaoa_llm_guided.yaml --dry-run
   ```
3. 4 パターンを同じ seed で一括実行
   ```bash
   astraia runs ab-template --config configs/quantum/qaoa_llm_guided.yaml \
     --seed 1234 --init-trials 6 --mix-ratio 0.4
   ```
   - `no-llm` / `llm-init` / `llm-mixed` / `llm-full` の run_id が連続で作成され、表形式で結果が表示される。
   - 各 `runs/<run_id>/summary.json` にベスト値や LLM 呼び出し数が保存される。

検証: 出力された表に 4 行分の run が表示され、`runs/<run_id>/summary.json` が存在する。

---

## シナリオ 4: 実行履歴を比較・整理する
1. 一覧を絞り込む
   ```bash
   astraia runs list --status completed --sort best_value --reverse --limit 5
   astraia runs list --filter metadata.name=qgan --json
   ```
2. 設定差分を確認
   ```bash
   astraia runs diff --run-id qgan_kl_minimal --run-id qgan_kl_minimal-02
   ```
3. メトリクスを表で比較
   ```bash
   astraia runs compare --runs qgan_kl_minimal qgan_kl_minimal-02 --metric kl --stat best --stat median
   ```
4. 後で見返すためのタグやベスト値を付ける
   ```bash
   astraia runs status --run-id qgan_kl_minimal --state completed \
     --best-value 0.12 --tag objectives=kl --tag multi_objective=false
   ```
5. 古い run を消す前に dry-run で確認
   ```bash
   astraia runs delete --run-id qgan_kl_minimal-02 --dry-run
   astraia runs delete --run-id qgan_kl_minimal-02 --yes
   ```

検証: `runs list` で絞り込みやタグ付けが反映され、不要な run が削除されている。

---

## 共有のトラブルシュート
- 症状: `Missing required secrets` が出る
  - 対処: `.env` に必要なキーを追加し、`ASTRAIA_ENV_FILE` を設定している場合はパスの存在を確認。
- 症状: ab-template で「llm_guidance が必要」と怒られる
  - 対処: 基本設定に `llm` と `llm_guidance` ブロックが入っているか確認する。LLM 無効の設定は ab-template では使えない。
- 症状: Pareto 描画時に「metrics が足りない」と出る
  - 対処: `--metric` を 2 つ以上渡す。`log.csv` にある列名と一致しているか確認する。

## 関連リンク
- `./cli_quickstart.md`
- `../reference/cli_reference.md`
- `../reference/run_artifacts.md`
- `../reference/sample_configs.md`
