# LLM あり/なし同時比較モードの使い方

このページでわかること:
- LLM を有効にした run と、LLM を無効にしたベースライン run を同時に起動する方法
- 比較レコード（メトリクス差分や成果物リンク）の確認場所
- 再現性を保つための seed や設定上書きのポイント

対象読者:
- LLM ガイダンスの効果を、同条件の古典（LLM 無効）実行と並べて評価したい人

## 前提
- Astraia の GUI バックエンド（FastAPI）を起動済み: `uvicorn main:app --port 8000 --reload`
- `.env` に LLM プロバイダのキーが入っていること（例: `OPENAI_API_KEY`）。`ASTRAIA_ENV_FILE` で別パスを指定可能。
- LLM 設定が含まれる YAML を使うこと（例: `configs/quantum/qaoa_llm_guided.yaml` や `configs/multiobj/zdt3_llm.yaml`）。`llm` ブロックがない設定では比較モードは拒否されます。

## 手順
1. **設定を確認する**  
   ```bash
   astraia --config configs/quantum/qaoa_llm_guided.yaml --summarize
   ```
   メトリクスや停止条件を把握しておきます。必要なら `--as-json` で検証済み設定を確認します。

2. **バックエンドを起動する**（未起動なら）  
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
   CORS を変えたい場合は `ASTRAIA_GUI_CORS_ORIGINS` を環境変数で指定します。

3. **LLM 比較モードで run を開始する**  
   REST API に `llm_comparison: true` を付けて POST します。`options.seed` を指定すると LLM 有効/無効の両方で同じ初期乱数が使われます（未指定の場合は config の `seed` → run_id 派生の順で決定）。
   ```bash
   curl -X POST http://localhost:8000/runs \
     -H "Content-Type: application/json" \
     -d '{
       "config_path": "configs/quantum/qaoa_llm_guided.yaml",
       "run_id": "qaoa_llm_pair",
       "llm_comparison": true,
       "perform_dry_run": true,
       "ping_llm": true,
       "options": {
         "max_trials": 24,
         "seed": 1234
       }
     }' | jq
   ```
   - レスポンスの `comparison` に LLM 側/ベースライン側の run_id と共通 seed が返ります。ベースライン run_id は自動で `<run_id>-no-llm` という形になります（重複があれば連番付与）。
   - React フロントエンドから実行する場合は「LLM 比較モード」チェックボックスを ON にして「Run を開始」を押すだけで同じ挙動になります。

4. **進行状況を確認する**  
   各 run_id で状態を見ます。`comparison.record_path` に比較レコードのパスが入っています。
   ```bash
   curl http://localhost:8000/runs/qaoa_llm_pair | jq '.meta.comparison'
   curl http://localhost:8000/runs/qaoa_llm_pair-no-llm | jq '.meta.status'
   ```
   両方が `completed` になると比較レコードが更新されます。

5. **比較結果を読む**  
   `runs/comparisons/<comparison_id>.json` にサマリがまとまります。メトリクス差分（LLM 有効 − 無効）が `summary.deltas` に、各 run の `log` / `report` へのパスも含まれます。
   ```bash
   jq '.summary' runs/comparisons/qaoa_llm_pair-llm-comparison.json
   # 例: { "shared_seed":1234, "llm_enabled":{...}, "llm_disabled":{...}, "deltas":{ "energy":-0.12, "depth":0.0 } }
   ```
   さらに詳細を見たい場合は `runs/<run_id>/log.csv` や `report.md` を個別に確認してください。CLI の比較表で見たい場合は `astraia runs compare --runs <llm> <baseline> --metric energy --stat best` も利用できます。

## 検証（成功判定）
- `runs/<run_id>/` と `runs/<run_id>-no-llm/` が両方生成され、`meta.json.artifacts.llm_comparison` が同じ比較レコードを指している。
- `runs/comparisons/<comparison_id>.json` の `status` が `completed` になり、`summary.deltas` にメトリクス差分が入っている。
- 各 run の `log.csv` / `report.md` にトライアル結果が記録されている。

## トラブルシュート
- 症状: リクエスト直後に 400/422 エラーになる  
  - 原因: config に `llm` ブロックがない、または `options.llm_enabled=false` を同時指定している。  
  - 対処: LLM 付き設定を使い、`llm_enabled` を無効化しない。
- 症状: ベースライン run が `failed` で比較が進まない  
  - 原因: LLM を切ったことで evaluator 側の依存に不整合が出ている（LLM 入力を必須にしているなど）。  
  - 対処: evaluator が LLM なしでも動くか確認し、必要なら config でデフォルト値やフォールバックを設定する。
- 症状: seed が毎回ズレる  
  - 原因: `options.seed` を指定していない場合は run_id から派生するため、run_id を変えると seed も変わる。  
  - 対処: 再現したいときは明示的に `options.seed` を指定する。

## 関連リンク
- `../index.md`
- `./gui_backend_usage.md`
- `../reference/sample_configs.md`
