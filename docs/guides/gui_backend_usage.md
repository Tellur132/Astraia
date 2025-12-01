# Astraia 実験管理 GUI バックエンド利用ガイド

FastAPI ベースの「実験管理 GUI バックエンド」をローカルで立ち上げ、設定の閲覧・検証、dry-run、実行開始、進行中ジョブの監視/キャンセルを HTTP 経由で行うための手順です。curl / API クライアント / 独自フロントから呼び出せるほか、付属の React フロントエンドを合わせて使うこともできます。

## 前提

- `pip install -e .` で FastAPI / uvicorn を含む依存が入っていること。
- `.env` に LLM の API キー（例: `OPENAI_API_KEY`）を用意しておくこと。別名の env ファイルを使う場合は `ASTRAIA_ENV_FILE` で明示できます。
- 既定の設定ルートはリポジトリ直下の `configs/`。任意のパスに置いた場合は `ASTRAIA_CONFIG_ROOT=/path/to/configs` をセットしてください。

## 起動方法

リポジトリルートで uvicorn を起動します。開発中は `--reload` を付けるとホットリロードされます。

```bash
cd /path/to/Astraia
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

起動後は `http://localhost:8000/health` でヘルスチェック、`/docs` で自動生成 API ドキュメント（Swagger UI）が確認できます。

## できること（主要エンドポイント）

- 設定一覧/閲覧  
  - `GET /configs` : `configs/` 配下の YAML をパスとタグ（親ディレクトリ名）付きで列挙。  
  - `GET /configs/{path}` : YAML 本文を取得。  
  - `GET /configs/{path}/as-json` : `OptimizationConfig` で検証した JSON を返却（422 で検証エラー詳細）。  
  - `GET /configs/{path}/summary` : メトリクス・停止条件・探索空間などのサマリ。
  - `config_path` には `/configs` で返る `path` をそのまま使います（config ルートからの相対パス）。
- 環境チェック  
  - `GET /env/status` : OpenAI/Gemini など必須環境変数の有無をマスク付きで返却。
- 実行前検証 / 実行開始  
  - `POST /runs/dry-run` : 設定のバリデーションと LLM 疎通確認だけを実施。`run_id` 未指定の場合は設定名から自動生成。`options.max_trials` / `options.sampler` / `options.llm_enabled=false` で一時的な上書きが可能。  
  - `POST /runs` : 非同期で最適化ジョブを開始。既定で dry-run と LLM ping を先に実施（`perform_dry_run=false` や `ping_llm=false` で無効化）。`runs/<run_id>/` に成果物を生成し、レスポンスには run_dir と meta_path が含まれます。
- 実行管理  
  - `GET /runs?status_filter=running` : 既存実行をステータスで絞り込み一覧。  
  - `GET /runs/{run_id}` : `meta.json`（ステータス・タグ・アーティファクト）と `config_resolved.json` の内容、アクティブジョブなら PID/開始時刻/キャンセル要求有無を返却。  
  - `POST /runs/{run_id}/cancel` : 実行中プロセスへ SIGINT を送信。完了済みの場合はメッセージのみ返却。

### LLM あり/なし比較をまとめて走らせる

- `POST /runs` のペイロードに `llm_comparison=true` を付けると、LLM 機能を有効にした run と、`llm_enabled=false` のベースライン run を同時に開始します。両者は共通の seed（`options.seed` → config 記載の seed → run_id から自動生成）で固定し、初期状態を揃えます。
- レスポンスの `comparison` フィールドに LLM 側 / ベースライン側の run_id と共有 seed が入ります。両 run の `meta.json.artifacts.llm_comparison` には比較レコード（`runs/comparisons/<comparison_id>.json`）へのパスが自動で追記されます。
- 比較レコードには best_value / best_metrics・差分などのサマリが保存され、ログやレポートへのリンクも含まれます。片方が失敗した場合もステータスを残すので再現調査に使えます。

## 典型フロー例（curl）

1. 設定を探す  
   ```bash
   curl http://localhost:8000/configs | jq
   ```
2. サマリを確認  
   ```bash
   curl "http://localhost:8000/configs/qgan_kl.yaml/summary" | jq
   ```
3. LLM キーの状態を確認  
   ```bash
   curl http://localhost:8000/env/status | jq
   ```
4. dry-run で事前検証  
   ```bash
   curl -X POST http://localhost:8000/runs/dry-run \
     -H "Content-Type: application/json" \
     -d '{"config_path":"qgan_kl.yaml","options":{"max_trials":10}}' | jq
   ```
5. 実行を開始（バックグラウンド）  
   ```bash
   curl -X POST http://localhost:8000/runs \
     -H "Content-Type: application/json" \
     -d '{"config_path":"qgan_kl.yaml","run_id":"qgan_via_gui"}' | jq
   ```
   LLM 比較を走らせたい場合は `llm_comparison=true` を追加します。seed を固定したいときは `options.seed` で指定可能です。
6. 進行状況を確認  
   ```bash
   curl http://localhost:8000/runs/qgan_via_gui | jq
   ```
   `meta.status` が `running/completed/failed/cancelling` などで更新され、`job.state` でプロセス存否を確認できます。
7. キャンセルしたいとき  
   ```bash
   curl -X POST http://localhost:8000/runs/qgan_via_gui/cancel | jq
   ```
8. 終了後は `runs/qgan_via_gui/` に生成された `log.csv` / `report.md` などを CLI と同様に閲覧できます。

## フロントエンドを立ち上げる（React + Vite）

- 前提: Node.js 18 以上
- バックエンドとポートを分けて起動し、開発中は Vite の `/api` プロキシを利用します。

1. **バックエンド**（ターミナル A）
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   # CORS を変えたい場合:
   # ASTRAIA_GUI_CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173" uvicorn main:app --reload
   ```
2. **フロントエンド**（ターミナル B）
   ```bash
   cd frontend
   npm install           # 1 回だけ実施
   cp .env.example .env  # 必要に応じて API URL を上書き
   npm run dev           # http://localhost:5173 で起動（/api → :8000 にプロキシ）
   ```
3. 本番ビルドしたい場合は `npm run build` で `frontend/dist/` が生成されます。`VITE_API_BASE_URL` を `.env` に指定すると、ビルド済みバンドルが直接バックエンドのホストを叩くようになります。

## 注意事項

- 認証・認可は未実装です。ローカル環境か、信頼できるネットワーク内でのみ起動してください。
- `run_id` を明示しない場合は設定の `metadata.name` からスラッグ生成されます。同名ディレクトリが存在する場合は `-02` のように連番が付きます。
- `options.llm_enabled=false` を指定すると設定内の `llm` / `llm_guidance` / `llm_critic` がまとめて無効化されます。LLM を使わないベンチマーク確認に便利です。
