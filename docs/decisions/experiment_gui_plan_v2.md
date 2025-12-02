# Astraia 実験管理 GUI 計画 v2

## 1. ゴール / 非ゴール

### 1.1 ゴール（MVP 時点）
- YAML/CLI 未経験者でも Astraia の実験を  
  「設定 → 検証 → 実行 → 状態確認 → 成果物閲覧」まで GUI で完結できる。
- 既存の `astraia` CLI と `OptimizationConfig`・`runs/<id>/...` をそのまま利用し、  
  **CLI 単体実行と完全互換**のワークフローを提供する。
- LLM キーや実行環境の安全性を損なわず、**ローカル環境完結**を前提とする
  （外部送信は LLM API のみ / プロキシ設定も考慮）。

### 1.2 非ゴール（MVP ではやらないこと）
- マルチユーザー対応（ワークスペース共有・権限管理）
- リモート実行クラスタ管理（別ホストへのジョブ投入）
- 高度な可視化ダッシュボード（大規模フロントエンドや BI ツール並みの機能）

これらは Phase 2 以降の拡張対象とする。

---

## 2. アーキテクチャ決定事項（MVP）

### 2.1 全体構成
- フロントエンド: React + Vite の SPA
- バックエンド: FastAPI（Python）  
  - 可能な限り **Python API 経由**で Astraia を呼び出す  
    - 例: `OptimizationConfig` の検証、`run_optimization` の起動
  - 必要に応じて CLI サブプロセスをラップ（`runs list|show|diff|compare`, `visualize` など）

### 2.2 プロセス・ジョブ管理
- 1 実行 = 1 ジョブとしてバックエンド内で管理
  - Python の `ThreadPoolExecutor` / `asyncio` で非同期実行
  - 実行ごとに `run_id` を払い出し、`runs/<run_id>/meta.json` を進捗のソースとして利用
- 停止処理
  - MVP ではまず「SIGINT 送信（Graceful 停止）」をサポート対象とする
  - ジョブテーブルに `state: running|completed|failed|cancelling` を持つ

### 2.3 設定ファイルポリシー
- 読み取り:
  - 既存の `configs/` をそのまま読み込み（CLI と同じファイルを扱う）
- 保存:
  - GUI 編集から保存されるファイルは **`configs/gui/` 以下に出す**（既存ファイルの破壊回避）
  - 元テンプレートのパスを `meta.json` や GUI のメタデータとして保持し、
    「どの config を元にしたか」を追跡できるようにする
- YAML 出力:
  - キー順・インデントを固定し、`git diff` が読みやすい形式にする

### 2.4 LLM キーの扱い
- `.env` 直接編集は **明示的な opt-in 操作**に限定
  - GUI 上のフォームは「ローカル保存のみ / Git 管理禁止」の警告を常に表示
- 読み取り:
  - UI 上ではマスク表示（`sk-****abcd` のような末尾のみ表示）
- 書き込み:
  - 書き換え可能なキーをホワイトリスト管理（`OPENAI_API_KEY`, `GEMINI_API_KEY` 等）

---

## 3. マイルストーン別ロードマップ

### M0: 基盤整備（1st スプリント）

**目的:** GUI バックエンド / フロントエンドの「土台」を作り、Astraia と最低限つながる状態にする。  

**完了条件 (DoD):**
- `uvicorn main:app` で起動できる FastAPI アプリがある
- `GET /health` が 200 を返す
- `GET /configs` が `configs/` 配下のファイル名一覧を返す
- フロントから `/health` と `/configs` を叩いて画面に一覧表示できる

**主なタスク:**
1. リポジトリ構成の決定（例: `src/astraia_gui/backend`, `frontend` ディレクトリ）
2. FastAPI スケルトン実装
   - `/health`
   - `/configs`（とりあえずファイル名一覧のみ）
3. React + Vite プロジェクトセットアップ
   - `ConfigsList` コンポーネントで `/configs` の結果を表示
4. 開発用コマンド整備
   - `make dev`（バックエンド + フロント同時起動）
   - or `poetry run` / `npm run dev` スクリプト

---

### M1: 設定ブラウザ & バリデーション（Config 周り MVP）

**目的:** 「config を選ぶ → 中身とサマリを GUI で確認 → `--summarize` / `--as-json` 相当が見える」状態にする。  

**完了条件:**
- GUI から config を選択すると、以下が見える:
  - YAML 本文（閲覧のみ）
  - 検証済み JSON 表現（`--as-json` 相当）
  - サマリ（メトリクス・探索空間・停止条件など）
- バックエンド側で `OptimizationConfig` 検証が行われ、エラー時は JSON で詳細エラーが返る

**主なタスク:**
1. バックエンド API 拡張
   - `GET /configs/{name}`: YAML 本文
   - `GET /configs/{name}/summary`: `astraia --summarize` or Python API で生成
   - `GET /configs/{name}/as-json`: 検証済 JSON
2. バリデーションエラーのフィールドマッピング
   - `OptimizationConfig.parse_file` のエラー → `path` / `msg` を抽出し JSON へ
3. フロント UX
   - 左ペイン: config 一覧（タグ/カテゴリ表示）
   - 右ペイン: サマリ・JSON・YAML Viewer のタブ
4. `.env` チェックの簡易実装
   - `GET /env/status`: 必須キーの有無を返す（OK/NG + 不足キー一覧）

---

### M2: 実行起動 & モニタリング（Run 周り MVP）

**目的:** 「GUI から dry-run → 実行開始 → 進捗 & 状態確認 → 完了後の成果物リンク」までを通す。  

**完了条件:**
- GUI から run_id を入力/自動生成して dry-run を実行できる
- dry-run 成功後にのみ実ジョブを開始できる
- 実行中はステータス（running/completed/failed）が一覧と詳細画面で見える
- 完了後、`runs/<run_id>/report.md` / `log.csv` へのリンクが GUI から辿れる

**主なタスク:**
1. バックエンド: 実行 API
   - `POST /runs/dry-run`: config 名 + 実行オプションを受けて dry-run
   - `POST /runs`: 本番実行開始（ジョブ登録 + 非同期開始）
   - `GET /runs`: `runs list` 相当の一覧
   - `GET /runs/{run_id}`: `runs show` + meta.json 内容
   - `POST /runs/{run_id}/cancel`: SIGINT 送信
2. モニタリング
   - ジョブテーブル（メモリ or 軽量 DB）で状態管理
   - `runs/<run_id>/meta.json` のポーリングで進捗更新
3. フロント UX
   - 実行ウィザード
     - Step1: config 選択
     - Step2: よく使うオプション（`max_trials`, サンプラー, LLM ON/OFF など）のフォーム
     - Step3: dry-run 実行 → 結果表示
     - Step4: 実行開始ボタン
   - 実行一覧画面
     - 状態フィルタ（running/completed/failed）
     - run_id / config / 開始日時 / 状態
   - 実行詳細画面
     - 基本メタデータ
     - ステータスと最新メトリクス（meta.json から）

---

### M3: 成果物ビュー & 比較機能（Phase 2 の一部）

**目的:** すでにある `runs` を GUI で探索し、log / report / 可視化を確認できるようにする。  

**完了条件:**
- GUI から任意の run を選択し、
  - `report.md` を Markdown としてレンダリング表示
  - `log.csv` をテーブル + 簡易グラフ（ベスト値推移）で表示
  - `log_history.png` / `log_pareto.png` が存在する場合はサムネイル表示
- 2 つ以上の run を選択して `runs diff` / `runs compare` 相当の結果要約を表示できる

**主なタスク:**
1. バックエンド: 成果物 API
   - `GET /runs/{run_id}/artifacts`: 利用可能な成果物一覧
   - `GET /runs/{run_id}/report`: report.md → HTML 変換して返す
   - `GET /runs/{run_id}/log`: log.csv の一部をページング / サマリ付きで返す
   - `GET /runs/compare`: run_id リスト + metric を受けて比較結果 JSON を返す
2. フロント UX
   - 成果物タブ: Report / Log / Plots のタブ切り替え
   - 比較ビュー:
     - 左側: run 一覧からチェックボックス選択
     - 右側: 指定メトリクスに対する best/median などの比較表
3. パフォーマンス
   - 大きな log.csv のときは「先頭 N 行」「末尾 N 行 + サマリ」のみ読み込む
   - 閾値（例: 10 万行）を超える場合は警告表示

---

### Phase 3: 拡張アイデア（バックログ）

- ワークスペース共有・権限管理（閲覧のみ / 実行可などのロール）
- リモート実行（別マシンの Astraia を HTTP 経由で叩く）
- LLM プロンプト編集 UX（planner/evaluator プロンプトを GUI から編集・バージョン管理）
- LLM コストダッシュボード（`llm_usage.csv` 集計 + コスト推計）

---

## 4. API 契約（MVP で最低限必要なもの）

例として、バックエンド側で最初に固めるべき主要エンドポイントだけ列挙:

- 設定関連
  - `GET /configs`  
    → `{ name, path, tags }[]`
  - `GET /configs/{name}`  
    → `{ yaml: string }`
  - `GET /configs/{name}/summary`  
    → `{ metrics: ..., search_space: ..., stop_criteria: ... }`
  - `GET /configs/{name}/as-json`  
    → `OptimizationConfig` の JSON
- 実行関連
  - `POST /runs/dry-run`
  - `POST /runs`
  - `GET /runs`
  - `GET /runs/{run_id}`
  - `POST /runs/{run_id}/cancel`
- 成果物関連
  - `GET /runs/{run_id}/artifacts`
  - `GET /runs/{run_id}/report`
  - `GET /runs/{run_id}/log`
- 環境関連
  - `GET /env/status`
  - `POST /env`（opt-in 時のみ有効）

---

## 5. リスクと対策（再整理）

- LLM キー漏洩
  - 書き込み対象キーをホワイトリスト化 + マスク表示 + 「Git 管理禁止」警告を徹底
- CLI 実行との競合
  - GUI 側での同時実行数を制限（例: デフォルト 1〜2 ジョブ）
  - ジョブ状態を `runs/` の meta.json に合わせて同期
- 成果物のサイズ
  - 初期表示はサマリのみ、詳細読み込みはユーザー操作時に限定
- 設定編集ミス
  - GUI 経由の実行は常に dry-run 強制
  - GUI 用 config と元テンプレートを分離しておく（`configs/gui/`）

---
