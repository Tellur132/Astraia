# セットアップとインストール

このページでは Astraia をローカルで動かすための最小限の準備手順をまとめます。

## 前提
- Python 3.11 以上。
- 仮想環境（`venv` / `conda` など）が有効になっていること。
- 必要に応じて LLM API キー（例: `OPENAI_API_KEY`）を用意。

## インストール手順
1. リポジトリを取得し、仮想環境を有効化します。
2. 依存関係をインストールします。用途に応じて extras を追加できます。

   ```bash
   pip install -e .
   pip install -e .[openai]    # OpenAI 連携が必要な場合
   pip install -e .[gemini]    # Gemini 連携が必要な場合
   pip install -e .[llm]       # OpenAI + Gemini 両方
   pip install -e .[quantum]   # Qiskit + Aer を含む量子実験向け
   ```

3. `.env.example` をコピーして API キーを設定します。

   ```bash
   cp .env.example .env
   echo "OPENAI_API_KEY=sk-..." >> .env
   ```

4. LLM の疎通と必須キーを確認します。

   ```bash
   astraia --dry-run
   ```

5. 任意: 単体テストで環境整合性を確認します。

   ```bash
   pytest
   ```

## 追加メモ
- GUI バックエンドを使う場合は Python 依存のみで起動できます。React フロントを開発する場合は Node.js 18+ を別途用意してください。
- `ASTRAIA_CONFIG_ROOT` で設定ファイルディレクトリを差し替えられます。`planner_prompts/` などのパスは既定のままでも動作します。
