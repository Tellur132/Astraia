# CLI クイックスタート

このページでは CLI で標準的な探索を一通り流す手順を示します。設定ファイルは `configs/` から任意に選択できます。

## 前提
- `pip install -e .` などで依存を導入済み。
- `.env` に必要な API キーを設定済み（LLM を使わない設定の場合は不要）。

## 手順
1. **設定のサマリを確認**
   ```bash
   astraia --config configs/qgan_kl.yaml --summarize
   astraia --config configs/qgan_kl.yaml --as-json    # 検証済み JSON を確認
   ```
2. **ドライランで疎通チェック**
   ```bash
   astraia --config configs/qgan_kl.yaml --dry-run
   ```
3. **探索を実行**
   ```bash
   astraia --config configs/qgan_kl.yaml
   # あるいは python -m astraia.cli --config configs/qgan_kl.yaml
   ```
4. **成果物を確認**
   - 出力先: `runs/<run_id>/`（例: `runs/qgan_kl_minimal/`）
   - 主なファイル: `log.csv`, `report.md`, `llm_usage.csv`, `comparison_summary.json`, `log_history.png`, `log_pareto.png`
5. **実行管理・可視化**
   ```bash
   astraia runs list
   astraia runs show --run-id qgan_kl_minimal
   astraia runs compare --runs qgan_kl_minimal another_run --metric kl --stat best

   astraia visualize --run-id qgan_kl_minimal --type history
   astraia visualize --run-id qgan_kl_minimal --type pareto --metric kl --metric depth
   ```

## コツ
- LLM を無効化したい場合は CLI に `--planner none` を付けるか、設定内の `llm_guidance`/`meta_search` を無効化してください。
- マルチ目的探索では `--metric` を複数指定し、Pareto front の PNG を `visualize --type pareto` で生成すると理解が早まります。
