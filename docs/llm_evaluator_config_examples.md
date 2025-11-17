# LLM 向け Evaluator / Config サンプル

LLM が evaluator や YAML 設定を自動生成する際に「どのような構造・コメントを入れればよいか」を即座に参照できるよう、
以下の 2 つのテンプレートを追加しました。

## 1. evaluator: `src/astraia/evaluators/llm_template.py`

```python
from astraia.evaluators.llm_template import create_evaluator
```

- `SimpleWaveEvaluator` は `__call__(params, seed)` だけを実装した dataclass で、BaseEvaluator を継承しなくても動作します。
- 4 ステップ（パラメタ取得 → 目的関数の計算 → 乱数ノイズ付与 → dict で返却）をコメントで明示し、LLM が同じ骨子を再利用しやすい構成にしています。
- `create_evaluator(config)` は YAML の `evaluator:` セクションをそのまま受け取り、1 行でインスタンス化できるため、LLM が任意の追加パラメタを拡張しやすくしています。
- 返却する dict のキー（`score`, `amplitude`, `frequency`, `phase`, `depth` など）はそのままレポートに掲載でき、必要に応じて増減可能です。

## 2. config: `configs/examples/llm_template.yaml`

```yaml
astraia --config configs/examples/llm_template.yaml --summarize
```

- 各セクションの冒頭に 1 文コメントを添えて「どこに何を書くか」を即説明できる形にしました。
- `search_space` の各パラメタにも短い用途メモを付与し、LLM が物理量や範囲を説明するときのテンプレートとして流用できます。
- `planner`/`llm`/`llm_guidance`/`meta_search`/`llm_critic` をすべて記述し、ON/OFF の切り替え例と最小限の文章を同居させています。
- `evaluator` セクションは上記テンプレートを指し示しつつ、`response_noise` のような任意パラメタを記述する例になっています。

## 使い方のヒント

1. evaluator を新規作成したい場合は `SimpleWaveEvaluator` のコメントを参考に、手順をほぼコピペで埋め替えると LL.M. でも破綻しにくくなります。
2. 設定ファイルは `llm_template.yaml` をベースに必要なセクションだけ残し、コメントを調整するだけで別の実験に転用できます。
3. CLI では `--summarize` で設定の整合性を確認し、`--dry-run` で `.env` のキーと LLM 疎通をチェックする運用を推奨します。
