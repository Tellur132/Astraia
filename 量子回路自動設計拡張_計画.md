# 1. 目的・ゴール

## 1.1 技術的な最終イメージ

* Astraia を「**LLM 駆動最適化フレームワーク**」として維持しつつ、その一アプリケーションとして

  * 量子回路の設計／最適化／多目的評価（忠実度・深さ・ゲート数など）
  * LLM による回路案・アンサッツ・書き換えルールの提案
  * 古典シミュレータを用いた厳密な評価
  * 評価結果のフィードバックを受けた LLM の逐次改善
    を実現する。

## 1.2 マイルストーン的ゴール

段階的に到達すべき目標を定めます。

1. **MVP-1（LLM なし）**

   * Qiskit 等を使って「小さな量子回路の評価器」を Astraia に統合
   * 例：2〜4 qubit の QAOA / QFT / 簡単なターゲットユニタリの忠実度を評価する Evaluator
   * 既存の `search_space`＋Optuna でパラメタ最適化が回る状態にする。

2. **MVP-2（LLM 単発生成）**

   * LLM に「回路案（QASM または Qiskit コード）」を出させ、それを Evaluator がチェックしてスコアを返す流れを実装。
   * 無効なコードや物理的におかしい回路をペナルティ付きで扱えるようにする。

3. **MVP-3（LLM フィードバックループ）**

   * `llm_guidance` や `meta_search` を活用して

     * 回路案生成 → 評価 → 履歴サマリ → LLM へのフィードバック
       のループを構築。
   * Astraia の枠組みのまま「回路自動設計ループ」が完成している状態。

4. **MVP-4（多目的最適化・可視化）**

   * 量子回路専用の 2〜3 目的（忠実度 vs 深さ／T-count 等）設定を複数用意。
   * `astraia visualize --type pareto` で量子回路の Pareto front を見られるようにする。

5. **安定版**

   * `configs/quantum/` 以下にいくつか代表的タスク（QFT, 小さなユニタリ合成, QAOA アンサッツ設計）を揃える。
   * README / docs に「Quantum モジュール」の章を追加。
   * テストも揃えて、ZDT3 や qGAN と同列の「標準ベンチマーク」として扱えるようにする。

---

# 2. 全体アーキテクチャ方針

## 2.1 サブパッケージ構成

* 新規ディレクトリ：`src/astraia/quantum/`

  * 例：

    * `__init__.py`
    * `circuit_models.py`（回路の内部表現）
    * `evaluators.py`（QuantumCircuitEvaluator 群）
    * `llm_agents.py`（量子回路用 LLM プランナー／ガイド）
    * `prompts/`（量子回路専用プロンプト）

→ 後で切り出したくなったときに、このディレクトリごと別パッケージにしやすくする。

## 2.2 依存関係の扱い

* `extras_require` に量子用オプションを追加（例：`[quantum]`）

  * 中身：`qiskit` または `qiskit-terra`、`numpy` など
* コア部分（Optuna／LLM 辺り）は現状のまま。量子関連はインポート時に try/except して、未インストールならわかりやすくエラーを返す。

## 2.3 Config スキーマの拡張

* `OptimizationConfig` に「量子問題」のメタ情報を持たせる案：

```yaml
metadata:
  name: qft_synthesis_small
  description: "4-qubit QFT circuit synthesis with fidelity-depth tradeoff"

problem:
  domain: quantum_circuit          # new
  task: synthesis                  # synthesis | ansatz_search | oracle
  n_qubits: 4
  target: qft                      # or "unitary", "truth_table" など
  backend: qiskit                  # 将来的に cirq/pennylane 切り替えも想定

evaluator:
  module: asatraia.quantum.evaluators
  callable: create_qft_synthesis_evaluator
```

* 既存の `search`, `stopping`, `llm`, `llm_guidance`, `meta_search`, `llm_critic` はそのまま活かしつつ、「quantum 用 evaluator／planner」を差し替えて使うイメージ。

---

# 3. 段階的ロードマップ（詳細）

## Phase 0：準備・設計整理

### やること

1. **現行 Astraia の設計を再確認**

   * Evaluator の抽象クラス (`BaseEvaluator`)
   * `llm_guidance` / `meta_search` / `llm_critic` のインタフェース
   * `OptimizationConfig` の schema

2. **quantum サブパッケージのひな形作成**

   * `astraia/quantum/__init__.py`
   * 空の `evaluators.py`, `circuit_models.py`, `llm_agents.py`

3. **依存追加**

   * `pyproject.toml` or `setup.cfg` に `[quantum]` extras

### この段階のゴール

* コード構造として「量子ドメインの置き場」ができている。
* まだ回路は動かなくてよい。

---

## Phase 1：LLM なしの量子 evaluator 統合（MVP-1）

### 1-1. Circuit 内部表現の決定

候補：

* **パターンA：Qiskit の `QuantumCircuit` を内部標準にする**

  * シンプルで実装速い。
~~* **パターンB：自前の薄いラッパー（Gate のリスト）**~~

~~  * 将来 Cirq/PennyLane にも対応しやすい。~~

最初は A（Qiskit そのまま）でよいと思います。

### 1-2. シンプルな Evaluator を実装

例：`QAOAEvaluator`（Ising / MaxCut の小さな例）か、`QFTFidelityEvaluator`。

* 入力：`params`（連続パラメタ、離散パラメタ）

  * 例：

    * `n_layers`: 1〜3（int）
    * `gamma_i`, `beta_i`: 各レイヤーの角度
* 評価フロー：

  1. `params` からアンサッツ構造（QAOA / QFT）を構築
  2. Qiskit のシミュレータで状態を生成
  3. 目的に応じて

     * 忠実度（ターゲット状態との inner product）
     * 期待値（<H>）
     * 回路深さ・ゲート数
       を計算。
  4. `{"metric_fidelity": ..., "metric_depth": ..., "metric_gate_count": ...}` のような dict で返す。

### 1-3. YAML 設定の追加

例：`configs/quantum/qaoa_small.yaml`

```yaml
metadata:
  name: qaoa_small
  description: "2-qubit QAOA on small Ising instance"

search:
  metrics:
    - name: energy
      direction: minimize
    - name: depth
      direction: minimize

search_space:
  int:
    n_layers:
      low: 1
      high: 3
  float:
    gamma_0:
      low: 0.0
      high: 3.14
    beta_0:
      low: 0.0
      high: 3.14
  # ...必要に応じて gamma_1, beta_1 など

evaluator:
  module: astraia.quantum.evaluators
  callable: create_qaoa_evaluator
```

### この段階での「嬉しいこと」

* Astraia のインフラ（Optuna、ログ、レポート、Pareto 可視化）をそのまま使って、量子回路ベースの実験ができるようになる。
* LLM なしで動く「量子ベンチマーク」が 1 つ追加され、動作確認・性能比較の土台ができる。

---

## Phase 2：LLM 単発生成による回路候補の評価（MVP-2）

ここから「LLM に回路案を出させて評価する」流れを作ります。

### 2-1. 回路候補の外部表現

LLM とやり取りするための表現：

* 推奨：**OpenQASM 3 か Qiskit スタイルの Python コードを「コードブロックで」生成**させる。
* 例（QASM）：

```qasm
OPENQASM 3;
qubit[2] q;
h q[0];
cx q[0], q[1];
```

Astraia 側の Evaluator では：

1. `params["circuit_code"]`（文字列）を受け取る
2. Qiskit でパース（try/except）
3. 成功時：

   * 指定されたターゲット（ユニタリ or truth table）に対して忠実度を計算
   * 深さ・ゲート数を算出
4. 失敗時：

   * fidelity=0 などの強いペナルティ＋`metric_valid=0` のようなフラグ

### 2-2. `search_space` と LLM ガイダンスの橋渡し

素直なやり方：

* `search_space` に「ダミーの 1 次元 categorical パラメタ」を置く：

```yaml
search_space:
  categorical:
    circuit_code:
      choices: ["<LLM_FILLED>"]  # 実際はこの値は使わない
```

* `llm_guidance` で

  * `circuit_code` の値を無視して、LLM プロンプトから直接サンプルを生成
  * 生成した文字列を `params["circuit_code"]` に上書き
* Optuna から見ると「文字列 1 個のパラメタ」に見えるので、履歴やログは今まで通り扱える。

※ 将来きれいにするなら、「LLM でのみ値を埋めるパラメタ型」を `config.py` 側に追加する余地あり。

### 2-3. 量子回路用 LLM プロンプトの設計

`planner_prompts/quantum_circuit_planner.md` のようなテンプレを追加：

* 入力情報：

  * ターゲットの説明（例：2-qubit SWAP/unitary、truth table）
  * 使用可能なゲート集合
  * トポロジ（隣接制約）
  * 最大ゲート数／深さ上限
* 出力フォーマット：

  * 必ず QASM / Qiskit コードを 1 つの `code` ブロックで
  * 解説はあってもよいが、Evaluator はコード部分だけを抽出

LLM の役割（この段階）：

* 単発で「それっぽい回路候補」を出す
* 簡単な小規模問題であれば、既知の標準解に近いものを生成してくれるはず

### この段階の「嬉しいこと」

* 「仕様（ターゲットユニタリや truth table）＋制約 → LLM が回路案 → Astraia で評価」という基本ループができる。
* LLM の出力がどれくらい物理的／構文的にまともか、エラー率がどれくらいか、という実験もできる。

---

## Phase 3：LLM フィードバックループの実装（MVP-3）

### 3-1. 評価結果のサマリを LLM に返す

* `meta_search` or 専用の「quantum_meta_agent」を使って、

  * 最近 N 個のトライアルの

    * fidelity / depth / gate_count
    * 失敗率（構文エラーなど）
      を要約。
* LLM に対して：

  * 「どのように回路構造を変えるべきか」
  * 「ゲート数を増やすべきか減らすべきか」
  * 「別のアーキテクチャ（例：チェーン vs 全結合）を試すべきか」
    を自然言語で問いかける。

### 3-2. LLM に「次の探索戦略」を提案させる

* 出力例：

  * 「次の 20 トライアルでは〇〇型のアンサッツを試せ」
  * 「ゲート数の上限を 10 → 15 に増やせ」
  * 「CNOT の位置を変える規則を試せ」

これを `llm_guidance` の内部状態や `search_space` の一時的な制約変更に反映させる。

### 3-3. ループ構造

1. `llm_guidance`：現在の方針に基づいて回路案サンプルを生成
2. Evaluator：回路を評価
3. 一定ステップごとに `meta_search`：

   * 履歴をサマリして LLM に投げる
   * 改善方針を受け取って、`llm_guidance` のプロンプト／制約をアップデート

→ Astraia がもともと持っている「メタ探索」機能に、quantum 用のロジックを差し込むイメージ。

### この段階の「嬉しいこと」

* 「LLM がただ回路を吐くだけ」から一歩進んで、

  * **探索戦略そのものを LLM にいじらせる**ことができる。
* Astraia の「LLM プランナー＋Optuna」の構造と完全に整合した「量子版 Astraia」になる。

---

## Phase 4：多目的最適化・ベンチマーク整備（MVP-4）

### 4-1. 多目的指標の設計

代表的な組み合わせ：

* fidelity vs depth
* fidelity vs gate_count
* fidelity vs T_gate_count（fault-tolerant 目線）
* depth vs エラー確率（ノイズモデルと組み合わせ）

### 4-2. Config テンプレートの追加

例：`configs/quantum/qft_fidelity_depth.yaml`

```yaml
search:
  metrics:
    - name: fidelity
      direction: maximize
    - name: depth
      direction: minimize

evaluator:
  module: astraia.quantum.evaluators
  callable: create_qft_synthesis_evaluator
```

実行後：

* `astraia visualize --type pareto --metric fidelity --metric depth` で
  「忠実度と深さのトレードオフ」を散布図で見られるようにする。

### この段階の「嬉しいこと」

* ZDT3 や qGAN と同様に、「量子回路」という新しいドメインでの多目的最適化結果を、他の問題と同一フレームで比較できる。
* 「LLM あり／なし」「ヒューリスティクス違い」などを `runs compare` で統一的に比較できる。

---

## Phase 5：ドキュメント・テスト・拡張余地

### 5-1. README / docs の拡張

* README に「Quantum Circuit Design」のセクションを追加：

  * セットアップ方法（`pip install -e .[quantum]`）
  * 代表的な config の使い方
  * 生成されるレポートの見方
* `docs/` 以下に、量子 evaluator 向けの解説（`llm_evaluator_config_examples.md` と似たノリで）を追加。

### 5-2. テスト

* ユニットテスト：

  * Evaluator が小さい回路で合理的な値を返すか（例：SWAP の忠実度が高い etc.）
  * 不正な QASM が来たときに安全に失敗ペイロードを返すか
* プロパティベーステスト（あると嬉しい）：

  * ランダムな小回路に対して

    * 「回路→行列→回路再構成」の一貫性など

### 5-3. 将来の拡張アイデア

* Qiskit 以外の backend（Cirq, PennyLane）への対応
* ノイズ付きシミュレーション（NISQ デバイス模倣）
* LLM に ZX-calculus 的な書き換えを学習させる実験
* 「量子オラクル自動合成」や「可逆回路合成」への拡張
* Astraia 内の他の問題（ZDT3 など）と並べて、「LLM がどのタイプの問題に強いか」の比較研究
