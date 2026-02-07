# IgakuQA 評価プロジェクト 申し送り

**最終更新**: 2026-02-08（セッション6終了時）
**実行環境**: Mac Studio M3 Ultra (192GB RAM)
**GitHub**: https://github.com/aki-wada/IgakuQA-evaluation

---

## 1. プロジェクト概要

ローカルLLM（LM Studio経由）で日本の医師国家試験（IgakuQA）を解かせ、モデル性能を比較するベンチマーク評価プロジェクト。

- **評価対象**: 第116回医師国家試験（2022年）A問題 75問（メイン）、全セクション400問（medgemma評価）
- **合格ライン**: セクションA単独 75%（56/75問）、全セクション総合 75%（300/400問）
- **評価済みモデル数**: 33モデル（うち合格12モデル、評価失敗8モデル）+ medgemma-27b/qwen3-32b全セクション評価完了
- **最高スコア**: gpt-oss-120b MLX 8bit = **92.0%**（案A, mt=1024）

---

## 2. ファイル構成

```
IgakuQA/
├── evaluate_prompt_comparison.py  # メイン評価スクリプト（v2）
├── evaluate_lmstudio_batch.py     # バッチ評価スクリプト（v1）
├── evaluate_llm.py                # 初期評価スクリプト
├── plot_size_vs_accuracy.py       # 可視化スクリプト（3種のプロット生成）
├── test_glm_settings.py           # glm-4.7-flash診断スクリプト
├── EVALUATION_PROGRESS.md         # 全評価結果・考察・手順の主要ドキュメント
├── HANDOVER.md                    # 本ファイル（申し送り）
├── data/
│   ├── 2018/ ～ 2022/             # 各年度の試験データ（JSONL）
│   └── 2022/116-{A..F}.jsonl      # 2022年 セクションA～F
├── results/                        # 評価結果JSON
│   ├── prompt_comparison_*.json   # v2プロンプト比較結果（33ファイル）
│   ├── medgemma-27b_*.json        # medgemma全セクション結果（12ファイル）
│   ├── qwen3-32b_fewshot_2022_{A..F}.json  # qwen3-32b全セクション結果
│   └── *_evaluation.json          # v1評価結果
├── plots/                          # 生成プロット
│   ├── size_vs_accuracy_scatter.png
│   ├── scaling_analysis.png
│   ├── pareto_and_budget.png
│   └── memory_efficiency.png
├── scripts/prompts/prompt.jsonl   # Few-shot用例題（第100回問題）
├── venv/                           # Python仮想環境（Python 3.9）
└── requirements.txt
```

---

## 3. 評価スクリプトの使い方

### 基本コマンド（v2プロンプト比較）

```bash
source venv/bin/activate

python evaluate_prompt_comparison.py \
  --model "モデルID" \
  --year 2022 \
  --section A
```

### 重要パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| `--model` | 必須 | LM Studioにロード済みのモデルID |
| `--year` | 2022 | 試験年度（2018-2022） |
| `--section` | A | セクション（A-F） |
| `--prompts` | 全4種 | テストするプロンプト（baseline format_strict chain_of_thought japanese_medical） |
| `--limit` | None | 問題数制限（デバッグ用） |
| `--use-few-shot` | OFF | 2-shot例の追加 |
| `--output` | 自動生成 | 出力ファイルパス |

### max_tokens設定（evaluate_prompt_comparison.py内のPROMPTS辞書）

```
デフォルト値（通常モデル用）:
  baseline:          50
  format_strict:     50
  chain_of_thought: 200
  japanese_medical:  50
```

**重要**: reasoningモデル（gpt-oss系）やqwen3-next-80b、medgemma-27b（thinking mode対策）では、max_tokensの変更が必要。詳細は後述。

### スクリプトの現在の状態

**注意: evaluate_prompt_comparison.py の max_tokens は現在すべて 512 に設定されている。**
medgemma-27b評価のために変更済み。通常モデルの評価前にデフォルト値に戻す必要がある。

```python
# 現在の値（要復元）
"baseline":          {"max_tokens": 512},  # デフォルト: 50
"format_strict":     {"max_tokens": 512},  # デフォルト: 50
"chain_of_thought":  {"max_tokens": 512},  # デフォルト: 200
"japanese_medical":  {"max_tokens": 512},  # デフォルト: 50
```

### qwen3モデルの自動対応

- モデル名に`qwen3`が含まれる場合、システムプロンプト末尾に`/no_think`を自動追加
- Thinking ON は性能を大幅に低下させるため、`/no_think`が正解

---

## 4. 主要な発見・知見

### max_tokensの影響（最重要）

- **gpt-ossシリーズ**: reasoningモデルのため内部推論がトークンを消費。mt=50では空回答が多発（0-33%）、mt=1024で解消（77-92%）
- **qwen3-next-80b**: mt=50でBaseline 66.7%と低く、案B(mt=200)は0%。mt=1024で案C 85.3%達成
- **medgemma-27b**: `<unused94>thought`タグで思考モード発動。mt=50ではセクションB-Fで大量の空回答。mt=512で解消
- **通常モデル**（qwen3, llama, gemma等）: mt=50で十分な性能を発揮

### プロンプト依存性の傾向

| モデルタイプ | 最適プロンプト | 備考 |
|-------------|---------------|------|
| 大規模(27B+) | Baseline | シンプルが最善 |
| 中規模(14-30B) | 案B (段階的思考) | CoTが有効 |
| 小規模(<10B) | 案A/案C | 回答形式の制約が有効 |
| reasoningモデル | 案A or 案C (mt=1024必須) | max_tokensが支配的 |

### スケーリング効果

- **qwen3**: 4B(54.7%) → 8B(61.3%) → 14B(73.3%) → 32B(80.0%) → VL-32B(82.7%) → Next-80B-A3B(85.3%) → 235B(88.0%) — 明確なスケーリング
- **gpt-oss**: 20B(77.3%) → 120B(92.0%) — +14.7%
- **Swallow FT**: 8B(53.3%) → 70B(81.3%) — +28%
- **Mistral**: small(76.0%) → Large(77.3%) — +1.3%のみ、コスパ悪い

### 日本語ファインチューニング

- **Swallow**: llama-3.3-70b(68.0%) → Swallow-70b(81.3%) = **+13.3%**（大幅改善）
- **Shisa**: llama-3.3-70b(68.0%) → Shisa-v2-70b(61.3%) = **-6.7%**（逆効果）
- **EZO**: gemma-3-12b(54.7%) → EZO-12b(60.0%) = +5.3%
- FT手法により結果が大きく異なる

---

## 5. medgemma-27b 全セクション評価（進行中）

### 評価済み結果

#### 1回目: mt=50, Baseline（セクションA既存 + B-F新規）

| セクション | 問題数 | 正答数 | 正答率 | 空回答数 | 備考 |
|-----------|--------|--------|--------|---------|------|
| A | 75 | 57 | 76.0% | 0 | 既存結果（prompt_comparison で評価済み） |
| B | 50 | 10 | 20.0% | 20 | thinking mode多発 |
| C | 75 | 24 | 32.0% | 36 | thinking mode多発 |
| D | 75 | 15 | 20.0% | 48 | thinking mode多発 |
| E | 50 | 26 | 52.0% | 0 | 比較的安定 |
| F | 75 | 21 | 28.0% | 33 | thinking mode多発 |
| **合計** | **400** | **170** | **42.5%** | **137** | **不合格** |

**根本原因**: medgemmaが `<unused94>thought` タグで英語の思考モードに入り、mt=50ではトークンが足りず回答を出力できない。

#### 2回目: mt=512, Baseline改変（「思考過程の出力は不要です。回答のみを出力してください。」追加）

| セクション | 問題数 | 正答数 | 正答率 | 空回答数 | 前回比 |
|-----------|--------|--------|--------|---------|--------|
| A | 75 | 48 | 64.0% | 0 | -12.0% |
| B | 50 | 37 | 74.0% | 0 | +54.0% |
| C | 75 | 51 | 68.0% | 0 | +36.0% |
| D | 75 | 52 | 69.3% | 0 | +49.3% |
| E | 50 | 36 | 72.0% | 0 | +20.0% |
| F | 75 | 47 | 62.7% | 0 | +34.7% |
| **合計** | **400** | **271** | **67.8%** | **0** | **+25.2%** |

**結果**: 不合格だが大幅改善（+25.2%）。セクションAはプロンプト改変により逆に精度低下（-12%）。

**注意**: 2回目はBaselineのプロンプト文とmax_tokensの両方を同時に変更したため、どちらが効いたか切り分けできていない。

#### 3回目: mt=512, オリジナル4プロンプト（few-shotなし）— セクションA

**評価日**: 2026-02-07（セッション3）
**目的**: プロンプト文をオリジナルのまま、max_tokensのみ512に変更。

| Prompt | 正答率 | 空回答 | 平均応答時間 | 状態 |
|--------|--------|--------|------------|------|
| Baseline | 17.3% (13/75) | 多数 | ~15s | 完了 |
| 案A | 57.3% (43/75) | 0 | ~0.9s | 完了 |
| 案B | — | 多数 | ~15s | 4問で中断 |
| 案C | — | — | — | 未実行 |

**発見**: 案Aのプロンプト（「アルファベットのみで回答」「余計な説明は不要」）が思考モードを完全に抑制。Baselineと案Bは思考モードが多発。

#### 4回目: mt=512, 全4プロンプト + Few-shot — セクションA

**評価日**: 2026-02-07（セッション3）
**条件**: オリジナル4プロンプト + 2-shot few-shot + max_tokens=512

| Rank | Prompt | 正答率 | 空回答 | 平均応答時間 |
|------|--------|--------|--------|------------|
| 1 | **Baseline + few-shot** | **76.0%** (57/75) | 0 | 1.0s |
| 2 | 案C + few-shot | 74.7% (56/75) | 0 | 1.0s |
| 3 | 案A + few-shot | 70.7% (53/75) | 0 | 0.9s |
| 4 | 案B + few-shot | 56.0% (42/75) | 0 | 5.3s |

**最重要発見**: **Few-shotがmedgemmaの思考モード抑制に決定的に有効**
- Baseline: few-shotなし17.3% → few-shotあり**76.0%**（+58.7%）
- 全プロンプトで空回答が0に

#### 5回目: mt=512, Baseline + Few-shot — 全セクション（最終評価）

**評価日**: 2026-02-07（セッション3）
**条件**: Baseline + 2-shot few-shot + max_tokens=512

| セクション | 問題数 | 正答数 | 正答率 | 空回答数 | 2回目比 |
|-----------|--------|--------|--------|---------|--------|
| A | 75 | 57 | **76.0%** | 0 | +12.0% |
| B | 50 | 41 | **82.0%** | 0 | +8.0% |
| C | 75 | 46 | 61.3% | 0 | -6.7% |
| D | 75 | 57 | **76.0%** | 0 | +6.7% |
| E | 50 | 38 | **76.0%** | 0 | +4.0% |
| F | 75 | 48 | 64.0% | 0 | +1.3% |
| **合計** | **400** | **287** | **71.8%** | **0** | **+4.0%** |

**総合判定: 不合格（287/400 = 71.8%, 合格ライン75% = 300/400, あと13問）**

**medgemma-27b全評価の推移**:
- 1回目（mt=50, no few-shot）: 42.5%
- 2回目（mt=512, 改変プロンプト）: 67.8%（+25.2%）
- **5回目（mt=512, Baseline+few-shot）: 71.8%（+4.0%）**

### 回答抽出の改善（このセッションで実施済み）

`extract_answer()` 関数に以下の改善を実施済み:
- `<unused\d+>thought` タグ（medgemma思考モード）の除去
- 抽出の優先順位: head match → 「正解は X」→「答え:」→ fallback（20文字以内）
- フォールバック範囲を100文字→20文字に縮小（誤検出防止）

---

## 6. 現在のLM Studio状態

- **ロード中のモデル**: openai/gpt-oss-20b @mxfp4 GGUF（12.11 GB）
- **API**: http://localhost:1234/v1

### モデル管理コマンド

```bash
# ロード済みモデル確認
lms status

# モデルロード
lms load "モデルID"

# モデルアンロード
lms unload "モデルID"

# 利用可能モデル一覧
curl -s http://localhost:1234/v1/models | python3 -c \
  "import sys,json; data=json.load(sys.stdin); print('\n'.join([m['id'] for m in data['data']]))"
```

---

## 7. 未評価モデル一覧

### 大規模（70B+）
- llama-3.1-swallow-70b-instruct-v0.3
- meta/llama-3.3-70b
- nousresearch/hermes-4-70b

### 中規模（10B-30B）
- qwen/qwen3-30b-a3b
- qwen2.5-14b-instruct-mlx
- qwen2.5-32b-instruct-mlx
- llama-4-scout-17b-16e-mlx-text
- llm-jp-3.1-13b-instruct4
- elyza-japanese-llama-2-13b-fast-instruct

### 小規模（<10B）
- google/gemma-3-4b, gemma-3-1b
- llama-3.2-3b-instruct
- internvl3-8b@bf16, internvl3_5-4b
- llama-3-swallow-8b-v0.1
- tanuki-8b-dpo-v1.0

---

## 8. 未実施タスク

### medgemma-27b関連（完了）
- [x] mt=512で全4プロンプト比較（セクションA）→ 最適プロンプト特定
- [x] 全4プロンプトにFew-shot追加して再評価 → **Baseline+few-shot = 76.0%が最良**
- [x] 最適設定で全セクション（B-F）評価 → **総合71.8%で不合格（あと13問）**

### qwen3-32b関連（完了）
- [x] Baseline+few-shot+mt=512で全セクション(A-F)評価 → **合格（79.3%）**

### 量子化比較（完了）
- [x] qwen3-32b 8bit vs 4bit → -0.5%（両方合格）
- [x] qwen3-vl-8b 8bit vs 4bit → -4.5%（両方不合格）
- [x] qwen3-vl-4b 8bit vs 4bit → -2.3%（両方不合格）

### gpt-oss-20b フォーマット比較（完了）
- [x] 6バリアント比較 → 動作する3モデルは全て~71%、全不合格
- [x] 詳細: results/gpt-oss-20b_model_comparison.md

### その他
- [ ] 年度別比較（2018-2022）
- [ ] カテゴリ別分析（神経科、放射線科など）
- [ ] 上記未評価モデルの評価
- [ ] 他の合格モデル（qwen3-235b, gpt-oss-120b等）の全セクション評価

---

## 9. 注意事項

### evaluate_prompt_comparison.py の変更時

**現在の設定**:
- max_tokensは全て512に固定済み
- few-shotはデフォルトで有効（`--no-few-shot`で無効化可能）
- qwen3モデルは自動的に`/no_think`が追加される

### 外付けドライブのモデル

一部の大規模モデル（gpt-oss-120b, qwen3-235b等）は外付けドライブにあるため、使用前にマウントが必要。

### プロット再生成

モデル追加後は `plot_size_vs_accuracy.py` にデータを追加して再生成:
```bash
python plot_size_vs_accuracy.py
# → plots/ に3つのPNG出力
```

新モデルを追加する際の編集箇所:
1. `models` リスト: `(名前, メモリGB, Best精度%, カテゴリ)` のタプルを追加
2. ラベル位置調整: `offset_x`, `offset_y` のカスタマイズ（重なり防止）
3. バジェットティア: 必要に応じて `budgets` リストにティアを追加

### Git状態

セッション6でコミット・プッシュ済み。主な変更:
- `EVALUATION_PROGRESS.md` — 量子化比較・gpt-oss-20bモデル比較結果追記
- `HANDOVER.md` — セッション6更新
- `results/gpt-oss-20b_model_comparison.md` — gpt-oss-20b 6バリアント比較詳細
- `results/gpt-oss-20b-*_2022_{A..F}.json` — gpt-oss-20b各バリアント結果
- `results/qwen3-32b-4bit_fewshot_2022_{A..F}.json` — qwen3-32b 4bit結果
- `results/qwen3-vl-8b-{8bit,4bit}_fewshot_2022_{A..F}.json` — qwen3-vl-8b結果
- `results/qwen3-vl-4b-{8bit,4bit}_fewshot_2022_{A..F}.json` — qwen3-vl-4b結果
- `plots/` — 量子化比較プロット3枚追加

---

## 10. 評価実行の典型的な流れ

1. LM Studioでモデルをロード（`lms load "モデルID"`）
2. APIの疎通確認（`lms status`）
3. 評価実行: `python evaluate_prompt_comparison.py --model "モデルID" --year 2022 --section A`
   - few-shotはデフォルト有効、max_tokensは512固定
   - few-shotなしで評価したい場合: `--no-few-shot` を追加
4. 結果確認: `results/prompt_comparison_モデルID_2022_A.json`
5. `EVALUATION_PROGRESS.md` に結果を追記（サマリーテーブル＋詳細セクション）
6. `plot_size_vs_accuracy.py` にデータ追加＆プロット再生成
7. 変更履歴を更新

---

## 11. medgemma-27b評価（完了）

全Phase完了済み（2026-02-07, セッション3）:

- **Phase 1**: 全4プロンプト比較（mt=512, few-shotなし）→ 案A(57.3%)が最良、Baselineは思考モードで17.3%
- **Phase 2**: 全4プロンプト + Few-shot → **Baseline+few-shot(76.0%)**が最良
- **Phase 3**: 全セクション(A-F)評価 → **287/400 = 71.8% → 不合格（あと13問）**
- **Phase 4**: EVALUATION_PROGRESS.md更新済み

### 結論: medgemma-27bの最適設定
- **プロンプト**: Baseline（シンプルが最善）
- **Few-shot**: 必須（思考モード抑制に決定的）
- **max_tokens**: 512（思考モード対策）
- **最終スコア**: 71.8%（合格ライン75%に届かず）

## 12. qwen3-32b 全セクション評価（完了）

**評価日**: 2026-02-07（セッション4）
**条件**: Baseline + 2-shot few-shot + max_tokens=512 + /no_think

| Section | Score | % |
|---------|-------|------|
| A | 60/75 | 80.0% |
| B | 43/50 | 86.0% |
| C | 51/75 | 68.0% |
| D | 65/75 | 86.7% |
| E | 42/50 | 84.0% |
| F | 56/75 | 74.7% |
| **Total** | **317/400** | **79.3%** |

**結果: 合格（317/400 = 79.3%）**

### 結論: qwen3-32bの最適設定
- **プロンプト**: Baseline（シンプルが最善）
- **Few-shot**: 必須（+10.7%改善）
- **max_tokens**: 512
- **/no_think**: 必須（自動付加済み）
- **最終スコア**: 79.3%（合格）

## 13. MLX量子化比較実験（8bit vs 4bit）

**評価日**: 2026-02-07（セッション5）
**条件**: Baseline + 2-shot few-shot + max_tokens=512 + /no_think
**目的**: MLX 8bit と 4bit の量子化による性能差とメモリ効率を比較

### 総合結果

| Model | Size(8bit) | Size(4bit) | 8bit | 4bit | 差分 | メモリ削減 |
|-------|-----------|-----------|------|------|------|-----------|
| qwen3-32b | 34.8GB | 18.5GB | 79.3% | 78.8% | **-0.5%** | 47% |
| qwen3-vl-8b | 9.9GB | 5.8GB | 69.8% | 65.3% | **-4.5%** | 41% |
| qwen3-vl-4b | 5.1GB | 3.0GB | 60.5% | 58.3% | **-2.3%** | 41% |

### セクション別詳細

**qwen3-32b**: A(80.0%→78.7%), B(86.0%→84.0%), C(68.0%→68.0%), D(86.7%→85.3%), E(84.0%→84.0%), F(74.7%→76.0%)
**qwen3-vl-8b**: A(62.7%→56.0%), B(76.0%→68.0%), C(60.0%→56.0%), D(73.3%→70.7%), E(76.0%→72.0%), F(74.7%→72.0%)
**qwen3-vl-4b**: A(58.7%→49.3%), B(68.0%→72.0%), C(50.7%→54.7%), D(69.3%→69.3%), E(70.0%→62.0%), F(52.0%→48.0%)

### 結論
- **大規模モデル（32B）**: 量子化の影響はほぼ無い（-0.5%）→ 4bitで十分実用的
- **中規模モデル（8B）**: -4.5%の低下 → メモリ制約がある場合のみ4bit推奨
- **小規模モデル（4B）**: -2.3%だがセクション間のばらつきが大きい
- メモリ削減は40-47%で一貫
- **合否判定は量子化で変わらない**（全モデルで8bit/4bit同一判定）
