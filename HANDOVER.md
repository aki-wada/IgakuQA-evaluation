# IgakuQA 評価プロジェクト 申し送り

**最終更新**: 2026-02-20（セッション10終了時）
**実行環境**: Mac Studio M3 Ultra (512GB RAM)
**GitHub**: https://github.com/aki-wada/IgakuQA-evaluation

---

## 1. プロジェクト概要

ローカルLLM（LM Studio経由）で日本の医師国家試験（IgakuQA）を解かせ、モデル性能を比較するベンチマーク評価プロジェクト。

- **評価対象**: 第116回医師国家試験（2022年）A問題 75問（メイン）、全セクション400問（medgemma評価）
- **合格ライン**: セクションA単独 75%（56/75問）、全セクション総合 75%（300/400問）
- **評価済みモデル数**: 37モデル+（Section A）、19モデル（全セクション評価完了）、うち合格11モデル
- **最高スコア（Section A）**: gpt-oss-120b MLX 8bit = **92.0%**（案A, mt=1024）
- **最高スコア（全セクション）**: gpt-oss-120b MLX 8bit = **84.5%** (338/400)

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
| `--max-tokens` | None | max_tokens上書き（thinking model用に4096等） |
| `--timeout` | 120 | APIタイムアウト秒数（thinkingモデルは300推奨） |
| `--output` | 自動生成 | 出力ファイルパス |

### max_tokens設定（evaluate_prompt_comparison.py内のPROMPTS辞書）

```
デフォルト値（全モデル共通）:
  baseline:         1024
  format_strict:    1024
  chain_of_thought: 1024
  japanese_medical: 1024
```

**備考**: セッション7で全プロンプトのmax_tokensを1024に統一。reasoningモデル（gpt-oss系）・medgemma・通常モデルすべてに対応可能。`--max-tokens` CLIオプションで個別上書きも可能。

### スクリプトの現在の状態

**evaluate_prompt_comparison.py の max_tokens は全て 1024 に統一済み。**
全モデルでそのまま使用可能（復元不要）。few-shotはデフォルトで有効。

```python
# 現在の値
"baseline":          {"max_tokens": 1024},
"format_strict":     {"max_tokens": 1024},
"chain_of_thought":  {"max_tokens": 1024},
"japanese_medical":  {"max_tokens": 1024},
```

**実行コマンド例**:
```bash
source venv/bin/activate

# 全セクション評価（Best Promptのみ）
for section in A B C D E F; do
  python evaluate_prompt_comparison.py \
    --model "モデルID" \
    --year 2022 \
    --section $section \
    --prompts BEST_PROMPT_KEY
done
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

- **ロード中のモデル**: nvidia/nemotron-3-nano（33.58 GB）
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

### 全セクション評価（大部分完了）

**方針**: Section Aで高得点だったプロンプト（Best Prompt）+ few-shot で、セクションA-Fの全問題を評価。

**全セクション評価 完了モデル一覧（合格順）**:

| Rank | モデル | 総合スコア | 合否 | サイズ | Best Prompt |
|------|--------|-----------|------|--------|-------------|
| 1 | gpt-oss-120b MLX | **84.5%** (338/400) | 合格 | 124.2GB | format_strict |
| 2 | gpt-oss-120b GGUF | **84.0%** (336/400) | 合格 | 63.4GB | format_strict |
| 3 | qwen3-next-80b | **83.5%** (334/400) | 合格 | 84.7GB | japanese_medical |
| 4 | qwen3-vl-32b | **82.8%** (331/400) | 合格 | 19.6GB | baseline |
| 5 | nemotron-3-nano | **80.2%** (321/400) | 合格 | 33.6GB | answer_first |
| 6 | qwen3-32b 8bit | **79.3%** (317/400) | 合格 | 34.8GB | baseline |
| 7 | qwen3-32b 4bit | **78.8%** (315/400) | 合格 | 18.5GB | baseline |
| 8 | Swallow-70b | **78.0%** (312/400) | 合格 | 40.4GB | baseline |
| 9 | qwen3-vl-30b | **77.8%** (311/400) | 合格 | 33.5GB | format_strict |
| 10 | mistral-small-3.2 | **76.8%** (307/400) | 合格 | 25.9GB | baseline |
| 11 | mistral-large | **75.8%** (303/400) | 合格 | 130.3GB | baseline |
| — | shisa-v2.1-70b | 74.2% (297/400) | 不合格 | 75.0GB | format_strict |
| — | magistral-small 8bit | 74.2% (297/400) | 不合格 | 47.2GB | baseline |
| — | magistral-small-2509 | 74.0% (296/400) | 不合格 | 47.2GB | baseline |
| — | qwen3-14b | 71.8% (287/400) | 不合格 | 15.7GB | format_strict |
| — | medgemma-27b | 71.8% (287/400) | 不合格 | 16.0GB | baseline+few-shot必須 |
| — | gemma-3-27b | 67.8% (271/400) | 不合格 | 16.9GB | chain_of_thought |
| — | phi-4 | 62.8% (251/400) | 不合格 | 15.6GB | format_strict |
| — | olmo-3-32b-think | 57.3% (Section Aのみ) | 不合格 | 34.3GB | baseline |
| — | phi-4-reasoning-plus | 56.0% (Section Aのみ) | 不合格 | 8.3GB | baseline |
| — | glm-4.6v-flash | 61.3% (Section Aのみ) | 不合格 | 11.8GB | answer_first |
| — | glm-4.7-flash | 評価不能 | API不安定 | 31.8GB | — |
| — | minimax-m2.5 | 34.7% (Section Aのみ) | 評価中止 | 128.7GB | — |

**未評価（全セクション未実施）**:
- [ ] llama-3.3-70b（Section A: 68.0%）
- [ ] qwen3-235b-2507（Section A: 88.0%、巨大モデル 249.8GB）

### その他の未実施タスク
- [ ] 年度別比較（2018-2022）
- [ ] カテゴリ別分析（神経科、放射線科など）
- [ ] 新規モデルの評価（セッション9で確認した未評価モデル一覧参照）
- [ ] 結果の git commit & push（大量の未コミット結果あり）

---

## 9. 注意事項

### evaluate_prompt_comparison.py の変更時

**現在の設定**:
- max_tokensは全て1024に統一済み（`--max-tokens`で上書き可能）
- few-shotはデフォルトで有効（`--no-few-shot`で無効化可能）
- qwen3モデルは自動的に`/no_think`が追加される
- thinkingタグ除去処理（`strip_thinking()`）が全評価スクリプトに統一追加済み
- 未閉じ`<think>`タグ、特殊トークン（`<|...|>`）除去対応済み（セッション10）
- `--timeout`オプション追加（デフォルト120s、thinkingモデルは300s推奨）

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

### Git状態（2026-02-20 更新）

ブランチ: `main`（origin/main より 1 commit 先行、未プッシュ）
最新コミット: `2392671` fix: 全評価スクリプトにthinking分離処理を統一追加

**未コミットファイル（大量）** — コミット推奨:

変更済み（modified）:
- `HANDOVER.md`
- `evaluate_prompt_comparison.py`（--timeout追加、未閉じthink除去、特殊トークン除去）
- `results/prompt_comparison_*_2022_A.json` ×9（gemma-3-27b, mistral-large, magistral-small-2509, gpt-oss-120b, qwen3-14b, qwen3-next-80b, qwen3-vl-30b, qwen3-vl-32b, Swallow-70b）

新規（untracked）— 全セクション評価結果:
- `plot_passing_models.py` — 合格モデル比較プロット生成スクリプト
- `plots/passing_models_comparison.png`, `plots/passing_models_comparison_en.png`
- `results/gpt-oss-120b-gguf_fewshot_2022_{A..F}.json` — gpt-oss-120b GGUF全セクション
- `results/prompt_comparison_nvidia_nemotron-3-nano_2022_{A..F}.json` — nemotron-3-nano全セクション（合格）
- `results/prompt_comparison_allenai_olmo-3-32b-think_2022_A.json` — olmo Section A
- `results/prompt_comparison_*_2022_{B..F}.json` — 以下モデルの全セクション結果:
  - google/gemma-3-27b
  - mistral-large-instruct-2407
  - mistralai/magistral-small-2509
  - mistralai/magistral-small（8bit版）
  - mistralai/mistral-small-3.2
  - mlx-community/phi-4
  - openai/gpt-oss-120b（MLX版）
  - qwen/qwen3-14b
  - qwen/qwen3-next-80b
  - qwen/qwen3-vl-30b
  - qwen/qwen3-vl-32b
  - shisa-v2.1-llama3.3-70b-mlx
  - tokyotech-llm-llama-3.3-swallow-70b-instruct-v0.4
- `results/prompt_comparison_minimax_minimax-m2.5_2022_A.json` — minimax-m2.5（評価中止）
- `results/prompt_comparison_qwen3-235b-a22b-thinking-2507-mlx_2022_A.json` — qwen3-235b Section A
- phi-4-reasoning-plus、glm-4.6v-flash、glm-4.7-flash の Section A 結果

---

## 10. 評価実行の典型的な流れ

1. LM Studioでモデルをロード（`lms load "モデルID"`）
2. APIの疎通確認（`lms status`）
3. 評価実行: `python evaluate_prompt_comparison.py --model "モデルID" --year 2022 --section A`
   - few-shotはデフォルト有効、max_tokensは1024固定
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

## 14. gpt-oss-120b 全セクション評価（完了）

**評価日**: 2026-02-08（セッション7）
**条件**: format_strict (案A) + 2-shot few-shot + max_tokens=1024
**モデル**: openai/gpt-oss-120b MLX 8bit (124.20 GB)

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 66 | 88.0% |
| B | 50 | 43 | 86.0% |
| C | 75 | 60 | 80.0% |
| D | 75 | 67 | 89.3% |
| E | 50 | 42 | 84.0% |
| F | 75 | 60 | 80.0% |
| **Total** | **400** | **338** | **84.5%** |

**結果: 合格（338/400 = 84.5%）**

### 比較（Section Aのみ）: few-shotなし vs few-shotあり
- few-shotなし（mt=1024）: 92.0% (69/75)
- few-shotあり（mt=1024）: 88.0% (66/75)
- **差分: -4.0%**（few-shotが逆効果）

### 結論: gpt-oss-120bの最適設定
- **プロンプト**: format_strict（案A）
- **Few-shot**: Section Aでは逆効果（-4.0%）だが全セクション合格に十分
- **max_tokens**: 1024（reasoning modelのため必須）
- **最終スコア**: 84.5%（合格、全評価モデル中最高）

## 15. gpt-oss-120b GGUF版 全セクション評価（完了）

**評価日**: 2026-02-08（セッション7）
**条件**: format_strict (案A) + 2-shot few-shot + max_tokens=1024
**モデル**: openai/gpt-oss-120b GGUF (63.39 GB)

| Section | 問題数 | 正答数 | 正答率 | MLX版比 |
|---------|--------|--------|--------|---------|
| A | 75 | 63 | 84.0% | -4.0% |
| B | 50 | 41 | 82.0% | -4.0% |
| C | 75 | 60 | 80.0% | ±0.0% |
| D | 75 | 68 | 90.7% | +1.3% |
| E | 50 | 42 | 84.0% | ±0.0% |
| F | 75 | 62 | 82.7% | +2.7% |
| **Total** | **400** | **336** | **84.0%** | **-0.5%** |

**結果: 合格（336/400 = 84.0%）**

### MLX 8bit vs GGUF 比較

| 項目 | MLX 8bit | GGUF |
|------|----------|------|
| メモリ | 124.20 GB | 63.39 GB |
| 総合スコア | 84.5% (338/400) | 84.0% (336/400) |
| 差分 | — | **-0.5%** |
| 平均応答時間 | ~2.0s | ~1.2s |
| 合否 | 合格 | 合格 |

### 結論: gpt-oss-120b量子化比較
- GGUF版はMLX版の**約半分のメモリ**（63GB vs 124GB）で **-0.5%** の精度差
- **応答速度はGGUF版が約40%高速**
- 合否判定は変わらず — **コストパフォーマンスはGGUF版が優秀**
- gpt-oss-20b（71%前後）と同様、reasoningモデルは量子化に強い

## 16. mistral-small-3.2 全セクション評価（完了）

**評価日**: 2026-02-08（セッション7）
**条件**: Baseline + 2-shot few-shot + max_tokens=1024
**モデル**: mistralai/mistral-small-3.2 (25.93 GB)

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 57 | 76.0% |
| B | 50 | 41 | 82.0% |
| C | 75 | 52 | 69.3% |
| D | 75 | 62 | 82.7% |
| E | 50 | 41 | 82.0% |
| F | 75 | 54 | 72.0% |
| **Total** | **400** | **307** | **76.8%** |

**結果: 合格（307/400 = 76.8%）**

### 結論: mistral-small-3.2の最適設定
- **プロンプト**: Baseline
- **Few-shot**: 有効（Section Aはfew-shotなしと同スコア76.0%）
- **max_tokens**: 1024
- **最終スコア**: 76.8%（合格、ギリギリだがクリア）
- **応答速度**: ~0.9s（全モデル中最速クラス）

## 17. qwen3-next-80b 全セクション評価（完了）

**評価日**: 2026-02-08（セッション7）
**条件**: japanese_medical (案C) + 2-shot few-shot + max_tokens=1024 + /no_think
**モデル**: qwen/qwen3-next-80b (84.67 GB, MoE A3B)

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 62 | 82.7% |
| B | 50 | 43 | 86.0% |
| C | 75 | 56 | 74.7% |
| D | 75 | 69 | **92.0%** |
| E | 50 | 45 | 90.0% |
| F | 75 | 59 | 78.7% |
| **Total** | **400** | **334** | **83.5%** |

**結果: 合格（334/400 = 83.5%）**

### 結論: qwen3-next-80bの最適設定
- **プロンプト**: japanese_medical（案C）
- **Few-shot**: 有効
- **max_tokens**: 1024
- **/no_think**: 必須（自動付加済み）
- **最終スコア**: 83.5%（合格、全モデル中3位）
- **応答速度**: ~0.5s（MoEモデルのため非常に高速）
- **Section D: 92.0%** は全モデル全セクション中の最高記録

## 18. mistral-large-instruct-2407 全セクション評価（完了）

**評価日**: 2026-02-08（セッション7）
**条件**: Baseline + 2-shot few-shot + max_tokens=1024
**モデル**: mistral-large-instruct-2407 (130.28 GB)

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 58 | 77.3% |
| B | 50 | 45 | **90.0%** |
| C | 75 | 50 | 66.7% |
| D | 75 | 59 | 78.7% |
| E | 50 | 35 | 70.0% |
| F | 75 | 56 | 74.7% |
| **Total** | **400** | **303** | **75.8%** |

**結果: 合格（303/400 = 75.8%）**

### 結論: mistral-large-instruct-2407の最適設定
- **プロンプト**: Baseline
- **Few-shot**: 有効
- **max_tokens**: 1024
- **最終スコア**: 75.8%（合格、ギリギリクリア）
- **応答速度**: ~6s/問（130GBの大型モデルのため遅い）
- **Section B: 90.0%** が最高、Section C: 66.7% が最低
- **メモリ使用量**: 130.28GB（最大サイズモデルの一つ）
- **コストパフォーマンス**: 130GB使って75.8%は効率が悪い（mistral-small 25.9GBで76.8%）

## 19. qwen3-vl-32b 全セクション評価（完了）

**評価日**: 2026-02-08（セッション8）
**条件**: Baseline + 2-shot few-shot + max_tokens=1024
**モデル**: qwen/qwen3-vl-32b (19.64 GB)

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 62 | 82.7% |
| B | 50 | 42 | 84.0% |
| C | 75 | 58 | 77.3% |
| D | 75 | 65 | **86.7%** |
| E | 50 | 43 | **86.0%** |
| F | 75 | 61 | 81.3% |
| **Total** | **400** | **331** | **82.8%** |

**結果: 合格（331/400 = 82.8%）**

### 結論: qwen3-vl-32bの最適設定
- **プロンプト**: Baseline
- **Few-shot**: 有効
- **max_tokens**: 1024
- **最終スコア**: 82.8%（合格、安定した高得点）
- **応答速度**: ~3.6s/問
- **メモリ使用量**: 19.64GB（合格モデル中最小）
- **メモリ効率**: 4.22%/GB（qwen3-32b 4bitに次ぐ2位）
- **全セクション77%以上**: セクション間のばらつきが小さく安定

## 20. Swallow-70b 全セクション評価（完了）

**評価日**: 2026-02-08（セッション8）
**条件**: Baseline + 2-shot few-shot + max_tokens=1024
**モデル**: tokyotech-llm-llama-3.3-swallow-70b-instruct-v0.4 (40.35 GB)

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 61 | 81.3% |
| B | 50 | 41 | 82.0% |
| C | 75 | 56 | 74.7% |
| D | 75 | 60 | 80.0% |
| E | 50 | 38 | 76.0% |
| F | 75 | 56 | 74.7% |
| **Total** | **400** | **312** | **78.0%** |

**結果: 合格（312/400 = 78.0%）**

### 結論: Swallow-70bの最適設定
- **プロンプト**: Baseline
- **Few-shot**: 有効
- **max_tokens**: 1024
- **最終スコア**: 78.0%（合格）
- **応答速度**: ~1.9s/問（高速）
- **メモリ使用量**: 40.35GB
- **日本語特化**: Llama-3.3-70bベースの日本語ファインチューン
- **Section C/F: 74.7%** が弱点（セクション単独では不合格ライン）

---

## 21. gemma-3-27b 全セクション評価

### モデル情報
- **モデル名**: google/gemma-3-27b
- **メモリ使用量**: 16.87 GB
- **量子化**: MLX (デフォルト)

### 評価設定
- **プロンプト**: 案B（段階的思考 / chain_of_thought）— Section Aで最高スコアだったプロンプト
- **Few-shot**: 有効（2-shot）
- **max_tokens**: 1024

### セクション別結果

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 56 | 74.7% |
| B | 50 | 37 | 74.0% |
| C | 75 | 44 | 58.7% |
| D | 75 | 54 | 72.0% |
| E | 50 | 34 | 68.0% |
| F | 75 | 46 | 61.3% |
| **Total** | **400** | **271** | **67.8%** |

**結果: 不合格（271/400 = 67.8%）**

### 結論: gemma-3-27bの最適設定
- **プロンプト**: 案B（段階的思考）
- **Few-shot**: 有効
- **max_tokens**: 1024
- **最終スコア**: 67.8%（不合格）
- **応答速度**: ~1.0s/問（高速）
- **メモリ使用量**: 16.87GB（小さい）
- **Section C: 58.7%、F: 61.3%** が特に弱い
- 全セクションで75%未満、合格ラインに大きく届かず

---

## 22. qwen3-vl-30b 全セクション評価

### モデル情報
- **モデル名**: qwen/qwen3-vl-30b
- **メモリ使用量**: 33.53 GB
- **量子化**: MLX (デフォルト)

### 評価設定
- **プロンプト**: 案A（回答形式強化 / format_strict）— Section Aで最高スコアだったプロンプト
- **Few-shot**: 有効（2-shot）
- **max_tokens**: 1024
- **/no_think**: 自動付与

### セクション別結果

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 55 | 73.3% |
| B | 50 | 44 | 88.0% |
| C | 75 | 53 | 70.7% |
| D | 75 | 63 | 84.0% |
| E | 50 | 39 | 78.0% |
| F | 75 | 57 | 76.0% |
| **Total** | **400** | **311** | **77.8%** |

**結果: 合格（311/400 = 77.8%）**

### 結論: qwen3-vl-30bの最適設定
- **プロンプト**: 案A（回答形式強化）
- **Few-shot**: 有効
- **max_tokens**: 1024
- **最終スコア**: 77.8%（合格）
- **応答速度**: ~2.3s/問
- **メモリ使用量**: 33.53GB
- **Section B: 88.0%、D: 84.0%** が特に高い
- **Section A: 73.3%、C: 70.7%** がやや弱い（セクション単独では不合格ライン）
- qwen3-vl-32bとの比較: 77.8% vs 82.8%（-5.0%）、メモリは33.5GB vs 19.6GB（qwen3-vl-32bが優位）

---

## 23. magistral-small-2509 全セクション評価

### モデル情報
- **モデル名**: mistralai/magistral-small-2509
- **メモリ使用量**: 47.16 GB
- **量子化**: MLX (デフォルト)

### 評価設定
- **プロンプト**: Baseline — Section Aで最高スコアだったプロンプト
- **Few-shot**: 有効（2-shot）
- **max_tokens**: 1024

### セクション別結果

| Section | 問題数 | 正答数 | 正答率 |
|---------|--------|--------|--------|
| A | 75 | 57 | 76.0% |
| B | 50 | 37 | 74.0% |
| C | 75 | 49 | 65.3% |
| D | 75 | 58 | 77.3% |
| E | 50 | 43 | 86.0% |
| F | 75 | 52 | 69.3% |
| **Total** | **400** | **296** | **74.0%** |

**結果: 不合格（296/400 = 74.0%）** — 合格ラインまであと4問

### 結論: magistral-small-2509の最適設定
- **プロンプト**: Baseline
- **Few-shot**: 有効
- **max_tokens**: 1024
- **最終スコア**: 74.0%（不合格、あと4問で合格）
- **応答速度**: ~0.9s/問（高速）
- **メモリ使用量**: 47.16GB
- **Section E: 86.0%** が最高、**Section C: 65.3%、F: 69.3%** が弱点
- mistral-small-3.2 (76.8%, 25.9GB) と比較: 精度-2.8%、メモリ+21GB（mistral-small-3.2が全面的に優位）

---

## 24. magistral-small 8bit版 全セクション評価

### モデル情報
- **モデル名**: mistralai/magistral-small (8bit版)
- **メモリ使用量**: 47.16 GB（※lms表示名は同じだが、APIモデルIDが `mistralai/magistral-small`）
- **注記**: LM Studioで `magistral-small-2509` (25.93GB, 8bit) と `magistral-small` (47.16GB) が同時ロード

### 評価設定
- **プロンプト**: Baseline
- **Few-shot**: 有効（2-shot）
- **max_tokens**: 1024

### セクション別結果（47GB版との比較）

| Section | 問題数 | 8bit(47GB) | 2509版(47GB) | 差分 |
|---------|--------|-----------|-------------|------|
| A | 75 | 53 (70.7%) | 57 (76.0%) | -5.3% |
| B | 50 | 38 (76.0%) | 37 (74.0%) | +2.0% |
| C | 75 | 53 (70.7%) | 49 (65.3%) | +5.4% |
| D | 75 | 59 (78.7%) | 58 (77.3%) | +1.4% |
| E | 50 | 41 (82.0%) | 43 (86.0%) | -4.0% |
| F | 75 | 53 (70.7%) | 52 (69.3%) | +1.4% |
| **Total** | **400** | **297 (74.2%)** | **296 (74.0%)** | **+0.2%** |

**結果: 不合格（297/400 = 74.2%）** — 47GB版とほぼ同じ（+1問）

### 結論: magistral-small量子化比較
- 8bit版(47GB) vs 2509版(47GB): ほぼ同等（74.2% vs 74.0%）
- セクションごとのブレは±5%あるが、合計はほぼ変わらず
- どちらも合格ラインに届かず（あと3-4問）

---

## セッション7（2026-02-09）

### 25. qwen3-14b 全セクション評価

**モデル**: qwen/qwen3-14b (15.71 GB)
**プロンプト**: format_strict + few-shot (2-shot)
**max_tokens**: 1024

### セクション別結果

| Section | 問題数 | 正答数 | 正答率 | 平均時間 |
|---------|--------|--------|--------|----------|
| A | 75 | 55 | 73.3% | 0.67s |
| B | 50 | 42 | 84.0% | 0.57s |
| C | 75 | 45 | 60.0% | 0.60s |
| D | 75 | 56 | 74.7% | 0.67s |
| E | 50 | 36 | 72.0% | 0.57s |
| F | 75 | 53 | 70.7% | 0.58s |
| **Total** | **400** | **287** | **71.8%** | **0.61s** |

**結果: 不合格（287/400 = 71.8%）** — 合格まであと13問

### 特徴
- 15.71GBと非常にコンパクト（合格モデル最小のqwen3-32b 4bitの18.5GBより小さい）
- 応答速度は極めて高速（0.61s/問）
- Section Bが84.0%と突出して高いが、Section Cが60.0%と低い
- medgemma-27b（71.8%）と全く同じ総合スコア

---

### 26. phi-4 全セクション評価

**モデル**: mlx-community/phi-4 (15.59 GB)
**プロンプト**: format_strict + few-shot (2-shot)
**max_tokens**: 1024

### セクション別結果

| Section | 問題数 | 正答数 | 正答率 | 平均時間 |
|---------|--------|--------|--------|----------|
| A | 75 | 42 | 56.0% | 0.64s |
| B | 50 | 35 | 70.0% | 0.62s |
| C | 75 | 48 | 64.0% | 0.91s |
| D | 75 | 47 | 62.7% | 0.65s |
| E | 50 | 34 | 68.0% | 0.82s |
| F | 75 | 45 | 60.0% | 0.67s |
| **Total** | **400** | **251** | **62.8%** | **0.72s** |

**結果: 不合格（251/400 = 62.8%）** — 合格まであと49問

### 特徴
- 15.59GBとコンパクトで高速（0.72s/問）
- 全セクションで60-70%台に留まり、突出して高いセクションがない
- 英語モデルのため日本語医学問題では厳しい
- 数値回答問題（B50, C75, F74）で長文回答を生成する傾向あり

---

### 27. shisa-v2.1-llama3.3-70b 全セクション評価

**モデル**: shisa-v2.1-llama3.3-70b-mlx (74.98 GB)
**プロンプト**: format_strict + few-shot (2-shot)
**max_tokens**: 1024
**備考**: v2のSection A結果（format_strict 61.3%が最高）を流用し、v2.1でfew-shot付き全セクション評価

### セクション別結果

| Section | 問題数 | 正答数 | 正答率 | 平均時間 |
|---------|--------|--------|--------|----------|
| A | 75 | 57 | 76.0% | 2.59s |
| B | 50 | 40 | 80.0% | 2.01s |
| C | 75 | 56 | 74.7% | 2.53s |
| D | 75 | 57 | 76.0% | 2.73s |
| E | 50 | 38 | 76.0% | 2.05s |
| F | 75 | 49 | 65.3% | 2.28s |
| **Total** | **400** | **297** | **74.2%** | **2.41s** |

**結果: 不合格（297/400 = 74.2%）** — 合格まであと3問

### 特徴
- 74.98GBと大型モデル（llama3.3-70bベース）
- A/D/E で76.0%、B で80.0%と安定しているが、Section Fが65.3%と低い
- v2（few-shotなし、Section A 61.3%）→ v2.1（few-shotあり、76.0%）で大幅改善
- あと3問で合格ライン到達の惜しい結果
- magistral-small-2509（74.0%、あと4問）と並ぶ「惜しい」グループ

---

## 28. minimax-m2.5 Section A 評価（中止）

**評価日**: 2026-02-19（セッション9）
**モデル**: minimax/minimax-m2.5 (128.68 GB, MoE)
**コンテキスト長**: 4096 → 12000 で再テスト

### 評価結果（Section Aのみ、75問）

| Prompt | ctx=4096 | ctx=12000 | 備考 |
|--------|----------|-----------|------|
| Baseline + few-shot | 21.3% | — | 分析モード突入 |
| 案A (format_strict) + few-shot | 18.7% | — | 分析モード突入 |
| **案B (chain_of_thought) + few-shot** | **34.7%** | **34.7%** | 最良だが低い |
| 案C (japanese_medical) + few-shot | 21.3% | — | 分析モード突入 |
| 案D (answer_first) + few-shot | 18.7% | 18.7% | 改善なし |

### 問題点: 「分析モード」

minimax-m2.5は医学問題に対して「この問題を分析します...」と冗長な日本語分析を開始し、max_tokens=1024のうちに回答アルファベットに到達しない。

- **コンテキスト長変更（4096→12000）**: 効果なし（完全に同一スコア）
- **プロンプト工夫（5種類）**: 効果なし（案Bの34.7%が上限）
- **案D（回答先出し）**: 効果なし（18.7%）

### 結論
- 128.68GBの巨大モデルだが、Section Aで最高34.7%
- 合格ライン75%には遠く及ばず、**評価中止**
- MoEアーキテクチャの日本語医学問題への適性に根本的な課題あり

---

## セッション10（2026-02-20）Thinkingモデル再評価

### 29. nemotron-3-nano 全セクション評価（合格）

**モデル**: nvidia/nemotron-3-nano (33.58 GB, hybrid thinking)
**プロンプト**: 案D (answer_first) + 2-shot few-shot + max_tokens=1024
**特徴**: thinkingモデルだがthinkingが短く収まり、回答品質が高い

#### Section A プロンプト比較（全5種）

| Prompt | 正答率 | 平均時間 |
|--------|--------|----------|
| 案D: 回答先出し | **84.0%** (63/75) | 8.3s |
| 案C: 日本医療文脈 | 82.7% (62/75) | 10.5s |
| 案A: 回答形式強化 | 81.3% (61/75) | 9.3s |
| Baseline | 80.0% (60/75) | 9.0s |
| 案B: 段階的思考 | 78.7% (59/75) | 11.7s |

全プロンプトが75%以上で非常に安定。案Dが最速かつ最高精度。

#### 全セクション結果（案D）

| Section | 問題数 | 正答数 | 正答率 | 平均時間 |
|---------|--------|--------|--------|----------|
| A | 75 | 63 | **84.0%** | 8.3s |
| B | 50 | 41 | 82.0% | 5.7s |
| C | 75 | 51 | 68.0% | 10.5s |
| D | 75 | 64 | **85.3%** | 9.6s |
| E | 50 | 41 | 82.0% | 5.2s |
| F | 75 | 61 | 81.3% | 7.7s |
| **Total** | **400** | **321** | **80.2%** | **~8.3s** |

**結果: 合格（321/400 = 80.2%）**

#### 結論: nemotron-3-nanoの最適設定
- **プロンプト**: 案D（回答先出し）
- **Few-shot**: 有効
- **max_tokens**: 1024
- **最終スコア**: 80.2%（合格、全モデル中5位）
- **応答速度**: ~8.3s/問（thinkingモデルのため通常モデルより遅い）
- **メモリ使用量**: 33.58GB
- **Section C: 68.0%** が唯一の弱点（セクション単独では不合格ライン）
- **Section D: 85.3%** が最高スコア
- hybridアーキテクチャにより、thinkingが適切な長さに収まる
- 案D（回答先出し）プロンプトが最適 — 他のthinkingモデルと異なる傾向

---

### 30. Thinkingモデル Section A 評価まとめ

セッション10で5つのthinkingモデルを評価。コード修正（未閉じ`<think>`タグ除去、特殊トークン除去）後の再評価を含む。

| モデル | サイズ | Section A最高 | Best Prompt | 問題点 |
|--------|--------|-------------|-------------|--------|
| **nemotron-3-nano** | 33.58GB | **84.0%** | 案D | なし（全セクション合格） |
| glm-4.6v-flash | 11.79GB | 61.3% | 案D | 遅い(~34s/問)、mt=4096必須 |
| olmo-3-32b-think | 34.26GB | 57.3% | baseline | thinking溢れ(mt=4096でも不足) |
| phi-4-reasoning-plus | 8.26GB | 56.0% | baseline | 特殊トークン汚染 |
| glm-4.7-flash | 31.84GB | 評価不能 | — | LM Studio APIクラッシュ |

#### コード修正（セッション10）

`evaluate_prompt_comparison.py` に以下を追加:
- 未閉じ`<think>`タグ除去: `re.sub(r'<(?:\w+:)?think>.*', '', response, flags=re.DOTALL)`
- 特殊トークン除去: `<|...|>` パターン（glm, phi-4-reasoning）
- `--timeout` CLIオプション追加（デフォルト120s、thinkingモデル用に300s推奨）
- `--max-tokens` CLIオプションによる上書き対応

---

## 未評価モデル一覧（2026-02-20更新）

LM Studioで利用可能だが未評価のモデル：

### 期待度高（合格可能性あり）
| モデル | 推定サイズ | 備考 |
|--------|-----------|------|
| google/gemma-3n-e4b | 不明 | 新アーキテクチャ |
| google/gemma-3-12b | ~12GB | gemma-3-27bが67.8%なので厳しいか |
| qwen/qwen3-30b-a3b-2507 | ~3GB活性 | MoE 2507版 |
| mistralai/devstral-small-2-2512 | 不明 | 最新版 |

### セッション10で評価済み（不合格/評価不能）
- allenai/olmo-3-32b-think — 57.3% FAIL
- phi-4-reasoning-plus — 56.0% FAIL
- zai-org/glm-4.6v-flash — 61.3% FAIL
- zai-org/glm-4.7-flash — API不安定、評価不能

### その他未評価
- qwen/qwen3-4b-thinking-2507, nousresearch/hermes-4-70b
- fallen-command-a-111b-v1.1-mlx, llama-4-scout-17b-16e-mlx-text
- ezo2.5-gemma-3-12b-it-preview, qwen2.5-14b/32b-instruct-mlx

### 非対応確認済み
- **qwen3.5-397b-a17b**: "Model type qwen3_5_moe not supported" エラーでロード不可（LM Studioアーキテクチャ未対応）
