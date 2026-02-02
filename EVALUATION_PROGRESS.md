# IgakuQA LLM評価 経過記録

**記録日時**: 2026-02-02
**実行環境**: macOS (初回評価)
**継続環境**: Mac Studio M3 Ultra (予定)

**GitHubリポジトリ**: https://github.com/aki-wada/IgakuQA-evaluation

---

## 評価完了モデル（第116回 2022年 A問題 75問）

| Rank | Model | Size | Accuracy | Correct | Avg Time | Status |
|------|-------|------|----------|---------|----------|--------|
| 1 | **medgemma-27b-text-it-mlx** | 27B | **73.3%** | 55/75 | 3.0s | ✓ **合格** |
| 2 | qwen/qwen3-4b-2507 | 4B | 57.3% | 43/75 | 0.8s | 不合格 |
| 3 | gemma-3-12b-it | 12B | 54.7% | 41/75 | 1.6s | 不合格 |
| 4 | openai/gpt-oss-20b | 20B | 28.0% | 21/75 | 1.8s | 不合格 |
| 5 | medgemma-1.5-4b-it@bf16 | 4B | 25.3% | 19/75 | 1.2s | 不合格 |
| 6 | medgemma-1.5-4b-it | 4B | 18.7% | 14/75 | 0.7s | 不合格 |

### ベースライン比較（論文より）
| Model | Accuracy | 備考 |
|-------|----------|------|
| GPT-4 | 80.0% | クラウドAPI |
| ChatGPT | 58.0% | クラウドAPI |
| GPT-3 | 42.0% | クラウドAPI |

---

## 評価失敗モデル（要再評価）

| Model | 理由 |
|-------|------|
| llama-3-elyza-jp-8b-mlx | API error - モデル未ロード |
| microsoft/phi-4-reasoning-plus | API error - モデル未ロード |

---

## 未評価モデル（LM Studioで利用可能）

```
google/gemma-3-4b
qwen/qwen3-vl-4b
qwen/qwen3-vl-8b
qwen/qwen3-vl-30b
liquid/lfm2.5-1.2b
zai-org/glm-4.7-flash
tinyswallow-1.5b-instruct
internvl3_5-14b
glm-4.6v-flash-mlx
olmocr-2-7b-1025
phi-4
internvl3-14b
translategemma-4b-it
translategemma-12b-it
translategemma-27b-it
ezo2.5-gemma-3-12b-it-preview
nvidia/nemotron-3-nano
mistralai/ministral-3-3b
google/gemma-3-12b
mistralai/magistral-small-2509
```

---

## Mac Studio M3 Ultraでの継続手順

### 0. リポジトリをクローン

```bash
git clone https://github.com/aki-wada/IgakuQA-evaluation.git
cd IgakuQA-evaluation
```

### 1. 環境セットアップ

```bash
# 仮想環境作成
python3 -m venv venv
source venv/bin/activate
pip install requests
```

### 2. LM Studio起動確認

```bash
# モデル一覧確認
curl -s http://localhost:1234/v1/models | python3 -c "import sys,json; data=json.load(sys.stdin); print('\n'.join([m['id'] for m in data['data']]))"
```

### 3. 評価実行コマンド

#### 単一モデル評価
```bash
python evaluate_lmstudio_batch.py \
  --models "モデル名" \
  --year 2022 \
  --section A \
  --use-few-shot \
  --output results/モデル名_evaluation.json
```

#### 複数モデル一括評価
```bash
python evaluate_lmstudio_batch.py \
  --models "model1" "model2" "model3" \
  --year 2022 \
  --section A \
  --use-few-shot \
  --output results/batch_evaluation.json
```

#### 全セクション評価（medgemma-27bで合否判定）
```bash
for section in A B C D E F; do
  python evaluate_lmstudio_batch.py \
    --models "medgemma-27b-text-it-mlx" \
    --year 2022 \
    --section $section \
    --use-few-shot \
    --output results/medgemma-27b_2022_${section}.json
done
```

### 4. おすすめ追加評価

M3 Ultraの192GB RAM環境では以下が推奨：

1. **大規模モデル追加評価**
   - qwen3-vl-30b
   - translategemma-27b-it
   - internvl3_5-14b

2. **日本語特化モデル再評価**
   - llama-3-elyza-jp-8b-mlx（要モデルロード）
   - ezo2.5-gemma-3-12b-it-preview

3. **medgemma-27bで全セクション評価**
   - A〜F全問で総合合否判定

---

## 結果ファイル一覧

```
results/
├── medgemma_evaluation.json      # medgemma 3モデル
├── qwen3_evaluation.json         # qwen3-4b
├── additional_evaluation.json    # gemma-3-12b等
├── gpt-oss-20b_evaluation.json   # gpt-oss-20b
```

---

## 主要な発見

1. **合格できたのはmedgemma-27bのみ（73.3%）**
2. **パラメータ数 ≠ 性能**: gpt-oss-20B(28%) < qwen3-4B(57%)
3. **医療特化 < 汎用（小規模時）**: medgemma-4B(19%) < qwen3-4B(57%)
4. **日本語対応が重要**: gpt-ossは空回答多発
5. **合格には20B以上の医療特化または70B以上の汎用が必要と推察**

---

## 次のステップ候補

- [ ] medgemma-27bで全セクション(A-F)評価 → 総合合否判定
- [ ] より大規模モデル(30B+)の評価
- [ ] 年度別比較(2018-2022)
- [ ] カテゴリ別分析（神経科、放射線科など）
- [ ] プロンプト最適化の効果検証
