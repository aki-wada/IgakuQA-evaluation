# IgakuQA 評価プロジェクト 申し送り

**最終更新**: 2026-02-07
**実行環境**: Mac Studio M3 Ultra (192GB RAM)
**GitHub**: https://github.com/aki-wada/IgakuQA-evaluation

---

## 1. プロジェクト概要

ローカルLLM（LM Studio経由）で日本の医師国家試験（IgakuQA）を解かせ、モデル性能を比較するベンチマーク評価プロジェクト。

- **評価対象**: 第116回医師国家試験（2022年）A問題 75問
- **合格ライン**: 75%（56/75問）
- **評価済みモデル数**: 33モデル（うち合格12モデル、評価失敗8モデル）
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
├── data/
│   ├── 2018/ ～ 2022/             # 各年度の試験データ（JSONL）
│   └── 2022/116-{A..F}.jsonl      # 2022年 セクションA～F
├── results/                        # 評価結果JSON（33ファイル）
│   ├── prompt_comparison_*.json   # v2プロンプト比較結果
│   └── *_evaluation.json          # v1評価結果
├── plots/                          # 生成プロット
│   ├── size_vs_accuracy_scatter.png  # モデルサイズ vs 精度
│   ├── scaling_analysis.png          # スケーリング分析（4パネル）
│   ├── pareto_and_budget.png         # パレートフロンティア＋メモリバジェット
│   └── memory_efficiency.png         # メモリ効率
├── venv/                           # Python仮想環境
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

### max_tokens設定（evaluate_prompt_comparison.py内のPROMPTS辞書）

```
現在のデフォルト:
  baseline:          50
  format_strict:     50
  chain_of_thought: 200
  japanese_medical:  50
```

**重要**: reasoningモデル（gpt-oss系）やqwen3-next-80bなど一部モデルでは、max_tokensが精度に極めて大きな影響を与える。これらのモデルではmax_tokens=1024に変更して実行する必要がある（実行後デフォルトに戻すこと）。

### qwen3モデルの自動対応

- モデル名に`qwen3`が含まれる場合、システムプロンプト末尾に`/no_think`を自動追加
- Thinking ON は性能を大幅に低下させるため、`/no_think`が正解

---

## 4. 主要な発見・知見

### max_tokensの影響（最重要）

- **gpt-ossシリーズ**: reasoningモデルのため内部推論がトークンを消費。mt=50では空回答が多発（0-33%）、mt=1024で解消（77-92%）
- **qwen3-next-80b**: mt=50でBaseline 66.7%と低く、案B(mt=200)は0%。mt=1024で案C 85.3%達成
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

## 5. 現在のLM Studio状態

- **ロード中のモデル**: qwen3-next-80b（84.67 GB）
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

## 6. 未評価モデル一覧

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

## 7. 未実施タスク

- [ ] medgemma-27bで全セクション(A-F)評価 → 総合合否判定
- [ ] 年度別比較（2018-2022）
- [ ] カテゴリ別分析（神経科、放射線科など）
- [ ] 上記未評価モデルの評価
- [ ] Gitへのコミット・プッシュ（多数の新規ファイルが未コミット）

---

## 8. 注意事項

### evaluate_prompt_comparison.py の変更時

max_tokensを変更して特定モデルを評価した場合、**評価後に必ずデフォルト値に戻すこと**:
```python
PROMPTS = {
    "baseline":          {"max_tokens": 50},
    "format_strict":     {"max_tokens": 50},
    "chain_of_thought":  {"max_tokens": 200},
    "japanese_medical":  {"max_tokens": 50},
}
```

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

- `EVALUATION_PROGRESS.md` は変更済み（未コミット）
- `evaluate_prompt_comparison.py`, `plot_size_vs_accuracy.py`, `results/`, `plots/` は未追跡
- `test_glm_settings.py` も未追跡

---

## 9. 評価実行の典型的な流れ

1. LM Studioでモデルをロード（`lms load "モデルID"`）
2. APIの疎通確認（`lms status`）
3. reasoningモデル/MoEモデルの場合、`evaluate_prompt_comparison.py` の max_tokens を 1024 に変更
4. 評価実行: `python evaluate_prompt_comparison.py --model "モデルID" --year 2022 --section A`
5. 結果確認: `results/prompt_comparison_モデルID_2022_A.json`
6. max_tokensを変更した場合はデフォルトに戻す
7. `EVALUATION_PROGRESS.md` に結果を追記（サマリーテーブル＋詳細セクション）
8. `plot_size_vs_accuracy.py` にデータ追加＆プロット再生成
9. 変更履歴を更新
