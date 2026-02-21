# IgakuQA 論文化計画

**作成日**: 2026-02-22
**ステータス**: 企画段階

---

## 1. 論文の位置づけ

### 新規性（先行研究との差別化）

IgakuQAベンチマークを用いたLLM評価の先行研究:

| 研究 | モデル数 | 評価対象 | 環境 |
|---|---|---|---|
| Kasai et al. (2023) | 3 (GPT-3/3.5/4) | クラウドAPI | API |
| PFN MedSwallow (2024) | ~10 (70Bクラス中心) | クラウド+オープン | API+GPU |
| doctorin IgakuQA119 (2025) | ~7 (32Bクラス中心) | オープン | Google Colab |
| eques (2025) | 4 (7-14B) | ローカル小型 | ローカル |
| メディックメディア (2025/2026) | 3-6 (クラウドAI) | クラウド最新 | API |
| **本研究** | **35+** | **1B〜235B全域** | **消費者Mac** |

**差別化ポイント**:
1. **規模**: 35モデル以上 — 既存研究の5〜10倍
2. **範囲**: 1.2B〜235Bまで全サイズ帯をカバー
3. **実用性**: 消費者向けハードウェア（Mac Studio）での実測
4. **体系的比較**: 4種プロンプト × 複数max_tokens × 複数量子化条件
5. **新知見**: max_tokensが推論モデルの正答率を支配する発見、日本語FTの質的差異

### 想定ターゲットジャーナル

| ジャーナル | IF | 適合度 | 備考 |
|---|---|---|---|
| **JMIR Medical Informatics** | ~3.2 | ★★★ | 医療IT・ベンチマーク系を多く掲載。第一候補 |
| **BMC Medical Informatics** | ~3.5 | ★★★ | オープンアクセス、比較研究歓迎 |
| **PLOS Digital Health** | 新興 | ★★☆ | デジタルヘルス全般 |
| **Scientific Reports** | ~4.6 | ★★☆ | 幅広い分野を受容 |
| **npj Digital Medicine** | ~15 | ★☆☆ | ハイインパクトだがハードル高い |
| **JAMIA** | ~7.9 | ★☆☆ | 医療情報学最高峰、競争激しい |

---

## 2. 現有データの棚卸し

### 評価済みモデル一覧（セクションA, 75問, Best Accuracy）

#### 合格モデル（75%以上）— 14モデル

| Model | Size | Best Accuracy | Best Prompt | Avg Time |
|---|---|---|---|---|
| gpt-oss-120b (MLX 8bit, mt=1024) | 120B MoE | **92.0%** | 案A | 2.1s |
| gpt-oss-120b (GGUF MXFP4, mt=1024) | 120B MoE | **90.7%** | 案C | 1.3s |
| Qwen3-235B-A22B-2507 (MLX 8bit) | 235B MoE | **88.0%** | 案B/案C | 1.9s |
| Qwen3-235B-A22B | 235B MoE | **88.0%** | 案B | 1.4s |
| Qwen3-Next-80B (MLX, mt=1024) | 80B MoE(A3B) | **85.3%** | 案C | 0.4s |
| Nemotron-3-Nano (mt=1024) | 33.6GB hybrid | **84.0%** | 案D | 8.3s |
| Qwen3-VL-32B | 32B VL | **82.7%** | Baseline/案B | 3.7s |
| Llama-3.3-Swallow-70B | 70B | **81.3%** | Baseline/案B | 2.1s |
| Qwen3-32B | 32B | **80.0%** | Baseline/案B | 1.5s |
| GPT-OSS-Swallow-20B (vLLM-MLX) | 20B MoE | **80.0%** | 案A | 11.2s |
| gpt-oss-20b (mt=1024) | 20B MoE | **77.3%** | 案A | 1.2s |
| Mistral-Large-2407 | 123B | **77.3%** | Baseline/案A/案C | 6.5s |
| MedGemma-27B | 27B | **76.0%** | Baseline | 3.0s |
| Mistral-Small-3.2 | 25.9GB | **76.0%** | Baseline | 0.9s |

#### 不合格モデル（75%未満）— 21+モデル

| Model | Size | Best Accuracy | Best Prompt |
|---|---|---|---|
| Qwen3-VL-30B | 30B VL | 74.7% | 案A/B/C |
| Gemma 3 27B | 27B | 74.7% | 案B |
| Qwen3-14B | 14B | 73.3% | 案A/案B |
| Llama-3.3-70B | 70B | 68.0% | Baseline/案A/案B |
| GLM-4.6V-Flash (mt=4096) | 11.8GB | 61.3% | 案D |
| Qwen3-8B | 8B | 61.3% | 案A |
| Shisa-v2-Llama3.3-70B | 70B | 61.3% | 案A/案B |
| EZO-Gemma-3-12B | 12B | 60.0% | 案C |
| Qwen3-VL-8B | 8B VL | 60.0% | Baseline |
| OLMo-3-32B-Think | 34.3GB | 57.3% | Baseline |
| Phi-4 | 14B | 56.0% | 案A |
| Phi-4-Reasoning-Plus | 8.3GB | 56.0% | Baseline |
| Gemma 3 12B | 12B | 54.7% | — |
| InternVL3.5-8B | 8B VL | 54.7% | Baseline |
| Qwen3-4B-2507 | 4B | 54.7% | 案C |
| Llama-3.1-Swallow-8B | 8B | 53.3% | 案B |
| Qwen3-VL-4B | 4B VL | 52.0% | 案B |
| ELYZA-JP-8B | 8B | 44.0% | 案A |
| MedGemma-4B (@bf16) | 4B | 29.3% | 案A |
| LFM2.5-1.2B | 1.2B | 28.0% | 案A |
| MedGemma-4B | 4B | 18.7% | — |

#### 評価不能モデル

| Model | 理由 |
|---|---|
| GLM-4.7-Flash | LM Studio APIクラッシュ |
| Fallen-Command-A-111B | 3bit量子化で指示追従不能 |
| MiniMax-M2.5 (128.7GB) | 34.7%で評価中止 |

### 全400問評価済みモデル

| Model | Section A | 全400問 | 合否 |
|---|---|---|---|
| Qwen3-32B | 80.0% | **79.3%** | **合格** |
| GPT-OSS-Swallow-20B | 80.0% | **77.8%** | **合格** |
| Nemotron-3-Nano | 84.0% | **80.2%** | **合格** |
| Mistral-Small-3.2 | 76.0% | **76.8%** | **合格** |
| Mistral-Large-2407 | 77.3% | **75.8%** | **合格** |
| MedGemma-27B | 76.0% | **71.8%** | **不合格** |

---

## 3. 査読で指摘されるであろう課題と対応策

### 必須対応（投稿前に解決すべき）

| # | 課題 | 対応策 | 工数 |
|---|---|---|---|
| 1 | **統計的検定がない** | McNemar検定（モデル間ペア比較）、95% CI（二項分布）、Bonferroni補正 | Python実装1日 |
| 2 | **セクションA(75問)のみの評価が多い** | 主要10モデルを全400問で再評価 | 数日（自動化済み） |
| 3 | **再現性の担保** | temperature=0, seed固定の明記。GitHubにスクリプト公開済み | 記述のみ |
| 4 | **量子化条件の統一性** | モデルごとの量子化条件を表に明記。量子化影響のサブ分析を追加 | 既存データで可能 |

### 論文内で議論すべき（Limitation節）

| # | 課題 | 対応 |
|---|---|---|
| 5 | **データ汚染リスク** | 2022年の国家試験問題は学習データに含まれる可能性。議論+可能なら別年度での検証 |
| 6 | **単一評価環境** | Mac Studio M3 Ultra + LM Studio。他環境（Linux/GPU）での再現性は未検証 |
| 7 | **プロンプト依存性** | 4種比較は体系的だが「最適」とは限らない |
| 8 | **画像問題の除外** | テキスト問題のみ評価。VLモデルの画像認識性能は未評価 |
| 9 | **単一年度** | 2022年のみ。複数年度での安定性は未確認 |

### あれば強化される（nice-to-have）

| # | 強化策 | 効果 |
|---|---|---|
| 10 | 複数年度（2018-2022）での評価 | データ汚染議論の説得力向上 |
| 11 | Linux/NVIDIA GPU環境での再現実験 | 一般化可能性の向上 |
| 12 | 医師（人間）との直接比較 | インパクト大（ただしIRB必要の可能性） |
| 13 | 回答の質的分析（誤答パターン分類） | 考察の深みが増す |

---

## 4. 論文構成案

### タイトル案

> **"Benchmarking 35+ Local Large Language Models on the Japanese National Medical Licensing Examination: Scaling Laws, Fine-tuning Effects, and Inference Configuration"**

短縮案:
> **"Local LLMs vs. the Japanese Medical Exam: A Comprehensive Benchmark of 35+ Models"**

### Abstract (構造)

- **Background**: ローカルLLMはプライバシー保護の観点から医療応用が期待されるが、医学知識の体系的評価は不足
- **Methods**: IgakuQAベンチマーク、35+モデル、Mac Studio M3 Ultra、4プロンプト戦略
- **Results**: gpt-oss-120B 92.0%（全モデル最高）。32B以上で合格率85%。max_tokensが推論モデルの正答率を支配。日本語FTは手法により+13%〜-7%の差
- **Conclusions**: ローカルLLMは医師国家試験合格水準に到達。設定最適化が性能を左右する最重要因子

### Sections

1. **Introduction**
   - 医療におけるLLMの可能性とプライバシー課題
   - ローカルLLMの台頭（LM Studio, Ollama等）
   - 先行研究: IgakuQA (Kasai 2023), MedSwallow (PFN 2024), IgakuQA119 (2025)
   - 本研究の目的: 包括的ベンチマーク

2. **Methods**
   - 2.1 Benchmark: IgakuQA (116th exam, 2022)
   - 2.2 Models: 35+ models across 1.2B-235B (Table 1)
   - 2.3 Evaluation Environment: Mac Studio M3 Ultra, LM Studio API
   - 2.4 Prompt Strategies: Baseline, Format-enforced (A), Chain-of-thought (B), Japan-medical-context (C)
   - 2.5 Inference Configuration: max_tokens, quantization, /no_think
   - 2.6 Statistical Analysis: McNemar test, 95% CI, Bonferroni correction

3. **Results**
   - 3.1 Overall Rankings (Table 2: 全モデル結果)
   - 3.2 Scaling Analysis (Figure 1: size vs accuracy scatter plot)
   - 3.3 Japanese Fine-tuning Effects (Figure 2: base vs FT comparison)
   - 3.4 MoE Architecture Efficiency (Figure 3: active params vs accuracy)
   - 3.5 Impact of max_tokens on Reasoning Models (Figure 4: mt sweep)
   - 3.6 Prompt Strategy × Model Size Interaction (Figure 5: heatmap)
   - 3.7 Medical-specialized vs General Models (Table 3: MedGemma vs Qwen3)
   - 3.8 Full Exam (400 questions) Validation (Table 4)

4. **Discussion**
   - 臨床的意義（どのモデルをどの環境で使うべきか）
   - ハードウェア要件と費用対効果
   - 推論モデルの設定の重要性（実務への示唆）
   - 日本語FTの品質問題
   - Limitations

5. **Conclusion**

### Figures/Tables 計画

| # | Type | 内容 | データ有無 |
|---|---|---|---|
| Table 1 | Table | モデル一覧（名前、サイズ、アーキテクチャ、量子化、メモリ使用量） | ✅あり |
| Table 2 | Table | 全モデル正答率ランキング | ✅あり |
| Figure 1 | Scatter | パラメータ数 vs 正答率（色=アーキテクチャ） | ✅plotsにあり |
| Figure 2 | Bar | 日本語FT効果（base vs FT、同一ベースで比較） | 要作成 |
| Figure 3 | Scatter | 稼働パラメータ vs 正答率（MoE効率性） | 要作成 |
| Figure 4 | Line | max_tokens vs 正答率（gpt-oss-120B/20B） | 要作成 |
| Figure 5 | Heatmap | プロンプト × モデルの正答率ヒートマップ | ✅plotsにあり |
| Table 3 | Table | MedGemma vs 汎用モデルの比較 | ✅あり |
| Table 4 | Table | 全400問評価結果 | ✅一部あり |

---

## 5. 追加実験の優先順位

### Priority 1（投稿に必須）

- [ ] 主要10モデルの全400問評価
  - gpt-oss-120B (MLX 8bit)
  - Qwen3-235B
  - Qwen3-Next-80B
  - Qwen3-VL-32B
  - Llama-3.3-Swallow-70B
  - Qwen3-14B
  - Qwen3-8B
  - Gemma 3 27B
  - Phi-4
  - Qwen3-4B
- [ ] 統計的検定の実装（McNemar, 95% CI）
- [ ] Figure 2-4 の作成

### Priority 2（論文の質を高める）

- [ ] 複数年度（2018-2021）での主要5モデル評価
- [ ] 回答の質的分析（誤答カテゴリ分類）
- [ ] 量子化影響のサブ分析（4bit vs 8bit vs bf16）

### Priority 3（差別化を強める）

- [ ] 推論時間 vs 正答率のパレート分析（既存プロットあり）
- [ ] メモリ使用量 vs 正答率の費用対効果分析
- [ ] 英語版IgakuQA（MedQA USMLE）との交差比較

---

## 6. 既存リソース

### コード

| ファイル | 用途 |
|---|---|
| `evaluate_lmstudio_batch.py` | メイン評価スクリプト |
| `evaluate_prompt_comparison.py` | プロンプト比較実験 |
| `analyze_results.py` | 結果分析 |
| `plot_size_vs_accuracy.py` | サイズ vs 正答率プロット |
| `plot_passing_models.py` | 合格モデル比較プロット |
| `plot_additional.py` | 追加分析プロット |

### プロット（既存）

| ファイル | 内容 |
|---|---|
| `plots/size_vs_accuracy_scatter.png` | パラメータ数 vs 正答率 |
| `plots/model_ranking.png` | モデルランキング |
| `plots/section_heatmap.png` | セクション別ヒートマップ |
| `plots/section_heatmap_full.png` | 全セクションヒートマップ |
| `plots/quantization_comparison.png` | 量子化比較 |
| `plots/quantization_tradeoff.png` | 量子化トレードオフ |
| `plots/scaling_analysis.png` | スケーリング分析 |
| `plots/prompt_effectiveness.png` | プロンプト効果 |
| `plots/gpt_oss_20b_variants.png` | gpt-oss-20B変種比較 |
| `plots/max_tokens_impact.png` | max_tokens影響 |
| `plots/memory_efficiency.png` | メモリ効率 |
| `plots/pareto_and_budget.png` | パレート分析 |
| `plots/passing_models_comparison.png` | 合格モデル比較（日本語） |
| `plots/passing_models_comparison_en.png` | 合格モデル比較（英語） |
| `plots/section_difficulty.png` | セクション難易度 |

### データ

- `results/` — 全評価結果JSON（150+ファイル）
- `results/SUMMARY.json` — 初期サマリー
- `EVALUATION_PROGRESS.md` — 全評価経過記録（詳細）
- `analysis/baseline_summary.csv` — ベースライン分析
- GitHub: `https://github.com/aki-wada/IgakuQA-evaluation`

---

## 7. 記事（書籍コラム版）

`article_igakuqa.md` に書籍向けの日本語記事を作成済み。
論文とは別に、一般医師向けの解説として `book/` にも配置。
