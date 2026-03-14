# IgakuQA 論文化計画

**作成日**: 2026-02-22
**最終更新**: 2026-03-13
**ステータス**: データ収集完了、論文執筆準備段階

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
| **本研究** | **45+** | **1B〜397B全域** | **消費者Mac** |

**国際的な先行研究を含む拡張比較**:

| 研究 | モデル数 | 評価対象 | 環境 | 試験 |
|---|---|---|---|---|
| Kasai et al. (2023) | 3 | クラウドAPI | API | IgakuQA |
| PFN MedSwallow (2024-2025) | ~10 | クラウド+オープン | API+GPU | IgakuQA |
| doctorin IgakuQA119 (2025) | 27 | クラウド+オープン+FT | Colab/API | IgakuQA119 |
| eques (2025) | 4-5 | オープン小型 | Colab A100 | IgakuQA+YakugakuQA |
| メディックメディア (2025/2026) | 3-6 | クラウド最新 | API | 119/120回 |
| Nissen et al. (2025) | 13 | モバイル(1-8B) | iPhone/iPad | AMEGA |
| Safavi-Naini et al. (2025) | 多数 | 量子化比較 | 混合 | 消化器病学 |
| Revalida BMJ (2025) | 31 | オープン+商用 | 混合 | ブラジル医師 |
| KorMedMCQA-V (2026) | 50+ | VLモデル | 混合 | 韓国KMLE |
| **本研究** | **45+** | **1B〜397B全域** | **消費者Mac** | **IgakuQA** |

**差別化ポイント**:
1. **規模**: 45モデル以上（49構成）— 日本語医療ベンチマークでは最大規模
2. **範囲**: 1.2B〜397Bまで全サイズ帯をカバー（他研究は特定帯域に集中）
3. **実用性**: 消費者向けハードウェア（Mac Studio M3 Ultra 512GB）での実測 — API/Colab不要
4. **体系的比較**: 5種プロンプト × 複数max_tokens × 複数量子化条件の交差分析
5. **新知見**: max_tokensが推論モデルの正答率を支配する発見（30%→92%）
6. **日本語FT質的差異**: 同一ベースモデルで+13%〜-7%の両方向を実証
7. **量子化×日本語医療**: MLX 8bit/4bit/GGUFの医療タスクでの体系的比較（先行研究なし）
8. **全400問評価**: 30+構成で全セクション(A-F)評価完了、18構成が合格

**先行研究が裏付ける本研究の発見**:
- 32B以上で合格圏 → Meerkat (7B合格は教科書FT必須), KorMedMCQA (72B=78.9%)
- 大型量子化 > 小型フル精度 → Safavi-Naini et al. (npj Digital Medicine 2025)
- 小型4bit劣化/大型4bit安定 → Lee et al. (IJCAI 2025), Li et al. (ICML 2024)
- 医療特化FTが汎用に劣るケース → EQUES, AfriMed-QA, IgakuQA119
- Mistral-Small-3.2の実力 → PHI抽出研究 (Scientific Reports 2026)

**詳細**: `docs/LITERATURE_REVIEW.md` 参照

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

#### 合格モデル（75%以上）— 20構成

| Model | Size | Best Accuracy | Best Prompt | Avg Time |
|---|---|---|---|---|
| gpt-oss-120b (MLX 8bit, mt=1024) | 120B MoE | **92.0%** | 案A | 2.1s |
| Qwen3.5-397B-A17B@8bit (mt=4096) | 397B MoE(17B) | **90.7%** | 案A | 55.8s |
| Qwen3.5-397B-A17B@4bit (mt=4096) | 397B MoE(17B) | **90.7%** | 案A/案C | 44.8s |
| gpt-oss-120b (GGUF MXFP4, mt=1024) | 120B MoE | **90.7%** | 案C | 1.3s |
| Qwen3.5-27B@8bit (mt=4096) | 27B dense | **89.3%** | 案A | 69.5s |
| Qwen3-235B-A22B-thinking-2507 (mt=8192) | 235B MoE | **89.3%** | Baseline | 66.8s |
| Qwen3-235B-A22B-2507 (MLX 8bit) | 235B MoE | **88.0%** | 案B/案C | 1.9s |
| Qwen3-235B-A22B | 235B MoE | **88.0%** | 案B | 1.4s |
| Qwen3-Next-80B (MLX, mt=1024) | 80B MoE(A3B) | **85.3%** | 案C | 0.4s |
| Nemotron-3-Nano (mt=1024) | 33.6GB hybrid | **84.0%** | 案D | 8.3s |
| Qwen3-VL-32B | 32B VL | **82.7%** | Baseline/案B | 3.7s |
| Llama-3.3-Swallow-70B | 70B | **81.3%** | Baseline/案B | 2.1s |
| GPT-OSS-Swallow-20B (vLLM-MLX) | 20B MoE | **80.0%** | 案A | 11.2s |
| Qwen3-32B | 32B | **80.0%** | Baseline/案B | 1.5s |
| MiniMax-M2.5 (mt=4096) | 128.7GB MoE | **77.3%** | Baseline | 9.6s |
| gpt-oss-20b (mt=1024) | 20B MoE | **77.3%** | 案A | 1.2s |
| Mistral-Large-2407 | 123B | **77.3%** | Baseline/案A/案C | 6.5s |
| Llama 4 Scout (17Bx16E) | 109B MoE | **76.0%** | Baseline/案A/案C | — |
| MedGemma-27B | 27B | **76.0%** | Baseline | 3.0s |
| Mistral-Small-3.2 | 25.9GB | **76.0%** | Baseline | 0.9s |

#### 不合格モデル（75%未満）— 21+モデル

| Model | Size | Best Accuracy | Best Prompt |
|---|---|---|---|
| Qwen3-VL-30B | 30B VL | 74.7% | 案A/B/C |
| Gemma 3 27B | 27B | 74.7% | 案B |
| GLM-4.7-Flash (mt=8192) | 31.8GB | 73.3% | Baseline |
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
| Qwen3.5-35B-A3B | 全問で"c,e"出力、MoE互換問題 |
| Fallen-Command-A-111B | 3bit量子化で指示追従不能 |
| gpt-oss-safeguard-20b-mlx | 空回答+エラー多発 |
| internlm3-8b-instruct | API不安定（完了率16%） |
| internvl3-14b / plamo-13b | 全リクエスト失敗 |

※ 以下は再評価で復活済み（合格モデルに含む）:
- ~~GLM-4.7-Flash~~ → mt=8192で73.3%
- ~~MiniMax-M2.5~~ → mt=4096で77.3% ✓合格
- ~~Nemotron-3-Nano~~ → mt=1024+案Dで84.0% ✓合格
- ~~Qwen3-235B-thinking-2507~~ → mt=8192で89.3% ✓合格

### 全400問評価済みモデル — 30+構成（18合格 / 14不合格）

| Rank | Model | Section A | 全400問 | 合否 | サイズ |
|---|---|---|---|---|---|
| 1 | **Qwen3.5-397B-A17B@8bit** | 90.7% | **89.5%** | **✓合格⭐⭐⭐** | 249.8GB |
| 2 | **Qwen3.5-397B-A17B@4bit** | 90.7% | **87.3%** | **✓合格⭐⭐** | 223.9GB |
| 2 | **Qwen3.5-27B@8bit** | 89.3% | **87.3%** | **✓合格⭐⭐** | 29.5GB |
| 4 | **Qwen3-235B-A22B-2507** | 88.0% | **86.0%** | **✓合格⭐⭐** | 132.3GB |
| 5 | **gpt-oss-120b MLX 8bit** | 88.0% | **84.5%** | **✓合格⭐⭐⭐** | 124.2GB |
| 6 | **Qwen3-235B-A22B** | 88.0% | **84.2%** | **✓合格⭐⭐** | 132.3GB |
| 7 | **gpt-oss-120b GGUF** | 84.0% | **84.0%** | **✓合格⭐⭐⭐** | 63.4GB |
| 8 | **Qwen3-Next-80B** | 82.7% | **83.5%** | **✓合格⭐** | 84.7GB |
| 9 | **Qwen3-VL-32B** | 82.7% | **82.8%** | **✓合格⭐** | 19.6GB |
| 10 | **Nemotron-3-Nano** | 84.0% | **80.2%** | **✓合格⭐** | 33.6GB |
| 11 | **Qwen3-32B 8bit** | 80.0% | **79.3%** | **✓合格⭐** | 34.8GB |
| 12 | **Qwen3-32B 4bit** | 78.7% | **78.8%** | **✓合格** | 18.5GB |
| 13 | **Swallow-70B** | 81.3% | **78.0%** | **✓合格⭐** | 40.4GB |
| 14 | **GPT-OSS-Swallow-20B** | 80.0% | **77.8%** | **✓合格⭐** | 45GB |
| 14 | **Qwen3-VL-30B** | 73.3% | **77.8%** | **✓合格** | 33.5GB |
| 16 | **Llama 4 Scout** | 76.0% | **77.5%** | **✓合格** | 109B MoE |
| 17 | **Mistral-Small-3.2** | 76.0% | **76.8%** | **✓合格** | 25.9GB |
| 18 | **Mistral-Large** | 77.3% | **75.8%** | **✓合格** | 130.3GB |
| — | shisa-v2.1-70b | 76.0% | 74.2% | 不合格 | 75.0GB |
| — | magistral-small 8bit | 70.7% | 74.2% | 不合格 | 47.2GB |
| — | magistral-small-2509 | 76.0% | 74.0% | 不合格 | 47.2GB |
| — | Qwen3-14B | 73.3% | 71.8% | 不合格 | 15.7GB |
| — | MedGemma-27B | 76.0% | 71.8% | 不合格 | 16.0GB |
| — | gpt-oss-20b @8bit | 76.0% | 71.5% | 不合格 | 22.3GB |
| — | Llama-3.3-70B | 68.0% | 71.0% | 不合格 | — |
| — | gpt-oss-20b @mxfp4 | 76.0% | 71.0% | 不合格 | 12.1GB |
| — | Qwen3-VL-8B 8bit | 62.7% | 69.8% | 不合格 | 9.9GB |
| — | Gemma 3 27B | 74.7% | 67.8% | 不合格 | 16.9GB |
| — | Qwen3-VL-8B 4bit | 56.0% | 65.3% | 不合格 | 5.8GB |
| — | Phi-4 | 56.0% | 62.8% | 不合格 | 15.6GB |
| — | Qwen3-VL-4B 8bit | 58.7% | 60.5% | 不合格 | 5.1GB |
| — | Qwen3-VL-4B 4bit | 49.3% | 58.3% | 不合格 | 3.0GB |

---

## 3. 査読で指摘されるであろう課題と対応策

### 必須対応（投稿前に解決すべき）

| # | 課題 | 対応策 | 工数 |
|---|---|---|---|
| 1 | **統計的検定がない** | McNemar検定（モデル間ペア比較）、95% CI（二項分布）、Bonferroni補正 | Python実装1日 |
| 2 | ~~セクションA(75問)のみの評価が多い~~ | ✅ **解決済み**: 30+構成で全400問評価完了（18合格/14不合格） | — |
| 3 | **再現性の担保** | temperature=0, seed固定の明記。GitHubにスクリプト公開済み | 記述のみ |
| 4 | ~~量子化条件の統一性~~ | ✅ **解決済み**: 8bit/4bit/GGUF比較実験完了（qwen3-32b, qwen3.5-397b, gpt-oss-20b 6バリアント） | — |

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

> **"Benchmarking 45+ Local Large Language Models on the Japanese National Medical Licensing Examination: Scaling Laws, Fine-tuning Effects, and Inference Configuration"**

短縮案:
> **"Local LLMs vs. the Japanese Medical Exam: A Comprehensive Benchmark of 45+ Models on Consumer Hardware"**

### Abstract (構造)

- **Background**: ローカルLLMはプライバシー保護の観点から医療応用が期待されるが、医学知識の体系的評価は不足
- **Methods**: IgakuQAベンチマーク、45+モデル（49構成）、Mac Studio M3 Ultra、5プロンプト戦略、全400問評価
- **Results**: Qwen3.5-397B 89.5%（全400問最高）、gpt-oss-120B 92.0%（Section A最高）。18/32構成が合格。max_tokensが推論モデルの正答率を支配（30%→92%）。日本語FTは手法により+13%〜-7%の差。27B denseが397B MoE@4bitと同スコア（87.3%）
- **Conclusions**: ローカルLLMは医師国家試験合格水準に到達。設定最適化が性能を左右する最重要因子。コンパクトモデル（19.6-29.5GB）でも80%超を達成

### Sections

1. **Introduction**
   - 医療におけるLLMの可能性とプライバシー課題
   - ローカルLLMの台頭（LM Studio, Ollama等）
   - 先行研究: IgakuQA (Kasai 2023), MedSwallow (PFN 2024), IgakuQA119 (2025)
   - 本研究の目的: 包括的ベンチマーク

2. **Methods**
   - 2.1 Benchmark: IgakuQA (116th exam, 2022), 400 questions (Sections A-F)
   - 2.2 Models: 45+ models (49 configurations) across 1.2B-397B (Table 1)
   - 2.3 Evaluation Environment: Mac Studio M3 Ultra, LM Studio API + vLLM-MLX
   - 2.4 Prompt Strategies: Baseline, Format-enforced (A), Chain-of-thought (B), Japan-medical-context (C), Answer-first (D)
   - 2.5 Inference Configuration: max_tokens, quantization (8bit/4bit/GGUF), /no_think, thinking models
   - 2.6 Statistical Analysis: McNemar test, 95% CI, Bonferroni correction

3. **Results**
   - 3.1 Overall Rankings (Table 2: 全モデル結果)
   - 3.2 Scaling Analysis (Figure 1: size vs accuracy scatter plot)
   - 3.3 Japanese Fine-tuning Effects (Figure 2: base vs FT comparison)
   - 3.4 MoE Architecture Efficiency (Figure 3: active params vs accuracy)
   - 3.5 Impact of max_tokens on Reasoning Models (Figure 4: mt sweep)
   - 3.6 Prompt Strategy × Model Size Interaction (Figure 5: heatmap)
   - 3.7 Medical-specialized vs General Models (Table 3: MedGemma vs Qwen3)
   - 3.8 Full Exam (400 questions) Results (Table 4: 30+構成の全セクション評価)
   - 3.9 Quantization Impact (Table 5: 8bit vs 4bit, GGUF vs MLX)
   - 3.10 Thinking Model Analysis (max_tokens dependency, /no_think effects)

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
| Table 4 | Table | 全400問評価結果（30+構成） | ✅あり |
| Table 5 | Table | 量子化比較（8bit vs 4bit, GGUF vs MLX） | ✅あり |
| Figure 6 | Bar | Thinking vs Non-thinking model comparison | 要作成 |

---

## 5. 追加実験の優先順位

### Priority 1（投稿に必須）

- [x] 主要モデルの全400問評価 → **30+構成完了**（18合格/14不合格）
  - [x] gpt-oss-120B (MLX 8bit) → 84.5% 合格
  - [x] gpt-oss-120B (GGUF) → 84.0% 合格
  - [x] Qwen3-235B-A22B → 84.2% 合格
  - [x] Qwen3-235B-A22B-2507 → 86.0% 合格
  - [x] Qwen3-Next-80B → 83.5% 合格
  - [x] Qwen3-VL-32B → 82.8% 合格
  - [x] Llama-3.3-Swallow-70B → 78.0% 合格
  - [x] Qwen3-14B → 71.8% 不合格
  - [x] Gemma 3 27B → 67.8% 不合格
  - [x] Phi-4 → 62.8% 不合格
  - [x] Qwen3.5-397B@8bit → 89.5% 全モデル最高
  - [x] Qwen3.5-397B@4bit → 87.3% 合格
  - [x] Qwen3.5-27B@8bit → 87.3% 合格
  - [x] Nemotron-3-Nano → 80.2% 合格
  - [x] GPT-OSS-Swallow-20B → 77.8% 合格
  - [x] Llama 4 Scout → 77.5% 合格
  - [x] その他12構成
- [ ] **統計的検定の実装（McNemar, 95% CI）** ← **最優先の残タスク**
- [ ] Figure 2-4 の作成（日本語FT比較、MoE効率、max_tokens影響）

### Priority 2（論文の質を高める）

- [ ] 複数年度（2018-2021）での主要5モデル評価 ← データ汚染議論に重要
- [ ] 回答の質的分析（誤答カテゴリ分類）
- [x] 量子化影響のサブ分析 → 完了（qwen3-32b -0.5%, qwen3.5-397b -2.3%, gpt-oss-20b 6バリアント）

### Priority 3（差別化を強める）

- [x] 推論時間 vs 正答率のパレート分析 → `plots/pareto_and_budget.png` 作成済み
- [x] メモリ使用量 vs 正答率の費用対効果分析 → `plots/memory_efficiency.png` 作成済み
- [ ] 英語版IgakuQA（MedQA USMLE）との交差比較

---

## 6. 既存リソース

### コード

| ファイル | 用途 |
|---|---|
| `evaluate_lmstudio_batch.py` | メイン評価スクリプト |
| `evaluate_prompt_comparison.py` | プロンプト比較実験（5種プロンプト、thinking対応） |
| `analyze_results.py` | 結果分析 |
| `statistical_analysis.py` | 統計的検定（McNemar, CI）※実装中 |
| `plot_size_vs_accuracy.py` | サイズ vs 正答率プロット |
| `plot_passing_models.py` | 合格モデル比較プロット |
| `plot_additional.py` | 追加分析プロット |
| `plot_time_vs_accuracy.py` | 推論時間 vs 正答率プロット |

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
| `plots/time_vs_accuracy.png` | 推論時間 vs 正答率 |
| `plots/time_vs_accuracy_en.png` | 推論時間 vs 正答率（英語） |
| `plots/size_vs_accuracy_full_sections.png` | サイズ vs 正答率（全セクション） |
| `plots/pareto_and_budget_full_sections.png` | パレート分析（全セクション） |

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

---

## 8. 先行研究レビュー

**調査日**: 2026-02-22
**詳細**: `docs/LITERATURE_REVIEW.md`

### 主要参考文献（論文で引用すべき）

#### IgakuQA / 日本語医療ベンチマーク
1. Kasai et al. (2023) arXiv:2303.18027 — IgakuQA原著（被引用123件）
2. Kawakami et al. (2025) arXiv:2504.18080 — PFN MedLLM-Qwen-72B, IgakuQA 86.8%
3. Sukeda (2024) arXiv:2409.11783 — JMedLLM-v1, 低リソース日本語医療LLM
4. Jiang et al. (2024) arXiv:2409.13317 — JMedBench (COLING 2025)
5. Liu et al. (2025) IJMI — GPT-4o日本医師国家試験89.2%
6. Kasai et al. (2024) JMIR — GPT-4V画像評価（画像追加で改善なし）
7. Yano et al. (2024) arXiv:2406.14882 — 70B日本語医療QA

#### USMLE / MedQA（スケーリング関連）
8. Kim et al. (2025) npj Digital Medicine — Meerkat-7B MedQA 74.3%（7B初合格）
9. Pal et al. (2024) Scientific Reports — OpenMedLM, Yi-34BプロンプトでFT超え
10. Nature Medicine (2025) — DeepSeek-R1 USMLE 92%（オープンソース）
11. Bolton et al. (2024) arXiv:2404.15894 — Mid-Sized Models, Mistral-7B MedQA 63%

#### 量子化
12. Zhan et al. (2025) arXiv:2509.04534 — 生物医学NLP量子化（12 LLM, GPU 75%削減）
13. Lee et al. (2025) IJCAI — 量子化×タスク難易度×モデルサイズ（1B-405B）
14. Kurtic et al. (2025) ACL — "BF16 or Death", FP8ロスレス
15. Ouyang et al. (2025) ACL — 量子化スケーリング則（1500+チェックポイント）
16. Li et al. (2024) ICML — 量子化LLM評価（125M-180B, 活性化量子化の逆方向発見）
17. Safavi-Naini et al. (2025) npj Digital Medicine — 大型量子化>小型フル精度

#### エッジ/ローカル展開
18. Nissen et al. (2025) arXiv:2502.08954 — Medicine on the Edge, 13モデルiPhone/iPad
19. arXiv:2511.05502 — Apple Silicon LLM推論ベンチマーク

#### 多言語・非英語
20. MedExpQA (2024) AI in Medicine — 非英語で-10pp
21. XLingHealth (2024) Web Conference — 非英語で正確性-18%
22. KorMedMCQA-V (2026) arXiv:2602.13650 — 韓国VL 50+モデル
23. Revalida BMJ (2025) — ブラジル31モデル

---

## 9. 論文化可能性評価

### 結論: **十分に論文化可能**

#### 強み（投稿可能な理由）

1. **独自のニッチ**: 「消費者ハードウェアでの日本語医療ベンチマーク×スケーリング×量子化」は先行研究に存在しない
2. **データ量**: 35+モデル、150+結果JSONファイル、複数条件の交差分析 — 十分な規模
3. **再現性**: temperature=0, スクリプト公開済み, GitHub管理
4. **臨床的関心**: ローカルLLMのプライバシー保護×医療応用は時宜を得たテーマ
5. **先行研究との整合性**: 量子化・スケーリングの発見が複数の独立研究と一致（信頼性高い）
6. **新発見**: max_tokensの決定的影響、日本語FTの両方向効果、MoE効率性 — いずれも新規

#### 弱み（対処が必要）

1. **統計的検定未実施**: McNemar検定、95%CI追加が必須（Python実装1日） ← **最優先**
2. ~~全400問評価の不足~~ → ✅ **解決済み**: 30+構成で全400問評価完了
3. **単一年度**: 2022年のみ → 最低1年度追加（2018年推奨、データ汚染議論用）
4. **画像問題未評価**: テキストのみ → Limitation節で明記

#### 推奨ターゲット（更新）

| ジャーナル | IF | 適合度 | 根拠 |
|---|---|---|---|
| **Scientific Reports** | ~4.6 | ★★★ | OpenMedLM, MedAlpaca等の類似ベンチマーク論文掲載実績 |
| **JMIR Medical Education** | ~3.1 | ★★★ | KMLE, ドイツ試験, GPT-4V IgakuQA論文掲載実績 |
| **BMC Medical Informatics** | ~3.5 | ★★★ | 比較研究歓迎、オープンアクセス |
| **npj Digital Medicine** | ~15 | ★★☆ | Meerkat掲載実績。インパクト高いが競争激しい |
