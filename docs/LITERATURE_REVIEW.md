# IgakuQA 論文化プロジェクト: 先行研究レビュー

**調査日**: 2026-02-22（更新: 2026-03-13）
**調査範囲**: 2023-2026年の関連研究

---

## 目次

1. [日本の先行研究](#1-日本の先行研究)
2. [国際的な医師国家試験でのLLM評価](#2-国際的な医師国家試験でのllm評価)
3. [スケーリング則・モデルサイズと医学的性能](#3-スケーリング則モデルサイズと医学的性能)
4. [量子化が医学タスクに与える影響](#4-量子化が医学タスクに与える影響)
5. [エッジ/消費者ハードウェアでの医療LLM展開](#5-エッジ消費者ハードウェアでの医療llm展開)
6. [プロンプト工学と推論戦略](#6-プロンプト工学と推論戦略)
7. [医療LLM評価方法論の批判と発展](#7-医療llm評価方法論の批判と発展)
8. [本研究が埋めるギャップ](#8-本研究が埋めるギャップ)

---

## 1. 日本の先行研究

### 1.1 Kasai et al. (2023) — IgakuQA原著論文

- **論文**: Kasai J, Kasai Y, Sakaguchi K, Yamada Y, Radev D. "Evaluating GPT-4 and ChatGPT on Japanese Medical Licensing Examinations." arXiv:2303.18027, 2023.
- **被引用数**: 123件（2026年2月時点）
- **モデル**: GPT-4, ChatGPT (GPT-3.5), GPT-3（3モデル、全てクラウドAPI）
- **データ**: 第112-117回（2018-2023）、各年400問、計~2,400問、5択選択式
- **結果**:
  - GPT-4: **全6年度で合格**（唯一の合格モデル）
  - ChatGPT: 全年度不合格
  - GPT-3: 全年度不合格
- **画像問題**: 約25%（除外して評価）
- **重要発見**:
  - GPT-4は禁忌肢選択が最少（0-1件/年）
  - 日本語テキストは英語の約2倍のトークンを消費
  - Chain-of-Thought は本タスクで改善なし
  - 人間受験者の91.7%が合格（2022年）
- **Limitation**: ブラックボックスAPI、データ汚染リスク、画像未評価、オープンソースモデル未評価
- **URL**: https://arxiv.org/abs/2303.18027
- **GitHub**: https://github.com/jungokasai/IgakuQA

```bibtex
@misc{kasai2023igakuqa,
  author = {Jungo Kasai and Yuhei Kasai and Keisuke Sakaguchi and Yutaro Yamada and Dragomir Radev},
  title = {Evaluating {GPT}-4 and {ChatGPT} on {J}apanese Medical Licensing Examinations},
  year = {2023},
  eprint = {2303.18027},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

### 1.2 PFN MedSwallow / Preferred-MedLLM (2024-2025)

#### 第1世代: Llama3-Preferred-MedSwallow-70B (2024年7月)

- **ブログ**: Iwasawa J, Suzuki K, Kawakami K. PFN Tech Blog, 2024-07-17.
- **ベース**: tokyotech-llm/Llama-3-Swallow-70B-v0.1
- **手法**: QLoRA（2x A100 GPU）
- **結果**: IgakuQA平均 **395.2点**（GPT-4: 388.8点を超過）— オープンモデル初のGPT-4超え
- **URL**: https://tech.preferred.jp/ja/blog/llama3-preferred-medswallow-70b/
- **HuggingFace**: pfnet/Llama3-Preferred-MedSwallow-70B

#### 第2世代: Preferred-MedLLM-Qwen-72B (2025)

- **論文**: Kawakami W, Suzuki K, Iwasawa J. "Stabilizing Reasoning in Medical LLMs with Continued Pretraining and Reasoning Preference Optimization." arXiv:2504.18080, 2025.
- **ベース**: Qwen/Qwen2.5-72B
- **手法**: CPT (QLoRA, 4x A100) + RPO (Reasoning Preference Optimization, 2x A100)
- **結果**: IgakuQA **86.8%**（GPT-4o: 86.6%を超過）
- **重要発見**:
  - 説明生成時にベースモデルは精度5-11%低下するが、RPOで完全に解消
  - CPTが+6.5%の最大寄与、RPOは推論安定化に寄与
  - DPOよりRPOが説明生成時の安定性で優位
- **URL**: https://arxiv.org/abs/2504.18080

```bibtex
@article{kawakami2025preferredmedllm,
  title = {Stabilizing Reasoning in Medical LLMs with Continued Pretraining and Reasoning Preference Optimization},
  author = {Kawakami, Wataru and Suzuki, Keita and Iwasawa, Junichiro},
  journal = {arXiv preprint arXiv:2504.18080},
  year = {2025}
}
```

### 1.3 doctorin IgakuQA119 (2025)

- **著者**: Naoto Iwase (岩瀬直人), 医学部5年生
- **公開**: 2025-04-12（更新: 2025-07-06）
- **試験**: 第119回（2025年2月実施）、500問（うち画像103問）
- **モデル数**: **27モデル**（クラウド+オープン+FT）
- **主要結果**:

| Rank | モデル | 精度 |
|---|---|---|
| 1 | Gemini 2.5 Pro | **97.25%** |
| 2 | OpenAI o3 | 96.00% |
| 3 | Gemini 2.5 Flash | 95.50% |
| 4 | Claude Sonnet 4 | 93.75% |
| 5 | Qwen3-235B (オープン最高) | **91.50%** |
| 10 | QwQ-32B | 83.50% |
| 11 | Qwen3-32B | 82.25% |
| 16 | Cogito-32B-Think | 77.50% |
| 24 | MedGemma-27B Q6_K | 62.50% |

- **FT実験**: SFT: +3.0%、CPT: -1.0%（逆効果）
- **重要発見**: CPTが回答フォーマット崩壊を引き起こす（8問で選択肢全文出力）
- **URL**: https://zenn.dev/doctorin/articles/e985df9bac7f99
- **GitHub**: https://github.com/docto-rin/IgakuQA119
- **DOI**: 10.5281/zenodo.15743221

### 1.4 メディックメディア INFORMA (2025-2026)

#### 第119回（2025年2月）— 6モデル評価

- **モデル**: GPT-4o, OpenAI o1, o3-mini-high, Claude 3.5 Sonnet, DeepSeek-R1, Gemini 2.0 Flash
- **結果**: **全6モデル合格**
  - o3-mini-high: 一般臨床 **96.0%**（受験者9,642人中**3位**相当）
  - o1: 必修 **98.0%**
- **全モデル共通不正解**: F-67（ビタミンB1 vs 葉酸）、C-52（インフルエンザ48時間計算）
- **URL**: https://informa.medilink-study.com/web-informa/post45927.html/

#### 第120回（2026年2月）— 3モデル評価

- **モデル**: GPT-5.2 Thinking, Gemini 3 Pro, Claude Opus 4.5
- **結果**: **全3モデル圧倒的合格**
  - Claude Opus 4.5: 一般臨床 **98.3%**（受験者中**1位**相当）
  - Gemini 3 Pro: 必修 **99.5%**（1問のみ不正解）
  - **全モデル共通不正解: 0問**（前年から大幅改善）
- **URL**: https://informa.medilink-study.com/web-informa/post51586.html/

### 1.5 GPT-OSS Swallow (2026) — 日本語強化大型オープンLLM

- **開発**: 東京工業大学 Swallow チーム
- **公開**: 2026-02-20
- **モデル**: GPT-OSS-Swallow-20B-{SFT,RL}, GPT-OSS-Swallow-120B-{SFT,RL}（4モデル）
- **ベース**: OpenAI GPT-OSS (20B, 120B)
- **手法**: 継続事前学習 (Swallow Corpus v3.2) + SFT + 強化学習
- **結果**: 英語タスク平均 **0.804**（120Bパラメータ以下のオープンLLM最高性能）。AIME 24-25で+15.0pt
- **本研究との関連**: 本研究で評価したgpt-oss-safeguard-120b-mlxの上流モデル。Safeguardバージョンとの比較が可能
- **URL**: https://swallow-llm.github.io/gptoss-swallow.en.html
- **HuggingFace**: tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1

### 1.6 EQUES (2025) — ローカルLLM医薬評価

- **著者**: 株式会社EQUES（東大松尾研発スタートアップ）
- **環境**: Google Colab A100（厳密にはローカルではない）
- **ベンチマーク**: IgakuQA (1,455問), YakugakuQA (4,485問), JMMLU
- **主要結果**:

| モデル | サイズ | IgakuQA | YakugakuQA |
|---|---|---|---|
| Phi-4 | 14B | **57.0%** | 43.8% |
| Qwen2.5-7B | 7B | 44.7% | 36.8% |
| Swallow-8B | 8B | 39.7% | 36.8% |
| EQUES-MedLlama-v2 | 8B | 42.7% | 30.6% |

- **評価方式**: Exact Match（厳格、回答抽出なし）
- **重要発見**: 医療特化モデル（EQUES-MedLlama-v2）が汎用モデルに劣る
- **URL**: https://zenn.dev/eques/articles/20cc5451ac9b09

### 1.7 その他の日本語医療LLMベンチマーク

#### JMedBench (COLING 2025)

- **論文**: Jiang J, Huang J, Aizawa A. "JMedBench: A Benchmark for Evaluating Japanese Biomedical Large Language Models." COLING 2025.
- **内容**: 日本語生物医学20データセット、5タスク（MCQA, NER, MT, DC, STS）
- **URL**: https://arxiv.org/abs/2409.13317

#### KokushiMD-10 (2025)

- **論文**: Liu J, et al. "KokushiMD-10: Benchmark for Evaluating LLMs on Ten Japanese National Healthcare Licensing Examinations." arXiv:2506.11114.
- **内容**: 医師含む10種の国家試験11,588問。GPT-4oでも70%しか合格せず
- **URL**: https://arxiv.org/abs/2506.11114

#### JMedEthicBench (2026)

- **論文**: Liu J, et al. "JMedEthicBench: A Multi-Turn Conversational Benchmark for Evaluating Medical Safety in Japanese LLMs." arXiv:2601.01627.
- **内容**: 50,000+の敵対的会話。**医療FTがsafety低下させる**発見
- **URL**: https://arxiv.org/abs/2601.01627

#### JPHARMATRON (2025)

- **論文**: Ono S, et al. "A Japanese Language Model and Three New Evaluation Benchmarks for Pharmaceutical NLP." arXiv:2505.16661.
- **内容**: 7BでIgakuQA 64.7%。YakugakuQA、NayoseQA、SogoCheckを新規作成
- **URL**: https://arxiv.org/abs/2505.16661

#### JMedLLM-v1 (2024)

- **論文**: Sukeda I. "Development and bilingual evaluation of Japanese medical large language model within reasonably low computational resources." arXiv:2409.11783.
- **内容**: 7BでIgakuQA 52.3%。英語基盤モデルの日本語医療FTで両言語改善（クロスリンガル転移）
- **URL**: https://arxiv.org/abs/2409.11783

#### Liu et al. (2025) — GPT-4oの日本医師国家試験評価

- **論文**: Liu M, et al. "Evaluating the Effectiveness of advanced LLMs in medical Knowledge: A Comparative study using Japanese national medical examination." Int J Med Inform, 193, 2025.
- **結果**: GPT-4o **89.2%**。画像問題で10%低下
- **URL**: https://pubmed.ncbi.nlm.nih.gov/39471700/

#### Liu et al. (2025) — 教科書レベル医学知識

- **論文**: Liu M, et al. "Textbook-Level Medical Knowledge in LLMs: A Comparative Evaluation Using the Japanese National Medical Examination." medRxiv 2025.09.10.25335398.
- **結果**: Gemini 2.5 Pro **97.2%**, GPT-5 **96.3%**, Claude Opus 4.1 **96.1%**
- **URL**: https://www.medrxiv.org/content/10.1101/2025.09.10.25335398v1

#### GPT-4V(ision) 多モーダル評価

- **論文**: Kasai J, et al. "Capability of GPT-4V(ision) in the Japanese National Medical Licensing Examination: Evaluation Study." JMIR Medical Education, 2024;10:e54393.
- **結果**: 画像あり68% vs なし72%（画像追加で改善なし、p=.36）
- **URL**: https://mededu.jmir.org/2024/1/e54393

#### ローカルLLM PHI抽出 (2026)

- **論文**: "Bridging the performance gap: systematic optimization of local LLMs for Japanese medical PHI extraction." Scientific Reports, 2026.
- **結果**: Mistral-Small-3.2 + Self-Refine: **91.54点**（GPT-4.1の97.8%）
- **URL**: https://www.nature.com/articles/s41598-026-36904-5

#### Yano et al. (2024) — 70Bモデル日本語医療QA

- **論文**: Yano T, et al. "70B-parameter large language models in Japanese medical question-answering." arXiv:2406.14882.
- **結果**: Swallow-70bが日本語医療Instruction Tuningで50%超え達成
- **URL**: https://arxiv.org/abs/2406.14882

---

## 2. 国際的な医師国家試験でのLLM評価

### 2.1 USMLE / MedQA（米国）

#### DeepSeek医療ベンチマーク (Nature Medicine 2025)

- **論文**: "Comparative benchmarking of the DeepSeek large language model on medical tasks and clinical reasoning." Nature Medicine, 31, 2546-2549, 2025.
- **モデル**: DeepSeek-R1 (671B MoE), ChatGPT-o1, Llama 3.1-405B
- **結果**: DeepSeek-R1 **92%** USMLE, ChatGPT-o1 95%, Llama-405B 83%
- **URL**: https://www.nature.com/articles/s41591-025-03726-3

#### Meerkat (npj Digital Medicine 2025)

- **論文**: Kim H, et al. "Small Language Models Learn Enhanced Reasoning Skills from Medical Textbooks." npj Digital Medicine, 8, 262, 2025.
- **モデル**: Meerkat-7B (Mistral基盤)
- **結果**: MedQA **74.3%** — **7Bで初のUSMLE合格**。教科書CoTで+7.5%
- **URL**: https://www.nature.com/articles/s41746-025-01653-8

#### OpenMedLM (Scientific Reports 2024)

- **論文**: Pal A, et al. "OpenMedLM: prompt engineering can out-perform fine-tuning in medical question-answering with open-source LLMs." Scientific Reports, 14, 14364, 2024.
- **モデル**: Yi-34B
- **結果**: MedQA **72.6%**, MMLU医療 **81.7%** — プロンプト工学のみでFT済み70B超え
- **URL**: https://www.nature.com/articles/s41598-024-64827-6

#### Llama-3-Meditron (OpenReview 2025)

- **論文**: Sallinen A, et al. "Llama-3-Meditron: An Open-Weight Suite of Medical LLMs Based on Llama-3.1."
- **モデル**: 8B, 70B
- **結果**: 70BがGPT-4超え。8Bも全Llama-3.1モデルを+3%超過
- **URL**: https://openreview.net/forum?id=ZcD35zKujO

#### Mid-Sized Models (arXiv 2024)

- **論文**: Bolton E, et al. "Assessing The Potential Of Mid-Sized Language Models For Clinical QA." arXiv:2404.15894.
- **モデル**: Mistral-7B, BioGPT-large, BioMedLM
- **結果**: Mistral-7B **63.0%** MedQA（汎用7Bが専門小型超え）
- **URL**: https://arxiv.org/abs/2404.15894

#### MedAlpaca (arXiv 2023/2025)

- **論文**: Bressem KK, et al. "MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data." arXiv:2304.08247.
- **モデル**: LLaMA-7B, 13B
- **結果**: 7B < 13B（スケーリング確認）。**8bit FTで精度低下**を早期報告
- **URL**: https://arxiv.org/abs/2304.08247

#### USMLE臨床推論ベンチマーク (Scientific Reports 2026)

- **論文**: Siam MK, et al. "Benchmarking LLMs on the USMLE for clinical reasoning and medical licensing scenarios." Scientific Reports, 16, 1387, 2026.
- **結果**: DeepSeek **93%** Step 2 CK。推論最適化アーキテクチャが同サイズ汎用モデルに優位
- **URL**: https://www.nature.com/articles/s41598-025-31010-4

### 2.2 韓国（KMLE）

#### KMLE 3年分析 (Scientific Reports 2025)

- **論文**: "Performance evaluation of large language models on Korean medical licensing examination: a three-year comparative analysis." Scientific Reports, 2025.
- **モデル**: GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro
- **結果**: GPT-4o **83.2%**（3年平均）。年々改善傾向
- **URL**: https://www.nature.com/articles/s41598-025-20066-x

#### KorMedMCQA (arXiv 2024)

- **論文**: "KorMedMCQA: Multi-Choice Question Answering Benchmark for Korean Healthcare Professional Licensing Examinations." arXiv:2403.01469.
- **データ**: 7,469問（医師・看護師・薬剤師・歯科）
- **結果**: Qwen2.5-72B **78.86%**（オープン最高）。CoTで+4.5%
- **URL**: https://arxiv.org/abs/2403.01469

#### KorMedMCQA-V (arXiv 2026)

- **論文**: "KorMedMCQA-V: A Multimodal Benchmark for Evaluating Vision-Language Models on the Korean Medical Licensing Examination." arXiv:2602.13650.
- **データ**: 1,534問+2,043画像
- **モデル**: 50+モデル
- **結果**: Qwen3-VL-32B-Thinking **83.7%**（オープン最高）。**Reasoning > 専門化**（+20pp差）
- **URL**: https://arxiv.org/abs/2602.13650

#### 韓国薬剤師試験 (medRxiv 2025)

- **論文**: "Proprietary and Open-Source LLMs on the Korean Pharmacist Licensing Examination." medRxiv, 2025.
- **モデル**: 27モデル（商用5+オープン22）
- **結果**: 7モデルが全6年度合格。Claude 3.5 Sonnet: 人間受験者上位12%
- **URL**: https://www.medrxiv.org/content/10.1101/2025.04.15.25325584v2

#### ソウル大学病院 (2024)

- **韓国医療特化LLM**: 3,800万臨床テキストで学習
- **結果**: KMLE **86.2%**（現役医師平均79.7%超え）

### 2.3 インド（MedMCQA / AIIMS・NEET-PG）

#### MedMCQA (PMLR 2022)

- **論文**: Pal A, Umapathi LK, Sankarasubbu M. "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering." CHIL 2022.
- **データ**: 193,000+問（AIIMS/NEET-PG、インド医学大学院入試）、21科目、2,400+テーマ
- **主要結果**:
  - OpenMedLM (Yi-34B): **68.3%**（プロンプト工学のみ）
  - Med-Gemini: **91.1%**（MedQAだが参考）
  - Clinical Camel 70B: 54.2%（5-shot）
- **重要性**: USMLE/MedQAに次ぐ世界第2の医療MCQAベンチマーク
- **URL**: https://proceedings.mlr.press/v174/pal22a.html

### 2.4 中国（CMExam / CMB）

#### CMExam (NeurIPS 2023)

- **論文**: Liu J, et al. "Benchmarking LLMs on CMExam -- A Comprehensive Chinese Medical Exam Dataset." NeurIPS 2023 D&B.
- **データ**: 60K+問（中国医師資格試験）
- **結果**: GPT-4 61.6%、ChatGLM-6B 45.3%（GPT-3.5の3%パラメータで匹敵）
- **URL**: https://arxiv.org/abs/2306.03030

#### CMB (NAACL 2024)

- **論文**: Wang X, et al. "CMB: A Comprehensive Medical Benchmark in Chinese." NAACL 2024.
- **結果**: Qwen-72B, Yi-34B, Yi-6BがCMB-ExamでGPT-4超え。ただし臨床シナリオ(CMB-Clin)ではGPT-4が優位
- **URL**: https://arxiv.org/abs/2308.08833

#### MedBench (2024)

- **論文**: Liu Y, et al. "MedBench: A Comprehensive, Standardized, and Reliable Benchmarking System for Evaluating Chinese Medical LLMs."
- **データ**: 300,901問、43臨床専門分野
- **URL**: https://arxiv.org/abs/2407.10990

### 2.5 欧州

#### ドイツ Staatsexamen (JMIR 2024)

- **論文**: Meyer A, et al. "Comparison of the Performance of GPT-3.5 and GPT-4 With That of Medical Students on the Written German Medical Licensing Examination." JMIR Medical Education, 2024.
- **結果**: GPT-4: M1 **93.1%**, M2 **94%**（学生平均: M1 73%, M2 74%）
- **URL**: https://mededu.jmir.org/2024/1/e50965

#### スペイン MIR (arXiv 2025)

- **論文**: "Evaluating LLMs on the Spanish Medical Intern Resident (MIR) Examination 2024/2025." arXiv:2503.00025.
- **モデル**: 22モデル（LLaMA含む）。FTモデル(Miri Pro)が臨床推論で優位
- **URL**: https://arxiv.org/abs/2503.00025

#### スウェーデン SMLB (Frontiers 2025)

- **論文**: Moell B, et al. "Swedish Medical LLM Benchmark." Frontiers in AI, 2025.
- **URL**: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1557920/full

### 2.6 ブラジル（ポルトガル語）

#### Revalida ベンチマーク (BMJ 2025)

- **論文**: "Benchmarking open-source LLMs on Portuguese Revalida multiple-choice questions." BMJ Health Care Informatics, 2025.
- **データ**: 399問、**31モデル（23オープン+8商用）**
- **結果**: Llama 3 70B **77.5%**（オープン最高）、Mixtral-8x7B 63.7%（MoE中型）
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12082654/

### 2.7 アラビア語

#### MedAraBench (arXiv 2026)

- **論文**: "MedAraBench: Large-Scale Arabic Medical Question Answering Dataset and Benchmark." arXiv:2602.01714.
- **結果**: GPT-o3 76.5%、オープンソースは大幅に低い
- **URL**: https://arxiv.org/abs/2602.01714

### 2.8 アフリカ

#### AfriMed-QA (ACL 2025, Best Social Impact Award)

- **論文**: "AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset." ACL 2025.
- **データ**: 15,275問、16国、32専門分野、30モデル
- **結果**: 小型エッジモデルは合格困難。**医療特化LLMが汎用に劣る**
- **URL**: https://arxiv.org/abs/2411.15640

### 2.9 多言語ベンチマーク

#### MedExpQA (AI in Medicine 2024)

- **論文**: Alonso I, et al. "MedExpQA: Multilingual benchmarking of LLMs for Medical Question Answering." AI in Medicine, 155, 2024.
- **言語**: 英語、フランス語、イタリア語、スペイン語
- **結果**: **非英語で精度-10ポイント**（全モデル共通）
- **URL**: https://www.sciencedirect.com/science/article/pii/S0933365724001805

#### XLingHealth (Web Conference 2024)

- **論文**: Yao Y, et al. "Better to Ask in English: Cross-Lingual Evaluation of LLMs for Healthcare Queries."
- **言語**: 英語、スペイン語、中国語、ヒンディー語
- **結果**: 正確性 **-18%**、一貫性 **-29%**、検証可能性 **-13%**（非英語）
- **URL**: https://dl.acm.org/doi/10.1145/3589334.3645643

#### WorldMedQA-V (NAACL Findings 2025)

- **論文**: "WorldMedQA-V: a multilingual, multimodal medical examination dataset." NAACL Findings 2025.
- **言語**: ポルトガル語、ヘブライ語、日本語、スペイン語
- **データ**: 568問+568画像（臨床医検証済み）
- **URL**: https://arxiv.org/abs/2410.12722

#### MedExamLLM (JMIR 2024)

- **論文**: "Large Language Models in Worldwide Medical Exams: Platform Development and Comprehensive Analysis." JMIR, 2024.
- **規模**: 15言語、28国、198試験、16モデル
- **URL**: https://www.jmir.org/2024/1/e66114

### 2.10 多言語性能ギャップまとめ

| 指標 | 英語比での低下 | 出典 |
|---|---|---|
| 精度 | -10ポイント | MedExpQA (仏/伊/西) |
| 正確性 | -18% | XLingHealth (西/中/ヒンディー) |
| 一貫性 | -29% | XLingHealth |
| 検証可能性 | -13% | XLingHealth |

---

## 3. スケーリング則・モデルサイズと医学的性能

### 3.1 モデルサイズしきい値

| サイズ帯 | 代表的結果 | 出典 |
|---|---|---|
| **1-3B** | USMLE 56, 医療QA不合格 | MedAide, Medicine on Edge |
| **7B** | MedQA 63-74%（FTで合格可能） | Meerkat, Bolton et al. |
| **14-34B** | MedQA 72.6%（プロンプト工学で70B超え） | OpenMedLM |
| **70B** | USMLE 62.5-77.5%（オープン合格圏） | Liévin et al., Revalida |
| **405B** | USMLE 83% | Nature Medicine 2025 |
| **671B MoE** | USMLE **92%** | DeepSeek-R1 |

### 3.2 主要スケーリング発見

| 発見 | 出典 |
|---|---|
| 学習データの質がモデルサイズを補償できる（7BでUSMLE合格） | Meerkat (npj Digital Med 2025) |
| プロンプト工学で34BがFT済み70B超え | OpenMedLM (Sci Rep 2024) |
| テスト時推論スケーリングで小型が大型超え。推論トークン最適: ~4K | m1 (arXiv 2025) |
| 小型モデルが推論予算増で大型に匹敵する「等価点」が存在 | Economics of Accuracy (medRxiv 2025) |
| 推論最適化アーキテクチャ(DeepSeek-R1)が同サイズ汎用を大幅超過 | Nature Medicine 2025 |
| MoE (Mixtral-8x7B) が一部の大型denseモデル超え | Revalida (BMJ 2025) |
| Reasoning variants が instruction-tuned を +20pp 超過 | KorMedMCQA-V (2026) |

### 3.3 推論モデルの医療性能

#### DeepSeek R1 医療推論分析 (Frontiers in AI 2025)

- **論文**: "Medical reasoning in LLMs: an in-depth analysis of DeepSeek R1." Frontiers in AI, 2025.
- **モデル**: DeepSeek-R1 (671B MoE)
- **結果**: MedQA臨床ケース100問で診断精度 **93%**。鑑別診断、ガイドラインベース治療選択、患者固有因子の統合を体系的に実施
- **眼科**: 中国語MCQ **86.2%**、英語MCQ **80.8%**（Gemini 2.0 Pro、o1、o3-miniを超過）
- **URL**: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616145/full

#### MedR-Bench: 臨床推論ベンチマーク (Nature Communications 2025)

- **論文**: "Quantifying the reasoning abilities of LLMs on clinical cases." Nature Communications, 2025.
- **データ**: 1,453症例（13臓器系、10専門分野、一般疾患+希少疾患）
- **結果**: GPT-4 **87.6%**（USMLE形式）。ただし現代医学は事実想起を超えた文脈適応・確率的推論・ガイドライン追従が必要
- **URL**: https://www.nature.com/articles/s41467-025-64769-1

#### 大規模推論モデルの医療MMLU-Pro評価 (medRxiv 2025)

- **論文**: "Evaluating Large Reasoning Model Performance on Complex Medical Scenarios In The MMLU-Pro Benchmark." medRxiv, 2025.
- **結果**: 推論モデル（o3-mini, DeepSeek-R1）がMMLU-Pro医療カテゴリで従来モデルを超過。ただしQwQ-32Bは精度24.4%（推論過多で非効率）
- **URL**: https://www.medrxiv.org/content/10.1101/2025.04.07.25325385v2.full

### 3.4 テスト時推論スケーリング

#### m1: 医療推論のテスト時スケーリング (arXiv 2025)

- **論文**: "m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models." arXiv:2504.00869, 2025.
- **モデル**: 7B, 32Bを23Kサンプルでファインチューニング
- **結果**:
  - m1-7B: MedQA **60.32%**（HuatuoGPT-o1-7B超え、新SoTA）
  - m1-32B: 70B級医療LLMに匹敵
- **重要発見**:
  - **最適推論トークン予算 ≈ 4K**（超過するとoverthinkingで精度低下）
  - 医学知識不足がテスト時スケーリングのボトルネック
  - 医学推論と数学推論は根本的に異なる
- **URL**: https://arxiv.org/abs/2504.00869
- **GitHub**: https://github.com/UCSC-VLAA/m1

#### Rethinking Test-Time Scaling for Medical AI (arXiv 2025)

- **論文**: "Rethinking Test-Time Scaling for Medical AI: Model and Task-Aware Strategies for LLMs and VLMs." arXiv:2506.13102, 2025.
- **結果**: テスト時スケーリングの効果はモデルとタスクに強く依存。画一的な推論予算増加は非効率
- **URL**: https://arxiv.org/abs/2506.13102

#### MedS3 (arXiv 2025)

- **論文**: Jiang S, et al. "MedS3: Towards Medical Slow Thinking with Self-Evolved Soft Dual-sided Process Supervision." arXiv:2501.12051.
- **結果**: 小型モデルが32Bクラスを **+8.57pt** 超過。MCTS推論でリソース制約デバイス向け
- **URL**: https://arxiv.org/abs/2501.12051

---

## 4. 量子化が医学タスクに与える影響

### 4.1 医療/生物医学特化の量子化研究

#### 生物医学NLPにおける量子化LLM (arXiv 2025)

- **論文**: Zhan Z, et al. "Quantized Large Language Models in Biomedical Natural Language Processing: Evaluation and Recommendation." arXiv:2509.04534.
- **モデル**: 12 LLM（ClinicalCamel-70B, HuatuoGPT-o1-70B, Med42-70B, Meditron-70B等）
- **結果**: GPU **75%削減**で性能維持。ただし一部生物医学FTモデルは8bitでも劣化
- **結論**: 4bit量子化は生物医学タスクで有効だが**モデル固有の影響**あり
- **URL**: https://arxiv.org/abs/2509.04534

#### 消化器病学ベンチマーク (npj Digital Medicine 2025)

- **論文**: Safavi-Naini SAA, et al. "Benchmarking proprietary and open-source language and vision-language models for gastroenterology clinical reasoning." npj Digital Medicine, 2025.
- **重要発見**: **大型量子化モデル > 小型フル精度モデル**（同メモリ予算）
  - 4bit 90Bモデル > フル精度 11Bモデル (61.0% vs 48.7%)
  - 4bit 11Bモデル > フル精度 3Bモデル (44.3% vs 35.7%)
- **URL**: https://www.nature.com/articles/s41746-025-02174-0

### 4.2 汎用量子化研究（医療に適用可能）

#### "BF16 or Death" (ACL 2025)

- **論文**: Kurtic E, et al. "'Give Me BF16 or Give Me Death?' Accuracy-Performance Trade-Offs in LLM Quantization." ACL 2025.
- **モデル**: Llama-3.1全族 (8B, 70B, 405B)、500,000+評価
- **結果**: FP8 **事実上ロスレス**、INT4 **予想以上に実用的**
- **URL**: https://arxiv.org/abs/2411.02355

#### 量子化スケーリング則 (ACL 2025)

- **論文**: Ouyang X, et al. "Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens." ACL 2025.
- **データ**: 1,500+量子化チェックポイント
- **重要発見**: **学習量が多いモデルほど量子化に弱い**。将来のモデルは量子化がより問題に
- **URL**: https://arxiv.org/abs/2411.17691

#### 量子化×タスク難易度×モデルサイズ (IJCAI 2025)

- **論文**: Lee J, et al. "Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and Model Size in LLMs From Edge to Giant." IJCAI 2025.
- **モデル**: 1B-405B、4量子化手法、13ベンチマーク
- **結果**: FP8が最もロバスト。**小型は4bitで大幅劣化、70B+は安定**
- **URL**: https://arxiv.org/abs/2409.11055

#### 量子化LLMの総合評価 (ICML 2024)

- **論文**: Li S, et al. "Evaluating Quantized Large Language Models." ICML 2024.
- **モデル**: 11モデル族、125M-180B
- **結果**: 70B 3bitで微小劣化、7B 同条件で大幅劣化。**活性化量子化は大型ほど不利**（逆方向）
- **URL**: https://arxiv.org/abs/2402.18158

#### 量子化×推論 (arXiv 2025)

- **論文**: "Quantization Meets Reasoning: Exploring LLM Low-Bit Quantization Degradation for Mathematical Reasoning." arXiv:2501.03035.
- **結果**: 4bit PTQでMATH最大32%劣化。**332例のリカバリ学習で回復可能**
- **URL**: https://arxiv.org/abs/2501.03035

### 4.3 量子化しきい値まとめ

| ビット幅 | 精度影響 | 備考 |
|---|---|---|
| **FP8 (8bit)** | **<2%劣化** | 全サイズでロバスト (ACL 2025, IJCAI 2025) |
| **4bit (INT4)** | **2-8%劣化** | 70B+で実用的、<14Bで問題あり (複数出典) |
| **3bit** | **5-20%劣化** | GPTQ崩壊、AWQ/QuIPで維持可 (Huang et al. 2024) |
| **2bit** | **15-30%+劣化** | 専門手法（PB-LLM等）のみ (ICML 2024) |

### 4.4 本研究の量子化結果との整合性

| 本研究の発見 | 先行研究の裏付け |
|---|---|
| 大型(32B) 4bit: -0.5%（無視可能） | Lee et al. (IJCAI 2025): 70B+は4bitで安定 |
| 中型(8B) 4bit: -4.5% | Lee et al.: 小型は4bitで大幅劣化 |
| gpt-oss-20b MLX Q4: 9.3%（崩壊） | 量子化×推論: 推論タスクで最大32%劣化 |
| 動作する3モデルは0.5%以内 | Red Hat: 適切な量子化で差なし |
| メモリ削減40-47% | Zhan et al.: GPU 75%削減可能 |

---

## 5. エッジ/消費者ハードウェアでの医療LLM展開

### 5.1 Medicine on the Edge (arXiv 2025)

- **論文**: Nissen L, et al. "Medicine on the Edge: Comparative Performance Analysis of On-Device LLMs for Clinical Reasoning." arXiv:2502.08954.
- **環境**: iPhone/iPad、4bit MLX量子化
- **モデル**: 13モデル（1B-8B）

| モデル | サイズ | AMEGA Score |
|---|---|---|
| Aloe 8B（医療） | 8B | 490.9 |
| Med42 8B（医療） | 8B | 490.0 |
| Llama 3.1 8B | 8B | 464.8 |
| Phi 3 Mini | 3.8B | 464.6 |
| Llama 3.2 1B | 1B | 256.5 |

- **結論**: Med42-8B推奨。Phi3-Mini(3.8B)がコスパ最良（6GBで稼働）
- **URL**: https://arxiv.org/abs/2502.08954

### 5.2 Apple Silicon推論ベンチマーク (arXiv 2025)

- **論文**: "Production-Grade Local LLM Inference on Apple Silicon." arXiv:2511.05502.
- **環境**: Mac Studio M2 Ultra 192GB
- **結果**: MLX ~230tok/s > MLC-LLM ~190tok/s > llama.cpp ~150tok/s > Ollama 20-40tok/s
- **URL**: https://arxiv.org/abs/2511.05502

### 5.3 MedAide (arXiv 2024)

- **論文**: "MedAide: Leveraging LLMs for On-Premise Medical Assistance on Edge Devices." arXiv:2403.00830.
- **環境**: 消費者GPU、Nvidia Jetson
- **結果**: USMLE 56（合格未満）。LoRA+RLHFで最適化
- **URL**: https://arxiv.org/abs/2403.00830

### 5.4 プライバシー保護関連

| 論文 | 内容 | URL |
|---|---|---|
| SoK: Privacy-aware LLM in Healthcare (2026) | 医療LLMプライバシー体系化 | arXiv:2601.10004 |
| Emergency Triage (Applied Sciences 2025) | 4/8bit Ollamaで緊急トリアージ | MDPI 15(15):8412 |
| Ollama Federated Learning (2025) | 連合学習で病院間プライバシー保護 | PubMed:41619364 |

---

## 6. プロンプト工学と推論戦略

### 6.1 CoTプロンプトの医療MCQAにおける体系的比較 (Computers in Biology and Medicine 2025)

- **論文**: "A comparative evaluation of chain-of-thought-based prompt engineering techniques for medical question answering." Computers in Biology and Medicine, 2025.
- **結果**: CoT + kNNベース few-shot + シャッフルオプション self-consistency が最高性能
- **重要発見**: CoTはzero-shot/few-shotを一貫して超過。アンサンブル変種 (self-consistency) がCoT単体を超過
- **URL**: https://www.sciencedirect.com/science/article/pii/S0010482525009655

### 6.2 プロンプト工学は普遍的に有効ではない (arXiv 2025)

- **論文**: "Prompt engineering does not universally improve Large Language Model performance across clinical decision-making tasks." arXiv:2512.22966, 2025.
- **モデル**: GPT-4o, Gemini 1.5 Pro, Llama 3.3 70B
- **データ**: 36臨床ケース × 5臨床推論タスク
- **結果**:
  - 最低精度タスク（検査選択）ではプロンプト工学が有意改善
  - **他タスクでは逆効果**になるケースあり
  - 動的 few-shot がランダム選択を一貫して上回らない
- **結論**: プロンプト工学の効果は**モデル・タスク依存**。文脈に応じた戦略が必要
- **本研究との関連**: 本研究の「プロンプト最適化はモデルサイズ依存」発見と整合
- **URL**: https://arxiv.org/abs/2512.22966

### 6.3 OpenMedLM: プロンプト工学がFTを凌駕 (Scientific Reports 2024)

- ※ セクション2.1に詳細記載
- **核心**: Yi-34BがMedQA **72.6%** — プロンプト工学のみでFT済み70B超え
- **本研究との関連**: 本研究でも大型モデルではBaselineが最適（複雑なプロンプトは不要）

### 6.4 プロンプト工学のスコーピングレビュー (JMIR 2024)

- **論文**: Zaghir J, et al. "Prompt engineering paradigms for medical applications: scoping review and recommendations for better practices." JMIR, 2024.
- **問題点**: **61%の研究がプロンプト非使用ベースラインを報告せず**
- **推奨**: モデル選択がプロンプト戦略と同等以上に重要
- **URL**: https://arxiv.org/abs/2405.01249

---

## 7. 医療LLM評価方法論の批判と発展

### 7.1 MedHELM: 包括的医療LLM評価フレームワーク (Nature Medicine 2025)

- **論文**: "Holistic evaluation of large language models for medical tasks with MedHELM." Nature Medicine, 2025.
- **機関**: Stanford HAI
- **構成**:
  - 臨床医検証済み分類法: 5カテゴリ（臨床意思決定支援、臨床記録生成、患者コミュニケーション、医学研究、事務）、22サブカテゴリ、121タスク
  - ベンチマークスイート: 37評価
  - LLM-jury評価: 複数AI評価者による自動評価（ICC=0.47、臨床医間合意ICC=0.43を超過）
- **結果**: 推論モデル (DeepSeek R1, o3-mini) が勝率 **66%**。ただしClaude 3.5 Sonnetが **15%低コスト**で同等性能
- **意義**: 国家試験スコアは実臨床の多様性を反映せず、包括評価が必要
- **URL**: https://www.nature.com/articles/s41591-025-04151-2

### 7.2 知識-実践ギャップ (JMIR 2025)

- **論文**: "Knowledge-Practice Performance Gap in Clinical Large Language Models." JMIR, 2025.
- **結果**:
  - 知識ベースベンチマーク: 平均精度 **70-79%**
  - 実践ベースベンチマーク: 平均精度 **46-70%**
  - MCQ形式 > 自由記述形式（大幅な差）
- **結論**: 国家試験(MCQ)での合格は実臨床能力を保証しない
- **URL**: https://www.jmir.org/2025/1/e84120/

### 7.3 構成概念妥当性の問題 (arXiv 2025)

- **論文**: "Medical Large Language Model Benchmarks Should Prioritize Construct Validity." arXiv:2503.10694, 2025.
- **指摘**: 現在の医療LLMベンチマークは構成概念妥当性（測定したいものを実際に測定しているか）を軽視
- **URL**: https://arxiv.org/abs/2503.10694

### 7.4 ベンチマーク飽和問題

- 2025-2026年時点で主要ベンチマーク (MedQA, USMLE) は飽和に近づいている
- 一部の不正解にはground truthラベルの誤りが含まれる（ベンチマーク自体の品質問題）
- MedQAが臨床性能の最も予測力の高いベンチマークだが、患者コミュニケーション・長期ケア・臨床情報抽出は捕捉できない

### 7.5 本研究への示唆

| 批判点 | 本研究の対応 |
|---|---|
| MCQ形式の限界 | 本研究はMCQ評価であり、この限界を認識して考察に記載すべき |
| ベンチマーク飽和 | IgakuQAは日本語・非英語であり、まだ飽和していない |
| 構成概念妥当性 | 医師国家試験は実際の臨床能力のproxy。限界は考察で言及 |
| 知識-実践ギャップ | 本研究のスコープは知識評価に限定。臨床実装への含意は慎重に |

---

## 8. 本研究が埋めるギャップ

### 8.1 先行研究との差別化

| 観点 | 既存研究の状況 | 本研究の独自性 |
|---|---|---|
| **日本語医療×ローカル** | APIかColab/GPU評価のみ | **消費者Mac完結で35+モデル** |
| **スケーリング分析** | 英語(MedQA)中心、2-5モデル | **日本語でQwen3 4B→235Bの体系的分析** |
| **量子化×医療** | 一般ベンチマーク中心 | **IgakuQAでMLX 8bit/4bit/GGUFを比較** |
| **プロンプト×サイズ相互作用** | 効果はモデル・タスク依存 (arXiv 2512.22966) | **4種プロンプト×35+モデルでサイズ依存性を体系的に実証** |
| **max_tokens影響** | m1で最適≈4Kトークンの示唆あり | **推論モデルで30%→92%の劇的影響を初めて体系的に実証** |
| **MoE効率性×医療** | DeepSeek-R1のみ | **gpt-oss, Qwen3-Next等で体系的に実証** |
| **日本語FTの質的差異** | 「FTは有効」の一般論 | **同一ベースで+13%〜-7%の両方向を実証** |

### 8.2 先行研究が裏付ける本研究の発見

| 本研究の発見 | 裏付ける先行研究 |
|---|---|
| 32B以上で合格圏 | Meerkat (7B合格は教科書FT必須)、KorMedMCQA (72B=78.9%) |
| 大型量子化 > 小型フル精度 | Safavi-Naini et al. (npj Digital Medicine 2025) |
| 医療特化FTが汎用に劣るケース | EQUES (MedLlama-v2)、AfriMed-QA、doctorin IgakuQA119 |
| プロンプト工学がFTを代替可能 | OpenMedLM (Yi-34B)、MedExpQA |
| 推論モデルのアーキテクチャ優位 | DeepSeek-R1 (92% USMLE)、KorMedMCQA-V (+20pp) |
| max_tokensが推論モデルの精度を支配 | m1: 最適推論トークン≈4K、超過でoverthinking劣化 |
| プロンプト最適化はモデルサイズ依存 | arXiv 2512.22966: プロンプト効果はモデル・タスク依存 |
| Mistral-Small-3.2の実力 | PHI抽出研究 (GPT-4.1の97.8%達成) |

### 8.3 本研究の位置づけ（性能比較）

| モデル/研究 | サイズ | 精度 | 環境 |
|---|---|---|---|
| Gemini 2.5 Pro (Liu et al.) | Cloud | 97.2% | API |
| GPT-4o (Liu et al.) | Cloud | 89.2% | API |
| Preferred-MedLLM-Qwen-72B | 72B | 86.8% | GPU (A100) |
| **gpt-oss-120B MLX 8bit (本研究)** | **120B** | **84.5%** | **Mac Studio (ローカル)** |
| **gpt-oss-120B GGUF (本研究)** | **120B** | **84.0%** | **Mac Studio (ローカル)** |
| **qwen3-next-80b (本研究)** | **80B** | **83.5%** | **Mac Studio (ローカル)** |
| Cogito-32B-Think (IgakuQA119) | 32B | 77.5% | Ollama |
| **qwen3-32b 8bit (本研究)** | **32B** | **79.3%** | **Mac Studio (ローカル)** |
| Llama 3 70B (Revalida) | 70B | 77.5% | — |
| Meerkat-7B (MedQA) | 7B | 74.3% | GPU |
