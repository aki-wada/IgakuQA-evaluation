# Materials and Methods — Draft

## 2.1 Benchmark Dataset

We used the IgakuQA benchmark, a publicly available dataset derived from the Japanese National Medical Licensing Examination (JMLE; 医師国家試験), originally released by Kasai et al. [1]. The dataset consists of text-based multiple-choice questions from the 112th–116th examinations (2018–2022), provided in JSONL format via GitHub (https://github.com/jungokasai/IgakuQA).

In the present study, we used the 116th examination (2022) as the primary evaluation set. This examination comprises 400 text-based questions across six sections: A (75 questions), B (50), C (75), D (75), E (50), and F (75). Each question includes a stem, five answer options labeled a–e, and one or more correct answers. Image-based questions were excluded from the original dataset by Kasai et al., and we used the text-only subset as provided.

In the actual JMLE, Sections B and E correspond to required questions (必修問題), while Sections A, C, D, and F correspond to general and clinical questions (一般問題・臨床実地問題). The official passing criteria for the 116th examination required ≥80% on required questions (with weighted scoring for clinical questions) and ≥72.1% (214/297 points) on the general-clinical section, plus no more than three prohibited answer selections (禁忌肢). In this study, we did not replicate the official JMLE scoring system, which involves differential point weights and prohibited answer detection. Instead, we adopted unweighted accuracy (correct answers / total questions) as the primary metric, consistent with the approach used by most prior IgakuQA studies [3–6]. As a practical reference for healthcare deployment potential, we adopted a 75% accuracy threshold based on the following considerations: (1) the 116th examination's general-clinical passing score was 72.1%, (2) recent passing thresholds have shown an upward trend, reaching 76.7% in the 118th examination (2024), and (3) 75% represents a conservative round-number benchmark within the observed eight-year range (69.6%–76.7%). This threshold is intended as a reference line for identifying models with potential clinical utility, not as a replication of the official JMLE passing criteria.

## 2.2 Models

We evaluated a total of 48 model configurations spanning 1.2B to 397B parameters, encompassing dense architectures, Mixture-of-Experts (MoE), vision-language models (VL), and thinking/reasoning models. All models were open-weight and publicly available. Models were obtained in quantized formats compatible with the local inference environment: MLX 8-bit, MLX 4-bit, and GGUF (MXFP4/Q4_K_M) formats. Table 1 summarizes the evaluated models with their architectures, parameter counts, quantization formats, and memory footprints.

Models were categorized into the following families and architectures:

- **Qwen3 / Qwen3.5 family**: Qwen3-4B through Qwen3-235B (dense and MoE), Qwen3-VL (4B–32B), Qwen3.5-27B (dense), Qwen3.5-397B (MoE, 17B active parameters)
- **GPT-OSS family**: gpt-oss-120B (MoE), gpt-oss-20B (MoE), GPT-OSS-Swallow-20B (Japanese fine-tuned)
- **Llama family**: Llama-3.3-70B, Llama-3.3-Swallow-70B (Japanese fine-tuned), Llama 4 Scout (17B×16E MoE), Shisa-v2.1-Llama3.3-70B (Japanese fine-tuned)
- **Gemma family**: Gemma-3-12B, Gemma-3-27B, MedGemma-4B, MedGemma-27B (medical-specialized)
- **Mistral family**: Mistral-Small-3.2, Mistral-Large-2407, Magistral-Small-2509
- **Other**: Phi-4, Nemotron-3-Nano (hybrid thinking model), OLMo-3-32B-Think, GLM-4.6V-Flash, EZO-Gemma-3-12B, ELYZA-JP-8B, LFM2.5-1.2B

For quantization comparison, we evaluated paired configurations (8-bit vs. 4-bit) for the following models: Qwen3.5-397B, Qwen3-32B, Qwen3-VL-8B, and Qwen3-VL-4B. Additionally, gpt-oss-120B was compared across MLX 8-bit and GGUF MXFP4 formats, and gpt-oss-20B was compared across six quantization variants (three functional, three non-functional).

## 2.3 Evaluation Environment

All evaluations were conducted on a single Apple Mac Studio equipped with an M3 Ultra system-on-chip (76-core GPU, 512 GB unified memory), running macOS Sonoma. Models were served locally using LM Studio (https://lmstudio.ai), which provides an OpenAI-compatible REST API at `http://localhost:1234/v1`. No cloud APIs or external network connections were used during evaluation.

Inference parameters were set as follows:
- **Temperature**: 0 (deterministic output)
- **max_tokens**: 1,024 for standard models; 4,096 for thinking/reasoning models (Qwen3.5, Nemotron-3-Nano, OLMo-3-32B-Think, GLM-4.6V-Flash, Phi-4-Reasoning-Plus)
- **Timeout**: 120 seconds per question for standard models; 300 seconds for thinking models
- **Rate limiting**: 100 ms interval between API calls

Each model was loaded individually into LM Studio before evaluation. Memory usage was recorded from LM Studio's model information display. Inference time was measured as wall-clock time per question, including API overhead.

## 2.4 Prompt Strategies

We designed four system prompt strategies to investigate the interaction between prompt engineering and model capability:

**Baseline**: A minimal instruction directing the model to answer as a medical expert, selecting from options a–e.

> あなたは医学の専門家です。選択肢から正解をa,b,c,d,eで答えてください。

**Strategy A (Format-strict)**: Emphasizes output format compliance, explicitly requesting only alphabetic answers without explanation.

> あなたは臨床医学に精通した医師です。問題を読み、正解の選択肢をアルファベット(a,b,c,d,e)のみで回答してください。複数選択の場合はカンマ区切りで回答してください。例: a,c 余計な説明は不要です。

**Strategy B (Chain-of-thought)**: Instructs step-by-step medical reasoning before providing a final answer.

> あなたは臨床医学に精通した医師です。回答手順: 1. 問題文のキーワードを確認 2. 各選択肢を医学的に検討 3. 正解を選択 最終行に「答え:」に続けて選択肢(a,b,c,d,e)のみを記載してください。

**Strategy C (Japanese medical context)**: Explicitly invokes Japanese clinical guidelines and healthcare system knowledge.

> あなたは日本の医療制度と臨床医学に精通した専門医です。医学の多肢選択問題に回答します。重要: 日本の診療ガイドラインに基づいて判断。選択肢はa,b,c,d,eのアルファベットで回答。指定された個数を必ず選択。回答形式: a または a,b,c（複数の場合）

For models evaluated on Sections B–F (full 400-question assessment), only the best-performing prompt from the Section A evaluation was used.

An additional **Strategy D (Answer-first)** was used selectively for thinking models:

> あなたは医学の専門家です。最初の行に正解の選択肢（a,b,c,d,eのアルファベットのみ）を出力してください。複数選択の場合はカンマ区切りで出力してください。例: a,c 2行目以降に解説を書いてもかまいません。

## 2.5 Few-Shot Examples

Two-shot in-context learning was employed for all evaluations. Few-shot examples were drawn from the 100th examination (a different year from the evaluation set) to avoid information leakage. The examples comprised one single-answer question and one multi-answer question, presented in a multi-turn chat format: each example was formatted as a user message (question with options) followed by an assistant message containing only the correct answer letter(s). This format served two purposes: (1) demonstrating the expected output format, and (2) suppressing undesired behaviors such as verbose reasoning or thinking-mode activation in certain models (e.g., MedGemma-27B).

## 2.6 Answer Extraction and Scoring

Model responses were processed through a multi-stage answer extraction pipeline:

1. **Thinking tag removal**: Internal reasoning tokens were stripped using regular expressions, handling multiple formats: `<think>...</think>` (Qwen3, DeepSeek), `</think>` without opening tag (Nemotron, OLMo), `<unused94>thought...` (MedGemma), `<|begin_of_box|>...<|end_of_box|>` (GLM), and truncated thinking blocks caused by max_tokens limits.
2. **Special token removal**: LLM control tokens in `<|...|>` format were removed.
3. **Character normalization**: Full-width alphabetic characters (ａ–ｅ) were converted to half-width (a–e).
4. **Answer pattern matching**: Answers were extracted using a priority-ordered sequence:
   - (a) Standalone answer letters on the first line (e.g., "a,c")
   - (b) Japanese answer patterns: 「正解は X」「答え: X」「回答: X」
   - (c) Fallback: first a–e letters within the initial 20 characters
5. **Scoring**: Extracted answer letters were sorted and compared against sorted gold-standard answers using exact set matching. A response was scored as correct only when the predicted set exactly matched the gold-standard set.

For Qwen3-series models, the `/no_think` token was automatically appended to system prompts to suppress the built-in thinking mode. For Qwen3.5-series models, where `/no_think` was ineffective, the thinking output was removed post-hoc via the extraction pipeline.

## 2.7 Evaluation Protocol

The evaluation was conducted in two phases:

**Phase 1 — Prompt comparison on Section A** (75 questions): Each model was evaluated with up to five prompt strategies (Baseline, A, B, C, D) using two-shot examples on Section A. The best-performing prompt was identified based on accuracy.

**Phase 2 — Full examination** (400 questions): For models achieving ≥75% on Section A, the full 400-question evaluation (Sections A–F) was conducted using the best prompt from Phase 1. This two-phase approach balanced computational cost against comprehensive evaluation of promising models. A total of 26 models completed the full 400-question assessment.

All evaluation scripts, prompts, and result data are publicly available at https://github.com/aki-wada/IgakuQA-evaluation.

## 2.8 Statistical Analysis

Accuracy for each model was calculated as the number of correct answers divided by the total number of questions, and reported with exact (Clopper-Pearson) 95% confidence intervals based on the binomial distribution. For the full 400-question evaluation, the 95% CI width was approximately ±4%, compared with ±9% for the 75-question Section A evaluation alone.

Pairwise comparisons between models evaluated on the same 400-question set were performed using McNemar's exact test, which accounts for the paired (matched) structure of the data—each question serves as its own matched pair across models. The McNemar test specifically evaluates whether the number of discordant pairs (questions answered correctly by only one of the two models) differs significantly in direction.

Twelve pre-specified comparisons were defined across four categories:

1. **Quantization pairs** (4 comparisons): 8-bit vs. 4-bit for Qwen3.5-397B, Qwen3-32B, Qwen3-VL-8B, and Qwen3-VL-4B.
2. **Inference format** (1 comparison): MLX 8-bit vs. GGUF for gpt-oss-120B.
3. **Japanese fine-tuned vs. base models** (3 comparisons): Swallow-70B vs. Llama-3.3-70B, Shisa-v2.1-70B vs. Llama-3.3-70B, and GPT-OSS-Swallow-20B vs. gpt-oss-20B.
4. **Medical-specialized vs. base model** (1 comparison): MedGemma-27B vs. Gemma-3-27B (same model family).
5. **Adjacent models near the 75% reference threshold** (3 comparisons): Mistral-Large-2407 vs. Shisa-v2.1-70B, Mistral-Large-2407 vs. Magistral-Small-2509, and Llama-4-Scout vs. Mistral-Small-3.2.

To control for multiple comparisons, Bonferroni correction was applied with significance set at α = 0.05/12 = 0.0042. Results significant at the nominal α = 0.05 but not surviving Bonferroni correction are reported separately.

All statistical analyses were performed using Python 3.9 with scipy 1.13.1 (Clopper-Pearson CI via `scipy.stats.beta`) and statsmodels 0.14.6 (McNemar's exact test via `statsmodels.stats.contingency_tables.mcnemar`).

---

## References (for this section)

[1] Kasai J, Kasai Y, Sakaguchi K, Yamada Y, Radev D. Evaluating GPT-4 and ChatGPT on Japanese Medical Licensing Examinations. arXiv:2303.18027, 2023.
[2] 厚生労働省. 第116回医師国家試験の合格発表について. 2022.
[3] Kawakami W, Suzuki K, Iwasawa J. Stabilizing Reasoning in Medical LLMs with Continued Pretraining and Reasoning Preference Optimization. arXiv:2504.18080, 2025.
[4] Iwase N. IgakuQA119: LLMベンチマーク構築とFine-tuningによる性能評価. Zenn, 2025.
[5] EQUES. 医薬分野のQ&AでローカルLLMを評価する. Zenn, 2025.
[6] Liu M, et al. Evaluating the Effectiveness of advanced LLMs in medical Knowledge. Int J Med Inform, 193, 2025.
