# Table 1. Evaluated Models

## Full Table (49 configurations, sorted by full-exam accuracy where available, then Section A accuracy)

| Model | Family | Params | Architecture | Active Params | Quant. | Format | Memory (GB) | Section A (%) | Full 400Q (%) | 95% CI |
|-------|--------|--------|-------------|---------------|--------|--------|------------|--------------|--------------|--------|
| Qwen3.5-397B-A17B | Qwen3.5 | 397B | MoE | 17B | 8-bit | MLX | 249.8 | 90.7 | 89.5 (358/400) | [86.1, 92.3] |
| Qwen3.5-397B-A17B | Qwen3.5 | 397B | MoE | 17B | 4-bit | MLX | 223.9 | 90.7 | 87.2 (349/400) | [83.6, 90.4] |
| Qwen3.5-27B | Qwen3.5 | 27B | Dense | 27B | 8-bit | MLX | 29.5 | 89.3 | 87.2 (349/400) | [83.6, 90.4] |
| gpt-oss-120B | GPT-OSS | 120B | MoE | — | 8-bit | MLX | 124.2 | 92.0 | 84.5 (338/400) | [80.6, 87.9] |
| gpt-oss-120B | GPT-OSS | 120B | MoE | — | MXFP4 | GGUF | 63.4 | 90.7 | 84.0 (336/400) | [80.0, 87.5] |
| Qwen3-Next-80B | Qwen3 | 80B | MoE | 3B | 8-bit | MLX | 84.7 | 85.3 | 83.5 (334/400) | [79.5, 87.0] |
| Qwen3-VL-32B | Qwen3-VL | 32B | Dense (VL) | 32B | 8-bit | MLX | 19.6 | 82.7 | 82.8 (331/400) | [78.7, 86.3] |
| Nemotron-3-Nano | Nemotron | 8B | Hybrid† | — | 8-bit | MLX | 33.6 | 84.0 | 80.2 (321/400) | [76.0, 84.0] |
| Qwen3-32B | Qwen3 | 32B | Dense | 32B | 8-bit | MLX | 34.8 | 80.0 | 79.2 (317/400) | [74.9, 83.1] |
| Qwen3-32B | Qwen3 | 32B | Dense | 32B | 4-bit | MLX | 18.5 | 78.7 | 78.8 (315/400) | [74.4, 82.7] |
| Qwen3-30B-A3B-2507 | Qwen3 | 30B | MoE | 3B | 8-bit | MLX | 32.5 | 78.7 | 78.8 (315/400) | [74.4, 82.7] |
| Llama-3.3-Swallow-70B | Llama (JP-FT) | 70B | Dense | 70B | 8-bit | MLX | 40.4 | 81.3 | 78.0 (312/400) | [73.6, 82.0] |
| Qwen3-VL-30B | Qwen3-VL | 30B | Dense (VL) | 30B | 8-bit | MLX | 33.5 | 74.7 | 77.8 (311/400) | [73.4, 81.7] |
| Llama 4 Scout | Llama 4 | 109B | MoE (17B×16E) | 17B | 4-bit | MLX | 63.9 | 76.0 | 77.5 (310/400) | [73.1, 81.5] |
| Mistral-Small-3.2 | Mistral | 24B | Dense | 24B | 8-bit | MLX | 25.9 | 76.0 | 76.8 (307/400) | [72.3, 80.8] |
| Mistral-Large-2407 | Mistral | 123B | Dense | 123B | 8-bit | MLX | 130.3 | 77.3 | 75.8 (303/400) | [71.2, 79.9] |
| Qwen3-235B-A22B | Qwen3 | 235B | MoE | 22B | 8-bit | MLX | 132.3 | 88.0 | 75.0 (300/400) | [70.5, 79.2] |
| GPT-OSS-Swallow-20B | GPT-OSS (JP-FT) | 20B | MoE | — | 8-bit | MLX | 45.0 | 80.0 | 74.5 (298/400) | [69.9, 78.7] |
| Magistral-Small (old) | Mistral | 24B | Dense | 24B | 8-bit | MLX | 47.2 | 76.0 | 74.2 (297/400) | [69.7, 78.5] |
| Shisa-v2.1-Llama3.3-70B | Llama (JP-FT) | 70B | Dense | 70B | 8-bit | MLX | 75.0 | 76.0‡ | 74.2 (297/400) | [69.7, 78.5] |
| Magistral-Small-2509 | Mistral | 24B | Dense | 24B | 8-bit | MLX | 47.2 | 76.0 | 74.0 (296/400) | [69.4, 78.2] |
| Qwen3-14B | Qwen3 | 14B | Dense | 14B | 8-bit | MLX | 15.7 | 73.3 | 71.8 (287/400) | [67.1, 76.1] |
| gpt-oss-20B | GPT-OSS | 20B | MoE | — | 8-bit | MLX | 22.3 | 77.3 | 71.5 (286/400) | [66.8, 75.9] |
| gpt-oss-20B | GPT-OSS | 20B | MoE | — | MXFP4 | GGUF | 12.1 | 76.0 | 71.0 (284/400) | [66.3, 75.4] |
| gpt-oss-20B | GPT-OSS | 20B | MoE | — | MXFP4+Q8 | MLX | 12.1 | 76.0 | 71.0 (284/400) | [66.3, 75.4] |
| Llama-3.3-70B-Instruct | Llama | 70B | Dense | 70B | 8-bit | MLX | 40.4 | 68.0 | 71.0 (284/400) | [66.3, 75.4] |
| MedGemma-27B | Gemma (Medical) | 27B | Dense | 27B | 8-bit | MLX | 16.0 | 76.0 | 67.8 (271/400) | [62.9, 72.3] |
| Gemma-3-27B | Gemma | 27B | Dense | 27B | 8-bit | MLX | 16.9 | 74.7 | 67.8 (271/400) | [62.9, 72.3] |
| Qwen3-VL-8B | Qwen3-VL | 8B | Dense (VL) | 8B | 8-bit | MLX | 9.9 | 60.0 | 69.8 (279/400) | [65.0, 74.2] |
| Qwen3-VL-8B | Qwen3-VL | 8B | Dense (VL) | 8B | 4-bit | MLX | 5.8 | — | 65.2 (261/400) | [60.4, 69.9] |
| Phi-4 | Phi | 14B | Dense | 14B | 8-bit | MLX | 15.6 | 56.0 | 62.8 (251/400) | [57.8, 67.5] |
| Qwen3-VL-4B | Qwen3-VL | 4B | Dense (VL) | 4B | 8-bit | MLX | 5.1 | 52.0 | 60.5 (242/400) | [55.5, 65.3] |
| Qwen3-VL-4B | Qwen3-VL | 4B | Dense (VL) | 4B | 4-bit | MLX | 3.0 | — | 58.2 (233/400) | [53.2, 63.1] |

## Section A Only Models (not evaluated on full 400Q)

| Model | Family | Params | Architecture | Active Params | Quant. | Format | Memory (GB) | Section A (%) |
|-------|--------|--------|-------------|---------------|--------|--------|------------|--------------|
| Qwen3-235B-A22B-2507 | Qwen3 | 235B | MoE | 22B | 8-bit | MLX | 249.8 | 88.0 |
| Qwen3-8B | Qwen3 | 8B | Dense | 8B | 8-bit | MLX | 9.4 | 61.3 |
| GLM-4.6V-Flash | GLM | — | Dense (VL) | — | 8-bit | MLX | 11.8 | 61.3 |
| EZO-Gemma-3-12B | Gemma (JP-FT) | 12B | Dense | 12B | 8-bit | MLX | 8.1 | 60.0 |
| OLMo-3-32B-Think | OLMo | 32B | Dense† | 32B | 8-bit | MLX | 34.3 | 57.3 |
| Phi-4-Reasoning-Plus | Phi | 14B | Dense† | 14B | 4-bit | MLX | 8.3 | 56.0 |
| Gemma-3-12B | Gemma | 12B | Dense | 12B | 8-bit | MLX | 8.1 | 54.7 |
| InternVL3.5-8B | InternVL | 8B | Dense (VL) | 8B | 8-bit | MLX | 9.5 | 54.7 |
| Qwen3-4B-2507 | Qwen3 | 4B | Dense | 4B | 8-bit | MLX | 5.2 | 54.7 |
| Llama-3.1-Swallow-8B | Llama (JP-FT) | 8B | Dense | 8B | BF16 | MLX | 16.1 | 53.3 |
| ELYZA-JP-8B | Llama (JP-FT) | 8B | Dense | 8B | 8-bit | MLX | 8.0 | 44.0 |
| LLM-JP-3.1-13B | LLM-JP (JP) | 13B | Dense | 13B | BF16 | MLX | 27.4 | 40.0 |
| LFM2-24B-A2B | Liquid | 24B | MoE | 2B | 8-bit | MLX | 23.6 | 36.0 |
| MedGemma-4B | Gemma (Medical) | 4B | Dense | 4B | BF16 | MLX | 8.6 | 29.3 |
| MedGemma-4B | Gemma (Medical) | 4B | Dense | 4B | 8-bit | MLX | 4.4 | 18.7 |
| LFM2.5-1.2B | Liquid | 1.2B | Dense | 1.2B | 8-bit | MLX | 1.3 | 28.0 |

## Notes

- † = Thinking/reasoning model (requires max_tokens ≥ 4096 and timeout ≥ 300s)
- ‡ = Shisa-v2.1 Section A score from format_strict prompt (61.3% with baseline; subsequent full evaluation used format_strict)
- JP-FT = Japanese fine-tuned variant
- VL = Vision-Language model (evaluated on text-only questions)
- MoE active parameters: Only a subset of total parameters are active per inference token
- Memory footprint measured from LM Studio's model information display
- 95% CI = Clopper-Pearson exact 95% confidence interval (full 400Q models only)
- All evaluations used temperature=0 and 2-shot in-context examples from the 100th JMLE
- "—" in Active Params indicates the information is not publicly available for that model
- Section A = 75 questions; Full 400Q = all six sections (A–F) of the 116th JMLE

## Evaluation Failures (excluded from analysis)

| Model | Reason |
|-------|--------|
| GLM-4.7-Flash (31.8 GB) | LM Studio API crash after ~11 questions |
| Qwen3.5-35B-A3B (37.8 GB) | All outputs "c,e" regardless of question (MoE compatibility issue) |
| Fallen-Command-A-111B (48.6 GB) | All outputs contain "d,e" multi-selection (3-bit quantization artifact) |
| InternLM3-8B-Instruct | API instability (16% completion rate) |
| InternVL3-14B | All API requests failed |
| PLaMo-13B | All API requests failed |
| MiniMax-M2.5 (128.7 GB) | Irrecoverable verbose analysis mode (best 34.7%, evaluation stopped) |
