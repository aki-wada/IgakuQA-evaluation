#!/usr/bin/env python3
"""
Statistical Analysis for IgakuQA Evaluation Paper

Performs:
1. Clopper-Pearson exact 95% CI for all models (400Q and 75Q)
2. McNemar's exact test for pre-specified pairwise comparisons
3. Bonferroni correction for multiple comparisons

Usage:
    python statistical_analysis.py
"""

import json
import glob
import os
import sys
from collections import OrderedDict

import numpy as np
from scipy.stats import binom, beta
from statsmodels.stats.contingency_tables import mcnemar

RESULTS_DIR = "results"
SECTIONS = ["A", "B", "C", "D", "E", "F"]
SECTION_SIZES = {"A": 75, "B": 50, "C": 75, "D": 75, "E": 50, "F": 75}  # total=400


# ============================================================
# Model file mapping: model_label -> {section -> (filepath, prompt_key)}
# ============================================================

# For Section A (prompt_comparison files): use the best prompt from summary
# For Sections B-F: single prompt_key in details

MODEL_FILES = OrderedDict({
    # --- Top performers ---
    "qwen3.5-397b-a17b@8bit": {
        "pattern": "prompt_comparison_qwen3.5-397b-a17b-8bit_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "qwen3.5-397b-a17b@4bit": {
        "pattern": "prompt_comparison_qwen3.5-397b-a17b@4bit_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "qwen3.5-27b@8bit": {
        "pattern": "prompt_comparison_qwen3.5-27b@8bit_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "gpt-oss-120b MLX 8bit": {
        "pattern": "prompt_comparison_openai_gpt-oss-120b_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "gpt-oss-120b GGUF": {
        "pattern": "gpt-oss-120b-gguf_fewshot_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "qwen3-next-80b": {
        "pattern": "prompt_comparison_qwen_qwen3-next-80b_2022_{section}.json",
        "section_a_prompt": "japanese_medical",
    },
    "qwen3-vl-32b": {
        "pattern": "prompt_comparison_qwen_qwen3-vl-32b_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "nemotron-3-nano": {
        "pattern": "prompt_comparison_nvidia_nemotron-3-nano_2022_{section}.json",
        "section_a_prompt": "answer_first",
    },
    "qwen3-32b@8bit": {
        "pattern": "qwen3-32b_fewshot_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "qwen3-32b@4bit": {
        "pattern": "qwen3-32b-4bit_fewshot_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "Swallow-70b": {
        "pattern": "prompt_comparison_tokyotech-llm-llama-3.3-swallow-70b-instruct-v0.4_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "qwen3-vl-30b": {
        "pattern": "prompt_comparison_qwen_qwen3-vl-30b_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "Llama-4-Scout-17Bx16E": {
        "pattern": "prompt_comparison_llama-4-scout-17b-16e_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "mistral-small-3.2": {
        "pattern": "prompt_comparison_mistralai_mistral-small-3.2_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "mistral-large-2407": {
        "pattern": "prompt_comparison_mistral-large-instruct-2407_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    # --- Near threshold ---
    "qwen3-235b-a22b": {
        "pattern": "prompt_comparison_qwen_qwen3-235b-a22b_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "GPT-OSS-Swallow-20B": {
        "pattern": "prompt_comparison_tokyotech-llm_GPT-OSS-Swallow-20B-RL-v0.1_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "magistral-small (old)": {
        "pattern": "prompt_comparison_mistralai_magistral-small_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "shisa-v2.1-70b": {
        "pattern": "prompt_comparison_shisa-v2.1-llama3.3-70b-mlx_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "magistral-small-2509": {
        "pattern": "prompt_comparison_mistralai_magistral-small-2509_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    # --- Below threshold ---
    "qwen3-14b": {
        "pattern": "prompt_comparison_qwen_qwen3-14b_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "gpt-oss-20b MLX 8bit": {
        "pattern": "gpt-oss-20b-gguf_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "gpt-oss-20b MXFP4 GGUF": {
        "pattern": "gpt-oss-20b-openai-mxfp4-gguf_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "gpt-oss-20b MLX-community": {
        "pattern": "gpt-oss-20b-mlx-community_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "llama-3.3-70b": {
        "pattern": "prompt_comparison_mlx-community_llama-3.3-70b-instruct_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "medgemma-27b": {
        "pattern": "medgemma-27b_mt512_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "gemma-3-27b": {
        "pattern": "prompt_comparison_google_gemma-3-27b_2022_{section}.json",
        "section_a_prompt": "chain_of_thought",
    },
    "qwen3-vl-8b@8bit": {
        "pattern": "qwen3-vl-8b-8bit_fewshot_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "qwen3-vl-8b@4bit": {
        "pattern": "qwen3-vl-8b-4bit_fewshot_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "phi-4": {
        "pattern": "prompt_comparison_mlx-community_phi-4_2022_{section}.json",
        "section_a_prompt": "format_strict",
    },
    "qwen3-vl-4b@8bit": {
        "pattern": "qwen3-vl-4b-8bit_fewshot_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
    "qwen3-vl-4b@4bit": {
        "pattern": "qwen3-vl-4b-4bit_fewshot_2022_{section}.json",
        "section_a_prompt": "baseline",
    },
})


# ============================================================
# Pre-specified pairwise comparisons for McNemar's test
# ============================================================

COMPARISONS = [
    # Category 1: Quantization pairs (8-bit vs 4-bit)
    ("qwen3.5-397b-a17b@8bit", "qwen3.5-397b-a17b@4bit", "Quantization"),
    ("qwen3-32b@8bit", "qwen3-32b@4bit", "Quantization"),
    ("qwen3-vl-8b@8bit", "qwen3-vl-8b@4bit", "Quantization"),
    ("qwen3-vl-4b@8bit", "qwen3-vl-4b@4bit", "Quantization"),
    # Category 1b: Format comparison (MLX vs GGUF)
    ("gpt-oss-120b MLX 8bit", "gpt-oss-120b GGUF", "Format"),

    # Category 2: Japanese fine-tuned vs base models
    ("Swallow-70b", "llama-3.3-70b", "Japanese FT"),
    ("shisa-v2.1-70b", "llama-3.3-70b", "Japanese FT"),
    ("GPT-OSS-Swallow-20B", "gpt-oss-20b MLX 8bit", "Japanese FT"),

    # Category 3: Medical-specialized vs base model (same family)
    ("medgemma-27b", "gemma-3-27b", "Medical spec."),

    # Category 4: Adjacent models near 75% threshold
    ("mistral-large-2407", "shisa-v2.1-70b", "Near threshold"),
    ("mistral-large-2407", "magistral-small-2509", "Near threshold"),
    ("Llama-4-Scout-17Bx16E", "mistral-small-3.2", "Near threshold"),
]


def clopper_pearson(k, n, alpha=0.05):
    """Compute exact Clopper-Pearson confidence interval."""
    if k == 0:
        lo = 0.0
    else:
        lo = beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def load_model_results(model_label, model_config):
    """Load per-question correct/incorrect results for all 400 questions.

    Returns: dict with 'correct_array' (400-element list of 0/1),
             'total_correct', 'total_questions', 'section_results'
    """
    pattern = model_config["pattern"]
    section_a_prompt = model_config["section_a_prompt"]

    all_results = []  # ordered list of (problem_id, correct)
    section_results = {}

    for section in SECTIONS:
        filepath = os.path.join(RESULTS_DIR, pattern.format(section=section))
        if not os.path.exists(filepath):
            print(f"  WARNING: Missing file {filepath} for {model_label}")
            return None

        with open(filepath) as f:
            data = json.load(f)

        details = data["details"]

        # For Section A with multiple prompt keys, use specified prompt
        if section == "A" and len(details) > 1:
            prompt_key = section_a_prompt
        else:
            # B-F: single prompt key
            prompt_key = list(details.keys())[0]

        if prompt_key not in details:
            print(f"  WARNING: prompt_key '{prompt_key}' not found in {filepath}")
            print(f"    Available: {list(details.keys())}")
            return None

        questions = details[prompt_key]
        section_correct = sum(1 for q in questions if q["correct"])
        section_total = len(questions)

        section_results[section] = {
            "correct": section_correct,
            "total": section_total,
            "prompt_key": prompt_key,
        }

        for q in questions:
            all_results.append((q["problem_id"], 1 if q["correct"] else 0))

    correct_array = [r[1] for r in all_results]
    problem_ids = [r[0] for r in all_results]

    return {
        "correct_array": correct_array,
        "problem_ids": problem_ids,
        "total_correct": sum(correct_array),
        "total_questions": len(correct_array),
        "section_results": section_results,
    }


def build_mcnemar_table(results_a, results_b):
    """Build 2x2 contingency table for McNemar's test.

    Returns: [[both_correct, a_only], [b_only, both_wrong]]
    """
    arr_a = results_a["correct_array"]
    arr_b = results_b["correct_array"]

    assert len(arr_a) == len(arr_b), "Arrays must have same length"

    both_correct = sum(1 for a, b in zip(arr_a, arr_b) if a == 1 and b == 1)
    a_only = sum(1 for a, b in zip(arr_a, arr_b) if a == 1 and b == 0)
    b_only = sum(1 for a, b in zip(arr_a, arr_b) if a == 0 and b == 1)
    both_wrong = sum(1 for a, b in zip(arr_a, arr_b) if a == 0 and b == 0)

    return [[both_correct, a_only], [b_only, both_wrong]]


def main():
    print("=" * 80)
    print("IgakuQA Statistical Analysis")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load all model results
    # --------------------------------------------------------
    print("\n[1] Loading model results...")
    model_results = {}
    for label, config in MODEL_FILES.items():
        result = load_model_results(label, config)
        if result is not None:
            model_results[label] = result
            n = result["total_questions"]
            k = result["total_correct"]
            print(f"  {label}: {k}/{n} = {k/n:.1%}")
        else:
            print(f"  SKIP: {label} (missing data)")

    print(f"\n  Loaded {len(model_results)} models with full 400Q data.")

    # --------------------------------------------------------
    # 2. Clopper-Pearson 95% CI for all models
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("[2] Clopper-Pearson Exact 95% Confidence Intervals (400 questions)")
    print("=" * 80)

    # Sort by accuracy descending
    sorted_models = sorted(model_results.items(),
                          key=lambda x: x[1]["total_correct"], reverse=True)

    print(f"\n{'Rank':<5} {'Model':<32} {'Score':>10} {'Accuracy':>9} {'95% CI':>20} {'Width':>7}")
    print("-" * 85)

    ci_results = {}
    for rank, (label, result) in enumerate(sorted_models, 1):
        k = result["total_correct"]
        n = result["total_questions"]
        acc = k / n
        lo, hi = clopper_pearson(k, n)
        ci_results[label] = (lo, hi)
        width = hi - lo

        # Mark models above/below 75% threshold
        marker = ""
        if lo >= 0.75:
            marker = " ***"  # CI entirely above 75%
        elif hi < 0.75:
            marker = ""
        elif acc >= 0.75:
            marker = " *"  # Point estimate above, but CI spans 75%

        print(f"{rank:<5} {label:<32} {k:>3}/{n:<4}  {acc:>7.1%}   [{lo:.1%}, {hi:.1%}]{marker}  {width:>5.1%}")

    print()
    print("  *** = 95% CI entirely above 75% reference threshold")
    print("  *   = Point estimate ≥75%, but CI spans threshold")

    # --------------------------------------------------------
    # 3. Section-level Clopper-Pearson CI for key models
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3] Section-Level 95% CI (Selected Models)")
    print("=" * 80)

    key_models = [
        "qwen3.5-397b-a17b@8bit", "gpt-oss-120b MLX 8bit",
        "qwen3-32b@8bit", "Swallow-70b",
        "mistral-large-2407", "shisa-v2.1-70b",
        "medgemma-27b", "llama-3.3-70b",
    ]

    for label in key_models:
        if label not in model_results:
            continue
        result = model_results[label]
        print(f"\n  {label}:")
        for section in SECTIONS:
            sr = result["section_results"][section]
            k, n = sr["correct"], sr["total"]
            lo, hi = clopper_pearson(k, n)
            print(f"    Section {section}: {k:>2}/{n:<2} = {k/n:>5.1%}  [{lo:.1%}, {hi:.1%}]")

    # --------------------------------------------------------
    # 4. McNemar's Exact Test for pre-specified comparisons
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[4] McNemar's Exact Test ({len(COMPARISONS)} pre-specified comparisons)")
    print(f"    Bonferroni-corrected α = 0.05 / {len(COMPARISONS)} = {0.05/len(COMPARISONS):.4f}")
    print("=" * 80)

    bonferroni_alpha = 0.05 / len(COMPARISONS)

    print(f"\n{'#':<3} {'Category':<16} {'Model A':<32} {'Model B':<32} {'Acc A':>6} {'Acc B':>6} {'Diff':>6} {'b':>4} {'c':>4} {'p-value':>10} {'Sig':>5}")
    print("-" * 140)

    comparison_results = []
    for i, (model_a, model_b, category) in enumerate(COMPARISONS, 1):
        if model_a not in model_results or model_b not in model_results:
            print(f"{i:<3} {category:<16} {model_a:<32} {model_b:<32} {'N/A':>6} {'N/A':>6}")
            continue

        res_a = model_results[model_a]
        res_b = model_results[model_b]

        table = build_mcnemar_table(res_a, res_b)
        # table = [[both_correct, a_only], [b_only, both_wrong]]
        a_only = table[0][1]  # b in McNemar notation
        b_only = table[1][0]  # c in McNemar notation

        # Use exact test (binomial) for small discordant counts
        n_discordant = a_only + b_only
        if n_discordant == 0:
            p_value = 1.0
        else:
            # exact McNemar: binomial test on discordant pairs
            result = mcnemar(table, exact=True)
            p_value = result.pvalue

        acc_a = res_a["total_correct"] / res_a["total_questions"]
        acc_b = res_b["total_correct"] / res_b["total_questions"]
        diff = acc_a - acc_b

        sig = ""
        if p_value < bonferroni_alpha:
            sig = "***"
        elif p_value < 0.05:
            sig = "*"

        comparison_results.append({
            "category": category,
            "model_a": model_a,
            "model_b": model_b,
            "acc_a": acc_a,
            "acc_b": acc_b,
            "diff": diff,
            "a_only": a_only,
            "b_only": b_only,
            "p_value": p_value,
            "significant_bonferroni": p_value < bonferroni_alpha,
            "significant_nominal": p_value < 0.05,
        })

        print(f"{i:<3} {category:<16} {model_a:<32} {model_b:<32} {acc_a:>5.1%} {acc_b:>5.1%} {diff:>+5.1%}  {a_only:>3}  {b_only:>3}  {p_value:>9.4f}  {sig:>4}")

    # --------------------------------------------------------
    # 5. Summary of significant results
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("[5] Summary of Significant Comparisons")
    print("=" * 80)

    sig_bonf = [r for r in comparison_results if r["significant_bonferroni"]]
    sig_nom = [r for r in comparison_results if r["significant_nominal"] and not r["significant_bonferroni"]]

    print(f"\n  Significant after Bonferroni correction (p < {bonferroni_alpha:.4f}): {len(sig_bonf)}")
    for r in sig_bonf:
        print(f"    [{r['category']}] {r['model_a']} vs {r['model_b']}: "
              f"{r['acc_a']:.1%} vs {r['acc_b']:.1%} (Δ={r['diff']:+.1%}, p={r['p_value']:.4f})")

    print(f"\n  Significant at nominal α=0.05 only (not after Bonferroni): {len(sig_nom)}")
    for r in sig_nom:
        print(f"    [{r['category']}] {r['model_a']} vs {r['model_b']}: "
              f"{r['acc_a']:.1%} vs {r['acc_b']:.1%} (Δ={r['diff']:+.1%}, p={r['p_value']:.4f})")

    not_sig = [r for r in comparison_results if not r["significant_nominal"]]
    print(f"\n  Not significant (p ≥ 0.05): {len(not_sig)}")
    for r in not_sig:
        print(f"    [{r['category']}] {r['model_a']} vs {r['model_b']}: "
              f"{r['acc_a']:.1%} vs {r['acc_b']:.1%} (Δ={r['diff']:+.1%}, p={r['p_value']:.4f})")

    # --------------------------------------------------------
    # 6. Key findings for paper
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("[6] Key Findings for Paper")
    print("=" * 80)

    print("\n  --- Quantization Impact ---")
    quant_comparisons = [r for r in comparison_results if r["category"] == "Quantization"]
    for r in quant_comparisons:
        sig_str = "significant" if r["significant_bonferroni"] else ("nominally sig." if r["significant_nominal"] else "not significant")
        print(f"    {r['model_a']} vs {r['model_b']}: {r['diff']:+.1%} ({sig_str}, p={r['p_value']:.4f})")

    print("\n  --- Japanese Fine-tuning ---")
    ft_comparisons = [r for r in comparison_results if r["category"] == "Japanese FT"]
    for r in ft_comparisons:
        sig_str = "significant" if r["significant_bonferroni"] else ("nominally sig." if r["significant_nominal"] else "not significant")
        print(f"    {r['model_a']} vs {r['model_b']}: {r['diff']:+.1%} ({sig_str}, p={r['p_value']:.4f})")

    print("\n  --- Medical Specialization ---")
    med_comparisons = [r for r in comparison_results if r["category"] == "Medical spec."]
    for r in med_comparisons:
        sig_str = "significant" if r["significant_bonferroni"] else ("nominally sig." if r["significant_nominal"] else "not significant")
        print(f"    {r['model_a']} vs {r['model_b']}: {r['diff']:+.1%} ({sig_str}, p={r['p_value']:.4f})")

    # --------------------------------------------------------
    # 7. Paper-ready text snippets
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("[7] Paper-Ready Text Snippets (English)")
    print("=" * 80)

    # Top model
    top_label, top_result = sorted_models[0]
    top_k = top_result["total_correct"]
    top_n = top_result["total_questions"]
    top_lo, top_hi = ci_results[top_label]
    print(f"\n  Top model:")
    print(f"    {top_label} achieved {top_k/top_n:.1%} ({top_k}/{top_n}; 95% CI: {top_lo:.1%}–{top_hi:.1%})")

    # Models with CI entirely above 75%
    print(f"\n  Models with 95% CI entirely above 75% threshold:")
    for label, result in sorted_models:
        lo, hi = ci_results[label]
        k = result["total_correct"]
        n = result["total_questions"]
        if lo >= 0.75:
            print(f"    {label}: {k/n:.1%} ({k}/{n}; 95% CI: {lo:.1%}–{hi:.1%})")

    # Quantization: largest pair
    print(f"\n  Quantization summary:")
    for r in quant_comparisons:
        print(f"    {r['model_a'].split('@')[0]}: 8-bit {r['acc_a']:.1%} vs 4-bit {r['acc_b']:.1%} "
              f"(Δ={r['diff']:+.1%}, McNemar p={r['p_value']:.3f})")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
