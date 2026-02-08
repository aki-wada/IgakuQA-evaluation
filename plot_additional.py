#!/usr/bin/env python3
"""
Additional IgakuQA Evaluation Visualizations
- Model ranking bar chart
- Prompt effectiveness heatmap
- max_tokens impact on reasoning models
- gpt-oss-20b variant comparison
- Updated section heatmap with gpt-oss-20b
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['axes.unicode_minus'] = False

# ========== Figure 1: Model Ranking Bar Chart ==========
print("Generating: model_ranking.png ...")

# (model_name, best_accuracy%, category, note)
ranking_data = [
    ("gpt-oss-120b MLX 8bit", 92.0, "gpt-oss", "mt=1024"),
    ("gpt-oss-120b GGUF", 90.7, "gpt-oss", "mt=1024"),
    ("qwen3-235b-2507", 88.0, "qwen3", ""),
    ("qwen3-235b-a22b", 88.0, "qwen3", ""),
    ("qwen3-next-80b", 85.3, "qwen3", "MoE, mt=1024"),
    ("qwen3-vl-32b", 82.7, "qwen3-vl", ""),
    ("Swallow-70b", 81.3, "jp-ft", "JP-FT"),
    ("qwen3-32b", 80.0, "qwen3", ""),
    ("gpt-oss-20b", 77.3, "gpt-oss", "mt=1024"),
    ("mistral-large", 77.3, "other", ""),
    ("medgemma-27b", 76.0, "gemma", "few-shot"),
    ("mistral-small", 76.0, "other", ""),
    ("qwen3-vl-30b", 74.7, "qwen3-vl", ""),
    ("gemma-3-27b", 74.7, "gemma", ""),
    ("qwen3-14b", 73.3, "qwen3", ""),
    ("llama-3.3-70b", 68.0, "llama", ""),
    ("shisa-v2-70b", 61.3, "jp-ft", "JP-FT (-)"),
    ("qwen3-8b", 61.3, "qwen3", ""),
    ("ezo2.5-12b", 60.0, "jp-ft", "JP-FT"),
    ("qwen3-vl-8b", 60.0, "qwen3-vl", ""),
    ("phi-4", 56.0, "other", ""),
    ("gemma-3-12b", 54.7, "gemma", ""),
    ("internvl3_5-8b", 54.7, "other", ""),
    ("qwen3-4b", 54.7, "qwen3", ""),
    ("Swallow-8b", 53.3, "jp-ft", "JP-FT"),
    ("qwen3-vl-4b", 52.0, "qwen3-vl", ""),
    ("elyza-jp-8b", 44.0, "jp-ft", "JP-FT"),
    ("medgemma-4b@bf16", 29.3, "gemma", ""),
    ("lfm2.5-1.2b", 28.0, "other", ""),
    ("medgemma-4b", 18.7, "gemma", ""),
]

cat_colors = {
    "qwen3": "#2196F3",
    "qwen3-vl": "#03A9F4",
    "gemma": "#4CAF50",
    "gpt-oss": "#FF9800",
    "llama": "#795548",
    "jp-ft": "#F44336",
    "other": "#607D8B",
}

fig, ax = plt.subplots(figsize=(12, 12))

names = [d[0] for d in ranking_data]
accs = [d[1] for d in ranking_data]
colors = [cat_colors[d[2]] for d in ranking_data]
notes = [d[3] for d in ranking_data]

y_pos = np.arange(len(names))
bars = ax.barh(y_pos, accs, color=colors, edgecolor='black', linewidth=0.3, height=0.7)

# Pass line
ax.axvline(x=75, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Pass Line (75%)')
ax.axvline(x=58, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='ChatGPT (58%)')
ax.axvline(x=80, color='blue', linestyle=':', linewidth=1, alpha=0.4, label='GPT-4 (80%)')

# Annotate accuracy values
for i, (acc, note) in enumerate(zip(accs, notes)):
    suffix = f"  ({note})" if note else ""
    color = '#1B5E20' if acc >= 75 else '#333333'
    fontw = 'bold' if acc >= 75 else 'normal'
    ax.text(acc + 0.5, i, f'{acc}%{suffix}', va='center', fontsize=8,
            color=color, fontweight=fontw)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Best Accuracy (%)', fontsize=12)
ax.set_title('IgakuQA 2022-A: All Models Ranked by Accuracy\n(30 models, Pass Line = 75%)',
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 105)
ax.invert_yaxis()

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=cat_colors["qwen3"], label="Qwen3"),
    Patch(facecolor=cat_colors["qwen3-vl"], label="Qwen3-VL"),
    Patch(facecolor=cat_colors["gemma"], label="Gemma / medgemma"),
    Patch(facecolor=cat_colors["gpt-oss"], label="GPT-OSS"),
    Patch(facecolor=cat_colors["jp-ft"], label="JP Fine-Tuned"),
    Patch(facecolor=cat_colors["llama"], label="Llama"),
    Patch(facecolor=cat_colors["other"], label="Others"),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3, axis='x')
ax.set_axisbelow(True)

# Shade pass region
ax.axvspan(75, 105, alpha=0.05, color='green')

plt.tight_layout()
plt.savefig('plots/model_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/model_ranking.png")

# ========== Figure 2: Prompt Effectiveness Heatmap ==========
print("Generating: prompt_effectiveness.png ...")

# Data: (model_name, baseline, format_strict(A), chain_of_thought(B), japanese_medical(C))
# All from Section A, 75 questions
prompt_data = [
    ("gpt-oss-120b MLX\n(mt=1024)", 85.3, 92.0, 90.7, 89.3),
    ("gpt-oss-120b GGUF\n(mt=1024)", 88.0, 85.3, 88.0, 90.7),
    ("qwen3-235b-2507\n(mt=1024)", 86.7, 86.7, 88.0, 88.0),
    ("qwen3-next-80b\n(mt=1024)", 66.7, 82.7, 84.0, 85.3),
    ("qwen3-vl-32b", 82.7, 80.0, 82.7, 80.0),
    ("Swallow-70b", 81.3, 77.3, 81.3, 77.3),
    ("qwen3-32b", 80.0, 77.3, 80.0, 78.7),
    ("gpt-oss-20b\n(mt=1024)", 74.7, 77.3, 76.0, 69.3),
    ("medgemma-27b\n(few-shot)", 76.0, 70.7, 56.0, 74.7),
    ("gemma-3-27b", 73.3, 72.0, 74.7, 73.3),
    ("qwen3-14b", 68.0, 73.3, 73.3, 72.0),
    ("llama-3.3-70b", 68.0, 68.0, 68.0, 66.7),
    ("qwen3-8b", 58.7, 61.3, 6.7, 60.0),
    ("qwen3-4b", 52.0, 50.7, 52.0, 54.7),
]

fig2, ax2 = plt.subplots(figsize=(10, 10))

model_names_p = [d[0] for d in prompt_data]
prompts = ["Baseline", "A: Format\nStrict", "B: Chain of\nThought", "C: Japanese\nMedical"]
matrix = np.array([[d[1], d[2], d[3], d[4]] for d in prompt_data])

# Custom colormap
cmap = LinearSegmentedColormap.from_list('prompt_cmap',
    [(0.0, '#E53935'), (0.4, '#FFCDD2'), (0.6, '#FFF9C4'),
     (0.75, '#C8E6C9'), (1.0, '#1B5E20')])

im = ax2.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=95)

ax2.set_xticks(np.arange(len(prompts)))
ax2.set_xticklabels(prompts, fontsize=10, fontweight='bold')
ax2.set_yticks(np.arange(len(model_names_p)))
ax2.set_yticklabels(model_names_p, fontsize=9)

# Cell annotations
for i in range(len(model_names_p)):
    # Find best prompt for this model
    best_idx = np.argmax(matrix[i])
    for j in range(len(prompts)):
        val = matrix[i, j]
        color = 'white' if val >= 85 or val < 20 else 'black'
        fontw = 'bold' if j == best_idx else 'normal'
        marker = ' *' if j == best_idx else ''
        ax2.text(j, i, f'{val:.1f}%{marker}', ha='center', va='center',
                 fontsize=9, color=color, fontweight=fontw)

ax2.set_title('IgakuQA 2022-A: Prompt Effectiveness by Model\n(* = best prompt for each model)',
              fontsize=13, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
cbar.set_label('Accuracy (%)', fontsize=10)

# Horizontal divider
ax2.axhline(y=7.5, color='white', linewidth=3)  # Pass/fail divide

plt.tight_layout()
plt.savefig('plots/prompt_effectiveness.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/prompt_effectiveness.png")

# ========== Figure 3: max_tokens Impact on Reasoning Models ==========
print("Generating: max_tokens_impact.png ...")

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

# --- (a) gpt-oss-120b GGUF ---
ax_a = axes3[0]
mt_tokens = [50, 200, 1024]
mt_labels = ["50", "200", "1024"]

# gpt-oss-120b GGUF data (best prompt at each mt)
gguf_baseline = [29.3, 84.0, 88.0]
gguf_a = [26.7, 82.7, 85.3]
gguf_b = [85.3, 85.3, 88.0]
gguf_c = [33.3, 84.0, 90.7]

ax_a.plot(mt_labels, gguf_baseline, 'o-', color='#607D8B', markersize=8, linewidth=2, label='Baseline')
ax_a.plot(mt_labels, gguf_a, 's--', color='#FF9800', markersize=8, linewidth=2, label='A: Format')
ax_a.plot(mt_labels, gguf_b, 'D-.', color='#2196F3', markersize=8, linewidth=2, label='B: CoT')
ax_a.plot(mt_labels, gguf_c, '^:', color='#4CAF50', markersize=8, linewidth=2, label='C: JP Medical')

ax_a.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax_a.fill_between(mt_labels, 75, 95, alpha=0.05, color='green')
ax_a.set_xlabel('max_tokens', fontsize=11)
ax_a.set_ylabel('Accuracy (%)', fontsize=11)
ax_a.set_title('(a) gpt-oss-120b GGUF\nBest: 90.7% (C, mt=1024)', fontsize=11, fontweight='bold')
ax_a.set_ylim(20, 95)
ax_a.legend(fontsize=8, loc='lower right')
ax_a.grid(True, alpha=0.3)
ax_a.set_axisbelow(True)

# --- (b) gpt-oss-20b ---
ax_b = axes3[1]

oss20_baseline = [32.0, None, 74.7]  # mt=200 not separately tested
oss20_a = [26.7, None, 77.3]
oss20_b = [68.0, None, 76.0]
oss20_c = [33.3, None, 69.3]

# Only plot 50 and 1024
mt_2 = ["50", "1024"]
ax_b.plot(mt_2, [32.0, 74.7], 'o-', color='#607D8B', markersize=8, linewidth=2, label='Baseline')
ax_b.plot(mt_2, [26.7, 77.3], 's--', color='#FF9800', markersize=8, linewidth=2, label='A: Format')
ax_b.plot(mt_2, [68.0, 76.0], 'D-.', color='#2196F3', markersize=8, linewidth=2, label='B: CoT')
ax_b.plot(mt_2, [33.3, 69.3], '^:', color='#4CAF50', markersize=8, linewidth=2, label='C: JP Medical')

ax_b.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax_b.fill_between(mt_2, 75, 95, alpha=0.05, color='green')
ax_b.set_xlabel('max_tokens', fontsize=11)
ax_b.set_ylabel('Accuracy (%)', fontsize=11)
ax_b.set_title('(b) gpt-oss-20b\nBest: 77.3% (A, mt=1024)', fontsize=11, fontweight='bold')
ax_b.set_ylim(20, 95)
ax_b.legend(fontsize=8, loc='center right')
ax_b.grid(True, alpha=0.3)
ax_b.set_axisbelow(True)

# Annotate dramatic improvement
ax_b.annotate('+50.6%', xy=(1, 77.3), xytext=(0.3, 85),
              fontsize=10, color='#1B5E20', fontweight='bold',
              arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.5))

# --- (c) medgemma-27b ---
ax_c = axes3[2]

# medgemma data: mt=50 (no few-shot) vs mt=512 (few-shot)
mt_med = ["50\n(no few-shot)", "512\n(prompt mod)", "512\n(+ few-shot)"]
med_vals = [76.0, 64.0, 76.0]  # Section A only: original, prompt mod, few-shot
# Full model comparison for total accuracy
med_total = [42.5, 67.8, 71.8]

ax_c.bar(np.arange(3), med_total, color=['#FFCDD2', '#FFF9C4', '#C8E6C9'],
         edgecolor='black', linewidth=0.5, width=0.6)
ax_c.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pass Line (75%)')

for i, val in enumerate(med_total):
    ax_c.text(i, val + 1.5, f'{val}%', ha='center', fontsize=11, fontweight='bold')

# Arrows showing improvement
ax_c.annotate('', xy=(1, 67.8), xytext=(0, 42.5),
              arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2))
ax_c.text(0.35, 53, '+25.2%', fontsize=9, color='#FF9800', fontweight='bold')
ax_c.annotate('', xy=(2, 71.8), xytext=(1, 67.8),
              arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
ax_c.text(1.35, 68, '+4.0%', fontsize=9, color='#4CAF50', fontweight='bold')

ax_c.set_xticks(np.arange(3))
ax_c.set_xticklabels(["mt=50\nno few-shot", "mt=512\nprompt mod", "mt=512\n+ few-shot"], fontsize=9)
ax_c.set_ylabel('Total Accuracy (%)', fontsize=11)
ax_c.set_title('(c) medgemma-27b (400Q Total)\nBest: 71.8% (FAIL)', fontsize=11, fontweight='bold')
ax_c.set_ylim(0, 85)
ax_c.legend(fontsize=8)
ax_c.grid(True, alpha=0.3, axis='y')
ax_c.set_axisbelow(True)

plt.suptitle('max_tokens & Few-shot Impact on Reasoning / Thinking Models',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/max_tokens_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/max_tokens_impact.png")

# ========== Figure 4: gpt-oss-20b Variant Comparison ==========
print("Generating: gpt_oss_20b_variants.png ...")

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 7))

# --- Left: Section-level comparison of 3 working variants ---
variants = [
    ("openai @8bit\nMLX (22.26GB)", [76.0, 78.0, 65.3, 76.0, 74.0, 58.7], 71.5, "#1976D2"),
    ("openai @mxfp4\nGGUF (12.11GB)", [76.0, 74.0, 65.3, 77.3, 74.0, 61.3], 71.0, "#FF9800"),
    ("mlx-community\nQ8 (12.10GB)", [74.7, 80.0, 62.7, 74.7, 72.0, 64.0], 71.0, "#4CAF50"),
]

sections = ["A", "B", "C", "D", "E", "F"]
x = np.arange(len(sections))
width = 0.25

for i, (name, scores, total, color) in enumerate(variants):
    offset = (i - 1) * width
    bars = ax4a.bar(x + offset, scores, width, label=f'{name} ({total}%)',
                    color=color, edgecolor='black', linewidth=0.3, alpha=0.85)

ax4a.axhline(y=75, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Pass Line')
ax4a.set_xticks(x)
ax4a.set_xticklabels(sections, fontsize=12)
ax4a.set_xlabel('Section', fontsize=11)
ax4a.set_ylabel('Accuracy (%)', fontsize=11)
ax4a.set_title('Working Variants: Section-Level Comparison', fontsize=12, fontweight='bold')
ax4a.set_ylim(50, 90)
ax4a.legend(fontsize=8, loc='lower left')
ax4a.grid(True, alpha=0.3, axis='y')
ax4a.set_axisbelow(True)

# --- Right: All 6 variants overview ---
all_variants = [
    ("openai @8bit\nMLX", 22.26, 71.5, "#1976D2", "OK"),
    ("openai @mxfp4\nGGUF", 12.11, 71.0, "#FF9800", "OK"),
    ("mlx-community\nMXFP4-Q8", 12.10, 71.0, "#4CAF50", "OK"),
    ("safeguard\nMLX MXFP4", 11.15, 36.0, "#BDBDBD", "BROKEN"),
    ("mlx-community\nMXFP4-Q4", 11.21, 9.3, "#BDBDBD", "BROKEN"),
    ("InferenceIll.\n4bit MLX", 11.80, 9.3, "#BDBDBD", "BROKEN"),
]

y_pos = np.arange(len(all_variants))
var_names = [v[0] for v in all_variants]
var_accs = [v[2] for v in all_variants]
var_colors = [v[3] for v in all_variants]
var_sizes = [v[1] for v in all_variants]

bars_r = ax4b.barh(y_pos, var_accs, color=var_colors, edgecolor='black',
                   linewidth=0.3, height=0.6)

ax4b.axvline(x=75, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Pass Line')

for i, (acc, size, status) in enumerate(zip(var_accs, var_sizes, [v[4] for v in all_variants])):
    if status == "OK":
        ax4b.text(acc + 1, i, f'{acc}%  ({size:.1f}GB)', va='center',
                  fontsize=9, fontweight='bold', color='#1B5E20')
    else:
        ax4b.text(acc + 1, i, f'{acc}%  ({size:.1f}GB)  BROKEN', va='center',
                  fontsize=9, fontweight='bold', color='#B71C1C')

ax4b.set_yticks(y_pos)
ax4b.set_yticklabels(var_names, fontsize=9)
ax4b.set_xlabel('Accuracy (%)', fontsize=11)
ax4b.set_title('All 6 Variants: Format/Quantization Impact', fontsize=12, fontweight='bold')
ax4b.set_xlim(0, 95)
ax4b.invert_yaxis()
ax4b.legend(fontsize=9)
ax4b.grid(True, alpha=0.3, axis='x')
ax4b.set_axisbelow(True)

# Divider between working and broken
ax4b.axhline(y=2.5, color='red', linewidth=1.5, linestyle='-', alpha=0.5)
ax4b.text(45, 2.7, 'MLX Q4 breaks reasoning', fontsize=8, color='#B71C1C',
          style='italic', ha='center')

plt.suptitle('gpt-oss-20b: Format & Quantization Variant Comparison (400 Questions)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/gpt_oss_20b_variants.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/gpt_oss_20b_variants.png")

# ========== Figure 5: Updated Section Heatmap (with gpt-oss-20b) ==========
print("Generating: section_heatmap_full.png ...")

fig5, ax5 = plt.subplots(figsize=(12, 10))

section_data = {
    "qwen3-32b 8bit":          [80.0, 86.0, 68.0, 86.7, 84.0, 74.7],
    "qwen3-32b 4bit":          [78.7, 84.0, 68.0, 85.3, 84.0, 76.0],
    "gpt-oss-20b @8bit MLX":   [76.0, 78.0, 65.3, 76.0, 74.0, 58.7],
    "gpt-oss-20b @mxfp4 GGUF": [76.0, 74.0, 65.3, 77.3, 74.0, 61.3],
    "gpt-oss-20b mlx-com Q8":  [74.7, 80.0, 62.7, 74.7, 72.0, 64.0],
    "medgemma-27b":             [76.0, 82.0, 61.3, 76.0, 76.0, 64.0],
    "qwen3-vl-8b 8bit":        [62.7, 76.0, 60.0, 73.3, 76.0, 74.7],
    "qwen3-vl-8b 4bit":        [56.0, 68.0, 56.0, 70.7, 72.0, 72.0],
    "qwen3-vl-4b 8bit":        [58.7, 68.0, 50.7, 69.3, 70.0, 52.0],
    "qwen3-vl-4b 4bit":        [49.3, 72.0, 54.7, 69.3, 62.0, 48.0],
}

# Compute totals with proper weighting (A=75, B=50, C=75, D=75, E=50, F=75)
q_counts = [75, 50, 75, 75, 50, 75]
section_totals = {}
for model, accs in section_data.items():
    correct = sum(acc / 100 * q for acc, q in zip(accs, q_counts))
    section_totals[model] = correct / 400 * 100

model_names = list(section_data.keys())
sections_labels = ["A\n(75)", "B\n(50)", "C\n(75)", "D\n(75)", "E\n(50)", "F\n(75)", "Total\n(400)"]
data_matrix = np.array([section_data[m] + [section_totals[m]] for m in model_names])

cmap = LinearSegmentedColormap.from_list('custom',
    [(0.0, '#FFCDD2'), (0.5, '#FFF9C4'), (0.75, '#C8E6C9'), (1.0, '#1B5E20')])

im = ax5.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=40, vmax=90)

ax5.set_xticks(np.arange(len(sections_labels)))
ax5.set_xticklabels(sections_labels, fontsize=11)
ax5.set_yticks(np.arange(len(model_names)))
ax5.set_yticklabels([f'{n} ({section_totals[n]:.1f}%)' for n in model_names], fontsize=9)

for i in range(len(model_names)):
    for j in range(len(sections_labels)):
        val = data_matrix[i, j]
        color = 'white' if val >= 78 or val < 52 else 'black'
        fontw = 'bold' if j == len(sections_labels) - 1 else 'normal'
        ax5.text(j, i, f'{val:.1f}%', ha='center', va='center',
                 fontsize=9, color=color, fontweight=fontw)

# Pass/fail annotation
for i, m in enumerate(model_names):
    total = section_totals[m]
    status = "PASS" if total >= 75 else "FAIL"
    scolor = '#1B5E20' if total >= 75 else '#B71C1C'
    ax5.text(len(sections_labels) - 0.35, i, status, ha='left', va='center',
             fontsize=8, color=scolor, fontweight='bold')

ax5.set_title('IgakuQA 2022: Full Section Accuracy Heatmap (10 Models)\n'
              '(Pass Line: 75% = 300/400)',
              fontsize=13, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax5, fraction=0.03, pad=0.04)
cbar.set_label('Accuracy (%)', fontsize=10)

# Divider lines between model groups
ax5.axhline(y=1.5, color='white', linewidth=2)   # qwen3-32b / gpt-oss-20b
ax5.axhline(y=4.5, color='white', linewidth=2)    # gpt-oss-20b / medgemma
ax5.axhline(y=5.5, color='white', linewidth=2)    # medgemma / qwen3-vl-8b
ax5.axhline(y=7.5, color='white', linewidth=2)    # qwen3-vl-8b / qwen3-vl-4b

plt.tight_layout()
plt.savefig('plots/section_heatmap_full.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/section_heatmap_full.png")

# ========== Figure 6: Section Difficulty Analysis ==========
print("Generating: section_difficulty.png ...")

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Average accuracy by section across all fully-evaluated models ---
all_models_sections = {
    "qwen3-32b 8bit":  [80.0, 86.0, 68.0, 86.7, 84.0, 74.7],
    "qwen3-32b 4bit":  [78.7, 84.0, 68.0, 85.3, 84.0, 76.0],
    "gpt-oss-20b 8bit": [76.0, 78.0, 65.3, 76.0, 74.0, 58.7],
    "gpt-oss-20b GGUF": [76.0, 74.0, 65.3, 77.3, 74.0, 61.3],
    "gpt-oss-20b Q8":  [74.7, 80.0, 62.7, 74.7, 72.0, 64.0],
    "medgemma-27b":     [76.0, 82.0, 61.3, 76.0, 76.0, 64.0],
    "qwen3-vl-8b 8bit": [62.7, 76.0, 60.0, 73.3, 76.0, 74.7],
    "qwen3-vl-8b 4bit": [56.0, 68.0, 56.0, 70.7, 72.0, 72.0],
    "qwen3-vl-4b 8bit": [58.7, 68.0, 50.7, 69.3, 70.0, 52.0],
    "qwen3-vl-4b 4bit": [49.3, 72.0, 54.7, 69.3, 62.0, 48.0],
}

sec_names = ["A", "B", "C", "D", "E", "F"]
all_values = np.array(list(all_models_sections.values()))
avg_by_section = np.mean(all_values, axis=0)
std_by_section = np.std(all_values, axis=0)
min_by_section = np.min(all_values, axis=0)
max_by_section = np.max(all_values, axis=0)

# Color by difficulty
colors_diff = ['#FF9800' if avg < 65 else '#FFC107' if avg < 72 else '#4CAF50' for avg in avg_by_section]

bars_d = ax6a.bar(np.arange(6), avg_by_section, color=colors_diff,
                  edgecolor='black', linewidth=0.5, width=0.6)
ax6a.errorbar(np.arange(6), avg_by_section, yerr=std_by_section,
              fmt='none', ecolor='black', capsize=5, capthick=1.5)

# Range markers
for i in range(6):
    ax6a.plot([i, i], [min_by_section[i], max_by_section[i]],
             color='#333333', linewidth=1, alpha=0.3)

ax6a.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pass Line')

for i, (avg, std) in enumerate(zip(avg_by_section, std_by_section)):
    ax6a.text(i, avg + std + 2, f'{avg:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax6a.text(i, avg - std - 3.5, f'SD={std:.1f}', ha='center', fontsize=7, color='#666666')

ax6a.set_xticks(np.arange(6))
ax6a.set_xticklabels([f'{s}\n({q})' for s, q in zip(sec_names, q_counts)], fontsize=10)
ax6a.set_ylabel('Average Accuracy (%)', fontsize=11)
ax6a.set_title('Section Difficulty\n(Mean ± SD across 10 models)', fontsize=12, fontweight='bold')
ax6a.set_ylim(40, 95)
ax6a.legend(fontsize=9)
ax6a.grid(True, alpha=0.3, axis='y')
ax6a.set_axisbelow(True)

# --- Right: Difficulty ranking ---
section_info = list(zip(sec_names, q_counts, avg_by_section, std_by_section))
section_info.sort(key=lambda x: x[2])  # Sort by difficulty (hardest first)

y_pos = np.arange(len(section_info))
sorted_names = [f'Section {s} ({q}Q)' for s, q, _, _ in section_info]
sorted_avgs = [a for _, _, a, _ in section_info]
sorted_stds = [s for _, _, _, s in section_info]
sorted_colors = ['#E53935' if a < 65 else '#FF9800' if a < 72 else '#4CAF50' for a in sorted_avgs]

bars_rank = ax6b.barh(y_pos, sorted_avgs, color=sorted_colors,
                      edgecolor='black', linewidth=0.5, height=0.6,
                      xerr=sorted_stds, capsize=4)

ax6b.axvline(x=75, color='red', linestyle='--', linewidth=1.5, alpha=0.6)

for i, (avg, std) in enumerate(zip(sorted_avgs, sorted_stds)):
    label = 'Hard' if avg < 65 else 'Medium' if avg < 72 else 'Easy'
    ax6b.text(avg + std + 1.5, i, f'{avg:.1f}% ({label})', va='center',
              fontsize=10, fontweight='bold')

ax6b.set_yticks(y_pos)
ax6b.set_yticklabels(sorted_names, fontsize=11)
ax6b.set_xlabel('Average Accuracy (%)', fontsize=11)
ax6b.set_title('Section Difficulty Ranking\n(Hardest → Easiest)', fontsize=12, fontweight='bold')
ax6b.set_xlim(40, 95)
ax6b.grid(True, alpha=0.3, axis='x')
ax6b.set_axisbelow(True)

plt.suptitle('IgakuQA 2022: Section Difficulty Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/section_difficulty.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/section_difficulty.png")

print("\nDone! 5 additional plots saved to plots/ directory.")
