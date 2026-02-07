#!/usr/bin/env python3
"""
Model Size (Memory GB) vs IgakuQA Accuracy - Scatter & Scaling Analysis
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['axes.unicode_minus'] = False

# ========== DATA ==========
# (model_name, memory_size_GB, best_accuracy%, category)
models = [
    # Passing models (75%+)
    ("gpt-oss-120b\n(MLX mt=1024)", 124.20, 92.0, "gpt-oss"),
    ("qwen3-235b-2507", 249.80, 88.0, "qwen3"),
    ("qwen3-235b-a22b", 132.26, 88.0, "qwen3"),
    ("qwen3-next-80b\n(MoE A3B mt=1024)", 84.67, 85.3, "qwen3"),
    ("qwen3-vl-32b", 19.64, 82.7, "qwen3-vl"),
    ("swallow-70b", 40.35, 81.3, "swallow"),
    ("qwen3-32b", 34.83, 80.0, "qwen3"),
    ("gpt-oss-20b\n(mt=1024)", 22.26, 77.3, "gpt-oss"),
    ("mistral-large", 130.28, 77.3, "mistral"),
    ("medgemma-27b", 16.03, 76.0, "gemma"),
    ("mistral-small", 25.93, 76.0, "mistral"),
    # Near-pass
    ("qwen3-vl-30b", 33.53, 74.7, "qwen3-vl"),
    ("gemma-3-27b", 16.87, 74.7, "gemma"),
    ("qwen3-14b", 15.71, 73.3, "qwen3"),
    ("llama-3.3-70b", 39.71, 68.0, "llama"),
    # Mid-range
    ("shisa-v2-70b", 39.71, 61.3, "llama-jp"),
    ("qwen3-8b", 8.72, 61.3, "qwen3"),
    ("ezo2.5-12b", 6.94, 60.0, "gemma-jp"),
    ("qwen3-vl-8b", 5.78, 60.0, "qwen3-vl"),
    ("phi-4", 9.05, 56.0, "other"),
    ("gemma-3-12b", 14.45, 54.7, "gemma"),
    ("internvl3_5-8b", 5.71, 54.7, "other"),
    ("qwen3-4b", 2.28, 54.7, "qwen3"),
    ("swallow-8b", 16.08, 53.3, "swallow"),
    ("qwen3-vl-4b", 3.11, 52.0, "qwen3-vl"),
    # Low accuracy
    ("elyza-jp-8b", 4.92, 44.0, "llama-jp"),
    ("medgemma-4b@bf16", 9.98, 29.3, "gemma"),
    ("lfm2.5-1.2b", 1.25, 28.0, "other"),
    ("medgemma-4b", 3.44, 18.7, "gemma"),
    # Evaluation failures
    ("fallen-111b", 48.61, 1.3, "failure"),
    ("nemotron-30b", 33.58, 1.3, "failure"),
    ("glm-4.7-flash", 31.84, 0.0, "failure"),
    ("glm-4.6v-flash", 11.79, 2.7, "failure"),
]

# Category styles
category_style = {
    "qwen3":    {"color": "#2196F3", "marker": "o", "label": "Qwen3"},
    "qwen3-vl": {"color": "#03A9F4", "marker": "D", "label": "Qwen3-VL"},
    "gemma":    {"color": "#4CAF50", "marker": "s", "label": "Gemma / medgemma"},
    "gemma-jp": {"color": "#8BC34A", "marker": "^", "label": "EZO (Gemma JP-FT)"},
    "gpt-oss":  {"color": "#FF9800", "marker": "p", "label": "GPT-OSS"},
    "mistral":  {"color": "#9C27B0", "marker": "h", "label": "Mistral"},
    "swallow":  {"color": "#F44336", "marker": "*", "label": "Swallow (JP-FT)"},
    "llama":    {"color": "#795548", "marker": "v", "label": "Llama"},
    "llama-jp": {"color": "#E91E63", "marker": "<", "label": "Llama JP-FT (Shisa/ELYZA)"},
    "other":    {"color": "#607D8B", "marker": "X", "label": "Others"},
    "failure":  {"color": "#BDBDBD", "marker": "x", "label": "Eval Failure"},
}

# ========== Figure 1: Full Scatter Plot ==========
fig, ax = plt.subplots(figsize=(14, 9))

# Reference lines
ax.axhline(y=75, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Pass Line (75%)')
ax.axhline(y=58, color='orange', linestyle=':', linewidth=1.2, alpha=0.6, label='ChatGPT (58%)')
ax.axhline(y=80, color='blue', linestyle=':', linewidth=1.0, alpha=0.4, label='GPT-4 (80%)')

# Plot by category
plotted_categories = set()
for name, size_gb, acc, cat in models:
    style = category_style[cat]
    label = style["label"] if cat not in plotted_categories else None
    plotted_categories.add(cat)
    ax.scatter(size_gb, acc, c=style["color"], marker=style["marker"],
               s=120, edgecolors='black', linewidths=0.5, label=label, zorder=5)
    # Label positioning
    offset_x, offset_y = 1.5, 1.2
    fontsize = 7
    if "gpt-oss-120b" in name:
        offset_y = -3.0
    elif name == "qwen3-235b-a22b":
        offset_x = -22
        offset_y = -3.0
    elif name == "qwen3-235b-2507":
        offset_y = 3.0
    elif "qwen3-next-80b" in name:
        offset_x = -25
        offset_y = -4.0
    elif name in ("swallow-70b", "llama-3.3-70b", "shisa-v2-70b"):
        offset_x = -18
    elif name in ("mistral-large",):
        offset_x = -20
        offset_y = -3
    elif name in ("medgemma-27b", "gemma-3-27b"):
        offset_x = -12
    elif name in ("qwen3-32b",):
        offset_x = -8
        offset_y = -3.5
    elif "gpt-oss-20b" in name:
        offset_y = -4.0
    ax.annotate(name, (size_gb, acc), fontsize=fontsize,
                xytext=(offset_x, offset_y), textcoords='offset points',
                alpha=0.85)

ax.set_xlabel('Model Size (Memory GB)', fontsize=13)
ax.set_ylabel('Best Accuracy (%)', fontsize=13)
ax.set_title('IgakuQA 2022-A: Model Size (Memory) vs Accuracy', fontsize=15, fontweight='bold')
ax.set_xlim(-5, 265)
ax.set_ylim(-5, 97)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('plots/size_vs_accuracy_scatter.png', dpi=150, bbox_inches='tight')
print("Saved: plots/size_vs_accuracy_scatter.png")
plt.close()

# ========== Figure 2: Scaling Analysis (2x2) ==========
fig2, axes = plt.subplots(2, 2, figsize=(14, 11))

# --- (a) Qwen3 Series ---
ax1 = axes[0, 0]
qwen3_data = [
    ("4B", 4, 54.7),
    ("8B", 8, 61.3),
    ("14B", 14, 73.3),
    ("32B", 32, 80.0),
    ("235B\n(MoE)", 235, 88.0),
]
qwen3_vl_data = [
    ("VL-4B", 4, 52.0),
    ("VL-8B", 8, 60.0),
    ("VL-30B\n(MoE)", 30, 74.7),
    ("VL-32B", 32, 82.7),
]

x_q = [d[1] for d in qwen3_data]
y_q = [d[2] for d in qwen3_data]
x_vl = [d[1] for d in qwen3_vl_data]
y_vl = [d[2] for d in qwen3_vl_data]

ax1.plot(x_q, y_q, 'o-', color='#2196F3', markersize=10, linewidth=2, label='Qwen3', zorder=5)
ax1.plot(x_vl, y_vl, 'D--', color='#03A9F4', markersize=9, linewidth=2, label='Qwen3-VL', zorder=5)
ax1.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pass Line')
ax1.axhline(y=80, color='blue', linestyle=':', linewidth=1, alpha=0.3, label='GPT-4')

for name, x, y in qwen3_data:
    ax1.annotate(name, (x, y), fontsize=8, xytext=(5, 5), textcoords='offset points')
for name, x, y in qwen3_vl_data:
    ax1.annotate(name, (x, y), fontsize=8, xytext=(5, -12), textcoords='offset points', color='#0288D1')

ax1.set_xscale('log')
ax1.set_xlabel('Parameters (B)', fontsize=11)
ax1.set_ylabel('Best Accuracy (%)', fontsize=11)
ax1.set_title('(a) Qwen3 Scaling', fontsize=12, fontweight='bold')
ax1.set_ylim(45, 93)
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_axisbelow(True)

# --- (b) Gemma Series ---
ax2 = axes[0, 1]
gemma_base = [
    ("gemma-12B", 12, 54.7),
    ("gemma-27B", 27, 74.7),
]
medgemma = [
    ("medgemma-4B", 4, 18.7),
    ("medgemma-4B\n@bf16", 4, 29.3),
    ("medgemma-27B", 27, 76.0),
]
ezo = [
    ("ezo2.5-12B", 12, 60.0),
]

ax2.plot([d[1] for d in gemma_base], [d[2] for d in gemma_base], 'o-', color='#4CAF50',
         markersize=10, linewidth=2, label='Gemma-3 (base)', zorder=5)
ax2.plot([medgemma[0][1], medgemma[2][1]], [medgemma[0][2], medgemma[2][2]], 's--', color='#2E7D32',
         markersize=10, linewidth=2, label='medgemma (Medical FT)', zorder=5)
ax2.scatter(medgemma[1][1], medgemma[1][2], c='#2E7D32', marker='s', s=100, edgecolors='black',
            linewidths=0.5, zorder=6)
ax2.scatter(ezo[0][1], ezo[0][2], c='#8BC34A', marker='^', s=120, edgecolors='black',
            linewidths=0.5, label='EZO (JP-FT)', zorder=5)
ax2.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5)

for name, x, y in gemma_base:
    ax2.annotate(name, (x, y), fontsize=8, xytext=(5, 5), textcoords='offset points')
for name, x, y in medgemma:
    oy = -15 if 'bf16' in name else 5
    ax2.annotate(name, (x, y), fontsize=8, xytext=(5, oy), textcoords='offset points', color='#2E7D32')
for name, x, y in ezo:
    ax2.annotate(name, (x, y), fontsize=8, xytext=(5, 5), textcoords='offset points', color='#558B2F')

ax2.set_xlabel('Parameters (B)', fontsize=11)
ax2.set_ylabel('Best Accuracy (%)', fontsize=11)
ax2.set_title('(b) Gemma / medgemma Scaling', fontsize=12, fontweight='bold')
ax2.set_ylim(10, 85)
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_axisbelow(True)

# --- (c) Llama-3.3-70B JP Fine-Tuning Comparison ---
ax3 = axes[1, 0]

bars_70b = [
    ("Shisa-v2\n(JP-FT)", 61.3, '#E91E63'),
    ("llama-3.3-70b\n(base)", 68.0, '#795548'),
    ("Swallow\n(JP-FT)", 81.3, '#F44336'),
]

x_pos = np.arange(len(bars_70b))
colors_bar = [b[2] for b in bars_70b]
vals = [b[1] for b in bars_70b]
names_bar = [b[0] for b in bars_70b]

bars = ax3.bar(x_pos, vals, color=colors_bar, edgecolor='black', linewidth=0.5, width=0.6)
ax3.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pass Line')
ax3.axhline(y=68, color='grey', linestyle=':', linewidth=1, alpha=0.4, label='Base (68%)')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(names_bar, fontsize=9)
ax3.set_ylabel('Best Accuracy (%)', fontsize=11)
ax3.set_title('(c) llama-3.3-70B: JP Fine-Tuning Effect', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 90)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_axisbelow(True)

for bar, val in zip(bars, vals):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.annotate('', xy=(2, 81.3), xytext=(1, 68.0),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax3.text(1.7, 74, '+13.3%', fontsize=9, color='green', fontweight='bold')
ax3.annotate('', xy=(0, 61.3), xytext=(1, 68.0),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3.text(0.15, 64, '-6.7%', fontsize=9, color='red', fontweight='bold')

# --- (d) Family Scaling Comparison ---
ax4 = axes[1, 1]

gpt_oss = [("20B\n(mt=1024)", 20, 77.3), ("120B\n(mt=1024)", 120, 92.0)]
mistral_s = [("Small 24B", 24, 76.0), ("Large 123B", 123, 77.3)]
swallow_s = [("8B", 8, 53.3), ("70B", 70, 81.3)]

ax4.plot([d[1] for d in gpt_oss], [d[2] for d in gpt_oss], 'p-', color='#FF9800',
         markersize=12, linewidth=2, label='GPT-OSS (+14.7%)', zorder=5)
ax4.plot([d[1] for d in mistral_s], [d[2] for d in mistral_s], 'h-', color='#9C27B0',
         markersize=12, linewidth=2, label='Mistral (+1.3%)', zorder=5)
ax4.plot([d[1] for d in swallow_s], [d[2] for d in swallow_s], '*-', color='#F44336',
         markersize=14, linewidth=2, label='Swallow (+28%)', zorder=5)
ax4.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pass Line')

for name, x, y in gpt_oss:
    ax4.annotate(name, (x, y), fontsize=9, xytext=(5, 5), textcoords='offset points')
for name, x, y in mistral_s:
    ax4.annotate(name, (x, y), fontsize=9, xytext=(5, -15), textcoords='offset points', color='#7B1FA2')
for name, x, y in swallow_s:
    ax4.annotate(name, (x, y), fontsize=9, xytext=(5, 5), textcoords='offset points', color='#C62828')

ax4.set_xlabel('Parameters (B)', fontsize=11)
ax4.set_ylabel('Best Accuracy (%)', fontsize=11)
ax4.set_title('(d) Family Scaling Comparison', fontsize=12, fontweight='bold')
ax4.set_ylim(45, 95)
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(True, alpha=0.3)
ax4.set_axisbelow(True)

plt.suptitle('IgakuQA 2022-A (75 Questions): Model Family Scaling Analysis', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plots/scaling_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: plots/scaling_analysis.png")
plt.close()

# ========== Figure 3: Pareto Frontier + Best Model per Memory Budget ==========
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 7))

# --- Left: Pareto Frontier ---
# Only functional models (exclude eval failures)
func_models = [(n, s, a, c) for n, s, a, c in models if c != "failure"]
func_models.sort(key=lambda x: x[1])  # sort by size

# Compute Pareto frontier
pareto = []
best_acc = 0
for n, s, a, c in func_models:
    if a > best_acc:
        pareto.append((n, s, a, c))
        best_acc = a

# Plot all models
for name, size_gb, acc, cat in func_models:
    style = category_style[cat]
    ax5.scatter(size_gb, acc, c=style["color"], marker=style["marker"],
                s=80, edgecolors='black', linewidths=0.3, zorder=4, alpha=0.6)
    ax5.annotate(name, (size_gb, acc), fontsize=6, xytext=(2, 2),
                 textcoords='offset points', alpha=0.7)

# Pareto line
px = [p[1] for p in pareto]
py = [p[2] for p in pareto]
ax5.step(px, py, where='post', color='red', linewidth=2, alpha=0.8, label='Pareto Frontier', zorder=5)
for n, s, a, c in pareto:
    ax5.scatter(s, a, c='red', marker='o', s=150, edgecolors='darkred',
                linewidths=1.5, zorder=6)
    ax5.annotate(f'{n}\n({a}%)', (s, a), fontsize=8, fontweight='bold',
                 xytext=(5, -15), textcoords='offset points', color='red')

ax5.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.4, label='Pass Line (75%)')
ax5.set_xlabel('Model Size (Memory GB)', fontsize=12)
ax5.set_ylabel('Best Accuracy (%)', fontsize=12)
ax5.set_title('Pareto Frontier: Best Accuracy per Memory', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9, loc='lower right')
ax5.grid(True, alpha=0.3)
ax5.set_axisbelow(True)
ax5.set_ylim(10, 97)

# --- Right: Best Model per Memory Budget (bar chart) ---
budgets = [
    ("~3 GB", 3, [m for m in func_models if m[1] <= 3.5]),
    ("~5 GB", 5, [m for m in func_models if m[1] <= 6]),
    ("~10 GB", 10, [m for m in func_models if m[1] <= 10]),
    ("~16 GB", 16, [m for m in func_models if m[1] <= 17]),
    ("~20 GB", 20, [m for m in func_models if m[1] <= 23]),
    ("~35 GB", 35, [m for m in func_models if m[1] <= 41]),
    ("~50 GB", 50, [m for m in func_models if m[1] <= 50]),
    ("~85 GB", 85, [m for m in func_models if m[1] <= 85]),
    ("~135 GB", 135, [m for m in func_models if m[1] <= 135]),
    ("~250 GB", 250, [m for m in func_models if m[1] <= 250]),
]

budget_labels = []
budget_accs = []
budget_names = []
budget_colors = []

for label, limit, candidates in budgets:
    if candidates:
        best = max(candidates, key=lambda x: x[2])
        budget_labels.append(label)
        budget_accs.append(best[2])
        budget_names.append(f"{best[0]}\n({best[1]:.1f}GB)")
        budget_colors.append(category_style[best[3]]["color"])

y_pos = np.arange(len(budget_labels))
bars = ax6.barh(y_pos, budget_accs, color=budget_colors, edgecolor='black', linewidth=0.5, height=0.6)

ax6.axvline(x=75, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Pass Line (75%)')
ax6.axvline(x=58, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='ChatGPT (58%)')

for i, (acc, name) in enumerate(zip(budget_accs, budget_names)):
    color = 'green' if acc >= 75 else ('orange' if acc >= 58 else 'gray')
    ax6.text(acc + 0.8, i, f'{acc}%  {name}', fontsize=8, va='center', fontweight='bold', color=color)

ax6.set_yticks(y_pos)
ax6.set_yticklabels([f'Budget: {l}' for l in budget_labels], fontsize=10)
ax6.set_xlabel('Best Accuracy (%)', fontsize=12)
ax6.set_title('Best Model per Memory Budget', fontsize=13, fontweight='bold')
ax6.set_xlim(0, 105)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='x')
ax6.set_axisbelow(True)
ax6.invert_yaxis()

plt.suptitle('IgakuQA 2022-A: Memory-Constrained Model Selection Guide', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/pareto_and_budget.png', dpi=150, bbox_inches='tight')
print("Saved: plots/pareto_and_budget.png")
plt.close()

# ========== Figure 4: Quantization Comparison (8bit vs 4bit) ==========
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 7))

quant_models = [
    {
        "name": "qwen3-32b",
        "params": "32B",
        "mem_8bit": 34.8, "mem_4bit": 18.5,
        "sections": ["A", "B", "C", "D", "E", "F"],
        "acc_8bit": [80.0, 86.0, 68.0, 86.7, 84.0, 74.7],
        "acc_4bit": [78.7, 84.0, 68.0, 85.3, 84.0, 76.0],
        "total_8bit": 79.3, "total_4bit": 78.8,
    },
    {
        "name": "qwen3-vl-8b",
        "params": "8B VL",
        "mem_8bit": 9.9, "mem_4bit": 5.8,
        "sections": ["A", "B", "C", "D", "E", "F"],
        "acc_8bit": [62.7, 76.0, 60.0, 73.3, 76.0, 74.7],
        "acc_4bit": [56.0, 68.0, 56.0, 70.7, 72.0, 72.0],
        "total_8bit": 69.8, "total_4bit": 65.3,
    },
    {
        "name": "qwen3-vl-4b",
        "params": "4B VL",
        "mem_8bit": 5.1, "mem_4bit": 3.0,
        "sections": ["A", "B", "C", "D", "E", "F"],
        "acc_8bit": [58.7, 68.0, 50.7, 69.3, 70.0, 52.0],
        "acc_4bit": [49.3, 72.0, 54.7, 69.3, 62.0, 48.0],
        "total_8bit": 60.5, "total_4bit": 58.3,
    },
]

for i, qm in enumerate(quant_models):
    ax = axes4[i]
    x = np.arange(len(qm["sections"]))
    width = 0.35

    bars_8 = ax.bar(x - width/2, qm["acc_8bit"], width, label=f'8bit ({qm["mem_8bit"]}GB)',
                    color='#1976D2', edgecolor='black', linewidth=0.5, alpha=0.85)
    bars_4 = ax.bar(x + width/2, qm["acc_4bit"], width, label=f'4bit ({qm["mem_4bit"]}GB)',
                    color='#FF7043', edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pass Line (75%)')
    ax.set_xticks(x)
    ax.set_xticklabels(qm["sections"], fontsize=11)
    ax.set_xlabel('Section', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)

    diff = qm["total_4bit"] - qm["total_8bit"]
    sign = "+" if diff >= 0 else ""
    ax.set_title(f'{qm["name"]} ({qm["params"]})\n'
                 f'Total: {qm["total_8bit"]}% → {qm["total_4bit"]}% ({sign}{diff:.1f}%)',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(35, 95)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Annotate differences on each bar pair
    for j, (v8, v4) in enumerate(zip(qm["acc_8bit"], qm["acc_4bit"])):
        d = v4 - v8
        color = '#2E7D32' if d >= 0 else '#C62828'
        sign_s = "+" if d >= 0 else ""
        ax.text(j, max(v8, v4) + 1.5, f'{sign_s}{d:.1f}%', ha='center', fontsize=7,
                color=color, fontweight='bold')

plt.suptitle('MLX Quantization Impact: 8bit vs 4bit (IgakuQA 2022, All Sections)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/quantization_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: plots/quantization_comparison.png")
plt.close()

# ========== Figure 5: Full Section Heatmap ==========
fig5, ax7 = plt.subplots(figsize=(10, 8))

section_data = {
    "qwen3-32b 8bit":    [80.0, 86.0, 68.0, 86.7, 84.0, 74.7],
    "qwen3-32b 4bit":    [78.7, 84.0, 68.0, 85.3, 84.0, 76.0],
    "medgemma-27b":      [76.0, 82.0, 61.3, 76.0, 76.0, 64.0],
    "qwen3-vl-8b 8bit":  [62.7, 76.0, 60.0, 73.3, 76.0, 74.7],
    "qwen3-vl-8b 4bit":  [56.0, 68.0, 56.0, 70.7, 72.0, 72.0],
    "qwen3-vl-4b 8bit":  [58.7, 68.0, 50.7, 69.3, 70.0, 52.0],
    "qwen3-vl-4b 4bit":  [49.3, 72.0, 54.7, 69.3, 62.0, 48.0],
}
section_totals = {
    "qwen3-32b 8bit": 79.3, "qwen3-32b 4bit": 78.8,
    "medgemma-27b": 71.8,
    "qwen3-vl-8b 8bit": 69.8, "qwen3-vl-8b 4bit": 65.3,
    "qwen3-vl-4b 8bit": 60.5, "qwen3-vl-4b 4bit": 58.3,
}

model_names = list(section_data.keys())
sections = ["A\n(75)", "B\n(50)", "C\n(75)", "D\n(75)", "E\n(50)", "F\n(75)", "Total\n(400)"]
data_matrix = np.array([section_data[m] + [section_totals[m]] for m in model_names])

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('custom',
    [(0.0, '#FFCDD2'), (0.5, '#FFF9C4'), (0.75, '#C8E6C9'), (1.0, '#1B5E20')])

im = ax7.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=40, vmax=90)

ax7.set_xticks(np.arange(len(sections)))
ax7.set_xticklabels(sections, fontsize=11)
ax7.set_yticks(np.arange(len(model_names)))
ax7.set_yticklabels([f'{n} ({section_totals[n]:.1f}%)' for n in model_names], fontsize=10)

for i in range(len(model_names)):
    for j in range(len(sections)):
        val = data_matrix[i, j]
        color = 'white' if val >= 78 or val < 52 else 'black'
        fontw = 'bold' if j == len(sections) - 1 else 'normal'
        ax7.text(j, i, f'{val:.1f}%', ha='center', va='center',
                 fontsize=9, color=color, fontweight=fontw)

# Pass line annotation on Total column
for i, m in enumerate(model_names):
    total = section_totals[m]
    status = "PASS" if total >= 75 else "FAIL"
    color = '#1B5E20' if total >= 75 else '#B71C1C'
    ax7.text(len(sections) - 0.35, i, status, ha='left', va='center',
             fontsize=8, color=color, fontweight='bold')

ax7.set_title('IgakuQA 2022: Section-Level Accuracy Heatmap\n(Pass Line: 75% = 300/400)',
              fontsize=13, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax7, fraction=0.03, pad=0.04)
cbar.set_label('Accuracy (%)', fontsize=10)

# Horizontal divider lines between model groups
ax7.axhline(y=1.5, color='white', linewidth=2)
ax7.axhline(y=2.5, color='white', linewidth=2)
ax7.axhline(y=4.5, color='white', linewidth=2)

plt.tight_layout()
plt.savefig('plots/section_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: plots/section_heatmap.png")
plt.close()

# ========== Figure 6: Quantization Summary (Memory vs Accuracy Trade-off) ==========
fig6, (ax8, ax9) = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Memory vs Accuracy scatter with arrows ---
for qm in quant_models:
    # 8bit point
    ax8.scatter(qm["mem_8bit"], qm["total_8bit"], c='#1976D2', marker='o',
                s=200, edgecolors='black', linewidths=1, zorder=5)
    # 4bit point
    ax8.scatter(qm["mem_4bit"], qm["total_4bit"], c='#FF7043', marker='s',
                s=200, edgecolors='black', linewidths=1, zorder=5)
    # Arrow from 8bit to 4bit
    ax8.annotate('', xy=(qm["mem_4bit"], qm["total_4bit"]),
                 xytext=(qm["mem_8bit"], qm["total_8bit"]),
                 arrowprops=dict(arrowstyle='->', color='#455A64', lw=2, connectionstyle='arc3,rad=0.1'))
    # Labels
    ax8.text(qm["mem_8bit"] + 0.5, qm["total_8bit"] + 1.2,
             f'{qm["name"]}\n8bit ({qm["mem_8bit"]}GB)',
             fontsize=8, color='#1565C0', fontweight='bold')
    ax8.text(qm["mem_4bit"] - 0.5, qm["total_4bit"] - 3,
             f'4bit ({qm["mem_4bit"]}GB)',
             fontsize=8, color='#D84315', fontweight='bold', ha='right' if qm["mem_4bit"] < 10 else 'left')

ax8.axhline(y=75, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Pass Line (75%)')
ax8.scatter([], [], c='#1976D2', marker='o', s=100, label='8bit')
ax8.scatter([], [], c='#FF7043', marker='s', s=100, label='4bit')

ax8.set_xlabel('Memory (GB)', fontsize=12)
ax8.set_ylabel('Total Accuracy (%)', fontsize=12)
ax8.set_title('Quantization: Memory vs Accuracy Trade-off', fontsize=12, fontweight='bold')
ax8.set_xlim(0, 40)
ax8.set_ylim(50, 85)
ax8.legend(fontsize=9, loc='upper left')
ax8.grid(True, alpha=0.3)
ax8.set_axisbelow(True)

# --- Right: Impact summary bar chart ---
model_labels = [qm["name"] + f'\n({qm["params"]})' for qm in quant_models]
acc_drops = [qm["total_4bit"] - qm["total_8bit"] for qm in quant_models]
mem_savings = [(1 - qm["mem_4bit"] / qm["mem_8bit"]) * 100 for qm in quant_models]

x_pos = np.arange(len(quant_models))
width = 0.35

bars_acc = ax9.bar(x_pos - width/2, acc_drops, width, label='Accuracy Change (%)',
                   color=['#C62828' if d < -2 else '#FF8F00' if d < 0 else '#2E7D32' for d in acc_drops],
                   edgecolor='black', linewidth=0.5)
bars_mem = ax9.bar(x_pos + width/2, [-s for s in mem_savings], width, label='Memory Saved (%)',
                   color='#1976D2', alpha=0.7, edgecolor='black', linewidth=0.5)

ax9.axhline(y=0, color='black', linewidth=0.8)

for j, (ad, ms) in enumerate(zip(acc_drops, mem_savings)):
    ax9.text(j - width/2, ad - 1.5 if ad < 0 else ad + 0.5, f'{ad:+.1f}%',
             ha='center', fontsize=9, fontweight='bold',
             color='#C62828' if ad < -2 else '#FF8F00')
    ax9.text(j + width/2, -ms - 1.5, f'-{ms:.0f}%',
             ha='center', fontsize=9, fontweight='bold', color='#1565C0')

ax9.set_xticks(x_pos)
ax9.set_xticklabels(model_labels, fontsize=9)
ax9.set_ylabel('Change (%)', fontsize=12)
ax9.set_title('8bit → 4bit: Accuracy vs Memory Trade-off', fontsize=12, fontweight='bold')
ax9.legend(fontsize=9, loc='lower left')
ax9.grid(True, alpha=0.3, axis='y')
ax9.set_axisbelow(True)
ax9.set_ylim(-55, 10)

plt.suptitle('MLX Quantization Analysis: Is 4-bit Worth It?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/quantization_tradeoff.png', dpi=150, bbox_inches='tight')
print("Saved: plots/quantization_tradeoff.png")
plt.close()

print("\nDone! 6 plots saved to plots/ directory.")
