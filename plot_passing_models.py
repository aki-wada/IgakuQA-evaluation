#!/usr/bin/env python3
"""合格モデルの精度・メモリ使用量・応答速度の関係を可視化"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']

# 合格モデルデータ（実測値）
models = [
    {"name": "gpt-oss-120b\nMLX 8bit",   "accuracy": 84.5, "memory_gb": 124.20, "avg_time": 1.979, "color": "#e74c3c"},
    {"name": "gpt-oss-120b\nGGUF",        "accuracy": 84.0, "memory_gb":  63.39, "avg_time": 1.289, "color": "#c0392b"},
    {"name": "qwen3-next\n-80b MoE",      "accuracy": 83.5, "memory_gb":  84.67, "avg_time": 0.555, "color": "#2ecc71"},
    {"name": "qwen3-32b\n8bit",           "accuracy": 79.3, "memory_gb":  34.80, "avg_time": 1.631, "color": "#3498db"},
    {"name": "qwen3-32b\n4bit",           "accuracy": 78.8, "memory_gb":  18.50, "avg_time": 1.513, "color": "#2980b9"},
    {"name": "mistral-small\n-3.2",       "accuracy": 76.8, "memory_gb":  25.93, "avg_time": 0.977, "color": "#f39c12"},
    {"name": "mistral-large\n-2407",      "accuracy": 75.8, "memory_gb": 130.28, "avg_time": 6.182, "color": "#9b59b6"},
]

names     = [m["name"] for m in models]
accuracy  = [m["accuracy"] for m in models]
memory    = [m["memory_gb"] for m in models]
avg_time  = [m["avg_time"] for m in models]
colors    = [m["color"] for m in models]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("合格モデル比較（7モデル・400問全セクション評価）", fontsize=16, fontweight="bold", y=0.98)

# --- Plot 1: メモリ vs 精度 (バブルサイズ=速度の逆数) ---
ax1 = axes[0, 0]
# バブルサイズ: 速度が速いほど大きい
bubble_size = [max(800 / t, 80) for t in avg_time]
scatter1 = ax1.scatter(memory, accuracy, s=bubble_size, c=colors, alpha=0.8, edgecolors='black', linewidth=0.8)
for i, m in enumerate(models):
    offset_y = 0.8
    if i == 1:  # GGUF - shift to avoid overlap
        offset_y = -1.2
    ax1.annotate(m["name"].replace("\n", " "), (memory[i], accuracy[i]),
                 textcoords="offset points", xytext=(8, offset_y), fontsize=7.5,
                 ha='left', va='center')
ax1.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='合格ライン (75%)')
ax1.set_xlabel("メモリ使用量 (GB)", fontsize=11)
ax1.set_ylabel("正答率 (%)", fontsize=11)
ax1.set_title("メモリ使用量 vs 精度\n（バブルサイズ = 応答速度、大きいほど高速）", fontsize=11)
ax1.legend(fontsize=9)
ax1.set_ylim(73, 87)
ax1.grid(True, alpha=0.3)

# --- Plot 2: 応答速度 vs 精度 ---
ax2 = axes[0, 1]
ax2.scatter(avg_time, accuracy, s=150, c=colors, alpha=0.8, edgecolors='black', linewidth=0.8)
for i, m in enumerate(models):
    offset_x = 8
    ha = 'left'
    if avg_time[i] > 5:
        offset_x = -8
        ha = 'right'
    ax2.annotate(m["name"].replace("\n", " "), (avg_time[i], accuracy[i]),
                 textcoords="offset points", xytext=(offset_x, 5), fontsize=7.5,
                 ha=ha, va='bottom')
ax2.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='合格ライン (75%)')
ax2.set_xlabel("平均応答時間 (秒/問)", fontsize=11)
ax2.set_ylabel("正答率 (%)", fontsize=11)
ax2.set_title("応答速度 vs 精度", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_ylim(73, 87)
ax2.grid(True, alpha=0.3)

# --- Plot 3: 効率スコア (精度/メモリ) 棒グラフ ---
ax3 = axes[1, 0]
efficiency = [a / m for a, m in zip(accuracy, memory)]
sorted_idx = np.argsort(efficiency)[::-1]
bars = ax3.barh([names[i] for i in sorted_idx],
                [efficiency[i] for i in sorted_idx],
                color=[colors[i] for i in sorted_idx], alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, idx in zip(bars, sorted_idx):
    w = bar.get_width()
    ax3.text(w + 0.02, bar.get_y() + bar.get_height()/2,
             f'{efficiency[idx]:.2f}%/GB', va='center', fontsize=9)
ax3.set_xlabel("メモリ効率 (精度%/GB)", fontsize=11)
ax3.set_title("メモリ効率ランキング\n（1GBあたりの正答率）", fontsize=11)
ax3.grid(True, axis='x', alpha=0.3)

# --- Plot 4: 総合比較レーダー風の棒グラフ ---
ax4 = axes[1, 1]
x = np.arange(len(names))
width = 0.25

# 正規化（0-1スケール）
acc_norm   = [(a - 70) / (90 - 70) for a in accuracy]       # 70-90% → 0-1
mem_norm   = [1 - (m / 140) for m in memory]                 # 少ないほど良い
speed_norm = [1 - (t / 7) for t in avg_time]                 # 速いほど良い

bars1 = ax4.bar(x - width, acc_norm, width, label='精度', color='#3498db', alpha=0.85)
bars2 = ax4.bar(x, mem_norm, width, label='メモリ効率', color='#2ecc71', alpha=0.85)
bars3 = ax4.bar(x + width, speed_norm, width, label='応答速度', color='#e74c3c', alpha=0.85)

ax4.set_xticks(x)
ax4.set_xticklabels([n.replace("\n", " ") for n in names], rotation=30, ha='right', fontsize=7.5)
ax4.set_ylabel("正規化スコア (高いほど良い)", fontsize=10)
ax4.set_title("総合比較（精度・メモリ効率・速度）", fontsize=11)
ax4.legend(fontsize=9, loc='upper right')
ax4.set_ylim(0, 1.15)
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/passing_models_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: plots/passing_models_comparison.png")
plt.close()

# ===== English version =====
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Passing Models Comparison (7 Models, 400 Questions Full-Section Eval)", fontsize=16, fontweight="bold", y=0.98)

# --- Plot 1: Memory vs Accuracy ---
ax1 = axes[0, 0]
bubble_size = [max(800 / t, 80) for t in avg_time]
ax1.scatter(memory, accuracy, s=bubble_size, c=colors, alpha=0.8, edgecolors='black', linewidth=0.8)
for i, m in enumerate(models):
    offset_y = 0.8
    if i == 1:
        offset_y = -1.2
    ax1.annotate(m["name"].replace("\n", " "), (memory[i], accuracy[i]),
                 textcoords="offset points", xytext=(8, offset_y), fontsize=7.5,
                 ha='left', va='center')
ax1.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='Pass threshold (75%)')
ax1.set_xlabel("Memory Usage (GB)", fontsize=11)
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_title("Memory Usage vs Accuracy\n(Bubble size = response speed, larger = faster)", fontsize=11)
ax1.legend(fontsize=9)
ax1.set_ylim(73, 87)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Response Time vs Accuracy ---
ax2 = axes[0, 1]
ax2.scatter(avg_time, accuracy, s=150, c=colors, alpha=0.8, edgecolors='black', linewidth=0.8)
for i, m in enumerate(models):
    offset_x = 8
    ha = 'left'
    if avg_time[i] > 5:
        offset_x = -8
        ha = 'right'
    ax2.annotate(m["name"].replace("\n", " "), (avg_time[i], accuracy[i]),
                 textcoords="offset points", xytext=(offset_x, 5), fontsize=7.5,
                 ha=ha, va='bottom')
ax2.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='Pass threshold (75%)')
ax2.set_xlabel("Avg Response Time (sec/question)", fontsize=11)
ax2.set_ylabel("Accuracy (%)", fontsize=11)
ax2.set_title("Response Time vs Accuracy", fontsize=11)
ax2.legend(fontsize=9)
ax2.set_ylim(73, 87)
ax2.grid(True, alpha=0.3)

# --- Plot 3: Memory Efficiency Ranking ---
ax3 = axes[1, 0]
bars = ax3.barh([names[i] for i in sorted_idx],
                [efficiency[i] for i in sorted_idx],
                color=[colors[i] for i in sorted_idx], alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, idx in zip(bars, sorted_idx):
    w = bar.get_width()
    ax3.text(w + 0.02, bar.get_y() + bar.get_height()/2,
             f'{efficiency[idx]:.2f}%/GB', va='center', fontsize=9)
ax3.set_xlabel("Memory Efficiency (Accuracy %/GB)", fontsize=11)
ax3.set_title("Memory Efficiency Ranking\n(Accuracy per GB of memory)", fontsize=11)
ax3.grid(True, axis='x', alpha=0.3)

# --- Plot 4: Overall Comparison ---
ax4 = axes[1, 1]
x = np.arange(len(names))
width = 0.25
ax4.bar(x - width, acc_norm, width, label='Accuracy', color='#3498db', alpha=0.85)
ax4.bar(x, mem_norm, width, label='Memory Eff.', color='#2ecc71', alpha=0.85)
ax4.bar(x + width, speed_norm, width, label='Speed', color='#e74c3c', alpha=0.85)
ax4.set_xticks(x)
ax4.set_xticklabels([n.replace("\n", " ") for n in names], rotation=30, ha='right', fontsize=7.5)
ax4.set_ylabel("Normalized Score (higher = better)", fontsize=10)
ax4.set_title("Overall Comparison (Accuracy / Memory Eff. / Speed)", fontsize=11)
ax4.legend(fontsize=9, loc='upper right')
ax4.set_ylim(0, 1.15)
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("plots/passing_models_comparison_en.png", dpi=150, bbox_inches='tight')
print("Saved: plots/passing_models_comparison_en.png")
plt.close()
