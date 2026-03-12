#!/usr/bin/env python3
"""
応答時間 vs 正答率の散布図（パレートフロンティア付き）
日本語版と英語版を同時生成
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

# ====== データ定義 ======
# (name, accuracy%, avg_time_sec, memory_gb, architecture)
# architecture: "dense", "moe", "vl", "thinking", "jp-ft"
MODELS = [
    ("Qwen3.5-397B@8bit", 89.5, 51.7, 249.8, "moe"),
    ("Qwen3.5-397B@4bit", 87.2, 45.9, 223.9, "moe"),
    ("Qwen3.5-27B", 87.2, 70.3, 29.5, "dense"),
    ("gpt-oss-120B MLX", 84.5, 2.0, 124.2, "moe"),
    ("gpt-oss-120B GGUF", 84.0, 1.3, 63.4, "moe"),
    ("Qwen3-Next-80B", 83.5, 0.6, 84.7, "moe"),
    ("Qwen3-VL-32B", 82.8, 3.6, 19.6, "vl"),
    ("Nemotron-3-Nano", 80.2, 8.1, 33.6, "thinking"),
    ("Qwen3-32B@8bit", 79.2, 1.6, 34.8, "dense"),
    ("Qwen3-32B@4bit", 78.8, 1.5, 18.5, "dense"),
    ("Qwen3-30B-A3B-2507", 78.8, 0.4, 32.5, "moe"),
    ("Swallow-70B", 78.0, 1.9, 40.4, "jp-ft"),
    ("Qwen3-VL-30B", 77.8, 2.3, 33.5, "vl"),
    ("Llama 4 Scout", 77.5, 5.0, 63.9, "moe"),
    ("Mistral-Small-3.2", 76.8, 1.0, 25.9, "dense"),
    ("Mistral-Large-2407", 75.8, 6.2, 130.3, "dense"),
    ("Qwen3-235B-A22B", 84.2, 6.2, 132.3, "moe"),
    ("Qwen3-235B-2507", 86.0, 1.9, 249.8, "moe"),
    ("GPT-OSS-Swallow-20B", 77.8, 11.2, 45.0, "jp-ft"),
    ("Magistral-Small (old)", 74.2, 1.2, 47.2, "dense"),
    ("Shisa-v2.1-70B", 74.2, 2.4, 75.0, "jp-ft"),
    ("Magistral-Small-2509", 74.0, 0.9, 47.2, "dense"),
    ("Qwen3-14B", 71.8, 0.6, 15.7, "dense"),
    ("gpt-oss-20B MLX", 71.5, 1.2, 22.3, "moe"),
    ("gpt-oss-20B GGUF", 71.0, 0.8, 12.1, "moe"),
    ("gpt-oss-20B Q8", 71.0, 0.8, 12.1, "moe"),
    ("Llama-3.3-70B", 71.0, 1.9, 40.4, "dense"),
    ("Qwen3-VL-8B@8bit", 69.8, 2.2, 9.9, "vl"),
    ("MedGemma-27B", 67.8, 0.8, 16.0, "dense"),
    ("Gemma-3-27B", 67.8, 1.0, 16.9, "dense"),
    ("Qwen3-VL-8B@4bit", 65.2, 2.2, 5.8, "vl"),
    ("Phi-4", 62.7, 0.7, 15.6, "dense"),
    ("Qwen3-VL-4B@8bit", 60.5, 2.0, 5.1, "vl"),
    ("Qwen3-VL-4B@4bit", 58.2, 1.9, 3.0, "vl"),
]

# Architecture -> color mapping
ARCH_COLORS = {
    "dense": "#1f77b4",    # blue
    "moe": "#ff7f0e",      # orange
    "vl": "#2ca02c",       # green
    "thinking": "#d62728", # red
    "jp-ft": "#9467bd",    # purple
}

# Architecture labels
ARCH_LABELS_JP = {
    "dense": "Dense",
    "moe": "MoE",
    "vl": "Vision-Language",
    "thinking": "Thinking",
    "jp-ft": "日本語FT",
}
ARCH_LABELS_EN = {
    "dense": "Dense",
    "moe": "MoE",
    "vl": "Vision-Language",
    "thinking": "Thinking",
    "jp-ft": "JP Fine-tuned",
}


def compute_pareto_frontier(points):
    """Compute Pareto frontier: maximize accuracy, minimize time."""
    # Sort by time ascending
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    max_acc = -1
    for t, acc, name in sorted_pts:
        if acc > max_acc:
            frontier.append((t, acc, name))
            max_acc = acc
    return frontier


def create_plot(lang="jp"):
    """Create the time vs accuracy scatter plot."""
    is_jp = lang == "jp"

    if is_jp:
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else:
        matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']

    fig, ax = plt.subplots(figsize=(14, 9))

    # Prepare data
    names = [m[0] for m in MODELS]
    accs = [m[1] for m in MODELS]
    times = [m[2] for m in MODELS]
    mems = [m[3] for m in MODELS]
    archs = [m[4] for m in MODELS]

    # Marker size proportional to memory (sqrt scale for area)
    mem_arr = np.array(mems)
    sizes = 30 + 200 * np.sqrt(mem_arr / mem_arr.max())

    # Plot each architecture group
    for arch_key in ["dense", "moe", "vl", "thinking", "jp-ft"]:
        idx = [i for i, a in enumerate(archs) if a == arch_key]
        if not idx:
            continue
        label = ARCH_LABELS_JP[arch_key] if is_jp else ARCH_LABELS_EN[arch_key]
        ax.scatter(
            [times[i] for i in idx],
            [accs[i] for i in idx],
            s=[sizes[i] for i in idx],
            c=ARCH_COLORS[arch_key],
            alpha=0.7,
            edgecolors='white',
            linewidth=0.8,
            label=label,
            zorder=3,
        )

    # 75% reference line
    ax.axhline(y=75.0, color='red', linestyle='--', alpha=0.6, linewidth=1.5, zorder=1)
    if is_jp:
        ax.text(0.35, 75.5, "合格参考ライン (75%)", color='red', fontsize=9, alpha=0.7)
    else:
        ax.text(0.35, 75.5, "Reference threshold (75%)", color='red', fontsize=9, alpha=0.7)

    # Pareto frontier
    pareto_pts = [(times[i], accs[i], names[i]) for i in range(len(MODELS))]
    frontier = compute_pareto_frontier(pareto_pts)
    if len(frontier) > 1:
        ft = [p[0] for p in frontier]
        fa = [p[1] for p in frontier]
        ax.plot(ft, fa, color='#333333', linestyle='-', alpha=0.4, linewidth=2.0, zorder=2)
        # Mark Pareto frontier points with stars
        ax.scatter(ft, fa, marker='*', s=15, color='black', alpha=0.5, zorder=4)

    # Annotate key models (Pareto frontier + notable ones)
    pareto_names = {p[2] for p in frontier}
    notable = {
        "Qwen3-Next-80B", "gpt-oss-120B MLX", "Qwen3.5-397B@8bit", "Qwen3.5-27B",
        "Qwen3-VL-32B", "Mistral-Small-3.2", "Qwen3-30B-A3B-2507",
        "Swallow-70B", "Nemotron-3-Nano", "MedGemma-27B", "Qwen3-14B",
        "Phi-4", "Qwen3-VL-4B@4bit",
    }
    annotate_set = pareto_names | notable

    # Annotation offsets (manually adjusted for readability)
    offsets = {
        "Qwen3.5-397B@8bit": (-15, 6),
        "Qwen3.5-397B@4bit": (-15, -10),
        "Qwen3.5-27B": (5, -8),
        "gpt-oss-120B MLX": (5, 5),
        "gpt-oss-120B GGUF": (-15, -10),
        "Qwen3-Next-80B": (-12, 7),
        "Qwen3-VL-32B": (5, 5),
        "Nemotron-3-Nano": (5, 5),
        "Qwen3-32B@8bit": (5, -10),
        "Qwen3-30B-A3B-2507": (-15, -10),
        "Swallow-70B": (5, -10),
        "Llama 4 Scout": (5, -10),
        "Mistral-Small-3.2": (-15, 5),
        "Mistral-Large-2407": (5, -10),
        "Qwen3-235B-A22B": (5, 5),
        "Qwen3-235B-2507": (-15, -10),
        "Qwen3-14B": (-12, 7),
        "MedGemma-27B": (5, 7),
        "Phi-4": (5, 7),
        "Qwen3-VL-4B@4bit": (5, -10),
    }

    for i, name in enumerate(names):
        if name in annotate_set:
            # Shorten long names
            display = name.replace("Qwen3.5-397B", "Q3.5-397B")
            display = display.replace("Qwen3-Next-80B", "Q3-Next-80B")
            display = display.replace("Qwen3-VL-32B", "Q3-VL-32B")
            display = display.replace("Qwen3-30B-A3B-2507", "Q3-30B-A3B")
            display = display.replace("Qwen3-32B", "Q3-32B")
            display = display.replace("Qwen3-235B-A22B", "Q3-235B")
            display = display.replace("Qwen3-235B-2507", "Q3-235B-2507")
            display = display.replace("Qwen3-14B", "Q3-14B")
            display = display.replace("Qwen3-VL-4B", "Q3-VL-4B")
            display = display.replace("Mistral-Small-3.2", "Mistral-Sm-3.2")
            display = display.replace("Mistral-Large-2407", "Mistral-Lg")
            display = display.replace("Nemotron-3-Nano", "Nemotron")

            dx, dy = offsets.get(name, (5, 5))
            fontsize = 7.5 if name in pareto_names else 7
            weight = 'bold' if name in pareto_names else 'normal'
            ax.annotate(
                display, (times[i], accs[i]),
                xytext=(dx, dy), textcoords='offset points',
                fontsize=fontsize, fontweight=weight,
                alpha=0.85,
                arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5) if abs(dx) > 10 or abs(dy) > 10 else None,
            )

    # Axis settings
    ax.set_xscale('log')
    ax.set_xlim(0.25, 120)
    ax.set_ylim(55, 93)

    if is_jp:
        ax.set_xlabel("平均応答時間 (秒/問, 対数スケール)", fontsize=12)
        ax.set_ylabel("正答率 (%)", fontsize=12)
        ax.set_title("応答時間 vs 正答率: ローカルLLMのパレート最適分析\n"
                      "第116回医師国家試験 (400問)  |  Mac Studio M3 Ultra",
                      fontsize=13, fontweight='bold')
    else:
        ax.set_xlabel("Mean Response Time (sec/question, log scale)", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Response Time vs. Accuracy: Pareto Efficiency of Local LLMs\n"
                      "116th JMLE (400 Questions)  |  Mac Studio M3 Ultra",
                      fontsize=13, fontweight='bold')

    # Custom x-axis ticks
    ax.set_xticks([0.3, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.3)

    # Legend with architecture types
    legend1 = ax.legend(
        loc='lower left', fontsize=9, framealpha=0.9,
        title="Architecture" if not is_jp else "アーキテクチャ",
        title_fontsize=10,
    )
    ax.add_artist(legend1)

    # Memory size legend
    mem_examples = [5, 30, 130, 250]
    size_examples = [30 + 200 * np.sqrt(m / mem_arr.max()) for m in mem_examples]
    mem_handles = [
        Line2D([0], [0], marker='o', color='gray', alpha=0.4,
               markersize=np.sqrt(s), linestyle='None',
               label=f"{m} GB")
        for m, s in zip(mem_examples, size_examples)
    ]
    legend2 = ax.legend(
        handles=mem_handles, loc='upper right', fontsize=8, framealpha=0.9,
        title="Memory" if not is_jp else "メモリ使用量",
        title_fontsize=9,
    )

    # Pareto note
    if is_jp:
        ax.text(0.02, 0.02, "黒線 = パレートフロンティア (速度・精度の最適トレードオフ)\n"
                             "バブルサイズ = メモリ使用量",
                transform=ax.transAxes, fontsize=7.5, alpha=0.6,
                verticalalignment='bottom')
    else:
        ax.text(0.02, 0.02, "Black line = Pareto frontier (optimal speed-accuracy trade-off)\n"
                             "Bubble size = memory footprint",
                transform=ax.transAxes, fontsize=7.5, alpha=0.6,
                verticalalignment='bottom')

    plt.tight_layout()

    suffix = "" if is_jp else "_en"
    outpath = f"plots/time_vs_accuracy{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_plot("jp")
    create_plot("en")
    print("Done!")
