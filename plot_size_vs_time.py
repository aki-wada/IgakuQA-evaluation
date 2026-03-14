#!/usr/bin/env python3
"""
モデルサイズ (Memory GB) vs 平均応答時間 (sec/question) の散布図
正答率をバブルサイズ・色で表現
日本語版と英語版を同時生成
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['axes.unicode_minus'] = False

# ====== データ定義 ======
# (name, memory_GB, avg_time_sec, accuracy%, architecture)
MODELS = [
    # Top tier
    ("Qwen3.5-397B@8bit", 249.80, 51.7, 89.5, "moe"),
    ("Qwen3.5-397B@4bit", 223.89, 45.9, 87.3, "moe"),
    ("Qwen3.5-27B@8bit", 29.53, 70.3, 87.3, "thinking"),
    ("gpt-oss-120B MLX", 124.20, 2.0, 84.5, "moe"),
    ("gpt-oss-120B GGUF", 63.39, 1.3, 84.0, "moe"),
    ("Qwen3-235B-2507", 249.80, 1.9, 86.0, "moe"),
    ("Qwen3-235B-A22B", 132.26, 6.2, 84.2, "moe"),
    ("Qwen3-Next-80B", 84.67, 0.6, 83.5, "moe"),
    # Passing
    ("Qwen3-VL-32B", 19.64, 3.6, 82.8, "vl"),
    ("Nemotron-3-Nano", 33.58, 8.1, 80.2, "thinking"),
    ("Qwen3-32B@8bit", 34.83, 1.6, 79.3, "dense"),
    ("Qwen3-32B@4bit", 18.45, 1.5, 78.8, "dense"),
    ("Qwen3-30B-A3B-2507", 32.50, 0.4, 78.8, "moe"),
    ("Swallow-70B", 40.35, 1.9, 78.0, "jp-ft"),
    ("GPT-OSS-Swallow-20B", 41.86, 11.2, 77.8, "jp-ft"),
    ("Qwen3-VL-30B", 33.53, 2.3, 77.8, "vl"),
    ("Llama 4 Scout", 61.14, 5.0, 77.5, "moe"),
    ("Mistral-Small-3.2", 25.93, 1.0, 76.8, "dense"),
    ("Mistral-Large", 130.28, 6.2, 75.8, "dense"),
    # Near-pass
    ("Shisa-v2.1-70B", 75.00, 2.4, 74.2, "jp-ft"),
    ("Magistral-Small", 47.16, 0.9, 74.0, "dense"),
    ("Qwen3-14B", 15.71, 0.6, 71.8, "dense"),
    ("MedGemma-27B", 16.03, 0.8, 71.8, "dense"),
    ("gpt-oss-20B MLX", 22.26, 1.2, 71.5, "moe"),
    ("Llama-3.3-70B", 39.71, 1.9, 71.0, "dense"),
    # Mid-range
    ("Qwen3-VL-8B@8bit", 9.87, 2.2, 69.8, "vl"),
    ("Gemma-3-27B", 16.87, 1.0, 67.8, "dense"),
    ("Qwen3-VL-8B@4bit", 5.78, 2.2, 65.3, "vl"),
    ("Phi-4", 15.59, 0.7, 62.8, "dense"),
    ("Qwen3-8B", 8.72, 0.4, 61.3, "dense"),
    ("Qwen3-VL-4B@8bit", 5.11, 2.0, 60.5, "vl"),
    ("Ezo2.5-12B", 6.94, 0.4, 60.0, "jp-ft"),
    ("Qwen3-VL-4B@4bit", 3.11, 1.9, 58.3, "vl"),
    ("Qwen3-4B-2507", 2.28, 0.8, 54.7, "dense"),
    ("Gemma-3-12B", 14.45, 1.6, 54.7, "dense"),
    ("Swallow-8B", 16.08, 0.4, 53.3, "jp-ft"),
    ("ELYZA-jp-8B", 4.92, 0.3, 44.0, "jp-ft"),
    ("Mistral-Nemo-12B", 12.0, 0.9, 34.7, "dense"),
    ("ELYZA-Llama2-13B", 7.0, 0.5, 24.0, "jp-ft"),
    ("MedGemma-4B@bf16", 9.98, 1.2, 29.3, "dense"),
    ("LFM2.5-1.2B", 1.25, 0.1, 28.0, "dense"),
    ("MedGemma-4B", 3.44, 0.7, 18.7, "dense"),
]

# Architecture styles
ARCH_STYLES = {
    "dense":    {"color": "#1f77b4", "marker": "o",  "label_jp": "Dense",           "label_en": "Dense"},
    "moe":      {"color": "#ff7f0e", "marker": "D",  "label_jp": "MoE",             "label_en": "MoE"},
    "vl":       {"color": "#2ca02c", "marker": "s",  "label_jp": "Vision-Language",  "label_en": "Vision-Language"},
    "thinking": {"color": "#d62728", "marker": "^",  "label_jp": "Thinking",         "label_en": "Thinking"},
    "jp-ft":    {"color": "#9467bd", "marker": "p",  "label_jp": "日本語FT",         "label_en": "JP Fine-tuned"},
}


def create_plot(lang="jp"):
    is_jp = lang == "jp"
    if is_jp:
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else:
        matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']

    fig, ax = plt.subplots(figsize=(14, 9))

    names = [m[0] for m in MODELS]
    mems = np.array([m[1] for m in MODELS])
    times = np.array([m[2] for m in MODELS])
    accs = np.array([m[3] for m in MODELS])
    archs = [m[4] for m in MODELS]

    # Bubble size proportional to accuracy (sqrt scale)
    sizes = 30 + 250 * (accs / accs.max()) ** 2

    # Color map for accuracy
    cmap = LinearSegmentedColormap.from_list('acc_cmap',
        [(0.0, '#BDBDBD'), (0.4, '#FF8A65'), (0.6, '#FFD54F'),
         (0.75, '#81C784'), (1.0, '#1B5E20')])
    norm = plt.Normalize(vmin=15, vmax=92)

    # Plot each architecture group with distinct markers
    for arch_key, style in ARCH_STYLES.items():
        idx = [i for i, a in enumerate(archs) if a == arch_key]
        if not idx:
            continue
        label = style["label_jp"] if is_jp else style["label_en"]
        scatter = ax.scatter(
            mems[idx], times[idx],
            s=sizes[idx],
            c=accs[idx],
            cmap=cmap, norm=norm,
            marker=style["marker"],
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5,
            zorder=3,
        )
        # Invisible scatter for legend (marker shape only)
        ax.scatter([], [], marker=style["marker"], c='gray', s=60,
                   edgecolors='black', linewidth=0.5, label=label)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("正答率 (%)" if is_jp else "Accuracy (%)", fontsize=11)

    # Annotate key models
    annotate_set = {
        "Qwen3.5-397B@8bit", "Qwen3.5-27B@8bit",
        "gpt-oss-120B MLX", "gpt-oss-120B GGUF",
        "Qwen3-Next-80B", "Qwen3-235B-2507", "Qwen3-235B-A22B",
        "Qwen3-VL-32B", "Nemotron-3-Nano",
        "Qwen3-32B@8bit", "Qwen3-30B-A3B-2507",
        "Swallow-70B", "GPT-OSS-Swallow-20B",
        "Mistral-Small-3.2", "Mistral-Large",
        "Llama 4 Scout",
        "Qwen3-14B", "MedGemma-27B",
        "Phi-4", "Qwen3-8B",
        "LFM2.5-1.2B", "ELYZA-Llama2-13B", "Mistral-Nemo-12B",
    }

    offsets = {
        "Qwen3.5-397B@8bit": (5, 8),
        "Qwen3.5-397B@4bit": (5, -12),
        "Qwen3.5-27B@8bit": (15, -20),
        "gpt-oss-120B MLX": (5, -8),
        "gpt-oss-120B GGUF": (5, -8),
        "Qwen3-Next-80B": (5, -8),
        "Qwen3-235B-2507": (5, -8),
        "Qwen3-235B-A22B": (5, 5),
        "Qwen3-VL-32B": (5, 5),
        "Nemotron-3-Nano": (5, 5),
        "Qwen3-32B@8bit": (5, -8),
        "Qwen3-30B-A3B-2507": (-15, -8),
        "Swallow-70B": (5, -8),
        "GPT-OSS-Swallow-20B": (5, 5),
        "Mistral-Small-3.2": (-15, -8),
        "Mistral-Large": (5, 5),
        "Llama 4 Scout": (5, 5),
        "Qwen3-14B": (-15, -8),
        "MedGemma-27B": (-15, 5),
        "Phi-4": (5, -8),
        "Qwen3-8B": (-12, -8),
        "LFM2.5-1.2B": (5, -5),
        "ELYZA-Llama2-13B": (5, -5),
        "Mistral-Nemo-12B": (5, 5),
    }

    for i, name in enumerate(names):
        if name in annotate_set:
            display = name.replace("Qwen3.5-397B@8bit", "Q3.5-397B@8bit")
            display = display.replace("Qwen3.5-27B@8bit", "Q3.5-27B")
            display = display.replace("Qwen3-Next-80B", "Q3-Next-80B")
            display = display.replace("Qwen3-235B-2507", "Q3-235B-2507")
            display = display.replace("Qwen3-235B-A22B", "Q3-235B")
            display = display.replace("Qwen3-VL-32B", "Q3-VL-32B")
            display = display.replace("Qwen3-32B@8bit", "Q3-32B")
            display = display.replace("Qwen3-30B-A3B-2507", "Q3-30B-A3B")
            display = display.replace("Qwen3-14B", "Q3-14B")
            display = display.replace("Qwen3-8B", "Q3-8B")
            display = display.replace("Nemotron-3-Nano", "Nemotron")
            display = display.replace("Mistral-Small-3.2", "Mistral-Sm")
            display = display.replace("Mistral-Large", "Mistral-Lg")
            display = display.replace("GPT-OSS-Swallow-20B", "OSS-Swallow-20B")
            display = display.replace("Mistral-Nemo-12B", "Mistral-Nemo")
            display = display.replace("ELYZA-Llama2-13B", "ELYZA-Llama2")
            display = display.replace("LFM2.5-1.2B", "LFM2.5-1.2B")

            display = f"{display} ({accs[i]:.0f}%)"

            dx, dy = offsets.get(name, (5, 5))
            ax.annotate(
                display, (mems[i], times[i]),
                xytext=(dx, dy), textcoords='offset points',
                fontsize=6.5, alpha=0.85,
                arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5) if abs(dx) > 10 or abs(dy) > 10 else None,
            )

    # Axis settings
    ax.set_yscale('log')
    ax.set_ylim(0.08, 120)
    ax.set_xlim(-5, 280)

    # Custom y-axis ticks
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if is_jp:
        ax.set_xlabel("モデルサイズ (メモリ使用量 GB)", fontsize=12)
        ax.set_ylabel("平均応答時間 (秒/問, 対数スケール)", fontsize=12)
        ax.set_title("モデルサイズ vs 応答時間: ローカルLLMの効率性分析\n"
                     "IgakuQA 第116回医師国家試験  |  Mac Studio M3 Ultra 512GB",
                     fontsize=13, fontweight='bold')
    else:
        ax.set_xlabel("Model Size (Memory GB)", fontsize=12)
        ax.set_ylabel("Mean Response Time (sec/question, log scale)", fontsize=12)
        ax.set_title("Model Size vs. Response Time: Efficiency Analysis of Local LLMs\n"
                     "116th JMLE (IgakuQA)  |  Mac Studio M3 Ultra 512GB",
                     fontsize=13, fontweight='bold')

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle='-', linewidth=0.3)

    # Architecture legend is placed later (bottom right)

    # Architecture legend - move to bottom right
    legend1 = ax.legend(
        loc='lower right', fontsize=9, framealpha=0.9,
        title="アーキテクチャ" if is_jp else "Architecture",
        title_fontsize=10,
    )
    ax.add_artist(legend1)

    # Note
    if is_jp:
        ax.text(0.98, 0.25,
                "バブルサイズ・色 = 正答率\n"
                "左下 = 小型・高速 / 右上 = 大型・低速\n"
                "MoEモデルは大容量でも高速な傾向",
                transform=ax.transAxes, fontsize=7.5, alpha=0.6,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        ax.text(0.98, 0.25,
                "Bubble size & color = accuracy\n"
                "Bottom-left = small & fast / Top-right = large & slow\n"
                "MoE models tend to be fast despite large total params",
                transform=ax.transAxes, fontsize=7.5, alpha=0.6,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()

    suffix = "" if is_jp else "_en"
    outpath = f"plots/size_vs_time{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_plot("jp")
    create_plot("en")
    print("Done!")
