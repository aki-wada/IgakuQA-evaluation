#!/usr/bin/env python3
"""
モデルサイズ × 応答時間 × 正答率の3Dプロット
精度-速度-コストのトリレンマを可視化
日本語版と英語版を同時生成
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['axes.unicode_minus'] = False

# (name, memory_GB, avg_time_sec, accuracy%, architecture)
MODELS = [
    ("Qwen3.5-397B@8bit", 249.80, 51.7, 89.5, "moe"),
    ("Qwen3.5-397B@4bit", 223.89, 45.9, 87.3, "moe"),
    ("Qwen3.5-27B@8bit", 29.53, 70.3, 87.3, "thinking"),
    ("gpt-oss-120B MLX", 124.20, 2.0, 84.5, "moe"),
    ("gpt-oss-120B GGUF", 63.39, 1.3, 84.0, "moe"),
    ("Qwen3-235B-2507", 249.80, 1.9, 86.0, "moe"),
    ("Qwen3-235B-A22B", 132.26, 6.2, 84.2, "moe"),
    ("Qwen3-Next-80B", 84.67, 0.6, 83.5, "moe"),
    ("Nemotron-3-Nano", 33.58, 8.1, 80.2, "thinking"),
    ("Qwen3-32B@8bit", 34.83, 1.6, 79.3, "dense"),
    ("Qwen3-32B@4bit", 18.45, 1.5, 78.8, "dense"),
    ("Qwen3-30B-A3B-2507", 32.50, 0.4, 78.8, "moe"),
    ("Swallow-70B", 40.35, 1.9, 78.0, "jp-ft"),
    ("GPT-OSS-Swallow-20B", 41.86, 11.2, 77.8, "jp-ft"),
    ("Llama 4 Scout", 61.14, 5.0, 77.5, "moe"),
    ("Mistral-Small-3.2", 25.93, 1.0, 76.8, "dense"),
    ("Mistral-Large", 130.28, 6.2, 75.8, "dense"),
    ("Shisa-v2.1-70B", 75.00, 2.4, 74.2, "jp-ft"),
    ("Qwen3-14B", 15.71, 0.6, 71.8, "dense"),
    ("MedGemma-27B", 16.03, 0.8, 71.8, "dense"),
    ("gpt-oss-20B MLX", 22.26, 1.2, 71.5, "moe"),
    ("Llama-3.3-70B", 39.71, 1.9, 71.0, "dense"),
    ("Gemma-3-27B", 16.87, 1.0, 67.8, "dense"),
    ("Phi-4", 15.59, 0.7, 62.8, "dense"),
    ("Qwen3-8B", 8.72, 0.4, 61.3, "dense"),
    ("Ezo2.5-12B", 6.94, 0.4, 60.0, "jp-ft"),
]

ARCH_STYLES = {
    "dense":    {"color": "#1f77b4", "marker": "o",  "label_jp": "Dense",          "label_en": "Dense"},
    "moe":      {"color": "#ff7f0e", "marker": "D",  "label_jp": "MoE",            "label_en": "MoE"},
    "thinking": {"color": "#d62728", "marker": "^",  "label_jp": "Thinking",        "label_en": "Thinking"},
    "jp-ft":    {"color": "#9467bd", "marker": "p",  "label_jp": "日本語FT",        "label_en": "JP Fine-tuned"},
}

# Key models to annotate
ANNOTATE = {
    "Qwen3.5-397B@8bit", "Qwen3.5-27B@8bit",
    "gpt-oss-120B GGUF", "Qwen3-Next-80B", "Qwen3-235B-2507",
    "Nemotron-3-Nano",
    "Qwen3-32B@4bit", "Qwen3-30B-A3B-2507",
    "Swallow-70B", "Mistral-Small-3.2",
    "Qwen3-14B", "Phi-4", "Ezo2.5-12B",
}


def create_plot(lang="jp"):
    is_jp = lang == "jp"
    if is_jp:
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else:
        matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']

    names = [m[0] for m in MODELS]
    mems = np.array([m[1] for m in MODELS])
    times = np.array([m[2] for m in MODELS])
    accs = np.array([m[3] for m in MODELS])
    archs = [m[4] for m in MODELS]

    # Color map
    cmap = LinearSegmentedColormap.from_list('acc',
        [(0.0, '#BDBDBD'), (0.4, '#FF8A65'), (0.6, '#FFD54F'),
         (0.75, '#81C784'), (1.0, '#1B5E20')])
    norm = plt.Normalize(vmin=55, vmax=92)

    fig = plt.figure(figsize=(18, 8))

    # ====== (a) 3D scatter ======
    ax1 = fig.add_subplot(121, projection='3d')

    # X=Memory, Y=Accuracy, Z=Response Time (tall = slow)
    log_times = np.log10(times)

    for arch_key, style in ARCH_STYLES.items():
        idx = [i for i, a in enumerate(archs) if a == arch_key]
        if not idx:
            continue
        label = style["label_jp"] if is_jp else style["label_en"]
        ax1.scatter(
            np.log10(mems[idx]), accs[idx], log_times[idx],
            c=accs[idx], cmap=cmap, norm=norm,
            marker=style["marker"], s=80, alpha=0.85,
            edgecolors='black', linewidth=0.3, zorder=3,
            label=label,
        )

    # Drop lines to floor (time=0 projection)
    z_floor = np.log10(0.08)
    for i in range(len(MODELS)):
        ax1.plot(
            [np.log10(mems[i]), np.log10(mems[i])],
            [accs[i], accs[i]],
            [z_floor, log_times[i]],
            color='gray', alpha=0.15, linewidth=0.5, zorder=1,
        )

    # Annotate key models
    for i, name in enumerate(names):
        if name in ANNOTATE:
            short = name.replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-")
            short = short.replace("gpt-oss-120B GGUF", "OSS-120B")
            short = short.replace("Nemotron-3-Nano", "Nemotron")
            short = short.replace("Mistral-Small-3.2", "Mistral-Sm")
            short = short.replace("GPT-OSS-Swallow-20B", "OSS-Sw-20B")
            ax1.text(np.log10(mems[i]), accs[i], log_times[i] + 0.08,
                     f'{short}', fontsize=5, alpha=0.8,
                     ha='center', zorder=10)

    # Axis formatting
    mem_ticks = [3, 10, 30, 100, 250]
    time_ticks = [0.1, 0.3, 1, 3, 10, 30, 70]
    ax1.set_xticks([np.log10(t) for t in mem_ticks])
    ax1.set_xticklabels([str(t) for t in mem_ticks], fontsize=8)
    ax1.set_zticks([np.log10(t) for t in time_ticks])
    ax1.set_zticklabels([str(t) for t in time_ticks], fontsize=8)

    if is_jp:
        ax1.set_xlabel('メモリ (GB, log)', fontsize=10, labelpad=8)
        ax1.set_ylabel('正答率 (%)', fontsize=10, labelpad=8)
        ax1.set_zlabel('応答時間 (秒, log)', fontsize=10, labelpad=5)
        ax1.set_title('(a) 3D: サイズ × 精度 × 速度\n高い塔 = 遅い', fontsize=12, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('Memory (GB, log)', fontsize=10, labelpad=8)
        ax1.set_ylabel('Accuracy (%)', fontsize=10, labelpad=8)
        ax1.set_zlabel('Response Time (sec, log)', fontsize=10, labelpad=5)
        ax1.set_title('(a) 3D: Size × Accuracy × Speed\nTall tower = slow', fontsize=12, fontweight='bold', pad=15)

    ax1.set_ylim(55, 93)
    ax1.set_zlim(np.log10(0.08), np.log10(100))
    ax1.view_init(elev=28, azim=-60)
    ax1.legend(fontsize=7, loc='upper left', ncol=1)

    # ====== (b) Efficiency score: accuracy / (log(mem) * log(time)) ======
    ax2 = fig.add_subplot(122)

    # Efficiency = accuracy per unit of "cost" (memory × time)
    # Higher = better (high accuracy, low resources)
    # Use log to handle wide ranges
    efficiency = accs / (np.log10(mems + 1) * np.log10(times + 1) + 0.5)

    # Sort by efficiency
    sort_idx = np.argsort(efficiency)[::-1]

    # Top 20
    top_n = 20
    top_idx = sort_idx[:top_n]

    colors_bar = [cmap(norm(accs[i])) for i in top_idx]
    y_pos = np.arange(top_n)

    short_names = []
    for i in top_idx:
        n = names[i].replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-")
        n = n.replace("gpt-oss-", "OSS-").replace("Nemotron-3-Nano", "Nemotron")
        n = n.replace("Mistral-Small-3.2", "Mistral-Sm").replace("Mistral-Large", "Mistral-Lg")
        n = n.replace("GPT-OSS-Swallow-20B", "OSS-Sw-20B")
        short_names.append(n)

    bars = ax2.barh(y_pos, efficiency[top_idx], color=colors_bar,
                    edgecolor='black', linewidth=0.3, alpha=0.85)

    # Add accuracy and time labels on bars
    for j, i in enumerate(top_idx):
        ax2.text(efficiency[i] + 0.5, j,
                 f'{accs[i]:.0f}% | {mems[i]:.0f}GB | {times[i]:.1f}s',
                 va='center', fontsize=7, alpha=0.7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(short_names, fontsize=8)
    ax2.invert_yaxis()

    if is_jp:
        ax2.set_xlabel('効率スコア = 正答率 / (log(メモリ) × log(応答時間))', fontsize=10)
        ax2.set_title('(b) 総合効率ランキング Top 20\n高精度・低コスト・高速 = 高スコア',
                      fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('Efficiency = Accuracy / (log(Memory) × log(Time))', fontsize=10)
        ax2.set_title('(b) Overall Efficiency Ranking Top 20\nHigh accuracy, low cost, fast = high score',
                      fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_axisbelow(True)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("正答率 (%)" if is_jp else "Accuracy (%)", fontsize=9)

    if is_jp:
        plt.suptitle('ローカルLLM: 精度-速度-コストのトリレンマ分析\n'
                     'IgakuQA 第116回医師国家試験 | Mac Studio M3 Ultra 512GB',
                     fontsize=14, fontweight='bold')
    else:
        plt.suptitle('Local LLM: Accuracy-Speed-Cost Trilemma Analysis\n'
                     '116th JMLE (IgakuQA) | Mac Studio M3 Ultra 512GB',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    suffix = "" if is_jp else "_en"
    outpath = f"plots/3d_trilemma{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_plot("jp")
    create_plot("en")
    print("Done!")
