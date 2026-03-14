#!/usr/bin/env python3
"""
Thinking機構の効果分析: モデルサイズとの関係
小型モデルほどThinkingの恩恵が大きいことを可視化
日本語版と英語版を同時生成
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.unicode_minus'] = False

# Paired comparisons: thinking model vs closest non-thinking baseline
# (thinking_name, thinking_acc, baseline_name, baseline_acc, approx_params_B, time_thinking, time_baseline)
PAIRS = [
    # Small: Qwen3.5-27B thinking vs Qwen3-32B dense (similar size)
    ("Qwen3.5-27B@8bit\n(Thinking)", 87.3, "Qwen3-32B@8bit\n(Dense)", 79.3, 30, 70.3, 1.6),
    # Medium: Nemotron-3-Nano thinking vs similar-size non-thinking
    ("Nemotron-3-Nano\n(Thinking)", 80.2, "Swallow-70B\n(JP-FT)", 78.0, 49, 8.1, 1.9),
    ("Nemotron-3-Nano\n(Thinking)", 80.2, "Llama 4 Scout\n(MoE)", 77.5, 49, 8.1, 5.0),
]

# All models for the scatter: (name, params_B, accuracy, time_sec, arch, is_thinking)
ALL_MODELS = [
    ("Qwen3.5-397B@8bit", 397, 89.5, 51.7, "MoE", False),
    ("Qwen3.5-397B@4bit", 397, 87.3, 45.9, "MoE", False),
    ("Qwen3.5-27B@8bit", 27, 87.3, 70.3, "Thinking", True),
    ("gpt-oss-120B MLX", 120, 84.5, 2.0, "MoE", False),
    ("gpt-oss-120B GGUF", 120, 84.0, 1.3, "MoE", False),
    ("Qwen3-235B-2507", 235, 86.0, 1.9, "MoE", False),
    ("Qwen3-235B-A22B", 235, 84.2, 6.2, "MoE", False),
    ("Qwen3-Next-80B", 80, 83.5, 0.6, "MoE", False),
    ("Nemotron-3-Nano", 49, 80.2, 8.1, "Thinking", True),
    ("Qwen3-32B@8bit", 32, 79.3, 1.6, "Dense", False),
    ("Qwen3-32B@4bit", 32, 78.8, 1.5, "Dense", False),
    ("Qwen3-30B-A3B-2507", 30, 78.8, 0.4, "MoE", False),
    ("Swallow-70B", 70, 78.0, 1.9, "JP-FT", False),
    ("GPT-OSS-Swallow-20B", 20, 77.8, 11.2, "JP-FT", False),
    ("Llama 4 Scout", 109, 77.5, 5.0, "MoE", False),
    ("Mistral-Small-3.2", 24, 76.8, 1.0, "Dense", False),
    ("Mistral-Large", 123, 75.8, 6.2, "Dense", False),
    ("Shisa-v2.1-70B", 70, 74.2, 2.4, "JP-FT", False),
    ("Qwen3-14B", 14, 71.8, 0.6, "Dense", False),
    ("MedGemma-27B", 27, 71.8, 0.8, "Dense", False),
    ("gpt-oss-20B MLX", 20, 71.5, 1.2, "MoE", False),
    ("Llama-3.3-70B", 70, 71.0, 1.9, "Dense", False),
    ("Gemma-3-27B", 27, 67.8, 1.0, "Dense", False),
    ("Phi-4", 14, 62.8, 0.7, "Dense", False),
    ("Qwen3-8B", 8, 61.3, 0.4, "Dense", False),
    ("Ezo2.5-12B", 12, 60.0, 0.4, "JP-FT", False),
]

ARCH_COLORS = {
    "Dense": "#1f77b4", "MoE": "#ff7f0e",
    "Thinking": "#d62728", "JP-FT": "#9467bd",
}
ARCH_MARKERS = {
    "Dense": "o", "MoE": "D", "Thinking": "^", "JP-FT": "p",
}


def create_plot(lang="jp"):
    is_jp = lang == "jp"
    if is_jp:
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else:
        matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ====== (a) Accuracy vs Size: Thinking models punch above their weight ======
    ax1 = axes[0]

    # Plot all non-thinking models
    for m in ALL_MODELS:
        name, params, acc, time_s, arch, is_think = m
        color = ARCH_COLORS[arch]
        marker = ARCH_MARKERS[arch]
        size = 180 if is_think else 80
        edge_w = 2 if is_think else 0.5
        zorder = 10 if is_think else 5
        alpha = 1.0 if is_think else 0.6

        ax1.scatter(params, acc, c=color, marker=marker, s=size,
                    edgecolors='black', linewidths=edge_w, zorder=zorder, alpha=alpha)

    # Regression line for non-thinking models only
    non_think = [(m[1], m[2]) for m in ALL_MODELS if not m[5]]
    x_nt = np.log10([p for p, _ in non_think])
    y_nt = np.array([a for _, a in non_think])
    slope, intercept = np.polyfit(x_nt, y_nt, 1)
    x_fit = np.linspace(np.log10(6), np.log10(500), 100)
    y_fit = slope * x_fit + intercept
    baseline_label = '非Thinking回帰線' if is_jp else 'Non-thinking regression'
    ax1.plot(10**x_fit, y_fit, color='gray', linestyle='--', linewidth=2, alpha=0.5,
             zorder=2, label=baseline_label)

    # Arrows showing "equivalent size" for thinking models
    # Qwen3.5-27B thinking (87.3%) — find where regression hits 87.3%
    equiv_27b = 10 ** ((87.3 - intercept) / slope)
    ax1.annotate('', xy=(equiv_27b, 87.3), xytext=(27, 87.3),
                 arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5, alpha=0.7))
    if is_jp:
        ax1.text((27 + equiv_27b) / 2, 88.0, f'Thinking効果\n27B → {equiv_27b:.0f}B相当',
                 fontsize=8, color='#d62728', ha='center', fontweight='bold', alpha=0.8)
    else:
        ax1.text((27 + equiv_27b) / 2, 88.0, f'Thinking effect\n27B → equiv. {equiv_27b:.0f}B',
                 fontsize=8, color='#d62728', ha='center', fontweight='bold', alpha=0.8)

    # Nemotron-3-Nano (80.2%)
    equiv_49b = 10 ** ((80.2 - intercept) / slope)
    ax1.annotate('', xy=(equiv_49b, 80.2), xytext=(49, 80.2),
                 arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5, alpha=0.7))
    if is_jp:
        ax1.text((49 + equiv_49b) / 2, 81.0, f'49B → {equiv_49b:.0f}B相当',
                 fontsize=7, color='#d62728', ha='center', alpha=0.8)
    else:
        ax1.text((49 + equiv_49b) / 2, 81.0, f'49B → equiv. {equiv_49b:.0f}B',
                 fontsize=7, color='#d62728', ha='center', alpha=0.8)

    # Annotate thinking models
    ax1.annotate('Qwen3.5-27B\n(Thinking)', (27, 87.3),
                 fontsize=7, xytext=(-50, -15), textcoords='offset points',
                 arrowprops=dict(arrowstyle='-', alpha=0.4, lw=0.5), alpha=0.8)
    ax1.annotate('Nemotron-3-Nano\n(Thinking)', (49, 80.2),
                 fontsize=7, xytext=(-60, 8), textcoords='offset points',
                 arrowprops=dict(arrowstyle='-', alpha=0.4, lw=0.5), alpha=0.8)

    # Annotate key non-thinking models for reference
    ref_annot = {
        "Qwen3.5-397B@8bit": (5, 5), "Qwen3-235B-2507": (5, -10),
        "gpt-oss-120B GGUF": (5, 5), "Qwen3-Next-80B": (5, -10),
        "Qwen3-32B@8bit": (5, -10), "Qwen3-14B": (-40, 5),
        "Phi-4": (-25, -10), "Qwen3-8B": (5, 5),
    }
    for m in ALL_MODELS:
        if m[0] in ref_annot and not m[5]:
            dx, dy = ref_annot[m[0]]
            short = m[0].replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-")
            short = short.replace("gpt-oss-120B GGUF", "OSS-120B")
            ax1.annotate(short, (m[1], m[2]), fontsize=6, xytext=(dx, dy),
                         textcoords='offset points', alpha=0.6)

    # Legend
    for arch in ["Dense", "MoE", "Thinking", "JP-FT"]:
        label = arch if not is_jp else {"Dense": "Dense", "MoE": "MoE",
                "Thinking": "Thinking", "JP-FT": "日本語FT"}[arch]
        ax1.scatter([], [], c=ARCH_COLORS[arch], marker=ARCH_MARKERS[arch],
                    s=80, edgecolors='black', linewidths=0.5, label=label)

    ax1.set_xscale('log')
    ax1.set_xticks([8, 14, 27, 49, 80, 120, 235, 400])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_ylim(55, 93)
    if is_jp:
        ax1.set_xlabel('モデルサイズ (パラメータ数 B, 対数)', fontsize=11)
        ax1.set_ylabel('正答率 (%)', fontsize=11)
        ax1.set_title('(a) Thinkingは小型モデルの「実効サイズ」を拡大\n赤矢印 = 回帰線上の等価サイズ',
                       fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Model Size (Parameters B, log scale)', fontsize=11)
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('(a) Thinking expands "effective size" of small models\nRed arrow = equivalent size on regression line',
                       fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7.5, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # ====== (b) Accuracy vs Time: cost of thinking ======
    ax2 = axes[1]

    for m in ALL_MODELS:
        name, params, acc, time_s, arch, is_think = m
        color = ARCH_COLORS[arch]
        marker = ARCH_MARKERS[arch]
        size = 180 if is_think else 80
        edge_w = 2 if is_think else 0.5
        zorder = 10 if is_think else 5
        alpha = 1.0 if is_think else 0.6

        ax2.scatter(time_s, acc, c=color, marker=marker, s=size,
                    edgecolors='black', linewidths=edge_w, zorder=zorder, alpha=alpha)

    # Highlight the trade-off zones
    # Fast & good zone
    ax2.axhspan(75, 93, xmin=0, xmax=0.15, alpha=0.05, color='green', zorder=0)
    # Slow zone
    ax2.axvspan(10, 80, alpha=0.04, color='red', zorder=0)
    # Passing line
    ax2.axhline(y=75, color='red', linestyle=':', linewidth=1, alpha=0.4)

    # Annotate thinking models with time cost
    ax2.annotate('Qwen3.5-27B\n(Thinking)\n70.3s', (70.3, 87.3),
                 fontsize=7, xytext=(-55, -15), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5, alpha=0.6),
                 color='#d62728', fontweight='bold', alpha=0.9)
    ax2.annotate('Nemotron-3-Nano\n(Thinking)\n8.1s', (8.1, 80.2),
                 fontsize=7, xytext=(10, 8), textcoords='offset points',
                 color='#d62728', fontweight='bold', alpha=0.9)

    # Annotate key fast models for contrast
    fast_annot = {
        "Qwen3-Next-80B": (5, -10), "Qwen3-30B-A3B-2507": (5, 5),
        "Qwen3-235B-2507": (5, 5), "Mistral-Small-3.2": (5, -10),
        "gpt-oss-120B GGUF": (5, 5), "Qwen3.5-397B@8bit": (-55, 5),
    }
    for m in ALL_MODELS:
        if m[0] in fast_annot:
            dx, dy = fast_annot[m[0]]
            short = m[0].replace("Qwen3.5-", "Q3.5-").replace("Qwen3-", "Q3-")
            short = short.replace("gpt-oss-120B GGUF", "OSS-120B")
            short = short.replace("Qwen3-30B-A3B-2507", "Q3-30B-A3B")
            short = short.replace("Mistral-Small-3.2", "Mistral-Sm")
            ax2.annotate(f'{short}\n{m[3]:.1f}s', (m[3], m[2]),
                         fontsize=6, xytext=(dx, dy), textcoords='offset points',
                         alpha=0.7)

    # Summary box
    if is_jp:
        summary = ('Thinkingのトレードオフ:\n'
                   '• Q3.5-27B: +8.0pp精度 → ×44倍遅い\n'
                   '• Nemotron: +1.4pp精度 → ×4.3倍遅い\n'
                   '→ 大型モデルほど恩恵は小さく\n'
                   '   速度コストは常に高い')
    else:
        summary = ('Thinking trade-off:\n'
                   '• Q3.5-27B: +8.0pp acc → 44× slower\n'
                   '• Nemotron: +1.4pp acc → 4.3× slower\n'
                   '→ Less benefit for larger models\n'
                   '   Speed cost is always high')
    ax2.text(0.97, 0.03, summary, transform=ax2.transAxes,
             fontsize=7.5, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    ax2.set_xscale('log')
    ax2.set_xticks([0.3, 1, 3, 10, 30, 70])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_ylim(55, 93)
    if is_jp:
        ax2.set_xlabel('応答時間 (秒/問, 対数)', fontsize=11)
        ax2.set_ylabel('正答率 (%)', fontsize=11)
        ax2.set_title('(b) Thinkingの代償: 精度向上 vs 速度低下\n赤領域 = 実用に課題のある速度帯',
                       fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('Response Time (sec/question, log scale)', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.set_title('(b) Cost of Thinking: Accuracy gain vs. Speed loss\nRed zone = impractical speed range',
                       fontsize=12, fontweight='bold')

    # Legend
    for arch in ["Dense", "MoE", "Thinking", "JP-FT"]:
        label = arch if not is_jp else {"Dense": "Dense", "MoE": "MoE",
                "Thinking": "Thinking", "JP-FT": "日本語FT"}[arch]
        ax2.scatter([], [], c=ARCH_COLORS[arch], marker=ARCH_MARKERS[arch],
                    s=80, edgecolors='black', linewidths=0.5, label=label)
    ax2.legend(fontsize=7.5, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    if is_jp:
        plt.suptitle('Thinking機構の効果分析: 小型モデルほど恩恵大、しかし速度コストは常に高い\n'
                     'IgakuQA 第116回医師国家試験 | Mac Studio M3 Ultra 512GB',
                     fontsize=13, fontweight='bold')
    else:
        plt.suptitle('Thinking Mechanism Analysis: Greater benefit for smaller models, but speed cost is always high\n'
                     '116th JMLE (IgakuQA) | Mac Studio M3 Ultra 512GB',
                     fontsize=13, fontweight='bold')

    plt.tight_layout()

    suffix = "" if is_jp else "_en"
    outpath = f"plots/thinking_effect{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_plot("jp")
    create_plot("en")
    print("Done!")
