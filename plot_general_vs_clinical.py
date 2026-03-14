#!/usr/bin/env python3
"""
一般問題 vs 臨床問題: モデルサイズとの関係
日本語版と英語版を同時生成
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.unicode_minus'] = False

# (name, total_params_B, arch, ippan%, rinsho%)
DATA = [
    ("Q3-14B", 14, "Dense", 77.6, 69.1),
    ("Mistral-Small", 24, "Dense", 78.4, 76.0),
    ("Q3.5-27B@8bit", 27, "Thinking", 89.6, 86.2),
    ("Gemma-3-27B", 27, "Dense", 74.4, 64.7),
    ("Q3-VL-30B", 30, "VL", 79.2, 77.1),
    ("Q3-VL-32B", 32, "VL", 83.2, 82.5),
    ("Nemotron-Nano", 49, "Thinking", 83.2, 78.9),
    ("Swallow-70B", 70, "JP-FT", 81.6, 76.4),
    ("Shisa-v2.1-70B", 70, "JP-FT", 77.6, 72.7),
    ("Llama-3.3-70B", 70, "Dense", 73.6, 69.8),
    ("Q3-Next-80B", 80, "MoE", 84.0, 83.3),
    ("Llama4-Scout", 109, "MoE", 79.2, 76.7),
    ("gpt-oss-120B", 120, "MoE", 87.2, 83.3),
    ("Mistral-Large", 123, "Dense", 82.4, 72.7),
    ("Q3.5-397B@8bit", 397, "MoE", 92.0, 88.4),
    ("Q3.5-397B@4bit", 397, "MoE", 89.6, 86.2),
]

ARCH_STYLE = {
    "Dense":    {"color": "#1f77b4", "marker": "o"},
    "MoE":      {"color": "#ff7f0e", "marker": "D"},
    "VL":       {"color": "#2ca02c", "marker": "s"},
    "Thinking": {"color": "#d62728", "marker": "^"},
    "JP-FT":    {"color": "#9467bd", "marker": "p"},
}

ARCH_LABEL_EN = {
    "JP-FT": "JP Fine-tuned",
}


def create_plot(lang="jp"):
    is_jp = lang == "jp"
    if is_jp:
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else:
        matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ====== (a) Model Size vs Accuracy: General vs Clinical ======
    ax1 = axes[0]

    for name, params, arch, ippan, rinsho in DATA:
        st = ARCH_STYLE[arch]
        # General (filled)
        ax1.scatter(params, ippan, c=st["color"], marker=st["marker"],
                    s=100, edgecolors='black', linewidths=0.5, zorder=5, alpha=0.9)
        # Clinical (hollow)
        ax1.scatter(params, rinsho, c='white', marker=st["marker"],
                    s=100, edgecolors=st["color"], linewidths=2, zorder=4, alpha=0.9)
        # Connect with line
        ax1.plot([params, params], [ippan, rinsho], color=st["color"], alpha=0.3, linewidth=1.5, zorder=3)

    ax1.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Regression lines (log scale)
    log_sizes = np.log10([d[1] for d in DATA])
    ippan_arr = np.array([d[3] for d in DATA])
    rinsho_arr = np.array([d[4] for d in DATA])

    x_fit = np.linspace(np.log10(10), np.log10(500), 100)
    x_fit_lin = 10 ** x_fit

    # General regression
    slope_i, intercept_i = np.polyfit(log_sizes, ippan_arr, 1)
    r_i = np.corrcoef(log_sizes, ippan_arr)[0, 1]
    y_fit_i = slope_i * x_fit + intercept_i
    gen_label = f'一般 回帰: {slope_i:.1f}%/decade (R²={r_i**2:.2f})' if is_jp else \
                f'General fit: {slope_i:.1f}%/decade (R²={r_i**2:.2f})'
    ax1.plot(x_fit_lin, y_fit_i, color='#1f77b4', linestyle='-', linewidth=2.5,
             alpha=0.4, zorder=2, label=gen_label)

    # Clinical regression
    slope_r, intercept_r = np.polyfit(log_sizes, rinsho_arr, 1)
    r_r = np.corrcoef(log_sizes, rinsho_arr)[0, 1]
    y_fit_r = slope_r * x_fit + intercept_r
    clin_label = f'臨床 回帰: {slope_r:.1f}%/decade (R²={r_r**2:.2f})' if is_jp else \
                 f'Clinical fit: {slope_r:.1f}%/decade (R²={r_r**2:.2f})'
    ax1.plot(x_fit_lin, y_fit_r, color='#e377c2', linestyle='--', linewidth=2.5,
             alpha=0.5, zorder=2, label=clin_label)

    # Slope comparison annotation
    if is_jp:
        slope_text = f'臨床の傾き {slope_r/slope_i:.2f}× 急勾配\n14B→400B: 一般+{slope_i*np.log10(400/14):.1f}pp / 臨床+{slope_r*np.log10(400/14):.1f}pp'
    else:
        slope_text = f'Clinical slope {slope_r/slope_i:.2f}× steeper\n14B→400B: General +{slope_i*np.log10(400/14):.1f}pp / Clinical +{slope_r*np.log10(400/14):.1f}pp'
    ax1.text(0.03, 0.12, slope_text, transform=ax1.transAxes,
             fontsize=7, alpha=0.7, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Legend
    for arch, st in ARCH_STYLE.items():
        label_arch = ARCH_LABEL_EN.get(arch, arch) if not is_jp else arch
        general_label = "General" if not is_jp else "一般"
        ax1.scatter([], [], c=st["color"], marker=st["marker"], s=80,
                    edgecolors='black', linewidths=0.5, label=f'{label_arch} ({general_label})')
    clinical_label = 'Clinical (hollow)' if not is_jp else '臨床 (白抜き)'
    ax1.scatter([], [], c='white', marker='o', s=80, edgecolors='gray',
                linewidths=2, label=clinical_label)

    # Annotate select models
    annot = {
        "Q3.5-397B@8bit": (5, 5), "Q3.5-27B@8bit": (-15, 8),
        "gpt-oss-120B": (5, 5), "Q3-Next-80B": (5, -12),
        "Mistral-Large": (5, 5), "Gemma-3-27B": (-15, -12),
        "Q3-14B": (-15, 5), "Swallow-70B": (-15, -12),
    }
    for name, params, arch, ippan, rinsho in DATA:
        if name in annot:
            dx, dy = annot[name]
            gap = rinsho - ippan
            ax1.annotate(f'{name}\n(Gap {gap:+.1f}%)', (params, (ippan+rinsho)/2),
                         fontsize=6.5, xytext=(dx, dy), textcoords='offset points',
                         alpha=0.8)

    ax1.set_xscale('log')
    if is_jp:
        ax1.set_xlabel('モデルサイズ (パラメータ数 B, 対数)', fontsize=11)
        ax1.set_ylabel('正答率 (%)', fontsize=11)
        ax1.set_title('(a) モデルサイズ vs 正答率\n塗り=一般問題, 白抜き=臨床問題', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Model Size (Total Parameters B, log scale)', fontsize=11)
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('(a) Model Size vs. Accuracy\nFilled=General, Hollow=Clinical', fontsize=12, fontweight='bold')
    ax1.set_ylim(60, 95)
    ax1.set_xticks([14, 27, 50, 80, 120, 200, 400])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend(fontsize=7.5, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    threshold_label = '合格ライン (75%)' if is_jp else 'Passing threshold (75%)'
    ax1.text(0.03, 0.97, threshold_label, transform=ax1.transAxes,
             fontsize=8, color='red', alpha=0.6, verticalalignment='top')

    # ====== (b) Same-model paired comparison: General vs Clinical ======
    ax2 = axes[1]

    # Diagonal line (y=x: clinical == general)
    diag = np.linspace(60, 95, 100)
    ax2.plot(diag, diag, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
    ax2.fill_between(diag, 60, diag, alpha=0.03, color='red', zorder=0)

    # Bubble size proportional to model size (log scale)
    all_params = np.array([d[1] for d in DATA])
    bubble_sizes = 40 + 160 * (np.log10(all_params) / np.log10(all_params.max()))

    # Plot each model
    for i, (name, params, arch, ippan, rinsho) in enumerate(DATA):
        st = ARCH_STYLE[arch]
        ax2.scatter(ippan, rinsho, c=st["color"], marker=st["marker"],
                    s=bubble_sizes[i], edgecolors='black', linewidths=0.5, zorder=5, alpha=0.85)

    # Regression line (general -> clinical)
    slope_gc, intercept_gc = np.polyfit(ippan_arr, rinsho_arr, 1)
    r_gc = np.corrcoef(ippan_arr, rinsho_arr)[0, 1]
    x_reg = np.linspace(72, 93, 100)
    y_reg = slope_gc * x_reg + intercept_gc
    if is_jp:
        reg_label = f'回帰: 臨床 = {slope_gc:.2f}×一般 {intercept_gc:+.1f} (R²={r_gc**2:.2f})'
    else:
        reg_label = f'Fit: Clinical = {slope_gc:.2f}×General {intercept_gc:+.1f} (R²={r_gc**2:.2f})'
    ax2.plot(x_reg, y_reg, color='#e377c2', linestyle='--', linewidth=2, alpha=0.6, zorder=2,
             label=reg_label)

    # Annotate offsets for readability
    annot2 = {
        "Q3.5-397B@8bit": (5, -10), "Q3.5-397B@4bit": (-50, -5),
        "Q3.5-27B@8bit": (5, -10), "gpt-oss-120B": (5, 5),
        "Q3-Next-80B": (5, 5), "Gemma-3-27B": (-50, 5),
        "Mistral-Large": (5, -10), "Swallow-70B": (-45, -8),
        "Q3-14B": (-35, -10), "Llama-3.3-70B": (-50, -5),
        "Nemotron-Nano": (5, 5), "Q3-VL-32B": (5, 5),
        "Mistral-Small": (5, -10), "Shisa-v2.1-70B": (-50, 5),
        "Q3-VL-30B": (-45, -8), "Llama4-Scout": (5, -10),
    }
    for name, params, arch, ippan, rinsho in DATA:
        dx, dy = annot2.get(name, (5, 5))
        gap = rinsho - ippan
        ax2.annotate(f'{name}\n({gap:+.1f}%)', (ippan, rinsho),
                     fontsize=6, xytext=(dx, dy), textcoords='offset points',
                     alpha=0.8,
                     arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5) if abs(dx) > 20 else None)

    # Legend for architectures
    for arch, st in ARCH_STYLE.items():
        label_arch = ARCH_LABEL_EN.get(arch, arch) if not is_jp else arch
        ax2.scatter([], [], c=st["color"], marker=st["marker"], s=80,
                    edgecolors='black', linewidths=0.5, label=label_arch)

    if is_jp:
        ax2.set_xlabel('一般問題 正答率 (%)', fontsize=11)
        ax2.set_ylabel('臨床問題 正答率 (%)', fontsize=11)
        ax2.set_title('(b) 同一モデル: 一般 vs 臨床 正答率\n対角線下 = 臨床で性能低下', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('General Question Accuracy (%)', fontsize=11)
        ax2.set_ylabel('Clinical Question Accuracy (%)', fontsize=11)
        ax2.set_title('(b) Same Model: General vs. Clinical Accuracy\nBelow diagonal = clinical penalty', fontsize=12, fontweight='bold')
    ax2.set_xlim(72, 94)
    ax2.set_ylim(62, 92)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # Mean gap annotation
    mean_gap = np.mean(rinsho_arr - ippan_arr)
    if is_jp:
        gap_text = f'全モデル平均Gap: {mean_gap:+.1f}%\n対角線(y=x)= 差なし\n赤領域 = 臨床が苦手'
    else:
        gap_text = f'Mean gap across models: {mean_gap:+.1f}%\nDiagonal (y=x) = no difference\nShaded = weaker on clinical'
    ax2.text(0.97, 0.03, gap_text, transform=ax2.transAxes,
             fontsize=7, alpha=0.6, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    if is_jp:
        plt.suptitle('IgakuQA 2022: 一般問題 (125問) vs 臨床問題 (275問) のモデルサイズ依存性',
                     fontsize=14, fontweight='bold')
    else:
        plt.suptitle('IgakuQA 2022: General (125Q) vs. Clinical (275Q) — Model Size Dependency',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    suffix = "" if is_jp else "_en"
    outpath = f"plots/general_vs_clinical{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_plot("jp")
    create_plot("en")
    print("Done!")
