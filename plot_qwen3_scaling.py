#!/usr/bin/env python3
"""
Qwen3ファミリーのみ: 一般問題 vs 臨床問題のスケーリング比較
日本語版と英語版を同時生成
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.unicode_minus'] = False

# Qwen3 models with full 400Q evaluation (general%, clinical%)
# Q3-235B-A22B excluded due to anomalous Section A/B data
QWEN3_DATA = [
    ("Q3-VL-4B@4bit", 4, "VL", 58.4, 58.2),
    ("Q3-VL-4B@8bit", 4, "VL", 62.4, 59.6),
    ("Q3-VL-8B@4bit", 8, "VL", 64.0, 65.8),
    ("Q3-VL-8B@8bit", 8, "VL", 69.6, 69.8),
    ("Q3-14B", 14, "Dense", 77.6, 69.1),
    ("Q3.5-27B@8bit", 27, "Thinking", 88.0, 86.2),
    ("Q3-30B-A3B", 30, "MoE", 81.6, 77.5),
    ("Q3-VL-30B", 30, "VL", 79.2, 77.1),
    ("Q3-32B@4bit", 32, "Dense", 81.6, 77.5),
    ("Q3-VL-32B", 32, "VL", 83.2, 82.5),
    ("Q3-Next-80B", 80, "MoE", 84.0, 83.3),
    ("Q3-235B-2507", 235, "MoE", 88.8, 84.7),
    ("Q3.5-397B@8bit", 397, "MoE", 90.4, 88.4),
    ("Q3.5-397B@4bit", 397, "MoE", 87.2, 86.2),
]

SUB_STYLE = {
    "Dense":    {"color": "#1f77b4", "marker": "o"},
    "MoE":      {"color": "#ff7f0e", "marker": "D"},
    "VL":       {"color": "#2ca02c", "marker": "s"},
    "Thinking": {"color": "#d62728", "marker": "^"},
}


def create_plot(lang="jp"):
    is_jp = lang == "jp"
    if is_jp:
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else:
        matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    log_sizes = np.log10([d[1] for d in QWEN3_DATA])
    ippan_arr = np.array([d[3] for d in QWEN3_DATA])
    rinsho_arr = np.array([d[4] for d in QWEN3_DATA])

    # ====== (a) Model Size vs Accuracy with regression ======
    ax1 = axes[0]

    for name, params, sub, ippan, rinsho in QWEN3_DATA:
        st = SUB_STYLE[sub]
        # General (filled)
        ax1.scatter(params, ippan, c=st["color"], marker=st["marker"],
                    s=100, edgecolors='black', linewidths=0.5, zorder=5, alpha=0.9)
        # Clinical (hollow)
        ax1.scatter(params, rinsho, c='white', marker=st["marker"],
                    s=100, edgecolors=st["color"], linewidths=2, zorder=4, alpha=0.9)
        # Connect
        ax1.plot([params, params], [ippan, rinsho], color=st["color"], alpha=0.3, linewidth=1.5, zorder=3)

    ax1.axhline(y=75, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Regression lines
    x_fit = np.linspace(np.log10(3), np.log10(500), 100)
    x_fit_lin = 10 ** x_fit

    slope_i, intercept_i = np.polyfit(log_sizes, ippan_arr, 1)
    r_i = np.corrcoef(log_sizes, ippan_arr)[0, 1]
    y_fit_i = slope_i * x_fit + intercept_i
    gen_label = f'一般: {slope_i:.1f}%/decade (R²={r_i**2:.2f})' if is_jp else \
                f'General: {slope_i:.1f}%/decade (R²={r_i**2:.2f})'
    ax1.plot(x_fit_lin, y_fit_i, color='#1f77b4', linestyle='-', linewidth=2.5,
             alpha=0.4, zorder=2, label=gen_label)

    slope_r, intercept_r = np.polyfit(log_sizes, rinsho_arr, 1)
    r_r = np.corrcoef(log_sizes, rinsho_arr)[0, 1]
    y_fit_r = slope_r * x_fit + intercept_r
    clin_label = f'臨床: {slope_r:.1f}%/decade (R²={r_r**2:.2f})' if is_jp else \
                 f'Clinical: {slope_r:.1f}%/decade (R²={r_r**2:.2f})'
    ax1.plot(x_fit_lin, y_fit_r, color='#e377c2', linestyle='--', linewidth=2.5,
             alpha=0.5, zorder=2, label=clin_label)

    # Slope comparison
    if is_jp:
        slope_text = (f'臨床の傾き {slope_r/slope_i:.2f}× 急勾配\n'
                      f'4B→397B: 一般+{slope_i*np.log10(397/4):.1f}pp / 臨床+{slope_r*np.log10(397/4):.1f}pp')
    else:
        slope_text = (f'Clinical slope {slope_r/slope_i:.2f}× steeper\n'
                      f'4B→397B: General +{slope_i*np.log10(397/4):.1f}pp / Clinical +{slope_r*np.log10(397/4):.1f}pp')
    ax1.text(0.03, 0.12, slope_text, transform=ax1.transAxes,
             fontsize=7, alpha=0.7, verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # Legend for sub-families
    for sub, st in SUB_STYLE.items():
        gen_word = "一般" if is_jp else "General"
        ax1.scatter([], [], c=st["color"], marker=st["marker"], s=80,
                    edgecolors='black', linewidths=0.5, label=f'{sub} ({gen_word})')
    clin_word = '臨床 (白抜き)' if is_jp else 'Clinical (hollow)'
    ax1.scatter([], [], c='white', marker='o', s=80, edgecolors='gray',
                linewidths=2, label=clin_word)

    # Annotate all models
    annot1 = {
        "Q3-VL-4B@4bit": (-10, -12), "Q3-VL-4B@8bit": (-10, 6),
        "Q3-VL-8B@4bit": (-10, -12), "Q3-VL-8B@8bit": (-10, 6),
        "Q3-14B": (5, -8), "Q3.5-27B@8bit": (5, 5),
        "Q3-30B-A3B": (5, -12), "Q3-VL-30B": (-40, -8),
        "Q3-32B@4bit": (5, 5), "Q3-VL-32B": (5, 5),
        "Q3-Next-80B": (5, -10), "Q3-235B-2507": (5, 5),
        "Q3.5-397B@8bit": (5, 5), "Q3.5-397B@4bit": (-55, -5),
    }
    for name, params, sub, ippan, rinsho in QWEN3_DATA:
        dx, dy = annot1.get(name, (5, 5))
        gap = rinsho - ippan
        ax1.annotate(f'{name}\n({gap:+.1f}%)', (params, (ippan + rinsho) / 2),
                     fontsize=5.5, xytext=(dx, dy), textcoords='offset points',
                     alpha=0.8,
                     arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5) if abs(dx) > 20 else None)

    ax1.set_xscale('log')
    if is_jp:
        ax1.set_xlabel('モデルサイズ (パラメータ数 B, 対数)', fontsize=11)
        ax1.set_ylabel('正答率 (%)', fontsize=11)
        ax1.set_title('(a) Qwen3 スケーリング: 一般 vs 臨床\n塗り=一般問題, 白抜き=臨床問題', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Model Size (Total Parameters B, log scale)', fontsize=11)
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('(a) Qwen3 Scaling: General vs. Clinical\nFilled=General, Hollow=Clinical', fontsize=12, fontweight='bold')
    ax1.set_ylim(52, 95)
    ax1.set_xticks([4, 8, 14, 30, 80, 235, 400])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend(fontsize=7, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    threshold = '合格ライン (75%)' if is_jp else 'Passing threshold (75%)'
    ax1.text(0.03, 0.97, threshold, transform=ax1.transAxes,
             fontsize=8, color='red', alpha=0.6, verticalalignment='top')

    # ====== (b) Paired comparison: General vs Clinical ======
    ax2 = axes[1]

    # Diagonal
    diag = np.linspace(50, 95, 100)
    ax2.plot(diag, diag, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
    ax2.fill_between(diag, 50, diag, alpha=0.03, color='red', zorder=0)

    # Bubble size by params
    all_params = np.array([d[1] for d in QWEN3_DATA])
    bubble_sizes = 50 + 150 * (np.log10(all_params) / np.log10(all_params.max()))

    for i, (name, params, sub, ippan, rinsho) in enumerate(QWEN3_DATA):
        st = SUB_STYLE[sub]
        ax2.scatter(ippan, rinsho, c=st["color"], marker=st["marker"],
                    s=bubble_sizes[i], edgecolors='black', linewidths=0.5, zorder=5, alpha=0.85)

    # Regression
    slope_gc, intercept_gc = np.polyfit(ippan_arr, rinsho_arr, 1)
    r_gc = np.corrcoef(ippan_arr, rinsho_arr)[0, 1]
    x_reg = np.linspace(55, 92, 100)
    y_reg = slope_gc * x_reg + intercept_gc
    if is_jp:
        reg_label = f'回帰: 臨床 = {slope_gc:.2f}×一般 {intercept_gc:+.1f} (R²={r_gc**2:.2f})'
    else:
        reg_label = f'Fit: Clinical = {slope_gc:.2f}×General {intercept_gc:+.1f} (R²={r_gc**2:.2f})'
    ax2.plot(x_reg, y_reg, color='#e377c2', linestyle='--', linewidth=2, alpha=0.6, zorder=2,
             label=reg_label)

    # Annotate
    annot2 = {
        "Q3-VL-4B@4bit": (5, -10), "Q3-VL-4B@8bit": (-50, 5),
        "Q3-VL-8B@4bit": (5, 5), "Q3-VL-8B@8bit": (5, -10),
        "Q3-14B": (5, -10), "Q3.5-27B@8bit": (5, -10),
        "Q3-30B-A3B": (-50, -8), "Q3-VL-30B": (5, 5),
        "Q3-32B@4bit": (-50, -8), "Q3-VL-32B": (5, 5),
        "Q3-Next-80B": (5, 5), "Q3-235B-2507": (5, -10),
        "Q3.5-397B@8bit": (5, -10), "Q3.5-397B@4bit": (-55, 5),
    }
    for name, params, sub, ippan, rinsho in QWEN3_DATA:
        dx, dy = annot2.get(name, (5, 5))
        gap = rinsho - ippan
        ax2.annotate(f'{name}\n({gap:+.1f}%)', (ippan, rinsho),
                     fontsize=5.5, xytext=(dx, dy), textcoords='offset points',
                     alpha=0.8,
                     arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5) if abs(dx) > 20 else None)

    # Sub-family legend
    for sub, st in SUB_STYLE.items():
        ax2.scatter([], [], c=st["color"], marker=st["marker"], s=80,
                    edgecolors='black', linewidths=0.5, label=sub)

    if is_jp:
        ax2.set_xlabel('一般問題 正答率 (%)', fontsize=11)
        ax2.set_ylabel('臨床問題 正答率 (%)', fontsize=11)
        ax2.set_title('(b) 同一モデル: 一般 vs 臨床\n対角線下 = 臨床で性能低下', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('General Question Accuracy (%)', fontsize=11)
        ax2.set_ylabel('Clinical Question Accuracy (%)', fontsize=11)
        ax2.set_title('(b) Same Model: General vs. Clinical\nBelow diagonal = clinical penalty', fontsize=12, fontweight='bold')
    ax2.set_xlim(55, 94)
    ax2.set_ylim(55, 92)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend(fontsize=7, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # Mean gap
    mean_gap = np.mean(rinsho_arr - ippan_arr)
    if is_jp:
        gap_text = f'Qwen3平均Gap: {mean_gap:+.1f}%\n対角線(y=x) = 差なし\nVLは小型でもGap小'
    else:
        gap_text = f'Qwen3 mean gap: {mean_gap:+.1f}%\nDiagonal (y=x) = no difference\nVL shows small gap even at small sizes'
    ax2.text(0.97, 0.03, gap_text, transform=ax2.transAxes,
             fontsize=7, alpha=0.6, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    if is_jp:
        plt.suptitle('Qwen3ファミリー: 一般問題 (125問) vs 臨床問題 (275問) スケーリング分析',
                     fontsize=14, fontweight='bold')
    else:
        plt.suptitle('Qwen3 Family: General (125Q) vs. Clinical (275Q) Scaling Analysis',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    suffix = "" if is_jp else "_en"
    outpath = f"plots/qwen3_scaling{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_plot("jp")
    create_plot("en")
    print("Done!")
