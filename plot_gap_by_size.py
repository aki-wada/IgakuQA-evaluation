#!/usr/bin/env python3
"""
一般問題 vs 臨床問題の異なるスケーリングパターンを可視化
一般: 対数線形成長 / 臨床: シグモイド成長（遅発的発達）
日本語版と英語版を同時生成
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

matplotlib.rcParams['axes.unicode_minus'] = False

# Qwen3 full family data (general%, clinical%)
QWEN3 = [
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


def sigmoid(x, L, k, x0, b):
    return L / (1 + np.exp(-k * (x - x0))) + b


def create_plot(lang="jp"):
    is_jp = lang == "jp"
    if is_jp:
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else:
        matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']

    sizes = np.array([d[1] for d in QWEN3])
    log_s = np.log10(sizes)
    gen = np.array([d[3] for d in QWEN3])
    clin = np.array([d[4] for d in QWEN3])
    subs = [d[2] for d in QWEN3]
    names = [d[0] for d in QWEN3]

    # Fit: General = log-linear
    slope_g, int_g = np.polyfit(log_s, gen, 1)
    r_g = np.corrcoef(log_s, gen)[0, 1]

    # Fit: Clinical = sigmoid
    popt_c, _ = curve_fit(sigmoid, log_s, clin, p0=[35, 4, 1.0, 55], maxfev=10000)
    pred_c = sigmoid(log_s, *popt_c)
    ss_res = np.sum((clin - pred_c) ** 2)
    ss_tot = np.sum((clin - clin.mean()) ** 2)
    r2_sig = 1 - ss_res / ss_tot

    # Also get log-linear R² for clinical (for comparison)
    r_c_lin = np.corrcoef(log_s, clin)[0, 1]

    # Smooth curves
    x_sm = np.linspace(np.log10(2), np.log10(600), 300)
    x_sm_lin = 10 ** x_sm
    y_gen_fit = slope_g * x_sm + int_g
    y_clin_fit = sigmoid(x_sm, *popt_c)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ====== (a) Main scaling plot with different curve fits ======
    ax1 = axes[0]

    # Data points
    for i, (name, p, sub, g, c) in enumerate(QWEN3):
        st = SUB_STYLE[sub]
        ax1.scatter(p, g, c=st["color"], marker=st["marker"],
                    s=100, edgecolors='black', linewidths=0.5, zorder=5, alpha=0.9)
        ax1.scatter(p, c, c='white', marker=st["marker"],
                    s=100, edgecolors=st["color"], linewidths=2, zorder=4, alpha=0.9)
        # Connect
        ax1.plot([p, p], [g, c], color=st["color"], alpha=0.25, linewidth=1.2, zorder=3)

    # General: log-linear fit (blue solid)
    if is_jp:
        label_g = f'一般: 対数線形  (R²={r_g**2:.2f})\nacc = {slope_g:.1f}·log₁₀(B) + {int_g:.0f}'
    else:
        label_g = f'General: Log-linear  (R²={r_g**2:.2f})\nacc = {slope_g:.1f}·log₁₀(B) + {int_g:.0f}'
    ax1.plot(x_sm_lin, y_gen_fit, color='#1565C0', linewidth=3, alpha=0.7,
             zorder=2, label=label_g)

    # Clinical: sigmoid fit (pink dashed)
    if is_jp:
        label_c = f'臨床: シグモイド  (R²={r2_sig:.2f})\nL={popt_c[0]:.0f}, k={popt_c[1]:.1f}, x₀={10**popt_c[2]:.0f}B'
    else:
        label_c = f'Clinical: Sigmoid  (R²={r2_sig:.2f})\nL={popt_c[0]:.0f}, k={popt_c[1]:.1f}, x₀≈{10**popt_c[2]:.0f}B'
    ax1.plot(x_sm_lin, y_clin_fit, color='#C62828', linewidth=3, alpha=0.7,
             linestyle='--', zorder=2, label=label_c)

    # Shade the gap
    ax1.fill_between(x_sm_lin, y_clin_fit, y_gen_fit,
                     where=(y_gen_fit > y_clin_fit),
                     alpha=0.08, color='red', zorder=1)

    # Mark the inflection point of sigmoid (where clinical growth accelerates)
    # Inflection at x0
    inflect_x = popt_c[2]  # in log10 space
    inflect_size = 10 ** inflect_x
    inflect_y = sigmoid(inflect_x, *popt_c)
    if is_jp:
        ax1.annotate(f'臨床の変曲点\n~{inflect_size:.0f}B',
                     (inflect_size, inflect_y),
                     fontsize=9, fontweight='bold', color='#C62828',
                     xytext=(30, -25), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.9))
    else:
        ax1.annotate(f'Clinical inflection\n~{inflect_size:.0f}B',
                     (inflect_size, inflect_y),
                     fontsize=9, fontweight='bold', color='#C62828',
                     xytext=(30, -25), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.9))

    # Max gap point
    gap_curve = y_gen_fit - y_clin_fit
    max_idx = np.argmax(gap_curve)
    max_gap_size = x_sm_lin[max_idx]
    max_gap_val = gap_curve[max_idx]
    if is_jp:
        ax1.annotate(f'最大Gap {max_gap_val:.1f}%\n({max_gap_size:.0f}B付近)',
                     (max_gap_size, (y_gen_fit[max_idx] + y_clin_fit[max_idx]) / 2),
                     fontsize=8, color='#B71C1C', alpha=0.8,
                     xytext=(-50, 0), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#B71C1C', alpha=0.6, lw=1))
    else:
        ax1.annotate(f'Max gap {max_gap_val:.1f}%\n(~{max_gap_size:.0f}B)',
                     (max_gap_size, (y_gen_fit[max_idx] + y_clin_fit[max_idx]) / 2),
                     fontsize=8, color='#B71C1C', alpha=0.8,
                     xytext=(-50, 0), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#B71C1C', alpha=0.6, lw=1))

    ax1.axhline(y=75, color='red', linestyle=':', linewidth=1, alpha=0.4)

    # Annotate key models
    annot1 = {
        "Q3-VL-4B@4bit": (-10, -12), "Q3-VL-8B@8bit": (-10, 6),
        "Q3-14B": (5, 5), "Q3.5-27B@8bit": (5, 5),
        "Q3-32B@4bit": (5, -12), "Q3-VL-32B": (5, 5),
        "Q3-Next-80B": (5, -10),
        "Q3.5-397B@8bit": (5, 5), "Q3.5-397B@4bit": (-55, -5),
    }
    for name, p, sub, g, c in QWEN3:
        if name in annot1:
            dx, dy = annot1[name]
            ax1.annotate(name, (p, (g + c) / 2),
                         fontsize=6, xytext=(dx, dy), textcoords='offset points', alpha=0.7,
                         arrowprops=dict(arrowstyle='-', alpha=0.2, lw=0.5) if abs(dx) > 20 else None)

    # Legend for sub-families
    for sub, st in SUB_STYLE.items():
        ax1.scatter([], [], c=st["color"], marker=st["marker"], s=60,
                    edgecolors='black', linewidths=0.5, label=sub)

    ax1.set_xscale('log')
    if is_jp:
        ax1.set_xlabel('モデルサイズ (パラメータ数 B, 対数)', fontsize=11)
        ax1.set_ylabel('正答率 (%)', fontsize=11)
        ax1.set_title('(a) 異なるスケーリングパターン\n一般 = 対数線形 / 臨床 = シグモイド（遅発的成長）',
                      fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Model Size (Parameters B, log scale)', fontsize=11)
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('(a) Different Scaling Patterns\nGeneral = Log-linear / Clinical = Sigmoid (delayed growth)',
                      fontsize=12, fontweight='bold')
    ax1.set_ylim(50, 95)
    ax1.set_xticks([4, 8, 14, 30, 80, 235, 400])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend(fontsize=7, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    threshold = '合格ライン (75%)' if is_jp else 'Passing threshold (75%)'
    ax1.text(0.03, 0.95, threshold, transform=ax1.transAxes,
             fontsize=8, color='red', alpha=0.5, verticalalignment='top')

    # ====== (b) Growth rate (derivative) comparison ======
    ax2 = axes[1]

    # Derivative of general (log-linear): d(acc)/d(log10(B)) = constant
    gen_rate = np.full_like(x_sm, slope_g)

    # Derivative of clinical (sigmoid):
    # d/dx sigmoid = L * k * exp(-k(x-x0)) / (1 + exp(-k(x-x0)))^2
    L, k, x0, b = popt_c
    exp_term = np.exp(-k * (x_sm - x0))
    clin_rate = L * k * exp_term / (1 + exp_term) ** 2

    ax2.plot(x_sm_lin, gen_rate, color='#1565C0', linewidth=3, alpha=0.7,
             label='一般: 一定 (対数線形)' if is_jp else 'General: Constant (log-linear)')
    ax2.plot(x_sm_lin, clin_rate, color='#C62828', linewidth=3, alpha=0.7, linestyle='--',
             label='臨床: ベル型 (シグモイド)' if is_jp else 'Clinical: Bell-shaped (sigmoid)')

    # Shade where clinical grows faster than general
    ax2.fill_between(x_sm_lin, gen_rate, clin_rate,
                     where=(clin_rate > gen_rate),
                     alpha=0.15, color='#66BB6A', zorder=1)
    ax2.fill_between(x_sm_lin, gen_rate, clin_rate,
                     where=(clin_rate <= gen_rate),
                     alpha=0.08, color='#EF5350', zorder=1)

    # Mark the crossover points
    crossover_idx = np.where(np.diff(np.sign(clin_rate - gen_rate)))[0]
    for idx in crossover_idx:
        cross_size = x_sm_lin[idx]
        cross_rate = gen_rate[idx]
        if is_jp:
            ax2.annotate(f'交差点\n~{cross_size:.0f}B',
                         (cross_size, cross_rate),
                         fontsize=8, fontweight='bold', color='#333333',
                         xytext=(15, 15), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='#333333', lw=1))
        else:
            ax2.annotate(f'Crossover\n~{cross_size:.0f}B',
                         (cross_size, cross_rate),
                         fontsize=8, fontweight='bold', color='#333333',
                         xytext=(15, 15), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='#333333', lw=1))

    # Mark sigmoid peak
    peak_idx = np.argmax(clin_rate)
    peak_size = x_sm_lin[peak_idx]
    peak_rate = clin_rate[peak_idx]
    if is_jp:
        ax2.annotate(f'臨床の成長ピーク\n~{peak_size:.0f}B\n(変曲点)',
                     (peak_size, peak_rate),
                     fontsize=8, fontweight='bold', color='#C62828',
                     xytext=(20, 5), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.9))
    else:
        ax2.annotate(f'Clinical growth peak\n~{peak_size:.0f}B\n(inflection point)',
                     (peak_size, peak_rate),
                     fontsize=8, fontweight='bold', color='#C62828',
                     xytext=(20, 5), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.9))

    ax2.set_xscale('log')
    if is_jp:
        ax2.set_xlabel('モデルサイズ (パラメータ数 B, 対数)', fontsize=11)
        ax2.set_ylabel('成長率 (Δ正答率 / Δlog₁₀B)', fontsize=11)
        ax2.set_title('(b) 成長速度の比較\n臨床推論は中型モデルで急成長', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('Model Size (Parameters B, log scale)', fontsize=11)
        ax2.set_ylabel('Growth Rate (Δacc / Δlog₁₀B)', fontsize=11)
        ax2.set_title('(b) Growth Rate Comparison\nClinical reasoning surges at mid-size', fontsize=12, fontweight='bold')
    ax2.set_xticks([4, 8, 14, 30, 80, 235, 400])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, max(clin_rate) * 1.3)

    # Phase labels
    if is_jp:
        phases = [
            (5, '臨床が遅い\n(一般が先行)', '#EF5350'),
            (peak_size, '臨床が急成長\n(Gap縮小)', '#66BB6A'),
            (300, '両方飽和\n(収束)', '#78909C'),
        ]
    else:
        phases = [
            (5, 'Clinical lags\n(General leads)', '#EF5350'),
            (peak_size, 'Clinical surges\n(Gap closes)', '#66BB6A'),
            (300, 'Both saturate\n(convergence)', '#78909C'),
        ]
    for x_pos, label, color in phases:
        ax2.text(x_pos, max(clin_rate) * 1.15, label,
                 ha='center', fontsize=7.5, fontweight='bold', color=color, alpha=0.8)

    if is_jp:
        plt.suptitle('Qwen3ファミリー: 一般問題と臨床問題の異なるスケーリング法則',
                     fontsize=14, fontweight='bold')
    else:
        plt.suptitle('Qwen3 Family: Different Scaling Laws for General vs. Clinical Questions',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    suffix = "" if is_jp else "_en"
    outpath = f"plots/gap_development{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_plot("jp")
    create_plot("en")
    print("Done!")
