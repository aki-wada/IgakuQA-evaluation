#!/usr/bin/env python3
"""
IgakuQA 分野別正答率分析
- 主要モデルの分野別ヒートマップ
- LLM vs 人間の正答率比較
- モデル間で差が大きい分野の特定
"""
import json
import os
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']

# ====== 全セクション評価済みモデルの結果ファイルマッピング ======
# (display_name, file_prefix, best_prompt_key)
MODELS = [
    ("Q3.5-397B@8bit", "prompt_comparison_qwen3.5-397b-a17b-8bit", "format_strict"),
    ("Q3.5-397B@4bit", "prompt_comparison_qwen3.5-397b-a17b@4bit", "format_strict"),
    ("Q3.5-27B@8bit", "prompt_comparison_qwen3.5-27b@8bit", "format_strict"),
    ("Q3-235B-2507", "prompt_comparison_qwen_qwen3-235b-a22b_2022", None),  # special
    ("gpt-oss-120B", "prompt_comparison_openai_gpt-oss-120b", "format_strict"),
    ("Q3-Next-80B", "prompt_comparison_qwen_qwen3-next-80b", "japanese_medical"),
    ("Q3-VL-32B", "prompt_comparison_qwen_qwen3-vl-32b", "baseline"),
    ("Nemotron-Nano", "prompt_comparison_nvidia_nemotron-3-nano", "answer_first"),
    ("Q3-32B@8bit", "qwen3-32b_allsections", "baseline"),
    ("Swallow-70B", "prompt_comparison_tokyotech-llm-llama-3.3-swallow-70b-instruct-v0.4", "baseline"),
    ("Q3-VL-30B", "prompt_comparison_qwen_qwen3-vl-30b", "format_strict"),
    ("Mistral-Small", "prompt_comparison_mistralai_mistral-small-3.2", "baseline"),
    ("Mistral-Large", "prompt_comparison_mistral-large-instruct-2407", "baseline"),
    ("Llama4-Scout", "prompt_comparison_llama-4-scout-17b-16e", "baseline"),
    ("Shisa-v2.1-70B", "prompt_comparison_shisa-v2.1-llama3.3-70b-mlx", "format_strict"),
    ("Llama-3.3-70B", "prompt_comparison_mlx-community_llama-3.3-70b-instruct", "baseline"),
    ("MedGemma-27B", "medgemma-27b", "baseline"),  # B-F; A from prompt_comparison
    ("Gemma-3-27B", "prompt_comparison_google_gemma-3-27b", "chain_of_thought"),
]

SECTIONS = ['A', 'B', 'C', 'D', 'E', 'F']


def load_metadata():
    """Load question metadata (category, human_accuracy) for all sections."""
    meta = {}
    for sec in SECTIONS:
        with open(f'data/2022/116-{sec}_metadata.jsonl') as f:
            for line in f:
                d = json.loads(line)
                meta[d['problem_id']] = d
    return meta


def load_model_results(file_prefix, prompt_key):
    """Load model results across all sections, return dict of problem_id -> correct."""
    results = {}
    for sec in SECTIONS:
        # Try different file naming patterns
        candidates = [
            f'results/{file_prefix}_2022_{sec}.json',
            f'results/{file_prefix}_{sec}.json',
            f'results/{file_prefix}_fewshot_2022_{sec}.json',
        ]
        # Special fallbacks for models with Section A in different files
        if sec == 'A' and 'medgemma-27b' in file_prefix:
            candidates.append('results/prompt_comparison_medgemma-27b-text-it-mlx_2022_A.json')
        if sec == 'A' and 'qwen3-32b_allsections' in file_prefix:
            candidates.append('results/qwen3-32b_fewshot_2022_A.json')
        found = False
        for fpath in candidates:
            if os.path.exists(fpath):
                with open(fpath) as f:
                    data = json.load(f)

                # Determine which key to use for details
                if 'details' in data and isinstance(data['details'], dict):
                    # Multi-prompt format
                    if prompt_key and prompt_key in data['details']:
                        details = data['details'][prompt_key]
                    else:
                        # Use first available key
                        first_key = list(data['details'].keys())[0]
                        details = data['details'][first_key]
                elif 'details' in data and isinstance(data['details'], list):
                    details = data['details']
                elif isinstance(data, list):
                    details = data
                else:
                    continue

                for item in details:
                    pid = item.get('problem_id', item.get('id', ''))
                    results[pid] = item.get('correct', False)
                found = True
                break
        if not found:
            pass  # silently skip missing sections
    return results


def main():
    meta = load_metadata()

    # Group questions by category
    cat_questions = collections.defaultdict(list)
    for pid, m in meta.items():
        cat_questions[m['category']].append(pid)

    # Sort categories by question count (descending)
    cat_sorted = sorted(cat_questions.keys(), key=lambda c: -len(cat_questions[c]))

    # Filter to categories with >= 7 questions for statistical relevance
    cat_filtered = [c for c in cat_sorted if len(cat_questions[c]) >= 7]

    print(f"Total categories: {len(cat_sorted)}")
    print(f"Categories with ≥7 questions: {len(cat_filtered)}")
    print()

    # Load all model results
    model_results = {}
    model_names = []
    for display_name, prefix, prompt_key in MODELS:
        res = load_model_results(prefix, prompt_key)
        if len(res) >= 300:  # need at least 300/400 questions
            model_results[display_name] = res
            model_names.append(display_name)
            print(f"  {display_name}: {len(res)} questions loaded")
        else:
            print(f"  {display_name}: SKIP (only {len(res)} questions)")

    print(f"\nModels loaded: {len(model_names)}")

    # Compute category-level accuracy for each model
    cat_acc = {}  # {model: {category: accuracy}}
    for mname in model_names:
        cat_acc[mname] = {}
        for cat in cat_filtered:
            pids = cat_questions[cat]
            correct = sum(1 for pid in pids if model_results[mname].get(pid, False))
            cat_acc[mname][cat] = correct / len(pids) * 100

    # Human accuracy by category
    human_cat_acc = {}
    for cat in cat_filtered:
        pids = cat_questions[cat]
        human_accs = []
        for pid in pids:
            ha = meta[pid].get('human_accuracy', '')
            if ha and ha.strip():
                try:
                    human_accs.append(float(ha))
                except ValueError:
                    pass
        human_cat_acc[cat] = np.mean(human_accs) if human_accs else 0

    # ====== Figure 1: Category Heatmap (top models) ======
    # Select representative models
    top_models = [m for m in model_names if m in [
        "Q3.5-397B@8bit", "Q3.5-27B@8bit", "gpt-oss-120B",
        "Q3-Next-80B", "Q3-VL-32B", "Q3-32B@8bit",
        "Swallow-70B", "Mistral-Small", "Llama-3.3-70B",
        "MedGemma-27B", "Shisa-v2.1-70B", "Gemma-3-27B",
    ]]

    fig, ax = plt.subplots(figsize=(18, 10))

    # Build matrix
    matrix = []
    row_labels = []
    for mname in top_models:
        row = [cat_acc[mname][cat] for cat in cat_filtered]
        matrix.append(row)
        # Add total accuracy
        total = sum(1 for pid, c in model_results[mname].items() if c) / len(model_results[mname]) * 100
        row_labels.append(f"{mname} ({total:.1f}%)")

    # Add human row
    human_row = [human_cat_acc[cat] for cat in cat_filtered]
    matrix.append(human_row)
    row_labels.append("受験者平均")

    matrix = np.array(matrix)
    col_labels = [f"{c}\n({len(cat_questions[c])}問)" for c in cat_filtered]

    cmap = LinearSegmentedColormap.from_list('custom',
        [(0.0, '#B71C1C'), (0.3, '#FF8A65'), (0.5, '#FFF9C4'),
         (0.75, '#81C784'), (1.0, '#1B5E20')])

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=30, vmax=100)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = 'white' if val >= 85 or val < 40 else 'black'
            fontw = 'bold' if i == len(row_labels) - 1 else 'normal'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight=fontw)

    # Divider before human row
    ax.axhline(y=len(top_models) - 0.5, color='white', linewidth=3)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('正答率 (%)', fontsize=10)

    ax.set_title('IgakuQA 2022: 医学分野別正答率ヒートマップ\n'
                 '主要LLMモデル vs 受験者平均  |  分野別問題数7問以上',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('plots/category_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: plots/category_heatmap.png")
    plt.close()

    # ====== Figure 2: LLM vs Human by Category ======
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Left: Best LLM vs Human scatter ---
    best_model = "Q3.5-397B@8bit"
    llm_accs = [cat_acc[best_model][cat] for cat in cat_filtered]
    human_accs_plot = [human_cat_acc[cat] for cat in cat_filtered]
    q_counts = [len(cat_questions[cat]) for cat in cat_filtered]

    sizes = [30 + 8 * n for n in q_counts]

    ax1.scatter(human_accs_plot, llm_accs, s=sizes, c='#1976D2', alpha=0.7,
                edgecolors='black', linewidths=0.5, zorder=3)

    # Diagonal line (LLM = Human)
    ax1.plot([30, 100], [30, 100], 'k--', alpha=0.3, linewidth=1, label='LLM = 受験者')

    # Annotate categories
    for i, cat in enumerate(cat_filtered):
        dx, dy = 3, 3
        if llm_accs[i] > human_accs_plot[i] + 10:
            dy = -10
        elif llm_accs[i] < human_accs_plot[i] - 10:
            dy = 8
        ax1.annotate(cat, (human_accs_plot[i], llm_accs[i]),
                     fontsize=7, xytext=(dx, dy), textcoords='offset points', alpha=0.8)

    ax1.set_xlabel('受験者平均正答率 (%)', fontsize=11)
    ax1.set_ylabel(f'{best_model} 正答率 (%)', fontsize=11)
    ax1.set_title(f'(a) LLM vs 受験者: 分野別比較\n({best_model})',
                  fontsize=12, fontweight='bold')
    ax1.set_xlim(30, 100)
    ax1.set_ylim(30, 100)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Count above/below diagonal
    above = sum(1 for l, h in zip(llm_accs, human_accs_plot) if l > h)
    below = len(llm_accs) - above
    ax1.text(0.05, 0.95, f'LLM > 受験者: {above}分野\nLLM < 受験者: {below}分野',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Right: Category difficulty ranking (LLM ensemble vs Human) ---
    # Average across top 5 models
    top5 = model_names[:5]
    ensemble_acc = {}
    for cat in cat_filtered:
        accs = [cat_acc[m][cat] for m in top5 if m in cat_acc]
        ensemble_acc[cat] = np.mean(accs)

    # Sort by LLM difficulty (hardest first)
    cat_by_difficulty = sorted(cat_filtered, key=lambda c: ensemble_acc[c])

    y_pos = np.arange(len(cat_by_difficulty))
    llm_vals = [ensemble_acc[c] for c in cat_by_difficulty]
    human_vals = [human_cat_acc[c] for c in cat_by_difficulty]

    ax2.barh(y_pos - 0.2, llm_vals, height=0.35, color='#1976D2', alpha=0.8,
             label='LLM Top5平均', edgecolor='black', linewidth=0.3)
    ax2.barh(y_pos + 0.2, human_vals, height=0.35, color='#FF7043', alpha=0.8,
             label='受験者平均', edgecolor='black', linewidth=0.3)

    ax2.axvline(x=75, color='red', linestyle='--', linewidth=1, alpha=0.5, label='合格ライン')

    cat_labels = [f"{c} ({len(cat_questions[c])}問)" for c in cat_by_difficulty]
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cat_labels, fontsize=8)
    ax2.set_xlabel('正答率 (%)', fontsize=11)
    ax2.set_title('(b) 分野別難易度: LLM vs 受験者\n(LLMはTop5モデル平均)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig('plots/category_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: plots/category_analysis.png")
    plt.close()

    # ====== Figure 3: Model variance by category ======
    fig3, ax3 = plt.subplots(figsize=(14, 7))

    # For each category, compute the range (max - min) across all models
    cat_variance = {}
    for cat in cat_filtered:
        accs = [cat_acc[m][cat] for m in model_names]
        cat_variance[cat] = {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'min': np.min(accs),
            'max': np.max(accs),
            'range': np.max(accs) - np.min(accs),
            'human': human_cat_acc[cat],
        }

    # Sort by variance
    cat_by_var = sorted(cat_filtered, key=lambda c: -cat_variance[c]['range'])

    y_pos = np.arange(len(cat_by_var))
    ranges = [cat_variance[c]['range'] for c in cat_by_var]
    means = [cat_variance[c]['mean'] for c in cat_by_var]
    mins = [cat_variance[c]['min'] for c in cat_by_var]
    maxs = [cat_variance[c]['max'] for c in cat_by_var]
    humans = [cat_variance[c]['human'] for c in cat_by_var]

    # Horizontal error bars
    for i, cat in enumerate(cat_by_var):
        ax3.plot([mins[i], maxs[i]], [i, i], 'b-', linewidth=2, alpha=0.3)
        ax3.scatter(means[i], i, c='#1976D2', s=80, zorder=5, edgecolors='black', linewidths=0.5)
        ax3.scatter(mins[i], i, c='#FF7043', s=40, marker='<', zorder=4)
        ax3.scatter(maxs[i], i, c='#4CAF50', s=40, marker='>', zorder=4)
        ax3.scatter(humans[i], i, c='red', s=40, marker='x', zorder=6, linewidths=1.5)

    cat_labels = [f"{c} ({len(cat_questions[c])}問) [±{cat_variance[c]['range']:.0f}%]" for c in cat_by_var]
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(cat_labels, fontsize=8)
    ax3.set_xlabel('正答率 (%)', fontsize=11)
    ax3.set_title('IgakuQA 2022: 分野別モデル間ばらつき\n'
                  '青丸=平均, 橙◁=最低, 緑▷=最高, 赤×=受験者平均',
                  fontsize=13, fontweight='bold')
    ax3.axvline(x=75, color='red', linestyle='--', linewidth=1, alpha=0.4)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_axisbelow(True)
    ax3.set_xlim(0, 105)
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.savefig('plots/category_variance.png', dpi=150, bbox_inches='tight')
    print("Saved: plots/category_variance.png")
    plt.close()

    # ====== Print summary table ======
    print("\n" + "=" * 80)
    print("分野別正答率サマリー (Top5 LLM平均 vs 受験者平均)")
    print("=" * 80)
    print(f"{'分野':8s} {'問数':>4s} {'LLM平均':>8s} {'受験者':>8s} {'差':>8s} {'モデル間幅':>10s}")
    print("-" * 50)
    for cat in cat_by_difficulty:
        llm_mean = ensemble_acc[cat]
        human = human_cat_acc[cat]
        diff = llm_mean - human
        var_range = cat_variance[cat]['range']
        sign = "+" if diff >= 0 else ""
        print(f"{cat:8s} {len(cat_questions[cat]):4d} {llm_mean:7.1f}% {human:7.1f}% {sign}{diff:6.1f}% {var_range:8.1f}%")

    # LLM が特に得意/苦手な分野
    print("\n--- LLMが受験者を大きく上回る分野 ---")
    for cat in sorted(cat_filtered, key=lambda c: -(ensemble_acc[c] - human_cat_acc[c]))[:5]:
        d = ensemble_acc[cat] - human_cat_acc[cat]
        print(f"  {cat}: LLM {ensemble_acc[cat]:.1f}% vs 受験者 {human_cat_acc[cat]:.1f}% (差 +{d:.1f}%)")

    print("\n--- LLMが受験者を大きく下回る分野 ---")
    for cat in sorted(cat_filtered, key=lambda c: ensemble_acc[c] - human_cat_acc[c])[:5]:
        d = ensemble_acc[cat] - human_cat_acc[cat]
        print(f"  {cat}: LLM {ensemble_acc[cat]:.1f}% vs 受験者 {human_cat_acc[cat]:.1f}% (差 {d:.1f}%)")


if __name__ == "__main__":
    main()
