#!/usr/bin/env python3
"""
選択数（1つ選べ vs 2つ以上選べ）× 問題タイプ（一般 vs 臨床）の正答率分析
日本語版と英語版を同時生成
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.unicode_minus'] = False

DATA_DIR = "/Users/macstudio/Desktop/IgakuQA/data/2022"
RESULTS_DIR = "/Users/macstudio/Desktop/IgakuQA/results"

# Models to analyze (name, params_B, file_prefix, prompt_key)
MODELS = [
    ("Q3.5-397B@8bit", 397, "prompt_comparison_qwen3.5-397b-a17b-8bit", "baseline"),
    ("Q3-235B-2507", 235, "prompt_comparison_qwen_qwen3-235b-a22b-2507", "baseline"),
    ("Q3.5-27B@8bit", 27, "prompt_comparison_qwen3.5-27b@8bit", "baseline"),
    ("Q3-Next-80B", 80, "prompt_comparison_qwen_qwen3-next-80b", None),
    ("Q3-VL-32B", 32, "prompt_comparison_qwen_qwen3-vl-32b", "baseline"),
    ("Q3-32B@4bit", 32, "qwen3-32b-4bit_fewshot", "baseline"),
    ("Q3-30B-A3B", 30, "prompt_comparison_qwen_qwen3-30b-a3b-2507", "baseline"),
    ("Q3-14B", 14, "prompt_comparison_qwen_qwen3-14b", None),
    ("Q3-VL-8B@8bit", 8, "qwen3-vl-8b-8bit_fewshot", "baseline"),
    ("Q3-VL-4B@4bit", 4, "qwen3-vl-4b-4bit_fewshot", "baseline"),
]


def load_question_map():
    """Load question metadata: problem_id -> {sec, num_select, type}"""
    q_map = {}
    for f in sorted(glob.glob(f"{DATA_DIR}/116-[A-F].jsonl")):
        sec = os.path.basename(f).split(".")[0].replace("116-", "")
        with open(f) as fh:
            for line in fh:
                q = json.loads(line)
                pid = q["problem_id"]
                ans = q["answer"]
                num_select = len(ans) if isinstance(ans, list) else 1
                q_type = "general" if sec in ["A", "B"] else "clinical"
                q_map[pid] = {"sec": sec, "num_select": num_select, "type": q_type}
    return q_map


def load_model_results(prefix, pkey, q_map):
    """Load results for a model, categorized by (type, multi_select)"""
    buckets = {("general", 1): [], ("general", 2): [],
               ("clinical", 1): [], ("clinical", 2): []}

    for sec in ["A", "B", "C", "D", "E", "F"]:
        fpath = os.path.join(RESULTS_DIR, f"{prefix}_2022_{sec}.json")
        if not os.path.exists(fpath):
            continue
        d = json.load(open(fpath))
        det = d.get("details", {})
        if isinstance(det, dict):
            items = det.get(pkey, []) if pkey else []
            if not items:
                first_key = list(det.keys())[0] if det else None
                items = det.get(first_key, []) if first_key else []
        else:
            items = det

        for item in items:
            pid = item.get("problem_id", "")
            if pid in q_map:
                info = q_map[pid]
                n = min(info["num_select"], 2)  # 1 or 2+
                correct = item.get("correct", item.get("is_correct", False))
                buckets[(info["type"], n)].append(1 if correct else 0)
    return buckets


def create_plot(lang="jp"):
    is_jp = lang == "jp"
    if is_jp:
        matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Arial Unicode MS', 'sans-serif']
    else:
        matplotlib.rcParams['font.family'] = ['Arial', 'Helvetica', 'sans-serif']

    q_map = load_question_map()

    # Collect data for all models
    model_data = []
    for name, params, prefix, pkey in MODELS:
        buckets = load_model_results(prefix, pkey, q_map)
        accs = {}
        for key, vals in buckets.items():
            accs[key] = 100 * sum(vals) / len(vals) if vals else 0
        model_data.append({
            "name": name, "params": params,
            "g1": accs[("general", 1)], "g2": accs[("general", 2)],
            "c1": accs[("clinical", 1)], "c2": accs[("clinical", 2)],
        })

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # ====== (a) Grouped bar: 4 conditions per model ======
    ax1 = axes[0]

    n_models = len(model_data)
    x = np.arange(n_models)
    width = 0.2

    colors = {
        "g1": "#1565C0",  # general single - dark blue
        "g2": "#64B5F6",  # general multi - light blue
        "c1": "#C62828",  # clinical single - dark red
        "c2": "#EF9A9A",  # clinical multi - light red
    }
    if is_jp:
        labels = {"g1": "一般・1つ選べ", "g2": "一般・2+選べ",
                  "c1": "臨床・1つ選べ", "c2": "臨床・2+選べ"}
    else:
        labels = {"g1": "General·Single", "g2": "General·Multi",
                  "c1": "Clinical·Single", "c2": "Clinical·Multi"}

    for i, key in enumerate(["g1", "g2", "c1", "c2"]):
        vals = [d[key] for d in model_data]
        bars = ax1.bar(x + (i - 1.5) * width, vals, width,
                       color=colors[key], edgecolor='black', linewidth=0.3,
                       label=labels[key], alpha=0.85)

    ax1.axhline(y=75, color='red', linestyle=':', linewidth=1, alpha=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([d["name"] for d in model_data], rotation=35, ha='right', fontsize=8)
    ax1.set_ylim(30, 100)
    if is_jp:
        ax1.set_ylabel('正答率 (%)', fontsize=11)
        ax1.set_title('(a) 選択数 × 問題タイプ別 正答率\n4条件の比較', fontsize=12, fontweight='bold')
    else:
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('(a) Accuracy by Selection Count × Question Type\n4-way comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7.5, loc='lower left', ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)

    # ====== (b) Multi-select penalty by model size ======
    ax2 = axes[1]

    params_arr = np.array([d["params"] for d in model_data])
    penalty_gen = np.array([d["g2"] - d["g1"] for d in model_data])
    penalty_clin = np.array([d["c2"] - d["c1"] for d in model_data])

    ax2.scatter(params_arr, penalty_gen, c='#1565C0', marker='o', s=120,
                edgecolors='black', linewidths=0.5, zorder=5, label='一般' if is_jp else 'General')
    ax2.scatter(params_arr, penalty_clin, c='#C62828', marker='D', s=120,
                edgecolors='black', linewidths=0.5, zorder=5, label='臨床' if is_jp else 'Clinical')

    # Connect same model
    for i in range(len(model_data)):
        ax2.plot([params_arr[i], params_arr[i]], [penalty_gen[i], penalty_clin[i]],
                 color='gray', alpha=0.3, linewidth=1, zorder=3)

    ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.4)

    # Annotate
    for i, d in enumerate(model_data):
        name = d["name"].replace("Q3.5-397B@8bit", "397B").replace("Q3-235B-2507", "235B")
        name = name.replace("Q3.5-27B@8bit", "27B").replace("Q3-Next-80B", "80B")
        name = name.replace("Q3-VL-32B", "VL32B").replace("Q3-32B@4bit", "32B")
        name = name.replace("Q3-30B-A3B", "30B").replace("Q3-14B", "14B")
        name = name.replace("Q3-VL-8B@8bit", "VL8B").replace("Q3-VL-4B@4bit", "VL4B")
        y_mid = (penalty_gen[i] + penalty_clin[i]) / 2
        ax2.annotate(name, (params_arr[i], y_mid), fontsize=7,
                     xytext=(8, 0), textcoords='offset points', alpha=0.7)

    # Shade penalty zone
    ax2.axhspan(-35, 0, alpha=0.04, color='red')

    ax2.set_xscale('log')
    ax2.set_xticks([4, 8, 14, 30, 80, 235, 400])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if is_jp:
        ax2.set_xlabel('モデルサイズ (B)', fontsize=11)
        ax2.set_ylabel('2+選べペナルティ (2+選べ - 1つ選べ, %)', fontsize=11)
        ax2.set_title('(b) 複数選択ペナルティ vs モデルサイズ\n0以下 = 複数選択で正答率低下', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('Model Size (B)', fontsize=11)
        ax2.set_ylabel('Multi-select Penalty (multi - single, %)', fontsize=11)
        ax2.set_title('(b) Multi-select Penalty vs. Model Size\nBelow 0 = accuracy drops on multi-select', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    # ====== (c) Confound decomposition: what explains clinical gap? ======
    ax3 = axes[2]

    # For each model: decompose clinical gap into
    # 1) pure difficulty gap (1-select only: c1 - g1)
    # 2) multi-select composition effect
    pure_gap = np.array([d["c1"] - d["g1"] for d in model_data])
    total_gap_approx = []
    for d in model_data:
        # Weighted average: general = 0.90*g1 + 0.10*g2, clinical = 0.83*c1 + 0.17*c2
        gen_avg = 0.896 * d["g1"] + 0.104 * d["g2"]
        clin_avg = 0.825 * d["c1"] + 0.175 * d["c2"]
        total_gap_approx.append(clin_avg - gen_avg)
    total_gap_approx = np.array(total_gap_approx)
    composition_effect = total_gap_approx - pure_gap

    x3 = np.arange(n_models)
    width3 = 0.35

    if is_jp:
        ax3.bar(x3 - width3 / 2, pure_gap, width3, color='#C62828', alpha=0.8,
                edgecolor='black', linewidth=0.3, label='純粋な臨床難易度\n(1つ選べのみで比較)')
        ax3.bar(x3 + width3 / 2, composition_effect, width3, color='#FF8A65', alpha=0.8,
                edgecolor='black', linewidth=0.3, label='選択数の構成差\n(臨床は2+選べが多い)')
    else:
        ax3.bar(x3 - width3 / 2, pure_gap, width3, color='#C62828', alpha=0.8,
                edgecolor='black', linewidth=0.3, label='Pure clinical difficulty\n(single-select only)')
        ax3.bar(x3 + width3 / 2, composition_effect, width3, color='#FF8A65', alpha=0.8,
                edgecolor='black', linewidth=0.3, label='Selection count composition\n(more multi-select in clinical)')

    # Total gap line
    ax3.plot(x3, total_gap_approx, 'ko-', markersize=5, linewidth=1.5, alpha=0.6,
             label='Total Gap' if not is_jp else '合計Gap')

    ax3.axhline(y=0, color='black', linewidth=0.8, alpha=0.4)

    # Value labels
    for i in range(n_models):
        if abs(total_gap_approx[i]) > 1:
            ax3.text(i, total_gap_approx[i] - 0.8, f'{total_gap_approx[i]:+.1f}%',
                     ha='center', fontsize=7, fontweight='bold', alpha=0.7)

    ax3.set_xticks(x3)
    ax3.set_xticklabels([d["name"] for d in model_data], rotation=35, ha='right', fontsize=8)
    if is_jp:
        ax3.set_ylabel('正答率差 (%)', fontsize=11)
        ax3.set_title('(c) 臨床Gapの要因分解\n赤=臨床推論の難しさ / 橙=選択数の交絡',
                      fontsize=12, fontweight='bold')
    else:
        ax3.set_ylabel('Accuracy Gap (%)', fontsize=11)
        ax3.set_title('(c) Clinical Gap Decomposition\nRed=Clinical difficulty / Orange=Selection confound',
                      fontsize=12, fontweight='bold')
    ax3.legend(fontsize=7.5, loc='lower left')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_axisbelow(True)

    if is_jp:
        plt.suptitle('IgakuQA 2022: 選択数（1つ vs 2+）が正答率に与える影響の分析',
                     fontsize=14, fontweight='bold')
    else:
        plt.suptitle('IgakuQA 2022: Impact of Selection Count (Single vs. Multi) on Accuracy',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    suffix = "" if is_jp else "_en"
    outpath = f"plots/selection_count{suffix}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    create_plot("jp")
    create_plot("en")
    print("Done!")
