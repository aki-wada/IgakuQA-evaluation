#!/usr/bin/env python3
"""
IgakuQA Results Analysis Script
評価結果の分析・比較・可視化スクリプト
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_jsonl(filepath: str) -> list:
    """JSONLファイルを読み込む"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze_baseline_results(baseline_dir: str = "baseline_results") -> pd.DataFrame:
    """既存のベースライン結果を分析"""
    baseline_path = Path(baseline_dir)
    results = []

    for year_dir in sorted(baseline_path.iterdir()):
        if not year_dir.is_dir():
            continue

        year = int(year_dir.name)
        year_to_exam = {2018: 112, 2019: 113, 2020: 114, 2021: 115, 2022: 116}
        exam_num = year_to_exam.get(year, 0)

        for result_file in sorted(year_dir.glob("*.jsonl")):
            filename = result_file.stem
            parts = filename.split("_")

            # Parse filename: e.g., "116-A_gpt4" or "116-A_translate_chatgpt-en"
            section = parts[0].split("-")[1]

            if "translate" in filename:
                model = "chatgpt-en"
                language = "en"
            else:
                model = parts[1]
                language = "ja"

            # Load predictions
            preds = load_jsonl(str(result_file))

            # Load gold answers
            data_dir = Path("data") / str(year)
            gold_file = data_dir / f"{exam_num}-{section}.jsonl"

            if not gold_file.exists():
                continue

            golds = load_jsonl(str(gold_file))

            # Calculate accuracy
            correct = 0
            total = len(preds)

            for pred, gold in zip(preds, golds):
                pred_ans = sorted(pred.get("prediction", "").split(","))
                gold_ans = sorted(gold.get("answer", []))

                # Special cases
                if gold["problem_id"] == "116A71":
                    correct += 1
                elif gold["problem_id"] == "112B30" and (pred_ans == ["a"] or pred_ans == ["d"]):
                    correct += 1
                elif pred_ans == gold_ans:
                    correct += 1

            accuracy = correct / total if total > 0 else 0

            results.append({
                "year": year,
                "exam": exam_num,
                "section": section,
                "model": model,
                "language": language,
                "total": total,
                "correct": correct,
                "accuracy": accuracy
            })

    return pd.DataFrame(results)


def analyze_by_category(year: int = 2022, section: str = "A",
                       result_file: str = None, data_dir: str = "data") -> pd.DataFrame:
    """カテゴリ別の正答率を分析"""
    year_to_exam = {2018: 112, 2019: 113, 2020: 114, 2021: 115, 2022: 116}
    exam_num = year_to_exam.get(year)

    # Load metadata
    meta_file = Path(data_dir) / str(year) / f"{exam_num}-{section}_metadata.jsonl"
    metadata = {m["problem_id"]: m for m in load_jsonl(str(meta_file))}

    # Load predictions
    if result_file:
        preds = {p["problem_id"]: p for p in load_jsonl(result_file)}
    else:
        return pd.DataFrame()

    # Load gold answers
    gold_file = Path(data_dir) / str(year) / f"{exam_num}-{section}.jsonl"
    golds = {g["problem_id"]: g for g in load_jsonl(str(gold_file))}

    # Analyze by category
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for problem_id, pred in preds.items():
        if problem_id not in metadata:
            continue

        category = metadata[problem_id].get("category", "その他")
        gold = golds.get(problem_id, {})

        pred_ans = sorted(pred.get("prediction", "").split(","))
        gold_ans = sorted(gold.get("answer", []))

        category_stats[category]["total"] += 1

        if pred_ans == gold_ans:
            category_stats[category]["correct"] += 1

    results = []
    for category, stats in category_stats.items():
        results.append({
            "category": category,
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        })

    return pd.DataFrame(results).sort_values("accuracy", ascending=False)


def plot_model_comparison(df: pd.DataFrame, output_file: str = "model_comparison.png"):
    """モデル比較のプロット"""
    plt.figure(figsize=(12, 6))

    # Aggregate by model
    model_acc = df.groupby("model")["accuracy"].mean().sort_values(ascending=True)

    colors = sns.color_palette("husl", len(model_acc))
    bars = plt.barh(model_acc.index, model_acc.values, color=colors)

    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.title("IgakuQA: Model Comparison (Average Accuracy)")
    plt.xlim(0, 1)

    # Add value labels
    for bar, val in zip(bars, model_acc.values):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_year_trend(df: pd.DataFrame, output_file: str = "year_trend.png"):
    """年度別トレンドのプロット"""
    plt.figure(figsize=(10, 6))

    # Filter main models
    main_models = ["gpt3", "chatgpt", "gpt4", "student-majority"]
    df_filtered = df[df["model"].isin(main_models)]

    # Pivot for plotting
    pivot = df_filtered.groupby(["year", "model"])["accuracy"].mean().unstack()

    pivot.plot(kind="line", marker="o", ax=plt.gca())

    plt.xlabel("Year")
    plt.ylabel("Accuracy")
    plt.title("IgakuQA: Accuracy by Year and Model")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_section_comparison(df: pd.DataFrame, output_file: str = "section_comparison.png"):
    """セクション別比較のプロット"""
    plt.figure(figsize=(12, 6))

    # Filter for GPT-4 to see section difficulty
    df_gpt4 = df[df["model"] == "gpt4"]

    if df_gpt4.empty:
        print("No GPT-4 results found")
        return

    pivot = df_gpt4.pivot_table(values="accuracy", index="section", columns="year")

    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", vmin=0.5, vmax=1.0)

    plt.xlabel("Year")
    plt.ylabel("Section")
    plt.title("IgakuQA: GPT-4 Accuracy by Section and Year")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def generate_report(df: pd.DataFrame) -> str:
    """分析レポートを生成"""
    report = []
    report.append("=" * 60)
    report.append("IgakuQA Baseline Results Analysis")
    report.append("=" * 60)
    report.append("")

    # Overall statistics
    report.append("## Overall Statistics")
    report.append(f"Total evaluations: {len(df)}")
    report.append(f"Years: {sorted(df['year'].unique())}")
    report.append(f"Models: {sorted(df['model'].unique())}")
    report.append("")

    # Model comparison
    report.append("## Model Performance (Average Accuracy)")
    model_acc = df.groupby("model")["accuracy"].agg(["mean", "std", "count"])
    model_acc = model_acc.sort_values("mean", ascending=False)
    for model, row in model_acc.iterrows():
        report.append(f"  {model:20s}: {row['mean']:.1%} (±{row['std']:.1%}, n={int(row['count'])})")
    report.append("")

    # Year comparison
    report.append("## Performance by Year")
    year_acc = df.groupby("year")["accuracy"].mean()
    for year, acc in year_acc.items():
        report.append(f"  {year}: {acc:.1%}")
    report.append("")

    # GPT-4 passing status
    report.append("## GPT-4 Passing Status (>60% threshold)")
    gpt4_df = df[df["model"] == "gpt4"]
    if not gpt4_df.empty:
        for year in sorted(gpt4_df["year"].unique()):
            year_data = gpt4_df[gpt4_df["year"] == year]
            avg_acc = year_data["accuracy"].mean()
            status = "PASS ✓" if avg_acc >= 0.6 else "FAIL ✗"
            report.append(f"  {year}: {avg_acc:.1%} - {status}")
    report.append("")

    report.append("=" * 60)
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze IgakuQA evaluation results")
    parser.add_argument("--baseline-dir", type=str, default="baseline_results",
                        help="Directory containing baseline results")
    parser.add_argument("--output-dir", type=str, default="analysis",
                        help="Output directory for plots and reports")
    parser.add_argument("--result-file", type=str, default=None,
                        help="Specific result file for category analysis")
    parser.add_argument("--year", type=int, default=2022,
                        help="Year for category analysis")
    parser.add_argument("--section", type=str, default="A",
                        help="Section for category analysis")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Analyze baseline results
    print("Analyzing baseline results...")
    df = analyze_baseline_results(args.baseline_dir)

    if df.empty:
        print("No baseline results found")
        return

    # Generate report
    report = generate_report(df)
    print(report)

    # Save report
    report_file = output_dir / "baseline_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {report_file}")

    # Generate plots
    print("\nGenerating plots...")
    plot_model_comparison(df, str(output_dir / "model_comparison.png"))
    plot_year_trend(df, str(output_dir / "year_trend.png"))
    plot_section_comparison(df, str(output_dir / "section_comparison.png"))

    # Category analysis if result file provided
    if args.result_file:
        print(f"\nAnalyzing by category: {args.result_file}")
        cat_df = analyze_by_category(args.year, args.section, args.result_file)
        if not cat_df.empty:
            print("\nCategory Analysis:")
            print(cat_df.to_string(index=False))

    # Save DataFrame
    df.to_csv(output_dir / "baseline_summary.csv", index=False)
    print(f"\nSummary saved to: {output_dir / 'baseline_summary.csv'}")


if __name__ == "__main__":
    main()
