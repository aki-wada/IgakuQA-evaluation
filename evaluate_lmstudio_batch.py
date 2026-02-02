#!/usr/bin/env python3
"""
IgakuQA LM Studio Batch Evaluation Script
ローカルLLMの一括評価とベンチマーク表作成
"""

import json
import argparse
import string
import time
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import requests


@dataclass
class ModelResult:
    """モデル評価結果"""
    model: str
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_time: float = 0.0
    errors: int = 0


def get_available_models(base_url: str = "http://localhost:1234/v1") -> list:
    """LM Studioで利用可能なモデル一覧を取得"""
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        data = response.json()
        return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def load_jsonl(filepath: str) -> list:
    """JSONLファイルを読み込む"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_question(question: dict, few_shot: list = None) -> list:
    """問題をメッセージ形式に整形"""
    messages = [{"role": "system", "content": "あなたは医師国家試験を解く専門家です。選択肢から正解をa,b,c,d,eで答えてください。"}]

    # Few-shot examples
    if few_shot:
        for ex in few_shot[:2]:  # 2例のみ使用（コンテキスト節約）
            q_text = create_question_text(ex)
            messages.append({"role": "user", "content": q_text})
            messages.append({"role": "assistant", "content": ",".join(ex['answer'])})

    # Target question
    q_text = create_question_text(question)
    messages.append({"role": "user", "content": q_text})

    return messages


def create_question_text(question: dict) -> str:
    """問題テキストを作成"""
    text = f"問題: {question['problem_text']}"

    if question.get('choices'):
        for choice, label in zip(question['choices'], string.ascii_lowercase):
            text += f"\n{label}: {choice}"

        n_answers = len(question.get('answer', ['']))
        text += f"\n\n{n_answers}個選んで答えてください。\n答え:"

    return text


def extract_answer(response: str) -> str:
    """回答から選択肢を抽出"""
    response = response.lower()
    # 全角→半角
    response = response.replace('ａ', 'a').replace('ｂ', 'b').replace('ｃ', 'c')
    response = response.replace('ｄ', 'd').replace('ｅ', 'e')

    # 選択肢パターン
    matches = re.findall(r'[a-e]', response[:50])  # 最初の50文字のみ

    if matches:
        unique = sorted(set(matches))
        return ','.join(unique)

    return response.strip()[:10]


def call_lmstudio(messages: list, model: str, base_url: str = "http://localhost:1234/v1", timeout: int = 60) -> str:
    """LM Studio APIを呼び出し"""
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 50,
                "stream": False
            },
            timeout=timeout
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        raise TimeoutError("API timeout")
    except Exception as e:
        raise RuntimeError(f"API error: {e}")


def evaluate_model(
    model: str,
    questions: list,
    few_shot: list = None,
    base_url: str = "http://localhost:1234/v1",
    limit: int = None,
    verbose: bool = True
) -> ModelResult:
    """1モデルを評価"""

    if limit:
        questions = questions[:limit]

    result = ModelResult(model=model, total=len(questions))
    times = []

    for i, q in enumerate(questions):
        if verbose:
            print(f"  [{i+1}/{len(questions)}] {q['problem_id']}", end=" ", flush=True)

        messages = format_question(q, few_shot)

        try:
            start = time.time()
            response = call_lmstudio(messages, model, base_url)
            elapsed = time.time() - start
            times.append(elapsed)

            prediction = extract_answer(response)
            gold = sorted(q['answer'])
            pred = sorted(prediction.split(','))

            is_correct = pred == gold
            if is_correct:
                result.correct += 1

            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"{status} ({prediction} vs {','.join(gold)}) [{elapsed:.1f}s]")

        except Exception as e:
            result.errors += 1
            if verbose:
                print(f"ERROR: {e}")

        # Rate limiting
        time.sleep(0.1)

    result.accuracy = result.correct / result.total if result.total > 0 else 0
    result.avg_time = sum(times) / len(times) if times else 0

    return result


def print_results_table(results: list[ModelResult], title: str = ""):
    """結果をMarkdown表形式で出力"""
    print("\n" + "=" * 70)
    if title:
        print(f"## {title}")
    print("=" * 70)

    # Sort by accuracy
    results = sorted(results, key=lambda x: x.accuracy, reverse=True)

    # Header
    print(f"\n| {'Rank':<4} | {'Model':<40} | {'Accuracy':<10} | {'Correct':<10} | {'Avg Time':<10} |")
    print(f"|{'-'*6}|{'-'*42}|{'-'*12}|{'-'*12}|{'-'*12}|")

    # Rows
    for i, r in enumerate(results, 1):
        acc_str = f"{r.accuracy:.1%}"
        correct_str = f"{r.correct}/{r.total}"
        time_str = f"{r.avg_time:.1f}s"
        error_note = f" ({r.errors}err)" if r.errors > 0 else ""

        model_name = r.model[:38] if len(r.model) > 38 else r.model

        print(f"| {i:<4} | {model_name:<40} | {acc_str:<10} | {correct_str:<10} | {time_str:<10} |{error_note}")

    print("\n" + "=" * 70)


def save_results(results: list[ModelResult], output_file: str):
    """結果をJSONファイルに保存"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "model": r.model,
                "total": r.total,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "avg_time": r.avg_time,
                "errors": r.errors
            }
            for r in results
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="IgakuQA LM Studio Batch Evaluation")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models to evaluate (default: all available)")
    parser.add_argument("--year", type=int, default=2022,
                        help="Exam year (2018-2022)")
    parser.add_argument("--section", type=str, default="A",
                        help="Exam section (A-F)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions per model")
    parser.add_argument("--url", type=str, default="http://localhost:1234/v1",
                        help="LM Studio API URL")
    parser.add_argument("--output", type=str, default="lmstudio_results.json",
                        help="Output JSON file")
    parser.add_argument("--use-few-shot", action="store_true",
                        help="Use few-shot examples")
    parser.add_argument("--exclude", type=str, nargs="+", default=[],
                        help="Models to exclude")

    args = parser.parse_args()

    # Map year to exam number
    year_to_exam = {2018: 112, 2019: 113, 2020: 114, 2021: 115, 2022: 116}
    exam_num = year_to_exam.get(args.year)
    if not exam_num:
        print(f"Invalid year: {args.year}")
        return

    # Load questions
    data_file = Path("data") / str(args.year) / f"{exam_num}-{args.section}.jsonl"
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return

    questions = load_jsonl(str(data_file))
    print(f"Loaded {len(questions)} questions from {data_file}")

    # Load few-shot examples
    few_shot = None
    if args.use_few_shot:
        prompt_file = Path("scripts/prompts/prompt.jsonl")
        if prompt_file.exists():
            few_shot = load_jsonl(str(prompt_file))
            print(f"Loaded {len(few_shot)} few-shot examples")

    # Get models
    if args.models:
        models = args.models
    else:
        models = get_available_models(args.url)
        print(f"Found {len(models)} models in LM Studio")

    # Apply exclusions
    if args.exclude:
        models = [m for m in models if m not in args.exclude]

    if not models:
        print("No models available")
        return

    print(f"\nModels to evaluate: {len(models)}")
    for m in models:
        print(f"  - {m}")

    # Evaluate each model
    results = []
    for i, model in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(models)}] Evaluating: {model}")
        print(f"{'='*60}")

        try:
            result = evaluate_model(
                model=model,
                questions=questions,
                few_shot=few_shot,
                base_url=args.url,
                limit=args.limit,
                verbose=True
            )
            results.append(result)

            print(f"\n  Result: {result.correct}/{result.total} ({result.accuracy:.1%})")

        except Exception as e:
            print(f"  Failed: {e}")
            results.append(ModelResult(model=model, errors=1))

    # Print summary table
    print_results_table(results, f"IgakuQA {args.year} Section {args.section}")

    # Save results
    save_results(results, args.output)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
