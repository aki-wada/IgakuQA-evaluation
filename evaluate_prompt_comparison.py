#!/usr/bin/env python3
"""
IgakuQA プロンプト比較実験スクリプト
異なるシステムプロンプトの効果を比較評価
"""

import json
import argparse
import string
import time
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import requests


# ============== プロンプト定義 ==============

PROMPTS = {
    "baseline": {
        "name": "Baseline (現行)",
        "system": "あなたは医学の専門家です。選択肢から正解をa,b,c,d,eで答えてください。",
        "max_tokens": 512
    },
    "format_strict": {
        "name": "案A: 回答形式強化",
        "system": """あなたは臨床医学に精通した医師です。
問題を読み、正解の選択肢をアルファベット(a,b,c,d,e)のみで回答してください。
複数選択の場合はカンマ区切りで回答してください。例: a,c
余計な説明は不要です。""",
        "max_tokens": 512
    },
    "chain_of_thought": {
        "name": "案B: 段階的思考",
        "system": """あなたは臨床医学に精通した医師です。

回答手順:
1. 問題文のキーワードを確認
2. 各選択肢を医学的に検討
3. 正解を選択

最終行に「答え:」に続けて選択肢(a,b,c,d,e)のみを記載してください。""",
        "max_tokens": 512
    },
    "japanese_medical": {
        "name": "案C: 日本医療文脈",
        "system": """あなたは日本の医療制度と臨床医学に精通した専門医です。
医学の多肢選択問題に回答します。

重要:
- 日本の診療ガイドラインに基づいて判断
- 選択肢はa,b,c,d,eのアルファベットで回答
- 指定された個数を必ず選択

回答形式: a または a,b,c（複数の場合）""",
        "max_tokens": 512
    }
}


@dataclass
class PromptResult:
    """プロンプト別評価結果"""
    prompt_key: str
    prompt_name: str
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_time: float = 0.0
    errors: int = 0
    empty_responses: int = 0


def load_jsonl(filepath: str) -> list:
    """JSONLファイルを読み込む"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_question_text(question: dict) -> str:
    """問題テキストを作成"""
    text = f"問題: {question['problem_text']}"

    if question.get('choices'):
        for choice, label in zip(question['choices'], string.ascii_lowercase):
            text += f"\n{label}: {choice}"

        n_answers = len(question.get('answer', ['']))
        text += f"\n\n{n_answers}個選んで答えてください。\n答え:"

    return text


def format_messages(question: dict, system_prompt: str, few_shot: list = None, model: str = "") -> list:
    """メッセージ形式に整形"""
    # qwen3 モデルの場合、/no_think を追加して thinking モードを無効化
    if "qwen3" in model.lower() and "/no_think" not in system_prompt:
        system_prompt = system_prompt + " /no_think"

    messages = [{"role": "system", "content": system_prompt}]

    # Few-shot examples
    if few_shot:
        for ex in few_shot[:2]:
            q_text = create_question_text(ex)
            messages.append({"role": "user", "content": q_text})
            messages.append({"role": "assistant", "content": ",".join(ex['answer'])})

    # Target question
    q_text = create_question_text(question)
    messages.append({"role": "user", "content": q_text})

    return messages


def extract_answer(response: str, prompt_key: str = "baseline") -> str:
    """回答から選択肢を抽出"""
    if not response or not response.strip():
        return ""

    # qwen3 の <think>...</think> タグを除去
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # medgemma の <unused94>thought 思考タグを除去
    response = re.sub(r'<unused\d+>thought.*', '', response, flags=re.DOTALL).strip()

    if not response:
        return ""

    response_lower = response.lower()

    # 全角→半角
    response_lower = response_lower.replace('ａ', 'a').replace('ｂ', 'b').replace('ｃ', 'c')
    response_lower = response_lower.replace('ｄ', 'd').replace('ｅ', 'e')

    # 1. 冒頭が単独の選択肢文字（a-e）のみの場合（最も確実）
    head_match = re.match(r'^([a-e](?:\s*[,、]\s*[a-e])*)\s*$', response_lower.split('\n')[0].strip())
    if head_match:
        matches = re.findall(r'[a-e]', head_match.group(1))
        if matches:
            return ','.join(sorted(set(matches)))

    # 2. 「正解は X」「正解: X」「選択肢は X」パターン
    seikai_patterns = [
        r'正解[はは]\s*\*{0,2}\s*([a-e](?:\s*[,、]\s*[a-e])*)',
        r'正解[:：]\s*\*{0,2}\s*([a-e](?:\s*[,、]\s*[a-e])*)',
        r'選択肢は\s*\*{0,2}\s*([a-e](?:\s*[,、]\s*[a-e])*)',
    ]
    for pattern in seikai_patterns:
        match = re.search(pattern, response_lower)
        if match:
            matches = re.findall(r'[a-e]', match.group(1))
            if matches:
                return ','.join(sorted(set(matches)))

    # 3. 「答え:」「回答:」パターン（CoT含む全プロンプト共通）
    answer_patterns = [r'答え[:：]\s*([a-e,\s]+)', r'回答[:：]\s*([a-e,\s]+)']
    for pattern in answer_patterns:
        match = re.search(pattern, response_lower)
        if match:
            found = match.group(1)
            matches = re.findall(r'[a-e]', found)
            if matches:
                return ','.join(sorted(set(matches)))

    # 4. フォールバック: 最初の20文字から抽出（短い範囲で誤検出を防止）
    matches = re.findall(r'[a-e]', response_lower[:20])

    if matches:
        unique = sorted(set(matches))
        return ','.join(unique)

    return response.strip()[:10]


def call_lmstudio(
    messages: list,
    model: str,
    max_tokens: int = 50,
    base_url: str = "http://localhost:1234/v1",
    timeout: int = 120
) -> str:
    """LM Studio APIを呼び出し"""
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
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


def evaluate_with_prompt(
    prompt_key: str,
    prompt_config: dict,
    model: str,
    questions: list,
    few_shot: list = None,
    base_url: str = "http://localhost:1234/v1",
    verbose: bool = True
) -> tuple[PromptResult, list]:
    """特定のプロンプトで評価"""

    result = PromptResult(
        prompt_key=prompt_key,
        prompt_name=prompt_config["name"],
        total=len(questions)
    )
    times = []
    details = []

    system_prompt = prompt_config["system"]
    max_tokens = prompt_config["max_tokens"]

    for i, q in enumerate(questions):
        if verbose:
            print(f"    [{i+1}/{len(questions)}] {q['problem_id']}", end=" ", flush=True)

        messages = format_messages(q, system_prompt, few_shot, model)

        try:
            start = time.time()
            response = call_lmstudio(messages, model, max_tokens, base_url)
            elapsed = time.time() - start
            times.append(elapsed)

            if not response or not response.strip():
                result.empty_responses += 1

            prediction = extract_answer(response, prompt_key)
            gold = sorted(q['answer'])
            pred = sorted(prediction.split(',')) if prediction else []

            is_correct = pred == gold
            if is_correct:
                result.correct += 1

            details.append({
                "problem_id": q["problem_id"],
                "prediction": prediction,
                "gold": ",".join(gold),
                "correct": is_correct,
                "response": response[:200] if response else "",
                "time": elapsed
            })

            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"{status} ({prediction} vs {','.join(gold)}) [{elapsed:.1f}s]")

        except Exception as e:
            result.errors += 1
            details.append({
                "problem_id": q["problem_id"],
                "prediction": "ERROR",
                "gold": ",".join(q['answer']),
                "correct": False,
                "error": str(e)
            })
            if verbose:
                print(f"ERROR: {e}")

        time.sleep(0.1)

    result.accuracy = result.correct / result.total if result.total > 0 else 0
    result.avg_time = sum(times) / len(times) if times else 0

    return result, details


def print_comparison_table(results: list[PromptResult], model: str):
    """比較結果をテーブル表示"""
    print("\n" + "=" * 80)
    print(f"プロンプト比較結果 - Model: {model}")
    print("=" * 80)

    # Sort by accuracy
    results = sorted(results, key=lambda x: x.accuracy, reverse=True)

    print(f"\n| {'Rank':<4} | {'Prompt':<25} | {'Accuracy':<10} | {'Correct':<10} | {'Empty':<6} | {'Avg Time':<10} |")
    print(f"|{'-'*6}|{'-'*27}|{'-'*12}|{'-'*12}|{'-'*8}|{'-'*12}|")

    for i, r in enumerate(results, 1):
        acc_str = f"{r.accuracy:.1%}"
        correct_str = f"{r.correct}/{r.total}"
        time_str = f"{r.avg_time:.1f}s"
        name = r.prompt_name[:23] if len(r.prompt_name) > 23 else r.prompt_name

        print(f"| {i:<4} | {name:<25} | {acc_str:<10} | {correct_str:<10} | {r.empty_responses:<6} | {time_str:<10} |")

    print("\n" + "=" * 80)

    # 改善率を計算
    baseline = next((r for r in results if r.prompt_key == "baseline"), None)
    if baseline:
        print("\n### ベースラインからの改善:")
        for r in results:
            if r.prompt_key != "baseline":
                diff = r.accuracy - baseline.accuracy
                sign = "+" if diff >= 0 else ""
                print(f"  {r.prompt_name}: {sign}{diff:.1%}")


def save_results(results: list[PromptResult], details: dict, model: str, output_file: str):
    """結果をJSONファイルに保存"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "summary": [
            {
                "prompt_key": r.prompt_key,
                "prompt_name": r.prompt_name,
                "total": r.total,
                "correct": r.correct,
                "accuracy": r.accuracy,
                "avg_time": r.avg_time,
                "errors": r.errors,
                "empty_responses": r.empty_responses
            }
            for r in results
        ],
        "details": details,
        "prompts": {k: v["system"] for k, v in PROMPTS.items()}
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="IgakuQA Prompt Comparison")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to evaluate")
    parser.add_argument("--year", type=int, default=2022,
                        help="Exam year (2018-2022)")
    parser.add_argument("--section", type=str, default="A",
                        help="Exam section (A-F)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions (for quick testing)")
    parser.add_argument("--url", type=str, default="http://localhost:1234/v1",
                        help="LM Studio API URL")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    parser.add_argument("--use-few-shot", action="store_true", default=True,
                        help="Use few-shot examples (default: True)")
    parser.add_argument("--no-few-shot", action="store_true",
                        help="Disable few-shot examples")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=list(PROMPTS.keys()),
                        choices=list(PROMPTS.keys()),
                        help="Prompts to test")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max_tokens (e.g. 1024 for reasoning models)")

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
    if args.limit:
        questions = questions[:args.limit]

    print(f"Loaded {len(questions)} questions from {data_file}")

    # Load few-shot examples (default: enabled, use --no-few-shot to disable)
    few_shot = None
    if args.no_few_shot:
        args.use_few_shot = False
    if args.use_few_shot:
        prompt_file = Path("scripts/prompts/prompt.jsonl")
        if prompt_file.exists():
            few_shot = load_jsonl(str(prompt_file))
            print(f"Loaded {len(few_shot)} few-shot examples")

    # Evaluate each prompt
    results = []
    all_details = {}

    for prompt_key in args.prompts:
        prompt_config = PROMPTS[prompt_key].copy()
        if args.max_tokens is not None:
            prompt_config["max_tokens"] = args.max_tokens

        print(f"\n{'='*60}")
        print(f"Testing: {prompt_config['name']}")
        print(f"System: {prompt_config['system'][:60]}...")
        print(f"{'='*60}")

        result, details = evaluate_with_prompt(
            prompt_key=prompt_key,
            prompt_config=prompt_config,
            model=args.model,
            questions=questions,
            few_shot=few_shot,
            base_url=args.url,
            verbose=True
        )

        results.append(result)
        all_details[prompt_key] = details

        print(f"\n  Result: {result.correct}/{result.total} ({result.accuracy:.1%})")

    # Print comparison
    print_comparison_table(results, args.model)

    # Save results
    if args.output is None:
        args.output = f"results/prompt_comparison_{args.model.replace('/', '_')}_{args.year}_{args.section}.json"

    Path(args.output).parent.mkdir(exist_ok=True)
    save_results(results, all_details, args.model, args.output)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
