#!/usr/bin/env python3
"""
IgakuQA LLM Evaluation Script
医師国家試験問題を使用したLLM性能評価スクリプト

Supports: OpenAI (GPT-4, GPT-4o), Anthropic (Claude), LM Studio (local models)
"""

import json
import argparse
import string
import time
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス"""
    model: str
    year: int
    section: str
    total_questions: int
    correct: int
    total_points: int
    earned_points: int
    accuracy: float
    score_rate: float
    results: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def load_jsonl(filepath: str) -> list:
    """JSONLファイルを読み込む"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, filepath: str):
    """JSONLファイルに保存"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def format_question(question: dict, prompt_examples: list = None) -> str:
    """問題をプロンプト形式に整形"""
    messages = []

    # Few-shot examples
    if prompt_examples:
        for ex in prompt_examples:
            q_text = format_single_question(ex)
            answer = ",".join(ex['answer'])
            messages.append({"role": "user", "content": q_text})
            messages.append({"role": "assistant", "content": answer})

    # Target question
    q_text = format_single_question(question)
    messages.append({"role": "user", "content": q_text})

    return messages


def format_single_question(question: dict) -> str:
    """単一の問題をテキスト形式に整形"""
    text = f"問題: {question['problem_text']}"

    if question.get('choices'):
        for choice, label in zip(question['choices'], string.ascii_lowercase):
            text += f"\n{label}: {choice}"

        n_answers = len(question.get('answer', ['']))
        text += f"\n必ずa,b,c,d,eの中からちょうど{n_answers}個選んでください。"
        text += "\n答え:"

    return text


def extract_answer(response: str) -> str:
    """LLMの回答から選択肢を抽出"""
    # 全角を半角に変換
    response = response.lower()
    response = response.replace('ａ', 'a').replace('ｂ', 'b').replace('ｃ', 'c')
    response = response.replace('ｄ', 'd').replace('ｅ', 'e')

    # 選択肢のパターンを抽出
    pattern = r'[a-e]'
    matches = re.findall(pattern, response)

    if matches:
        # 重複を除いてソート
        unique_answers = sorted(set(matches))
        return ','.join(unique_answers)

    return response.strip()


def check_answer(prediction: str, gold: dict) -> bool:
    """回答の正誤判定"""
    pred_sorted = sorted(prediction.split(','))
    gold_sorted = sorted(gold['answer'])

    # 特殊ケース (採点除外問題など)
    if gold['problem_id'] == "116A71":
        return True
    if gold['problem_id'] == "112B30" and (pred_sorted == ["a"] or pred_sorted == ["d"]):
        return True

    return pred_sorted == gold_sorted


# ============== LLM Provider Classes ==============

class LLMProvider:
    """LLMプロバイダーの基底クラス"""
    def __init__(self, model: str):
        self.model = model

    def generate(self, messages: list) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API (GPT-4, GPT-4o, etc.)"""
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        super().__init__(model)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai パッケージをインストールしてください: pip install openai")

    def generate(self, messages: list) -> str:
        system_msg = [{"role": "system", "content": "あなたは医師国家試験を解く専門家です。"}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=system_msg + messages,
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic API (Claude)"""
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = None):
        super().__init__(model)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic パッケージをインストールしてください: pip install anthropic")

    def generate(self, messages: list) -> str:
        # Convert to Anthropic format
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            system="あなたは医師国家試験を解く専門家です。",
            messages=anthropic_messages
        )
        return response.content[0].text


class LMStudioProvider(LLMProvider):
    """LM Studio (local models via OpenAI-compatible API)"""
    def __init__(self, model: str = "local-model", base_url: str = "http://localhost:1234/v1"):
        super().__init__(model)
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        except ImportError:
            raise ImportError("openai パッケージをインストールしてください: pip install openai")
        self.base_url = base_url

    def get_available_models(self) -> list:
        """利用可能なモデル一覧を取得"""
        models = self.client.models.list()
        return [m.id for m in models.data]

    def generate(self, messages: list) -> str:
        system_msg = [{"role": "system", "content": "あなたは医師国家試験を解く専門家です。"}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=system_msg + messages,
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content


def get_provider(provider_name: str, model: str, **kwargs) -> LLMProvider:
    """プロバイダーインスタンスを取得"""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "lmstudio": LMStudioProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")

    return providers[provider_name](model=model, **kwargs)


def evaluate_section(
    provider: LLMProvider,
    questions: list,
    prompt_examples: list = None,
    delay: float = 0.5,
    verbose: bool = True
) -> list:
    """1セクション分の問題を評価"""
    results = []

    for i, q in enumerate(questions):
        if verbose:
            print(f"  [{i+1}/{len(questions)}] {q['problem_id']}...", end=" ", flush=True)

        messages = format_question(q, prompt_examples)

        try:
            response = provider.generate(messages)
            prediction = extract_answer(response)
            is_correct = check_answer(prediction, q)

            results.append({
                "problem_id": q["problem_id"],
                "prediction": prediction,
                "gold": ",".join(q["answer"]),
                "correct": is_correct,
                "points": int(q.get("points", 1)),
                "raw_response": response
            })

            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"{status} (pred: {prediction}, gold: {','.join(q['answer'])})")

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "problem_id": q["problem_id"],
                "prediction": "ERROR",
                "gold": ",".join(q["answer"]),
                "correct": False,
                "points": int(q.get("points", 1)),
                "error": str(e)
            })

        time.sleep(delay)

    return results


def calculate_metrics(results: list) -> dict:
    """評価メトリクスを計算"""
    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    earned_points = sum(r["points"] for r in results if r["correct"])
    total_points = sum(r["points"] for r in results)

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "total_points": total_points,
        "earned_points": earned_points,
        "score_rate": earned_points / total_points if total_points > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="IgakuQA LLM Evaluation")
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "lmstudio"],
                        help="LLM provider")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model name (e.g., gpt-4o, claude-sonnet-4-20250514, local-model)")
    parser.add_argument("--year", type=int, default=2022,
                        help="Exam year (2018-2022)")
    parser.add_argument("--section", type=str, default="A",
                        help="Exam section (A-F)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls (seconds)")
    parser.add_argument("--use-few-shot", action="store_true",
                        help="Use few-shot examples from prompts/prompt.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of questions (for testing)")
    parser.add_argument("--lmstudio-url", type=str, default="http://localhost:1234/v1",
                        help="LM Studio API URL")

    args = parser.parse_args()

    # Map year to exam number
    year_to_exam = {
        2018: 112,
        2019: 113,
        2020: 114,
        2021: 115,
        2022: 116
    }
    exam_num = year_to_exam.get(args.year)
    if not exam_num:
        raise ValueError(f"Invalid year: {args.year}")

    # Load data
    data_file = Path(args.data_dir) / str(args.year) / f"{exam_num}-{args.section}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"\n{'='*60}")
    print(f"IgakuQA Evaluation")
    print(f"{'='*60}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Year: {args.year} (第{exam_num}回)")
    print(f"Section: {args.section}")
    print(f"{'='*60}\n")

    questions = load_jsonl(str(data_file))
    if args.limit:
        questions = questions[:args.limit]

    print(f"Loaded {len(questions)} questions from {data_file}")

    # Load few-shot examples if requested
    prompt_examples = None
    if args.use_few_shot:
        prompt_file = Path("scripts/prompts/prompt.jsonl")
        if prompt_file.exists():
            prompt_examples = load_jsonl(str(prompt_file))
            print(f"Loaded {len(prompt_examples)} few-shot examples")

    # Initialize provider
    provider_kwargs = {}
    if args.provider == "lmstudio":
        provider_kwargs["base_url"] = args.lmstudio_url

    provider = get_provider(args.provider, args.model, **provider_kwargs)

    # Run evaluation
    print(f"\nStarting evaluation...")
    results = evaluate_section(provider, questions, prompt_examples, args.delay)

    # Calculate metrics
    metrics = calculate_metrics(results)

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Total Questions: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    print(f"Points: {metrics['earned_points']}/{metrics['total_points']} ({metrics['score_rate']:.1%})")
    print(f"{'='*60}\n")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{exam_num}-{args.section}_{args.provider}_{args.model.replace('/', '_')}.jsonl"
    save_jsonl(results, str(output_file))
    print(f"Results saved to: {output_file}")

    # Save summary
    summary = {
        "model": args.model,
        "provider": args.provider,
        "year": args.year,
        "section": args.section,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    summary_file = output_dir / f"summary_{exam_num}-{args.section}_{args.provider}_{args.model.replace('/', '_')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
