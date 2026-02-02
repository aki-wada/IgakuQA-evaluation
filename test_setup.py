#!/usr/bin/env python3
"""
IgakuQA Setup Test Script
環境設定のテスト用スクリプト（3問のみで動作確認）
"""

import sys
import os
from pathlib import Path

def test_data_files():
    """データファイルの存在確認"""
    print("1. データファイルの確認...")

    required_files = [
        "data/2022/116-A.jsonl",
        "data/2022/116-A_metadata.jsonl",
        "scripts/prompts/prompt.jsonl"
    ]

    all_exist = True
    for f in required_files:
        exists = Path(f).exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {f}")
        if not exists:
            all_exist = False

    return all_exist


def test_dependencies():
    """依存パッケージの確認"""
    print("\n2. 依存パッケージの確認...")

    packages = {
        "openai": "OpenAI API",
        "anthropic": "Anthropic API",
        "pandas": "データ分析",
        "matplotlib": "可視化"
    }

    results = {}
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print(f"   ✓ {pkg} ({desc})")
            results[pkg] = True
        except ImportError:
            print(f"   ✗ {pkg} ({desc}) - pip install {pkg}")
            results[pkg] = False

    return results


def test_api_keys():
    """APIキーの確認"""
    print("\n3. APIキーの確認...")

    keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic"
    }

    results = {}
    for key, name in keys.items():
        value = os.environ.get(key)
        if value:
            masked = value[:8] + "..." + value[-4:]
            print(f"   ✓ {name}: {masked}")
            results[key] = True
        else:
            print(f"   - {name}: 未設定 (export {key}=...)")
            results[key] = False

    return results


def test_evaluation(provider: str = "openai", model: str = "gpt-4o"):
    """評価スクリプトの動作テスト（3問のみ）"""
    print(f"\n4. 評価テスト ({provider}/{model})...")

    from evaluate_llm import (
        load_jsonl, get_provider, format_question,
        extract_answer, check_answer
    )

    # Load 3 questions
    questions = load_jsonl("data/2022/116-A.jsonl")[:3]
    prompts = load_jsonl("scripts/prompts/prompt.jsonl")

    print(f"   問題数: {len(questions)}")

    try:
        # Initialize provider
        prov = get_provider(provider, model)
        print(f"   プロバイダー初期化: ✓")

        # Test one question
        q = questions[0]
        messages = format_question(q, prompts)

        print(f"   問題 {q['problem_id']}: {q['problem_text'][:30]}...")

        response = prov.generate(messages)
        prediction = extract_answer(response)
        is_correct = check_answer(prediction, q)

        status = "✓" if is_correct else "✗"
        print(f"   回答: {prediction} (正解: {','.join(q['answer'])}) {status}")

        return True

    except Exception as e:
        print(f"   エラー: {e}")
        return False


def main():
    print("=" * 50)
    print("IgakuQA 環境テスト")
    print("=" * 50)

    # Test data files
    data_ok = test_data_files()

    # Test dependencies
    deps = test_dependencies()

    # Test API keys
    keys = test_api_keys()

    # Summary
    print("\n" + "=" * 50)
    print("サマリ")
    print("=" * 50)

    if data_ok:
        print("✓ データファイル: OK")
    else:
        print("✗ データファイル: 不足あり")

    if all(deps.values()):
        print("✓ 依存パッケージ: OK")
    else:
        missing = [k for k, v in deps.items() if not v]
        print(f"✗ 依存パッケージ: {', '.join(missing)} が必要")

    # Determine available provider
    available_provider = None
    if keys.get("OPENAI_API_KEY") and deps.get("openai"):
        available_provider = ("openai", "gpt-4o")
    elif keys.get("ANTHROPIC_API_KEY") and deps.get("anthropic"):
        available_provider = ("anthropic", "claude-sonnet-4-20250514")

    if available_provider:
        print(f"\n使用可能なプロバイダー: {available_provider[0]}")

        response = input("\n評価テストを実行しますか? (y/N): ")
        if response.lower() == "y":
            test_evaluation(*available_provider)
    else:
        print("\n評価テストをスキップ (APIキー未設定)")

    print("\n" + "=" * 50)
    print("テスト完了")
    print("=" * 50)


if __name__ == "__main__":
    main()
