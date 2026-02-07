#!/usr/bin/env python3
"""GLM-4.7-flash の設定調整テスト"""

import requests
import json

API_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "zai-org/glm-4.7-flash"

# テスト問題（116A1）
TEST_QUESTION = """問題: 医師の職業倫理に関する記述で正しいのはどれか。
a 患者の自己決定権は、医師の裁量権に優先する。
b 医療行為に対するインフォームド・コンセントは、口頭で行えばよい。
c 患者の個人情報は、いかなる場合も第三者に開示してはならない。
d 医師は、患者の要求があれば、治療の適応がなくても治療を行う。
e セカンドオピニオンを求めることは、患者の権利である。

正解を1つ選んでください。"""

CORRECT_ANSWER = "e"

def test_config(name, messages, temperature=0, max_tokens=50, extra_params=None):
    """設定をテストする"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    if extra_params:
        payload.update(extra_params)

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        result = response.json()
        if "choices" in result:
            content = result["choices"][0]["message"]["content"]
            print(f"\n=== {name} ===")
            print(f"Settings: temp={temperature}, max_tokens={max_tokens}")
            print(f"Response: {content[:200]}")
            return content
        else:
            print(f"\n=== {name} ===")
            print(f"Error: {result}")
            return None
    except Exception as e:
        print(f"\n=== {name} ===")
        print(f"Exception: {e}")
        return None

print("GLM-4.7-flash 設定テスト")
print("=" * 60)

# テスト1: 通常設定（System prompt あり）
test_config(
    "1. 通常設定 (system prompt あり)",
    [
        {"role": "system", "content": "あなたは医学の専門家です。選択肢から正解をa,b,c,d,eで答えてください。"},
        {"role": "user", "content": TEST_QUESTION}
    ]
)

# テスト2: System prompt なし
test_config(
    "2. System prompt なし",
    [
        {"role": "user", "content": "あなたは医学の専門家です。選択肢から正解をa,b,c,d,eで答えてください。\n\n" + TEST_QUESTION}
    ]
)

# テスト3: 非常に短い max_tokens
test_config(
    "3. max_tokens=10",
    [
        {"role": "system", "content": "Answer with only one letter (a,b,c,d,e)."},
        {"role": "user", "content": TEST_QUESTION}
    ],
    max_tokens=10
)

# テスト4: 英語プロンプト
test_config(
    "4. 英語プロンプト",
    [
        {"role": "system", "content": "You are a medical expert. Choose the correct answer from a,b,c,d,e. Reply with only one letter."},
        {"role": "user", "content": TEST_QUESTION}
    ],
    max_tokens=10
)

# テスト5: 中国語プロンプト
test_config(
    "5. 中国語プロンプト",
    [
        {"role": "system", "content": "你是医学专家。请从选项中选择正确答案，只回答一个字母(a,b,c,d,e)。"},
        {"role": "user", "content": TEST_QUESTION}
    ],
    max_tokens=10
)

# テスト6: 超シンプル
test_config(
    "6. 超シンプル（回答のみ要求）",
    [
        {"role": "user", "content": TEST_QUESTION + "\n\n答え:"}
    ],
    max_tokens=5
)

# テスト7: temperature=0.1
test_config(
    "7. temperature=0.1",
    [
        {"role": "system", "content": "Answer with a single letter only."},
        {"role": "user", "content": TEST_QUESTION}
    ],
    temperature=0.1,
    max_tokens=10
)

# テスト8: stop tokens
test_config(
    "8. stop tokens追加",
    [
        {"role": "user", "content": "正解の選択肢を1つだけ答えてください。\n\n" + TEST_QUESTION}
    ],
    max_tokens=20,
    extra_params={"stop": ["\n", "。", "、"]}
)

# テスト9: 思考禁止の明示的指示
test_config(
    "9. 思考禁止指示",
    [
        {"role": "system", "content": "Do not analyze. Do not explain. Just output a single letter."},
        {"role": "user", "content": "Which is correct? Output ONLY one letter (a/b/c/d/e).\n\n" + TEST_QUESTION}
    ],
    max_tokens=5
)

# テスト10: アシスタントのプレフィル
test_config(
    "10. アシスタントプレフィル",
    [
        {"role": "user", "content": TEST_QUESTION},
        {"role": "assistant", "content": "答え: "}
    ],
    max_tokens=5
)

# テスト11: より長いmax_tokensで出力を確認
test_config(
    "11. max_tokens=200で全出力確認",
    [
        {"role": "system", "content": "医学の多肢選択問題に答えてください。"},
        {"role": "user", "content": TEST_QUESTION}
    ],
    max_tokens=200
)

print("\n" + "=" * 60)
print(f"正解: {CORRECT_ANSWER}")
