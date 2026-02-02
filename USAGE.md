# IgakuQA LLM評価ガイド

## クイックスタート

### 1. 環境構築

```bash
cd /Users/wadaakihiko/Desktop/wada_work/IgakuQA

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存パッケージインストール
pip install -r requirements.txt
```

### 2. APIキーの設定

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 評価の実行

### OpenAI (GPT-4o)

```bash
python evaluate_llm.py \
    --provider openai \
    --model gpt-4o \
    --year 2022 \
    --section A \
    --use-few-shot
```

### Anthropic (Claude)

```bash
python evaluate_llm.py \
    --provider anthropic \
    --model claude-sonnet-4-20250514 \
    --year 2022 \
    --section A
```

### LM Studio (ローカルモデル)

```bash
# LM Studioを起動し、モデルをロード後:
python evaluate_llm.py \
    --provider lmstudio \
    --model "your-model-name" \
    --year 2022 \
    --section A \
    --lmstudio-url http://localhost:1234/v1
```

## コマンドオプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `--provider` | LLMプロバイダー (openai/anthropic/lmstudio) | openai |
| `--model` | モデル名 | gpt-4o |
| `--year` | 試験年度 (2018-2022) | 2022 |
| `--section` | セクション (A-F) | A |
| `--use-few-shot` | Few-shotプロンプト使用 | False |
| `--delay` | API呼び出し間隔(秒) | 0.5 |
| `--limit` | テスト用に問題数制限 | None |
| `--output-dir` | 結果出力ディレクトリ | results |

## 結果の分析

```bash
# ベースライン結果の分析
python analyze_results.py

# 特定の結果ファイルをカテゴリ別分析
python analyze_results.py \
    --result-file results/116-A_openai_gpt-4o.jsonl \
    --year 2022 \
    --section A
```

## データ構造

```
data/
├── 2022/
│   ├── 116-A.jsonl          # 問題データ (正解含む)
│   ├── 116-A_metadata.jsonl # カテゴリ、正答率等
│   └── 116-A_translate.jsonl # 英語翻訳
```

### 問題データ形式

```json
{
  "problem_id": "116A1",
  "problem_text": "睡眠時無呼吸症候群による高血圧について...",
  "choices": ["a: 選択肢1", "b: 選択肢2", ...],
  "text_only": true,
  "answer": ["c"],
  "points": "1"
}
```

### メタデータ形式

```json
{
  "problem_id": "116A1",
  "category": "呼吸器",
  "human_accuracy": "87.8",
  "breakdown": ["2.6", "0.0", "87.8", "4.7", "4.9"]
}
```

## ベースライン結果サマリ

| モデル | 平均正答率 |
|--------|----------|
| GPT-4 | 77.5% |
| ChatGPT (GPT-3.5) | 55.1% |
| ChatGPT (英語版) | 57.4% |
| GPT-3 | 41.6% |
| 受験生多数派 | 94.4% |

※ 医師国家試験合格基準は約60%

## 注意事項

- 画像問題は含まれていません（`text_only: true`のみ）
- API使用料に注意（GPT-4oは高コスト）
- LM Studioは日本語対応モデルを推奨
