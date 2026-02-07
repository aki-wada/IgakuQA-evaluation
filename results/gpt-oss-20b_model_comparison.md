# gpt-oss-20b モデル詳細・量子化比較

**評価日**: 2026-02-07〜08
**評価データ**: IgakuQA 2022（第116回医師国家試験）全セクション 400問
**合格ライン**: 75%（300/400）
**評価条件**: Baseline + 2-shot few-shot + max_tokens=1024

---

## 共通仕様（Hugging Face より取得）

| 項目 | 値 |
|------|-----|
| ベースモデル | openai/gpt-oss-20b |
| パラメータ数 | 20B (BF16: 20.9GB) |
| アーキテクチャ | GptOssForCausalLM (gpt_oss) |
| コンテキスト長 | 131,072 tokens |
| 知識カットオフ | 2024-06 |
| ライセンス | Apache 2.0 |
| Vision | No |
| Tool Use | Yes（browser, python） |
| 推論 (Reasoning) | Yes（reasoning effort: low/medium/high） |
| トークナイザ | BOS: `<\|startoftext\|>`, EOS: `<\|return\|>`, PAD: `<\|endoftext\|>` |

---

## モデル一覧

### 1. openai/gpt-oss-20b @8bit

| 項目 | 値 |
|------|-----|
| LM Studio表示名 | openai/gpt-oss-20b @8bit |
| HuggingFace ID | lmstudio-community/gpt-oss-20b-MLX-8bit |
| ベースモデル | openai/gpt-oss-20b |
| フォーマット | safetensors (MLX) |
| 量子化 | 8bit |
| サイズ | 22.26 GB |
| パラメータ数 | 20B (20.9GB BF16) |
| アーキテクチャ | GptOssForCausalLM |
| コンテキスト長 | 131,072 |
| Vision | No |
| Tool Use | Yes |
| ライセンス | Apache 2.0 |
| 知識カットオフ | 2024-06 |
| HF Downloads | 2,358 |
| 公開日 | 2025-08-05 |

**評価結果**: 286/400 = **71.5%** (FAIL)

| Section | A (75) | B (50) | C (75) | D (75) | E (50) | F (75) | Total (400) |
|---------|--------|--------|--------|--------|--------|--------|-------------|
| 正答数 | 57 | 39 | 49 | 57 | 37 | 44 | 286 |
| 正答率 | 76.0% | 78.0% | 65.3% | 76.0% | 74.0% | 58.7% | 71.5% |
| 空回答 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

---

### 2. openai/gpt-oss-20b @mxfp4

| 項目 | 値 |
|------|-----|
| LM Studio表示名 | openai/gpt-oss-20b @mxfp4 |
| HuggingFace ID | lmstudio-community/gpt-oss-20b-GGUF |
| ベースモデル | openai/gpt-oss-20b |
| フォーマット | GGUF |
| 量子化 | MXFP4 (4bit) |
| ファイル名 | gpt-oss-20b-MXFP4.gguf |
| サイズ | 12.11 GB |
| パラメータ数 | 20B (20.9GB BF16) |
| アーキテクチャ | gpt-oss |
| コンテキスト長 | 131,072 |
| Vision | No |
| Tool Use | Yes |
| ライセンス | Apache 2.0 |
| 知識カットオフ | 2024-06 |
| HF Downloads | 157,984 |
| 公開日 | 2025-08-05 |

**評価結果**: 284/400 = **71.0%** (FAIL)

| Section | A (75) | B (50) | C (75) | D (75) | E (50) | F (75) | Total (400) |
|---------|--------|--------|--------|--------|--------|--------|-------------|
| 正答数 | 57 | 37 | 49 | 58 | 37 | 46 | 284 |
| 正答率 | 76.0% | 74.0% | 65.3% | 77.3% | 74.0% | 61.3% | 71.0% |
| 空回答 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

---

### 3. mlx-community/gpt-oss-20b-MXFP4-Q8

| 項目 | 値 |
|------|-----|
| LM Studio表示名 | mlx-community/gpt-oss-20b-mxfp4-q8 |
| HuggingFace ID | mlx-community/gpt-oss-20b-MXFP4-Q8 |
| ベースモデル | openai/gpt-oss-20b |
| フォーマット | safetensors (MLX) |
| 量子化 | MXFP4 (4bit weights) + Q8 (non-weights) |
| サイズ | 12.10 GB |
| パラメータ数 | 20B (20.9GB BF16) |
| アーキテクチャ | GptOssForCausalLM |
| コンテキスト長 | 131,072 |
| Vision | No |
| Tool Use | Yes |
| ライセンス | Apache 2.0 |
| HF Downloads | 702,005 |
| 公開日 | 2025-08-29 |

**評価結果**: 284/400 = **71.0%** (FAIL)

| Section | A (75) | B (50) | C (75) | D (75) | E (50) | F (75) | Total (400) |
|---------|--------|--------|--------|--------|--------|--------|-------------|
| 正答数 | 56 | 40 | 47 | 56 | 36 | 48 | 284 |
| 正答率 | 74.7% | 80.0% | 62.7% | 74.7% | 72.0% | 64.0% | 71.0% |
| 空回答 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

---

### 4. mlx-community/gpt-oss-20b-MXFP4-Q4

| 項目 | 値 |
|------|-----|
| LM Studio表示名 | mlx-community/gpt-oss-20b-mxfp4-q4 |
| HuggingFace ID | mlx-community/gpt-oss-20b-MXFP4-Q4 |
| ベースモデル | openai/gpt-oss-20b |
| フォーマット | safetensors (MLX) |
| 量子化 | MXFP4 (4bit weights) + Q4 (non-weights) |
| サイズ | 11.21 GB |
| パラメータ数 | 20B (20.9GB BF16) |
| アーキテクチャ | GptOssForCausalLM |
| コンテキスト長 | 131,072 |
| Vision | No |
| Tool Use | Yes |
| ライセンス | Apache 2.0 |
| HF Downloads | 4,528 |
| 公開日 | 2025-08-29 |

**評価結果**: BROKEN（セクションA-Bのみ実施、中断）

| Section | A (75) | B (50) |
|---------|--------|--------|
| 正答数 | 7 | 10 |
| 正答率 | 9.3% | 20.0% |
| 空回答 | 60 | 27 |

**障害内容**: 推論トークンが max_tokens を消費し、回答が生成されない（大量の空回答）

---

### 5. InferenceIllusionist/gpt-oss-20b-MLX-4bit

| 項目 | 値 |
|------|-----|
| LM Studio表示名 | inferenceillusionist/gpt-oss-20b-mlx |
| HuggingFace ID | InferenceIllusionist/gpt-oss-20b-MLX-4bit |
| ベースモデル | openai/gpt-oss-20b |
| フォーマット | safetensors (MLX) |
| 量子化 | 4bit |
| サイズ | 11.80 GB |
| パラメータ数 | 20B (20.9GB BF16) |
| アーキテクチャ | GptOssForCausalLM |
| コンテキスト長 | 131,072 |
| Vision | No |
| Tool Use | Yes |
| ライセンス | Apache 2.0 |
| HF Downloads | 628 |
| 公開日 | 2025-08-06 |

**評価結果**: BROKEN（セクションAのみ実施、中断）

| Section | A (75) |
|---------|--------|
| 正答数 | 7 |
| 正答率 | 9.3% |
| 空回答 | 60+ |

**障害内容**: mlx-community Q4 と同一症状。推論トークンで max_tokens を消費。

---

### 6. lmstudio-community/gpt-oss-safeguard-20b-MLX-MXFP4

| 項目 | 値 |
|------|-----|
| LM Studio表示名 | gpt-oss-safeguard-20b-mlx |
| HuggingFace ID | lmstudio-community/gpt-oss-safeguard-20b-MLX-MXFP4 |
| ベースモデル | openai/gpt-oss-safeguard-20b |
| フォーマット | safetensors (MLX) |
| 量子化 | MXFP4 (4bit) |
| サイズ | 11.15 GB |
| パラメータ数 | 20B (20.9GB BF16) |
| アーキテクチャ | GptOssForCausalLM |
| コンテキスト長 | 131,072 |
| Vision | No |
| Tool Use | Yes |
| ライセンス | Apache 2.0 |
| 知識カットオフ | 2024-06 |
| HF Downloads | 7,780 |
| 公開日 | 2025-10-28 |
| 備考 | 安全性フィルター付きバリアント (safeguard) |

**評価結果**: BROKEN（セクションAのみ実施、中断）

| Section | A (75) |
|---------|--------|
| 正答数 | 27 |
| 正答率 | 36.0% |
| 空回答 | 39 |
| APIエラー | 6 |

**障害内容**: 空回答多数 + APIエラー。他の MLX 4bit より若干ましだが実用不可。

---

## 比較サマリー

| # | モデル | フォーマット | 量子化 | サイズ | 正答率 | 状態 | HF DL数 |
|---|--------|------------|--------|--------|--------|------|---------|
| 1 | openai @8bit | MLX | 8bit | 22.26 GB | **71.5%** | 動作OK | 2,358 |
| 2 | openai @mxfp4 | GGUF | MXFP4 4bit | 12.11 GB | **71.0%** | 動作OK | 157,984 |
| 3 | mlx-community Q8 | MLX | MXFP4+Q8 | 12.10 GB | **71.0%** | 動作OK | 702,005 |
| 4 | mlx-community Q4 | MLX | MXFP4+Q4 | 11.21 GB | ~9% | BROKEN | 4,528 |
| 5 | InferenceIllusionist | MLX | 4bit | 11.80 GB | ~9% | BROKEN | 628 |
| 6 | safeguard | MLX | MXFP4 | 11.15 GB | ~36% | BROKEN | 7,780 |

---

## 重要な発見

### 1. 動作する3モデルの精度はほぼ同一
- 71.0%〜71.5% の範囲（最大差 0.5%）
- 8bit → MXFP4 の量子化による精度低下はほぼ無視できる

### 2. フォーマットと量子化方式が推論モデルの動作を左右する
- **MLX 8bit (22.26GB)**: 正常動作（空回答ゼロ）
- **GGUF MXFP4 (12.11GB)**: 正常動作（空回答ゼロ）
- **MLX MXFP4+Q8 (12.10GB)**: 正常動作（空回答ゼロ）
- **MLX 4bit系 (11-12GB)**: 推論トークン生成が壊れて空回答多発
- 動作する12GB級モデルは GGUF MXFP4 と MLX MXFP4+Q8 の2種

### 3. MLX の non-weight 量子化が鍵
- mlx-community **MXFP4+Q8**（non-weight 8bit）: 正常動作
- mlx-community **MXFP4+Q4**（non-weight 4bit）: 壊れる
- 重みの量子化は同じ MXFP4 でも、non-weight パラメータの精度が推論モデルの動作に影響

### 4. コストパフォーマンス
- **最良**: openai @mxfp4 GGUF (12.11GB) または mlx-community MXFP4-Q8 (12.10GB)
  - MLX 8bit (22.26GB) 対比で **46%のメモリ節約**、精度低下は **-0.5%**
- 8bit を選ぶメリットはほぼ無い

### 5. 全モデル不合格
- 合格ライン 75%（300/400）に対し、最高でも 71.5%（286/400）
- あと 14問で合格
- gpt-oss-20b はパラメータ数の制約で医師国家試験レベルには届かない
  - 参考: gpt-oss-120b は 92.0%（セクションA）で余裕の合格
