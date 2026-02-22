# IgakuQA 論文化: 統計的検定ガイド

**作成日**: 2026-02-22
**対象**: IgakuQA評価データの論文化に必要な統計処理

---

## 1. なぜ統計的検定が必要か

現状のデータ例:
- Qwen3-32B: 80.0% (60/75問)
- MedGemma-27B: 76.0% (57/75問)

「Qwen3-32Bの方が4%高い」と記述できるが、査読者は**「それは偶然ではないか？」**と問う。75問のサンプルでは、たまたま得意な問題が出ただけかもしれない。統計的検定は「偶然ではない」ことを数学的に示す手段。

---

## 2. 95%信頼区間（Confidence Interval）

### 意味

「このモデルの**真の実力**はどの範囲にあるか」を示す。

### 計算方法

#### 正規近似法（簡易）

75問中60問正解（80.0%）の場合:

```
p = 60/75 = 0.800
SE = √(p × (1-p) / n) = √(0.800 × 0.200 / 75) = 0.0462
95% CI = p ± 1.96 × SE = 0.800 ± 0.0905
       = [70.9%, 89.1%]
```

#### Clopper-Pearson法（正確法、論文推奨）

正規近似は簡便だが、サンプルが少ない場合や極端な正答率（95%以上、5%以下）では不正確。Clopper-Pearson法は二項分布の正確な信頼区間を計算する。

```python
from scipy.stats import binom

def clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson exact confidence interval"""
    lo = binom.ppf(alpha/2, n, k/n) / n if k > 0 else 0
    hi = binom.ppf(1 - alpha/2, n, k/n) / n if k < n else 1
    return lo, hi

# 例: 75問中60問正解
lo, hi = clopper_pearson(60, 75)
print(f"95% CI: [{lo:.1%}, {hi:.1%}]")
```

### 本研究での計算例

| モデル | 正答率 | 95% CI (正規近似) | 解釈 |
|---|---|---|---|
| gpt-oss-120B | 92.0% (69/75) | [83.9%, 96.7%] | 真の実力は84-97% |
| Qwen3-32B | 80.0% (60/75) | [70.9%, 89.1%] | 真の実力は71-89% |
| MedGemma-27B | 76.0% (57/75) | [65.7%, 84.5%] | 真の実力は66-85% |
| Qwen3-8B | 61.3% (46/75) | [50.2%, 71.6%] | 真の実力は50-72% |

### 重要な注意点

- 75問では信頼区間が**約±9%**と広い
- Qwen3-32B [71-89%] と MedGemma-27B [66-85%] は**重なっている** → 75問だけでは「確実に優秀」とは言えない
- 全400問だと幅が**±4%**に縮小 → **全400問評価が統計的に必要な理由**

```
# 400問の場合
p = 317/400 = 0.793
SE = √(0.793 × 0.207 / 400) = 0.0203
95% CI = [75.4%, 83.1%]  ← 幅が大幅に縮小
```

---

## 3. McNemar検定（モデル間のペア比較）

### なぜ普通のカイ二乗検定ではダメか

普通のカイ二乗検定は「独立な2群」を比較する（例: 薬A群 vs 薬B群の治癒率）。

本研究では**同じ75問（または400問）**を2つのモデルに解かせている。問題ごとに「正解/不正解」が**対応（paired）**している。この対応を無視すると検出力を失う。

### McNemar検定の考え方

75問それぞれについて、2つのモデルの正解/不正解を4分類:

```
                    モデルB
                 正解    不正解
モデルA  正解  |  a   |   b   |
         不正解|  c   |   d   |
```

- **a**: 両方正解（差を生まない）
- **d**: 両方不正解（差を生まない）
- **b**: Aだけ正解（Aが優位な問題）
- **c**: Bだけ正解（Bが優位な問題）

**検定したいのは「bとcに有意な差があるか」**。

### 計算式

```
χ² = (|b - c| - 1)² / (b + c)    # 連続性補正あり
```

自由度1のカイ二乗分布で検定。p < 0.05 なら有意差あり。

### 具体例（仮想データ）

Qwen3-32B vs MedGemma-27B（75問）:

```
                    MedGemma-27B
                   正解(57)  不正解(18)
Qwen3-32B 正解(60)  |  50  |   10   |  = 60
          不正解(15) |   7  |    8   |  = 15
                       57      18       75
```

- b = 10（Qwen3だけ正解）
- c = 7（MedGemmaだけ正解）
- χ² = (|10-7| - 1)² / (10+7) = 4/17 = 0.235
- p = 0.628（有意差なし）

**4%の差があっても、75問では統計的有意差を示せないことがある。**
これは「差がない」ことを意味するのではなく、「サンプルが足りない」ことを意味する。

### 実装

```python
from statsmodels.stats.contingency_tables import mcnemar

# 2x2テーブル: [[a, b], [c, d]]
table = [[50, 10], [7, 8]]
result = mcnemar(table, exact=True)  # exact=True: 小サンプル向け
print(f"p-value: {result.pvalue:.4f}")
```

### 検出力（Power Analysis）

- **75問**: McNemar検定が有意差を検出できるのは、だいたい**15%以上の差**がある場合
- **400問**: **7-8%の差**から検出可能
- これも全400問評価を推奨する理由の一つ

---

## 4. Bonferroni補正（多重比較の問題）

### 問題

35モデルを全ペアで比較すると 35×34/2 = **595回**の検定を行う。
p < 0.05 で検定すると、何も差がなくても**約30回**は「有意差あり」と誤判定する（偽陽性）。

### 解決法

有意水準を検定回数で割る:

```
補正後の有意水準 = 0.05 / 検定回数
```

### 実用的な対処（全ペア比較は不要）

論文では比較を限定する:

1. **合格ライン前後のモデル同士**（例: 76% vs 74%）
2. **サイズ帯別の代表モデル**（4B, 8B, 14B, 32B, 70B, 120B）
3. **同一ベースのFT比較**（Swallow vs ベース、MedGemma vs Gemma）
4. **量子化比較**（8bit vs 4bit、同一モデル）

比較回数を10-20回に絞れば、補正後も p < 0.0025-0.005 で実用的に判定可能。

---

## 5. Cochran's Q検定（3モデル以上の同時比較、オプション）

### 用途

「このサイズ帯の複数モデルに全体として差があるか」を検定。
McNemar検定の多群版。

```python
from statsmodels.stats.contingency_tables import cochrans_q

# 各列がモデル、各行が問題、値は0/1
result = cochrans_q(data_matrix)
print(f"Q statistic: {result.statistic:.2f}, p-value: {result.pvalue:.4f}")
```

有意であれば、事後検定としてMcNemar検定をペアワイズで実施。

---

## 6. 本研究で実施すべき検定のまとめ

| 検定 | 目的 | 適用場面 | 優先度 |
|---|---|---|---|
| **95% CI (Clopper-Pearson)** | 各モデルの精度の信頼区間 | 全モデルの結果テーブル | **必須** |
| **McNemar検定** | 2モデル間のペア比較 | FT効果、量子化影響、合格ライン付近 | **必須** |
| **Bonferroni補正** | 多重比較の偽陽性防止 | McNemar検定の結果に適用 | **必須** |
| **Cochran's Q検定** | 3モデル以上の同時比較 | サイズ帯別グループ比較 | オプション |

---

## 7. 実装に必要なデータ

### 現在のJSON構造の確認が必要

McNemar検定には**問題ごとの正解/不正解リスト**が必要:

```python
# 必要なデータ構造
model_a_results = [1, 0, 1, 1, 0, ...]  # 1=正解, 0=不正解
model_b_results = [1, 1, 1, 0, 0, ...]  # 同じ問題順序で対応
```

結果JSONに各問題の正解/不正解が記録されているか確認すること。

---

## 8. 論文における記載例

### Methods節

> Model accuracy was reported with exact (Clopper-Pearson) 95% confidence intervals. Pairwise comparisons between models evaluated on the same question set were performed using McNemar's exact test. To control for multiple comparisons, Bonferroni correction was applied, with significance set at p < 0.005 (α = 0.05 / 10 pre-specified comparisons). All statistical analyses were performed using Python 3.9 with scipy and statsmodels.

### Results節での記載例

> Qwen3-32B achieved 79.3% (317/400; 95% CI: 75.0%–83.1%) on the full 400-question evaluation, significantly outperforming MedGemma-27B at 71.8% (287/400; 95% CI: 67.2%–76.0%; McNemar's test, p = 0.003).

### 注意: 有意差が出ない場合の記載

> The difference between Qwen3-32B (80.0%) and MedGemma-27B (76.0%) on Section A (75 questions) did not reach statistical significance (McNemar's test, p = 0.63), likely due to the limited sample size.

---

## 9. 75問 vs 400問の検出力比較

| 比較 | 75問での検出可能性 | 400問での検出可能性 |
|---|---|---|
| 92% vs 76% (16%差) | 検出可能（高確率） | 確実に検出可能 |
| 80% vs 76% (4%差) | 検出困難 | 検出可能（場合による） |
| 80% vs 77% (3%差) | 検出不可能 | 検出困難 |

**結論**: 合格ライン付近（75-80%）のモデル間の差を議論するには、**全400問評価が統計的に必須**。

---

## 10. 先行研究での統計処理の扱い

| 研究 | 統計処理 | 備考 |
|---|---|---|
| Kasai et al. (2023) | なし（記述統計のみ） | arXivプレプリント |
| PFN MedLLM (2025) | なし（記述統計のみ） | arXivプレプリント |
| doctorin IgakuQA119 (2025) | なし | Zenn記事 |
| Liu et al. (2025) IJMI | **あり**（カイ二乗検定） | 査読付きジャーナル |
| Kim et al. (2025) Meerkat | **あり**（95%CI, bootstrap） | npj Digital Medicine |
| Safavi-Naini et al. (2025) | **あり**（McNemar検定） | npj Digital Medicine |
| OpenMedLM (2024) | **あり**（95%CI） | Scientific Reports |

**傾向**: arXiv/ブログでは統計処理なしが多いが、**査読付きジャーナル（Scientific Reports, npj Digital Medicine, JMIR）では統計処理が必須**。本研究が目指すジャーナルでは統計的検定は避けられない。
