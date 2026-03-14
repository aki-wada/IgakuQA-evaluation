# CLAUDE.md — IgakuQA Evaluation Project

## プロジェクト概要

日本の医師国家試験（IgakuQA）を用いたローカルLLMベンチマーク評価プロジェクト。
35モデル以上を Mac Studio M3 Ultra + LM Studio 環境で体系的に評価。

**GitHub**: https://github.com/aki-wada/IgakuQA-evaluation
**現在の目標**: 評価結果の学術論文化（英文、査読付きジャーナル）

---

## 重要ファイル

| ファイル | 内容 |
|---|---|
| `EVALUATION_PROGRESS.md` | **最重要** — 全評価結果・発見・考察の詳細記録 |
| `docs/PAPER_PLAN.md` | 論文化計画（構成案、追加実験、ジャーナル候補） |
| `article_igakuqa.md` | 書籍向け日本語記事（コラム版） |
| `results/` | 全評価結果JSON（150+ファイル） |
| `plots/` | 既存分析プロット（15枚） |
| `evaluate_lmstudio_batch.py` | メイン評価スクリプト |
| `evaluate_prompt_comparison.py` | プロンプト比較実験スクリプト |
| `analyze_results.py` | 結果分析スクリプト |
| `plot_*.py` | プロット生成スクリプト群 |

---

## 評価条件

- **マシン**: Mac Studio M3 Ultra（192GB）
- **LLMサーバ**: LM Studio（localhost:1234）
- **評価問題**: 第116回 医師国家試験 2022年
- **主要評価**: セクションA（75問）、一部モデルは全400問（A〜F）
- **Few-shot**: 2例（同年度の別問題）
- **Temperature**: 0
- **プロンプト4種**: Baseline / 案A（回答形式強化）/ 案B（段階的思考）/ 案C（日本医療文脈）

---

## 主要な発見（論文のコア）

1. **gpt-oss-120B が92.0%で全モデル最高** — GPT-4(80%)を大幅超え
2. **32B以上で合格圏内**（14/35+モデルが75%超え）
3. **max_tokensが推論モデルの正答率を支配**: gpt-oss-120B mt=50→33% / mt=1024→92%
4. **日本語FTの質的差異**: Swallow-70B(+13.3%) vs Shisa-70B(-6.7%)
5. **MoE効率**: Qwen3-Next-80B（稼働3B）で85.3%、0.4s/問
6. **医療特化の逆転**: MedGemma-27B(71.8%) < Qwen3-32B(79.3%)
7. **プロンプト最適化はモデルサイズ依存**: 大型→Baseline / 中型→段階的思考 / 小型→形式強化

---

## 論文化の次のステップ

### 必須（Priority 1）
- [ ] 主要10モデルの全400問評価
- [ ] 統計的検定の実装（McNemar, 95% CI, Bonferroni）
- [ ] 論文用Figure作成（日本語FT比較、max_tokens影響、MoE効率）

### 推奨（Priority 2）
- [ ] 複数年度（2018-2021）での主要5モデル評価
- [ ] 誤答パターンの質的分析
- [ ] 量子化影響のサブ分析

### 強化（Priority 3）
- [ ] 推論時間 vs 正答率のパレート最適分析
- [ ] USMLE (MedQA) との交差比較

---

## ユーザーの環境・好み

- 放射線科医、大学教員
- 日本語での対話を好む
- Mac中心（M2 Max 96GB / M4 Max 128GB / M3 Ultra 512GB）
- Windows RTX 5090 も使用（読影業務）
- 逐次修正型ワークフロー（小さな変更を積み重ね）
- ドキュメントの正確性・保存性を重視

---

## 親プロジェクト CLAUDE.md の行動規則を継承

- 既存内容の保持（NO SILENT DELETION）
- UIは「見たものだけを書く」
- 壊すくらいなら、進めるな。分からないなら、書くな。削るくらいなら、聞け。
