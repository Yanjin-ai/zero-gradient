# Phase D 提交版（当前最佳，实测 T4）

提交 notebook：`kaggle_zerograd_moe.ipynb`（KAGGLE 配置默认 = **BPE + 2 层 MLP 头 + Phase C 稳定 schedule**，无需任何 env 变量）。

## 当前最佳结果（实测）
- 结构：4.160B 零梯度 MoE（专家=关键路径 FFN）+ BPE subword 词表 + 2 层 MLP 读出头。
- **WikiText-103 test perplexity ≈ 1391（subword）**，相对 unigram(2814) **优 51%**。
- 训练**单调、漂移=0、早停 t\*≈54min**（150min 预算内提前锁定最佳）。
- **7/7 合规门**；峰值显存 **8.33GB/16GB**；**8546 tok/s**；零 autograd、确定性。
- BP 训 4B 需 ~31GB → T4 OOM（核心对比）。

## 质量演进（同设定内可比，跨设定换基准）
| 设定 | WikiText-103 test ppl | 说明 |
|---|---|---|
| word-level 32k | 1360 | 被 `<unk>` 通缩的虚低数 |
| BPE subword | 1998 | 诚实基准（无 `<unk>`）|
| **BPE + MLP 头** | **1391** | 单杠杆：头从单层→2层，−30%，优 unigram 51% |

## 合规与叙事（不变）
两条腿：① 零梯度特化 MoE 真能学（内容路由 + 局部规则）；② 4B 常驻 + 每步仅更新 0.42% 专家（sparse learning）→ T4/3h 可训、击败 BP 显存。**不声称智能调度提升性能**——调度器是确定性 round-robin（为确定性+稳定）。

## 运行
- 本机逻辑验证（小配置，word/linear，CPU）：`python kaggle_zerograd_moe.py`
- Kaggle 提交：上传 `kaggle_zerograd_moe.ipynb`，GPU=T4，挂 WikiText-103 数据集，运行（自动 bpe+mlp+稳定 schedule，~54min 早停，3h 预算内）。
- 实验开关（覆盖默认）：`ZG_TOKENIZER=word|bpe`、`ZG_HEAD=linear|mlp`、`ZG_RUN_MIN=<分钟>`。

## 下一步（Phase D 单杠杆，叠在 1391 基线上）
D-2 局部损失增强（小配置先验证单调性）→ 4B 单杠杆 run；之后视情况 D-3 加深主干 / 可训练 attention，及并行 C.1 后训练试点。
