[English →](README.md) | **中文**

# ZeroBP-4B 与 Phase H

**一个资源受限的研究项目**:(1) 在单块 GPU 上构建一个*纯无反向传播*的 40 亿参数语言模型,(2) 系统地测出它的推理能力在哪里失效,(3) 用一个隔离的*可训练注意力对照骨架*把**算法**性限制与**架构**性限制分开。

**成果入口:** [Kaggle 说明](KAGGLE_README.zh.md) · [Workshop 论文](PAPER_workshop.zh.md) · [完整报告](PAPER_DRAFT.zh.md) · [幻灯](slides_outline.zh.md) · [总档索引](MASTER_ARCHIVE.md)

`单块 Tesla T4` · `零 autograd` · `41.6 亿参数` · `确定性可复现`

---

## 一句话摘要(TL;DR)

- 构建并锁定了一个**纯 ZeroBP 4B Kaggle 基线** —— 41.6 亿参数,**全程无反向传播**(无 `autograd`/`.backward()`),单块 T4,提交路径确定、可复现。
- 测出一道清晰的**能力边界**:ZeroBP 在 bag 式**情感(79%)**上有效,但在真实 **NLI 和多步算术**上停在随机水平。
- 尝试了五种"骨架内"修法(更丰富数据、结构目标、不塌缩读出、只训 attention 的 BP、更深 BP)—— **都无法消除 4B 边界**。
- 引入 **Phase H**,一个完全隔离的*可训练注意力对照骨架*,在**真实 SNLI 上达 69.97%**(vs 随机),并解出浅多步算术(k ≤ 3)。
- **核心结论:** ZeroBP 在关系与多步任务上的失败主要是**架构性的,而非任务太难** —— 而且连对照骨架也有诚实的上限(k ≥ 4 的抗规模多步之墙)。

## 关键结果总表

| 维度 | ZeroBP-4B(无反向传播) | Phase H(可训练注意力对照) | 解读 |
|---|---:|---:|---|
| Kaggle 语言建模基线 | **困惑度 1391 / 1355** | — | 资源受限下的强基线 |
| 情感(bag) | **79%**(一点 BP) | — | 简单组合信号可学 |
| 真实 SNLI(关系) | 33.4%(随机) | **69.97%** | 关系结构是**架构性的** |
| 多步算术 | 任何深度随机 | **k ≤ 3 解出,k ≥ 4 墙** | 浅推理可装进对照骨架 |
| 生成式 GSM8K | — | 2.0% EM | 小模型诚实的上限 |

![同样的任务,两种设计 —— 限制在架构,不在任务](figures/capability_comparison.png)

> 第一次来?配图的白话讲解见下文,方法与证据见[完整报告](PAPER_DRAFT.zh.md)。

---

## 本仓库的两条线

仓库把**产品级提交**与**研究调查**干净地分开 —— 二者从不共享代码。

**1 · Kaggle 提交线** —— 干净、合规、可复现的纯 ZeroBP 基线。
→ 从 [`kaggle_zerograd_moe.py`](kaggle_zerograd_moe.py) + [`KAGGLE_README.zh.md`](KAGGLE_README.zh.md) 开始。

**2 · 研究线** —— 完整实验 arc + 对照骨架调查。
→ 从 [`MASTER_ARCHIVE.md`](MASTER_ARCHIVE.md) → [`PAPER_workshop.zh.md`](PAPER_workshop.zh.md) → [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) → [`phase_h/`](phase_h/) 开始。

## 项目地图

| 路径 | 层级 | 是什么 |
|---|---|---|
| `kaggle_zerograd_moe.py` · `KAGGLE_README.md` | 提交 | 纯 ZeroBP 4B 模型 + 评审文档 |
| `PAPER_workshop.md` · `PAPER_DRAFT.md` · `slides_outline.md` | **故事** | 论文(短 + 全)与演讲提纲 |
| `MASTER_ARCHIVE.md` · `项目总档案.md` | **档案** | 权威索引(双语)+ 中文深度叙事 |
| `EXPERIMENT_LEDGER.md` · `docs/adr/` · `ARCHITECTURE.md` | 证据 | 逐实验配置/commit/指标 + 决策 |
| `phase_h/` | 研究代码 | 隔离的可训练注意力对照骨架 |
| `phase_e*.py` · `v2_*.py` · `task_*.py` · `track1_sst2_4b.py` | 研究代码 | 边界线实验(BP,默认关,隔离) |
| `results/` · `figures/` · `runs/` | 数据 | 权威结果 JSON、图、运行摘要 |

*按合适的颗粒度阅读:**落地页**(本页)→ **故事**(论文)→ **档案**(上下文)→ **证据**(台账/脚本)。*

## 快速开始

```bash
# 研究材料(从这里开始理解项目)
open MASTER_ARCHIVE.md            # 叙事 + 锁定结论
open PAPER_workshop.zh.md         # 8 页故事版

# 本机复现(无需 GPU)
python3 kaggle_zerograd_moe.py    # 无反向传播模型,小配置 -> final_ppl 6.251,零 autograd 通过,确定性
python3 phase_h/ph_nli.py         # 对照骨架解合成关系推理 -> 100%
python3 make_figures.py           # 重新生成 figures/ 里的图

# Kaggle 基线(需 Kaggle 账号 + T4)
python3 build_kaggle_kernels.py && kaggle kernels push -p kaggle_run   # 见 KAGGLE_README.zh.md
```

预期提交门:`final_ppl = 6.251` · 零 autograd **通过** · 确定性 **通过**。

## 仓库结构

```text
.
├── README.md  /  README.zh.md          # 本落地页(英文 / 中文)
├── kaggle_zerograd_moe.py              # ZeroBP-4B 模型 + 纯 ZeroBP 提交路径
├── KAGGLE_README.md  /  .zh.md         # 面向 Kaggle 的基线文档
├── PAPER_DRAFT.md  /  .zh.md           # 完整技术报告
├── PAPER_workshop.md  /  .zh.md        # 8 页精简论文
├── slides_outline.md  /  .zh.md        # 演讲提纲
├── MASTER_ARCHIVE.md                   # 权威项目索引(双语)
├── 项目总档案.md                        # 中文深度档案
├── EXPERIMENT_LEDGER.md                # 实验台账:commit、配置、指标
├── ARCHITECTURE.md · ENGINEERING.md · SUBMISSION.md
├── phase_h/                            # 隔离的可训练注意力对照骨架
├── phase_e*.py · v2_*.py · task_*.py · track1_sst2_4b.py   # 边界线研究
├── results/                            # 权威真实运行结果 JSON
├── figures/  · make_figures.py         # 图 + 生成器
└── docs/adr/                           # 架构决策记录(ADR-001..005)
```

## 配图讲解(白话版)

- **我们造了什么**:一个 40 亿参数、单 GPU、**完全不用反向传播**训练的语言模型(每步只改 0.42% 的模型,峰值 ~8GB;同规模用反向传播需 ~31GB 装不下)。它作为语言模型是能用的。
- **发现的边界**:情感任务它能做好(79%),但关系推理和多步推理无论怎么调都停在随机水平 —— 且我们证明所需结构**根本不在模型内部表示里**。
- **对照实验**:换成一个标准可训练注意力骨架 + 全反向传播,同样的任务被解出(真实 SNLI 34%→70%)—— 证明失败是**设计**造成的,不是任务不可学。
- **诚实上限**:对照骨架也在 4 步以上推理撞墙(放大 5 倍也不动),真实 GSM8K 仅 2% —— 指向具体的下一步(curriculum / chain-of-thought),而非空谈。

完整方法、全部结果与**诚实纠错日志**见 [PAPER_DRAFT.zh.md](PAPER_DRAFT.zh.md)。

## 论文与报告

- [`PAPER_workshop.zh.md`](PAPER_workshop.zh.md) —— 精简论文版(~8 页),面向 workshop 投稿。
- [`PAPER_DRAFT.zh.md`](PAPER_DRAFT.zh.md) —— 完整技术报告,含扩展背景与诚实**纠错日志**。
- [`slides_outline.zh.md`](slides_outline.zh.md) —— 11 页演讲提纲。

论文按**三条线**(提交 → 边界 → 对照)组织,而非按时间顺序。每个数字都可追溯至 [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md);原始结果在 [`results/`](results/)。

## 引用与关于

独立研究项目(单 GPU、无反向传播 LM + 架构对照研究)。引用格式见英文 README 的 BibTeX。

*完整项目历史备份镜像:[github.com/Yanjin-ai/zerogradient](https://github.com/Yanjin-ai/zerogradient)。*
