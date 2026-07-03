[English →](README.md) | **中文**

# ZeroBP-4B 与 Phase H

**一个资源受限的研究项目**:(1) 在单块 GPU 上构建一个*纯无反向传播*的 40 亿参数语言模型,(2) 系统地测出它的推理能力在哪里失效,(3) 用一个隔离的*可训练注意力对照骨架*把**算法**性限制与**架构**性限制分开。

**接着读:** [完整报告](paper/PAPER_DRAFT.zh.md) · [Workshop 论文(8 页)](paper/PAPER_workshop.zh.md) · [幻灯](paper/slides_outline.zh.md) · [Kaggle 提交](docs/KAGGLE_README.zh.md) · [总档索引](docs/MASTER_ARCHIVE.md)

![单块 Tesla T4](https://img.shields.io/badge/hardware-single%20Tesla%20T4-blue) ![零反向传播](https://img.shields.io/badge/training-zero%20backprop-8A2BE2) ![参数](https://img.shields.io/badge/params-4.16B-green) ![可复现](https://img.shields.io/badge/results-deterministic%20%26%20logged-brightgreen)

---

## 一句话摘要(TL;DR)

- 构建并锁定了一个**纯 ZeroBP 4B 基线** —— 41.6 亿参数,**全程无反向传播**(无 `autograd`/`.backward()`),单块 T4,提交路径确定、可复现。
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

---

## 研究如何展开

五个阶段,每个阶段回答上一个阶段提出的问题。*(它们对应日志中使用的内部阶段代号 A–H;下表给出每个阶段实际做了什么的通俗描述。)*

| 阶段 | 做了什么 | 结果 |
|---|---|---|
| **1 · 构建无反向传播模型** *(A–D)* | 在极小规模上原型化局部学习规则,找出让它能学的关键(基于内容的专家特化 + 每步只更新极小比例的模型),再扩到 40 亿参数、单 GPU 并稳定训练。 | 一个可用的语言模型(困惑度 ≈ 1355–1391) |
| **2 · 找出它在哪里失效** *(E)* | 固定模型,在三类任务上做后训练 —— 情感(表层)、关系蕴含(理解)、多步算术(顺序推理),逐步注入反向传播。 | 情感 79%;关系与多步停在随机水平 |
| **3 · 从模型内部尝试修复** *(F–G)* | 五种模型内干预:更丰富数据、结构训练目标、不塌缩读出、只训练注意力、更深反向传播。 | 4B 上都无法移动边界 |
| **4 · 定位瓶颈** *(G)* | 探针检验关系信息存在于模型内部何处。 | 它在内部表示中完全不存在 |
| **5 · 对照实验** *(H)* | 只替换架构 —— 一个标准可训练注意力模型 + 普通反向传播 —— 复跑相同任务。 | 真实 SNLI 34%→70%;浅多步被解出 —— 障碍在架构 |

---

## 两个入口

**A · GitHub 落地页(即本页)** —— 30 秒定位:这是什么、主要结果、去哪看。

**B · 论文 / 技术报告** —— 供认真评审:问题、方法、实验、限制、结论 → **[paper/PAPER_DRAFT.zh.md](paper/PAPER_DRAFT.zh.md)**(完整)或 **[paper/PAPER_workshop.zh.md](paper/PAPER_workshop.zh.md)**(8 页)。

## 本仓库的两条线

仓库把**产品级提交**与**研究调查**干净地分开 —— 二者从不共享训练路径。

**1 · 提交线** —— 干净、合规、可复现的纯 ZeroBP 基线。
→ [`kaggle_zerograd_moe.py`](kaggle_zerograd_moe.py)(模型 + 默认路径)· [`kaggle_run/`](kaggle_run/)(提交 kernel)· [`docs/KAGGLE_README.zh.md`](docs/KAGGLE_README.zh.md)。

**2 · 研究线** —— 完整实验 arc + 对照骨架调查。
→ [`docs/MASTER_ARCHIVE.md`](docs/MASTER_ARCHIVE.md) → [`paper/PAPER_workshop.zh.md`](paper/PAPER_workshop.zh.md) → [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md) → [`phase_h/`](phase_h/)。

## 仓库结构

```text
.
├── README.md · README.zh.md            # 本落地页(英文 / 中文)
├── LICENSE · CITATION.cff · requirements.txt
│
├── kaggle_zerograd_moe.py              # 无反向传播 4B 模型 + 纯 ZeroBP 提交路径
├── kaggle_run/                         # 官方提交 kernel(notebook + metadata)
├── selfcheck.py · build_kaggle_kernels.py · orchestrate_kaggle.py
│
├── phase_e*.py · c1_4b.py · adapt_sentiment.py       # 阶段 2:后训练能力上限
├── f1_data.py · f2_aux.py · h1_attn.py               # 阶段 3:模型内修法(数据 / 目标 / 注意力)
├── v2_readout.py · v2_attn.py · v2_deepbp.py         # 阶段 4:定位瓶颈的探针
├── task_nli.py · task_arith.py                       # 共享合成推理任务
├── track1_sst2_4b.py · track1_radar.py · make_figures.py · build_track1_kernels.py
│
├── phase_h/                            # 阶段 5 —— 对照:隔离的可训练注意力骨架
│
├── paper/                             # 报告(英文 + 中文)
│   ├── PAPER_DRAFT.md · PAPER_workshop.md · slides_outline.md (+ .zh.md)
├── docs/                             # 权威文档
│   ├── MASTER_ARCHIVE.md · EXPERIMENT_LEDGER.md · ARCHITECTURE.md
│   ├── ENGINEERING.md · SUBMISSION.md · KAGGLE_README.md · archive_zh.md
│   └── adr/                           # 架构决策记录(ADR-001..005)
├── figures/                          # 图 + make_figures.py 输出
└── results/                          # 权威真实运行结果 JSON(+ 索引)
```

**按合适的颗粒度阅读:** 落地页(本页)→ 故事([paper/](paper/))→ 上下文([docs/MASTER_ARCHIVE.md](docs/MASTER_ARCHIVE.md))→ 证据([docs/EXPERIMENT_LEDGER.md](docs/EXPERIMENT_LEDGER.md)、脚本、[results/](results/))。

## 配图讲解(白话版)

- **系统构成**:一个 40 亿参数、单 GPU、**完全不用反向传播**训练的语言模型(每步只改 0.42% 的模型,峰值 ~8GB;同规模用反向传播需 ~31GB 装不下)。它作为语言模型是能用的。
- **发现的边界**:情感任务能做好(79%),但关系推理和多步推理无论怎么调都停在随机水平 —— 且探针表明所需结构**根本不在模型内部表示里**。
- **对照实验**:换成一个标准可训练注意力骨架 + 全反向传播,同样的任务被解出(真实 SNLI 34%→70%)—— 证明失败是**设计**造成的,不是任务不可学。
- **诚实上限**:对照骨架也在 4 步以上推理撞墙(放大 5 倍也不动),真实 GSM8K 仅 2%。

完整方法、全部结果与**诚实纠错日志**见 [paper/PAPER_DRAFT.zh.md](paper/PAPER_DRAFT.zh.md)。

## 快速开始

```bash
pip install -r requirements.txt

# 本机复现(无需 GPU)
python3 kaggle_zerograd_moe.py    # 无反向传播模型,小配置 -> final_ppl 6.251,零 autograd 通过,确定性
python3 phase_h/ph_nli.py         # 对照骨架解合成关系推理 -> 100%
python3 make_figures.py           # 重新生成 figures/ 里的图

# Kaggle 基线(需 Kaggle 账号 + T4)
python3 build_kaggle_kernels.py && kaggle kernels push -p kaggle_run   # 见 docs/KAGGLE_README.zh.md
```

预期提交门:`final_ppl = 6.251` · 零 autograd **通过** · 确定性 **通过**。每个数字都可追溯至 [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md);原始结果在 [`results/`](results/)。

## 引用与许可

MIT 许可([LICENSE](LICENSE));机器可读引用见 [CITATION.cff](CITATION.cff)。独立研究项目,BibTeX 见英文 README。

*完整项目历史(含探索阶段与过程笔记)保存在备份镜像:[github.com/Yanjin-ai/zerogradient](https://github.com/Yanjin-ai/zerogradient)。*
