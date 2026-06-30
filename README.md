# Zero-Gradient Sparse-Learning MoE — 项目全记录

> Kaggle [The Post-Backprop Challenge: Zero-Gradient Learning for Efficiency](https://www.kaggle.com/competitions/the-post-backprop-challenge-zero-gradient-learning-for-efficiency)
> 从 0 到可提交：竞赛侦察 → nano 机制验证（A）→ 循序渐进 scale → 诚实定稿 → 真实 T4 上跑 ≥4B（B）→ schedule 稳定化（C）。
> 全程**零反向传播**（无 `torch.autograd`/`.backward()`/optimizer），手写局部规则。

---

## 0. 一句话

在**单 T4 / 3h / 无全局梯度**约束下，自研一个**内容路由的 MoE**，用**逐专家局部规则**训练 **4.156B 常驻参数**模型：每步只更新 **0.42% 的专家**（sparse *learning*，非仅 sparse forward），峰值显存 **8.3GB**（BP 训 4B 需 ~31GB → T4 直接 OOM）。在 WikiText-103 上 test perplexity ≈ **1391**（BPE subword，早停）/ **1355**（跑满预算）、7/7 合规门全过、确定性可复现。**核心价值是"4B 可训练的可行性 + 系统效率"，不是"调度更聪明降 PPL"——后者经诚实实验被否。** 后续 Phase D–F 的能力边界与少量 BP 研究见 §0.5。

---

## 0.5 当前阶段技术总结（Phase D–F + 架构发现，2026-06）

> 项目已从"能否跑通"推进到"ZeroBP 的能力边界在哪、少量 BP 能补到哪、架构瓶颈是什么"。以下是当前定论（4B 真实 T4 + 小配置交叉验证；细节见 [项目总档案.md](项目总档案.md) §1.5🔒 与 [Phase-F charter.md](Phase-F%20charter.md) §10🔒）。

**① ZeroBP 预训练成立，但有清晰的能力边界。** 纯 ZeroBP 4.16B 在 WikiText-103 上可训得有用 LM（test ppl ≈**1391**（早停）/≈**1355**（跑满预算），BPE+MLP 头，零 autograd、7/7 门）。但在后训练上，**能力随任务结构复杂度递减**（任务-方法矩阵）：

| 任务 | 结构 | 4B zero-shot | 4B 少量 BP（embedding） | 小配置最好 |
|---|---|---|---|---|
| 情感 | bag 组合 | 60% | **79%** | 100% |
| NLI | 关系对齐 | 33%(chance) | 33%(不转移) | 62% |
| 2 步算术 | 多步计算 | — | gate 失败 | ~chance |

**② 少量 BP 部分有效——但是 embedding 在扛，且有限。** 在情感（bag 组合）上，少量真实 BP（仅 embedding+头）把 4B 从 60%→**79%**、近零遗忘，而 6 种纯 ZeroBP 后训练全 plateau ~62%——**墙的本质是 BP vs ZeroBP**。但对关系对齐（NLI），三条 ZeroBP 路线（更丰富数据 +2pp / 结构辅助目标 +0.5pp / attention 局部规则失败）都装不进关系几何；少量 BP 在小配置能撬到 59%（**公平读出下是 embedding 扛，attention 只 +0.3pp**），但**不转移到 4B**（4B NLI 即便 3000 步 +高 attention lr 仍 chance）。

**③ 架构瓶颈新线索：末位塌缩。** 模型在 `context()` 里把**整个序列塌缩成"最后一个位置"一个向量**，所有 block 都在这一个向量上算。对关系/对齐/多步任务，信息很可能在塌缩时丢失——这比"attention 弱"或"BP 不够"更根本，是关系任务难的疑似真因。

**④ 下一阶段（Phase G 方向）需要动结构归纳偏置。** 要靠拢现代 LLM 的关系/多步能力，"ZeroBP backbone + 少量顶层 BP"不够，需要更强的结构归纳偏置：**不塌缩的序列读出（pooling/多位置）/ 真正可训练的 attention / 更深 BP**——这超出当前 Hybrid 边界，留作 Phase G。

**提交版隔离（红线）**：官方 Kaggle 提交 = 纯 ZeroBP 4B（`kaggle_run`，见 [SUBMISSION.md](SUBMISSION.md)），全程零 autograd；所有 Phase E/F 的少量 BP 都在**独立研究脚本 + 默认 off**，不进提交。

---

## 1. 竞赛与硬约束（已核实）

| 维度 | 要求 |
|---|---|
| 参数门禁 | 训练前常驻 **≥4,000,000,000 fp16 参数** |
| 零梯度 | 完全替换 BPTT + 所有 autograd/`.backward()`/optimizer，用手写局部规则 |
| 硬件/时间 | 单 Tesla T4 16GB 或 4 核 CPU，**3 小时** |
| 复现 | 强确定性（SEED=42、deterministic algorithms） |
| 数据 | 自备；评测 WikiText-103 perplexity（offline gate）|
| 提交 | 公开 Kaggle Notebook + 可审计产物 |

详见 [memory: competition-mechanics] / 文中各定稿文档。

---

## 2. 全景路线（从 0 到现在）

```
竞赛侦察 ──▶ Phase A（nano 机制验证，本机 CPU）──▶ 诚实定稿 ──▶ Phase B（真实 T4 ≥4B）──▶ Phase C（schedule 稳定化）
                │                                                        │
                ├ E01-E08  局部学习 + 重要性 controller（单层）              ├ smoke/v1/v2/baseline-3h/PhaseC
                ├ E09-E10  Stage3/4 nano-MoE 内容路由 + 特化 + 控制器        └ 4.156B 真跑、击败 BP 显存
                ├ E11-E13  scale 3.1/3.2/3.3（多层/真实语料/4B-surrogate）
                └ E14      覆盖诊断（controller 角色坐实）
```

**方法论**（教科书式、也是加分项）：**toy 验证机制 → real 看哪些 survive**。toy 排除架构混淆、确认机制级真相，再带到真实分布看退化——比"直接上真实数据迷路"干净。

---

## 3. 完整实验表（A 阶段 nano，E01–E14）

| ID | 做了什么 | 结果 / 发现 |
|---|---|---|
| E01 | nano 切片 v1：deeply-supervised 局部头 + 重要性 vs 随机 | ppl 24、没过 unigram、多数类塌缩。bug：注意力缺残差 + embedding 没训 |
| E02 | 修注意力残差 + embedding 训练 | ppl 24→**7.36**，能学了；headline 仍噪声 |
| E03 | v2 系统：锁定 controller（leverage 等）+ 4 基线 + 指标 + 历史 | score 稳定性爆（1155）；importance≈random；model<bigram |
| E04 | z-score 裁剪 | score 稳定性 1155→**0.45** |
| E05 | 上下文依赖语料 + 统一读出头 + lr0.3 + 专家32 + 小初始化 | backbone **5.75 < bigram 6.37**；但加法式专家在强 backbone 上冗余 |
| E06 | 诊断 backbone-only vs full | 专家"帮还是害"取决于 backbone 强弱（强则害） |
| E07 | **B 实验**：瘛颈 backbone 让专家承重 | 专家承重（7.0→5.7）；importance vs random 仍噪声级 |
| E08 | v3 controller（coverage/deficit） | 无稳定胜；**定因：专家"不可区分"（固定哈希无特化）→ 需 A** |
| E09 | **Stage 3** nano-MoE：EMA-prototype 内容路由 + 容量限制 | ppl **5.97**；**purity 0.674 vs 随机 0.25**，SPECIALIZED✓。关键：路由键须载判别信号（topic 0.67/mean-pool 0.35/末token 0.29）|
| E10 | **Stage 4** controller v3 接入特化 MoE | **importance>random 稳定**（4/5 seed）；gap 随预算收紧放大（k=10:+0.017→k=2:+0.079）|
| E11 | **Scale 3.1** nano→small 堆叠 MoE | 特化更锐（purity 0.80→0.94）；**controller 优势随 N 侵蚀反转**（+0.048→−0.119）。修：coverage 随 N 放大（v4 (N/16)²）→ 64e 翻盘 +0.024 |
| E12 | **Scale 3.2** 真实语料（P&P+莎翁，无标记） | **特化在真实数据成立**（coherence +0.224）；**controller 被噪声吞掉**（gap −0.24±3.25）|
| E13 | **Scale 3.3** 4B-surrogate（大 N 极紧预算） | gap 各档冲不出噪声；**关键副产品：k=1 ppl ≈ k=16 → 极稀疏更新不掉质量** |
| E14 | **覆盖诊断**（用覆盖指标而非 PPL 评 controller）| controller 在 PPL **和**覆盖上都不胜 random；根因：capacity 路由已均衡 → random 近最优 |

完整细节：[EXPERIMENTS.md](EXPERIMENTS.md)。

## 3b. Kaggle 真实 T4 跑（Phase B/C）

| 跑 | 配置 | 结果 |
|---|---|---|
| smoke | 4B，4min（先 P100 失败→强制 T4）| 4.156B、峰值 **7.52GB**、**6226 tok/s**、ppl 31→11.9、6/6 门 |
| v1 | lr 0.1，40min | test ppl 1788，**震荡发散** |
| v2 | lr 0.03 + warmup，40min | test ppl **1437**，单调稳定，6/6 门 |
| baseline-3h | lr 0.03，满 3h | 触底 1752@6000步 → **漂移回 2383**；test 1919。"**3h 反而变差**" |
| **Phase C** | lr 余弦衰减 + 路由冻结@5k + 早停 | **单调降到 1680，test ppl 1360**，route_drift=0，**7/7 门**，漂移消除 |

定论文档：[Phase-B baseline 定论.md](Phase-B%20baseline%20定论.md)、runs/kaggle/。

---

## 4. 发现的规律（深层结论）

1. **特化机制鲁棒可扩、真实数据成立**：内容路由（EMA-prototype 最近邻 + 容量限制）+ 局部规则 → 专家按内容特化（合成 purity 0.67、真实 coherence +0.40）。这是确定能赢的一半。
2. **"智能调度"在真实数据上不胜 random**：PPL 和覆盖两个指标都不胜。根因——**capacity 路由已把 forward 负载均衡，random 选已近最优，controller 无发挥空间**。
3. **真正的硬通货 = training-time *update* sparsity**：每步只更新 k_update≪N 个专家而质量不掉（标准 MoE 是 sparse forward + dense backward；本工作 sparse forward + **sparse learning**）。使能者是 **regime**（局部规则独立可跳 + capacity 路由 + basin 鲁棒），**不是聪明 controller**（random 选也成立）。
4. **指标错配 + basin 鲁棒**：PPL 测终点 basin、controller 作用在训练路径；`k=1≈k=16` 的 PPL 证明 basin 对"更新谁"高度鲁棒。
5. **信号按噪声分层**：coverage/deficit 是**精确计数（零噪、可辨识、扩到任意 N）**；leverage/learnability 是**噪声估计（大 N 排序不可辨识）**。解释了 v4 coverage 缩放有效、value 信号在 scale 上失效。
6. **稳定性规律**：lr 太高→震荡；无 lr 衰减/路由冻结→后期漂移（类灾难性遗忘）；**lr 余弦衰减 + 后期冻结路由 → 单调、无漂移、整个预算都有用**。

---

## 5. 最终定论（详见 [Phase-A 定论.md](Phase-A%20定论.md)）

**核心创新**：无 BP 约束下，内容路由 MoE + 逐专家局部规则可在**更新维度极度稀疏化**（k_update≪N）而不损质量 → 每步训练算力与总参数量解耦 → **4B 常驻在 T4/3h 可训，BP 不行**。是**可行性/效率**结论，非质量超越。

**被实验否定的假设**：H1 智能 controller 降 PPL（否）/ H2 优势在覆盖上显现（否）/ H3 加法式专家有用（否→须上关键路径）/ H4 value 信号在 scale 可用（否→不可辨识）/ H5 特化依赖标记/真实数据失效（否，正向）/ H6 controller 是稀疏训练使能机制（否→regime 才是）。

**controller 角色**：从"卖点"降级为**确定性 budget 调度器**——价值在 k_update 预算接口 + 可复现性 + 最坏 backlog 上界，**不是性能优化**。最终形态 = 确定性 deficit round-robin，value 信号去掉，只暴露 k_update。

**两条腿（Kaggle 叙事，都不依赖聪明 controller）**：① 零梯度特化 MoE 真能学；② 4B 常驻 + 极稀疏训练击败 BP memory/speed。

---

## 6. 代码与文件地图

**算法/实验代码（演进顺序）**
- `zerograd_nano.py` — Phase A 单层 nano 切片（E01–E08）：deeply-supervised 局部规则 + 重要性 controller + 数据/评测/日志。
- `zerograd_moe.py` — Stage 3/4 nano-MoE：EMA-prototype 内容路由 + 容量限制 + `Ctrl`（v3/v4/v5 controller）。
- `zerograd_moe_scale.py` — 多层堆叠 MoE（scale 3.1，per-layer 路由/专家/控制器，v4 coverage 自动缩放）。
- `zerograd_moe_nat.py` — 真实语料（3.2/3.3）+ 覆盖诊断（E14）；无标签特化度量（coherence）。
- `kaggle_zerograd_moe.py` / `.ipynb` — **≥4B Kaggle 提交**（Phase B/C，自包含，配置驱动 SMALL↔KAGGLE）。

**仪表盘**（自包含 HTML，读各自 dataX.js）
- `dashboard/index.html`（主）、`stageA.html`（特化）、`stage4.html`（控制器消融）、`scale.html`（可扩性）。

**数据 / 产物**
- `data/natural.txt`（真实语料子集）、`runs/`（本机）、`runs/kaggle/`（T4 产物）。

**设计 / 定稿文档**（思路与结论）
- [zero gradient.md](zero%20gradient.md) — 最初思考笔记（BP/PC/FF/LOCO/ADMM 拆解 + Dynamic Layerwise 设想）
- [项目路线与设计.md](项目路线与设计.md) — 初版路线与设计
- [EXPERIMENTS.md](EXPERIMENTS.md) — E01–E14 实验日志
- [阶段A设计.md](阶段A设计.md) — Phase A（MoE-faithful）锁定设计
- [Phase-A 复盘与校准.md](Phase-A%20复盘与校准.md) — 指标错配校准（PPL vs 路径）
- [Phase-A 定论.md](Phase-A%20定论.md) — 核心创新 / 被否假设 / controller 角色
- [Phase-B 说明.md](Phase-B%20说明.md) — Kaggle 运行说明
- [Phase-B baseline 定论.md](Phase-B%20baseline%20定论.md) — 原设定 baseline 实测定论
- [路线图 Phase C-E.md](路线图%20Phase%20C-E.md) — C/D/C.1/E 路线图

---

## 7. 复现 / 运行

```bash
# 本机逻辑验证（小配置，CPU，零 autograd，全部合规门）
python kaggle_zerograd_moe.py

# Kaggle T4（≥4B 真跑）：上传 kaggle_zerograd_moe.ipynb，GPU=T4，(可选)挂 WikiText-103，运行
# 自动化：kaggle kernels push（machine_shape="NvidiaTeslaT4"）→ status 轮询 → output 拉回
```

---

## 8. 当前状态与下一步

- ✅ Phase A（机制）、诚实定稿、Phase B（真实 T4 ≥4B 跑通、击败 BP 显存）、**Phase C（schedule 稳定化：漂移消除、test ppl 1360、7/7 门）**。
- 原设定（32k word-level 词表）已干净收口：**稳定、单调、test ppl ~1360**。
- **下一步（路线图）**：Phase D（BPE/更小词表 + 更强读出头 + 局部损失增强 → 更低/更可读 ppl）；Phase C.1（从稳定 best checkpoint 做后训练试点）；Phase E（混合 BP，长期研究）。
