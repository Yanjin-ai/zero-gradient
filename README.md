# 无反向传播的大模型学习，与"能力的架构性边界"

*Zero-Backprop Learning at Scale, and the Architecture of Capability — a controlled boundary study.*

> 一句话：在**单块 T4 GPU / 无全局反向传播**的约束下，我们训练了一个 **4.16B 参数**的语言模型（全程零 `autograd`，每步只用手写局部规则更新 0.42% 的专家），然后系统地测量它在关系推理、多步计算等现代 LLM 核心维度上的**能力边界**；再用一个**标准可训练注意力骨架 + 全反向传播**作为对照实验，证明这些边界**是架构导致的、不是任务不可学**——新骨架在真实 SNLI 上达到 **69.97%**（旧骨架为随机水平 33.4%），同时也暴露出它自己诚实的上限。

> 完整技术稿（英文）：[PAPER_DRAFT.md](PAPER_DRAFT.md)（~9–11pp）· 8 页精简 [PAPER_workshop.md](PAPER_workshop.md) · 幻灯 [slides_outline.md](slides_outline.md)｜真实实验数据 [results/](results/)｜锁定结论索引 [MASTER_ARCHIVE.md](MASTER_ARCHIVE.md)｜实验台账 [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md)｜决策记录 [docs/adr/](docs/adr/)。

---

## 1. 研究问题

两个正交的问题贯穿全程：

1. **在严格的单 GPU 预算下，一个完全不使用全局反向传播（BP）的语言模型能有多少能力？** 局部学习 / 无 BP 方法在内存和硬件受限场景很有吸引力，但它们在大规模下的能力天花板一直缺乏刻画。
2. **当这样的模型在某个任务上失败时，失败的根因是算法（梯度不够）、架构（骨架无法表示这种结构）、还是任务本身（不可学）？** 三者通常纠缠在一起、无法区分——这正是本项目要用受控实验拆开的。

## 2. 方法论

整个研究被组织成**三条互相隔离的实验线**，配合三条贯穿始终的纪律：

- **提交线** — 一个干净、确定性、纯 ZeroBP 的 4B 基线（§4），作为被研究的固定对象。
- **边界线** — 把这个骨架冻结成研究对象，测量"任务 × 方法"能力矩阵，逐步注入少量到深度的 BP（§5–6）。
- **控制线（Phase H）** — 一个严格隔离的**标准可训练注意力骨架 + 全 BP**，作为架构对照，回答边界线的失败到底是不是骨架造成的（§7）。

三条纪律：**① 公平读出** —— 能力一律用一个全新的、与任务无关的读出头来测量（我们两次发现：不公平的读出会凭空制造或抹掉表面的能力）；**② 单杠杆** —— 每个实验只改动一个结构自由度；**③ 诚实纠错** —— 我们自己的两个工作假设被后续实验推翻，报告里保留纠正而非最初的猜测（§8）。

## 3. 更早的机制研究：无 BP 训练为什么能成立（Phase A–C）

在做大模型之前，我们先在 nano 规模上确认机制、排除架构混淆（"toy 验证机制 → 真实数据看哪些 survive"）。关键发现：

- **内容路由 + 局部规则确实能产生专家特化。** EMA-prototype 最近邻路由 + 容量限制，使专家按内容分工（合成数据 purity 0.67 vs 随机 0.25；真实语料 coherence +0.40）。这是"能赢的一半"。
- **真正的硬通货是"训练时的更新稀疏"，不是"聪明的调度器"。** 标准 MoE 是 sparse forward + dense backward；本工作是 sparse forward + **sparse learning**：每步只更新 k_update ≪ N 个专家而质量不掉（**k=1 的 ppl ≈ k=16**）。我们原本押注一个"重要性 controller"能降 ppl，但它在 ppl 和覆盖率两个指标上**都不胜随机**——根因是容量路由已经把负载均衡，随机选已近最优。该假设被否，controller 降级为一个**确定性预算调度器**（价值在可复现性与最坏 backlog 上界，而非性能）。
- **训练稳定性有清晰规律。** lr 过高 → 震荡；无 lr 衰减 / 不冻结路由 → 后期漂移（类灾难性遗忘，"跑满 3h 反而更差"）；**余弦衰减 + 后期冻结路由 → 单调、无漂移、整个预算都有用**。

## 4. 系统：ZeroBP-4B（无反向传播的 4B 基线）

一个 decoder 式内容路由 MoE 语言模型。数据流（代码级事实）：

```
token x ─► E[x] + pos                                       (embedding；|V|=32000 BPE, d=1024)
        ─► 冻结随机因果注意力 (Wq,Wk 不训练)                    (单层"reservoir"注意力)
        ─► h = (emb + att·emb)[:, -1]     ← 末位塌缩：整条序列 → 一个 [B,d] 向量
        ─► 4× 堆叠 MoE block：内容路由(EMA-prototype+capacity) → 950 专家取 top-2
        ─► 每块深监督 2 层 MLP 读出头 + 确定性 round-robin 预算(每块每步更新 k_update 个专家)
```

`d=1024, |V|=32000, seq_len=64, 4 层, 950 专家/层, k_route=2, k_update=4` → **4.16B fp16 常驻参数**。三个性质贯穿后文：**(a) 无全局梯度** —— 全程 `set_grad_enabled(False)` 并断言，专家由局部规则更新；**(b) 确定性稀疏预算** —— 每步只更新 0.42% 的专家；**(c) 冻结注意力 reservoir + 末位塌缩** —— §5–6 指认的两个架构疑点。

**基线结果**：BPE + 2 层 MLP 头 + 路由冻结的余弦 schedule，在 WikiText-103 上 test perplexity ≈ **1391**（早停 ~54min）/ ≈ **1355**（跑满 ~2.9h 预算），比 unigram 好约 51%；峰值显存 ~8.3GB（同规模 BP 训练需 ~31GB → T4 直接 OOM），~8546 tok/s，确定性可复现。**核心价值是"4B 无 BP 可训练的可行性 + 系统效率"，以及它作为能力边界研究的干净对象。**

## 5. 结果一：ZeroBP 的能力边界（Phase E/F）

冻结骨架、只向 embedding / attention / 任务头注入 BP（不动骨架），用公平读出测量三种任务结构：

| 任务 | 结构 | 4B zero-shot | 4B 少量 BP | 小配置最好 |
|---|---|---|---|---|
| 情感 | bag 组合 | 60% | **79%**（embedding+头） | 100% |
| NLI | 关系对齐 | 33.4%（chance） | 33.4%（不转移） | 62% → 65.7%（深 BP） |
| 两步算术 | 多步计算 | 24.7% | ~19–21%（chance） | ~chance |

- **情感突破。** head-only 后训到 59.9%、近零遗忘；一点 embedding BP → **79%**（+3.0 ppl），且消融证明**embedding 才是杠杆**。
- **NLI 不转移。** 4B 上 zero-shot / emb-BP / emb+attn-BP **全是 chance**，即使 3000 步 + 高 attention lr 重试仍 chance。
- **多步不可安装。** 任何 BP 组件下都是 chance。三条"ZeroBP 原生"结构路线对 NLI 几乎无效：更丰富数据 +2.1pp、结构辅助目标 +0.5pp、attention 局部规则 ≈0。

## 6. 结果二：瓶颈在哪里？（Phase G，三个单杠杆探针）

- **不塌缩读出**：在同一冻结表示上换 mean-pool / all-positions，NLI **不升反降至 chance**（34.9% / 32.9%）→ **关系结构根本不在表示的任何位置**，换读出捞不出来。
- **可训练 attention 隔离**：冻结 embedding、只训 Wq/Wk，NLI **+0.0pp**（51.3→51.3；已验证 ‖ΔWq‖≈0.08 权重确有更新、‖ΔE‖=0 embedding 确冻）→ emb+attn 的 59.1% **全部来自 embedding**，attention 单独不是杠杆。
- **更深 BP**：让 MoE block 也参与 BP，小配置 NLI 抬到 65.7%，但**不转移到 4B**；多步算术在**任何深度**仍是 chance。

**边界结论**：对 bag 任务，ZeroBP + 少量 embedding BP 有效（79%）；对**关系**与**多步**这两个现代 LLM 核心维度，当前 ZeroBP-4B 架构**装不进结构**——它不在表示里、可训练 attention 单独无用、多步在任何 BP 深度都死。决定因素是：任务有多少落在 embedding（BP 可装），有多少需要通过冻结 block 的顺序计算 / 跨句注意力对齐（少量甚至大量 BP 都装不进）。**那么这是算法、BP 预算、还是骨架的问题？—— 控制线来回答。**

## 7. 结果三：架构控制实验（Phase H）

**设计与隔离**：一个刻意"普通"的标准 pre-LN Transformer —— 多头**双向**自注意力、GELU-MLP、残差、**mean-pool 读出**、**全反向传播（AdamW）**。严格隔离在 `phase_h/`（纯 PyTorch、零依赖 ZeroBP 代码；提交线永不 import 它）。在**同分布**的合成任务 + 真实 SNLI/GSM8K/SST-2 上复测。

- **G0（合成，回答"是不是骨架"）**：一个 **0.8M** 参数的 4 层模型，全 BP，把 ZeroBP 卡在 65.7%/chance 的同一合成 NLI 和两步算术都做到 **100%** → 边界是**架构性的，不是任务不可学**。
- **G1（真实 SNLI）**：12.4M 模型在真实 SNLI 上 **val 69.97%**（majority 33.8%），而 ZeroBP-4B 是 chance（33.4%）→ 新骨架在真实关系 benchmark 上进入现代小模型合格区间。
- **G2（多步深度）**：k=2 → **100%**，k=3 → 100%（本地 2.67M），但 **k≥4 是一堵抗规模的墙**：参数从 4.7M 扩到 **21.3M**、步数 6k→15k，k≥4 仍近 chance（21%）。
- **G2b / SST-2（诚实上限）**：小 char-LM 生成式解真实 GSM8K exact-match 仅 **2%**（预告的 stretch）；真实 SST-2 上 ZeroBP-4B 经读出诊断后确认 ≈ **chance**（公平 BP 探针 51.5→53.3%，majority 50.9%）——真实词汇情感远难于合成 bag 任务。

**最终 scorecard**：

| 维度 | Phase H（新骨架） | ZeroBP-4B（锁定） | 结论 |
|---|---|---|---|
| 合成 NLI + 算术 | 100% / 100% | 65.7%(小) / chance | 骨架是根本限制 |
| 真实 SNLI | **69.97%** | 33.4%（chance） | 关系结构可安装 |
| 多步深度 | k≤3=100%, k≥4 抗规模墙 | 任何深度 chance | 新栈有自身天花板 |
| 生成式 GSM8K | EM ~2% | — | 小模型 stretch 失败 |
| 真实 SST-2 | ≈chance（探针 53%） | ≈chance | 真实情感 ≫ 合成 bag |

## 8. 中途的发现与被推翻的假设

这项研究的一个特点是**多个自己的假设被数据推翻**，我们保留了纠正过程（完整见 [PAPER_DRAFT.md](PAPER_DRAFT.md) 附录 B）：

| 最初的说法 | 纠正 |
|---|---|
| 情感 4B ≈ 57% | 79%（原来是欠训的 SGD 读出头 → 公平闭式头） |
| "末位塌缩是瓶颈" | 证伪：结构在冻结表示的**任何位置**都不存在 |
| "可训练 attention 是真杠杆" | 证伪：attention 单独 +0.0pp，是 embedding 在扛 |
| SST-2（ZeroBP-4B）= 49% 平坦 | 读出坍缩 artifact；公平探针真值 ≈ 53% |
| k≥4 多步是"欠容量" | 证伪：5× 参数 + 2.5× 步数完全不动 → 真的是墙 |

**方法论教训**：其中两个"表面数字"是**读出的假象、不是表示的事实**（欠训的情感头；坍缩的 SST-2 头）——关于表示能力的论断必须用公平、任务无关的探针，最好用两个（闭式 + BP 线性）。

## 9. 结论

- **无 BP 的 4B 预训练成立，但有清晰、架构性的能力边界。** ZeroBP-4B 在单 T4 预算下对语言建模和 bag 式任务（ppl ≈1355–1391、情感 79%）确实有用，且是一个可控的边界测量对象；但关系对齐与多步计算在它上面**装不进**，且这**不是 BP 预算不够**——大量甚至全深度 BP 在 4B 上仍失败。
- **梯度必要但不充分：它需要一个可训练的结构底座。** 同样的任务在一个可训练注意力骨架 + 全 BP 上被轻易装入。所以 BP 只在架构给结构留出可训练位置时才有用。**"ZeroBP 做不了 NLI"被改写成："任何 BP 注入一个冻结 reservoir + 末位塌缩的骨架都装不进关系结构；一个标准注意力骨架轻松做到。"**
- **任务结构决定难度序，好骨架也有诚实上限。** 两条栈上的难度序一致（bag ≪ 关系 ≪ 多步深度 ≪ 真实生成式数学）。Phase H 跨过了关系和浅多步，但它的 **k≥4 深度墙是抗规模的**，提示深层顺序推理需要的不只是"骨架 + 规模"，而是 curriculum / 中间监督 / chain-of-thought（本项目未测，是最可能的正道）。

**未来方向**：用 curriculum / chain-of-thought（而非纯扩规模）打破 k≥4 墙；把 SNLI 扩到 MNLI、Phase H 上规模；在**可训练注意力底座**上验证更先进的 ZeroBP 局部规则（两条线合流）；在保持零 autograd 纯净的前提下继续提升 ZeroBP 的 LM 质量。

---

## 10. 仓库导航

**对外成果（研究报告 + 展示）**
- [PAPER_DRAFT.md](PAPER_DRAFT.md) — 完整英文技术稿（~9–11pp，三线结构 + scorecard + 附录）。
- [PAPER_workshop.md](PAPER_workshop.md) — 8 页精简 workshop 版。
- [slides_outline.md](slides_outline.md) — 11 页幻灯提纲。
- [results/](results/) — 真实 GPU 实验数据（SNLI / 多步深度 / GSM8K / SST-2 诊断 的原始 JSON + 索引）。
- [runs/track1_radar.png](runs/track1_radar.png) — 能力雷达图。

**研究档案与治理**
- [MASTER_ARCHIVE.md](MASTER_ARCHIVE.md) — 锁定结论的双语索引（【FACT】/【INTERPRETATION】/【PROPOSAL】分层）。
- [项目总档案.md](项目总档案.md) — 中文深度总档。
- [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) — 每个实验的配置 · commit · 指标 · gate · 结论。
- [ARCHITECTURE.md](ARCHITECTURE.md) — v1.0（固定骨架）/ v2.0（可改区）边界；[docs/adr/](docs/adr/) — ADR-001…005 决策记录；各 `Phase-*.charter.md` — 阶段章程。
- [ENGINEERING.md](ENGINEERING.md) — 工程规范（reset/clone 纪律、selfcheck、隔离红线）。

**代码**
- `kaggle_zerograd_moe.py` — ZeroBP-4B 主实现（配置驱动 SMALL↔4B，纯 ZeroBP）。
- `zerograd_nano.py` / `zerograd_moe*.py` — Phase A 机制研究（nano → 多层 → 真实语料）。
- `phase_e*.py` / `phasee_nli_4b.py` / `f1_data.py` / `f2_aux.py` / `h1_attn.py` / `v2_readout.py` / `v2_attn.py` / `v2_deepbp.py` / `task_*.py` — 边界线研究脚本（含少量 BP，默认 off，与提交隔离）。
- `phase_h/` — 控制线新骨架栈（`ph_base.py` Transformer + 各 G0/G1/G2/G2b 脚本），零依赖提交代码。
- `track1_sst2_4b.py` / `track1_radar.py` — Track 1 真实 SST-2 探针与能力雷达。

**复现（本机，无 GPU）**
```bash
python3 kaggle_zerograd_moe.py     # ZeroBP 小配置：零 autograd、确定性、合规门（final_ppl 6.251）
python3 phase_h/ph_nli.py          # Phase H G0：合成 NLI → 100%（对照 ZeroBP 的边界）
python3 track1_radar.py            # 生成能力雷达图
```

> Kaggle 提交与平台细节见 [SUBMISSION.md](SUBMISSION.md) / [KAGGLE_README.md](KAGGLE_README.md)（与本研究报告解耦）。
