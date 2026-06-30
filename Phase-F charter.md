# Phase F Charter — 向现代 LLM 能力光谱靠拢（分层：严格 ZeroBP + 受控 Hybrid BP）

> **状态**：charter（2026-06-30）。Phase E 已锁定能力边界（见 `项目总档案.md` §1.5🔒）。Phase F 不放弃 ZeroBP，而是把它从"能在 T4 训出 4B LM"推进到"在关键结构能力上向现代成熟 LLM 靠拢"。
> **纪律不变**：small-config 先行 → 过 gate → 才迁 4B；每实验记录 + 推 GitHub；reset 必 clone、多 reset 必过 `selfcheck.py`、4B 单发隔离。

## 1. 出发点（Phase E 锁定结论）
ZeroBP 4B backbone 在 LM 上成立；少量 BP（embedding 路径）是唯一能突破 ZeroBP 后训练天花板的机制；但能力边界由**任务结构复杂度**控制——bag 组合可突破（情感 4B 79%）、关系对齐很难（NLI 4B chance，3000 步+attn 重试仍 chance）、多步计算不可安装（算术连小配置都失败）。**根因**：关系对齐结构（attention）与顺序计算结构（blocks）没在预训练表示里形成，少量 post-hoc BP 装不进。→ **必须升级预训练目标 / attention 训练方式 / 少量 BP 介入策略**。

## 2. Phase F 总目标
在**当前 ZeroBP 4B 栈 + Kaggle T4/3h 约束内**，分两层并行推进：
- **第一层（严格 ZeroBP 预训练加强）**：不引入全局 BP，提升 backbone（尤其 blocks 与 attention 周边）的句子级 / 长程 / 跨段结构表达，延续 ZeroBP 作为核心算法创新。
- **第二层（Hybrid 受控 BP）**：保持 ZeroBP backbone 主体不变，**仅在 attention 或少量顶层结构**上引入受控 BP，补齐目前 ZeroBP 难形成的关系对齐 / 高层结构。

> **能力对标，非规模对标**：不复刻工业预训练规模；对标其**能力形成路径**（更丰富上下文分布 + 让 attention/高层学到跨句结构 + 用 NLI/QA/推理小任务当能力温度计）。

## 3. ZeroBP 与 Hybrid 的边界（写进所有文档，硬约束）

**严格 ZeroBP（禁止全局 autograd / 端到端 BP）**：

| 模块 | 原因 |
|---|---|
| 主干 MoE-FFN 专家更新 | 核心算法创新，局部规则主导 |
| EMA routing / training-time sparsity | 资源优势与训练范式核心 |
| 大部分底层 backbone blocks | 维持"ZeroBP backbone"身份与资源可控 |
| 绝大部分预训练 token LM 主循环 | 确保主线是 ZeroBP 预训练而非 disguised BP |

**允许 Hybrid BP（唯一试验区）**：

| 模块 | 允许方式 | 目的 |
|---|---|---|
| 顶层 attention（Wq/Wk，必要时 Wo） | 少量 / 周期性 BP | 跨句对齐、长程引用、结构依赖 |
| 顶层 1–2 个 blocks | 少量 BP / 局部梯度近似 | 更强句子级表示与任务结构 |
| 任务头 / 读出头 | BP 或闭式 | 任务适配 + 公平评测 |
| 后训练中的 embedding 路径 | 少量 BP | Phase E 已验证有效，保留 |

> **红线**：若上述"严格 ZeroBP"被大量 BP 替代，主叙事会从"ZeroBP 4B 预训练栈"滑向"普通混合训练模型"。Hybrid 只能动 attention + 极少量顶层。

## 4. 第一层：严格 ZeroBP 预训练加强册（F1–F3）

**F1-data（数据结构增强，不改算法）**：从 WikiText 式 token LM 扩到更"文档级"分布——更长段落/多段、对话型/问答型/网页段落型；分桶（普通 LM / 长上下文 / 结构化对话）。硬条件：可清洗、Kaggle/本地可读、长度与 batch 组织不显著抬高显存峰值。成本在数据准备，非算法。

**F2-aux-zeroBP（结构辅助目标，仍 ZeroBP）**：在 token LM 主目标旁加轻量结构信号，但必须**局部读出 + 局部更新**（不做全局监督分支）：段内相邻句一致性、简化句对匹配、局部对比式句子级信号。让 backbone 在 ZeroBP 训练中逐步获得句子级几何。

**F3-attn-zeroBP（attention 的 ZeroBP 局部规则增强）**：Phase F 第一大缺口。为 Wq/Wk 设计更稳定的局部更新规则；对 attention 输出引入局部监督信号再反馈 Wq/Wk。**优先级高**（不破坏 ZeroBP 主叙事，却可能直接补最弱维度）。小配置指标：单调、确定、route drift 不恶化、句对结构小任务 zero-shot 提升。

## 5. 第二层：Hybrid 少量 BP 介入册（H1–H3）

**H1-attn-hybrid-pretrain（核心实验）**：大部分 backbone/MoE/路由/token LM 主循环仍 ZeroBP；仅顶层 attention（最后 1–2 层）周期性少量 BP，来源仍是 LM 目标，只更新顶层 attention 参数；**只对注意力参数建 autograd 图，不扩散到 4B 主干**（显存可控）。测：少量 attention BP 是否让 backbone 获得更强句子级/跨句结构。

**H2-topblock-hybrid（若 H1 不够）**：再开放顶层 1 个 block 对 LM 读出相关高层表示做少量 BP，其余 blocks/专家仍严格 ZeroBP。增强句子级语义汇聚。

**H3-PhaseE-2.0（检验环节）**：在更强的 Phase F backbone 上重跑后训练矩阵——情感（79% 是否保持/上升）、NLI（zero-shot 是否脱离 chance、少量 BP 是否开始生效）、算术（是否出现基础 zero-shot 信号）。若无改善→预训练增强仍不对症。

## 6. 执行顺序与 gate
**顺序**：F1-data → F2-aux-zeroBP → F3-attn-zeroBP → H1/H2-hybrid-pretrain（Hybrid 放最后）。
**进 4B 的门槛**（每阶段）：① 小配置单调 / 确定 / 无 reset 污染（过 `selfcheck.py`）；② 小配置在 ≥1 个句对/结构任务上提升 zero-shot 或 Mixed-BP 上限；③ 4B 资源仍在 T4/3h 内。

## 7. 后训练册（Phase E 2.0，固定矩阵）
每个新 backbone 版本都重跑**固定三类任务 × 四路线**，看"是否把更多任务从无信号/不可安装推进到有信号/可安装"（而非追单任务偶然涨分）：
- 任务：bag 组合（情感 XOR）/ 关系对齐（NLI）/ 多步计算（2 步算术）。
- 路线：Zero-shot head / ZeroBP adapt / Mixed-BP(embedding+头) / Mixed-BP(embedding+attention)；必要时 +top block、+更长 BP budget、闭式 vs SGD 头公平读出。
- 脚本已就位：`task_*.py`（小配置）、`c1_4b.py`/`phase_e_4b.py`/`phasee_nli_4b.py`（4B）、`orchestrate_kaggle.py`（5 stage）。

## 8. 第一条实验线：F1-data（已跑，gate 失败）
`f1_data.py`：小配置，严格 ZeroBP，只改预训练**分布**（random vs richer = random + 一致关系句对 + QA 式），测 NLI zero-shot。

| 预训练语料 | NLI zero-shot |
|---|---|
| random（当前基线） | 49.1% |
| **richer（一致关系对 + QA）** | **51.3%（仅 +2.1pp）** |

**【事实】数据分布单独不够**（+2.1pp < 5pp 门）。加一致关系句对**不能**让 ZeroBP 表示形成关系几何 → **瓶颈在学习机制（attention），不在数据**（呼应 NLI 4B chance + Track A）。**gate 失败，不迁 4B。**

## 9. 修正后的下一步（F1 把矛头指向 attention）
F1 排除了"只要更好数据"。**下一步直接攻 attention**（charter 已列为高优先）：
- **F3-attn-zeroBP**：给 Wq/Wk 设计 ZeroBP 局部规则（对 attention 输出引入局部监督再反馈），小配置看 NLI zero-shot 是否脱离 49%；
- **H1-attn-hybrid**：若 ZeroBP 局部规则仍不够，对顶层 attention 引入**少量周期性 BP**（已知 Phase E 的 emb+attn BP 在小配置 NLI 到 62%——但那是后训练；H1 是**预训练阶段**让 attention 形成结构，使 zero-shot 抬升）。
- F2-aux（结构辅助目标）可与 F3 并行或其后。
