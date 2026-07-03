# ADR-005: 开 Phase H / v3.0 新骨架栈（多层可训练注意力 + 开放 BP）

- **Status**: Proposed (2026-07-01) — PROPOSAL（小配置原型验证中）
- **Context**: ADR-002 锁定"关系/多步是当前 ZeroBP 骨架（末位塌缩 + 单冻结 attention + ZeroBP MoE）的真实能力上限"；ADR-004 已 CLOSED——v2.0 三类结构杠杆（读出 / 可训练 attention / 更深 BP）在**当前骨架内**全部测完，无突破 4B 路径。要在 NLI / GSM8K 这类现代 LLM 核心维度上跑出**对照小模型**，必须换骨架，而非继续在 reservoir 上打补丁。
- **Decision（提案）**:
  - 新开 **Phase H / v3.0** = 一套**全新、标准的多层可训练注意力 base**（GPT/BERT 风格），独立目录 `phase_h/`，**完全 research-only**。三层设计见 `Phase-H charter.md`：① 架构层（多层多头 self-attention + mean/CLS pooling，**抛弃末位塌缩**）；② 训练层（**开放 BP**：整 base 标准 BP 预训练 + 下游可调 BP 深度）；③ 数据层（更丰富预训练 + 面向 NLI/GSM8K 的下游数据）。
  - **目标不是追平 GPT-4**，而是在现代 benchmark 维度（LM/NLI/GSM8K/GLUE）上给出一个**可比较的小模型行为**，与"ZeroBP 4B + 少量 BP"的能力结构做对照（论文用）。
- **硬隔离约束（红线，与 ADR-003 提交完整性并列）**:
  1. **不碰提交线**：`phase_h/` **不 import** `kaggle_zerograd_moe.py`，提交线**不 import** `phase_h/`。提交默认路径仍 `6.251` / 零 autograd / 7 门不变。
  2. **物理分层**：Phase H 全部代码/产物在 `phase_h/`（或未来独立 repo）；与 v1.0/v2.0 文件不混。
  3. **单独命名**：Phase H checkpoint/kernel 用 `ph_*` 前缀，绝不覆盖 1355/6.251 提交产物。
  4. **小配置先行**：先在小配置 + 合成任务（与 ZeroBP 矩阵同分布）验证骨架有没有"腿"，再决定是否上 GPU/真实数据/更大规模。
- **Consequences**: 给"靠拢现代 LLM 关系/多步能力"一条**架构级**路径，同时把它与已锁定的 ZeroBP 4B 提交线、v1.0/v2.0 研究结论**完全隔离**。Phase H 的成败**不回改** ADR-002/004 的任何锁定结论——它是另一条栈的独立故事。

## 计算现实（must know）
- 本地 = **CPU-only**（无 CUDA）。可本地跑：小配置 Phase H 原型 + 合成任务（NLI/算术，与 ZeroBP 矩阵同分布）。
- **需 GPU/Kaggle**（later，按现有 4B 流程）：真实 SNLI/MNLI/GSM8K、文档级预训练、几百 M~B 级规模。charter 里规划，不在本地跑。

## 首个验证（pending → 见 ledger / charter 更新）
小配置原型 `phase_h/ph_nli.py`：标准多层双向 attention + 全 BP，在**与 ZeroBP 完全同分布**的合成 NLI 上，能否明显超过 ZeroBP 骨架的小配置上限（深 BP 65.7% / 4B chance）？若能（预期 →~95-100%），即证明**瓶颈是骨架不是任务** → 本 ADR 方向确认。
