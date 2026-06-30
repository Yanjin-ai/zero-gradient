# ADR-004: 开放 v2.0（允许动结构归纳偏置）

- **Status**: Proposed (2026-06-30) — PROPOSAL（待 v2.0 小配置验证后转 Accepted）
- **Context**: ADR-002 锁定"关系/多步是 v1.0 架构的真实上限"。Phase F 发现疑似真因 = **末位塌缩**（`context()` 把序列塌成单 [B,d] 向量，丢跨位置信息）。靠 v1.0 骨架 + 少量顶层 BP 已无法靠拢现代 LLM 的关系/多步能力。
- **Decision（提案）**:
  - 新开 **v2.0 研究层**，允许动**三类结构归纳偏置**（见 `ARCHITECTURE.md §3`）：① **不塌缩的序列读出**（pooling/多位置/注意力池化，**首选**：最便宜最对症）；② **真正可训练 attention**；③ 更深 BP。
  - **硬约束**：v2.0 一律 **research-only / non-submission**，新建独立路径，**默认 off**，**永不破坏** `kaggle_zerograd_moe.py` 默认 `6.251` / 零 autograd；单杠杆 + 小配置先行 + 过门才迁 4B；结果标 research-only、写 ledger。
  - 首个 v2.0 实验（Phase G 首线）：小配置测**不塌缩读出**能否松动 NLI/算术 zero-shot。
- **Consequences**: 给"靠拢现代 LLM 关系/多步能力"一条架构级路径，同时**完全隔离**提交版与 v1.0 锁定结论。若 v2.0 不塌缩读出在小配置松动关系任务 → 本 ADR 转 Accepted 并定义 v2.0 提交策略；若无效 → 记负结果，关系/多步限制升级为"架构级硬墙"。
