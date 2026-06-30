# ADR-001: Phase F 的 ZeroBP / Hybrid 边界

- **Status**: Accepted (2026-06-30) — LOCKED
- **Context**: Phase E 锁定"少量 BP（embedding）是唯一突破 ZeroBP 后训练天花板的机制"。Phase F 要向现代 LLM 能力靠拢，必须决定**哪些保持严格 ZeroBP、哪些允许 BP**，否则主叙事会从"ZeroBP 4B 栈"滑向"普通混合训练模型"。
- **Decision**:
  - **严格 ZeroBP（红线，禁全局 autograd/端到端 BP）**：主干 MoE-FFN 专家更新、EMA routing / training-time sparsity、大部分底层 blocks、绝大部分预训练 token LM 主循环。
  - **允许 Hybrid BP（唯一试验区）**：顶层 attention（Wq/Wk，必要时 Wo）、顶层 1–2 个 blocks、任务/读出头、后训练 embedding 路径。
  - 执行顺序：F1-data → F2-aux-zeroBP → F3-attn-zeroBP → H1/H2-hybrid（Hybrid 放最后）；小配置过门才迁 4B。
- **Consequences**: ZeroBP 算法身份与资源效率得以保持；Hybrid 仅在 attention+少量顶层试验。**若上述严格-ZeroBP 部分被大量 BP 替代，即违反本 ADR。** 详见 `Phase-F charter.md`。
