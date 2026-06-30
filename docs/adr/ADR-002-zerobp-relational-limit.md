# ADR-002: ZeroBP 的关系/多步结构上限（锁定）

- **Status**: Accepted (2026-06-30) — LOCKED 🔒（不得改写）
- **Context**: NLI（关系对齐）4B zero-shot = chance（33.4%），大预算重试 3000 步 + attn lr 0.5 仍 chance（commit `ef26ef0`）。需判定这是"欠训"还是"结构性缺失"，并决定 ZeroBP 路线是否还值得投入。
- **Decision（基于实测，LOCKED FACT）**:
  - 三条**纯 ZeroBP** 路线对"把关系几何装进表示"都失败：F1 更丰富数据 **+2.1pp**（commit `c03db5d`）、F2 结构辅助目标 **+0.5pp**（commit `dd9fa31`）、Track A attention 局部规则（借 readout 信号）失败（commit `0ff347f`）。→ **ZeroBP 通往关系结构的路线判定穷尽。**
  - 少量 BP **能部分装入**：H1 小配置 NLI zero-shot 51.3%→**58.8%（embedding）**，attention 仅 +0.3pp（公平新鲜头读出，commit `dd9fa31`）；**墙的本质是 BP-vs-ZeroBP**。
  - 但少量 BP **有限且不转移 4B**（小配置 59% < 65%；4B NLI 仍 chance）。
- **Consequences**: 关系/多步对齐是**当前架构（v1.0）的真实上限**，非调参问题。要突破需 v2.0 结构归纳偏置（见 ADR-004）。**【不得用更多调参/更长 BP 声称推翻本结论，除非在 v2.0 架构上重测。】**
