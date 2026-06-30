# ADR-002: ZeroBP 的关系/多步结构上限（锁定）

- **Status**: Accepted (2026-06-30) — LOCKED 🔒（不得改写）
- **Context**: NLI（关系对齐）4B zero-shot = chance（33.4%），大预算重试 3000 步 + attn lr 0.5 仍 chance（commit `ef26ef0`）。需判定这是"欠训"还是"结构性缺失"，并决定 ZeroBP 路线是否还值得投入。
- **Decision（基于实测，LOCKED FACT）**:
  - 三条**纯 ZeroBP** 路线对"把关系几何装进表示"都失败：F1 更丰富数据 **+2.1pp**（commit `c03db5d`）、F2 结构辅助目标 **+0.5pp**（commit `dd9fa31`）、Track A attention 局部规则（借 readout 信号）失败（commit `0ff347f`）。→ **ZeroBP 通往关系结构的路线判定穷尽。**
  - 少量 BP **能部分装入**：H1 小配置 NLI zero-shot 51.3%→**58.8%（embedding）**，attention 仅 +0.3pp（公平新鲜头读出，commit `dd9fa31`）；**墙的本质是 BP-vs-ZeroBP**。
  - 但少量 BP **有限且不转移 4B**（小配置 59% < 65%；4B NLI 仍 chance）。
- **Consequences**: 关系/多步对齐是**当前架构（v1.0）的真实上限**，非调参问题。要突破需 v2.0 结构归纳偏置（见 ADR-004）。**【不得用更多调参/更长 BP 声称推翻本结论，除非在 v2.0 架构上重测。】**

## 更新（2026-06-30，v2.0 更深 BP 重测 — 本 ADR 的 line-9 授权探针）
> 锁定 Decision **不改写**；以下为该结论的**补强 + 小配置细化**，均在本骨架内、未重测 4B。`v2_deepbp.py`（EXPLORATION）。
- **【LOCKED FACT】多步（算术）补强**：放开 BP 到**全 block 深度**（emb+所有 block experts+attn），2 步算术仍 **19–21%（chance）**，适配反而抹掉 zero-shot 24.7% 的微弱优势。→ **多步计算在当前骨架下任何 BP 深度都不可安装**——比关系更硬，原结论**加强**。
- **【LOCKED FACT】关系（NLI）细化**：小配置 NLI 随 BP 深度上升——floor 49.1 → 浅 emb 57.9 → **全深 65.7%**。→ 小配置那句"少量 BP 有限（~59%）"是**浅 BP 的下估**；小配置关系上限是 **BP 深度的函数**。**但不改 4B 头条**：NLI 4B BP 已锁 chance（commit `ef26ef0`），深 BP **未重测 4B**、转移先验**弱**。
- **净判定**：本 ADR 的"关系/多步是当前架构真实上限、4B 不可突破"**仍成立**；细化为——多步=任何深度不可装；关系=小配置可深 BP 装一点但不转 4B。**突破仍需换骨架（Phase H，见 ADR-004 收口）。**
