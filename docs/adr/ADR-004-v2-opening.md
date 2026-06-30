# ADR-004: 开放 v2.0（允许动结构归纳偏置）

- **Status**: Proposed → **CLOSED**（2026-06-30）— v2.0 三类结构杠杆（不塌缩读出、可训练 attention 隔离、更深 BP）已**全部小配置实测**：读出/attention 证伪；更深 BP 关系有小配置信号但不转 4B、多步任何深度仍 chance。在当前骨架上无突破 4B 路径。**未转 Accepted；突破须换骨架（Phase H / 新 ADR）。** 见文末三条更新。
- **Context**: ADR-002 锁定"关系/多步是 v1.0 架构的真实上限"。Phase F 发现疑似真因 = **末位塌缩**（`context()` 把序列塌成单 [B,d] 向量，丢跨位置信息）。靠 v1.0 骨架 + 少量顶层 BP 已无法靠拢现代 LLM 的关系/多步能力。
- **Decision（提案）**:
  - 新开 **v2.0 研究层**，允许动**三类结构归纳偏置**（见 `ARCHITECTURE.md §3`）：① **不塌缩的序列读出**（pooling/多位置/注意力池化，**首选**：最便宜最对症）；② **真正可训练 attention**；③ 更深 BP。
  - **硬约束**：v2.0 一律 **research-only / non-submission**，新建独立路径，**默认 off**，**永不破坏** `kaggle_zerograd_moe.py` 默认 `6.251` / 零 autograd；单杠杆 + 小配置先行 + 过门才迁 4B；结果标 research-only、写 ledger。
  - 首个 v2.0 实验（Phase G 首线）：小配置测**不塌缩读出**能否松动 NLI/算术 zero-shot。
- **Consequences**: 给"靠拢现代 LLM 关系/多步能力"一条架构级路径，同时**完全隔离**提交版与 v1.0 锁定结论。

## 更新（2026-06-30，Phase G 首线结果）
- **【LOCKED FACT】不塌缩读出子方向被证伪**（`v2_readout.py`）：在冻结 ZeroBP base 上换 mean/all-pos/concat 读出，NLI zero-shot 不升反降（mean/all-pos 即 chance）→ **关系结构不在冻结表示，读出不是杠杆**。
- **【INTERPRETATION】** 真瓶颈 = **冻结 attention 不形成对齐** → v2.0 的真杠杆调整为 **"真正可训练 attention"**（首选），不塌缩读出降级。
- **Status 仍为 PROPOSAL**：v2.0 空间仍开放，但首个子实验（读出）已否；下一个 v2.0 实验 = **可训练 attention 小配置**（看 NLI zero-shot 是否脱离 49%）。该实验确认方向后再把本 ADR 转 Accepted。

## 更新（2026-06-30，Phase G 第二线结果 — 可训练 attention 隔离）
- **【LOCKED FACT】可训练 attention 子方向也被证伪**（`v2_attn.py`）：同 F1 base + 公平新鲜闭式头 + NLI 标签 CE，**冻结 embedding、只训 Wq/Wk**：51.3%→**51.3%（+0.0pp）**（已验证 Wq/Wk 真更新、embedding 真冻结）。参考臂复现锁定 H1：emb-only 58.8% / emb+attn 59.1%。→ **隔离后可训练 attention 单独零作用；59.1% 全部来自 embedding**。上面那条"真杠杆=可训练 attention"的 INTERPRETATION 被**自身实验否掉**。
- **Decision 变更**：v2.0 的三类结构杠杆中，**读出 + 可训练 attention 两类已实测证伪**；仅剩**更深/更多层 BP** 未测，但先验偏弱（embedding 已是关系/组合主导且不转移 4B）。
- **Status → REJECTED（部分）/ 降级**：本 ADR 提出的"动结构归纳偏置即可靠拢现代 LLM 关系能力"假设，在前两个最对症的子方向上**未通过验证**。ADR-002（关系/多步是真实能力边界）由此**进一步坐实**。v2.0 不再"首选 attention"；若要继续，需以**更深 BP** 或**换骨架（多层真注意力 base）** 立新 ADR，而非在冻结 reservoir 上打补丁。

## 更新（2026-06-30，Phase G 第三线 = 更深 BP，v2.0 三类杠杆测完收口）
- **【LOCKED FACT】第三类 v2.0 杠杆（更深 BP）已测**（`v2_deepbp.py`，EXPLORATION）：放开 BP 到全 block 深度。**关系 NLI**：49.1→**65.7%**（深 BP 真有信号，小配置上限是 BP 深度的函数）；**多步算术**：任何深度仍 **19–21% chance**（不可安装）。细节并入 ADR-002 更新。
- **Decision 收口**：本 ADR 列的**三类结构杠杆全部测完**——读出（证伪）、可训练 attention（+0.0pp 证伪）、更深 BP（关系小配置有信号但不转 4B、多步任何深度 chance）。**在当前骨架（末位塌缩 + 单冻结 attention + ZeroBP MoE）上加结构/BP 已无突破 4B 的路径。**
- **Status → CLOSED**：v2.0（"在现有骨架上动归纳偏置/BP"）作为靠拢现代 LLM 关系/多步能力的路线**结束**。真要突破须**换骨架**：新立 **ADR-005 / Phase H（多层可训练注意力 + 非 reservoir base）**，或接受边界、转向巩固已成立能力（情感 79% / LM 1355）。
