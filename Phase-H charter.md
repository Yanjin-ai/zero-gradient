# Phase H charter — 新骨架栈（v3.0）+ 当前骨架展示线（两条并行路）

> **定位**：v2.0 在当前 ZeroBP 骨架上的结构探索已 CLOSED（ADR-004）。本 charter 开两条**明显分层、互不污染**的并行路。治理见 [ADR-005](docs/adr/ADR-005-phase-h-new-backbone.md)。标注 **【LOCKED FACT】/【INTERPRETATION】/【PROPOSAL】/【EXPLORATION】**。最后更新 2026-07-01。
>
> **铁律**：两条路都**完全 research-only**，**永不污染** ZeroBP 4B 提交线（`kaggle_zerograd_moe.py` 默认 `6.251` / 零 autograd / 7 门不变；`phase_h/` 与提交线互不 import）。

---

## Track 1 — 当前骨架（ZeroBP 4B + 少量 BP）：对齐现代 LLM 的轻量评测与展示
**不改架构、不大改训练**，只在现有栈上做评测/demo + 把关系/多步边界做成对照证据。

### 1.1 LM / 情感 / 简单理解（展示线）
- **LM**：在 WikiText（已有 1355/1391）+ 一个小 open corpus 上报 ppl；与同参数量 BP Transformer 给个**段位区间**（不追平，说明"资源受限 + ZeroBP"处于哪一档）。
- **情感/分类**：现有合成组合情感（79% 4B / 100% 小）之外，加 1–2 个公开小数据集（SST-2 类）；跑 ZeroBP baseline + Mixed-BP(embedding)，看能否稳定到现代小 LLM 合理水平（~80-85%）。
- **简单理解**：一个轻量单句/句对任务，给 Track 2 的 NLI 做对照基线。
- **产出**：写 `EXPERIMENT_LEDGER` + `MASTER_ARCHIVE`；README "当前技术状态"给简短总结 + 能力雷达图。
- **计算**：公开数据集 + 4B 评测多数需 **GPU/Kaggle**（按现有 kernel 流程）；本地可先做合成任务与脚手架。

### 1.2 继续攻 NLI/多步 = 明确【EXPLORATION】（边界证据，非主路）
- 架构不变（末位塌缩 + 单冻结 attention + ZeroBP MoE）。
- 在小配置 + 4B 上加更多结构任务（更复杂 NLI 变体 / 简单链式推理 / 不同长度算术），对每个重复矩阵：zero-shot / ZeroBP adapt / Mixed-BP(emb) / Mixed-BP(emb+attn) / 更深 BP(`v2_deepbp`)。
- 整合成一张大表：**哪种结构任务"嵌一点 BP 就能装"，哪种"怎么 BP 都不动"** → 为 Phase H 提供对照与动机。

---

## Track 2 — Phase H 新骨架栈（v3.0）：多层可训练注意力 + 开放 BP + 更丰富数据
专攻 NLI / GSM8K 等现代 LLM 核心维度，目标 = **可比较的对照小模型**（非追平规模）。独立目录 `phase_h/`。

### 2.1 架构层 — 多层可训练注意力 base
- GPT/BERT 风格标准 Transformer：多层 self-attention（先 4 层小配置 → 12/24 层按资源）、多头（4 → 8-16）、pre-LN、residual、FFN(4×)、GELU。
- **读出**：**抛弃末位塌缩** → mean-pool / CLS / attention-pool（句对/段落表示）。
- 规模：先**几百 M 以内**小配置验证能力，再决定是否上 B 级。
- 文件：`phase_h/ph_base.py`（**纯 torch、零依赖 ZeroBP 模块**，可整体迁出到独立 repo）。

### 2.2 训练层 — 开放 BP（不再节省到极限）
- 预训练：标准 BP（AdamW）在 LM/MLM 目标上训练**整个 base**（attention+FFN）。
- 下游：NLI/GSM8K 上做 BP 微调；设 **BP 深度阶梯**（只 head / 顶几层 / 全模型），记录每档能力 → 与 Phase F/G 的"少量 BP/更深 BP"形成跨栈对照。

### 2.3 数据层 — 更丰富预训练 + 面向下游结构
- 预训练：文档级文本 + 对话/QA + 部分 GSM8K 风格算术解释文本。
- 下游：NLI（SNLI/MNLI 或轻量版）；多步 = **G2 核心 = 可扩展多步算术深度扫描**（classification，与 ZeroBP 矩阵同分布的直接延伸）。
- **【诚实边界】G2 vs G2b**：**G2（本 scaffold）** = 控制的多步算术，n_steps 递增——ZeroBP k=2 就 chance，看 Phase H 能装到多深，**可学、可对照**。**G2b（stretch，未做）** = 真实自然语言 **GSM8K**：生成式 + 需 causal-LM 变体 + 预训练/规模；一个从零小模型 **不会**直接刷出好成绩——**不伪装成小模型的胜利**，留作独立生成式栈。
- 合成对照：先用**与 ZeroBP 矩阵完全同分布**的合成 NLI/算术，确保跨栈 apples-to-apples。

### 2.4 门槛 / Gates
- **G0（本地，首验）**：小配置 Phase H base + 全 BP 在**合成 NLI（同分布）**上 **明显超过** ZeroBP 小配置上限（深 BP 65.7%），预期 →~95-100% → 证明"瓶颈是骨架不是任务"。**过 G0 才值得上 GPU。**
- **G1（GPU）**：真实 SNLI/MNLI 上跑出现代小模型合理水平。
- **G2（GPU）**：GSM8K 部分题上有非平凡正确率（多步计算这条 ZeroBP 完全装不进的维度）。

---

## 三、推进顺序 & 计算现实
1. **本地现在**：Track 2 的 **G0 小配置原型**（`phase_h/ph_nli.py`，CPU 可跑）；Track 1 的归档/脚手架。
2. **GPU/Kaggle later**：Track 1 的公开数据集评测 + Track 2 的真实数据/规模（G1/G2）。**本地 CPU-only，跑不了真实大模型**——charter 规划，不本地跑。
3. **论文对齐**：两条线都用现代 benchmark 维度（LM/NLI/GSM8K/GLUE）定位，对齐的是**能力结构**而非规模——"资源受限 ZeroBP 4B + 少量 BP" vs "标准多层注意力 base + 全 BP"。

## 四、隔离自检（每次 Phase H 提交必过）
- `phase_h/` 不 import `kaggle_zerograd_moe`；提交线不 import `phase_h/`。
- `python3 kaggle_zerograd_moe.py` 默认仍 `6.251` / 零 autograd；`selfcheck.py` 过。
- Phase H 产物 `ph_*` 前缀，不覆盖提交 checkpoint。

## 五、运行手册（脚手架已就绪）
- **G0（本地 CPU）**：`python3 phase_h/ph_nli.py` · `python3 phase_h/ph_arith.py`（合成，已 100%）。
- **G1 本地 smoke**：`python3 phase_h/ph_nli_gpu.py --source synthetic --steps 800`（验证 loop/summary）。
- **G2 本地 smoke**：`python3 phase_h/ph_gsm_gpu.py --steps_list 2,3,4 --train_steps 1500`（深度扫描；k=2=100%，k≥3 需更大/更长）。
- **G1+G2 上 GPU**：`python3 phase_h/build_ph_kernels.py`（生成两个 kernel：`kaggle_ph_nli` 真实 SNLI / `kaggle_ph_gsm` 多步深度扫描）→ `python3 phase_h/orchestrate_ph.py`（默认跑 `phnli phgsm`；push→poll→pull，需 Kaggle creds；结果入 `runs/experiments.jsonl` + `runs/ph_{nli,gsm}_run_summary.json`）。改脚本后必重跑 build。
- **Track 1 雷达**：`python3 track1_radar.py` → `runs/track1_radar.png`（读 `runs/track1_metrics.json`；真实 Kaggle 数到手后改 JSON 再跑）。
- **监控**：`runs/ph_orchestrator.log`（人读）+ `runs/experiments.jsonl`（机器台账，含 push/finished/metrics 事件）。
