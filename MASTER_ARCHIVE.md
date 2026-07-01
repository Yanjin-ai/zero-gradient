# MASTER ARCHIVE — 锁定结论索引（接棒必读）

> **定位**：本文件是**锁定结论的英文/双语索引**，给新接手者一页看清"什么已成定论、什么是解读、什么是提案"。深度叙事见 [`项目总档案.md`](项目总档案.md)（中文总档）；工程规范见 [`ENGINEERING.md`](ENGINEERING.md)；提交完整性见 [`SUBMISSION.md`](SUBMISSION.md)；架构边界见 [`ARCHITECTURE.md`](ARCHITECTURE.md)；决策记录见 [`docs/adr/`](docs/adr/)；实验台账见 [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md)。
>
> **标注约定**：**【LOCKED FACT】**=已实测、不得改写的结论；**【INTERPRETATION】**=对事实的推断/判读；**【PROPOSAL】**=未验证的下一步建议。**最后更新 2026-06-30。**
>
> **一致性铁律**：本文件所有数字与 commit 必须与 `项目总档案.md` / `runs/experiments.jsonl` 一致。任何会破坏默认路径 `final_ppl 6.251` 的改动 → 标 **v2.0 / non-submission / research-only**，不进提交。

---

## 0. 项目一句话
Kaggle Post-Backprop Challenge：单 T4 / 3h / **零全局梯度**下训练 **4.16B 常驻** 内容路由 MoE，逐专家手写局部规则（每步更新 0.42% 专家）。已从"能否跑通"推进到"ZeroBP 能力边界 + 少量 BP 能补到哪 + 架构瓶颈"。

## 1. Submission integrity（提交完整性）— LOCKED
- **【LOCKED FACT】** 官方提交 = `kaggle_run/` kernel，跑**纯 ZeroBP** 4.16B（BPE subword + 2 层 MLP 头 + Phase C schedule）。WikiText-103 test ppl ≈ **1391**（早停 t\*≈54min）/ ≈ **1355**（跑满 ~2.9h 预算）；小配置默认 `final_ppl 6.251`、7/7 合规门、零 autograd、确定性、峰值 ~8.33GB、~8546 tok/s。
- **【LOCKED FACT】** 提交**文件** `kaggle_zerograd_moe.py` 内**零** `.backward()/autograd.grad/enable_grad`；所有 BP 只在独立研究脚本（`phase_e*.py`/`phasee_nli_4b.py`/`h1_attn.py`），**提交从不导入**；所有研究 flag（`attn_train/save_ckpt/freeze_heads/backbone_lr_scale/aux_w`）默认 no-op。
- **【LOCKED FACT】** 修复记录（commit `ef7f213`）：此前 `kaggle_run.ipynb` 内嵌 **pre-D-1 word-level 快照**（非 bpe+mlp），会跑出旧结果；已用 `build_kaggle_kernels.py` 从当前代码重生成，锁定 bpe+mlp。完整 checklist 见 `SUBMISSION.md`。

## 2. Phase A–D（基础栈）— LOCKED（摘要，细节见总档/EXPERIMENTS.md）
- **【LOCKED FACT】** A：nano 机制验证（E01–E14）——内容路由 MoE + 局部规则可学；**controller 不胜 random（被否）**；**极稀疏更新不掉质量（k=1≈k=16）**。B：4.156B 真 T4 可跑、8.3GB（BP 需 ~31GB→OOM），但 baseline 后期漂移（test 1919）。C：lr 余弦 + 路由冻结 + 早停 → 单调、7/7 门。D-1：BPE 词表 + 2 层 MLP 头 → test 1391（优 unigram 51%），**头是大杠杆**。D-2：next-2 辅助损失失败（小配置拦下）。

## 3. Phase E（后训练能力边界）— LOCKED 🔒
**任务-方法矩阵（4B 真实 + 小配置交叉验证）**：

| 任务 | 结构 | 4B zero-shot | 4B 少量 BP | 小配置最好 |
|---|---|---|---|---|
| 情感 | bag 组合 | 60%(59.9) | **79%**(embedding+头, 公平闭式读出) | 100% |
| NLI | 关系对齐 | 33%(chance) | 33%(不转移) | 62% |
| 2 步算术 | 多步计算 | — | gate 失败 | ~chance |

- **【LOCKED FACT】** C.1 4B head-only（commit `3630b1f`）：MLP 59.9% / 线性 51.9% / majority 50.3%，LM ppl **1355.4→1355.4（零遗忘）**、零 autograd、确定性。
- **【LOCKED FACT】** Phase E 4B Mixed-BP（commit `a1e5e29`，公平**闭式头**读出）：60%→**79%**、遗忘 **+3.0 ppl**。注意：第一个点（lr0.05/400 + **SGD 头**读出）假性 57%——是**测量混淆**（SGD 头欠训），公平闭式头修正为 79%。
- **【LOCKED FACT】** v1.1 缓解 4B：full-adapt 61.4%/ppl 3546(**+2190 灾难遗忘**)；**冻 LM 头 62.2%/1359(+3.4)**（遗忘主因=LM 头）；结构分区 61.3%/0。
- **【LOCKED FACT】** NLI 4B 不转移（commit `8d5e0bb`），大预算重试 3000 步+attn lr 0.5 **仍 chance**（commit `ef26ef0`）。算术连小配置都 gate 失败（commit `f611679`）。
- **🔒【LOCKED】Phase E 阶段结论**：① ZeroBP 4B backbone 在 LM 上成立；② **少量 BP（embedding 路径）是唯一突破 ZeroBP 后训练天花板的机制**；③ **能力边界由任务结构复杂度控制**（bag 可突破+转移 / 关系对齐难 / 多步计算不可安装）。
- **【INTERPRETATION】** 根因：任务有多少落在 embedding（可 BP 装入）vs 需冻结 block 的顺序计算 / attention 跨句对齐（少量 BP 装不进）。

## 4. Phase F（向现代 LLM 靠拢 + 关系结构调查）— LOCKED 🔒
charter 见 `Phase-F charter.md`。两层：第一层严格 ZeroBP 预训练加强（F1/F2/F3）；第二层 Hybrid 受控 BP（H1/H2，只允许 attention+少量顶层）。
- **【LOCKED FACT】** F1-data（commit `c03db5d`）：只改预训练分布（random→richer 一致关系对+QA），NLI zero-shot **49.1%→51.3%（+2.1pp）**——数据单独不够。
- **【LOCKED FACT】** F2-aux-zeroBP（commit `dd9fa31`）：结构目标喂 ZeroBP 局部规则，NLI zero-shot **51.3%→51.8%（+0.5pp）**——ZeroBP 即便喂结构目标也刻不出关系几何。
- **【LOCKED FACT】** H1-attn-hybrid（commit `dd9fa31`，公平**新鲜闭式头**读出）：F1 base 51.3% → **BP-emb 58.8%** → BP-emb+attn 59.1%。
- **🔒【LOCKED】Phase F 关系子结论**：① 三条 ZeroBP 路线（数据/结构目标/attention 局部规则 Track A）**都装不进**关系几何（≤+2pp）；② 少量 BP **能部分装入**（小配置 51→59%），**墙的本质是 BP-vs-ZeroBP**；③ 但**公平读出下是 embedding 在扛，attention 只 +0.3pp**，且**有限、不转移 4B**。
- **【INTERPRETATION】** "attention 帮 NLI 到 62%"（Phase E task#2）部分是 BP 头拟合、非表示本身——公平新鲜头读出才是表示真值。

## 5. 架构发现：末位塌缩假说 — 被 Phase G 首线**证伪**
- **【LOCKED FACT（代码事实）】** `context()` 把整个序列塌缩成**最后一个位置**一个向量 `(emb+att@emb)[:, -1]`，之后所有 MoE block 都在这**一个** [B,d] 向量上算。
- **【LOCKED FACT】Phase G v2.0 首线证伪"塌缩是瓶颈"**（`v2_readout.py`，commit 见 ledger）：给**同一冻结 ZeroBP base** 换**不塌缩读出**（mean-pool / all-positions / concat），NLI zero-shot 不升反降——v1.0 last-h **50.4%**、mean-pool 34.9%(chance)、all-positions 32.9%(chance)、concat 47.5%。→ **关系结构不在冻结表示的任何位置**，更好的读出捞不出来。
- **【INTERPRETATION（更新）】** 真瓶颈在**上游**：**冻结随机 attention 从不形成对齐**（匹配实体不互相注意），所以任何读出都无可恢复。"末位塌缩=主因"**已弱化/证伪**；曾推断**关系任务的真正杠杆是可训练 attention**——但下一条线（§5b）已**证伪**此推断。
- **【LOCKED FACT】Phase G v2.0 第二线证伪"可训练 attention 是杠杆"**（`v2_attn.py`，commit 见 ledger）：同 F1 base + 公平新鲜闭式头、NLI 标签 CE 直接监督，**冻结 embedding、只训 Wq/Wk**：51.3%→**51.3%（+0.0pp）**（已验 ‖dWq‖≈0.080/‖dWk‖≈0.079 真动、‖dE‖=0 真冻）。参考臂复现锁定 H1：emb-only **58.8%** / emb+attn **59.1%**。→ **隔离后可训练 attention 单独毫无作用；59.1% 全是 embedding 的功劳，attention 真实边际 = 0**。
- **【INTERPRETATION（再更新）】** v2.0 已测的两个结构杠杆（不塌缩读出 + 可训练 attention）**均被证伪**：关系结构既不在冻结表示的任何位置，也无法靠训练 attention 装入。这把 ADR-002（关系/多步是真实能力边界）**进一步坐实**，并把 ADR-004 的"attention 首选杠杆"假设否掉。
- **【LOCKED FACT】Phase G v2.0 第三线 = 更深 BP 探针**（`v2_deepbp.py`，EXPLORATION，commit 见 ledger）：放开 BP 深度（emb→+top-block→+ALL-blocks→+attn），公平新鲜闭式头，**任务分化**：
  - **NLI（关系）**：floor 49.1 → emb 57.9 → emb+top **64.3** → emb+all+attn **65.7%**（**深 BP +16.5pp over floor，+7.8pp over 浅 emb**）。已验深臂两 block 各 ~22-23/48 专家真训。
  - **算术（多步）**：floor 24.7 → 所有 BP 臂 **19–21%（chance）**——**任何 BP 深度都装不进多步计算**，适配反而抹掉 zero-shot 的微弱优势。
- **【INTERPRETATION（关系线修正）】** 小配置的"关系上限"**是 BP 深度的函数，不是绝对**：浅 BP 的 59% 低估了；放开 block 深度可到 ~64-66%。但这**不改 4B 头条结论**——NLI 4B BP 已锁 chance（ADR-002），深 BP 的 4B 转移先验**弱**（未重测 4B）。**多步（算术）则在任何深度都是真·不可安装**，比关系更硬。

## 6. v1.0 / v2.0 分界 — 提案/边界（见 `ARCHITECTURE.md`）
- **【LOCKED】v1.0（固定骨架，= 提交版 + Phase E/F 研究）**：架构**不动**（末位塌缩 readout、单 attention、MoE/路由/局部规则）；提交版纯 ZeroBP；研究分支可在 embedding/attention/头加默认-off 的少量 BP，但**不改架构骨架**、不破坏 `6.251`。
- **【PROPOSAL】v2.0（允许动架构）**：放开**结构归纳偏置**——不塌缩的序列读出（pooling/多位置/注意力池化）/ 真正可训练 attention / 更深 BP。**任何 v2.0 改动 = non-submission / research-only**，必须新建独立路径、不污染默认 6.251。

## 7. 工程规范（硬约束）— LOCKED（见 `ENGINEERING.md §6b`）
- **【LOCKED FACT】** `load_state_dict` 的 CPU 别名 bug 已修（commit `66d5cc4`，clone）：CPU 上 `.to(cpu)` 不复制 → reset 后 base.E 别名 golden，原地 `index_add_` 污染 golden。**影响**多 reset 小配置实验（adapt/mitigate/d2/nli 已用 corrected 数）；**不影响** Phase E + 所有 4B（CUDA 复制+单发）。
- **规则**：reset 永远 clone；golden 只读；原地更新只在本 run 局部拷贝；多 reset 实验必过 `selfcheck.py`；4B 单发隔离；每实验记录目标/改动/配置/参数量/资源/指标/gate/结论；每实验后同步 master/charter/ENGINEERING/ledger，稳定则更 README。

## 8. 下一步 — PROPOSAL
- **【DONE】Phase G 三线全部测完**：① 不塌缩读出（`v2_readout.py`）证伪；② 可训练 attention 隔离（`v2_attn.py`）+0.0pp 证伪；③ 更深 BP（`v2_deepbp.py`）= 关系小配置有真信号（→65.7%）但 4B 转移先验弱、多步任何深度仍 chance。v2.0 的三类结构杠杆**已全部测到头**。
- **【INTERPRETATION】结论收口**：在 v1.0/v2.0 当前骨架（末位塌缩 + 单冻结 attention + ZeroBP MoE）下，**多步计算真·不可安装**（任何 BP 深度 chance）；**关系对齐**可在小配置靠深 BP 装一点（~66%）但**不转移 4B**。继续在此骨架上加 BP 已无新增空间。
- **【ACTIVE】两条并行路（用户拍板，2026-07-01，见 [Phase-H charter.md](Phase-H%20charter.md)）**：**Track 1** = 当前骨架的现代-LLM-对齐轻量评测/展示（LM/情感/简单理解）+ 关系/多步边界做对照证据【EXPLORATION】；**Track 2** = **Phase H / v3.0 新骨架栈**（多层可训练注意力 + 开放 BP + 更丰富数据，攻 NLI/GSM8K），独立 `phase_h/`，治理 [ADR-005](docs/adr/ADR-005-phase-h-new-backbone.md)。两条均 research-only，**不污染** ZeroBP 4B 提交线。

## 9. Phase H / v3.0 新骨架栈（research-only，独立 stack）— G0 已过 ✅
- **【LOCKED】定位**：抛弃 v1.0/v2.0 的"冻结 reservoir attention + 末位塌缩 MoE"，换成**标准多层双向 self-attention + mean-pool 读出 + 全 BP（AdamW）**的干净 Transformer，独立目录 `phase_h/`、**零依赖** `kaggle_zerograd_moe`、可整体迁出独立 repo。治理 [ADR-005](docs/adr/ADR-005-phase-h-new-backbone.md)，设计 [Phase-H charter.md](Phase-H%20charter.md)。**与提交线完全隔离**（互不 import；默认 6.251 / 零 autograd 不变，已复核）。
- **【LOCKED FACT】G0（本地，小配置，与 ZeroBP 矩阵同分布的合成任务）**：4L×4H d128 **0.80M** 全 BP base —— 合成 **NLI 100.0%**（vs ZeroBP 深 BP 65.7% 小 / 4B chance）+ 2 步**算术 100.0%**（vs ZeroBP 任何 BP 深度 chance＝不可安装）。
- **【INTERPRETATION】** 同一标准 trainable-attn base 同时攻克关系 + 多步两个 ZeroBP 装不进的维度 → **瓶颈是骨架（冻结 attention + 末位塌缩），不是任务、不是 BP 预算**。这与 ADR-002 一致（ADR-002 说的是"**当前骨架**的上限"），**不回改**任何 ZeroBP 锁定结论——Phase H 是另一条栈。
- **【作用域 / 诚实边界】** G0 = **合成、小模型、完全可学**；G1/G2 已上 GPU 真实数据（下）。
- **【LOCKED FACT】G1 真实 SNLI（Kaggle T4，DONE）**：标准 6L×8H d256 **12.4M** 全 BP，真实 SNLI **val 69.97%**（majority 33.8%）。**vs ZeroBP 4B NLI 锁定 = chance 33.4%** → **新骨架在真实关系 benchmark 上进入现代小 LLM 合格区间；ZeroBP 结构上到不了**。这是 Phase H 相对 ADR-002 边界的**实证突破**（不同栈，不回改 ZeroBP 锁定结论）。
- **【LOCKED FACT】G2 多步深度扫描（Kaggle T4，DONE）**：6L×8H d256，k=2 **100%** / k=4 22.5% / k=6 19.9% / k=8 20.6%（chance 20%）。→ **装得进 2 步**（ZeroBP 任何 BP 深度连 2 步都 chance），**但此规模/训练下深多步（k≥4）仍 chance**——是欠容量非绝对墙（本地 k=3 已达 100%@2.67M）。**深多步是 Phase H 待推的下一个边界**。
- **【LOCKED FACT】G2b 生成式 GSM8K（GPU，DONE）**：4.87M char-LM 8000 步，真实 GSM8K **exact-match 2.0%**——**如预告的诚实 stretch 失败**（小 char LM 无规模/预训练；合成 smoke 曾 66.6%）。机器工作、能力不足，不伪装。
- **【LOCKED FACT】Track1 SST-2（ZeroBP 4B，GPU，已查清）**：诊断 rerun 查明之前"49.08% 全同"是**闭式读出头退化**（三变体全预测 class-0 `[872,0]`）的 artifact。**公平 BP 线性探针（真值）**：zero-shot **51.5%** → emb 52.5% → emb+attn **53.3%**（majority 50.9%）。→ **真实 SST-2 上 ZeroBP 4B ≈ chance**（仅 +0.6~+2.4pp over majority），Mixed-BP 微小真实 lift 但不可用。**真实词汇情感与 NLI/多步同属 ZeroBP 装不进的边界**——合成 bag-sentiment 的 79% 是任务可 embedding-分离的特例，不代表真实情感能力。
- **【PROPOSAL】下一步**：推深多步（更大规模映射 k≥4 上限）；G1 扩 MNLI；或把 SNLI 70% / 多步 / SST-2≈chance 的跨栈对照直接收束进论文。

---
**接棒一句话**：提交版是**干净的纯 ZeroBP 4B**（已锁定、已修复）；研究已测清 **ZeroBP 的关系/多步能力上限**（少量 BP 部分有效但 **embedding 主导**、不转移 4B）。Phase G 三条 v2.0 结构线已**全部测到头**：**不塌缩读出**证伪、**可训练 attention 隔离**+0.0pp 证伪、**更深 BP** = 关系小配置有信号（→65.7%）但不转移 4B、多步任何深度仍 chance。在当前骨架上加 BP 已无新增空间。现走**两条并行路**：Track 1 = 当前骨架现代-LLM-对齐评测/展示（+ 边界对照）；Track 2 = **Phase H 新骨架栈**（`phase_h/`，ADR-005），**G0 已过**（小配置标准 trainable-attn base 全 BP 把合成 NLI/算术打到 100%——证明瓶颈是骨架）。两条 research-only，**勿改已锁定结论；勿破坏默认 6.251；`phase_h/` 与提交线互不 import**。
