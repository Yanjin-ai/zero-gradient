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
- **【INTERPRETATION（更新）】** 真瓶颈在**上游**：**冻结随机 attention 从不形成对齐**（匹配实体不互相注意），所以任何读出都无可恢复。"末位塌缩=主因"**已弱化/证伪**；**关系任务的真正杠杆是可训练 attention**（让它学会对齐），非读出。（注：此为"冻结 base 上换读出"的测试；"用不塌缩读出从头训练的 base"未测，但 mean/all-pos 即 chance 已强烈指向冻结 attention。）

## 6. v1.0 / v2.0 分界 — 提案/边界（见 `ARCHITECTURE.md`）
- **【LOCKED】v1.0（固定骨架，= 提交版 + Phase E/F 研究）**：架构**不动**（末位塌缩 readout、单 attention、MoE/路由/局部规则）；提交版纯 ZeroBP；研究分支可在 embedding/attention/头加默认-off 的少量 BP，但**不改架构骨架**、不破坏 `6.251`。
- **【PROPOSAL】v2.0（允许动架构）**：放开**结构归纳偏置**——不塌缩的序列读出（pooling/多位置/注意力池化）/ 真正可训练 attention / 更深 BP。**任何 v2.0 改动 = non-submission / research-only**，必须新建独立路径、不污染默认 6.251。

## 7. 工程规范（硬约束）— LOCKED（见 `ENGINEERING.md §6b`）
- **【LOCKED FACT】** `load_state_dict` 的 CPU 别名 bug 已修（commit `66d5cc4`，clone）：CPU 上 `.to(cpu)` 不复制 → reset 后 base.E 别名 golden，原地 `index_add_` 污染 golden。**影响**多 reset 小配置实验（adapt/mitigate/d2/nli 已用 corrected 数）；**不影响** Phase E + 所有 4B（CUDA 复制+单发）。
- **规则**：reset 永远 clone；golden 只读；原地更新只在本 run 局部拷贝；多 reset 实验必过 `selfcheck.py`；4B 单发隔离；每实验记录目标/改动/配置/参数量/资源/指标/gate/结论；每实验后同步 master/charter/ENGINEERING/ledger，稳定则更 README。

## 8. 下一步 — PROPOSAL
- **【PROPOSAL】Phase G 首线**：小配置先测**不塌缩的序列读出**（pooling / 多位置 / 注意力池化读出头）能否松动关系/多步任务——末位塌缩线索指向的、疑似最对症且最便宜的杠杆。过门才迁 4B；属 v2.0 / research-only，提交版不动。
- **【PROPOSAL】** 其后：真正可训练 attention（非冻结 reservoir）；更深 BP。均 v2.0。

---
**接棒一句话**：提交版是**干净的纯 ZeroBP 4B**（已锁定、已修复）；研究已测清 **ZeroBP 的关系/多步能力上限**（少量 BP 部分有效但 embedding 主导、不转移 4B）；**疑似真因是末位塌缩**；下一步是 **v2.0 的不塌缩读出**（research-only，先小配置）。**勿改已锁定结论；勿破坏默认 6.251。**
