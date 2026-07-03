# EXPERIMENT LEDGER — Phase E/F + 提交修复（关键实验·commit·结果）

> 本台账记录 **Phase D-1 之后**的关键实验（机器台账 = `runs/experiments.jsonl`；Phase A 的 E01–E14 见 `EXPERIMENTS.md`）。所有数字与 commit 与 `MASTER_ARCHIVE.md` / `项目总档案.md` 一致。标注 **[FACT]** = 已实测锁定，**[INTERP]** = 判读。
> ⚠️ **勘误已应用**：`load_state_dict` CPU 别名 bug（commit `66d5cc4`）污染过多 reset 小配置实验；下表小配置数为**修复后 corrected**；4B + Phase E 不受影响（见 §勘误）。最后更新 2026-06-30。

## 1. C.1 后训练（4B 真实 T4，全自动）
| 实验 | 结果 [FACT] | commit |
|---|---|---|
| C.1 pipeline（checkpoint save/reload + 局部任务头） | 小配置 5/5；reload 复现 best | `47f4dc4` `19644c3` |
| C.1 自动化（orchestrator + ckpt kernel + 台账） | push→poll→pull→record；修 kernelspec/slug/状态解析 | `ffc5221` `670815a` |
| **C.1 4B head-only** | MLP **59.9%** / 线性 51.9% / majority 50.3%；LM ppl **1355.4→1355.4（零遗忘）**；零 autograd、确定性 | `3630b1f` |
| [INTERP] | 机制成立；零样本迁移弱（WikiText 表示对域外组合情感线性可分性有限） | — |

## 2. v1.1 域适配 + 缓解（4B + 小配置 corrected）
| 实验 | 结果 [FACT] | commit |
|---|---|---|
| 4B-Adapt 无缓解 | acc 61.4% / WikiText ppl 3546（**+2190 灾难遗忘**） | `3ef529d` `98bae43` |
| 4B-Adapt 冻 LM 头 | acc 62.2% / 1359（**+3.4**，遗忘主因=LM 头） | `526759d` |
| 2.1 缓解（小配置 corrected） | full 97.3%/+30.7；**冻头 86.7%/+13.3**；backbone×0.1 51.5%/+50.4；冻头+bb×0.3 51.8%/+2.9 | `6305f89`(数据已 corrected `66d5cc4`) |
| 2.2 结构分区（小配置） | concat[h,pooled] **95.9% / 零遗忘**（原模型全冻） | `fda8daf` |
| 2.2 分区 4B | 61.3%（vs frozen-h 59.9%，仅 +1.4pp）/ 零遗忘——**不转移** | `526759d` `7e68283` |
| 适配-遗忘曲线（小配置 corrected） | N=0/150/400/1000 acc 51.2/76.5/97.9/99.9%，遗忘 +0/+7.4/+19.1/**+44.0** | `060b511`(corrected `66d5cc4`) |

## 3. Phase E：Mixed-BP 突破（小配置 + 4B）
| 实验 | 结果 [FACT] | commit |
|---|---|---|
| D.2 ZeroBP-DeepSignal（小配置） | 纵向任务信号 49.5%→**61.8%**/+0.06——plateau，gate 失败 | `6d49b2c`(corrected `66d5cc4`) |
| Phase E 小配置（embedding+头 BP） | 49.5%→**100%** / +0.02；同组件 zero-BP(D.2) 仅 61% → **墙=BP vs ZeroBP** | `693d998` |
| 消融：embedding-only == emb+top | 都 100%/+0.02 → **embedding 是杠杆** | `693d998` |
| **Phase E 4B（embedding+头，公平闭式读出）** | 60%→**79%** / **+3.0 ppl** | `a1e5e29` |
| └ 混淆修正 [FACT] | 第一点 lr0.05/400 + **SGD 头**读出假性 57%；SGD 头欠训，**闭式头**修正为 79% | `e1fa6d5` `a1e5e29` |

## 4. 任务-方法矩阵（task#2 NLI / task#3 算术）
| 实验 | 结果 [FACT] | commit |
|---|---|---|
| NLI 小配置 4 路线 | zero-shot 49.1 / zero-BP 45-47 / Mixed-BP emb 57.9 / emb+attn 61.7% | `66d5cc4` |
| **NLI 4B** | zero-shot/emb/emb+attn **全 33.4%（chance）**——不转移 | `8d5e0bb` |
| NLI 4B 大预算重试（3000 步/attn lr 0.5） | **仍全 chance** → 结构限制非欠训 | `ef26ef0` `bfed728` |
| 算术（2 步）小配置 | zero-shot 24.7 / Mixed-BP(任意组件) ~19-20%（chance）→ **gate 失败，不迁 4B** | `f611679` |
| [INTERP] 矩阵 | Phase E 威力随**任务结构复杂度**递减：bag 组合可突破+转移 / 关系对齐难 / 多步计算不可安装 | — |

## 5. Phase F：预训练加强 + Hybrid（小配置）
| 实验 | 结果 [FACT] | commit |
|---|---|---|
| F1-data（更丰富分布，严格 ZeroBP） | NLI zero-shot 49.1%→**51.3%（+2.1pp）**——数据单独不够 | `c03db5d` |
| F2-aux-zeroBP（结构目标 + 局部规则） | 51.3%→**51.8%（+0.5pp）**——ZeroBP 喂结构目标也刻不出关系几何 | `dd9fa31` |
| H1-attn-hybrid（少量 BP，公平新鲜头） | F1 base 51.3% → **BP-emb 58.8%** → BP-emb+attn 59.1% | `dd9fa31` |
| 🔒 Phase F 关系子结论 | ZeroBP 三路线都装不进（≤+2pp）；少量 BP 部分装入（51→59%，embedding 主导）；有限、不转移 4B | `dd9fa31` |

## 6. 架构发现 + 提交修复
| 项 | 结果 [FACT] | commit |
|---|---|---|
| 末位塌缩（架构线索） | `context()` 把序列塌成单 [B,d] 向量 [FACT 代码] → 曾疑似关系瓶颈 [INTERP] | `d4f35df` |
| **Phase G v2.0 不塌缩读出（`v2_readout.py`）** | 冻结 base 换 mean/all-pos/concat：last-h 50.4 / mean 34.9 / all-pos 32.9 / concat 47.5%——**不升反降，证伪"塌缩=瓶颈"** [FACT]；真瓶颈=冻结 attention 不对齐 [INTERP] → 杠杆转可训练 attention | (本轮) |
| **Phase G v2.0 可训练 attention 隔离（`v2_attn.py`）** | 同 F1 base + 公平新鲜闭式头；NLI 标签 CE 直接监督。冻结 emb、**只训 Wq/Wk**：51.3%→**51.3%（+0.0pp）**（已验 ‖dWq‖≈0.080/‖dWk‖≈0.079 真动、‖dE‖=0 真冻）。参考臂复现锁定 H1：emb-only 58.8 / emb+attn 59.1%。→ **可训练 attention 单独不是杠杆，59.1% 全是 embedding 的功劳，attention 真实边际=0** [FACT]；ADR-004"真杠杆=可训练 attention"被**证伪** [INTERP] | (本轮, post-`eeb2b09`) |
| **Phase G v2.0 更深 BP 探针（`v2_deepbp.py`，EXPLORATION）** | gen_O base + 公平新鲜闭式头；放开 BP 深度（emb→+top-block→+ALL-blocks→+attn）。**NLI**：floor 49.1→emb 57.9→emb+top **64.3**→emb+all 63.9→emb+all+attn **65.7%**（**深 BP +16.5pp over floor / +7.8pp over 浅 emb**）。**算术**：floor 24.7→所有 BP 臂 **19–21%（chance）**，任何深度都装不进。已验深臂两 block 各 ~22-23/48 专家真训。→ [FACT] **关系：深 BP 在小配置有真信号（小配置上限是 BP 深度的函数，非绝对）；多步：任何深度仍 chance**。[INTERP] 4B 转移先验弱（NLI 4B 已锁 chance），未重测 4B | (本轮, post-`eeb2b09`) |
| **提交修复** | `kaggle_run.ipynb` 曾内嵌 pre-D-1 word-level 快照；已从当前代码重生成（bpe+mlp，纯 ZeroBP） | `ef7f213` |
| load_state_dict 别名 bug 修复 | clone-on-load；`selfcheck.py` 守护；勘误已应用 | `66d5cc4` |

## 7. Phase H / v3.0 新骨架栈（research-only，独立 `phase_h/`，零依赖 ZeroBP）
> 治理 = [ADR-005](adr/ADR-005-phase-h-new-backbone.md)；设计 = Phase-H charter.md。与提交线**完全隔离**（互不 import；提交默认 6.251 / 零 autograd 不变）。标准多层双向 attention + mean-pool 读出 + 全 BP（AdamW）。本地 CPU 跑小配置合成任务（与 ZeroBP 矩阵**同分布**，apples-to-apples）。
| 实验 | 结果 [FACT] | 文件 |
|---|---|---|
| **G0-NLI**（4L×4H d128, 0.80M, 全 BP） | 合成 NLI（同 task_nli 分布）val **100.0%**@step500 → 收敛 100%。**vs ZeroBP 锁定：深 BP 65.7%（小）/ 4B chance** | `phase_h/ph_nli.py` |
| **G0-arith**（同 base） | 2 步算术（同 task_arith 分布）val **100.0%**@step1000。**vs ZeroBP 锁定：任何 BP 深度 19–21% chance＝不可安装** | `phase_h/ph_arith.py` |
| 🔒 G0 子结论 [INTERP] | 同一标准 trainable-attn base + 全 BP **同时攻克**关系（NLI）与多步（算术）两个 ZeroBP 骨架装不进的维度 → **瓶颈是骨架（冻结 reservoir attention + 末位塌缩），非任务/非 BP 预算**。**作用域**：合成、小模型、完全可学；**未**触及真实 SNLI/MNLI/GSM8K（G1/G2 需 GPU）。G0 过 → Phase H 有腿，值得上 GPU | — |
| **🎯 G1 真实 SNLI（GPU，DONE）** | Kaggle T4 跑真实 SNLI（12.4M，6L×8H d256，全 BP，549k 训练/9842 验证，12873 步）：**val_acc 69.97%**（majority 33.8%）。**vs ZeroBP 4B NLI = chance 33.4%** → **新骨架在真实关系 benchmark 上进入现代小 LLM 合格区间，ZeroBP 结构上做不到** [FACT]。**G1 gate PASS** | `phase_h/ph_nli_gpu.py`（kernel `post-backprop-phase-h-nli`） |
| **🎯 G2 多步深度扫描（GPU，v1+deep，DONE）** | v1（6L d256 4.7M/6k 步）：k2 **100%** / k4 22.5% / k6 19.9% / k8 20.6%。**deep（12L d384 21.3M/15k 步）**：k2 **100%** / k4 21.3% / k6 19.8% / k8 21.4%（chance 20%）。→ [FACT] **k=2 装得进（ZeroBP 任何 BP 深度连 k=2 都 chance）；k=3 本地 100%@2.67M；但 k≥4 = chance，且 5× 参数 + 2.5× 步数完全没动它**。**"欠容量"假设被 deep run 证伪** → 新骨架有**自身多步深度墙 ~k≤3**，raw scale 不破（本任务=直接答案分类 + mean-pool，无 curriculum/CoT——这两条未测） | `phase_h/ph_gsm_gpu.py`（kernel `post-backprop-phase-h-multi-step` v2） |
| **G2b 生成式 GSM8K（GPU，DONE）** | causal LM `PhCausalLM` 4.87M/6L，8000 步，真实 GSM8K：**exact-match 2.0%**。**如预告的诚实 stretch 失败**——小 char LM 无规模/预训练，真实 NL GSM8K 装不进。本地 synth smoke 曾 66.6%（合成可学）。[FACT] 生成式机器工作，模型能力不足；不伪装 | `phase_h/ph_gsm_gen.py`（kernel `post-backprop-phaseh-gsmgen-v2`） |
| **🔒 Track 1 SST-2（ZeroBP 4B，GPU，已查清并锁定）** | 诊断 rerun（`track1_sst2_4b.py` 加预测分布 + 标准 BP 线性探针）：**闭式头退化**——三变体全预测 class-0（`[872,0]`）→ 之前的"49.08% 全同"是**读出坍缩 artifact**，非真值。**公平 BP 线性探针（真值）**：zero-shot **51.5%** `[255,617]` → emb **52.5%** → emb+attn **53.3%** `[229,643]`（majority 50.9%，true `[428,444]`）。→ [FACT] **真实 SST-2 上 ZeroBP 4B ≈ chance（探针仅 +0.6~+2.4pp over majority），Mixed-BP 有微小真实 lift 但远不可用**。对照合成 bag-sentiment 79%：**真实词汇情感 ZeroBP 栈装不进**（与 NLI/多步同类边界，非 bag-可分任务） | `track1_sst2_4b.py`（kernel `post-backprop-track1-sst2-v2` v3） |
| **Track 1 SST-2（ZeroBP 4B，pending GPU）** | 复用 `phasee_nli_4b` helpers，在 4B checkpoint 上真实 GLUE/SST-2：zero-shot / Mixed-BP(emb) / (emb+attn)，公平闭式头 + WikiText ppl 漂移。import/cross-ref 校验通过；kernel（挂 wikitext + 4B ckpt + internet）就绪 | `track1_sst2_4b.py` |
| **Track 1 能力雷达脚手架** | 数据驱动雷达（`runs/track1_metrics.json`→`runs/track1_radar.png`）：ZeroBP-4B（锁定真值 LM~.51/情感.79/NLI.334/算术.20）vs Phase H（G0 合成 关系/多步=1.0，LM/情感 pending）。已本地渲染验证 | `track1_radar.py` |
| **Kaggle 自动跑（全部就绪，2 批）** | G1+G2 **DONE**（上表）。G2b+SST-2 已修复重跑：**RUNNING**（后台 monitor `orchestrate_ph phgsmgen` + `orchestrate_kaggle sst2` → poll/pull/record）。两个环境坑已解：① **CLI**：`pip install -U kaggle`→kagglesdk 0.1.32（打包 bug 缺多个模块）弄坏 CLI，修复=**pin `kagglesdk==0.1.31`**（配 kaggle 2.2.1，auth=`~/.kaggle/access_token`）；② **slug**：新 kernel 若 `id` 的 slug==title-slug 且有 ghost 态会 404/409，改用**全新唯一 slug（`-v2`）+ camelCase title（id==title-slug）**干净创建 → G2b=`post-backprop-phaseh-gsmgen-v2`、SST-2=`post-backprop-track1-sst2-v2` | `orchestrate_ph.py` · `orchestrate_kaggle.py sst2` |

## 勘误（Erratum）— 必须知道
`load_state_dict` 在 CPU 上 `.to(cpu)` 不复制 → reset 后 base.E 别名 golden checkpoint，随后原地 `index_add_` 污染 golden，使**多 reset 小配置实验**串味。**已修**（commit `66d5cc4`，clone）。
- **受影响（已用 corrected 数）**：adapt_sentiment / adapt_mitigate / d2_deepsignal / task_nli。例：适配遗忘 N1000 原 +63 → corrected **+44**；2.1 冻头原 92.4 → corrected **86.7%**；backbone 缩放变体原虚高 95% → corrected **~chance 失效**。
- **不受影响**：Phase E（重新赋值 E，从不原地）+ **所有 4B 实验**（CUDA `.to` 复制 + 单发）→ **79% 突破、矩阵 4B 行、提交版均不变。**
