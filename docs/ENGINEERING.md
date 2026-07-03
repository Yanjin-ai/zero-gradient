# ENGINEERING — 系统操作说明 (v1.0 工程基座)

> 本文件是 v1.0 完整栈的"操作手册"：怎么跑本机验证、怎么一键跑 4B 预训练 + C.1、怎么看实验台账、关键 commit 地图。算法/研究叙事见 `项目总档案.md`。

## 1. 环境
- 本机：Python 3.9（Xcode），`pip3 install torch numpy nbformat tokenizers`（CPU 即可）。Kaggle CLI 2.2.x，凭证已配（`kaggle kernels list -m` 可认证）。
- 提交版默认路径（小配置，CPU，秒级）：`python3 kaggle_zerograd_moe.py` → 6/7 门、final_ppl 6.251、零 autograd、确定性。**这是 anchor，永远字节不变。**

## 2. 本机实验（全部零 autograd、确定性）
| 命令 | 作用 | 关键产物 |
|---|---|---|
| `python3 kaggle_zerograd_moe.py` | 提交版小配置 sanity（默认路径） | 6/7 门、ppl 6.251 |
| `ZG_TOKENIZER=bpe ZG_HEAD=mlp python3 kaggle_zerograd_moe.py` | BPE+MLP 小配置 | — |
| `ZG_CKPT=1 python3 kaggle_zerograd_moe.py` | 训练并落盘 best_ckpt.pt（save-once） | `runs/best_ckpt.pt` |
| `python3 ckpt_verify.py` | checkpoint save/reload 正确性 | reload==best |
| `python3 c1_posttrain.py` | C.1 pipeline（topic 任务，head-only） | 5/5 |
| `python3 c1_sentiment.py` | 组合情感 in-domain（线性 vs MLP 头） | MLP 100% |
| `python3 c1_4b.py` | 从 best_ckpt.pt 重载 + head-only MLP 情感（`ZG_C1_PARTITION=1` 用 2.2 分区头 [h,pooled]） | `runs/c1_run_summary.json` |
| `python3 adapt_sentiment.py` | v1.1 in-domain 适配 + 遗忘权衡扫描（N=0/150/400/1000） | 权衡表 |
| `python3 adapt_mitigate.py` | v1.1 缓解 2.1：冻结 LM 头 / backbone 重要性加权 | acc-vs-遗忘表 |
| `python3 adapt_partition.py` | v1.1 缓解 2.2：结构分区（原模型全冻，新头读冻结特征 → 零遗忘） | acc 表 |
| `python3 adapt_4b.py` | 4B 域适配 + C.1（带缓解 `ZG_ADAPT_FREEZE_HEADS=1` 等） | `runs/adapt_run_summary.json` |
| `python3 track_a_probe.py` / `track_a_diag.py` | Track A（attention）负结果探针/诊断 | — |

实验开关（env，默认 off）：`ZG_TOKENIZER=word|bpe`、`ZG_HEAD=linear|mlp`、`ZG_AUX=<f>`、`ZG_ATTN=1`/`ZG_ATTN_LR`/`ZG_ATTN_WK`、`ZG_CKPT=1`、`ZG_C1_STEPS=<int>`、`ZG_RUN_MIN`、`ZG_SMOKE=1`。

## 3. 4B Kaggle 自动化（一键 push→轮询→拉取→记录）
```
python3 build_kaggle_kernels.py          # 从 .py 重新生成两个自包含 notebook（改了代码就重跑）
python3 orchestrate_kaggle.py ckpt c1    # 全自动：ckpt(4B 预训练→best_ckpt.pt) → c1(head-only 情感)
python3 orchestrate_kaggle.py c1         # 只跑 C.1（ckpt 已 complete 时）
```
- `kaggle_ckpt/`：4B 预训练 + `ZG_CKPT=1` → `best_ckpt.pt`（8GB，留在 kernel output）。
- `kaggle_c1/`：通过 `kernel_sources` **自动挂载** ckpt 输出（无需手建 dataset）+ WikiText 数据集。
- orchestrator：幂等（已在跑/已完成则跳过 push）、不下载 8GB（C.1 服务端读）、只拉小结果 JSON、写台账 + `runs/orchestrator.log`。
- **坑（已修，勿再犯）**：① notebook 必须有 `kernelspec`（否则 papermill 报 "No kernel name"）；② Kaggle 用**标题**推 slug，不是 metadata `id` → id 与标题 slug 必须一致；③ 状态字串是 `KernelWorkerStatus.COMPLETE`，解析取最后一段。

## 4. 实验台账（本机 + 远程统一）
```
python3 experiments.py seed   # 录入本机已有结果（幂等追加）
python3 experiments.py show   # 汇总表（site/track/status/关键指标）
python3 experiments.py log '<json>'   # 追加一条
```
台账文件 `runs/experiments.jsonl`（gitignore，运行时数据）。orchestrator 自动追加远程事件与指标。

## 5. 关键 commit 地图
| commit | 内容 |
|---|---|
| `0ff347f` | Track A (A1) 可训练 attention → 负结果（探针+诊断），暂缓 |
| `19644c3` | C.1 checkpoint save/reload + 后训练 pipeline（5/5） |
| `907d45e` | C.1 组合情感 in-domain，MLP 头 100%（5/5） |
| `47f4dc4` | C.1 4B 包（save-once checkpoint + c1_4b 真实词 + kaggle_c1） |
| `ffc5221` | C.1 4B 自动化（orchestrator + ckpt kernel + 台账） |
| `670815a` | 修 Kaggle 自动化（kernelspec / slug / 状态解析） |
| `3630b1f` | C.1 4B 实测结果（机制成立、零样本迁移弱 60%） |

## 6b. 工程规范（状态管理，踩坑后定的硬规则）

> 起因：`load_state_dict` 的 **CPU 别名 bug**——CPU 上 `.to(cpu)` 不复制 → reset 后 `base.E` 别名 golden `baseA["E"]`，随后原地 `index_add_` 污染了 golden，使多 reset 小配置实验串味（见档案勘误）。规则如下，**违反即可能静默出错**：

1. **复位永远 clone，golden 只读**：`load_state_dict` 对每个张量 `.clone()`（已修）；`baseA = base.state_dict()` 视为**只读 golden checkpoint**，任何路径都不得原地改它。
2. **禁止"隐式别名 + 原地更新"混用**：对共享/复用对象（尤其 `E`）优先**重新赋值**（`base.E = new`）而非原地 `index_add_/index_copy_`；若必须原地，**只能在当前 run 的局部拷贝上**（如 `E_p = base.E.detach().clone()`，BP 改 `E_p` 不碰 `base.E`）。
3. **多 reset 实验必带 reset 自检**：跑 `python3 selfcheck.py` —— 同配置"fresh vs 串跑后再 reset"两次，结果必须**逐位一致**；不一致说明 reset 漏了状态。开发期先过自检再信数。
4. **gate + 日志分层**：每实验记 small-config gate（任务提升 vs 遗忘 vs 稳定性）；失败要写"失败原因 + 是否疑似 bug 影响"；区分**首跑 raw 数**与**修 bug 后 corrected 数**，用勘误段连起来（档案已这样做）。
5. **4B 与小配置风险隔离**：小配置探索脚本（多 reset、CPU）自带 reset 自检；**4B 单发脚本保持无共享状态、不原地写 checkpoint**（CUDA `.to` 本就复制、且单发）——即使小配置模块有 bug 也不污染 4B 主结果。

## 6c. 文件地图（工程）
- 算法/提交版：`kaggle_zerograd_moe.py`（含 ZG_* 开关、state_dict、save-once）。
- C.1：`ckpt_verify.py` / `c1_posttrain.py` / `c1_sentiment.py` / `c1_4b.py` / `adapt_sentiment.py`。
- 自动化：`build_kaggle_kernels.py` / `orchestrate_kaggle.py` / `experiments.py` / `kaggle_ckpt/` / `kaggle_c1/`。
- 文档：`项目总档案.md`（总）/ 本文件 / `C1-Kaggle 运行说明.md` / `EXPERIMENTS.md` / 各 Phase 说明。
