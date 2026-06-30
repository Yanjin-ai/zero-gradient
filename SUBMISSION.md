# Kaggle 提交（官方版，纯 ZeroBP）— 准备与完整性

> 一句话：**官方提交 = `kaggle_run/` kernel，跑纯 ZeroBP 4.16B（BPE+MLP+Phase C），零 autograd、确定性、单 T4/3h，WikiText-103 test ppl ≈1391（早停）/≈1355（跑满预算）。** 所有 BP / Phase E / Phase F 研究都在**独立脚本**里，**不进提交**。

## 什么是提交版
- kernel：`kaggle_run/`（id `yanjinli2001/post-backprop-zerograd-moe`，T4，挂 `vadimkurochkin/wikitext-103`）。
- 代码：内嵌**当前** `kaggle_zerograd_moe.py`（`build_kaggle_kernels.py` 自动生成，单一真源），默认 `KAGGLE` 配置 = `tokenizer="bpe", head="mlp"` + Phase C schedule（lr 余弦 0.03→0.003、freeze_routing@5000、patience 8）。
- 运行：`runpy.run_path('kaggle_zerograd_moe.py','__main__')`，**不设任何 `ZG_*` 环境变量** → 走纯 ZeroBP 默认路径。

## 完整性保证（已逐项实测）
| 检查 | 结果 |
|---|---|
| 默认路径零 autograd（gate 断言 `not torch.is_grad_enabled()`） | ✅ PASS |
| 提交文件 `kaggle_zerograd_moe.py` 内**无** `.backward()/autograd.grad/enable_grad` | ✅（BP 只在 `phase_e*.py`/`phasee_nli_4b.py`，不被提交导入） |
| 任何研究 flag 默认开？（attn_train/save_ckpt/freeze_heads/backbone_lr_scale/aux_w） | ✅ 全默认 no-op |
| 提交 notebook 含研究 import / ZG_ flag？ | ✅ 无 |
| 内嵌代码 == 当前 `kaggle_zerograd_moe.py` | ✅ 一致（单一真源） |
| KAGGLE 配置 = bpe+mlp（文档化最佳 1355/1391） | ✅ |
| 小配置默认跑 = final_ppl 6.251、6/7 门、确定性 | ✅ |

> **修复记录**：此前 `kaggle_run.ipynb` 内嵌的是 **pre-D-1 word-level 快照**（非 bpe+mlp），若提交会跑出旧的 word-level 结果而非文档化最佳。已用 `build_kaggle_kernels.py` 从当前代码重生成，锁定为 bpe+mlp 提交版。

## 怎么推送 / 运行
```
python3 build_kaggle_kernels.py            # 改了 kaggle_zerograd_moe.py 后重生成提交 notebook
kaggle kernels push -p kaggle_run          # 推送官方提交 kernel
kaggle kernels status yanjinli2001/post-backprop-zerograd-moe
kaggle kernels output yanjinli2001/post-backprop-zerograd-moe -p kaggle_run/out
```
产物：`run_summary.json`（含 7 门、wikitext103_test_ppl、显存、tok/s）、`loss_curve.png`、`memory_profile.png`。预算内：~2.9h 跑满（Phase C 单调，未早停时 ppl 更低）或 ~54min 早停。

## 与研究分支的隔离（红线）
- **提交版永不含 BP**：`kaggle_zerograd_moe.py` 全程 `set_grad_enabled(False)`，仅手写闭式局部规则。
- **研究分支（Phase E/F，含少量 BP）**全部在独立脚本（`phase_e.py`/`phase_e_4b.py`/`phasee_nli_4b.py`/`h1_attn.py`/`f1_data.py`/`f2_aux.py`/`task_*.py`）+ 独立 kernel（`kaggle_c1`/`kaggle_adapt`/`kaggle_phase_e`/`kaggle_phasee_nli`），**默认 off、不进提交**。
- 改提交版前的硬检查：默认路径 6.251 + 零 autograd；`load_state_dict` clone；`selfcheck.py` 过（见 `ENGINEERING.md §6b`）。

## 维护规则
任何对 `kaggle_zerograd_moe.py` 的改动**必须**：① 保持默认路径零 autograd / 6.251 不变（研究 flag 一律默认 no-op）；② 跑 `python3 build_kaggle_kernels.py` 重生成提交 notebook；③ 本说明的完整性表重新过一遍。

> **v1.0 / v2.0 边界**：任何会破坏默认 `6.251` / 零 autograd 的改动（如改架构 readout、可训练 attention、更深 BP）= **v2.0 / non-submission / research-only**，必须新建独立路径、默认 off，**不进本提交**（见 [ARCHITECTURE.md](ARCHITECTURE.md) 与 [docs/adr/ADR-003](docs/adr/ADR-003-submission-purity.md)/[ADR-004](docs/adr/ADR-004-v2-opening.md)）。
