[English →](KAGGLE_README.md) | **中文**

# Post-Backprop 挑战赛 —— ZeroGrad MoE(纯 ZeroBP 4B)

**提交 notebook 顶部说明 / 评审者 README。** 本参赛项在单块 T4 GPU 上,用**零全局反向传播**训练一个 **41.6 亿参数**的语言模型 —— 没有 `torch.autograd`、没有 `loss.backward()`、没有 optimizer。每次参数更新都是一条手写**局部规则**。

---

## 这是什么

- **模型:** 内容路由的**专家混合(MoE)**语言模型。`d=1024`、`|V|=32000`(BPE 子词)、`seq_len=64`、`4 层`、`每层 950 专家`(top-2 路由),确定性预算为每块每步 `k_update=4` 个专家。**41.6 亿 fp16 常驻参数。**
- **训练(ZeroBP):** 全程 `torch.set_grad_enabled(False)`(由 gate 断言)。专家由局部规则更新,信号来自每块的深监督读出头;路由是不可微的内容分配(EMA 原型 + 容量)。一个冻结的随机注意力层("reservoir")把末位表示喂给路由块。
- **读出 / schedule:** 2 层 MLP 读出头 + 带路由冻结和早停的余弦学习率("Phase C")。

## 结果(WikiText-103)

| 配置 | 测试困惑度 | 备注 |
|---|---|---|
| BPE + 2 层 MLP 头,早停(t\* ≈ 54 分钟) | **≈ 1391** | 省预算 |
| BPE + 2 层 MLP 头,跑满预算(~2.9 小时) | **≈ 1355** | 单调,无早停 |
| (参考)unigram 基线 | — | 模型好约 51% |
| 小配置 smoke 默认 | `final_ppl = 6.251` | 确定性,用于 CI/gate |

峰值显存 ~8.3GB,~8546 tok/s,舒适地落在单 T4 / <3 小时预算内。

## 纯净性与合规门(为什么这是有效的零反向传播参赛项)

默认运行会断言一套 gate;有效参赛项必须通过:

- **零 autograd** —— 训练路径中断言 `not torch.is_grad_enabled()`。
- **常驻参数 ≥ 门禁**、**损失单调**、**val-ppl < unigram**、**确定性(重跑一致)**、**无后期漂移**。
- 提交源码中**没有**真实的 `.backward()` / `autograd.grad` / `enable_grad` 调用(唯一的文本匹配是这条纯净性声明本身的注释),且**不 import 任何研究/BP 脚本**(`phase_e*`、`phase_h/*`、`v2_*` 等)。所有研究开关默认为空操作。

> 注:在极小 smoke 配置上通过 6/7 门 —— 那一个"失败"的门(`BP-4B would OOM on T4`)是一个**演示门**,只在完整 4B 运行时触发,在 smoke 配置上预期不激活。

## 复现

```bash
# 完整提交 notebook(嵌入当前训练器,纯 ZeroBP 默认路径):
python3 build_kaggle_kernels.py          # 从 kaggle_zerograd_moe.py 重新生成提交 notebook
kaggle kernels push -p kaggle_run        # 推送官方提交 kernel(挂 WikiText-103)
kaggle kernels status  yanjinli2001/post-backprop-zerograd-moe
kaggle kernels output  yanjinli2001/post-backprop-zerograd-moe -p kaggle_run/out

# 本机完整性检查(无需 GPU):
python3 kaggle_zerograd_moe.py           # 默认路径 → final_ppl 6.251,零 autograd 通过,确定性
python3 selfcheck.py                     # reset/读出守护通过
```

产物:`run_summary.json`(门、`wikitext103_test_ppl`、显存、tok/s)、`loss_curve.png`、`memory_profile.png`。

## 范围

这是**纯 ZeroBP 基线**参赛项。所有基于反向传播的研究(能力边界研究与 Phase H 可训练注意力对照)都在提交永不 import 的独立脚本里;完整性清单见 `SUBMISSION.md`,完整研究见 `PAPER_DRAFT.md`(中文研究报告见 [README.zh.md](README.zh.md))。
