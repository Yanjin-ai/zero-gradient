# Phase B：≥4B Kaggle 提交（运行说明）

围绕 Phase A 定稿的两条腿搭的真实提交。文件：`kaggle_zerograd_moe.py`（自包含）/ `kaggle_zerograd_moe.ipynb`（直接上传 Kaggle）。

## 两条腿（都不声称智能调度提升性能）
1. **零梯度特化 MoE 真能学**：内容路由（EMA-prototype 最近邻 + capacity）+ 逐专家局部规则（deeply-supervised 闭式更新，无 autograd）。
2. **4B 常驻 + 极稀疏局部训练**：每步只更新 `k_update·n_layers` 个专家（4B 配置 = 16/3800 ≈ 0.42%），训练 FLOPs ∝ k_update、与 N 解耦 → 塞进 T4/3h；BP 训 4B 需 ~31GB → T4 OOM。

## 4B 配置（已核实）
- d=1024, vocab=32000, seq_len=64, 4 个 MoE 层 × 950 专家, k_route=2, k_update=4, fp16。
- **参数 = 4.156B ≥ 4B 门禁 ✓；fp16 常驻 = 7.74GB（塞进 T4 16GB）✓**。
- 每步更新专家 = 16（0.42%）；BP 需 ≈31GB → OOM。

## 合规（硬规则）
- 全局 `torch.set_grad_enabled(False)`，无 `.backward()`/autograd/optimizer——已断言。
- 确定性 SEED=42 + deterministic algorithms（小配置跑两遍自检一致）。
- 参数门禁断言 ≥4e9。
- 数据自备；WikiText-103 perplexity 仅在数据集挂载时报（offline gate，`find_wikitext()`）。
- 产物：`/kaggle/working/run_summary.json`、`loss_curve.png`、`memory_profile.png` + 6 项合规门自检。

## 怎么跑
- **本机（逻辑验证）**：`python kaggle_zerograd_moe.py` → 自动用小配置（3.2M，CPU），跑通、5/6 门过（BP-OOM 门仅 4B 有意义）。
- **Kaggle（正式）**：上传 `.ipynb`，选 **GPU T4** runtime；（可选）attach 一个 WikiText-103 数据集；运行——4B 配置在 GPU 上自动选中，3h wall-clock 守卫，产物写到 `/kaggle/working`。

## 本机已验证
小配置：ppl 35→6（打过 unigram）、零 autograd、确定性复现、产物齐全。4B 配置：参数公式 4.156B、显存 7.74GB、BP-OOM 演示数值正确。

## 待办（Kaggle 上）
- 真实 T4 上的峰值显存 / tokens·s⁻¹ / 3h 内步数。
- 挂 WikiText-103 报真实 perplexity。
- 跑 BP-训-4B 演示截 OOM（叙事核心图）。
- 调 4B 下的 lr / steps / 路由参数（沿用小配置验证过的值起步）。
