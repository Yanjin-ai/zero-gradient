# Phase B baseline 定论（原设定，实测于真实 Kaggle T4）

设定：WikiText-103 + 32000 词表（word-level）+ 4.156B 零梯度稀疏-学习 MoE + lr0.03 + warmup。
全程实测于真实 T4（kernel `yanjinli2001/post-backprop-zerograd-moe`）。产物在 `runs/kaggle/`。

## 资源 profile（全部成立 ✅）
- 参数 **4.156B**（≥4B 门禁 ✓），fp16 常驻峰值显存 **8.32GB / 16GB**。
- 吞吐 **8000 tok/s**，3h 内 **19,926 步**。
- 每步只更新 **16 / 3800 专家（0.42%）**——sparse learning，训练 FLOPs ∝ k_update、与 N 解耦。
- **BP 训 4B 需 ~31GB → T4 OOM**；我们 8.3GB 跑通。零 autograd、确定性。

## 质量结论（诚实）
- **best WikiText-103 test ppl ≈ 1437**，在 **~40–50 分钟 / ~4600–6000 步**达到（val 触底 1752 @ step6000）。
- **之后继续训练会漂移变差**：val 1752(step6000) → 2383(step19900)；满 3h 的 final test ppl 退化到 **1919**。
- → **原设定下 3h 预算用满有害；baseline final PPL ≈ 1437（需早停/最佳checkpoint才能保住）。**

## 根因
无 lr 衰减 / 早停 / best-checkpoint；EMA prototype 持续漂移 → 路由不断变 → 专家被重分配 → 类灾难性遗忘的漂移。绝对 ppl 高也部分因 32000 词表（word-level，天然 ppl 高）。

## 留给 Phase C 的修法（复用同一指标面板）
1. **lr 余弦衰减 + 早停 / 按 val 存最佳checkpoint**（直接解决"3h 反而变差"）。
2. **路由稳定化**：prototype EMA 衰减率随训练降低 / 训练后期冻结路由，减少漂移。
3. **更有意义的 ppl**：换 BPE/更小词表，或更强读出（当前单层 d×V over 32k 是瓶颈）。
4. 目标：在新设定上用同一面板（资源 profile + 稳定性 + WikiText test ppl）对比改进幅度。

**一句话**：系统两条腿（4B 常驻 + 稀疏训练击败 BP 显存）在真实 T4 完全坐实；质量这条腿是个诚实的弱 LM（best ~1437），且当前缺早停/衰减导致 3h 反而退化——这是 Phase C 的首要修复项。
