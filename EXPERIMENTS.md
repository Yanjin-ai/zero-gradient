# 实验日志（Experiment Log）

> 每个实验：目标 / 设置 / 结果 / 判决。完整跑通的 main() 运行也会进 `runs/history.json`（仪表盘"训练历史"表可见）。本文件额外沉淀手动 sweep 和判决。

约定：`importance/random/uniform/fixed_topk` 是 4 种**更新预算分配策略**（不是 forward 路由）；gap = random_ppl − importance_ppl（>0 表示 importance 更好）。

---

## E01 · nano 纵切片 v1（deeply-supervised 局部学习）
- **目标**：跑通零 autograd 的"模型+局部规则+importance路由+评测"整条线。
- **设置**：mean-pool context + ReLU 块 + 逐块 local head；importance vs random。
- **结果**：确定性✓、零autograd✓、loss降✓，但 val_ppl=24.6 **没打过 unigram 23.96**，acc 卡 0.236（多数类塌缩）。4/6 门。
- **判决**：模型没学到结构。根因→E02。

## E02 · 修 attention 残差 + embedding 训练
- **目标**：让模型真学起来。
- **改动**：context 加注意力残差（修复"整窗 mean-pool 把末token身份冲掉"）；embedding 经注意力权重近似局部更新（此前 `lr_embed` 从未生效）。
- **结果**：val_ppl 24→**7.36**，5/6 门。headline importance≈random（噪声）。
- **判决**：模型能学了；headline 未定，进 v2 系统化。

## E03 · v2 实验系统（锁定 controller + 4 基线 + 指标 + 历史）
- **目标**：把 controller 数学锁定（leverage/learnability/act/cost → EMA → z-score → softmax预算+top-k）、加 uniform/fixed_topk 基线、bigram baseline、多数类塌缩、score稳定性、跨run历史、7槽设计元数据、重做仪表盘。
- **结果**：7/9 门。**score_stability=1155（爆了）**；importance≈random；**model(7.4) 没打过 bigram(4.47)**。
- **判决**：① z-score 数值不稳→E04；② 模型质量不足（没过 bigram）→E05。

## E04 · z-score 裁剪
- **改动**：`_z` 裁剪到 ±4 + 方差守卫。
- **结果**：score_stability **1155→0.45**（5.2 条件达标）。

## E05 · loop-1：上下文依赖语料 + 统一读出头 + 调 lr/稀疏比
- **目标**：让模型打过 bigram（5.1）+ 稀疏比往 4B 靠。
- **改动**：① 语料重做为**多 topic、subject→verb 取决于句首 topic 标记**（bigram 抓不住）；② 砍掉单独 Hf，统一用末块 deeply-supervised 头（修 train/eval 头不一致）；③ lr 0.05→0.3、steps 500→1000、experts 16→32、专家小初始化。
- **结果**：backbone_only **10.8→5.75 < bigram 6.37**（5.1 达标）；7/9 门。但 full(6.27) > backbone_only(5.75)。
- **判决**：**加法式专家在强 backbone 上冗余**（净噪声）。架构岔路 B/A。

## E06 · 诊断：backbone-only vs full
- **结果**：低 lr 时 full < backbone（专家帮忙）；高 lr 强 backbone 时 full > backbone（专家拖后腿）。
- **判决**：专家"帮还是害"取决于 backbone 强弱；强 backbone 吃光结构→专家无残差可加。

## E07 · B 实验：瘛颈 backbone，让专家承重
- **设置**：`lr_backbone=0`（冻结）、`n_backbone=1`、`load_balance`(覆盖压力) sweep。
- **结果**：backbone_only **7.0 → full 5.7**（专家**真正承重**✓）。importance vs random：load_balance 0→8 gap 单调改善（−0.145→+0.023），lb=8 时 importance 微弱反超，lb=20 又掉。
- **判决**：**稀疏专家要"覆盖"不要"集中"**；但反超只有 +0.023，噪声级。

## E08 · v3 controller（coverage/deficit 一等项）+ 判决
- **改动**：加 `deficit = 路由流量 − 更新次数` 作为评分一等项 + update_ema 追踪；`lam_cov` sweep。
- **结果**：deficit 项**没能稳定复现** lb=8 的反超；importance vs random 全在 **±0.15 ppl 噪声带内**。
- **判决（B 的最终结论）**：**controller 不是瓶颈，专家"不可区分"才是**。固定哈希路由 → 专家无特化 → 不存在"哪个专家更值得更新"的真实价值差。coverage 只能避免最差、拉到≈random，**造不出不存在的信号**。→ **决定性 importance>random 需要专家特化 = 必须进 A（内容路由 + 专家即 FFN）**。带 v3 coverage-aware controller 进 A。

---

## E09 · 阶段 A / Stage 3：nano-MoE + 内容路由（controller 关）
- **目标**：造一个专家在关键路径、内容路由的 MoE，验证**专家是否特化**（controller 全关，所有被路由专家等量更新）。
- **结构**：薄主干（embedding + 冻结注意力，无可训练 backbone FFN）；16 专家 top-2 即主路 FFN；router = EMA-prototype 最近邻（cosine）+ 容量限制（防坍塌）；prototype 在线 k-means 更新。文件 `zerograd_moe.py`，仪表盘 `dashboard/stageA.html`。
- **过程问题与修复**：① lr=0.3 发散→降 0.1；② EMA-prototype 坍塌到 2 专家→加**容量限制路由**（每专家上限+溢出 reroute）；③ purity≈随机→定位路由键问题。
- **路由键消融（关键发现）**：用**末位置 h**（编码 subject）→ purity 0.29≈随机；**mean-pool**（topic 稀释）→ 0.35；**句首 topic 标记 embedding**（载判别信号）→ **purity 0.67**。
- **结果**：ppl **5.97 < bigram 6.37**；**purity 0.674 vs random 0.25**；route_H 2.70/2.77（不坍塌、16 专家均衡）；无 NaN。**SPECIALIZED = True**。
- **判决**：✅ **特化机制成立**——EMA-prototype 内容路由 + 容量管理 → 专家按 topic 特化，前提是**路由键要载到判别性内容**（对 4B/真实语料的启示：routing key 必须捕获判别上下文）。专家现在有真实价值差 → **可进 Stage 4（controller ON，重测 importance>random）**。

---

## 下一步：Stage 4（controller v3 接入特化的 MoE）
在已特化的专家上打开 controller v3，做 importance vs random/uniform 消融——这次有专家特化撑着，预期 importance>random 能决定性成立。
