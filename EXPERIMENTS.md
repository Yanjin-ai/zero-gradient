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

## E10 · 阶段 A / Stage 4：controller v3 接入特化 MoE（importance vs random/uniform 消融）
- **目标**：模型+局部规则不动，只引入 controller v3（在被路由专家里按 score 选 top-k_update 做全量更新，其余 soft_floor）；多 seed 测 importance(D) 是否**稳定**优于 random(B)。
- **模式**：uniform（全更新，预算无限上界）/ random（随机选 k）/ fixed_topk / importance（v3 score 选 k）。`zerograd_moe.py stage4`，仪表盘 `dashboard/stage4.html`。
- **结果（5 seed，k_update=6）**：uniform 6.48±0.34 < importance 6.99±0.51 ≈ random 7.03±0.49 < fixed_topk 7.44。**paired gap(random−importance) = +0.042 ± 0.027，4/5 seed 为正**。
- **预算扫描（关键）**：gap 随预算收紧**单调放大**——k=10:+0.017 / k=6:+0.042 / k=4:+0.057 / k=3:+0.062 / **k=2:+0.079**（每档 4/5 seed 正）。
- **判决**：✅ **importance > random 稳定成立**（与 B 实验的"importance≈或<random"形成鲜明对比），且**预算越紧优势越大 = 正中 4B regime**（N 巨大、k_update/N→0）。margin 仍小（≤0.08 ppl）但**方向一致、机理清晰**。**nano 级整套"结构感知训练资源分配"想法第一轮闭环完成。**
- **对比总览**：B（无特化）gap −0.04~−0.18 → Stage 4（特化）gap +0.02~+0.08。**特化是 controller 价值的前提，已被双向验证。**

---

## E11 · 阶段 A / Stage 5 (3.1)：nano → small 堆叠 MoE（机制是否可扩）
- **目标**：同语料同门禁，沿结构轴放大（加宽 d、堆 2–3 个 MoE block、增专家数），EMA-routing + controller v3 + coverage 全不变，看特化和 importance>random gap 是否保持/放大。`zerograd_moe_scale.py`（多层堆叠，每层独立 prototypes/experts/local head/controller）。
- **过程修复**：多层重构漏了 embedding 末-token 直连更新 → 模型学坏（ppl 13+，不及 bigram）；补回后 ppl 恢复（7.2/7.0/6.9，与 stage4 importance@k=4 一致）。
- **结果（修复后，4 seed）**：
  - nano  1L d64  16e: purity 0.80, gap **+0.048**±0.036 (3/4)
  - small 2L d128 32e: purity 0.84, gap **−0.014**±0.021 (1/4)
  - small 3L d128 64e: purity 0.94, gap **−0.119**±0.014 (0/4)
- **判决**：✅ **特化随规模更锐利**（purity 0.80→0.94，熵不坍塌）；❌ **controller 优势不可扩——随专家数增加而侵蚀并反转**（gap +0.05→−0.01→−0.12）。**根因 = B 实验的"覆盖"张力在大 N 重现**：专家越多、每步预算固定 → 欠训练专家越多 → 覆盖压倒价值 → "集中"(importance)饿死多数专家、输给"铺开"(random)。**v3 权重在 N=16 调的，不迁移到 N=64。对 4B(~3700 专家)是强警告。**
- **修复验证（lam_cov sweep @3L/64e）**：lam_cov 1→4→12→30 时 gap −0.219→+0.015→+0.038→+0.041（12/30 时 3/3 全赢）。**确认"coverage 必须随 N 放大"是解。**
- **v4 controller（自动缩放）**：`lam_cov_eff = lam_cov·(N/16)²`（N=16→1, 32→4, 64→16）。三配置重测：nano 16e **+0.048**(3/4)、small 32e **−0.007**(0/4, 平手)、small 64e **+0.024**(4/4)。**大 N 灾难性反转修好**（64e −0.119→+0.024，通往 4B 最关键方向）；32e 卡平手→缩放律还需精细化（非完美单调全胜，已诚实标注）。仪表盘 `dashboard/scale.html`。
- **3.1 总结**：特化可扩（purity 0.80→0.94）；controller 默认权重不可扩，但**coverage 随 N 缩放后大 N 优势恢复**。下一步 3.2 用更自然语料 + 精细化缩放律。

## E12 · 阶段 A / Stage 5 (3.2)：自然语料（机制在真实分布下是否存在）
- **目标**：换真实噪声分布（Pride & Prejudice + Shakespeare，`data/natural.txt`，9万词，无 topic 标记），机制不变，路由键改 context mean（无标记），重测特化 + importance>random。`zerograd_moe_nat.py`。无标签特化度量 = 路由 coherence（key 到自己 prototype 的相似度 − 到随机的，>0=内容聚类）。
- **结果（small 2L d128 32e，4 seed）**：vocab 4000，unigram 821，bigram 649。uniform 398 / importance 393.6±5.7 / random 393.4 —— **全部碾压 bigram**。coherence **+0.224**（>0），entropy 2.94/3.47（不坍塌）。gap **−0.24 ± 3.25**（2/4）。
- **判决**：✅ **特化机制在真实分布下成立**——EMA-routing 在无标记真实文本里找到内容簇（coherence>0），模型 ppl 远超 bigram。❌ **controller 优势被真实数据 run 间方差吞掉**（gap std 3.25 ≫ |mean| 0.24）——synthetic 上 ~0.05 ppl 的优势在真实噪声（±3 ppl）下不可见。与"优势小、依赖紧预算+大 N 才放大"一致。
- **启示**：① 特化是机制的鲁棒部分，真实数据照样成立，可放心带去 4B；② controller 的资源分配优势在 nano/small 真实数据上是 wash，**4B 的赌注押在"紧预算 + 大 N 放大优势"这条尚未在真实数据上证实的趋势上**——3.3 surrogate 要专门验证它，或先做信号去噪（更强 EMA / 更鲁棒 leverage）+ 紧预算复测。

## E13 · 阶段 A / Stage 5 (3.3)：4B-surrogate（大 N + 极紧预算，PPL gap 能否冲出噪声）
- **目标**：唯一问题——"N 大、预算极紧的 surrogate 里 importance vs random 的 gap 能否超过真实噪声方差"。自然语料 + N=64 + k_update 扫 {1,4,16}（k/N 0.016/0.062/0.25），各 5 seed paired，1800 步。`zerograd_moe_nat.py surrogate`。
- **结果**：ppl 377–383（全碾压 bigram 649）；coherence **+0.396**（比 3.2 更强）、熵 3.53/4.16 不坍塌。
  - k=1: gap −1.14±1.50 (2/5) | k=4: −0.78±2.13 (2/5) | k=16: +1.30±2.98 (4/5)。**每档 std≫|mean|，无一冲出噪声**；收紧预算也没让 importance 冒头。
  - **关键副产品**：imp@k=1(377) ≈ @k=16(381) → **每步只更新 1.6% 的专家，质量几乎不掉**。
- **判决（决定性）**：❌ **controller 的 PPL 优势在真实数据上不存在**（任何预算档都被噪声吞没）→ **Kaggle headline 不能押在"importance 改进 PPL"**。✅ 特化 + 内容路由在大 N 真实数据上更锐利、稳健。✅ **极稀疏更新（k_update≪N）不损质量 = 系统效率的硬证据**。
- **战略转向（最终）**：Kaggle 两条腿 = ①"专家特化 + 内容路由的零梯度 MoE 真能学"（已 toy+真实文本验证）；②"4B 常驻但每步训练 FLOPs ∝ k_update、与 N 解耦 → 击败 BP 的 memory/speed（BP 训 4B 在 T4 直接 OOM）"。**不再把 headline 押在 importance>random 的可见 PPL 改进上**——它在真实数据是 wash；controller 的价值是"让极稀疏更新不掉质量"，即效率，不是 PPL。

## nano 闭环完成 → 下一步：scale 设计
dense+local 规则 ✓ → B 实验定因 ✓ → Stage 3 特化 ✓ → Stage 4 importance>random ✓。
下一步谈 scale（更大 MoE / 4B surrogate），保持结构比值（深度/宽度、稀疏专家/主干预算、controller 控制比例、state-cache/参数比），并在更紧预算下复测 gap 是否保持/放大。
