# Phase A 定论：核心创新 · 被否假设 · controller 角色

> Phase A（B → Stage3/4 → scale 3.1/3.2/3.3 → 覆盖诊断 E14，共 14 个实验）的最终交接结论。
> 基调：这是一个**可行性 + 效率（系统）结论**，不是"质量超越 baseline"的结论。所有声明都经得起"和 random 比"的拷问。

---

## 一、核心创新（Core Innovation）

**一句话：**

> 在**无反向传播**约束下，一个**内容路由的 MoE** 用**逐专家局部规则**训练时，可以在**更新（学习）维度**上做到极度稀疏——每步只更新 `k_update ≪ N_active` 个专家——而**质量无可测损失**；这把**每步训练算力与总参数量解耦**，使 **≥4B 常驻模型在单 T4 / 3h 内可训练**，而 BP 训 4B 在 T4 直接 OOM。

**为什么质量不掉（三个使能条件，缺一不可）：**
1. **局部规则** → 每个专家的更新相互独立、可跳过（BP 做不到：梯度跨层耦合）。
2. **capacity-balanced 路由** → forward 负载均匀，被触达的专家集近均匀，随机/轮转地更新其中一小撮即可覆盖。
3. **basin 鲁棒性** → 终点 likelihood 对"这一步更新了谁"高度不敏感（实测 `k_update=1` 与 `k_update=16/32` 的 PPL 几乎相同）。

**相对已有工作的新颖点：**
- 标准 MoE = **sparse forward（推理省算力）+ dense backward（激活的专家全部走 BP 拿梯度）**；训练成本仍随激活参数量走。
- 本工作 = **sparse forward + sparse LEARNING**（激活的专家里只更新 k_update 个）**且全程零反向传播**。
- 组合"**no-backprop + 局部规则 + 稀疏学习 + 内容路由特化**"以实现"**4B 常驻、T4 可训**"，是面向本竞赛的可辩护新颖包。

**诚实边界（必须明说）：**
- 这是 **feasibility / efficiency** 结论（memory、per-step FLOPs），**不是** "我们的方法 PPL 更好"。
- "不掉质量"= 相对它自己在更宽预算下的质量；不是相对一个强 BP baseline 的质量（那个 baseline 在 4B/T4 上根本跑不起来——这恰是卖点）。

---

## 二、被实验否定的假设（Refuted Hypotheses）

逐条列出"我们一度相信、然后被实验证否"的，附证据。**这部分是诚信的核心，也是防 reviewer 的护城河。**

| # | 曾相信的假设 | 判决 | 证据 |
|---|---|---|---|
| H1 | 智能 importance controller（leverage/learnability）能比 random 更好地分配更新预算 → 降 PPL | **否** | 真实数据上 importance ≈ random（在噪声带内），每个预算档都如此（3.2 gap −0.24±3.25；3.3 各档 std≫\|mean\|） |
| H2 | controller 在 PPL 上看不见的优势，会在**覆盖**（它真正的目标）上清晰显现 | **否** | 覆盖指标 undertrained_traffic 上 **random 每档最好**；controller 仅略降最坏 backlog（E14） |
| H3 | 加法式专家（强 backbone 上的残差）提供有用容量 | **否** | 强 backbone 上加法专家是冗余噪声（full ppl > backbone-only），必须把专家放上关键路径（MoE-faithful）（E05/E06） |
| H4 | value 信号（‖dW‖、learnability）在 scale 上可用作重要性排序 | **否** | 大 N + 小 batch 下 Var(噪声)≫Var(信号) → 排序不可辨识 ≈ 随机置换；只有 coverage 计数无噪，但它也不胜 random（3.1/3.3/E14） |
| H5 | 特化依赖显式结构标记 / 在真实数据上会失效 | **否（正向）** | 无标记真实文本上特化照样成立，coherence +0.40，比合成更强（3.2/3.3） |
| H6 | controller 是稀疏训练的**使能机制**（没它稀疏更新会饿死专家、质量崩） | **否** | random 稀疏选也能达到同质量（k=1 ppl ≈ k=32）、且覆盖更好；使能者是 regime，不是 controller（E14） |

**元教训**：PPL 是路径/约束类模块的**错配指标**（k=1≈k=16 的鲁棒性证实了这点）——但把指标校正到覆盖后，controller **依然**不胜 random。所以问题不只是"测错了"，而是"在已被 capacity 路由均衡的系统里，智能选择本就没有发挥空间"。

---

## 三、controller 在最终系统中的角色

经过 H1/H2/H6 的连环否定，诚实的角色定义：

**它不是什么：**
- ❌ 不是性能优化器（不降 PPL）。
- ❌ 不是覆盖优化器（覆盖上不胜 random）。
- ❌ 不是稀疏训练的使能机制（random 也行）。
- ❌ 不是 headline / 卖点。

**它是什么（保留的真实价值）：**
1. **预算接口**：它定义并执行"每步只更新 k_update 个专家"这件**真正重要**的事（k_update≪N 才是效率来源）。**该更新多少（预算）有价值；该更新谁（选择）没有。**
2. **确定性调度**：用 deterministic deficit round-robin 给出**逐位可复现**的更新计划——满足 Kaggle 的确定性硬要求，比 random（需小心 seed、概念上"任意"）更干净。
3. **最坏情况上界**：deficit 轮转略微压低单专家最坏 backlog（一个公平性保证），即便平均覆盖不优于 random。

**推荐最终形态：**
- 名字从"importance controller"改为诚实的 **"budget scheduler"**。
- 实现 = **确定性 deficit round-robin**；**value 信号全部去掉**（它们是噪声，H4）。
- 对外只暴露一个旋钮：`k_update`（效率/质量的连续 trade-off 控制点）。
- 定位 = "**约束满足 + 可复现性 + 最坏情况保证**"，不是"智能优化"。

---

## 四、对 Phase B 的两条腿（都不依赖聪明 controller）

1. **零梯度特化 MoE 真能学**：内容路由（EMA-prototype）+ 局部规则 → 专家特化 → 模型碾压 unigram/bigram（toy + 真实文本验证）。
2. **4B 常驻 + 极稀疏训练不掉质量**：每步训练 FLOPs ∝ k_update、与 N 解耦 → 塞进 T4/3h → 击败 BP 的 memory/speed（BP 训 4B 在 T4 OOM，截图为证）。

调度器 = 最简确定性 round-robin（为确定性与最坏上界，不为性能）。**主动不声称"智能调度提升质量"** —— 这反而是更硬、更难被打的定位。
