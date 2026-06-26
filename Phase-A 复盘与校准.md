# Phase A 复盘与校准（定稿）

> 目的：在进入 Phase B（真实 4B Kaggle notebook）前，把已确定的结论分层钉死、修正指标错配、明确 controller 的真实定位与该测的量。
> 一句话定调：**controller 是无噪声的「覆盖调度器」，不是「PPL 优化器」；真正的硬通货是「training-time update sparsity 不损质量」。**

---

## 0. 关键校正（本次复盘的核心）

**指标错配，不是训练/实验 bug。** controller 作用在「训练路径」（coverage / learning dynamics / 稳定性），PPL 测的是「终点 basin」。用终点 log-likelihood 测一个约束「可达性 + 覆盖公平」的模块，本就不该有大信号。
`k_update=1 ≈ k_update=16`（PPL 几乎相同）是**正向证据**：basin 对"谁被更新"高度鲁棒 → scheduler 的职责是"可行地到达 basin"，不是"移动 basin"。

---

## 1. PPL gap 消失的完整分解（三因叠加）

1. **噪声大**：small model + few steps + heavy-tailed data，seed 间 PPL 方差 ±1–3。
2. **效应本就小**：在已是 content-aware routing + self-specialized experts + MoE-faithful 主干的系统上，controller 只能做"二阶修正"，理论收益本就是 o(1)（~0.05–0.1 PPL），不是被噪声吃掉的 0.5。
3. **指标错配**：PPL = 终点分布；controller 优化的是路径质量 / 覆盖公平 / 稳定性。改变了训练过程结构，未必改变 stationary point 的 likelihood。

---

## 2. 信号的噪声分解 → 只有 coverage 可辨识（最 sharp 的一层）

对每个 expert，观测信号 `ŝ_i = s_i + ε_i`，controller 做 `rank_k(ŝ_i)` 选 top-k。

| 信号 | 性质 | 噪声 Var(ε) | 可辨识性 |
|---|---|---|---|
| **deficit / coverage**(usage − update_count) | 精确计数 | **0** | **永远可辨识，扩到任意 N** |
| leverage(‖dW‖) / learnability / activation | 估计量 | 大 N + 小 batch 下 ≫ Var(s) | rank ≈ 随机置换 → **不可辨识** |

- 解释 3.1：v4 coverage 缩放有效，是因为它放大**唯一无噪的项**。
- 解释 3.2/3.3：importance≈random 必然，因为剩下的"价值"信号在真实大 N 下排序不可辨识。
- **v5 设计指令**：scale 上 controller 几乎纯靠 coverage（确定性 deficit 轮转），leverage/learnability 降为极小权重 tie-breaker。

---

## 3. 指标校正：controller 该测什么（路径/可行性，不是 PPL）

**controller 真实评判指标（低噪声、可辨识）：**
- **覆盖均匀度**：累计更新次数 min/mean、Gini、**dead-expert 比例**。纯计数，importance vs random 可直接比。
- **稳定性**：per-expert loss 方差、更新范数尾部、极稀疏下发散/NaN 率。
- **收敛效率**：到阈值步数 / PPL-步数曲线下面积（样本效率），非终点 PPL。

**PPL 降级为 sanity check**：只问"学没学动 / 没发散 / 过没过 bigram"，不再是 controller 的目标函数。

> 待验证的关键预测（Phase B 第一个该测）：覆盖均匀度上 importance(deficit 调度) 应**清晰可辨识地**优于 random（random 选 k 因"生日碰撞"重复更新刚训过的、饿死其余；deficit 轮转不会）。这是 controller 唯一该有、也该能看见的硬差异，且正是"极稀疏不掉质量"的成因。

---

## 4. 真正的贡献：training-time update sparsity（被低估的硬通货）

- 现有 MoE = sparse forward（省推理）+ **dense backward**（激活专家全更新）。
- 本工作 = **sparse learning**：`k_update ≪ N_activated`，每步只更新 ~1.6% 专家，质量不掉（3.3 实测 imp@k=1 ≈ imp@k=16）。
- **使能条件**：① 零梯度**局部规则**（每专家更新独立 → 可跳过）；② **coverage 调度**（跳过不永久饿死）。
- → controller 不是可选优化，是**使能机制**；没它，极稀疏更新会因专家饿死而崩。

---

## 5. 定稿核心 claim

> **局部规则使 MoE 的「更新维度」可极度稀疏化（k_update ≪ N）而不损质量，前提是覆盖感知调度防止专家饿死；调度器的价值在于可行性与效率（约束满足），而非终点 likelihood——后者对更新策略高度鲁棒。**

正向断言（证明可行性）+ 把 PPL-null 变成支撑证据（basin 鲁棒）。

---

## 6. 分层定稿：哪些 confirm / 哪些被部分否定

**已确认（confirmed）：**
- MoE-faithful（专家在 FFN 主路）+ 内容路由 + 专家特化：toy → real 都成立（coherence 真实数据 +0.40，比合成更强）。
- 极稀疏更新可行：k_update ≪ N 不损质量（real data 实测）。
- coverage 在大 N 下是刚性约束、不是调味项（无噪声、可辨识、随 N 必须放大）。

**被部分否定（refuted）：**
- "controller 显著提升 PPL"：在现实 Kaggle 条件（real + 小模型 + few steps + 大 N）下不现实——指标错配 + 效应本就小 + 噪声。

**方法论（加分）：**
- toy → real 的 controlled-to-natural transfer：toy 排架构混淆(B)、Stage3 确认 specialization 是 mechanism-level truth、再带 real 看 degradation。比"直接上真实数据迷路"干净。

---

## 7. Phase A 全局 schema

**7.1 算法依赖（前提）**
- 架构：MoE-faithful、多专家、稀疏路由。
- 路由：内容相关 key + EMA-prototype 聚类 + capacity/reroute 防 collapse。
- 学习：零梯度局部规则（per-expert local loss）。
- 调度：controller 读 per-expert 信号(error/usage/deficit) + 可调 update 预算接口(k_update)。

**7.2 监控指标层次**
- 性能(sanity)：PPL train/val vs unigram/bigram。
- 结构：coherence/purity、routing entropy、expert usage、prototype 漂移。
- 调度(controller 主指标)：coverage 均匀度/Gini/dead-expert、importance vs random/uniform gap(仅 toy 看趋势)。
- 资源：k_update/N、每步更新专家比例、(Phase B) 峰值显存、step 时间。

**7.3 实验构造与假设**
- B/Stage3/4：验证 basic mechanism + controller 在理想 toy 的效果。
- 3.1：扩展性 + 暴露 coverage scaling 问题。
- 3.2/3.3：真实数据 + 大 N + 紧预算下哪些 survive（特化、稀疏更新）/ 哪些沉入噪声（PPL 微差）。

---

## 8. 对 Phase B（4B Kaggle）的定位

- **Headline 不押 PPL 提升**。两条腿：① 零梯度特化 MoE 真能学（特化 + 内容路由，已验证）；② 4B 常驻、每步训练 FLOPs ∝ k_update（与 N 解耦）→ 击败 BP memory/speed（BP 训 4B 在 T4 OOM）。
- **controller 的证据用覆盖/稳定性指标呈现**，不用 PPL gap。
- **评测**：PPL 做 sanity（WikiText-103，证明能学）；系统指标(峰值显存/step 时间/FLOPs)做 headline；覆盖均匀度/dead-expert/稳定性做 controller 必要性的证据。
- **controller v5**：coverage-dominant（deficit 轮转），value 信号极小权重。
