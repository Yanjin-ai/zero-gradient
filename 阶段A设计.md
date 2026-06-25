# 阶段 A 设计：MoE-faithful nano（锁定）

> 由 B 实验判决进入（见 `EXPERIMENTS.md` E07–E08）：决定性 importance>random 需要**专家特化**，而特化需要**内容路由 + 专家在关键路径**。
> 严格区分：**router = forward 结构，决定谁参与前向**；**controller = 训练时，决定谁重点更新**。两者分开。

---

## 决策 1（锁定）：content routing —— EMA-prototype 最近邻（在线 k-means 式）

零梯度、内容相关、可特化。

- **routing key**：`k_t = P · h_t`，`P` 固定随机投影（d→d_key）；`h_t` = MoE block 输入 hidden（已含注意力聚合的上下文/topic）→ 路由天然内容相关。
- **prototypes**：每个专家一个 centroid `c_e`（key 空间，随机初始化）。
- **路由**：token 取到 prototype 最近的 **top-k** 专家（cosine 或欧氏）。
- **prototype 更新**（零梯度）：`c_e ← (1−η)·c_e + η·mean(被分配到 e 的 keys)`（在线 k-means）。诱导特化：centroid 漂向一类输入 → 专家只见同类样本。
- **负载管理**：每专家每 batch capacity 上限；溢出 reroute 到次近；prototype 轻微排斥防坍塌；controller 的 coverage/deficit 项兜底。

**淘汰项**：固定哈希（无特化，B 已证）；learned softmax router（要 BP，禁）。

## 决策 2（锁定）：薄密集主干 + 专家做关键路径主力 FFN

- **主干（薄）**：embedding + 注意力（冻结/薄，做上下文混合）+ 一个薄的常驻共享变换 = **质量地板 + 数值稳定**。
- **专家（主力）**：1–2 个指定 MoE block 把 FFN 换成 N 个专家；token **必经**被路由的专家（无可绕过的全强度旁路）；专家承载绝大多数参数（4B 在此）。
- **关键**：主干**故意做薄到单独解不了任务**——避免上一轮"强主干把专家挤成噪声"的陷阱。主干给地板，专家抬升+特化。

---

## nano-MoE 形态（起步配置）
- backbone：tiny Transformer block，保留 attention（薄）。
- 在 1–2 个 block 把 FFN 改成 MoE FFN。
- experts：8 或 16 起步；top-k：top-1 或 top-2。
- controller：v3 coverage-aware（`lam_cov=1, lam_lev=0.8, lam_learn=0.8, lam_act=0.2, lam_cost=0.2`），只控**更新预算**（哪些 routed experts 真更新、update_scale、local_iters、cache_tier），不碰 forward 路由。

## 预算可行性（4B）
每步训练成本 **∝ k_update（被选中真更新的专家数），与专家总数 N 无关**。专家越多只增加常驻显存（fp16 摆着），不增加每步训练量。→ 4B 常驻 + 稀疏激活能进 3h/T4。

---

## 阶段化验证门（对齐 stage 3→5）

**stage 3 — 内容路由 + 专家，controller 先关**：先验证机制本身。
- 专家在主路径、不可被 backbone 绕过；
- 路由内容相关（不是固定哈希）；
- **专家出现特化**：监控每专家接收 token 的 topic 分布、activation 统计、输出差异、对不同 token 群的 loss 贡献差异。看不出差异 → controller 无东西可分。

**stage 4 — 接入 controller v3**：
- importance vs random vs uniform；
- coverage-aware 是否 > 集中更新；
- 目标：**importance > random 决定性成立**（这次有特化撑着）。

**stage 5 — scale 设计**：保持结构比值（深度/宽度、attention/MLP、稀疏专家预算/主干预算、controller 控制层比例、state-cache/参数比），放大"同类系统"而非机械放大参数。

## 新增监控（仪表盘要能看到）
- 每专家的 token 分配分布 / topic 纯度（特化度量）；
- prototype 漂移轨迹 / 专家覆盖率；
- 路由熵、capacity 溢出率、reroute 率；
- 沿用：importance vs {random,uniform,fixed_topk}、coverage、score 稳定性、4 条件闭环。
