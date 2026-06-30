# ARCHITECTURE — v1.0 固定骨架 vs v2.0 可改区（边界定义）

> **用途**：明确**什么是不可动的 v1.0 骨架**（提交版 + Phase E/F 研究都基于它），**什么是 v2.0 才允许动的结构**。任何 v2.0 改动 = **non-submission / research-only**，必须新建独立路径、**不破坏默认 `final_ppl 6.251` / 零 autograd**。
> 标注：**【LOCKED】**=现状事实/不可动；**【v2.0 OPEN】**=允许在 v2.0 研究分支试改。最后更新 2026-06-30。

## 1. v1.0 数据流（当前骨架，代码事实）
```
token x ──► E[x] + pos                                  (embedding，词表 32000，d=1024)
        ──► 冻结随机因果 attention (Wq,Wk 不训练)          (单 attention，reservoir)
        ──► h = (emb + att@emb)[:, -1]    ← 【末位塌缩】整个序列 → 一个 [B,d] 向量
        ──► 4×堆叠 MoE block：每块 内容路由(EMA-prototype+capacity) → 950 专家 取 top-2
        ──► 每块 deeply-supervised 读出头(2 层 MLP) + 确定性 round-robin 预算调度(每块每步更新 4 个专家)
d=1024, V=32000, seq_len=64, n_layers=4, n_experts=950/层, k_route=2, k_update=4 → 4.160B fp16
```

## 2. v1.0 固定骨架（LOCKED，不可动）

| 组件 | 现状 | 为什么锁 |
|---|---|---|
| **末位塌缩 readout** `h=(emb+att@emb)[:,-1]` | 序列塌成单向量 | 是当前架构身份；**疑似关系任务瓶颈**（见 MASTER_ARCHIVE §5），但改它 = v2.0 |
| **单 attention（Wq/Wk 冻结 reservoir）** | 不训练 | Track A 证明 ZeroBP 局部规则训不动它；改成可训练 = v2.0 |
| **MoE-FFN 专家 + EMA 内容路由 + training-time 稀疏** | 严格 ZeroBP 局部规则 | **核心算法创新**；提交版主体，绝不可换成 BP |
| **确定性 round-robin 调度** | k_update=4 | 可复现 + 最坏 backlog 上界（controller 已被否） |
| **默认训练循环 + 提交配置（bpe+mlp+Phase C）** | 零 autograd，6.251/1391 | **提交完整性红线** |

> **【LOCKED】v1.0 研究分支（Phase E/F）允许的范围**：只在 **embedding / attention 参数 / 任务头** 上加**默认-off 的少量 BP**（autograd 图只覆盖这些 + 被路由专家），**不改上面的架构骨架**、不进提交、不破坏 6.251。这条线已穷尽：少量 BP 对 bag 组合有效（情感 79%），对关系/多步有限且不转移 4B。

## 3. v2.0 可改区（OPEN，research-only）

| 方向 | 【v2.0 OPEN】允许动什么 | 目的 | 先验门槛 |
|---|---|---|---|
| **不塌缩序列读出（首选）** | 把 `[:, -1]` 换成 pooling / 多位置 / 注意力池化读出头；blocks 读多位置而非单向量 | 保住跨位置信息，直攻关系/多步瓶颈 | 小配置 NLI/算术 zero-shot 是否松动；最便宜最对症 |
| **真正可训练 attention** | Wq/Wk(/Wo) 可训（ZeroBP 新规则 或 v2.0 少量 BP） | 跨句对齐、长程依赖 | 单调/确定/route-drift 不爆；句对小任务提升 |
| **更深 / 更多层 BP** | 顶层多个 block 的 BP（超出 Phase F 顶层 1–2 边界） | 顺序计算能力 | 小配置算术是否脱离 chance |

## 4. v2.0 硬规则（必须遵守）
1. **【LOCKED】不污染提交版**：v2.0 改动一律新建独立文件/路径（如 `v2_*.py`），默认 off；`kaggle_zerograd_moe.py` 默认路径必须仍 `6.251` / 零 autograd / 7 门。改后必跑 `python3 kaggle_zerograd_moe.py` + `selfcheck.py` 复核。
2. **【LOCKED】单杠杆 + 小配置先行**：每次只动一个结构维度；小配置过门（任务提升 + 稳定性 + 确定性 + reset 自检）才迁 4B。
3. **【LOCKED】记录分层**：v2.0 结果标 **research-only**，写 EXPERIMENT_LEDGER + ADR；不得改写已锁定的 v1.0 结论。
4. **【PROPOSAL】命名**：v2.0 分支 checkpoint / kernel 用新名（不覆盖 1355 提交 checkpoint）。

## 5. 一句话边界
**v1.0 = 末位塌缩 + 单冻结 attention + ZeroBP MoE，骨架不动（提交版纯 ZeroBP）；v2.0 = 允许动 readout / attention / 更深 BP 这三类结构归纳偏置，但仅 research-only、小配置先行、永不破坏 6.251。**
