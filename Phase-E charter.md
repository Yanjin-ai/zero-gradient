# Phase E — Hybrid BP（混合反向传播）Charter

> **状态**：charter 草案（2026-06-29），**未写代码**。等签字 + scope 确认再实现（遵守 charter-before-code 纪律）。
> **前置已满足**：v1.x 封存（见 `项目总档案.md` §1.5）；v1.1 + D.2 已证明**纯零 BP 后训练在 4B 组合任务上 plateau ~62%**。

## 1. 动机（来自实证）
跨 6 种零 BP 后训练路线全部卡在 ~60–62%：任务特征无法从冻结表示提取，零 BP 局部规则无法在适配时把新特征装进表示而不遗忘。**Phase E 的研究问题**：在适配阶段引入**少量真实 BP**，能否突破 ~62% 天花板，且遗忘可控？

## 2. 合规边界（重要）
- Phase E **故意打破"零全局 BP"性质** → 是**研究分支**，**不是竞赛提交版**。
- **竞赛提交版永远零 BP**（6.251 / 4B 1355 不变）。Phase E 用独立 flag（默认 off）+ 独立 kernel + 独立文档，绝不替换提交版。
- Phase E 回答的是科学问题"少量 BP 买到什么"，作为零 BP 的对照上界。

## 3. 单杠杆（首发）
- **只对顶层 1 个 MoE block + 任务头开 autograd**；embedding、下层 block、EMA 路由、attention **全冻**。
- 预训练保持现有零 BP（不动）；只有**适配阶段**对这一小部分启用 `torch.enable_grad()` + 真实 `loss.backward()` + 手动 SGD（仅更新这两组参数）。
- 实现要点：全局 `torch.set_grad_enabled(False)` 需在适配段对目标参数局部开 grad（`requires_grad_(True)` + 局部 autograd 上下文），其余张量 detach。

## 4. 小配置 gate（过门才迁 4B）
对照基线（小配置，组合情感）：零样本 ~51% / 2.1 冻头 92%/+11 / 2.2 分区 96%/+0 / D.2 61%/+0。
- **G1 任务**：Mixed-BP 顶层适配 acc **明显 > 零 BP 最好的可比档**（即 > D.2 的 61%，目标向 2.1 的 92% 看齐或更高）。
- **G2 遗忘**：原域 PPL 的 ΔPPL **不比零 BP full-adapt 更糟**（理想接近冻头/分区档）。
- **G3 量纲**：明确 BP 触及的参数量、额外显存/算力（必须远小于全模型 BP；这是"少量 BP"的意义）。
- **G4 纪律**：单杠杆（只这一处改动）、确定性、小配置先行。
- 失败（连 G1 都过不了）→ 记负结果：在当前架构下少量顶层 BP 也不够，需更大改动或换路线。

## 5. 4B 实验（gate 过后，用现有 orchestrator）
- 新 kernel `kaggle_hybrid/`（或复用 adapt kernel + `ZG_HYBRID_BP=1`）：从 1355 checkpoint 载入 → 顶层 1 block + 任务头 BP 适配 → 测 acc + WikiText ppl 前后。
- **四档对照表**（最终产物）：
  | 路线 | 4B acc | 遗忘 ΔPPL |
  |---|---|---|
  | Zero-BP ZeroShot | 60% | 0 |
  | Zero-BP Adapt(freeze-head) | 62% | +3.4 |
  | Zero-BP Partition | 61% | 0 |
  | **Mixed-BP（本阶段）** | ? | ? |
- 判定：Mixed-BP 是否在"任务提升 vs 遗忘"上给出**明显优于所有零 BP 档**的操作点。

## 6. 阶段顺序（与 D.2 的关系）
- D.2（ZeroBP-DeepSignal）小配置已跑：plateau 61%，gate 失败、不迁 4B（见档案 §1.5）。→ 给了零 BP 公平的最后机会，仍突破不了。
- 因此 **Phase E 优先级提升**：D.2 的失败正是 Phase E 的直接动机。
- 若 Phase E 首发（顶层 1 block + 头）过门 → 可选 stage 2：顶层 2 block；或对照 D.2 的 4B 点。

## 7. 升级单杠杆（备选，若首发不够）
顶层 2 block + 头 BP；或 BP 仅作用于"新增任务专家"（结构分区 + BP，把分区思想与少量 BP 结合，遗忘更可控）。每次仍只改一维度。

---
**待你确认**：① 首发单杠杆 = 顶层 1 block + 任务头 BP（推荐）是否 OK；② gate 阈值（G1 目标 acc、G2 遗忘上界）是否按上面定；③ 是否现在就实现小配置 Phase E，还是先把 v1.x 推上 GitHub 收尾再开工。
