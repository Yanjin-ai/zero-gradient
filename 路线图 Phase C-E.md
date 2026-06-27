# 路线图：Phase C → D → (C.1) → E

> 四原则贯穿不变：**MoE-FFN（专家在主路）+ EMA 内容路由 + 零梯度局部规则 + training-time 稀疏更新**。
> 已有 baseline 定论（原设定，实测 T4）：系统两条腿坐实；质量是诚实弱 LM，best WikiText-103 test ppl ≈1437 出现在 ~50min/~6000 步，之后漂移退化（无早停/lr 衰减）。

## Phase C — 锁定最佳点 + 稳定（当前）
**目标**：不改大结构，把训练稳定到最佳点 t* 附近，避免走进退化区。
- **lr schedule**：warmup + 余弦衰减（后半程明显降 lr），防止"后期高 lr 破坏已学结构"。
- **早停 + best checkpoint**：每 M 步 val 评估，记录 best PPL + 步数；连续 K 次无新低 → 早停（停在最佳附近）。
- **路由稳定化**：后期降低/冻结 prototype EMA，减少路由 churn → 减轻专家重分配的类灾难性遗忘。
- **不做**：不改词表、不重构读出头、不加新局部损失。
- **指标面板**（复用 + 2 个辅助）：val/test PPL；MoE 特化（routing entropy/coherence）；资源（显存/tok·s⁻¹）；稳定性（loss/NaN）；**新增** 路由漂移度（相邻评估期路由分布的 cosine/KL 距离）、checkpoint 历史（每次评估的 best PPL + 步数）。
- **输出**：一条稳定训练路径，在 T4/3h 内到达并停在最佳 LM 状态，明确 t*，并证明漂移被压住。

## Phase D — 结构/损失/词表增强（提升能力）
**目标**：在已稳定的框架上提升表达力与信号质量。换基准，需新一轮 sanity check。
- **词表**：32k word-level → BPE/SentencePiece（PPL 更可读、更细粒度）。
- **读出头**：单层 d×V → 更强（多层 MLP / tie weights / adaptive softmax），缓解大 vocab 瓶颈。
- **局部损失扩展**：加对比/句级属性/局部 reconstruction 强化表示；视资源逐步开放部分 attention 可训练（用局部/替代规则）。
- **输出**：更低、更有意义的 PPL + 更强结构指标。

## Phase C.1 — 基于 best checkpoint 的初步后训练（可与 D 并行）
**前提**：Phase C 已有稳定 best checkpoint。
- 从锁定的 best checkpoint 出发，用同样局部规则在新域/小任务上做少量额外训练；或顶层挂局部任务头只训练任务头。
- 监控 PPL + 任务 metric，看后训练是否改善、是否严重遗忘。
- 判断现有零 BP MoE 的下游潜力 + 后训练中哪些模式稳定/危险。

## Phase E — 混合训练探索（长期研究，超出当前 Kaggle scope）
- pretrain 继续零 BP MoE + training-time 稀疏；finetune 在少数层/adapter 引入 BP 做精细优化（BP 只在顶层/关键层，主干仍局部规则稀疏更新）。
- 命题：在保持训练效率与结构优势前提下，引入少量 BP 提升最终 LM 质量。

## 核心评估理念（贯穿）
- 3h = 最大预算（证明能在此约束下跑 4B，BP 不行）。
- 真正有意义的是：**最低 PPL 出现在哪个时间/步数（t*）**，以及**如何用 schedule + 早停在 3h 内对齐到 t* 并避免退化区**。
- 不依赖 BP，用合适指标监控"到最低点了没"。
