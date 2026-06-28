# Track A 设计文档：可训练 Attention（零 autograd 局部规则）

> **状态**：设计阶段（2026-06-28）。本文件列出候选局部规则、推导、风险与 sanity 计划，**先列候选再写代码**（遵守档案 §13.2 纪律）。代码实现与小配置结果将在通过 sanity 后追加。
>
> **单杠杆约束**：本 track 唯一变量 = attention 是否可训练。保持 BPE + MLP 头 + Phase C schedule 不变，与 1391 基线干净对照。

---

## 0. A1 立项冻结（CHARTER — 编码前必须先确认这一节）

> 纪律：在改 `kaggle_zerograd_moe.py` 前，先冻结「改动范围 / 过门标准 / 失败回滚边界」三件事。未确认本节不写代码。

**0.1 最小改动范围**
- 训练对象 = **唯一那个 attention 的 Wq + Wk**（`context()` 内，行 179）。**不含 Wo**（当前无 value/output 投影；加 Wo = 新参数 = 另一个结构杠杆，违反单杠杆，A1 不做）。
- 单步局部信号 = **block-0 readout 的 `dh₀`**（见 §2、§3-A1 推导），不串联多块。
- **默认关闭**：新增 `ZG_ATTN`（默认 0），必须 `ZG_ATTN=1` 才开启 attention 训练；提交默认路径（bpe+mlp+Phase C）**字节不变**。
- 更小回退面 **A4**：若 Wq+Wk 不稳，退到**只训 Wq**（Wk 保持冻结）；仍不稳叠 **A3**（attn 专属小 `η_attn` / 晚开放）。

**0.2 small-config 过门标准（全过才允许迁 4B）**
- 对照锚点 = 冻结-attention 小配置基线 **final val_ppl 6.25**（word/linear 合成，已实测、确定性）。
- ① 稳定且可复现地**优于基线**：`ZG_ATTN=1` 的 final val_ppl **≤ 6.20**（明显且非噪声）。
- ② 确定性不破：两次运行 final val_ppl 差 **< 1e-6**（现基线满足）。
- ③ 零 autograd 不破：`autograd_used == False`。
- ④ 单调性保持：val_ppl 单调下降、无发散、无 NaN。
- ⑤ 路由不被扰乱：`route_drift` 不显著高于冻结基线（基线末端 ~0.001）。

**0.3 失败回滚边界**
- 若**连续 2–3 个候选局部规则**（A1 → A4/A3 稳定化变体 → 视情况 A2）在 small-config 上都无法稳定优于 6.25，**暂停 Track A，转 C.1**。
- 代码回滚：所有改动锁在 `ZG_ATTN` 开关后，默认路径永不受影响；实验 commit 可直接 `git revert`；**默认 CONFIG 在 4B run 证明增益前绝不改动**。

**0.4 联动规则（三轨共用）**
- **共用 master baseline** = BPE+MLP 提交版 **1391**（4B）/ 6.25（small）。每个实验只许相对它改一个维度。
- **共用 gate** = §0.2 五项（zero-autograd / 单调 / 确定性 / route-drift / 优于同类 baseline）→ 过门才上 4B。
- **共享 checkpoint** = `best-pretrain-vCurrent`（现=1391）。A 成功且更优更稳 → 升级 `vNext`。**C.1 永远从当前 best-pretrain 起步**，不从任意实验分支起步。
- **依赖关系**：B 大部分**不**依赖 A（B1 可在 1391 上独立做；C.1 只依赖稳定 checkpoint）；只有 "attention-aware 局部损失"（B1.5）才依赖 A 先训起来。A 成功是 B/C.1 的**更强底座**，非前置门槛。
- **C.1 的硬依赖 = checkpoint 落盘/重载**：当前 notebook **不产出可重载 4B checkpoint**，所以 C.1 第一工程任务**不是任务头，而是 best checkpoint 的 save/reload**。在 A 推进的同时并行铺这条低风险准备线。
- **B1 暂不编码**：D-2 刚失败，不立刻再赌局部损失；只整理候选池，等 A1 第一轮结果出来再决定。

---

## 1. 现状（被冻结的 attention）

`kaggle_zerograd_moe.py` 的 `context()`（行 194–202）：

```
emb = E[x] + pos                       # [B,T,d]
q   = emb @ Wq ; k = emb @ Wk          # Wq,Wk 冻结随机 reservoir（行 179）
sc  = q @ kᵀ / sqrt(d)                 # [B,T,T]，因果 mask
att = softmax(masked sc)
h   = (emb + att @ emb)[:, -1]         # [B,d]，取最后位置作为 context 向量
```

`h` 喂给 4 个 MoE block。**Wq/Wk 全程不更新** → attention 只做固定的随机上下文混合。

**【判断】** 类比 D-1「单层线性头是隐藏瓶颈，加一层 MLP 榨出 30%」：冻结随机 attention 很可能是下一个真实结构瓶颈——它决定了喂给所有专家的上下文表示质量，但目前完全不学。这是当前剩余 EV 最大的结构杠杆。

---

## 2. 关键约束：什么算「合规的局部规则」

竞赛红线 = 不调 `torch.autograd` / 不 `.backward()` / 不跨层全局 BPTT。**手算闭式、单步、deeply-supervised 的局部更新是允许的**（softmax-CE 的 `δ=p−onehot` 即标准技巧）。

**已有先例**：当前代码的 embedding 更新（行 344）已经用 block 的 readout 信号 `dh` 做近似局部信用分配（单步，不跨多块真实 BPTT）。attention 局部规则应**同性质**：用某个 block 的 deeply-supervised readout 信号，单步手算回传到 Wq/Wk，**不串联 4 个 block**。

**最干净的信号源 = block-0 的 readout `dh₀`**：block-0 的输入正是 `context()` 的输出 `h`，所以 block-0 的局部头给出的 `dh₀ = ∂loss₀/∂h` 是训练 attention 最自然的单步信号——与项目的 deeply-supervised 哲学完全一致，且只跨「context→block0 头」一步，不跨多块。

---

## 3. 候选规则

### A1（推荐首选）：block-0 readout 信号 → 闭式 attention 梯度

**思路**：取 block-0 deeply-supervised 头的 `dh₀`（已在训练循环里算出，shape `[B,d]`，对应最后位置的 context 向量），手算回传到 Wq/Wk。低风险、最对齐、复用现成信号。

**推导**（只对最后位置 `L=T-1`，因为 `h` 只取该位置）：

```
h_L = emb_L + Σ_j att_Lj · emb_j            (j ≤ L)
g_j  = dh₀ · emb_j                           # [B,T]，对每个 key 位置 j 的标量
                                             # 来自 ∂h_L/∂att_Lj = emb_j
s_j  = att_Lj · (g_j − Σ_{j'} att_Lj'·g_j')  # softmax 雅可比，[B,T]
                                             # = ∂loss/∂sc_Lj
dq_L = (1/√d) · Σ_j s_j · k_j                # [B,d]
dk_j = (1/√d) · s_j · q_L                    # [B,T,d]
ΔWq  = emb_Lᵀ @ dq_L            (batch 求和) # [d,d]
ΔWk  = Σ_j emb_jᵀ @ dk_j        (batch 求和) # [d,d]
Wq  -= η_attn · ΔWq ;  Wk -= η_attn · ΔWk
```

全部张量代数、零 autograd、单步、确定性。**复杂度**：`g_j`/`s_j` 是 `[B,T]`，`ΔWq/ΔWk` 是 `[d,d]` 外积——与现有 block 更新同量级，便宜。

**风险**：attention 与 MoE 路由共享 `emb` 表示；attention 训练若扰动 `emb` 的统计，可能让 EMA-prototype 路由漂移（route_drift 爆）。**缓解**：给 attention 单独的 `η_attn`（远小于主 lr）+ 监控 route_drift。

### A2：Forward-Forward goodness（正/负样本）

**思路**：不借 readout 信号，给 attention 自己的局部目标——真实上下文的 goodness（如 `‖h‖²` 或 h 与下一 token embedding 的对齐）应高于「打乱/corrupt 上下文」的负样本。手算闭式 goodness 梯度更新 Wq/Wk。

**风险/成本**：需构造负样本、设计 goodness 函数、调阈值——比 A1 复杂、超参多。**作为 A1 失败后的备选**，不首发。

### A3：attention 专属 lr / 温度（旋钮，叠加在 A1 上）

给 attention softmax 一个温度 `τ_attn`，并给 `η_attn` 单独调度（如比主 lr 小 1–2 个数量级、或更晚才开放训练）。**不是独立候选，是 A1 的稳定化旋钮**，防止和路由打架。

### A4：低风险消融——只训 Wq（Wk 冻结）/ 晚开放

只更新 Wq（保留 Wk 随机），或在 `freeze_routing_step` 之后才开放 attention 训练（先让路由稳定）。用于在 A1 不稳时缩小改动面、定位是「attention 训练本身有益」还是「与路由的交互有害」。

---

## 4. 小配置 sanity 计划（全过才允许迁 4B）

锚点 = 本机小配置 word/linear（或 bpe/mlp）冻结-attention 基线 val_ppl（Task #1 先测）。

逐个候选，严格单杠杆：

1. **正确性**：实现后先验证 `autograd_used == False`、两次运行 val_ppl 完全一致（确定性）。
2. **稳定性**：val_ppl 仍单调下降、不发散、无 NaN；`route_drift` 不显著高于冻结基线（attention 没把路由搅乱）。
3. **增益**：可训 attention 的小配置 val_ppl **明显优于**冻结基线。
4. **旋钮**：若不稳，先上 A3（小 η_attn / 晚开放 / A4 只训 Wq）再判。

**允许迁 4B 的门槛**：A1（必要时 +A3/A4）在小配置上稳定且明显优于冻结基线。迁 4B 后判读：是否把 1391 进一步压低、7/7 门是否保持、显存/吞吐是否仍在预算内、t* 在哪。

---

## 5. 实现落点（代码 map，供编码时参考）

- 加 `Wq/Wk` 更新：训练循环（行 315 `context()` 之后需缓存 `emb/q/k/att`；行 324 拿到 block-0 的 `dh` 后做 A1 更新）。
- 现 `context()` 不返回 `att/q/k/emb`——需让它在训练态额外吐出中间量（或新增 `context_train()`），推理态不变。
- 加 `ZG_ATTN=0|1`（默认 0，保持当前提交版不变）、`attn_lr`/`attn_warmup` 配置，与 §14.2 的 env 开关风格一致。
- 监控：在 curve 里加 `attn_dW_norm`（更新幅度）便于诊断。

---

## 5b. 已实测结果（2026-06-28，本机 CPU，A1 已实现并跑过）

> **重要**：A1 已由前序工作完整实现（`ZG_ATTN`/`attn_lr_scale` + `context()` cache + `attn_update` + 训练循环 wiring + `attn_dw` 监控），并已实测。**结论：A1 目前不是有效杠杆。**

**【事实】① 主小配置 = 错配指标（attention 测不出来）**：主小配置（合成 topic-prefix，seq_len 16）接近 bag-of-words，几乎不需要 attention → 冻结 val_ppl **6.251** vs 可训 **6.245** = **噪声**。这推翻了 §0.2 我原定的"≤6.20 on 主小配置"过门标准——它对 attention 是错配指标（同 D-1 的"小配置低估"，但这次是"小配置根本测不到"）。**正确的 instrument 是关联召回探针**（`track_a_probe.py`）。

**【事实】② 关联召回探针（frozen 必败的任务）**：`seq=[k1 v1 … km vm q]`，target=与 q 匹配的 key 的 value；frozen 随机 Wq/Wk 只能 chance 级。实测：

| 设定 | val_ppl (unigram 40.4) | val_acc | max_attn_dw |
|---|---|---|---|
| FROZEN-ATTN | 45.034 | **2.7%** | — |
| TRAIN lr×1 | 45.046 | 2.6% | 0.00016 |
| TRAIN lr×10 | 45.046 | 2.6% | 0.00016 |
| TRAIN lr×100 | 45.030 | 2.7% | 0.00018 |
| TRAIN lr×1000 | 44.246 | 2.4% | 0.00131 |

**判读**：
- val_acc 全程 ≈ **2.5% = chance**（1/40 个 value）；连 frozen 和 train 都**做不了召回**，且 ppl(45) **比 unigram(40.4) 还差**。整套（内容路由 MoE + 局部规则）解不了 recall，可训 attention 也没解。
- `max_attn_dw` 极小（1e-4 量级）；lr×1 与 lr×10 **结果逐字节相同** → 在这些尺度上 attention 训练对模型轨迹**几乎零影响**。只有 lr×1000 才让 attention 稍动（ppl 45.03→44.25），但 val_acc 仍是 chance。
- 零 autograd 保持（`autograd_used=False`）。

**【判断】根因**：block-0 readout 的 `dh₀` 传到 Wq/Wk 的信用信号**太弱**——`h_L = emb_L + Σ att·emb` 里 `emb_L` 直连项主导，attention 贡献只是小扰动 → 对 Wq/Wk 的梯度被稀释到 1e-4。换句话说 **A1 的局部目标设计本身让 attention 收不到有效信号**，不是 lr 问题。

**【事实】③ dead-or-savable 诊断（`track_a_diag.py`，assoc-recall 上 instrument attn_update）**：

| 量 | 值 | 含义 |
|---|---|---|
| `r_attn = ‖att@emb‖/‖emb_L‖` | **0.38** | attention **确实贡献** h 的 ~38%，**不是**可忽略扰动（推翻"emb_L 主导"的原根因猜测） |
| `‖dh₀‖` | 0.0157 | readout 信号小但非零 |
| `‖dWq+dWk‖` | 7.9e-5 | 原始 attention 梯度极小 |
| `rel_step = lr·‖dW‖/‖Wq‖` | **7e-7**（max 2e-6） | 每步相对权重movement≈0；`‖Wq‖` drift **0.00%**——Wq 根本不动 |

**【事实】④ 调大 attn lr 的判别实验**（绕过"梯度太小"，直接把 rel_step 拉到 ~1e-1）：

| attn_lr_scale | val_ppl | val_acc |
|---|---|---|
| frozen | 45.03 | 2.7% |
| ×3e3 | 44.24 | 2.8% |
| ×1e4 | 44.14 | 2.1% |
| ×1e5 | 43.75 | 2.5% |
| ×3e5 | 43.80 | 2.3% |

**【结论：A1 不是 scale 问题，是方向不足】** 即便把步长拉到健康量级，acc 仍是 chance、ppl 只微动 3% 且 ×1e5 后**饱和**。即 **block-0 readout 的闭式梯度方向，教不会 attention 做 content lookup**——与 FF/局部学习文献一致：结构性模块需要**自己的局部目标**，借来的端信号不够。A1（naive + scale-corrected 两版）= **第 1 个候选，彻底失败**。

**【判断】下一步**：A1 已充分证伪。若继续 Track A 需 A2（给 attention 自己的 goodness/对比目标，绕开 readout）——但 A2 投机、重、且无强理论保证能解决"闭式局部梯度训 content-lookup"。按用户决策框架（无明确 A2 把握则转 C.1）：**Track A 标记为"首轮 A1 充分失败后暂缓"，主线转 C.1**；A2 留作未来候选，需先有强理论 mini-charter 才启动。

> **元结论（项目资产）**：「零 BP 栈里 attention 难用局部规则训起来」本身是一个诚实、有价值的负结果（含定量诊断 r_attn=0.38 但 readout 梯度方向不足），与"头是大杠杆"形成对照——不是所有结构模块都能用同一套 readout-信号局部规则训。

---

## 6. 待办

- [ ] Task #1 装好 torch → 锚住小配置冻结-attention 基线。
- [ ] 实现 A1（context_train 吐中间量 + 闭式 ΔWq/ΔWk + ZG_ATTN 开关）。
- [ ] 小配置跑 A1，过 §4 四项门。
- [ ] 不稳则上 A3/A4；仍不行考虑 A2。
- [ ] 过门 → 迁 4B 单杠杆 run，对照 1391。结果（成败）写回总档案 §6.4 + EXPERIMENTS.md。
