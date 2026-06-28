# C.1 在 4B 上的后训练 — Kaggle 运行说明

> 目标：在真实 4.160B BPE+MLP 提交版（test ppl ≈1391）上，做 **head-only 零-autograd 后训练**，验证零 BP 预训练表示的下游可用性。本机已验证全链路（见 §结果预期）。

## 两步流程

### 步骤 1：产出 4B best checkpoint（`ZG_CKPT=1`）
在现有提交 kernel（`kaggle_run`，4B BPE+MLP）上开启 checkpoint 落盘，**其它 env/config 全不变**：

- 在 notebook **最前面加一个 cell**（在主代码 cell 之前）：
  ```python
  import os; os.environ["ZG_CKPT"] = "1"
  ```
  （主代码在模块顶层读 `ZG_CKPT`，所以必须在它之前设置。）
- 正常跑完整 4B 预训练（~54min 早停）。训练**结束时**自动把 best 模型写到 `/kaggle/working/best_ckpt.pt`（~8GB；CPU-RAM 暂存、末尾一次性落盘，训练中无 I/O 抖动）。
- `*.pt` 已在 `.gitignore`，不会进代码库。
- 跑完后把该 kernel 的 output（含 `best_ckpt.pt`）**保存为一个 Kaggle Dataset**（例如 `yanjinli2001/zerograd-4b-ckpt`）。

### 步骤 2：C.1 后训练 kernel（不重训 LM，只训头）
- 目录 `kaggle_c1/`：`c1_kernel.ipynb`（自包含，`%%writefile` 内嵌 `kaggle_zerograd_moe.py` + `c1_4b.py`）+ `kernel-metadata.json`。
- **在 `kaggle_c1/kernel-metadata.json` 的 `dataset_sources` 里补上步骤 1 的 checkpoint dataset slug**（现在只挂了 WikiText）：
  ```json
  "dataset_sources": ["vadimkurochkin/wikitext-103", "yanjinli2001/zerograd-4b-ckpt"]
  ```
- 重新生成 notebook（若改了代码）：`python3 build_c1_kernel.py`。
- 推送并运行：
  ```bash
  kaggle kernels push -p kaggle_c1
  kaggle kernels status yanjinli2001/post-backprop-zerograd-c1
  kaggle kernels output yanjinli2001/post-backprop-zerograd-c1 -p kaggle_c1/out
  ```

## C.1 kernel 做了什么（`c1_4b.py`）
1. 从 `/kaggle/input/**/best_ckpt.pt` 载入，按 blob 里的 cfg **重建 4B 模型结构**并 `load_state_dict`。
2. `build_data(cfg)` 用**和 checkpoint 相同的确定性 BPE tokenizer**（从 WikiText 重建）→ 既给 LM ppl 评估，也给任务编码器 `_encode`。
3. 构造**词表重叠的组合情感任务**（真实词："the {subj} is [not] {word}"，sentiment = polarity XOR negation，不能靠 vocab 切分），用模型自己的 tokenizer 编码、左 pad 到 seq_len。
4. 挂一个 **2 层 MLP 任务头**（与本机一致，手算闭式反向、零 autograd），**只更新任务头**；另跑一个线性头作对照。
5. 监控：sentiment acc（+否定/非否定子集分解）、majority 基线、**WikiText LM ppl 前后对比**（head-only → 应零遗忘）、determinism、零 autograd。
6. 写 `/kaggle/working/c1_run_summary.json`。

环境开关：`ZG_C1_STEPS=<int>`（头训练步数，默认 1000）、`ZG_CKPT_PATH=<path>`（手动指定 checkpoint）。

## 结果预期（本机已验证的对照）
- **机制成立（in-domain）**：`c1_sentiment.py` —— base 训过情感域时，MLP 头 **100%**、线性头 52.8%（XOR 线性不可分）、零遗忘。
- **off-domain 警示**：`c1_4b.py` 对小合成/小BPE checkpoint（base 没学过 good/bad）→ 近 chance（57%）。**这正是为什么 4B 必须用真实词**：WikiText 4B 学过 good/bad/not，表示**有望**线性编码这些原子特征 → MLP 头组合出情感。
- **4B 是真正的实验**：若 acc 明显超 majority 且 LM ppl 零退化 → 证明"零 BP 4B LM + head-only MLP 头后训练"完整成立；若仍近 chance → 说明 WikiText 表示对这种组合情感的零样本可分性不足（也是诚实结论）。

## 结果回写
跑完后把 `c1_run_summary.json` 的关键字段（任务类型/head-only/MLP 头/sentiment acc/LM ppl 前后）补进 `项目总档案.md` §6（Phase C.1）与本说明，并标注与本机 100% 的关系。
