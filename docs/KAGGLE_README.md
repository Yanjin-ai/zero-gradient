<!-- **English** | [中文](KAGGLE_README.zh.md) -->
**English** | [中文 →](KAGGLE_README.zh.md)

# Post-Backprop Challenge — ZeroGrad MoE (pure ZeroBP 4B)

**Submission notebook top-matter / reviewer README.** This entry trains a **4.16-billion-parameter**
language model with **zero global backpropagation** — no `torch.autograd`, no `loss.backward()`, no
optimizer — under a single T4 GPU. Every parameter update is a hand-written **local rule**.

---

## What this is

- **Model:** a content-routed **Mixture-of-Experts** LM. `d=1024`, `|V|=32000` (BPE subword), `seq_len=64`,
  `n_layers=4`, `n_experts=950/layer` (top-2 routing), deterministic budget of `k_update=4` experts per
  block per step. **4.160 B fp16 resident parameters.**
- **Training (ZeroBP):** runs entirely under `torch.set_grad_enabled(False)` (asserted by a gate). Experts
  are updated by local rules driven by per-block deeply-supervised readout signals; routing is a
  non-differentiable content assignment (EMA prototypes + capacity). A single frozen random attention layer
  ("reservoir") feeds a last-position representation into the routed blocks.
- **Readout / schedule:** 2-layer MLP readout head + a cosine LR schedule with routing-freeze and early stop
  ("Phase C").

## Results (WikiText-103)

| Configuration | Test perplexity | Notes |
|---|---|---|
| BPE + 2-layer MLP head, early stop (t\* ≈ 54 min) | **≈ 1391** | budget-efficient |
| BPE + 2-layer MLP head, full budget (~2.9 h) | **≈ 1355** | monotone, no early stop |
| (reference) unigram baseline | — | model is ~51% better |
| Small-config smoke default | `final_ppl = 6.251` | deterministic, for CI/gates |

Peak GPU ~8.3 GB, ~8546 tok/s on T4 — comfortably within a single-T4 / <3 h budget.

## Purity & gates (why this is a valid zero-backprop entry)

The default run asserts a gate suite; a valid entry must pass:

- **Zero autograd** — `not torch.is_grad_enabled()` asserted in the training path.
- **Resident params ≥ target**, **monotone loss**, **val-ppl < unigram**, **deterministic (re-run
  identical)**, **no late drift**.
- The submission source contains **no** real `.backward()` / `autograd.grad` / `enable_grad` calls (the only
  textual match is this purity statement in a comment), and **imports no research/BP scripts**
  (`phase_e*`, `phase_h/*`, `v2_*`, etc.). All research flags default to no-ops.

> Note: on the tiny smoke config, 6/7 gates pass — the one "failing" gate (`BP-4B would OOM on T4`) is a
> *demonstration* gate that only triggers on the full 4B run; it is expected to be inert on the smoke config.

## Reproduce

```bash
# Full submission notebook (embeds the current trainer, pure-ZeroBP default path):
python3 build_kaggle_kernels.py          # regenerate the submission notebook from kaggle_zerograd_moe.py
kaggle kernels push -p kaggle_run        # push the official submission kernel (attach WikiText-103)
kaggle kernels status  yanjinli2001/post-backprop-zerograd-moe
kaggle kernels output  yanjinli2001/post-backprop-zerograd-moe -p kaggle_run/out

# Local integrity check (no GPU needed):
python3 kaggle_zerograd_moe.py           # default path -> final_ppl 6.251, zero-autograd PASS, deterministic
python3 selfcheck.py                     # reset/readout guard PASS
```

Outputs: `run_summary.json` (gates, `wikitext103_test_ppl`, memory, tok/s), `loss_curve.png`,
`memory_profile.png`.

## Scope

This is the **pure-ZeroBP baseline** entry. All backprop-based research (capability-boundary studies and the
Phase H trainable-attention control) lives in separate scripts that the submission never imports; see
`SUBMISSION.md` for the full integrity checklist and `PAPER_DRAFT.md` for the complete study.
