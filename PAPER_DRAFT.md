<!-- **English** | [中文](PAPER_DRAFT.zh.md) -->
**English** | [中文 →](PAPER_DRAFT.zh.md)

# ZeroBP at Scale and the Architecture of Capability: A Boundary Study of a Backprop-Free 4B Language Model, with a Trainable-Attention Control

**Draft technical report — v0.1.** Target: workshop paper (8–12 pp). All numbers are traceable to the experiment ledger and per-run summaries in the accompanying repository; commit hashes are given where a result is locked.

---

## Abstract

We study, under a hard resource budget (a single T4 GPU, 3 h), how far a *backprop-free* large language model can go, and — more importantly — *why* it stops. We build **ZeroBP-4B**, a 4.16-billion-parameter content-routed mixture-of-experts (MoE) language model whose entire training uses hand-written **local update rules** and **zero global backpropagation** (no `autograd`), and lock it as a deterministic, reproducible Kaggle submission (WikiText-103 test perplexity ≈ 1391 early-stop / ≈ 1355 full-budget). On this fixed backbone we then map a **capability boundary** across three task structures that probe modern-LLM competencies — bag-style sentiment, relational entailment (NLI), and multi-step modular arithmetic — under increasing amounts of injected backprop (BP). The boundary is sharp and *structural*: a little embedding-path BP lifts sentiment from 60% to 79%, but neither a little nor a *deep* BP installs relational or multi-step structure at 4B (both stay at chance), and three "ZeroBP-native" routes (richer data, structural auxiliary targets, local attention rules) each add ≤ 2 pp. A non-collapsing-readout probe shows the relational structure is *not present anywhere* in the frozen representation.

To test whether these failures are properties of the *tasks* or of the *architecture*, we introduce **Phase H**, a strictly isolated control: a standard multi-layer trainable-attention Transformer trained with ordinary full backprop. The same synthetic tasks that cap ZeroBP at 65.7% / chance are solved to 100% by a 0.8M-parameter Phase-H model; on **real SNLI** Phase H reaches **69.97%** where ZeroBP-4B is at chance (33.4%). The picture is not a clean win, and we report it honestly: Phase H installs shallow multi-step reasoning (k ≤ 3 = 100%) but hits its *own* **scale-resistant depth wall** at k ≥ 4 (unchanged from 4.7M/6k-step to 21.3M/15k-step models); a from-scratch character LM reaches only 2% exact-match on real GSM8K; and *neither* stack does real sentiment (SST-2 ≈ chance for ZeroBP once a readout-collapse artifact is corrected). Two of our own working hypotheses were refuted by the data along the way and are corrected in the record. The result is a complete arc: **the ZeroBP relational/multi-step ceiling is architectural, not fundamental — a trainable-attention backbone crosses it on real relational data and shallow multi-step reasoning, while exposing new, honest limits of its own.**

---

## 1. Introduction

Modern large language models are trained with global backpropagation at enormous compute cost. Two orthogonal questions motivate this work:

1. **How much language capability survives without global BP, under a strict single-GPU budget?** Backprop-free / local-learning methods are attractive for memory- and hardware-constrained settings, but their capability ceiling at scale is poorly characterized.
2. **When such a model fails a task, is the failure algorithmic (not enough gradient), architectural (the backbone cannot represent the structure), or intrinsic to the task?** Disentangling these is usually impossible because researchers change many things at once.

We attack both with a deliberately layered study organized as **three lines**:

- **The submission line** (§3): a pure-ZeroBP 4B MoE LM, engineered to be a clean, deterministic, resource-budget baseline with *zero* autograd anywhere in the training path.
- **The boundary line** (§4): with that backbone *frozen as an object of study*, we measure a task × method capability matrix (Phase E/F/G), injecting increasing amounts of BP into embeddings, attention, and blocks, and testing three ZeroBP-native structural levers.
- **The control line** (§5): a strictly isolated **Phase H** stack — a standard trainable-attention Transformer with ordinary full BP — used as an architectural control to ask whether the boundary of the second line is caused by the backbone.

The contribution is not a new state-of-the-art number. It is a **controlled boundary study**: a reproducible backprop-free 4B baseline, a systematic capability matrix with a clean submission/research isolation discipline, and an architectural control that turns "ZeroBP can't do NLI" into the sharper, falsifiable claim "*this backbone* can't, and *here is a backbone that can*, and *here is where even that one stops*."

We foreground three methodological commitments that shaped the results: (i) **fair readout** — capabilities are measured with a fresh, task-agnostic readout head, because we found that an unfair readout can fabricate or destroy apparent capability (§4.4, §5.5); (ii) **single-lever changes** — each experiment moves one structural degree of freedom; and (iii) **honest correction** — two of our own hypotheses were refuted by later runs, and we report the corrections rather than the original guesses.

---

## 2. Related context (brief)

Our ZeroBP trainer sits in the family of **local-learning / backprop-alternative** methods (local losses, target propagation, forward-only and Hebbian-style updates) and of **sparsely-activated MoE** models, but with an unusual combination: a large resident expert bank whose experts are updated by hand-written local rules under a deterministic sparse budget, with a frozen random attention "reservoir" (in the spirit of reservoir computing) feeding a last-position-collapsed representation into stacked routed blocks. The evaluation axes (LM perplexity, sentiment, NLI, multi-step arithmetic / GSM8K, GLUE-SST2) are chosen to span the standard capability dimensions used to characterize modern LLMs, at a scale where controlled ablation is affordable. We do not claim novelty of any single component; the contribution is the **boundary methodology and the architectural control**.

---

## 3. The submission line: a pure-ZeroBP 4B baseline

### 3.1 Motivation and constraints

The target environment is a single T4 (16 GB) with a 3-hour wall clock and self-supplied data — a setting in which standard BP training of a 4B model is infeasible (our measurements put a BP forward+backward of this model at ~31 GB, i.e. OOM). ZeroBP sidesteps the backward pass entirely: there is no global loss gradient; every parameter update is a local rule.

### 3.2 Architecture and the zero-BP update

ZeroBP-4B is a decoder-style LM with the following data flow (code-level fact):

```
token x ─► E[x] + pos                                   (embedding; |V|=32000, d=1024)
        ─► frozen random causal attention (Wq,Wk not trained)   (single "reservoir" attention)
        ─► h = (emb + att·emb)[:, -1]     ← last-position COLLAPSE: whole sequence → one [B,d] vector
        ─► 4× stacked MoE block: content routing (EMA-prototype + capacity) → 950 experts, top-2
        ─► per-block deeply-supervised 2-layer-MLP readout head
           + deterministic round-robin budget schedule (k_update experts / block / step)
```

with `d=1024, |V|=32000, seq_len=64, n_layers=4, n_experts=950/layer, k_route=2, k_update=4`, giving **4.160 B fp16** resident parameters. Three properties matter for the rest of the paper:

- **No global gradient.** Training runs under `torch.set_grad_enabled(False)`, asserted by a gate. Each expert is updated by a hand-written local rule driven by the per-block deeply-supervised readout signal; the routing is a non-differentiable content assignment (EMA prototypes + capacity).
- **Deterministic sparse budget.** A round-robin schedule updates only `k_update` experts per block per step (0.42% of the expert bank), which we found does not degrade LM quality relative to dense updates (k=1 ≈ k=16 in nano ablations) and bounds worst-case backlog. A learned importance controller did **not** beat this random/round-robin schedule and was rejected.
- **Frozen attention + last-position collapse.** The single attention layer is a frozen random reservoir (`Wq,Wk` never trained), and the whole sequence is collapsed to the last position *before* the routed blocks. These two facts become the prime suspects in §4.

### 3.3 The Kaggle submission: purity, gates, reproducibility

We build a self-contained submission that embeds the current trainer and runs the **default (pure-ZeroBP) path** with no research flags. Integrity is enforced and checked:

- The submission file contains **no** `.backward` / `autograd.grad` / `enable_grad`; all BP lives in separate research scripts that the submission never imports; every research flag (`attn_train`, `save_ckpt`, `freeze_heads`, `backbone_lr_scale`, `aux_w`) defaults to a no-op.
- A gate suite asserts: resident params ≥ target, **zero autograd**, monotone loss, val-ppl < unigram, **determinism (re-run identical)**, and no late drift. The small-config smoke default reproduces `final_ppl = 6.251` deterministically; the full 4B run fits the T4 at ~8.3 GB peak and ~8546 tok/s.
- We fixed a real reproducibility bug: an earlier submission notebook embedded a *pre-improvement word-level snapshot*; we rebuilt the notebook from the current single-source trainer so the submission reflects the documented BPE+MLP configuration.

**Baseline result.** With BPE subword tokenization + a 2-layer MLP readout head + a cosine-with-routing-freeze ("Phase C") schedule, ZeroBP-4B reaches **WikiText-103 test perplexity ≈ 1391** (early stop, t\* ≈ 54 min) / **≈ 1355** (full ~2.9 h budget), improving on an earlier word-level stage (≈ 1360) and on the unigram baseline by ~51%. The 2-layer MLP head was a large lever; a naïve linear head is much weaker. This gives a clean, deterministic, backprop-free 4B object for the boundary study.

**Engineering erratum (relevant to §4).** We discovered and fixed a `load_state_dict` CPU-aliasing bug (clone-on-load) that had contaminated some *multi-reset small-config* experiments (the reset base aliased the golden checkpoint, which was then polluted in place). All small-config numbers below are the corrected values; **4B and single-shot experiments were unaffected** (CUDA copies + single-shot isolation), so the 79% breakthrough and the 4B matrix rows are unchanged. This episode is why we treat small-config results as *indicative* and require 4B confirmation before locking a capability claim.

---

## 4. The boundary line: a ZeroBP capability matrix (Phase E/F/G)

We now hold the ZeroBP backbone fixed and ask what post-training can install, injecting BP only into **embedding / attention / task-head** parameters (never changing the architecture skeleton). All accuracies use a **fair readout**: a fresh closed-form (autograd-free) classification head trained on the *frozen* adapted representation, so we measure whether the *representation* gained structure rather than whether a BP head overfit.

### 4.1 Three task structures

- **Sentiment** — a bag/compositional label (order-insensitive lexical composition).
- **NLI** — a 3-class relational entailment requiring cross-clause alignment of *two* entity pairs and their relations (entail / contradict / neutral).
- **2-step arithmetic** — a 5-class genuinely sequential fold `((d1 ∘ d2) ∘ d3) mod 5`.

### 4.2 The task × method matrix

| Task | Structure | 4B zero-shot | 4B little-BP | Small-config best |
|---|---|---|---|---|
| Sentiment | bag composition | 60% (59.9) | **79%** (embedding + head, fair readout) | 100% |
| NLI | relational alignment | 33.4% (chance) | 33.4% (does **not** transfer) | 62% → 65.7% (deep BP) |
| 2-step arithmetic | multi-step compute | 24.7% (small) | ~19–21% (chance) | ~chance (gate fails) |

Key locked results:

- **Sentiment breaks (79%).** 4B head-only post-training reaches 59.9% (MLP head) with *zero forgetting* (LM ppl 1355.4 → 1355.4). Adding a little embedding-path BP lifts it to **79%** at a modest cost of +3.0 ppl forgetting (fair closed-form readout). An ablation shows **embedding is the lever** (embedding-only ≈ embedding+top-block). We also traced a **measurement confound**: an early "57%" point used an undertrained SGD readout head; the fair closed-form head corrected it to 79% — a first sign that readout choice can fabricate/hide capability.
- **NLI does not transfer.** At 4B, zero-shot / emb-BP / emb+attn-BP are **all chance (33.4%)**; a large-budget retry (3000 steps, attn lr 0.5) is still chance, so this is a structural limit, not undertraining.
- **Multi-step is uninstallable.** Even at small config the 2-step task fails its gate under any BP component (~chance); it is not carried to 4B.

### 4.3 Three ZeroBP-native structural levers (Phase F)

Can ZeroBP install relational structure *without* crossing into BP? Three routes:

- **F1 — richer pretraining data** (consistent relation pairs + QA): NLI zero-shot 49.1% → **51.3% (+2.1 pp)**. Data alone is insufficient.
- **F2 — structural auxiliary target** fed to the ZeroBP local rule: 51.3% → **51.8% (+0.5 pp)**. ZeroBP cannot carve the relational geometry even when handed the structural objective.
- **Track A — local attention rule** (feedback to `Wq,Wk` from the readout signal): no useful alignment forms.

A little BP, by contrast, *partially* installs it in the small config (fresh-head NLI 51.3% → **58.8%** via embedding, → 59.1% adding attention) — but the extra from attention is only **+0.3 pp**, embedding dominates, and it **does not transfer to 4B**. The wall is **BP-vs-ZeroBP**, and within BP it is **embedding**, not attention, that carries the (limited) signal.

### 4.4 Where is the relational structure? (Phase G, v2.0 probes)

We opened a research-only "v2.0" space to test three structural levers on the backbone, each single-lever and never touching the submission:

- **Non-collapsing readout** (`v2_readout`). On the *same frozen* base, replace last-position collapse with mean-pool / all-positions / concat. Result: NLI does **not** improve — it *drops* (last-h 50.4%, mean-pool 34.9% ≈ chance, all-positions 32.9% ≈ chance, concat 47.5%). **The relational structure is not present anywhere in the frozen representation**; a better readout cannot recover what is not there. This *refuted* our "last-position collapse is the bottleneck" hypothesis.
- **Trainable attention, isolated** (`v2_attn`). Freeze the embedding, train *only* `Wq,Wk` on the NLI CE loss, fair fresh head: **51.3% → 51.3% (+0.0 pp)**. We verified the weights genuinely moved (‖ΔWq‖ ≈ 0.080, ‖ΔWk‖ ≈ 0.079) and the embedding stayed frozen (‖ΔE‖ = 0). So the 59.1% of emb+attn is **entirely embedding**; trainable attention *in isolation* on this backbone is not a lever. This *refuted* the updated "trainable attention is the real lever" hypothesis.
- **Deeper BP** (`v2_deepbp`). Let the MoE blocks themselves be BP-trained. NLI (small): floor 49.1 → emb 57.9 → emb+top-block 64.3 → emb+all-blocks 63.9 → **emb+all+attn 65.7%** (verified both blocks train, ~22–23/48 experts each). Arithmetic: **every** BP-depth arm stays at chance (19–21%). So depth gives a real *small-config* relational signal but **does not transfer to 4B** (NLI-4B remains chance), and multi-step is dead at every depth.

### 4.5 Boundary conclusion (ADR-002)

The evidence supports a sharp, structural boundary for the ZeroBP-4B backbone:

> For **bag-style** tasks (sentiment), ZeroBP + a little embedding BP is effective (79% at 4B). For **relational** (NLI) and **multi-step** (arithmetic) tasks — core modern-LLM competencies — the current ZeroBP-4B architecture and training regime cannot install the structure: the relational geometry is *absent from the representation at every position* (readout cannot recover it), trainable attention in isolation adds nothing, and multi-step computation is uninstallable at any BP depth. The determining factor is **how much of the task lands in the embedding** (BP-installable) versus how much requires *sequential computation through the frozen blocks* or *cross-clause attention alignment* (which a little BP, and even a lot of it at 4B, cannot install).

The natural objection: is this a property of ZeroBP, of the amount of BP, or of the *backbone* (frozen reservoir attention + last-position collapse)? That question is exactly what the control line answers.

---

## 5. The control line: Phase H, a trainable-attention backbone

### 5.1 Design and isolation

Phase H (a "v3.0" research stack, governed by ADR-005 and a dedicated charter) is a clean break: a **standard pre-LN Transformer** with multi-head **bidirectional self-attention**, GELU-MLP blocks, residual connections, and a **non-collapsing mean-pool readout**, trained by **ordinary full backprop (AdamW)**. It is deliberately *not* clever — the point is to use a conventional backbone as an architectural control. It is **strictly isolated**: `phase_h/` has zero dependency on the ZeroBP trainer (pure PyTorch, movable to a separate repo), the submission never imports it, and the submission's default path (`final_ppl 6.251`, zero autograd) is re-verified after every change.

We evaluate on the *same* synthetic distributions used in §4 (apples-to-apples) plus real SNLI, real GSM8K, and real SST-2, with the same fair-readout discipline.

### 5.2 G0 — is the backbone the limit? (synthetic)

A 0.80M-parameter, 4-layer, 4-head Phase-H model trained with full BP on the *same* synthetic tasks:

| Task (synthetic, same dist as §4) | Phase H | ZeroBP-4B / small (§4) |
|---|---|---|
| NLI (3-class) | **100%** @ step 500 | deep-BP 65.7% (small) / chance (4B) |
| 2-step arithmetic (5-class) | **100%** @ step 1000 | chance at *every* BP depth |

A single 0.8M standard trainable-attention model installs *both* the relational and the multi-step structure that the ZeroBP backbone cannot — strong evidence that the §4 boundary is **architectural, not task-intrinsic**.

### 5.3 G1 — real relational data (SNLI)

Scaling to a 12.4M-parameter (6-layer, 8-head, d=256) Phase-H model trained with full BP on **real SNLI** (549k train / 9.8k val, 12.9k steps): **val accuracy = 69.97%** (majority 33.8%). ZeroBP-4B on NLI is **chance (33.4%)**. The new backbone reaches a modern small-LLM tier on a real relational benchmark that the ZeroBP stack structurally cannot — an empirical crossing of the ADR-002 boundary. (Because Phase H is a separate stack, this does *not* revise any locked ZeroBP conclusion; it *contextualizes* it.)

### 5.4 G2 — how deep does multi-step go?

We swept the number of sequential steps k on the arithmetic fold, training a fresh Phase-H model per depth, at two scales:

| n_steps | v1: 6L×8H d256, 4.7M, 6k steps | deep: 12L×8H d384, 21.3M, 15k steps |
|---|---|---|
| k = 2 | **100%** | **100%** |
| k = 3 | 100% (local, 2.67M) | — |
| k = 4 | 22.5% | **21.3%** |
| k = 6 | 19.9% | **19.8%** |
| k = 8 | 20.6% | **21.4%** |

(chance = 20%). Phase H installs **shallow multi-step (k ≤ 3 = 100%)**, decisively past ZeroBP (chance at *any* depth, failing even the k = 2 gate). But k ≥ 4 is a **wall**: **5× the parameters and 2.5× the steps moved it essentially nothing**. This **refuted our own "under-capacity" hypothesis** — the k ≥ 4 failure is not solved by scale under this formulation. We flag the formulation honestly: the task is *direct answer-classification* with a mean-pool readout and **no curriculum and no chain-of-thought** — the two mechanisms most likely to break the k ≥ 4 wall, and untested here.

### 5.5 G2b and SST-2 — honest limits and a readout-collapse lesson

- **G2b — generative GSM8K (stretch).** A 4.87M-parameter causal char-LM (6-layer), trained to *generate* the answer token-by-token and scored by exact-match, reaches **2.0%** on real GSM8K. We labeled this a stretch in advance (a small from-scratch char LM lacks the scale/pretraining for natural-language GSM8K; the *synthetic* generative smoke test reached 66.6%). We report it plainly rather than dress it up: the machinery works, the model does not.
- **SST-2 — a corrected negative.** The ZeroBP-4B stack on real GLUE/SST-2 first returned a *bit-identical* 49.08% across zero-shot / emb-BP / emb+attn-BP. A diagnostic rerun exposed the cause: the **closed-form readout head collapsed** (predicting class-0 for all 872 examples, `[872,0]`), so 49.08% was "always negative," an artifact. A fair **BP linear probe** (which uses both classes) reveals the true picture: zero-shot **51.5%** → emb **52.5%** → emb+attn **53.3%** (majority 50.9%, true `[428,444]`). So ZeroBP-4B on *real* sentiment is **≈ chance** (+0.6…+2.4 pp over majority) — far from the synthetic bag-sentiment's 79%. **Real lexical sentiment is in the same "cannot install" class as NLI/multi-step**; the synthetic 79% was an embedding-separable special case, not a general sentiment capability. This is the second place a readout choice fabricated an apparent number, reinforcing the fair-readout discipline.

### 5.6 Final scorecard

| Dimension | Phase H (new backbone) | ZeroBP-4B (locked) | Conclusion |
|---|---|---|---|
| Synthetic NLI + arithmetic (G0) | 100% / 100% | 65.7% (small) / chance | backbone is the limit |
| Real SNLI (G1) | **69.97%** | 33.4% (chance) | relational structure installable in the new stack |
| Multi-step depth (G2) | k ≤ 3 = 100%, k ≥ 4 = scale-resistant wall | chance at any depth | new stack has its own multi-step ceiling |
| Generative GSM8K (G2b) | EM ~2% | — | small-model stretch failure (honest) |
| Real SST-2 sentiment | — | ≈ chance (BP probe 53%) | real sentiment ≫ synthetic bag task; ZeroBP boundary confirmed on real data |

---

## 6. Discussion: algorithm vs. architecture vs. task

Three readings are now separable because we varied them one at a time:

**The ZeroBP-4B ceiling is architectural.** The relational geometry is absent from the frozen representation at every position (§4.4), so no readout recovers it; trainable attention in isolation adds nothing (§4.4); and multi-step is uninstallable at any BP depth (§4.4). The prime architectural suspects are exactly the two frozen structural choices in §3.2: a **frozen random attention reservoir** (no cross-clause alignment can form) and a **last-position collapse** (sequence structure is discarded before the routed blocks). BP into embeddings can only install what is *linearly composable* at the token level — which is why bag-style sentiment (79%) works and relational alignment / sequential computation do not.

**BP is necessary but not sufficient — it needs a trainable structural substrate.** A little BP installs bag structure but not relational/multi-step structure *on this backbone*; the same tasks are trivially installed by full BP *on a trainable-attention backbone* (§5.2). So gradient is only useful where the architecture provides a trainable place for structure to live (attention alignment, sequential depth). This reframes "ZeroBP can't do NLI" as "**no amount of BP into a frozen-reservoir/last-collapse backbone installs relational structure; a standard attention backbone does so easily.**"

**Task structure sets the difficulty ordering, and even a good backbone has honest limits.** Across both stacks the ordering is consistent — bag ≪ relational ≪ multi-step-depth ≪ real generative math. Phase H crosses the relational boundary and shallow multi-step, but its **k ≥ 4 depth wall is scale-resistant** (§5.4), suggesting that deep sequential reasoning needs *more than backbone + scale* — plausibly curriculum, intermediate supervision, or chain-of-thought (explicitly generating the running value), none of which we tested. The generative-GSM8K and real-SST-2 results bound the claim further: crossing a boundary on a controlled task is not the same as general capability.

**Methodological takeaway.** Two apparent results in this study were artifacts of the *readout*, not the representation (the 57% sentiment SGD-head confound in §4.2, and the 49% SST-2 head collapse in §5.5). Capability claims about representations should always be made through a **fair, task-agnostic probe**, and ideally *two* (closed-form and BP-linear), which here disagreed exactly when one of them had degenerated.

---

## 7. Conclusion and future work

We presented a three-line boundary study. **ZeroBP-4B** is a reproducible, deterministic, backprop-free 4B language model that is genuinely useful under a single-T4 budget for LM and bag-style tasks (WikiText-103 ppl ≈ 1355–1391; sentiment 79%), and serves as a controllable object for measuring capability boundaries. Those boundaries — relational and multi-step failure — are shown to be **architectural** by a strictly isolated **Phase H** control: a standard trainable-attention Transformer with full BP solves the same synthetic tasks to 100%, reaches **69.97%** on real SNLI (vs. chance), and installs shallow multi-step reasoning — while exposing its *own* honest limits: a scale-resistant multi-step depth wall at k ≥ 4, ~2% on real GSM8K, and near-chance real SST-2 that also afflicts ZeroBP.

**Roles.** ZeroBP-4B: a resource-constrained pretraining + boundary-measurement tool, effective on LM/simple composition, hard-limited on relational/multi-step. Phase H: a modern-backbone baseline demonstrating that NLI and shallow multi-step are *learnable* (not task-intrinsic barriers) and surfacing a new depth-of-reasoning research question.

**Future work.** (i) Break the k ≥ 4 wall with curriculum / chain-of-thought / intermediate supervision rather than raw scale; (ii) test whether *advanced* ZeroBP local rules on a *trainable-attention* substrate can recover part of the Phase-H capability without full BP (combining the two lines); (iii) scale Phase H (data + parameters, MNLI/GSM8K) toward modern small-LLM behavior on the relational/reasoning axes; (iv) push ZeroBP-4B's LM quality with better local rules while preserving zero-autograd purity.

---

## Appendix A — Reproducibility and artifacts

- **Submission (pure ZeroBP):** `kaggle_zerograd_moe.py` (single source) + generated submission notebook; default path asserts zero autograd and reproduces `final_ppl 6.251` deterministically. Purity checklist in `SUBMISSION.md`.
- **Boundary line (research BP, isolated):** `phase_e*.py`, `phasee_nli_4b.py`, `f1_data.py`, `f2_aux.py`, `h1_attn.py`, `v2_readout.py`, `v2_attn.py`, `v2_deepbp.py`, `task_nli.py`, `task_arith.py`.
- **Control line (Phase H, isolated):** `phase_h/ph_base.py` (Transformer + causal LM), `ph_nli.py`/`ph_arith.py` (G0), `ph_nli_gpu.py` (G1), `ph_gsm_gpu.py` (G2 depth sweep), `ph_gsm_gen.py` (G2b), `track1_sst2_4b.py` (SST-2 with prediction-distribution + BP-linear-probe diagnostics).
- **Governance:** ADR-001…005, `ARCHITECTURE.md`, `Phase-*/charter.md`, `EXPERIMENT_LEDGER.md`, `MASTER_ARCHIVE.md`. Each locked number carries a commit hash in the ledger.
- **Compute:** single Kaggle T4; concurrency capped at 2 GPU sessions; per-run summaries pulled to `runs/*.json`.

## Appendix B — Honest corrections log

| Claim as first stated | Correction | Evidence |
|---|---|---|
| Sentiment 4B ≈ 57% (early point) | 79% (undertrained SGD head → fair closed head) | §4.2 |
| "Last-position collapse is the bottleneck" | Refuted: structure absent from frozen rep at all positions | §4.4 (`v2_readout`) |
| "Trainable attention is the real lever" | Refuted: attention-only = +0.0 pp; embedding carries it | §4.4 (`v2_attn`) |
| SST-2 (ZeroBP-4B) = 49% flat | Readout-collapse artifact; true ≈ 53% via fair BP probe | §5.5 |
| k ≥ 4 multi-step is under-capacity | Refuted: 5× params + 2.5× steps unchanged → real wall | §5.4 |
