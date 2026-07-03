<!-- **English** | [中文](PAPER_workshop.zh.md) -->
**English** | [中文 →](PAPER_workshop.zh.md)

# Is It the Algorithm or the Architecture? A Boundary Study of a Backprop-Free 4B LM with a Trainable-Attention Control

**Workshop cut (~8 pp).** Condensed from the full technical report (`PAPER_DRAFT.md`); all numbers trace to `EXPERIMENT_LEDGER.md`.

---

## Abstract

This report asks how far a *backprop-free* language model gets under a single-T4 budget, and — the real question — *why it stops*. The study builds **ZeroBP-4B**, a 4.16B content-routed MoE LM trained with **no global backpropagation** (no `autograd`, no `.backward()`; every update is a local rule), locked as a deterministic Kaggle submission (WikiText-103 test ppl ≈ 1391 early-stop / ≈ 1355 full-budget). On this fixed backbone the study maps a capability boundary across three task structures (bag sentiment, relational NLI, multi-step arithmetic) under increasing injected backprop. The boundary is sharp and *structural*: a little embedding-path BP lifts sentiment 60%→79%, but neither shallow nor deep BP installs relational or multi-step structure at 4B (both stay at chance), and three ZeroBP-native routes add ≤2 pp; a non-collapsing-readout probe shows the relational structure is absent from the representation at every position. To separate task-difficulty from architecture, the study adds a strictly isolated **architectural control** (the final stage) — a standard trainable-attention Transformer with full backprop. The same synthetic tasks that cap ZeroBP at 65.7% / chance are solved to 100% by a 0.8M control model; on **real SNLI** the control backbone reaches **69.97%** where ZeroBP-4B is at chance. The limits are reported just as plainly: the control backbone installs shallow multi-step (k≤3=100%) but hits a **scale-resistant wall at k≥4** (unchanged 4.7M/6k→21.3M/15k), scores 2% on real GSM8K, and — like ZeroBP — is near-chance on real SST-2 once a readout-collapse artifact is corrected. **The ZeroBP relational/multi-step ceiling is architectural, not fundamental; a trainable-attention backbone crosses it, while exposing new honest limits of its own.**

## 1. Introduction

Two questions motivate this work: (i) how much language capability survives without global BP under a strict single-GPU budget, and (ii) when a model fails a task, is the failure *algorithmic* (too little gradient), *architectural* (the backbone cannot represent the structure), or *task-intrinsic*? These are normally impossible to separate because many things change at once. A layered design of **three lines**, run as a strict progression (each stage motivated by the previous result), separates them: a **submission line** (build a clean pure-ZeroBP 4B baseline, §2), a **boundary line** (with it fixed, measure where post-training succeeds/fails, try in-backbone fixes, then diagnose the bottleneck, §3), and a **control line** (test whether the diagnosed limit is architectural by swapping only the backbone, §4). The contribution is not a new SOTA number but a **controlled boundary study**: a reproducible backprop-free 4B baseline, a systematic capability matrix with strict submission/research isolation, and an architectural control that turns "ZeroBP can't do NLI" into "*this backbone* can't, *this one* can, and *here* is where even that one stops." A recurring lesson is that capabilities must be read through a **fair, task-agnostic probe**: twice in this study an unfair readout fabricated or destroyed an apparent number.

## 2. The submission line: a pure-ZeroBP 4B baseline

**Setting.** A single T4 (16 GB), 3 h wall-clock, self-supplied data — where standard BP of a 4B model OOMs (~31 GB). ZeroBP removes the backward pass entirely.

**Model and update (code-level).** A decoder-style MoE LM: `token → E[x]+pos → frozen random causal attention (Wq,Wk untrained) → h=(emb+att·emb)[:,-1]` (last-position collapse) `→ 4× routed MoE block (EMA-prototype content routing + capacity, 950 experts, top-2) → per-block 2-layer-MLP readout`. With `d=1024, |V|=32000, seq_len=64, 4 layers, 950 experts/layer, k_route=2, k_update=4`: **4.160 B fp16** resident. Three properties recur below: (a) **no global gradient** — training runs under `set_grad_enabled(False)`, asserted; experts update by local rules driven by the deeply-supervised readout signal; (b) a **deterministic sparse budget** updates 0.42% of experts/step (a learned importance controller did *not* beat round-robin and was rejected); (c) a **frozen attention reservoir + last-position collapse** — the two structural choices that §3 implicates.

**Submission integrity.** The default path asserts a gate suite (resident params ≥ target, **zero autograd**, monotone loss, val-ppl<unigram, **determinism**, no late drift). The submission source has no real `.backward`/`autograd.grad`/`enable_grad` and imports no research script; all research flags default to no-ops. The smoke config reproduces `final_ppl=6.251` deterministically; the 4B run fits the T4 at ~8.3 GB, ~8546 tok/s.

**Baseline.** With BPE + a 2-layer MLP head + a routing-freeze cosine schedule: **WikiText-103 test ppl ≈ 1391** (early stop) / **≈ 1355** (full budget), ~51% better than unigram. This is a clean, deterministic, backprop-free 4B object for the boundary study. *(Erratum: a `load_state_dict` CPU-aliasing bug had contaminated some multi-reset small-config runs; all small-config numbers below are corrected, and 4B/single-shot results were unaffected — hence 4B confirmation is required before locking a claim.)*

## 3. The boundary line: a ZeroBP capability matrix

Holding the backbone fixed, BP is injected only into embedding / attention / task-head (never the skeleton) and measure with a **fair readout**: a fresh closed-form head on the *frozen* adapted representation.

**Three task structures.** Sentiment (bag composition), NLI (3-class relational alignment of two entity pairs), 2-step arithmetic (`((d1∘d2)∘d3) mod 5`).

| Task | Structure | 4B zero-shot | 4B little-BP | Small-config best |
|---|---|---|---|---|
| Sentiment | bag | 60% | **79%** (embedding+head) | 100% |
| NLI | relational | 33.4% (chance) | 33.4% (no transfer) | 62% → 65.7% (deep BP) |
| 2-step arithmetic | multi-step | 24.7% | ~19–21% (chance) | ~chance |

**Findings.** *Sentiment breaks* — head-only reaches 59.9% with zero forgetting; a little embedding BP → **79%** (+3.0 ppl), and embedding is the lever. *NLI does not transfer* — all 4B variants are chance, even a 3000-step / high-attn-lr retry. *Multi-step is uninstallable* at any BP component. Three **ZeroBP-native** levers barely move NLI: richer data +2.1 pp, structural auxiliary target +0.5 pp, local attention rule ≈0.

**Where is the relational structure?** Three single-lever v2.0 probes on the frozen/adapted backbone: (1) **non-collapsing readout** — mean-pool/all-positions *drop to chance* (34.9% / 32.9%), so the structure is *absent from the representation at every position* (refuting the "collapse is the bottleneck" hypothesis); (2) **trainable attention in isolation** — freezing the embedding and training only `Wq,Wk` on NLI gives **+0.0 pp** (51.3→51.3), with verified weight movement (‖ΔWq‖≈0.08) and frozen embedding (‖ΔE‖=0), so emb+attn's 59.1% is *entirely embedding* (refuting "attention is the lever"); (3) **deep BP** — letting the blocks train lifts small-config NLI to 65.7% but does **not** transfer to 4B, and arithmetic stays chance at every depth.

**Boundary conclusion.** For bag tasks, ZeroBP + a little embedding BP is effective (79%). For **relational** and **multi-step** tasks — core modern-LLM competencies — the ZeroBP-4B architecture cannot install the structure: it is absent from the representation, trainable attention adds nothing in isolation, and multi-step is dead at any BP depth. The determinant is how much of a task lands in the embedding (BP-installable) vs. requires sequential computation through frozen blocks or cross-clause attention alignment. Is this a property of the algorithm, the BP budget, or the *backbone*? — the control line answers.

## 4. The control line: an isolated trainable-attention backbone (stage 5)

**Design & isolation.** A deliberately conventional pre-LN Transformer: multi-head **bidirectional** self-attention, GELU-MLP blocks, residuals, **mean-pool readout**, trained by **full backprop (AdamW)**. Strictly isolated — pure PyTorch, zero dependency on the ZeroBP trainer; the submission never imports it and its `6.251`/zero-autograd default is re-verified after every change. Evaluated on the *same* synthetic distributions as §3, plus real SNLI/GSM8K/SST-2.

**G0 (synthetic): is the backbone the limit?** A 0.80M (4-layer) control model, full BP: synthetic **NLI 100%** @step 500, **2-step arithmetic 100%** @step 1000 — vs ZeroBP's 65.7% (small) / chance (4B) and chance-at-any-depth. One small standard model installs *both* structures ZeroBP cannot: the boundary is **architectural, not task-intrinsic**.

**G1 (real SNLI).** A 12.4M (6-layer) control model on real SNLI (549k train): **val 69.97%** (majority 33.8%); ZeroBP-4B is chance (33.4%). The new backbone reaches a modern small-LLM tier on a real relational benchmark ZeroBP structurally cannot.

**G2 (multi-step depth).** Per-depth training on the arithmetic fold, two scales:

| n_steps | 4.7M, 6k steps | 21.3M, 15k steps |
|---|---|---|
| k=2 | 100% | 100% |
| k=3 | 100% (2.67M, local) | — |
| k=4 | 22.5% | 21.3% |
| k=6 | 19.9% | 19.8% |
| k=8 | 20.6% | 21.4% |

the control backbone installs **shallow multi-step (k≤3=100%)**, past ZeroBP (chance at any depth, failing even k=2). But k≥4 is a **wall**: **5× params + 2.5× steps moved it nothing**, refuting the "under-capacity" hypothesis. Formulation caveat: direct answer-classification, mean-pool, **no curriculum / chain-of-thought** (the untested routes most likely to break it).

**G2b / SST-2 (honest limits + a readout lesson).** A 4.87M char-LM *generating* GSM8K answers scores **2% exact-match** (flagged a stretch in advance; synthetic generative smoke was 66.6%). On real SST-2, ZeroBP-4B first returned a bit-identical 49.08%; a diagnostic showed the closed-form head had **collapsed** (all-negative, `[872,0]`). A fair **BP linear probe** gives the truth: **51.5%→52.5%→53.3%** (majority 50.9%) — i.e. **≈ chance**. Real lexical sentiment is in the same "cannot install" class as NLI/multi-step; the synthetic 79% was an embedding-separable special case.

### Final scorecard

| Dimension | Control backbone (new arch) | ZeroBP-4B (locked) | Conclusion |
|---|---|---|---|
| Synthetic NLI + arithmetic | 100% / 100% | 65.7% (small) / chance | backbone is the limit |
| Real SNLI | **69.97%** | 33.4% (chance) | relational structure installable |
| Multi-step depth | k≤3=100%, k≥4 wall | chance at any depth | new stack has its own ceiling |
| Generative GSM8K | EM ~2% | — | small-model stretch failure |
| Real SST-2 | ≈ chance (probe 53%) | ≈ chance | real sentiment ≫ synthetic bag |

## 5. Discussion

Because each factor was varied one at a time, three readings separate. **(i) The ZeroBP ceiling is architectural** — the relational structure is absent from the frozen representation, trainable attention adds nothing in isolation, and multi-step is dead at any BP depth; the prime suspects are the frozen attention reservoir (no alignment can form) and last-position collapse (sequence structure discarded before the blocks). Embedding BP installs only what is linearly composable at the token level — hence bag sentiment works and relational/sequential structure does not. **(ii) BP is necessary but not sufficient** — the same tasks are trivially installed by full BP on a trainable-attention backbone, so gradient helps only where the architecture gives structure a trainable place to live. **(iii) Task structure sets a consistent difficulty ordering** (bag ≪ relational ≪ multi-step-depth ≪ real generative math), and even a good backbone has honest limits: the control backbone's k≥4 wall is scale-resistant, suggesting deep sequential reasoning needs more than backbone+scale (curriculum, intermediate supervision, chain-of-thought). **Methodologically**, two apparent numbers here were readout artifacts, not representation facts (an undertrained SGD sentiment head; a collapsed SST-2 head) — capability claims about representations should use a fair, task-agnostic probe, ideally two.

## 6. Conclusion and future work

ZeroBP-4B is a reproducible, deterministic, backprop-free 4B LM that is genuinely useful under a single-T4 budget for LM and bag-style tasks (ppl ≈1355–1391; sentiment 79%) and serves as a controllable object for measuring capability boundaries. Those boundaries are shown *architectural* by an isolated trainable-attention control that solves the same synthetic tasks (100%), reaches 69.97% on real SNLI, and installs shallow multi-step — while exposing its own limits (k≥4 wall, 2% GSM8K, near-chance real SST-2). **Future work:** break the k≥4 wall with curriculum / chain-of-thought rather than raw scale; test advanced ZeroBP local rules on a *trainable-attention* substrate (combining the two lines); scale the control backbone toward modern small-LLM behavior on relational/reasoning axes; and improve ZeroBP LM quality while preserving zero-autograd purity.

## Limitations

Small-config results are indicative and confirmed at 4B only where stated (see the erratum). the control backbone's real-data evaluation is at small scale (≤21M) and single-benchmark per axis (SNLI, GSM8K, SST-2); the k≥4 result is specific to answer-classification without curriculum/CoT. Two initial claims were corrected by later runs and are retained transparently: the SST-2 flat-49% was a readout-collapse artifact (true ≈53%), and the multi-step k≥4 failure is a scale-resistant wall, not under-capacity. Full artifact map and per-run summaries accompany the report.
