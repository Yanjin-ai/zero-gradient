# Slides outline — ZeroBP-4B boundary study + Phase H control (11 slides)

Talk length ~12–15 min. Key visual assets: capability radar (`runs/track1_radar.png`), the task×method
matrix table (§3), the final scorecard table (§4). Numbers trace to `EXPERIMENT_LEDGER.md`.

---

**1 — Title / one-line thesis**
- "Is it the algorithm or the architecture?" — a backprop-free 4B LM, its capability boundary, and a
  trainable-attention control that explains it.
- Names, affiliation, repo/report pointer.

**2 — Motivation (the two questions)**
- How much LM capability survives with **no global backprop**, on a **single T4 / 3 h**?
- When it fails a task: **algorithm** (too little gradient), **architecture** (backbone can't represent it),
  or **task-intrinsic**? Usually inseparable — we separate it.
- Method preview: **three lines** (Submission → Boundary → Control).

**3 — The submission line: ZeroBP-4B**
- 4.16B content-routed MoE; **zero autograd**, every update a local rule; deterministic sparse budget (0.42%
  experts/step); frozen attention reservoir + last-position collapse.
- Result: WikiText-103 ppl **≈1391 / ≈1355**, ~51% > unigram; fits T4 (~8.3 GB, ~8546 tok/s); deterministic,
  gated, purity-checked. → a clean object to probe.

**4 — The boundary line: task × method matrix**  *(table)*
- Sentiment (bag) / NLI (relational) / 2-step arithmetic (multi-step), with fair readout.
- Sentiment **60→79%** (embedding BP is the lever); NLI **chance→chance** at 4B; arithmetic **chance** at any BP.
- Punchline: bag structure installs; relational & multi-step do not.

**5 — "Where is the relational structure?" (three v2.0 probes)**
- Non-collapsing readout → **drops to chance** (structure absent at every position).
- Trainable attention *alone* → **+0.0 pp** (verified weights moved; embedding frozen).
- Deep BP → 65.7% small-config only, **no 4B transfer**; arithmetic dead at every depth.
- → Boundary is **structural**; two hypotheses of ours were refuted here (honesty flag).

**6 — The question this forces**
- Is "ZeroBP can't do NLI" about ZeroBP, the BP budget, or the **backbone** (frozen reservoir + collapse)?
- Enter the **control**: swap only the backbone, keep everything else standard.

**7 — Phase H: the control (design + G0)**
- Standard trainable-attention Transformer, mean-pool, full BP; strictly isolated from the submission.
- **G0 (same synthetic tasks):** NLI **100%**, 2-step arithmetic **100%** at **0.8M** params.
- → the boundary is **architectural, not task-intrinsic**.

**8 — Phase H on real data (G1 + G2)**  *(scorecard table begins here)*
- **Real SNLI: 69.97%** (12.4M) vs ZeroBP chance 33.4% — relational structure installs.
- **Multi-step depth:** k≤3 = 100%; **k≥4 = wall**, unchanged from 4.7M/6k → 21.3M/15k → **not** under-capacity.

**9 — Honest limits (G2b + SST-2) + the readout lesson**
- Real **GSM8K: 2%** EM (small char-LM stretch, flagged in advance).
- Real **SST-2 ≈ chance** — *after* fixing a closed-head **collapse** (flat 49% was an artifact; BP probe → 53%).
- Lesson: read capability through a **fair, task-agnostic probe** (twice bitten here).

**10 — Final scorecard + capability radar**  *(scorecard table + `runs/track1_radar.png`)*
- One slide: the 5-row Phase H vs ZeroBP table + the radar figure.
- Story in one line: **new backbone crosses the ZeroBP relational/shallow-multi-step boundary; both have honest limits.**

**11 — Conclusion & future work**
- ZeroBP-4B: a resource-constrained backprop-free baseline + boundary-measurement tool (good at LM/bag,
  hard-limited on relational/multi-step).
- Phase H: proves NLI/shallow-multi-step are *learnable*, surfaces a multi-step *depth* research question.
- Next: break k≥4 with curriculum / chain-of-thought (not scale); advanced ZeroBP rules on a trainable
  substrate; scale Phase H on relational/reasoning axes.

*(Optional backup slides: ZeroBP local-rule detail; erratum + honest-corrections log; SST-2 diagnostic table.)*
