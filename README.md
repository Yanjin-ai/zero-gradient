<!-- **English** | [中文](README.zh.md) -->
**English** | [中文版 →](README.zh.md)

# ZeroBP-4B and Phase H

**A resource-constrained research project** that (1) builds a *pure backprop-free* 4-billion-parameter language model on a single GPU, (2) systematically maps where its reasoning breaks, and (3) uses an isolated *trainable-attention control backbone* to separate **algorithmic** limits from **architectural** ones.

**Artifacts:** [Kaggle README](KAGGLE_README.md) · [Workshop paper](PAPER_workshop.md) · [Full report](PAPER_DRAFT.md) · [Slides](slides_outline.md) · [Master archive](MASTER_ARCHIVE.md) — *(每份均有中文版 `*.zh.md`)*

`single Tesla T4` · `zero autograd` · `4.16B params` · `deterministic & reproducible`

---

## TL;DR

- Built and locked a **pure ZeroBP 4B Kaggle baseline** — 4.16B parameters, **no backpropagation anywhere** (no `autograd`/`.backward()`), single T4, with a deterministic, reproducible submission path.
- Measured a clear **capability boundary**: ZeroBP is effective on bag-style **sentiment (79%)** but stays at **chance on real NLI and multi-step arithmetic**.
- Tested five in-backbone fixes (richer data, structural objectives, non-collapsing readout, attention-only BP, deeper BP) — **none removes the 4B boundary**.
- Introduced **Phase H**, a fully isolated *trainable-attention control backbone*, which reaches **69.97% on real SNLI** (vs chance) and solves shallow multi-step arithmetic (k ≤ 3).
- **Main conclusion:** the ZeroBP failures on relational and multi-step tasks are primarily **architectural, not task difficulty** — and even the control backbone has an honest limit (a scale-resistant multi-step wall at k ≥ 4).

## Headline results

| Dimension | ZeroBP-4B (backprop-free) | Phase H (trainable-attn control) | Interpretation |
|---|---:|---:|---|
| Kaggle LM baseline | **ppl 1391 / 1355** | — | strong resource-constrained baseline |
| Sentiment (bag) | **79%** (a little BP) | — | simple compositional signal is learnable |
| Real SNLI (relations) | 33.4% (chance) | **69.97%** | relational structure is **architectural** |
| Multi-step arithmetic | chance at any depth | **k ≤ 3 solved, k ≥ 4 wall** | shallow reasoning installs in the control backbone |
| Generative GSM8K | — | 2.0% EM | honest small-model limit |

![Same tasks, two designs — the limit is the architecture, not the task](figures/capability_comparison.png)

> New here? A plain-language walkthrough with figures is in **[README.zh.md](README.zh.md)** / see the [full report](PAPER_DRAFT.md) for methods and evidence.

---

## Two lines in this repository

The repo cleanly separates a **product-grade submission** from the **research investigation** — they never share code.

**1 · Kaggle submission line** — a clean, rule-compliant, reproducible pure-ZeroBP baseline.
→ start at [`kaggle_zerograd_moe.py`](kaggle_zerograd_moe.py) + [`KAGGLE_README.md`](KAGGLE_README.md).

**2 · Research line** — the full experimental arc + the control-backbone investigation.
→ start at [`MASTER_ARCHIVE.md`](MASTER_ARCHIVE.md) → [`PAPER_workshop.md`](PAPER_workshop.md) → [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) → [`phase_h/`](phase_h/).

## Project map

| Path | Layer | What it is |
|---|---|---|
| `kaggle_zerograd_moe.py` · `KAGGLE_README.md` | submission | the pure-ZeroBP 4B model + reviewer doc |
| `PAPER_workshop.md` · `PAPER_DRAFT.md` · `slides_outline.md` | **story** | the paper (short + full) and talk outline |
| `MASTER_ARCHIVE.md` · `项目总档案.md` | **archive** | authoritative index (bilingual) + deep Chinese narrative |
| `EXPERIMENT_LEDGER.md` · `docs/adr/` · `ARCHITECTURE.md` | evidence | per-experiment configs/commits/metrics + decisions |
| `phase_h/` | research code | the isolated trainable-attention control backbone |
| `phase_e*.py` · `v2_*.py` · `task_*.py` · `track1_sst2_4b.py` | research code | the boundary-line experiments (BP, default-off, isolated) |
| `results/` · `figures/` · `runs/` | data | canonical result JSONs, figures, run summaries |

*Read at the right granularity: **landing** (this page) → **story** (paper) → **archive** (context) → **evidence** (ledger/scripts).*

## Quickstart

```bash
# Research artifacts (start here to understand the project)
open MASTER_ARCHIVE.md            # the narrative + locked conclusions
open PAPER_workshop.md            # the 8-page story

# Reproduce locally (no GPU needed)
python3 kaggle_zerograd_moe.py    # backprop-free model, small config -> final_ppl 6.251, zero-autograd PASS, deterministic
python3 phase_h/ph_nli.py         # control backbone solving synthetic relational reasoning -> 100%
python3 make_figures.py           # regenerate the figures in figures/

# Kaggle baseline (needs Kaggle account + T4)
python3 build_kaggle_kernels.py && kaggle kernels push -p kaggle_run   # see KAGGLE_README.md
```

Expected submission gates: `final_ppl = 6.251` · zero-autograd **PASS** · deterministic **PASS**.

## Repository structure

```text
.
├── README.md  /  README.zh.md          # this landing page (EN / 中文)
├── kaggle_zerograd_moe.py              # ZeroBP-4B model + pure-ZeroBP submission path
├── KAGGLE_README.md  /  .zh.md         # Kaggle-facing baseline doc
├── PAPER_DRAFT.md  /  .zh.md           # full technical report
├── PAPER_workshop.md  /  .zh.md        # concise 8-page paper
├── slides_outline.md  /  .zh.md        # presentation outline
├── MASTER_ARCHIVE.md                   # authoritative project index (bilingual)
├── 项目总档案.md                        # detailed Chinese archive
├── EXPERIMENT_LEDGER.md                # experiment log: commits, configs, metrics
├── ARCHITECTURE.md · ENGINEERING.md · SUBMISSION.md
├── phase_h/                            # isolated trainable-attention control backbone
├── phase_e*.py · v2_*.py · task_*.py · track1_sst2_4b.py   # boundary-line research
├── results/                            # canonical real-run result JSONs
├── figures/  · make_figures.py         # figures + generator
└── docs/adr/                           # architecture decision records (ADR-001..005)
```

## Papers and reports

- [`PAPER_workshop.md`](PAPER_workshop.md) — concise paper version (~8 pp) for workshop-style submission.
- [`PAPER_DRAFT.md`](PAPER_DRAFT.md) — full technical report with extended context and an honest **corrections log**.
- [`slides_outline.md`](slides_outline.md) — 11-slide talk outline.

The paper is organized around the **three lines** (submission → boundary → control), not chronologically. Every number traces to [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md); raw results are in [`results/`](results/).

## Citation & about

Independent research project (single-GPU, backprop-free LM + architectural control study). If you reference this work:

```bibtex
@misc{zerobp4b_phaseh_2026,
  title  = {ZeroBP-4B and Phase H: A Backprop-Free 4B Language Model, its Capability Boundary,
            and a Trainable-Attention Control},
  author = {Yanjin Li},
  year   = {2026},
  note   = {https://github.com/Yanjin-ai/zero-gradient}
}
```

*Backup mirror of the full project history: [github.com/Yanjin-ai/zerogradient](https://github.com/Yanjin-ai/zerogradient).*
