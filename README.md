<!-- **English** | [中文](README.zh.md) -->
**English** | [中文版 →](README.zh.md)

# ZeroBP-4B and Phase H

**A resource-constrained research project** that (1) builds a *pure backprop-free* 4-billion-parameter language model on a single GPU, (2) systematically maps where its reasoning breaks, and (3) uses an isolated *trainable-attention control backbone* to separate **algorithmic** limits from **architectural** ones.

**Read next:** [Full report](paper/PAPER_DRAFT.md) · [Workshop paper (8pp)](paper/PAPER_workshop.md) · [Slides](paper/slides_outline.md) · [Kaggle submission](docs/KAGGLE_README.md) · [Master archive](docs/MASTER_ARCHIVE.md) — *(每份均有中文 `*.zh.md`)*

![single Tesla T4](https://img.shields.io/badge/hardware-single%20Tesla%20T4-blue) ![zero autograd](https://img.shields.io/badge/training-zero%20backprop-8A2BE2) ![params](https://img.shields.io/badge/params-4.16B-green) ![reproducible](https://img.shields.io/badge/results-deterministic%20%26%20logged-brightgreen)

---

## TL;DR

- Built and locked a **pure ZeroBP 4B baseline** — 4.16B parameters, **no backpropagation anywhere** (no `autograd`/`.backward()`), single T4, with a deterministic, reproducible submission path.
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

---

## Two entry points

**A · GitHub landing page (you are here)** — 30-second orientation: what this is, the main results, where to look.

**B · The paper / technical report** — for careful review: problem, method, experiments, limitations, conclusions → **[paper/PAPER_DRAFT.md](paper/PAPER_DRAFT.md)** (full) or **[paper/PAPER_workshop.md](paper/PAPER_workshop.md)** (8 pp).

## Two lines in this repository

The repo cleanly separates a **product-grade submission** from the **research investigation** — they never share a training path.

**1 · Submission line** — a clean, rule-compliant, reproducible pure-ZeroBP baseline.
→ [`kaggle_zerograd_moe.py`](kaggle_zerograd_moe.py) (model + default path) · [`kaggle_run/`](kaggle_run/) (submission kernel) · [`docs/KAGGLE_README.md`](docs/KAGGLE_README.md).

**2 · Research line** — the full experimental arc + the control-backbone investigation.
→ [`docs/MASTER_ARCHIVE.md`](docs/MASTER_ARCHIVE.md) → [`paper/PAPER_workshop.md`](paper/PAPER_workshop.md) → [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md) → [`phase_h/`](phase_h/).

## Repository layout

```text
.
├── README.md · README.zh.md            # this landing page (EN / 中文)
├── LICENSE · CITATION.cff · requirements.txt
│
├── kaggle_zerograd_moe.py              # the backprop-free 4B model + pure-ZeroBP submission path
├── kaggle_run/                         # official submission kernel (notebook + metadata)
├── selfcheck.py · build_kaggle_kernels.py · orchestrate_kaggle.py
│
├── phase_e*.py · c1_4b.py · adapt_sentiment.py       # boundary line: post-training limits (Phase E)
├── f1_data.py · f2_aux.py · h1_attn.py               # boundary line: ZeroBP-native fixes (Phase F)
├── v2_readout.py · v2_attn.py · v2_deepbp.py         # boundary line: structural probes (Phase G)
├── task_nli.py · task_arith.py                       # shared synthetic tasks
├── track1_sst2_4b.py · track1_radar.py · make_figures.py · build_track1_kernels.py
│
├── phase_h/                            # CONTROL: isolated trainable-attention backbone (research-only)
│
├── paper/                             # the report (EN + 中文)
│   ├── PAPER_DRAFT.md · PAPER_workshop.md · slides_outline.md  (+ .zh.md)
├── docs/                             # authoritative docs
│   ├── MASTER_ARCHIVE.md · EXPERIMENT_LEDGER.md · ARCHITECTURE.md
│   ├── ENGINEERING.md · SUBMISSION.md · KAGGLE_README.md · archive_zh.md
│   └── adr/                           # architecture decision records (ADR-001..005)
├── figures/                          # figures + make_figures.py output
└── results/                          # canonical real-run result JSONs (+ index)
```

**Read at the right granularity:** landing (this page) → story ([paper/](paper/)) → context ([docs/MASTER_ARCHIVE.md](docs/MASTER_ARCHIVE.md)) → evidence ([docs/EXPERIMENT_LEDGER.md](docs/EXPERIMENT_LEDGER.md), scripts, [results/](results/)).

## Quickstart

```bash
pip install -r requirements.txt

# Reproduce locally (no GPU needed)
python3 kaggle_zerograd_moe.py    # backprop-free model, small config -> final_ppl 6.251, zero-autograd PASS, deterministic
python3 phase_h/ph_nli.py         # control backbone solving synthetic relational reasoning -> 100%
python3 make_figures.py           # regenerate the figures in figures/

# Kaggle baseline (needs a Kaggle account + T4)
python3 build_kaggle_kernels.py && kaggle kernels push -p kaggle_run   # see docs/KAGGLE_README.md
```

Expected submission gates: `final_ppl = 6.251` · zero-autograd **PASS** · deterministic **PASS**. Every headline number traces to [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md); raw results are in [`results/`](results/).

## Papers and reports

- [`paper/PAPER_workshop.md`](paper/PAPER_workshop.md) — concise paper (~8 pp) for workshop-style submission.
- [`paper/PAPER_DRAFT.md`](paper/PAPER_DRAFT.md) — full technical report with extended context and an honest **corrections log**.
- [`paper/slides_outline.md`](paper/slides_outline.md) — 11-slide talk outline.

The report is organized around the **three lines** (submission → boundary → control), not chronologically.

## Citation & license

MIT-licensed ([LICENSE](LICENSE)); machine-readable [CITATION.cff](CITATION.cff). Independent research project.

```bibtex
@misc{zerobp4b_phaseh_2026,
  title  = {ZeroBP-4B and Phase H: A Backprop-Free 4B Language Model, its Capability Boundary,
            and a Trainable-Attention Control},
  author = {Yanjin Li},
  year   = {2026},
  note   = {https://github.com/Yanjin-ai/zero-gradient}
}
```

*Full project history (including exploratory phases and process notes) is preserved in the backup mirror: [github.com/Yanjin-ai/zerogradient](https://github.com/Yanjin-ai/zerogradient).*
