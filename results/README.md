# results/ — canonical real-run result data

Version-controlled copies of the **real GPU run summaries** (Kaggle T4), decoupled from the transient
`*/out/` pull directories (which are gitignored). Numbers here are the source for the ledger, the archive,
and the papers. See `EXPERIMENT_LEDGER.md` for full context and commit provenance.

| File | Run | Headline |
|---|---|---|
| `phase_h_g1_snli.json` | Phase H G1 — real SNLI, 12.4M (6L×8H d256), full BP | **val_acc 69.97%** (majority 33.8%) vs ZeroBP-4B chance 33.4% |
| `phase_h_g2_multistep_deep.json` | Phase H G2 — multi-step depth sweep, 21.3M (12L×8H d384), 15k steps/depth | k=2 **100%**, k=4 21.3%, k=6 19.8%, k=8 21.4% (chance 20%) — k≥4 scale-resistant wall |
| `phase_h_g2b_gsm8k.json` | Phase H G2b — generative GSM8K, 4.87M char-LM causal | exact-match **2.0%** (honest small-model stretch) |
| `track1_sst2_diagnostic.json` | Track 1 — ZeroBP-4B on real SST-2 (diagnostic) | closed head **collapsed** ([872,0]); fair BP linear probe **51.5→52.5→53.3%** ≈ chance |

Notes:
- **Synthetic G0** (Phase H NLI/arithmetic = 100%) is reproduced locally on CPU via `phase_h/ph_nli.py` /
  `ph_arith.py`; the local smoke summaries land in `runs/ph_*_run_summary.json`.
- **ZeroBP-4B research results** (Phase E sentiment 79%, C.1 head-only, NLI-4B chance) are recorded in
  `EXPERIMENT_LEDGER.md` with commit hashes; their machine ledger is `runs/experiments.jsonl` (gitignored,
  local only).
