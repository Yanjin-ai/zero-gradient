# phase_h/ — Phase H / v3.0 new-backbone stack (RESEARCH-ONLY, isolated)

A clean-break standard multi-layer trainable-attention Transformer (full backprop), separate from the
ZeroBP 4B submission. **Zero dependency on `kaggle_zerograd_moe`** — this directory can move to its own repo
unchanged. Never imported by the submission; submission default path stays `6.251` / zero autograd.

Governance: [ADR-005](../docs/adr/ADR-005-phase-h-new-backbone.md) · design: [Phase-H charter](../Phase-H%20charter.md)

## Files
| file | role |
|---|---|
| `ph_base.py` | standard pre-LN multi-head Transformer + mean-pool readout (the base) |
| `ph_nli.py` | **G0** — synthetic NLI (same dist as ZeroBP matrix). `python3 phase_h/ph_nli.py` → 100% |
| `ph_arith.py` | **G0** — synthetic 2-step arithmetic. `python3 phase_h/ph_arith.py` → 100% |
| `ph_nli_gpu.py` | **G1** — real SNLI/MNLI (or `--source synthetic` smoke). writes `runs/ph_nli_run_summary.json` |
| `ph_gsm_gpu.py` | **G2** — multi-step arithmetic depth sweep (classification). writes `runs/ph_gsm_run_summary.json` |
| `build_ph_kernels.py` | build the Kaggle notebooks `kaggle_ph_nli/` + `kaggle_ph_gsm/` (T4) from the .py files |
| `orchestrate_ph.py` | push→poll→pull→record the G1/G2 kernels (needs Kaggle creds) |

## Status
G0 PASS (local, synthetic). G1 + G2 scaffolds ready + smoke-tested; real SNLI + deep multi-step (GPU) pending.
G2b (real natural-language GSM8K, generative) = documented stretch, not scaffolded here.
