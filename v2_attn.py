"""Phase G / v2.0 (RESEARCH-ONLY, never touches the submission) -- trainable attention, ISOLATED.

The Phase G readout line (`v2_readout.py`) REFUTED the last-position-collapse hypothesis: on a frozen
ZeroBP base, non-collapsing readouts (mean / all-pos) are chance -> relational structure simply is NOT in
the frozen representation. The updated INTERPRETATION (ADR-004): the real bottleneck is UPSTREAM -- the
frozen random attention never forms cross-sentence alignment -- so the true lever is TRAINABLE ATTENTION.

This file tests that lever in isolation. The existing engine already co-trains embedding+attention on the
NLI CE loss (task_nli: emb 57.9% -> emb+attn 61.7%; H1 fresh-head: emb 58.8% -> emb+attn 59.1%, attn's
marginal share only +0.3pp). What was NEVER run is attention ALONE: freeze the embedding (the known
dominant lever) and train ONLY Wq/Wk on the NLI label CE. Same F1 base + same FRESH closed-form head as
H1 (so we measure whether the REPRESENTATION gained relational structure, not whether a BP head fit).

Decisive read:
  - attn-only ~= 51% floor      -> attention alignment alone is NOT a learnable lever here; embedding does
                                   the work in the 61.7% -> strengthens ADR-002 (relational = real limit).
  - attn-only jumps to ~59%+    -> trainable attention IS the lever -> ADR-004 PROPOSAL -> Accepted.

Strict research branch; does NOT import or modify the submission path. Run: python3 v2_attn.py
"""
import torch
import kaggle_zerograd_moe as Z
import task_nli as T
import f1_data as F1
import phase_e as PE


def main():
    cfg = Z.Config(name="v2attn", vocab=T.VOCAB, seq_len=T.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Xtr, Ytr = T.gen_cls(6000, Z.SEED+2); Xv, Yv = T.gen_cls(1500, Z.SEED+3); d = cfg.d_model
    maj = max(float((Yv == c).float().mean()) for c in range(T.NCLS))
    base = Z.ZeroGradMoE(cfg, T.VOCAB); Z.train(base, T.lm_data(F1.gen_richer(8000, Z.SEED), cfg), cfg)   # same F1 base as H1
    baseA = base.state_dict()
    Odata = T.lm_data(F1.gen_richer(2000, Z.SEED), cfg)
    def fresh_zs(): return T.acc(base, T.train_head(base, Xtr, Ytr, d, T.NCLS), Xv, Yv)   # FRESH closed-form head on (shaped) frozen rep

    print(f"\n==== Phase G / v2.0 trainable attention ISOLATED: FRESH-head NLI zero-shot  (majority {maj*100:.1f}%) ====")
    print(f"  {'shaping (NLI-label CE)':40} {'NLI zero-shot (fresh head)':>26}")

    base.load_state_dict(baseA); zs0 = fresh_zs(); print(f"  {'none (F1 base)':40} {zs0*100:>25.1f}%")
    # attention ONLY (embedding frozen) -- the decisive, never-before-run arm
    PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 1000, lr=0.1, bp_top=False, ncls=T.NCLS, bp_attn=True, bp_emb=False)
    zs_a = fresh_zs(); print(f"  {'BP attention ONLY (emb frozen)  [NEW]':40} {zs_a*100:>25.1f}%")
    # references (reproduce H1) -- embedding lever, with/without attention
    PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 1000, lr=0.1, bp_top=False, ncls=T.NCLS, bp_attn=False, bp_emb=True)
    zs_e = fresh_zs(); print(f"  {'BP embedding only (ref H1-emb)':40} {zs_e*100:>25.1f}%")
    PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 1000, lr=0.1, bp_top=False, ncls=T.NCLS, bp_attn=True, bp_emb=True)
    zs_ea = fresh_zs(); print(f"  {'BP embedding+attention (ref H1-attn)':40} {zs_ea*100:>25.1f}%")

    print(f"\n  F1 base {zs0*100:.1f}%  |  attn-only {zs_a*100:.1f}%  |  emb-only {zs_e*100:.1f}%  |  emb+attn {zs_ea*100:.1f}%")
    print(f"  attention's STANDALONE lift over floor: {(zs_a-zs0)*100:+.1f}pp")
    verdict = ("TRAINABLE ATTENTION IS A STANDALONE LEVER (attn-only escapes the floor) -> ADR-004 -> Accepted"
               if zs_a > 0.58 else
               "attention alone does NOT escape the floor -> embedding is the lever; relational limit stands (ADR-002)")
    print(f"  [{verdict}]")
    return dict(base=zs0, attn_only=zs_a, emb_only=zs_e, emb_attn=zs_ea)


if __name__ == "__main__":
    main()
