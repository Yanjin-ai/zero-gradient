"""Phase F / H1-attn-hybrid: a little BP on the TOP attention (Wq/Wk) during shaping -- does it install
relational structure that a FRESH (zero-shot) head can read? Three ZeroBP routes failed (F1 data +2pp,
F2 structural +0.5pp, Track A attention local rule). H1 crosses into the Hybrid layer: autograd touches
ONLY embedding/attention/head; the MoE experts / routing / lower blocks stay ZeroBP.

We shape the F1 base on the structural relation task with a little BP, then evaluate NLI with a FRESH
closed-form head on the shaped frozen rep (so we measure whether the REPRESENTATION gained structure,
not whether the BP head fit). Variants: BP(emb) vs BP(emb+attn). Success = zero-shot lifts well past 49%.

Run:  python3 h1_attn.py
"""
import torch
import kaggle_zerograd_moe as Z
import task_nli as T
import f1_data as F1
import phase_e as PE

def main():
    cfg = Z.Config(name="h1", vocab=T.VOCAB, seq_len=T.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Xtr, Ytr = T.gen_cls(6000, Z.SEED+2); Xv, Yv = T.gen_cls(1500, Z.SEED+3); d = cfg.d_model
    maj = max(float((Yv == c).float().mean()) for c in range(T.NCLS))
    base = Z.ZeroGradMoE(cfg, T.VOCAB); Z.train(base, T.lm_data(F1.gen_richer(8000, Z.SEED), cfg), cfg)
    baseA = base.state_dict()
    def fresh_zs(): return T.acc(base, T.train_head(base, Xtr, Ytr, d, T.NCLS), Xv, Yv)   # fresh head on (shaped) frozen rep

    print(f"\n==== Phase F / H1-attn-hybrid: little BP shaping -> FRESH-head NLI zero-shot  (majority {maj*100:.1f}%) ====")
    print(f"  {'shaping':34} {'NLI zero-shot (fresh head)':>26}")
    base.load_state_dict(baseA); zs0 = fresh_zs(); print(f"  {'none (F1 base)':34} {zs0*100:>25.1f}%")
    PE.run_phase_e(base, baseA, Xtr, Ytr, T.lm_data(F1.gen_richer(2000, Z.SEED), cfg), cfg, 1000, lr=0.1, bp_top=False, ncls=T.NCLS, bp_attn=False)
    zs_e = fresh_zs(); print(f"  {'BP embedding (H1-emb)':34} {zs_e*100:>25.1f}%")
    PE.run_phase_e(base, baseA, Xtr, Ytr, T.lm_data(F1.gen_richer(2000, Z.SEED), cfg), cfg, 1000, lr=0.1, bp_top=False, ncls=T.NCLS, bp_attn=True)
    zs_ea = fresh_zs(); print(f"  {'BP embedding+attention (H1-attn)':34} {zs_ea*100:>25.1f}%")

    best = max(zs_e, zs_ea)
    print(f"\n  F1 base {zs0*100:.1f}%  ->  H1-emb {zs_e*100:.1f}%  ->  H1-emb+attn {zs_ea*100:.1f}%")
    print(f"  attention's extra: {(zs_ea-zs_e)*100:+.1f}pp")
    print(f"  [{'H1 BREAKS the wall (fresh-head NLI zero-shot lifts well past 49%)' if best > 0.62 else 'H1 limited'}]")
    return zs0, zs_e, zs_ea

if __name__ == "__main__":
    main()
