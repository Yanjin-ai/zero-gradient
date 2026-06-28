"""v1.1 mitigation 2.1: importance weighting / component freezing for the adaptation-forgetting trade-off.

Hypothesis from the 4B-Adapt failure: catastrophic forgetting was largely the LM readout head being
overwritten toward the narrow sentiment vocab. Test mitigations on the small config at a fixed adapt
budget N: protect the LM head (freeze_heads) and/or slow the backbone (backbone_lr_scale = importance
weighting; 0 = head-only). Re-draw sentiment acc vs original-domain PPL. A good mitigation moves the
point down-right (task stays high, forgetting drops).

Run:  python3 adapt_mitigate.py
"""
import torch
from dataclasses import replace
import kaggle_zerograd_moe as Z
import adapt_sentiment as A

def adapt_and_eval(baseA, cfg, Sdata, Odata, cls, N, freeze_heads, blr):
    m = Z.ZeroGradMoE(cfg, A.VOCAB); m.load_state_dict(baseA)
    if N > 0:
        acfg = replace(cfg, steps=N, warmup_steps=min(40, max(1, N//4)), eval_every=10**9, time_limit_s=120,
                       freeze_heads=freeze_heads, backbone_lr_scale=blr, freeze_routing_step=0)
        Z.train(m, Sdata, acfg)
    oppl = Z.evaluate(m, Odata["Xval"], Odata["Yval"], cfg)
    (Xtr_c, Ytr_c), (Xv_c, Yv_c) = cls
    hp = A.train_head_mlp(m, Xtr_c, Ytr_c, cfg.d_model); acc = A.acc_mlp(m, hp, Xv_c, Yv_c)
    return acc, oppl

def main():
    cfg = Z.Config(name="mit", vocab=A.VOCAB, seq_len=A.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Odata = A.lm_data(A.gen_O(8000, Z.SEED), cfg); Sdata = A.lm_data(A.gen_S_stream(8000, Z.SEED+1), cfg)
    Xtr_c, Ytr_c = A.gen_S_cls(6000, Z.SEED+2); Xv_c, Yv_c = A.gen_S_cls(1500, Z.SEED+3); cls = ((Xtr_c, Ytr_c), (Xv_c, Yv_c))
    maj = max(float((Ytr_c == 0).float().mean()), float((Ytr_c == 1).float().mean()))
    base = Z.ZeroGradMoE(cfg, A.VOCAB); Z.train(base, Odata, cfg); baseA = base.state_dict()
    oppl_A = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)

    N = 400
    configs = [("zero-shot (N=0)", 0, False, 1.0),
               ("full adapt", N, False, 1.0),
               ("freeze LM head", N, True, 1.0),
               ("backbone x0.1", N, False, 0.1),
               ("freeze head + bb x0.3", N, True, 0.3)]
    print(f"\n==== v1.1 MITIGATION (importance / freezing) N={N}  majority {maj*100:.1f}%  O-ppl(Base A)={oppl_A:.2f} ====")
    print(f"  {'config':24} {'sent_acc':>9} {'O_ppl':>8} {'forget_dPPL':>12}")
    rows = []
    for name, n, fh, blr in configs:
        acc, oppl = adapt_and_eval(baseA, cfg, Sdata, Odata, cls, n, fh, blr)
        rows.append((name, acc, oppl, oppl-oppl_A))
        print(f"  {name:24} {acc*100:>8.1f}% {oppl:>8.2f} {oppl-oppl_A:>+12.2f}")
    zs = rows[0]
    good = [r for r in rows[1:] if r[1] > zs[1] + 0.10 and r[3] < 5.0]      # acc up >10pp AND forgetting < +5 ppl
    print(f"\n  zero-shot: {zs[1]*100:.1f}% / dPPL +0.0")
    if good:
        b = max(good, key=lambda r: r[1])
        print(f"  [PASS] mitigation found: '{b[0]}' -> {b[1]*100:.1f}% acc at dPPL {b[3]:+.2f} (vs full adapt's heavy forgetting)")
    else:
        print(f"  [WEAK] no config beats zero-shot by >10pp while keeping forgetting < +5 ppl; try structural partition (2.2)")
    return rows

if __name__ == "__main__":
    main()
