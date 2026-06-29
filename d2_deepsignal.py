"""Phase D.2 -- ZeroBP-DeepSignal: give zero-BP one more shot at lifting the 4B ~62% ceiling.

Idea (no autograd): instead of adapting the top block by its own next-token readout (2.1, which barely
moved the task), adapt it by a VERTICAL TASK signal -- the task head's closed-form backward gives the exact
error w.r.t. the top block's output (dh_top), and we feed THAT into the top block's local expert update.
This is a single-step vertical propagation (task head -> top block), same closed-form class as the existing
deeply-supervised readout; no torch.autograd, no weight transport. Lower blocks / embedding / context /
LM head all FROZEN.

Question: does a task-aligned vertical signal reorganize the top representation better (higher task acc per
unit forgetting) than 2.1's next-token backbone adaptation? Compare on the small config:
  zero-shot (head on frozen top)        vs   D.2 (adapt top block by task signal, sweep N)
and watch original-domain PPL (forgetting; the top block is shared with the LM head).

Run:  python3 d2_deepsignal.py
"""
import math, torch
import kaggle_zerograd_moe as Z
import adapt_sentiment as A

def partial(base, xb, upto):                                   # frozen forward through blocks [0, upto)
    h, rrep = base.context(xb)
    for l in range(upto):
        a, _ = base.route(rrep, base.C[l]); h = base.block(h, a, l)
    return h, rrep

def top_forward(base, hpre, rrep, l):                          # top block with caching (for the local update)
    a, _ = base.route(rrep, base.C[l]); cache = {}; moe = torch.zeros_like(hpre)
    for e in torch.unique(a).tolist():
        mm = (a == e).any(1); inp = hpre[mm]; z = inp @ base.We[l][e] + base.be[l][e]
        moe[mm] = moe[mm] + torch.relu(z); cache[e] = (inp, z, mm)
    return hpre + moe/base.cfg.k_route, cache

def run_d2(base, baseA, Xtr, Ytr, Odata, cfg, N, lr=0.1, htask=64):
    base.load_state_dict(baseA)                                # reset to Base A
    L = cfg.n_layers-1; d = cfg.d_model; g = torch.Generator().manual_seed(Z.SEED)
    W1 = (torch.randn(d, htask, generator=g)/math.sqrt(d)).to(Z.DEVICE); b1 = torch.zeros(htask, device=Z.DEVICE)
    W2 = (torch.randn(htask, 2, generator=g)/math.sqrt(htask)).to(Z.DEVICE); b2 = torch.zeros(2, device=Z.DEVICE)
    for step in range(N):
        gi = torch.Generator().manual_seed(Z.SEED+step); ix = torch.randint(0, len(Xtr), (64,), generator=gi)
        xb, yb = Xtr[ix].to(Z.DEVICE), Ytr[ix].to(Z.DEVICE); B = xb.shape[0]
        hpre, rrep = partial(base, xb, L)                      # frozen lower blocks
        htop, cache = top_forward(base, hpre, rrep, L)         # adapted top block
        z1 = htop.float() @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        dz1 = (p @ W2.T) * (z1 > 0).float(); dh_top = dz1 @ W1.T          # vertical task signal -> top block
        W2 = W2 - lr*(a1.T @ p); b2 = b2 - lr*p.sum(0); W1 = W1 - lr*(htop.float().T @ dz1); b1 = b1 - lr*dz1.sum(0)
        for e in cache:                                        # local expert update driven by the TASK signal
            inp, z, mm = cache[e]; dz = (dh_top[mm]/cfg.k_route) * (z.float() > 0).float()
            base.We[L][e] = (base.We[L][e].float() - lr*(inp.float().T @ dz)).to(cfg.td)
            base.be[L][e] = (base.be[L][e].float() - lr*dz.sum(0)).to(cfg.td)
        base.E.index_add_(0, xb[:, -1], (-lr*dh_top).to(cfg.td))   # also push the vertical signal into the embedding
    # task acc
    def acc(Xc, Yc):
        c = 0
        for i in range(0, len(Xc), 64):
            xb = Xc[i:i+64].to(Z.DEVICE); hpre, rrep = partial(base, xb, L); htop, _ = top_forward(base, hpre, rrep, L)
            lg = torch.relu(htop.float() @ W1 + b1) @ W2 + b2; c += int(lg.argmax(-1).cpu().eq(Yc[i:i+64]).sum())
        return c/len(Xc)
    oppl = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)            # forgetting (top block shared with LM head)
    return acc, oppl

def main():
    cfg = Z.Config(name="d2", vocab=A.VOCAB, seq_len=A.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Odata = A.lm_data(A.gen_O(8000, Z.SEED), cfg)
    Xtr_c, Ytr_c = A.gen_S_cls(6000, Z.SEED+2); Xv_c, Yv_c = A.gen_S_cls(1500, Z.SEED+3)
    maj = max(float((Ytr_c == 0).float().mean()), float((Ytr_c == 1).float().mean()))
    base = Z.ZeroGradMoE(cfg, A.VOCAB); Z.train(base, Odata, cfg); baseA = base.state_dict()
    oppl_A = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)
    print(f"\n==== Phase D.2 ZeroBP-DeepSignal (vertical task signal -> top block)  majority {maj*100:.1f}%  O-ppl(A)={oppl_A:.2f} ====")
    print(f"  {'adapt_N':>8} {'sent_acc':>9} {'O_ppl':>8} {'forget_dPPL':>12}")
    rows = []
    for N in [0, 150, 400, 1000]:
        accf, oppl = run_d2(base, baseA, Xtr_c, Ytr_c, Odata, cfg, N, lr=0.3)
        a = accf(Xv_c, Yv_c); rows.append((N, a, oppl, oppl-oppl_A))
        print(f"  {N:>8} {a*100:>8.1f}% {oppl:>8.2f} {oppl-oppl_A:>+12.2f}")
    zs = rows[0][1]; best = max(rows[1:], key=lambda r: r[1])
    print(f"\n  zero-shot {zs*100:.1f}%  ->  best D.2 {best[1]*100:.1f}% @ N={best[0]} (forgetting {best[3]:+.2f})")
    print(f"  reference: 2.1 freeze-head was 92.4% / +11.0 ;  2.2 partition 95.9% / +0")
    print(f"  [{'PASS' if best[1] > 0.924 and best[3] < 11.0 else 'WEAK'}] D.2 beats 2.1 freeze-head (higher acc AND less forgetting)")
    return rows

if __name__ == "__main__":
    main()
