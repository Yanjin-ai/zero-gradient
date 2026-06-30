"""Phase F / F2-aux-zeroBP (strict ZeroBP): can a structural OBJECTIVE (not just data) make the local
rule install relational geometry? Train a 3-class sentence-pair relation head by closed-form CE and flow
its dh into the backbone (top-layer experts + embedding) via the existing ZeroBP local rule -- NO autograd.
Then read NLI zero-shot with a fresh closed-form head. If it barely moves, ZeroBP local rules can't carve
relational structure even when fed a structural target -> hard boundary -> H1 (hybrid) is necessary.

Run:  python3 f2_aux.py
"""
import math, torch
import kaggle_zerograd_moe as Z
import task_nli as T
import f1_data as F1

def shape_zerobp(base, Xc, Yc, cfg, steps=1000, lr=0.1, ncls=3):
    d = cfg.d_model; g = torch.Generator().manual_seed(Z.SEED)
    W1 = (torch.randn(d, d, generator=g)/math.sqrt(d)).to(Z.DEVICE); b1 = torch.zeros(d, device=Z.DEVICE)
    W2 = (torch.randn(d, ncls, generator=g)/math.sqrt(d)).to(Z.DEVICE); b2 = torch.zeros(ncls, device=Z.DEVICE)
    sched = [Z.RoundRobin(cfg.n_experts) for _ in range(cfg.n_layers)]
    for step in range(steps):
        gi = torch.Generator().manual_seed(Z.SEED+step); ix = torch.randint(0, len(Xc), (64,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, rrep = base.context(xb); hin = h; top = None
        for l in range(cfg.n_layers):
            assign, _ = base.route(rrep, base.C[l]); cache = {}; moe = torch.zeros_like(hin)
            for e in torch.unique(assign).tolist():
                mm = (assign == e).any(1); inp = hin[mm]; z = inp @ base.We[l][e] + base.be[l][e]
                moe[mm] = moe[mm] + torch.relu(z); cache[e] = (inp, z, mm)
            hin = hin + moe/cfg.k_route; top = (assign, cache)
        z1 = hin.float() @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2          # structural head (closed form)
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        dz1 = (p @ W2.T) * (z1 > 0).float(); dh = dz1 @ W1.T
        W2 = W2 - lr*(a1.T @ p); b2 = b2 - lr*p.sum(0); W1 = W1 - lr*(hin.float().T @ dz1); b1 = b1 - lr*dz1.sum(0)
        assign, cache = top; cand = torch.tensor(sorted(cache.keys())); sel = sched[-1].select(cand, cfg.k_update)
        for e in sel:                                                              # flow dh -> experts (ZeroBP local rule)
            inp, z, mm = cache[e]; dz = (dh[mm]/cfg.k_route) * (z.float() > 0).float()
            base.We[-1][e] = (base.We[-1][e].float() - lr*(inp.float().T @ dz)).to(cfg.td)
            base.be[-1][e] = (base.be[-1][e].float() - lr*dz.sum(0)).to(cfg.td)
        base.E.index_add_(0, xb[:, -1], (-lr*dh).to(cfg.td))                       # ...and embedding (last token)

def main():
    cfg = Z.Config(name="f2", vocab=T.VOCAB, seq_len=T.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Xtr, Ytr = T.gen_cls(6000, Z.SEED+2); Xv, Yv = T.gen_cls(1500, Z.SEED+3); d = cfg.d_model
    maj = max(float((Yv == c).float().mean()) for c in range(T.NCLS))
    base = Z.ZeroGradMoE(cfg, T.VOCAB); Z.train(base, T.lm_data(F1.gen_richer(8000, Z.SEED), cfg), cfg)   # F1 richer base
    zs_before = T.acc(base, T.train_head(base, Xtr, Ytr, d, T.NCLS), Xv, Yv)
    shape_zerobp(base, Xtr, Ytr, cfg)                                                                     # ZeroBP structural shaping
    zs_after = T.acc(base, T.train_head(base, Xtr, Ytr, d, T.NCLS), Xv, Yv)
    print(f"\n==== Phase F / F2-aux-zeroBP: structural objective via ZeroBP local rule  (majority {maj*100:.1f}%) ====")
    print(f"  NLI zero-shot  before(F1 base)={zs_before*100:.1f}%  ->  after ZeroBP structural shaping={zs_after*100:.1f}%")
    print(f"  gain {(zs_after-zs_before)*100:+.1f}pp")
    print(f"  [{'SIGNAL: ZeroBP structural objective lifts NLI' if zs_after > zs_before + 0.05 else 'NO: ZeroBP cannot carve relational geometry even with a structural target -> H1 needed'}]")
    return zs_before, zs_after

if __name__ == "__main__":
    main()
