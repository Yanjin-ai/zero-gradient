"""Phase E -- Hybrid BP (small-config first). RESEARCH BRANCH: deliberately uses autograd, never the
zero-BP submission. Single lever: during adaptation, enable real BP on ONLY the top MoE block + task head;
embedding / lower blocks / routing / attention frozen; pretraining stays zero-BP.

Question: does a little real gradient (top block + head) install the compositional task features that
6 zero-BP routes could NOT (all plateaued ~60-62%), and at what forgetting cost?

Compare on the small config (vs the zero-BP references): zero-shot ~51% / 2.1 freeze-head 92%/+11 /
2.2 partition 96%/+0 / D.2 61%/+0. Gate G1: acc approach/beat 92%. Gate G2: forgetting << full-adapt(+31).

Run:  python3 phase_e.py
"""
import math, torch
import kaggle_zerograd_moe as Z
import adapt_sentiment as A

def block_fwd(hin, a, We, be, k):                              # functional MoE block (grad flows through hin; We may be const or leaf)
    out = torch.zeros_like(hin)
    for e in torch.unique(a).tolist():
        idx = (a == e).any(1).nonzero(as_tuple=True)[0]
        if idx.numel() == 0: continue
        out = out.index_add(0, idx, torch.relu(hin[idx] @ We[e] + be[e]))
    return hin + out/k

def ctx_fwd(base, xb, E_p):                                    # differentiable context with trainable embedding E_p
    T = xb.shape[1]; emb = E_p[xb] + base.pos[:T].unsqueeze(0)
    q = emb @ base.Wq; k = emb @ base.Wk
    sc = (q @ k.transpose(1, 2))/math.sqrt(emb.shape[-1])
    m = torch.triu(torch.ones(T, T, device=xb.device), 1).bool()
    att = torch.softmax(sc.masked_fill(m, float("-inf")), -1)
    return (emb + att @ emb)[:, -1], emb

def run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, N, lr=0.1, htask=64, bp_top=True):
    base.load_state_dict(baseA); L = cfg.n_layers-1; d = cfg.d_model
    E_p = base.E.detach().float().clone().requires_grad_(True)                     # embedding (trainable; the real lever)
    if bp_top:                                                                     # top-block experts trainable...
        We_p = [w.detach().float().clone().requires_grad_(True) for w in base.We[L]]
        be_p = [b.detach().float().clone().requires_grad_(True) for b in base.be[L]]
    else:                                                                          # ...or frozen (BP still reaches E through it)
        We_p = [w.detach().float() for w in base.We[L]]; be_p = [b.detach().float() for b in base.be[L]]
    Wlo = [[w.detach().float() for w in base.We[l]] for l in range(L)]             # lower experts (frozen constants)
    blo = [[b.detach().float() for b in base.be[l]] for l in range(L)]
    g = torch.Generator().manual_seed(Z.SEED)
    W1 = (torch.randn(d, htask, generator=g)/math.sqrt(d)).to(Z.DEVICE).requires_grad_(True)
    b1 = torch.zeros(htask, device=Z.DEVICE, requires_grad=True)
    W2 = (torch.randn(htask, 2, generator=g)/math.sqrt(htask)).to(Z.DEVICE).requires_grad_(True)
    b2 = torch.zeros(2, device=Z.DEVICE, requires_grad=True)
    params = [E_p] + (We_p + be_p if bp_top else []) + [W1, b1, W2, b2]
    for step in range(N):
        gi = torch.Generator().manual_seed(Z.SEED+step); ix = torch.randint(0, len(Xtr), (64,), generator=gi)
        xb, yb = Xtr[ix].to(Z.DEVICE), Ytr[ix].to(Z.DEVICE)
        with torch.no_grad():                                  # routing is non-diff (argsort) -> fix assignments per step
            _, rrep = base.context(xb)
            al = [base.route(rrep, base.C[l])[0] for l in range(cfg.n_layers)]
        with torch.enable_grad():                              # real BP through embedding -> lower blocks -> top block -> head
            h, _ = ctx_fwd(base, xb, E_p)
            for l in range(L): h = block_fwd(h, al[l], Wlo[l], blo[l], cfg.k_route)   # frozen experts (constants)
            htop = block_fwd(h, al[L], We_p, be_p, cfg.k_route)                        # trainable top block
            lg = torch.relu(htop @ W1 + b1) @ W2 + b2
            loss = torch.nn.functional.cross_entropy(lg, yb)
        grads = torch.autograd.grad(loss, params, allow_unused=True)   # unrouted experts -> None grad
        with torch.no_grad():
            for p, gr in zip(params, grads):
                if gr is not None: p -= lr*gr
    base.E = E_p.detach().to(cfg.td)                           # write back (for base forward / O-PPL)
    if bp_top:
        for e in range(len(base.We[L])):
            base.We[L][e] = We_p[e].detach().to(cfg.td); base.be[L][e] = be_p[e].detach().to(cfg.td)
    head = (W1.detach(), b1.detach(), W2.detach(), b2.detach())
    def acc(Xc, Yc):
        c = 0
        for i in range(0, len(Xc), 64):
            xb = Xc[i:i+64].to(Z.DEVICE); h, _, _ = base.forward(xb)
            lg = torch.relu(h.float() @ head[0] + head[1]) @ head[2] + head[3]
            c += int(lg.argmax(-1).cpu().eq(Yc[i:i+64]).sum())
        return c/len(Xc)
    oppl = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)
    return acc, oppl

def main():
    cfg = Z.Config(name="phaseE", vocab=A.VOCAB, seq_len=A.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Odata = A.lm_data(A.gen_O(8000, Z.SEED), cfg)
    Xtr_c, Ytr_c = A.gen_S_cls(6000, Z.SEED+2); Xv_c, Yv_c = A.gen_S_cls(1500, Z.SEED+3)
    maj = max(float((Ytr_c == 0).float().mean()), float((Ytr_c == 1).float().mean()))
    base = Z.ZeroGradMoE(cfg, A.VOCAB); Z.train(base, Odata, cfg); baseA = base.state_dict()
    oppl_A = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)
    print(f"\n==== Phase E HYBRID BP (top block + head, autograd)  majority {maj*100:.1f}%  O-ppl(A)={oppl_A:.2f} ====")
    print(f"  {'bp_N':>8} {'sent_acc':>9} {'O_ppl':>8} {'forget_dPPL':>12}")
    rows = []
    for N in [0, 150, 400, 1000]:
        accf, oppl = run_phase_e(base, baseA, Xtr_c, Ytr_c, Odata, cfg, N); a = accf(Xv_c, Yv_c)
        rows.append((N, a, oppl, oppl-oppl_A)); print(f"  {N:>8} {a*100:>8.1f}% {oppl:>8.2f} {oppl-oppl_A:>+12.2f}")
    best = max(rows[1:], key=lambda r: r[1])
    print(f"\n  zero-shot {rows[0][1]*100:.1f}%  ->  best Mixed-BP {best[1]*100:.1f}% @ N={best[0]} (forgetting {best[3]:+.2f})")
    print(f"  refs: zeroBP best small = 2.2 partition 96%/+0, 2.1 freeze-head 92%/+11, D.2 61%/+0")
    g1 = best[1] >= 0.92; g2 = best[3] < 31.0
    print(f"  [{'PASS' if g1 else 'FAIL'}] G1 acc >= 92%   [{'PASS' if g2 else 'FAIL'}] G2 forgetting < full-adapt(+31)")
    print(f"  -> {'migrate to 4B' if (g1 and g2) else 'gate fails; record negative'}")
    return rows

if __name__ == "__main__":
    main()
