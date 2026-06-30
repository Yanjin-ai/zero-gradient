"""Phase G / v2.0 (RESEARCH-ONLY, never touches the submission) -- non-collapsing readout.

Hypothesis (INTERPRETATION, charter/ADR-004): the relational-task wall is caused by LAST-POSITION
COLLAPSE -- the v1.0 model squashes the whole sequence to a single vector `(emb+att@emb)[:,-1]` before
the blocks, losing cross-position info. Test it cheaply: take the SAME frozen ZeroBP base and only change
the READOUT (no BP on the backbone), comparing v1.0 last-position vs non-collapsing pools (mean / all-
positions / concat[last, mean]). If a non-collapsing readout lifts NLI zero-shot well past the ~49% floor,
the collapse WAS the bottleneck -> Phase G direction confirmed (then ADR-004 -> Accepted).

This is a v2.0 architecture probe: it reads pre-collapse per-position features `emb + att@emb` [B,T,d].
Strict research branch; does NOT import or modify the submission path. Run: python3 v2_readout.py
"""
import math, torch
import kaggle_zerograd_moe as Z
import task_nli as T
import f1_data as F1

def rep_seq(base, xb):                                        # pre-collapse per-position rep [B,T,d] (frozen base)
    Tn = xb.shape[1]; emb = (base.E[xb] + base.pos[:Tn].unsqueeze(0)).float()
    q = emb @ base.Wq.float(); k = emb @ base.Wk.float()
    sc = (q @ k.transpose(1, 2))/math.sqrt(emb.shape[-1]); m = torch.triu(torch.ones(Tn, Tn, device=xb.device), 1).bool()
    att = torch.softmax(sc.masked_fill(m, float("-inf")), -1)
    return emb + att @ emb

def feats(base, X, kind):
    out = []
    for i in range(0, len(X), 64):
        xb = X[i:i+64].to(Z.DEVICE)
        if kind == "last-h (v1.0)":
            h, _, _ = base.forward(xb); f = h.float()
        else:
            rs = rep_seq(base, xb); pm = (xb != 0).float().unsqueeze(-1)
            if kind == "mean-pool": f = (rs*pm).sum(1)/(pm.sum(1)+1e-6)
            elif kind == "all-positions": f = rs.reshape(rs.shape[0], -1)
            elif kind == "concat[last,mean]":
                h, _, _ = base.forward(xb); mp = (rs*pm).sum(1)/(pm.sum(1)+1e-6); f = torch.cat([h.float(), mp], -1)
        out.append(f)
    return torch.cat(out)

def train_head(F, Y, ncls, hid=128, steps=1000, lr=0.2):     # closed-form 2-layer MLP on precomputed features (no autograd)
    din = F.shape[1]; g = torch.Generator().manual_seed(Z.SEED)
    W1 = (torch.randn(din, hid, generator=g)/math.sqrt(din)).to(Z.DEVICE); b1 = torch.zeros(hid, device=Z.DEVICE)
    W2 = (torch.randn(hid, ncls, generator=g)/math.sqrt(hid)).to(Z.DEVICE); b2 = torch.zeros(ncls, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(Z.SEED+step); ix = torch.randint(0, len(F), (64,), generator=gi)
        f, y = F[ix], Y[ix].to(Z.DEVICE); B = f.shape[0]
        z1 = f @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), y] -= 1.0; p /= B
        W2 = W2 - lr*(a1.T @ p); b2 = b2 - lr*p.sum(0); dz1 = (p @ W2.T)*(z1 > 0).float()
        W1 = W1 - lr*(f.T @ dz1); b1 = b1 - lr*dz1.sum(0)
    return (W1, b1, W2, b2)

def acc(hp, F, Y):
    W1, b1, W2, b2 = hp; return float((torch.relu(F @ W1 + b1) @ W2 + b2).argmax(-1).cpu().eq(Y).float().mean())

def main():
    cfg = Z.Config(name="v2ro", vocab=T.VOCAB, seq_len=T.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Xtr, Ytr = T.gen_cls(6000, Z.SEED+2); Xv, Yv = T.gen_cls(1500, Z.SEED+3)
    maj = max(float((Yv == c).float().mean()) for c in range(T.NCLS))
    base = Z.ZeroGradMoE(cfg, T.VOCAB); Z.train(base, T.lm_data(F1.gen_richer(8000, Z.SEED), cfg), cfg)   # frozen ZeroBP base
    print(f"\n==== Phase G / v2.0 non-collapsing readout: NLI zero-shot (frozen ZeroBP base)  majority {maj*100:.1f}% ====")
    print(f"  {'readout':24} {'NLI acc':>9}")
    rows = []
    for kind in ["last-h (v1.0)", "mean-pool", "all-positions", "concat[last,mean]"]:
        a = acc(train_head(feats(base, Xtr, kind), Ytr, T.NCLS), feats(base, Xv, kind), Yv)
        rows.append((kind, a)); print(f"  {kind:24} {a*100:>8.1f}%")
    v1 = rows[0][1]; best = max(rows[1:], key=lambda r: r[1])
    print(f"\n  v1.0 last-h {v1*100:.1f}%  ->  best non-collapsing '{best[0]}' {best[1]*100:.1f}%  (gain {(best[1]-v1)*100:+.1f}pp)")
    print(f"  [{'COLLAPSE WAS THE BOTTLENECK: non-collapsing readout lifts NLI well past 49%' if best[1] > 0.62 else 'collapse not the whole story (non-collapsing readout limited)'}]")
    return rows

if __name__ == "__main__":
    main()
