"""Phase E task #3 -- 2-step modular arithmetic (multi-step reasoning), 5-class.

  seq = [d1 op1 d2 op2 d3]   (digits 0..4, ops +/-)
  label = ((d1 op1 d2) op2 d3) mod 5     (genuine 2-step left-to-right computation)

Runs the same 4-way comparison as the other tasks (zero-shot / zero-BP full adapt / Mixed-BP(embedding)
/ Mixed-BP(embedding+attention)) on the small config, to place this task in the task-method matrix:
does multi-step reasoning behave like sentiment (Mixed-BP breaks the plateau) or like NLI (stays near
chance)? Uses the verified reset path (load_state_dict clones; see selfcheck.py).

Run:  python3 task_arith.py
"""
import math, random, torch
from dataclasses import replace
import kaggle_zerograd_moe as Z
import phase_e as PE

PAD = 0; PLUS, MINUS = 6, 7; VOCAB = 8; L = 5; NCLS = 5      # digits 0..4 -> tokens 1..5
def tok(d): return d + 1

def compute(d1, o1, d2, o2, d3):
    v = d1 + d2 if o1 == PLUS else d1 - d2
    v = v + d3 if o2 == PLUS else v - d3
    return v % 5

def gen_O(n, seed):                                          # general corpus: random well-formed expressions
    rng = random.Random(seed); out = []
    for _ in range(n):
        out += [tok(rng.randrange(5)), rng.choice([PLUS, MINUS]), tok(rng.randrange(5)),
                rng.choice([PLUS, MINUS]), tok(rng.randrange(5))]
    return out

def gen_cls(n, seed):
    rng = random.Random(seed); X, Y = [], []
    for _ in range(n):
        d1, d2, d3 = (rng.randrange(5) for _ in range(3)); o1, o2 = rng.choice([PLUS, MINUS]), rng.choice([PLUS, MINUS])
        X.append(torch.tensor([tok(d1), o1, tok(d2), o2, tok(d3)], dtype=torch.long)); Y.append(compute(d1, o1, d2, o2, d3))
    return torch.stack(X), torch.tensor(Y)

def lm_data(stream, cfg):
    ids = torch.tensor(stream, dtype=torch.long); Xs, Ys = [], []
    for i in range(len(ids)-cfg.seq_len-1): Xs.append(ids[i:i+cfg.seq_len]); Ys.append(ids[i+cfg.seq_len])
    X, Y = torch.stack(Xs), torch.stack(Ys); nval = len(X)//10
    cnt = torch.bincount(Y[nval:], minlength=VOCAB).float()+1e-6; p = cnt/cnt.sum()
    return dict(corpus="arith/word", vocab=list(range(VOCAB)), Xtr=X[nval:].to(Z.DEVICE), Ytr=Y[nval:].to(Z.DEVICE),
                Y2tr=Y[nval:].to(Z.DEVICE), Xval=X[:nval].to(Z.DEVICE), Yval=Y[:nval].to(Z.DEVICE),
                unigram_ppl=math.exp(float(-(p[Y[nval:]]).log().mean())))

def train_head(model, Xc, Yc, d, ncls, steps=1000, lr=0.2):
    g = torch.Generator().manual_seed(Z.SEED)
    W1 = (torch.randn(d, d, generator=g)/math.sqrt(d)).to(Z.DEVICE); b1 = torch.zeros(d, device=Z.DEVICE)
    W2 = (torch.randn(d, ncls, generator=g)/math.sqrt(d)).to(Z.DEVICE); b2 = torch.zeros(ncls, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(Z.SEED+step); ix = torch.randint(0, len(Xc), (64,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, _, _ = model.forward(xb); h = h.float(); z1 = h @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        W2 = W2 - lr*(a1.T @ p); b2 = b2 - lr*p.sum(0); dz1 = (p @ W2.T)*(z1 > 0).float()
        W1 = W1 - lr*(h.T @ dz1); b1 = b1 - lr*dz1.sum(0)
    return (W1, b1, W2, b2)

def acc(model, hp, Xc, Yc):
    W1, b1, W2, b2 = hp; c = 0
    for i in range(0, len(Xc), 64):
        xb = Xc[i:i+64].to(Z.DEVICE); h, _, _ = model.forward(xb)
        c += int((torch.relu(h.float() @ W1 + b1) @ W2 + b2).argmax(-1).cpu().eq(Yc[i:i+64]).sum())
    return c/len(Xc)

def main():
    cfg = Z.Config(name="arith", vocab=VOCAB, seq_len=L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Odata = lm_data(gen_O(8000, Z.SEED), cfg); Sdata = lm_data(gen_O(8000, Z.SEED+1), cfg)
    Xtr, Ytr = gen_cls(6000, Z.SEED+2); Xv, Yv = gen_cls(1500, Z.SEED+3)
    maj = max(float((Ytr == c).float().mean()) for c in range(NCLS))
    base = Z.ZeroGradMoE(cfg, VOCAB); Z.train(base, Odata, cfg); baseA = base.state_dict()
    oppl_A = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg); d = cfg.d_model; rows = []
    def headacc(m): return acc(m, train_head(m, Xtr, Ytr, d, NCLS), Xv, Yv)
    rows.append(("Zero-shot head", headacc(base), oppl_A))
    for name, fh in [("Zero-BP full adapt", False), ("Zero-BP freeze-head", True)]:
        base.load_state_dict(baseA)
        Z.train(base, Sdata, replace(cfg, steps=1000, warmup_steps=40, eval_every=10**9, freeze_routing_step=0, freeze_heads=fh))
        rows.append((name, headacc(base), Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)))
    PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 1000, lr=0.1, bp_top=False, ncls=NCLS)
    rows.append(("Phase E Mixed-BP (emb)", headacc(base), Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)))
    PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 1000, lr=0.1, bp_top=False, ncls=NCLS, bp_attn=True)
    rows.append(("Phase E Mixed-BP (emb+attn)", headacc(base), Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)))

    print(f"\n==== Phase E task #3: 2-step arithmetic (5-class)  majority {maj*100:.1f}%  O-ppl(A)={oppl_A:.2f} ====")
    print(f"  {'route':28} {'acc':>7} {'O_ppl':>8} {'forget':>9}")
    for nm, a, op in rows: print(f"  {nm:28} {a*100:>6.1f}% {op:>8.2f} {op-oppl_A:>+9.2f}")
    zb = max(a for nm, a, _ in rows if "Mixed-BP" not in nm); mb = rows[-1][1]
    print(f"\n  best zero-BP {zb*100:.1f}%  ->  Mixed-BP(emb+attn) {mb*100:.1f}%  (gain {(mb-zb)*100:+.1f}pp)")
    print(f"  pattern: {'sentiment-like (Mixed-BP breaks plateau)' if mb > zb + 0.10 else 'NLI-like (Mixed-BP cannot break it)'}")
    return rows

if __name__ == "__main__":
    main()
