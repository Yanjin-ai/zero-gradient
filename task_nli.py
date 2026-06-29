"""Phase E task #2 -- synthetic NLI (logical relation entailment), 3-class, harder than the XOR sentiment.

  seq = [E1 r1 than E2 SEP E3 r2 than E4]   (r in {GT, LT})
  label: entailment   if (E3,E4)=(E2,E1) and r2 = inverse(r1)   ("A>B"  =>  "B<A")
         contradiction if (E3,E4)=(E2,E1) and r2 = r1            ("A>B"  vs  "B>A")
         neutral       if the hypothesis is about a different entity pair
Requires comparing BOTH entity pairs AND the relations across the SEP -> compositional, 3-class.

Runs the SAME 4-way comparison as the sentiment task (zero-shot / zero-BP full adapt / zero-BP
freeze-head / Phase E Mixed-BP) on the small config, to test whether the pattern repeats:
zero-BP post-training plateaus, a little embedding+head BP breaks it.

Run:  python3 task_nli.py
"""
import math, random, torch
from dataclasses import replace
import kaggle_zerograd_moe as Z
import phase_e as PE

PAD, THAN, SEP, GT, LT = 0, 1, 2, 3, 4
ENT = list(range(5, 13)); VOCAB = 13; L = 9; NCLS = 3
INV = {GT: LT, LT: GT}

def gen_O(n, seed):                                            # general corpus: random well-formed sequences (no label logic)
    rng = random.Random(seed); out = []
    for _ in range(n):
        e1, e2 = rng.sample(ENT, 2); e3, e4 = rng.sample(ENT, 2)
        out += [e1, rng.choice([GT, LT]), THAN, e2, SEP, e3, rng.choice([GT, LT]), THAN, e4]
    return out

def gen_cls(n, seed):                                          # labeled NLI examples
    rng = random.Random(seed); X, Y = [], []
    for _ in range(n):
        e1, e2 = rng.sample(ENT, 2); r1 = rng.choice([GT, LT]); lab = rng.randrange(3)
        if lab == 0: e3, e4, r2 = e2, e1, INV[r1]              # entailment
        elif lab == 1: e3, e4, r2 = e2, e1, r1                 # contradiction
        else:                                                  # neutral: a different entity pair
            while True:
                e3, e4 = rng.sample(ENT, 2)
                if {e3, e4} != {e1, e2}: break
            r2 = rng.choice([GT, LT])
        X.append(torch.tensor([e1, r1, THAN, e2, SEP, e3, r2, THAN, e4], dtype=torch.long)); Y.append(lab)
    return torch.stack(X), torch.tensor(Y)

def lm_data(stream, cfg):
    ids = torch.tensor(stream, dtype=torch.long); Xs, Ys = [], []
    for i in range(len(ids)-cfg.seq_len-1): Xs.append(ids[i:i+cfg.seq_len]); Ys.append(ids[i+cfg.seq_len])
    X, Y = torch.stack(Xs), torch.stack(Ys); nval = len(X)//10
    cnt = torch.bincount(Y[nval:], minlength=VOCAB).float()+1e-6; p = cnt/cnt.sum()
    return dict(corpus="nli/word", vocab=list(range(VOCAB)), Xtr=X[nval:].to(Z.DEVICE), Ytr=Y[nval:].to(Z.DEVICE),
                Y2tr=Y[nval:].to(Z.DEVICE), Xval=X[:nval].to(Z.DEVICE), Yval=Y[:nval].to(Z.DEVICE),
                unigram_ppl=math.exp(float(-(p[Y[nval:]]).log().mean())))

def train_head(model, Xc, Yc, d, ncls, steps=1000, lr=0.2):    # closed-form n-class MLP head (zero autograd)
    g = torch.Generator().manual_seed(Z.SEED)
    W1 = (torch.randn(d, d, generator=g)/math.sqrt(d)).to(Z.DEVICE); b1 = torch.zeros(d, device=Z.DEVICE)
    W2 = (torch.randn(d, ncls, generator=g)/math.sqrt(d)).to(Z.DEVICE); b2 = torch.zeros(ncls, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(Z.SEED+step); ix = torch.randint(0, len(Xc), (64,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, _, _ = model.forward(xb); h = h.float()
        z1 = h @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        dW2 = a1.T @ p; db2 = p.sum(0); dz1 = (p @ W2.T)*(z1 > 0).float()
        W2 = W2 - lr*dW2; b2 = b2 - lr*db2; W1 = W1 - lr*(h.T @ dz1); b1 = b1 - lr*dz1.sum(0)
    return (W1, b1, W2, b2)

def acc(model, hp, Xc, Yc):
    W1, b1, W2, b2 = hp; c = 0
    for i in range(0, len(Xc), 64):
        xb = Xc[i:i+64].to(Z.DEVICE); h, _, _ = model.forward(xb)
        c += int((torch.relu(h.float() @ W1 + b1) @ W2 + b2).argmax(-1).cpu().eq(Yc[i:i+64]).sum())
    return c/len(Xc)

def main():
    cfg = Z.Config(name="nli", vocab=VOCAB, seq_len=L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Odata = lm_data(gen_O(8000, Z.SEED), cfg); Sdata = lm_data(gen_O(8000, Z.SEED+1), cfg)   # adapt uses NLI-structured stream
    Xtr, Ytr = gen_cls(6000, Z.SEED+2); Xv, Yv = gen_cls(1500, Z.SEED+3)
    maj = max(float((Ytr == c).float().mean()) for c in range(NCLS))
    base = Z.ZeroGradMoE(cfg, VOCAB); Z.train(base, Odata, cfg); baseA = base.state_dict()
    oppl_A = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)
    d = cfg.d_model; rows = []

    def headacc(m): return acc(m, train_head(m, Xtr, Ytr, d, NCLS), Xv, Yv)
    rows.append(("Zero-shot head", headacc(base), oppl_A, 0.0))                               # frozen rep
    for name, fh in [("Zero-BP full adapt", False), ("Zero-BP freeze-head", True)]:
        base.load_state_dict(baseA)
        Z.train(base, Sdata, replace(cfg, steps=1000, warmup_steps=40, eval_every=10**9, freeze_routing_step=0, freeze_heads=fh))
        rows.append((name, headacc(base), Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg), 0.0))
    _, oe1 = PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 1000, lr=0.1, bp_top=False, ncls=NCLS)               # BP embedding+head
    rows.append(("Phase E Mixed-BP (emb)", headacc(base), oe1, 0.0))
    _, oe2 = PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 1000, lr=0.1, bp_top=False, ncls=NCLS, bp_attn=True)  # +attention (relational)
    rows.append(("Phase E Mixed-BP (emb+attn)", headacc(base), oe2, 0.0))
    for i in range(1, len(rows)): rows[i] = (rows[i][0], rows[i][1], rows[i][2], rows[i][2]-oppl_A)

    print(f"\n==== Phase E task #2: NLI (3-class)  majority {maj*100:.1f}%  O-ppl(A)={oppl_A:.2f} ====")
    print(f"  {'route':22} {'nli_acc':>8} {'O_ppl':>8} {'forget':>9}")
    for nm, a, op, fg in rows: print(f"  {nm:22} {a*100:>7.1f}% {op:>8.2f} {fg:>+9.2f}")
    zb = max(a for nm, a, _, _ in rows if "Mixed-BP" not in nm); mb = rows[-1][1]
    print(f"\n  best zero-BP {zb*100:.1f}%  ->  Mixed-BP(emb+attn) {mb*100:.1f}%  (gain {(mb-zb)*100:+.1f}pp)")
    print(f"  [{'PATTERN REPEATS' if mb > zb + 0.10 else 'NO'}] Mixed-BP breaks the zero-BP plateau by >10pp")
    return rows

if __name__ == "__main__":
    main()
