"""v1.1 domain-adaptation + forgetting trade-off (zero autograd, single lever).

Question the 4B zero-shot C.1 raised: head-only MLP sentiment on a WikiText-pretrained model only
hit ~60% (vs 100% in-domain). Does a SHORT in-domain LM adaptation lift it, and at what forgetting cost?

Setup (small, local, all zero-BP):
  - General corpus O: "the {subj} {pol} {obj}"  -> polarity words appear in NEUTRAL (adjective) contexts,
    so the base learns their embeddings but never the sentiment-XOR structure (mirrors WikiText zero-shot).
  - Sentiment corpus S: "the {subj} is [not] {pol}", label = (pol in POS) XOR negated (vocab-overlapping).
  - Base A = pretrain on O. Then for each N, reset to Base A and adapt (continue zero-BP LM) on S for N
    steps; measure head-only MLP sentiment acc AND O-PPL drift (forgetting). N=0 is the zero-shot baseline.

Output: a trade-off table  N(adapt steps) -> sentiment acc / O-PPL / forgetting delta.
Run:  python3 adapt_sentiment.py
"""
import math, random, torch
from dataclasses import replace
import kaggle_zerograd_moe as Z

SEED = Z.SEED
PAD, THE, IS, NOT = 0, 1, 2, 3
SUBJ = list(range(4, 12)); POS = list(range(12, 17)); NEG = list(range(17, 22)); OBJ = list(range(22, 27))
VOCAB = 27; L = 6

def gen_O(n, seed):                                            # general: polarity words as plain adjectives
    rng = random.Random(seed); out = []
    for _ in range(n):
        out += [THE, rng.choice(SUBJ), rng.choice(POS+NEG), rng.choice(OBJ)]
    return out

def gen_S_stream(n, seed):                                     # sentiment sentences as an LM stream (for adaptation)
    rng = random.Random(seed); out = []
    for _ in range(n):
        s, pos = rng.choice(SUBJ), rng.random() < 0.5; w = rng.choice(POS if pos else NEG)
        out += [THE, s, IS] + ([NOT] if rng.random() < 0.5 else []) + [w]
    return out

def gen_S_cls(n, seed):                                        # sentiment as a labeled classification set
    rng = random.Random(seed); X, Y = [], []
    for _ in range(n):
        s, pos = rng.choice(SUBJ), rng.random() < 0.5; w = rng.choice(POS if pos else NEG); neg = rng.random() < 0.5
        toks = [THE, s, IS] + ([NOT] if neg else []) + [w]; toks = [PAD]*(L-len(toks)) + toks
        X.append(torch.tensor(toks, dtype=torch.long)); Y.append(int(pos ^ (not neg)))
    return torch.stack(X), torch.tensor(Y)

def lm_data(stream, cfg):
    ids = torch.tensor(stream, dtype=torch.long)
    Xs, Ys = [], []
    for i in range(len(ids)-cfg.seq_len-1): Xs.append(ids[i:i+cfg.seq_len]); Ys.append(ids[i+cfg.seq_len])
    X, Y = torch.stack(Xs), torch.stack(Ys); nval = len(X)//10
    cnt = torch.bincount(Y[nval:], minlength=VOCAB).float()+1e-6; p = cnt/cnt.sum()
    return dict(corpus="adapt/word", vocab=list(range(VOCAB)), Xtr=X[nval:].to(Z.DEVICE), Ytr=Y[nval:].to(Z.DEVICE),
                Y2tr=Y[nval:].to(Z.DEVICE), Xval=X[:nval].to(Z.DEVICE), Yval=Y[:nval].to(Z.DEVICE),
                unigram_ppl=math.exp(float(-(p[Y[nval:]]).log().mean())))

def train_head_mlp(model, Xc, Yc, d, steps=800, lr=0.2):
    g = torch.Generator().manual_seed(SEED)
    W1 = (torch.randn(d, d, generator=g)/math.sqrt(d)).to(Z.DEVICE); b1 = torch.zeros(d, device=Z.DEVICE)
    W2 = (torch.randn(d, 2, generator=g)/math.sqrt(d)).to(Z.DEVICE); b2 = torch.zeros(2, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xc), (64,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, _, _ = model.forward(xb); h = h.float()
        z1 = h @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        W2 = W2 - lr*(a1.T @ p); b2 = b2 - lr*p.sum(0); dz1 = (p @ W2.T)*(z1 > 0).float()
        W1 = W1 - lr*(h.T @ dz1); b1 = b1 - lr*dz1.sum(0)
    return W1, b1, W2, b2

def acc_mlp(model, hp, Xc, Yc):
    W1, b1, W2, b2 = hp; c = 0
    for i in range(0, len(Xc), 64):
        xb, yb = Xc[i:i+64].to(Z.DEVICE), Yc[i:i+64].to(Z.DEVICE); h, _, _ = model.forward(xb)
        c += int((torch.relu(h.float() @ W1 + b1) @ W2 + b2).argmax(-1).eq(yb).sum())
    return c/len(Xc)

def main():
    cfg = Z.Config(name="adapt", vocab=VOCAB, seq_len=L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Odata = lm_data(gen_O(8000, SEED), cfg); Sdata = lm_data(gen_S_stream(8000, SEED+1), cfg)
    Xtr_c, Ytr_c = gen_S_cls(6000, SEED+2); Xv_c, Yv_c = gen_S_cls(1500, SEED+3)
    maj = max(float((Ytr_c == 0).float().mean()), float((Ytr_c == 1).float().mean()))

    base = Z.ZeroGradMoE(cfg, VOCAB); Z.train(base, Odata, cfg)         # Base A: pretrain on general corpus O
    baseA = base.state_dict(); oppl_A = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)

    print(f"\n==== v1.1 IN-DOMAIN ADAPTATION + FORGETTING (majority {maj*100:.1f}%, O-ppl(Base A)={oppl_A:.2f}) ====")
    print(f"  {'adapt_N':>8} {'sent_acc':>9} {'O_ppl':>8} {'forget_dPPL':>12}")
    rows = []
    for N in [0, 150, 400, 1000]:
        m = Z.ZeroGradMoE(cfg, VOCAB); m.load_state_dict(baseA)        # reset to Base A each time
        if N > 0:
            acfg = replace(cfg, steps=N, warmup_steps=min(40, max(1, N//4)), eval_every=10**9, time_limit_s=120)
            Z.train(m, Sdata, acfg)                                    # in-domain zero-BP LM adaptation on S
        oppl = Z.evaluate(m, Odata["Xval"], Odata["Yval"], cfg)        # forgetting on the original domain
        hp = train_head_mlp(m, Xtr_c, Ytr_c, cfg.d_model); acc = acc_mlp(m, hp, Xv_c, Yv_c)
        rows.append((N, acc, oppl, oppl - oppl_A))
        print(f"  {N:>8} {acc*100:>8.1f}% {oppl:>8.2f} {oppl-oppl_A:>+12.2f}")
    best = max(rows, key=lambda r: r[1])
    print(f"\n  zero-shot (N=0) acc = {rows[0][1]*100:.1f}%  ->  best adapted acc = {best[1]*100:.1f}% @ N={best[0]} "
          f"(forgetting dPPL {best[3]:+.2f})")
    print(f"  [{'PASS' if best[1] > rows[0][1] + 0.15 else 'WEAK'}] adaptation lifts sentiment acc by >15pp over zero-shot")
    return rows

if __name__ == "__main__":
    main()
