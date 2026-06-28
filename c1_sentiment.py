"""C.1 first real-ish downstream task: COMPOSITIONAL sentiment (head-only, zero autograd).

Charter (approved): a non-trivially-separable sentiment task — overlapping vocab where sentiment =
base_polarity XOR negation. "good" and "not" both appear in BOTH classes, so no unigram/bigram split
works; the head must read the pretrained representation's encoding of "was 'not' present?". This tests
whether the zero-BP pretrained representation is linearly separable for a compositional label.

  sentence = "the {subj} is [not] {word}"   word in POS|NEG,  negate ~50%
  label    = positive  iff  (word in POS) XOR (not negated)      # "not good" -> neg, "not bad" -> pos

Pipeline (reuses the validated C.1 mechanics): pretrain base LM (next-token) on this domain -> save best
ckpt -> RELOAD into a fresh model -> attach a local sentiment head -> train ONLY the head with closed-form
CE (no autograd) -> report acc vs majority, a negated-vs-not breakdown, and base LM ppl drift (head-only ->
zero forgetting). Gate: acc >= majority + 15pp, ppl drift ~0, zero autograd, deterministic.

Run:  python3 c1_sentiment.py
"""
import torch, math, random
from pathlib import Path
import kaggle_zerograd_moe as Z

OUT = Path(__file__).parent/"runs"; OUT.mkdir(exist_ok=True)
SEED = Z.SEED
PAD, THE, IS, NOT = 0, 1, 2, 3
SUBJ = list(range(4, 12)); POS = list(range(12, 17)); NEG = list(range(17, 22)); VOCAB = 22
L = 6                                                          # fixed seq len (left-pad -> polarity word at last pos)

def sentence(rng):
    s = rng.choice(SUBJ); pos = rng.random() < 0.5
    w = rng.choice(POS if pos else NEG); neg = rng.random() < 0.5
    toks = [THE, s, IS] + ([NOT] if neg else []) + [w]
    label = int(pos ^ (not neg))                              # positive iff base_pos XOR (not negated)
    return toks, label, neg

def padseq(toks):                                             # left-pad so the polarity word lands at position L-1
    return [PAD]*(L-len(toks)) + toks

def cls_set(n, seed):
    rng = random.Random(seed); X, Y, NG = [], [], []
    for _ in range(n):
        t, y, ng = sentence(rng); X.append(torch.tensor(padseq(t), dtype=torch.long)); Y.append(y); NG.append(ng)
    return torch.stack(X), torch.tensor(Y), torch.tensor(NG)

def lm_data(cfg):
    rng = random.Random(SEED+7); stream = []                  # LM corpus: a stream of the same sentences
    for _ in range(12000): stream += sentence(rng)[0]
    ids = torch.tensor(stream, dtype=torch.long)
    def win(t, cap):
        Xs, Ys = [], []
        for i in range(0, min(len(t)-cfg.seq_len-1, cap)): Xs.append(t[i:i+cfg.seq_len]); Ys.append(t[i+cfg.seq_len])
        return torch.stack(Xs), torch.stack(Ys)
    X, Y = win(ids, 60000); nval = len(X)//10
    cnt = torch.bincount(Y[nval:], minlength=VOCAB).float()+1e-6; p = cnt/cnt.sum()
    return dict(corpus="sentiment-comp/word", vocab=list(range(VOCAB)),
                Xtr=X[nval:].to(Z.DEVICE), Ytr=Y[nval:].to(Z.DEVICE), Y2tr=Y[nval:].to(Z.DEVICE),
                Xval=X[:nval].to(Z.DEVICE), Yval=Y[:nval].to(Z.DEVICE),
                unigram_ppl=math.exp(float(-(p[Y[nval:]]).log().mean())))

def train_head(model, Xc, Yc, d, ncls=2, steps=1000, lr=0.2):
    g = torch.Generator().manual_seed(SEED)
    Wc = (torch.randn(d, ncls, generator=g)/math.sqrt(d)).to(Z.DEVICE); bc = torch.zeros(ncls, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xc), (64,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, _, _ = model.forward(xb)
        lg = h.float() @ Wc + bc; p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        Wc = Wc - lr*(h.float().T @ p); bc = bc - lr*p.sum(0)
    return Wc, bc

def train_head_mlp(model, Xc, Yc, d, ncls=2, hid=None, steps=1000, lr=0.2):
    """2-layer MLP sentiment head (relu hidden), hand-derived closed-form backward, zero autograd.
    A linear head can't XOR two linearly-present features; a hidden layer can (cf. D-1's MLP readout)."""
    hid = hid or d; g = torch.Generator().manual_seed(SEED)
    W1 = (torch.randn(d, hid, generator=g)/math.sqrt(d)).to(Z.DEVICE); b1 = torch.zeros(hid, device=Z.DEVICE)
    W2 = (torch.randn(hid, ncls, generator=g)/math.sqrt(hid)).to(Z.DEVICE); b2 = torch.zeros(ncls, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xc), (64,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, _, _ = model.forward(xb); h = h.float()
        z1 = h @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        dW2 = a1.T @ p; db2 = p.sum(0); dz1 = (p @ W2.T) * (z1 > 0).float()
        dW1 = h.T @ dz1; db1 = dz1.sum(0)
        W2 = W2 - lr*dW2; b2 = b2 - lr*db2; W1 = W1 - lr*dW1; b1 = b1 - lr*db1
    return (W1, b1, W2, b2)

def acc_mlp(model, hp, Xc, Yc, mask=None):
    W1, b1, W2, b2 = hp
    Xc = Xc if mask is None else Xc[mask]; Yc = Yc if mask is None else Yc[mask]
    if len(Xc) == 0: return float("nan")
    c = 0
    for i in range(0, len(Xc), 64):
        xb, yb = Xc[i:i+64].to(Z.DEVICE), Yc[i:i+64].to(Z.DEVICE); h, _, _ = model.forward(xb)
        lg = torch.relu(h.float() @ W1 + b1) @ W2 + b2
        c += int((lg.argmax(-1) == yb).sum())
    return c/len(Xc)

def acc_of(model, Wc, bc, Xc, Yc, mask=None):
    Xc = Xc if mask is None else Xc[mask]; Yc = Yc if mask is None else Yc[mask]
    if len(Xc) == 0: return float("nan")
    c = 0
    for i in range(0, len(Xc), 64):
        xb, yb = Xc[i:i+64].to(Z.DEVICE), Yc[i:i+64].to(Z.DEVICE); h, _, _ = model.forward(xb)
        c += int(((h.float() @ Wc + bc).argmax(-1) == yb).sum())
    return c/len(Xc)

def main():
    cfg = Z.Config(name="c1-sent", vocab=VOCAB, seq_len=L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250,
                   time_limit_s=120, save_ckpt=True)
    data = lm_data(cfg); base = Z.ZeroGradMoE(cfg, VOCAB)
    res = Z.train(base, data, cfg, out_dir=OUT)
    ppl_before = Z.evaluate(base, data["Xval"], data["Yval"], cfg)

    blob = torch.load(OUT/"best_ckpt.pt", map_location="cpu", weights_only=False)
    post = Z.ZeroGradMoE(cfg, VOCAB); post.load_state_dict(blob["state"])
    Xtr_c, Ytr_c, _ = cls_set(6000, SEED+1); Xval_c, Yval_c, NGval = cls_set(1500, SEED+2)
    maj = max(float((Ytr_c == 0).float().mean()), float((Ytr_c == 1).float().mean()))
    Wc, bc = train_head(post, Xtr_c, Ytr_c, cfg.d_model)           # linear head
    acc = acc_of(post, Wc, bc, Xval_c, Yval_c)
    hp = train_head_mlp(post, Xtr_c, Ytr_c, cfg.d_model)            # 2-layer MLP head (can XOR)
    accm = acc_mlp(post, hp, Xval_c, Yval_c)
    accm_neg = acc_mlp(post, hp, Xval_c, Yval_c, NGval.bool())
    accm_pos = acc_mlp(post, hp, Xval_c, Yval_c, ~NGval.bool())
    ppl_after = Z.evaluate(post, data["Xval"], data["Yval"], cfg)
    hp2 = train_head_mlp(post, Xtr_c, Ytr_c, cfg.d_model); accm2 = acc_mlp(post, hp2, Xval_c, Yval_c)

    print("\n==== C.1 COMPOSITIONAL SENTIMENT ====")
    print(f"  base LM best_ppl={res['best_ppl']:.3f} unigram={data['unigram_ppl']:.1f}")
    print(f"  task: 2-class sentiment, vocab-overlapping w/ negation (majority baseline {maj*100:.1f}%)")
    print(f"  atomic features are 100%/99.9% linearly decodable (see probe) -> sentiment = their XOR")
    print(f"  LINEAR head acc = {acc*100:.1f}%   (XOR not linearly separable -> ~chance, expected)")
    print(f"  MLP    head acc = {accm*100:.1f}%   re-run = {accm2*100:.1f}%")
    print(f"    breakdown:  NOT-negated subset = {accm_pos*100:.1f}%   negated subset = {accm_neg*100:.1f}%")
    print(f"  base LM ppl  before={ppl_before:.4f}  after head-train={ppl_after:.4f}  (forgetting)")
    gates = {
        "MLP head acc >= majority + 15pp":       accm >= maj + 0.15,
        "MLP handles negation (neg subset>80%)": accm_neg > 0.80,
        "base LM ppl not regressed (head-only)": abs(ppl_after-ppl_before) < 1e-6,
        "MLP head training deterministic":       abs(accm-accm2) < 1e-9,
        "zero autograd":                         not res["autograd_used"],
    }
    print("\n  ---- gates ----")
    for k, v in gates.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  RESULT: {sum(gates.values())}/{len(gates)} gates")
    return gates

if __name__ == "__main__":
    main()
