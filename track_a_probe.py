"""Track A probe: does trainable attention (A1) help on a task that REQUIRES attention?

The main small config (synthetic topic-prefix sentences, seq_len 16) is near bag-of-words and
barely exercises attention -> it can't tell us if trainable attention is a real lever (val_ppl
6.251 frozen vs 6.245 trainable = noise). This probe builds an ASSOCIATIVE-RECALL task that frozen
random attention provably cannot solve: the answer depends on content-based lookup of an earlier
position. If A1 clearly beats frozen here, that's evidence the lever is real (and the main probe
just under-tests it, like D-1's small-config understatement). Zero autograd throughout (reuses
ZeroGradMoE + its local rules unchanged).

Task: seq = [k1 v1 k2 v2 ... km vm  q]  where q equals one of the keys ki; target = vi.
  keys and values are disjoint vocab ranges -> the model must (a) find the matching key by content
  and (b) read out the value next to it. Frozen random Wq/Wk -> chance-level; trainable -> can align.

Run:  python3 track_a_probe.py
"""
import os, math, random, torch
import kaggle_zerograd_moe as Z

SEED = Z.SEED
NKEY, NVAL, NPAIR = 40, 40, 4          # key vocab, value vocab, pairs per sequence
VOCAB = 1 + NKEY + NVAL                # 0=pad, [1..NKEY]=keys, [NKEY+1..]=values
SEQ = NPAIR*2 + 1                      # k v k v ... q

def make_data(n, seed):
    rng = random.Random(seed); Xs, Ys = [], []
    for _ in range(n):
        keys = rng.sample(range(1, NKEY+1), NPAIR)
        vals = [rng.randrange(NKEY+1, VOCAB) for _ in range(NPAIR)]
        seq = []
        for kk, vv in zip(keys, vals): seq += [kk, vv]
        qi = rng.randrange(NPAIR); seq.append(keys[qi])     # query = one of the keys
        Xs.append(torch.tensor(seq, dtype=torch.long)); Ys.append(vals[qi])
    return torch.stack(Xs), torch.tensor(Ys, dtype=torch.long)

def build(cfg):
    Xtr, Ytr = make_data(6000, SEED); Xval, Yval = make_data(1000, SEED+1)
    # majority-class / unigram baseline ppl on val
    import collections
    cnt = collections.Counter(Ytr.tolist()); tot = sum(cnt.values())
    p = {k: v/tot for k, v in cnt.items()}; nll = -sum(math.log(p.get(int(y), 1e-9)) for y in Yval)/len(Yval)
    return dict(corpus="assoc-recall", vocab=list(range(VOCAB)),
                Xtr=Xtr.to(Z.DEVICE), Ytr=Ytr.to(Z.DEVICE), Y2tr=Ytr.to(Z.DEVICE),
                Xval=Xval.to(Z.DEVICE), Yval=Yval.to(Z.DEVICE),
                unigram_ppl=math.exp(nll))

def acc(model, X, Y, cfg, n=1000):
    c = 0
    for i in range(0, min(len(X), n), cfg.batch_size):
        xb, yb = X[i:i+cfg.batch_size], Y[i:i+cfg.batch_size]
        h, _, _ = model.forward(xb); pred = model.logits(h).argmax(-1)
        c += int((pred == yb).sum())
    return c/min(len(X), n)

def run(attn_train, attn_lr_scale=0.3):
    cfg = Z.Config(name=("trainA" if attn_train else "frozen"), vocab=VOCAB, seq_len=SEQ,
                   d_model=128, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1200, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100,
                   eval_every=400, time_limit_s=120, attn_train=attn_train, attn_lr_scale=attn_lr_scale)
    data = build(cfg); model = Z.ZeroGradMoE(cfg, VOCAB)
    res = Z.train(model, data, cfg)
    a = acc(model, data["Xval"], data["Yval"], cfg)
    return res, a, data["unigram_ppl"]

if __name__ == "__main__":
    print(f"task=assoc-recall vocab={VOCAB} seq_len={SEQ} pairs={NPAIR}")
    import sys
    res, a, uni = run(False)
    print(f"  [FROZEN-ATTN        ] val_ppl {res['best_ppl']:.3f} (unigram {uni:.1f})  val_acc {a*100:.1f}%")
    for scale in (1.0, 10.0, 100.0, 1000.0):
        res, a, uni = run(True, scale)
        amax = max((c.get('attn_dw') or 0) for c in res['curve'])
        print(f"  [TRAIN-ATTN lr*{scale:<6g}] val_ppl {res['best_ppl']:.3f}  val_acc {a*100:.1f}%  "
              f"max_attn_dw={amax:.4g}  autograd={res['autograd_used']}")
