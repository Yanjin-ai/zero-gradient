"""Phase H / v3.0 -- G0 gate (harder contrast): the SAME standard trainable-attn base + FULL BP on the
2-step modular arithmetic the ZeroBP backbone could NOT install at ANY BP depth (every arm = chance 19-21%).

Synthetic distribution REPLICATED verbatim from `task_arith.py`. If Phase H cracks this, BOTH the relational
(NLI) and multi-step (arithmetic) dimensions were backbone-limited, not capability-limited. Self-contained
(no ZeroBP import). Run: python3 phase_h/ph_arith.py
"""
import os, sys, random
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ph_base import PhConfig, PhTransformer

SEED = 1234
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- 2-step modular arithmetic, IDENTICAL distribution to task_arith.py ----
PAD = 0; PLUS, MINUS = 6, 7; VOCAB = 8; L = 5; NCLS = 5
def tok(d): return d + 1


def compute(d1, o1, d2, o2, d3):
    v = d1 + d2 if o1 == PLUS else d1 - d2
    v = v + d3 if o2 == PLUS else v - d3
    return v % 5


def gen_cls(n, seed):
    rng = random.Random(seed); X, Y = [], []
    for _ in range(n):
        d1, d2, d3 = (rng.randrange(5) for _ in range(3)); o1, o2 = rng.choice([PLUS, MINUS]), rng.choice([PLUS, MINUS])
        X.append(torch.tensor([tok(d1), o1, tok(d2), o2, tok(d3)], dtype=torch.long)); Y.append(compute(d1, o1, d2, o2, d3))
    return torch.stack(X), torch.tensor(Y)


@torch.no_grad()
def evaluate(model, X, Y, bs=256):
    model.eval(); c = 0
    for i in range(0, len(X), bs):
        c += int(model(X[i:i+bs].to(DEVICE)).argmax(-1).cpu().eq(Y[i:i+bs]).sum())
    return c / len(X)


def main():
    torch.manual_seed(SEED)
    Xtr, Ytr = gen_cls(6000, SEED+2); Xv, Yv = gen_cls(1500, SEED+3)
    maj = max(float((Yv == c).float().mean()) for c in range(NCLS))
    cfg = PhConfig(vocab=VOCAB, seq_len=L, d_model=128, n_layers=4, n_heads=4, n_cls=NCLS, dropout=0.1)
    model = PhTransformer(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01); lossf = nn.CrossEntropyLoss()
    Xtr_d, Ytr_d = Xtr.to(DEVICE), Ytr.to(DEVICE); steps = 3000

    print(f"\n==== Phase H / v3.0 G0 (hard): standard trainable-attn + FULL BP on 2-step arithmetic ====")
    print(f"  base: {cfg.n_layers}L x {cfg.n_heads}H d={cfg.d_model}  params={model.n_params()/1e6:.2f}M  "
          f"majority {maj*100:.1f}%  chance {100/NCLS:.1f}%")
    g = torch.Generator().manual_seed(SEED)
    for step in range(1, steps+1):
        model.train()
        ix = torch.randint(0, len(Xtr_d), (64,), generator=g)
        loss = lossf(model(Xtr_d[ix]), Ytr_d[ix])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            print(f"  step {step:>4}  loss {loss.item():.4f}  val_acc {evaluate(model, Xv, Yv)*100:.1f}%")
    acc = evaluate(model, Xv, Yv)
    print(f"\n  Phase H arithmetic val_acc = {acc*100:.1f}%")
    print(f"  vs ZeroBP backbone (locked): EVERY BP depth = chance (19-21%) -- uninstallable")
    print(f"  [{'G0 PASS: new backbone INSTALLS multi-step -> backbone-limited, not capability-limited' if acc > 0.90 else 'G0 not cleared'}]")
    return acc


if __name__ == "__main__":
    main()
