"""Phase H / v3.0 -- G0 gate: does a STANDARD multi-layer trainable-attention base + FULL backprop crack the
SAME synthetic NLI that the ZeroBP backbone capped at (deep-BP 65.7% small / chance at 4B)?

Apples-to-apples: the synthetic NLI distribution below is REPLICATED verbatim from `task_nli.py` (entities,
GT/LT relations, entailment/contradiction/neutral logic) so the only thing that changes vs the locked ZeroBP
matrix is the BACKBONE. If this lifts to ~95-100%, the bottleneck was the architecture, not the task -> ADR-005
direction confirmed. Self-contained (no ZeroBP import). Run: python3 phase_h/ph_nli.py
"""
import os, sys, math, random
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ph_base import PhConfig, PhTransformer

SEED = 1234
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- synthetic NLI, IDENTICAL distribution to task_nli.py (apples-to-apples vs the ZeroBP matrix) ----
PAD, THAN, SEP, GT, LT = 0, 1, 2, 3, 4
ENT = list(range(5, 13)); VOCAB = 13; L = 9; NCLS = 3
INV = {GT: LT, LT: GT}


def gen_cls(n, seed):
    rng = random.Random(seed); X, Y = [], []
    for _ in range(n):
        e1, e2 = rng.sample(ENT, 2); r1 = rng.choice([GT, LT]); lab = rng.randrange(3)
        if lab == 0: e3, e4, r2 = e2, e1, INV[r1]                 # entailment
        elif lab == 1: e3, e4, r2 = e2, e1, r1                    # contradiction
        else:
            while True:
                e3, e4 = rng.sample(ENT, 2)
                if {e3, e4} != {e1, e2}: break                    # neutral: different entity pair
            r2 = rng.choice([GT, LT])
        X.append(torch.tensor([e1, r1, THAN, e2, SEP, e3, r2, THAN, e4], dtype=torch.long)); Y.append(lab)
    return torch.stack(X), torch.tensor(Y)


@torch.no_grad()
def evaluate(model, X, Y, bs=256):
    model.eval(); c = 0
    for i in range(0, len(X), bs):
        lg = model(X[i:i+bs].to(DEVICE))
        c += int(lg.argmax(-1).cpu().eq(Y[i:i+bs]).sum())
    return c / len(X)


def main():
    torch.manual_seed(SEED)
    Xtr, Ytr = gen_cls(6000, SEED+2); Xv, Yv = gen_cls(1500, SEED+3)
    maj = max(float((Yv == c).float().mean()) for c in range(NCLS))
    cfg = PhConfig(vocab=VOCAB, seq_len=L, d_model=128, n_layers=4, n_heads=4, n_cls=NCLS, dropout=0.1)
    model = PhTransformer(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    Xtr_d, Ytr_d = Xtr.to(DEVICE), Ytr.to(DEVICE); steps = 3000

    print(f"\n==== Phase H / v3.0 G0: standard multi-layer trainable-attn + FULL BP on synthetic NLI ====")
    print(f"  base: {cfg.n_layers}L x {cfg.n_heads}H d={cfg.d_model}  params={model.n_params()/1e6:.2f}M  "
          f"majority {maj*100:.1f}%  chance {100/NCLS:.1f}%")
    g = torch.Generator().manual_seed(SEED)
    for step in range(1, steps+1):
        model.train()
        ix = torch.randint(0, len(Xtr_d), (64,), generator=g)
        lg = model(Xtr_d[ix]); loss = lossf(lg, Ytr_d[ix])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            print(f"  step {step:>4}  loss {loss.item():.4f}  val_acc {evaluate(model, Xv, Yv)*100:.1f}%")
    acc = evaluate(model, Xv, Yv)
    print(f"\n  Phase H NLI val_acc = {acc*100:.1f}%")
    print(f"  vs ZeroBP backbone (locked): deep-BP 65.7% (small) / 4B = chance(33.4%)")
    print(f"  [{'G0 PASS: new backbone CRACKS synthetic NLI -> bottleneck was the architecture (ADR-005 confirmed)' if acc > 0.90 else 'G0 not cleared'}]")
    return acc


if __name__ == "__main__":
    main()
