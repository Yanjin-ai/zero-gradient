"""Phase H / v3.0 -- G2: multi-step reasoning depth. RESEARCH-ONLY, isolated (no ZeroBP import).

Honest framing: the ZeroBP contrast on "multi-step" was CLASSIFICATION of k-step modular arithmetic
(ZeroBP: chance at k=2, at every BP depth -- uninstallable). The defensible, apples-to-apples G2 is to
scale that difficulty: does the standard trainable-attn base + full BP INSTALL multi-step reasoning, and
how far does it scale in the number of steps? We sweep n_steps and report accuracy per depth.

  (Real natural-language GSM8K = G2b STRETCH: it is generative + needs a causal-LM variant + pretraining/
   scale; a tiny from-scratch classifier does NOT do it. Not scaffolded here as a win -- see charter §2.3.)

Writes runs/ph_gsm_run_summary.json (per-depth acc + config) for orchestrator monitoring.
Run locally (smoke):  python3 phase_h/ph_gsm_gpu.py --steps_list 2,3,4 --train_steps 1500
On Kaggle T4:         python3 phase_h/ph_gsm_gpu.py --steps_list 2,4,6,8 --train_steps 6000 --d_model 256 --layers 6
"""
import os, sys, json, random, argparse, time
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ph_base import PhConfig, PhTransformer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_vocab(digit_range):                                      # PAD=0, digits 1..digit_range, then +/- ops
    PLUS, MINUS = digit_range + 1, digit_range + 2
    return PLUS, MINUS, digit_range + 3


def gen(n, seed, n_steps, mod, digit_range):
    PLUS, MINUS, _ = make_vocab(digit_range)
    rng = random.Random(seed); X, Y = [], []
    seq_len = 2 * n_steps + 1                                     # (n_steps+1) digits + n_steps ops
    for _ in range(n):
        ds = [rng.randrange(digit_range) for _ in range(n_steps + 1)]
        ops = [rng.choice([PLUS, MINUS]) for _ in range(n_steps)]
        toks, v = [ds[0] + 1], ds[0]
        for i in range(n_steps):
            v = v + ds[i+1] if ops[i] == PLUS else v - ds[i+1]
            toks += [ops[i], ds[i+1] + 1]
        X.append(toks); Y.append(v % mod)
    return X, Y, seq_len


@torch.no_grad()
def evaluate(model, X, Y, bs=512):
    model.eval(); c = 0
    for i in range(0, len(X), bs):
        c += int(model(X[i:i+bs].to(DEVICE)).argmax(-1).cpu().eq(Y[i:i+bs]).sum())
    return c / max(1, len(X))


def train_depth(n_steps, a):
    _, _, V = make_vocab(a.digit_range)
    Xtr, Ytr, seq = gen(a.n_train, a.seed+2, n_steps, a.mod, a.digit_range)
    Xv, Yv, _ = gen(a.n_val, a.seed+3, n_steps, a.mod, a.digit_range)
    Xtr_d = torch.tensor(Xtr, dtype=torch.long).to(DEVICE); Ytr_d = torch.tensor(Ytr, dtype=torch.long).to(DEVICE)
    Xv_t = torch.tensor(Xv, dtype=torch.long); Yv_t = torch.tensor(Yv, dtype=torch.long)
    cfg = PhConfig(vocab=V, seq_len=seq, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, n_cls=a.mod, dropout=0.1)
    model = PhTransformer(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(a.seed)
    for step in range(1, a.train_steps+1):
        model.train(); ix = torch.randint(0, len(Xtr_d), (a.batch,), generator=g)
        loss = lossf(model(Xtr_d[ix]), Ytr_d[ix]); opt.zero_grad(); loss.backward(); opt.step()
    return evaluate(model, Xv_t, Yv_t), model.n_params()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps_list", default="2,3,4"); ap.add_argument("--mod", type=int, default=5)
    ap.add_argument("--digit_range", type=int, default=5)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--train_steps", type=int, default=1500); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=1500); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "ph_gsm_run_summary.json"))
    a = ap.parse_args(); torch.manual_seed(a.seed); t0 = time.time()
    depths = [int(s) for s in a.steps_list.split(",")]
    chance = 100.0 / a.mod
    print(f"\n==== Phase H G2: multi-step arithmetic depth sweep  ({a.layers}L x {a.heads}H d={a.d_model}, "
          f"mod {a.mod}, chance {chance:.1f}%, dev={DEVICE.type}) ====")
    print(f"  {'n_steps':>8} {'val_acc':>9} {'params(M)':>10}")
    rows = []
    for k in depths:
        acc, nparams = train_depth(k, a)
        rows.append({"n_steps": k, "val_acc": round(acc, 4), "params_M": round(nparams/1e6, 3)})
        print(f"  {k:>8} {acc*100:>8.1f}% {nparams/1e6:>10.2f}")
    summ = {"phase": "H", "gate": "G2", "task": "multi-step-arithmetic", "mod": a.mod,
            "config": {"layers": a.layers, "heads": a.heads, "d_model": a.d_model, "train_steps": a.train_steps},
            "chance": round(chance/100, 4), "by_depth": rows,
            "zerobp_contrast": "ZeroBP: chance at k=2, every BP depth (uninstallable)",
            "wall_s": round(time.time()-t0, 1), "device": DEVICE.type}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f: json.dump(summ, f, indent=2)
    best = max(rows, key=lambda r: r["n_steps"])
    print(f"\n  deepest k={best['n_steps']} -> {best['val_acc']*100:.1f}%  (ZeroBP: chance at k=2 already)")
    print(f"  summary -> {a.out}")
    return summ


if __name__ == "__main__":
    main()
