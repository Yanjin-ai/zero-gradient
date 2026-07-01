"""Phase H / v3.0 -- G1: real NLI (SNLI/MNLI) on GPU. RESEARCH-ONLY, isolated (no ZeroBP import).

Trains the standard multi-layer trainable-attention base (ph_base.PhTransformer) with FULL backprop on
real SNLI/MNLI and reports accuracy -- the modern-LLM NLI dimension where the ZeroBP 4B backbone is locked
at chance. Goal (charter G1): reach a respectable mid-tier acc (~70-85%) that ZeroBP structurally cannot,
NOT to match GPT-4.

Data sources (auto / --source):
  synthetic : the SAME synthetic NLI as ph_nli.py (int tokens) -- CPU smoke test of the full loop.
  hf        : HuggingFace `datasets` (snli / glue-mnli). Needs internet or a cached dataset.
  jsonl     : a mounted Kaggle dataset dir of {premise,hypothesis,label} jsonl (label 0/1/2; -1 dropped).

Writes runs/ph_nli_run_summary.json (metrics + config + ZeroBP contrast) for orchestrator monitoring.
Run locally (smoke):  python3 phase_h/ph_nli_gpu.py --source synthetic --steps 800
On Kaggle T4:         python3 phase_h/ph_nli_gpu.py --source hf --dataset snli --epochs 3
"""
import os, sys, json, math, random, argparse, time
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ph_base import PhConfig, PhTransformer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PAD, UNK, SEP = 0, 1, 2                                            # reserved ids for the word-level tokenizer
LABELS = {"entailment": 0, "neutral": 1, "contradiction": 2}


# ---------- data: synthetic (smoke) ----------
def synthetic(n, seed):                                           # identical distribution to task_nli.py / ph_nli.py
    THAN, GT, LT = 1, 3, 4; ENT = list(range(5, 13)); INV = {GT: LT, LT: GT}
    rng = random.Random(seed); X, Y = [], []
    for _ in range(n):
        e1, e2 = rng.sample(ENT, 2); r1 = rng.choice([GT, LT]); lab = rng.randrange(3)
        if lab == 0: e3, e4, r2 = e2, e1, INV[r1]
        elif lab == 1: e3, e4, r2 = e2, e1, r1
        else:
            while True:
                e3, e4 = rng.sample(ENT, 2)
                if {e3, e4} != {e1, e2}: break
            r2 = rng.choice([GT, LT])
        X.append([e1, r1, 2, e2, 2, e3, r2, THAN, e4]); Y.append(lab)
    return X, Y, 13, 9                                            # X(list[int]), Y, vocab, seq_len


# ---------- data: real text (hf / jsonl) -> word-level tokenized ----------
def load_pairs(source, dataset, data_dir, split):
    """Return list of (premise:str, hypothesis:str, label:int in 0/1/2); drops label==-1 (no gold)."""
    pairs = []
    if source == "hf":
        from datasets import load_dataset
        ds = load_dataset("glue", "mnli")[split] if dataset == "mnli" else load_dataset("snli")[split]
        for r in ds:
            if r["label"] in (0, 1, 2): pairs.append((r["premise"], r["hypothesis"], int(r["label"])))
    elif source == "jsonl":
        path = os.path.join(data_dir, f"{split}.jsonl")
        with open(path) as f:
            for line in f:
                r = json.loads(line); lab = r.get("label")
                lab = LABELS.get(lab, lab) if isinstance(lab, str) else lab
                if lab in (0, 1, 2): pairs.append((r["premise"], r["hypothesis"], int(lab)))
    return pairs


def build_vocab(pairs, max_vocab=30000):
    from collections import Counter
    c = Counter(w for p, h, _ in pairs for w in (p + " " + h).lower().split())
    vocab = {w: i + 3 for i, (w, _) in enumerate(c.most_common(max_vocab - 3))}   # 0/1/2 reserved
    return vocab


def encode(pairs, vocab, max_len):
    X, Y = [], []
    for p, h, lab in pairs:
        pt = [vocab.get(w, UNK) for w in p.lower().split()]
        ht = [vocab.get(w, UNK) for w in h.lower().split()]
        ids = (pt + [SEP] + ht)[:max_len]; ids += [PAD] * (max_len - len(ids))
        X.append(ids); Y.append(lab)
    return X, Y


def to_tensors(X, Y):
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


@torch.no_grad()
def evaluate(model, X, Y, bs=512):
    model.eval(); c = 0
    for i in range(0, len(X), bs):
        c += int(model(X[i:i+bs].to(DEVICE)).argmax(-1).cpu().eq(Y[i:i+bs]).sum())
    return c / max(1, len(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "hf", "jsonl"], default="synthetic")
    ap.add_argument("--dataset", choices=["snli", "mnli"], default="snli")
    ap.add_argument("--data_dir", default="/kaggle/input/nli")
    ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "ph_nli_run_summary.json"))
    a = ap.parse_args(); torch.manual_seed(a.seed); t0 = time.time()

    if a.source == "synthetic":
        Xtr, Ytr, V, seq = synthetic(6000, a.seed+2); Xv, Yv, _, _ = synthetic(1500, a.seed+3)
        a.layers, a.heads, a.d_model, a.max_len = 4, 4, 128, seq            # small config for CPU smoke
    else:
        tr = load_pairs(a.source, a.dataset, a.data_dir, "train")
        val_split = "validation_matched" if (a.source == "hf" and a.dataset == "mnli") else ("validation" if a.source == "hf" else "dev")
        va = load_pairs(a.source, a.dataset, a.data_dir, val_split)
        vocab = build_vocab(tr); V = len(vocab) + 3; seq = a.max_len
        Xtr, Ytr = encode(tr, vocab, a.max_len); Xv, Yv = encode(va, vocab, a.max_len)
    Xtr, Ytr = to_tensors(Xtr, Ytr); Xv, Yv = to_tensors(Xv, Yv)
    Xtr_d, Ytr_d = Xtr.to(DEVICE), Ytr.to(DEVICE)
    maj = max(float((Yv == c).float().mean()) for c in range(3))

    cfg = PhConfig(vocab=V, seq_len=seq, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, n_cls=3, dropout=0.1)
    model = PhTransformer(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); lossf = nn.CrossEntropyLoss()
    steps = a.steps or (a.epochs * max(1, len(Xtr) // a.batch))
    print(f"\n==== Phase H G1: {a.source}/{a.dataset if a.source!='synthetic' else '-'}  "
          f"{cfg.n_layers}L x {cfg.n_heads}H d={cfg.d_model}  params={model.n_params()/1e6:.2f}M  "
          f"train={len(Xtr)} val={len(Xv)} vocab={V} maj={maj*100:.1f}% steps={steps} dev={DEVICE.type} ====")
    g = torch.Generator().manual_seed(a.seed)
    for step in range(1, steps+1):
        model.train(); ix = torch.randint(0, len(Xtr_d), (a.batch,), generator=g)
        loss = lossf(model(Xtr_d[ix]), Ytr_d[ix]); opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps//10) == 0 or step == 1:
            print(f"  step {step:>5}/{steps}  loss {loss.item():.4f}  val_acc {evaluate(model, Xv, Yv)*100:.2f}%")
    acc = evaluate(model, Xv, Yv)
    summ = {"phase": "H", "gate": "G1", "task": "NLI", "source": a.source,
            "dataset": (a.dataset if a.source != "synthetic" else "synthetic"),
            "params_M": round(model.n_params()/1e6, 3), "config": {"layers": a.layers, "heads": a.heads, "d_model": a.d_model, "max_len": a.max_len},
            "train_size": len(Xtr), "val_size": len(Xv), "steps": steps, "val_acc": round(acc, 4),
            "majority": round(maj, 4), "zerobp_contrast": "ZeroBP 4B NLI = chance (33.4%)",
            "wall_s": round(time.time()-t0, 1), "device": DEVICE.type}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f: json.dump(summ, f, indent=2)
    print(f"\n  Phase H G1 val_acc = {acc*100:.2f}%   (ZeroBP 4B NLI = chance 33.4%)")
    print(f"  summary -> {a.out}")
    return summ


if __name__ == "__main__":
    main()
