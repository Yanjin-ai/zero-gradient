"""C.1 on a saved checkpoint: head-only post-training (zero autograd) of an MLP sentiment head.

Loads a best_ckpt.pt (produced by a ZG_CKPT=1 run -> on Kaggle, the 4.160B BPE+MLP submission),
reloads the zero-BP MoE, builds a vocab-overlapping COMPOSITIONAL sentiment task (sentiment =
polarity XOR negation) in the model's own id space, attaches a 2-layer MLP head, trains ONLY the
head with closed-form CE (no autograd), and reports task acc + a linear-head contrast + LM-ppl
drift (head-only -> zero forgetting). Same code path runs on the small local checkpoint (plumbing
sanity) and the 4B Kaggle checkpoint.

Run locally:  python3 c1_4b.py                          (uses runs/best_ckpt.pt)
On Kaggle:    attach the checkpoint dataset; this finds /kaggle/input/**/best_ckpt.pt.
Env: ZG_CKPT_PATH=<path>, ZG_C1_STEPS=<int> (head-train steps, default 1000).
"""
import os, json, math, random, torch
from dataclasses import fields
from pathlib import Path
import kaggle_zerograd_moe as Z

SEED = Z.SEED
STEPS = int(os.environ.get("ZG_C1_STEPS", 1000))

def find_ckpt():
    p = os.environ.get("ZG_CKPT_PATH")
    if p and Path(p).exists(): return Path(p)
    cands = [Path(__file__).parent/"runs"/"best_ckpt.pt"]
    ki = Path("/kaggle/input")
    if ki.exists(): cands += sorted(ki.rglob("best_ckpt.pt"))
    for c in cands:
        if c.exists(): return c
    raise FileNotFoundError("no best_ckpt.pt found; set ZG_CKPT_PATH")

# --- compositional sentiment with REAL words encoded by the model's OWN tokenizer. "good"/"bad"/"not"
#     are common in WikiText so the pretrained rep has learned them; sentiment = polarity XOR negation so
#     it is NOT vocab-separable (every word appears in both classes). This is a zero-shot head-only
#     transfer test: the base LM is NOT fine-tuned, only the task head is trained. ---
SUBJ = ["man", "day", "film", "book", "city", "story", "road", "king", "river", "house"]
POS  = ["good", "great", "fine", "nice", "strong"]
NEG  = ["bad", "poor", "weak", "small", "wrong"]

def gen(encode, seq_len, n, seed):
    rng = random.Random(seed); X, Y, NG = [], [], []
    for _ in range(n):
        s = rng.choice(SUBJ); pos = rng.random() < 0.5
        w = rng.choice(POS if pos else NEG); neg = rng.random() < 0.5
        sent = f"the {s} is {'not ' if neg else ''}{w}"        # polarity word is last -> lands at last position
        ids = encode(sent)[-seq_len:]                          # encode with the model's tokenizer; cap to seq_len
        ids = [0]*(seq_len-len(ids)) + ids                     # left-pad
        X.append(torch.tensor(ids, dtype=torch.long)); Y.append(int(pos ^ (not neg))); NG.append(int(neg))
    return torch.stack(X), torch.tensor(Y), torch.tensor(NG)

def _batches(Xc, Yc, bs):
    for i in range(0, len(Xc), bs): yield Xc[i:i+bs].to(Z.DEVICE), Yc[i:i+bs].to(Z.DEVICE)

def train_head_linear(model, Xc, Yc, d, ncls=2, lr=0.2, bs=64):
    g = torch.Generator().manual_seed(SEED)
    Wc = (torch.randn(d, ncls, generator=g)/math.sqrt(d)).to(Z.DEVICE); bc = torch.zeros(ncls, device=Z.DEVICE)
    for step in range(STEPS):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xc), (bs,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, _, _ = model.forward(xb); lg = h.float() @ Wc + bc
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        Wc = Wc - lr*(h.float().T @ p); bc = bc - lr*p.sum(0)
    return ("lin", Wc, bc)

def train_head_mlp(model, Xc, Yc, d, ncls=2, lr=0.2, bs=64):
    g = torch.Generator().manual_seed(SEED)
    W1 = (torch.randn(d, d, generator=g)/math.sqrt(d)).to(Z.DEVICE); b1 = torch.zeros(d, device=Z.DEVICE)
    W2 = (torch.randn(d, ncls, generator=g)/math.sqrt(d)).to(Z.DEVICE); b2 = torch.zeros(ncls, device=Z.DEVICE)
    for step in range(STEPS):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xc), (bs,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, _, _ = model.forward(xb); h = h.float()
        z1 = h @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        dW2 = a1.T @ p; db2 = p.sum(0); dz1 = (p @ W2.T) * (z1 > 0).float(); dW1 = h.T @ dz1; db1 = dz1.sum(0)
        W2 = W2 - lr*dW2; b2 = b2 - lr*db2; W1 = W1 - lr*dW1; b1 = b1 - lr*db1
    return ("mlp", W1, b1, W2, b2)

def acc(model, head, Xc, Yc, mask=None, bs=64):
    if mask is not None: Xc, Yc = Xc[mask], Yc[mask]
    if len(Xc) == 0: return float("nan")
    c = 0
    for xb, yb in _batches(Xc, Yc, bs):
        h, _, _ = model.forward(xb); h = h.float()
        lg = (h @ head[1] + head[2]) if head[0] == "lin" else (torch.relu(h @ head[1] + head[2]) @ head[3] + head[4])
        c += int(lg.argmax(-1).eq(yb).sum())
    return c/len(Xc)

def main():
    ck = find_ckpt(); print(f"loading checkpoint: {ck}")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    fld = {f.name for f in fields(Z.Config)}; cfg = Z.Config(**{k: v for k, v in blob["cfg"].items() if k in fld})
    model = Z.ZeroGradMoE(cfg, cfg.vocab); model.load_state_dict(blob["state"])
    del blob                                                   # free the 8GB CPU copy after it's on the GPU
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  reloaded {cfg.name}: {cfg.param_count()/1e9:.3f}B params  vocab={cfg.vocab}  seq_len={cfg.seq_len}  head={cfg.head}")

    data = Z.build_data(cfg)                                   # rebuilds same tokenizer/windows -> valid LM eval + encoder
    Xlm, Ylm = (data["Xtest"], data["Ytest"]) if "Xtest" in data else (data["Xval"], data["Yval"])
    ppl_before = Z.evaluate(model, Xlm, Ylm, cfg, batches=10**9)

    enc = data["_encode"]
    Xtr, Ytr, _ = gen(enc, cfg.seq_len, 6000, SEED+1)
    Xv, Yv, NG = gen(enc, cfg.seq_len, 1500, SEED+2)
    maj = max(float((Ytr == 0).float().mean()), float((Ytr == 1).float().mean()))
    lin = train_head_linear(model, Xtr, Ytr, cfg.d_model); acc_lin = acc(model, lin, Xv, Yv)
    mlp = train_head_mlp(model, Xtr, Ytr, cfg.d_model)
    acc_mlp = acc(model, mlp, Xv, Yv); acc_neg = acc(model, mlp, Xv, Yv, NG.bool()); acc_pos = acc(model, mlp, Xv, Yv, ~NG.bool())
    ppl_after = Z.evaluate(model, Xlm, Ylm, cfg, batches=10**9)
    mlp2 = train_head_mlp(model, Xtr, Ytr, cfg.d_model); acc_mlp2 = acc(model, mlp2, Xv, Yv)

    summary = dict(checkpoint=str(ck), config=cfg.name, param_gigaparams=round(cfg.param_count()/1e9, 3),
                   corpus=data["corpus"], task="compositional-sentiment (head-only)", head="2-layer MLP (hand-derived)",
                   majority_baseline=round(maj, 3), linear_head_acc=round(acc_lin, 4),
                   mlp_head_acc=round(acc_mlp, 4), mlp_head_acc_rerun=round(acc_mlp2, 4),
                   acc_negated=round(acc_neg, 4), acc_not_negated=round(acc_pos, 4),
                   lm_ppl_before=round(ppl_before, 3), lm_ppl_after=round(ppl_after, 3),
                   zero_forgetting=bool(abs(ppl_after-ppl_before) < 1e-6),
                   deterministic=bool(abs(acc_mlp-acc_mlp2) < 1e-9), zero_autograd=not torch.is_grad_enabled(),
                   head_train_steps=STEPS)
    out = Path("/kaggle/working") if Z.ON_KAGGLE else Path(__file__).parent/"runs"; out.mkdir(parents=True, exist_ok=True)
    (out/"c1_run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n==== C.1 POST-TRAIN (head-only MLP sentiment on reloaded checkpoint) ====")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\n  wrote {out/'c1_run_summary.json'}")
    return summary

if __name__ == "__main__":
    main()
