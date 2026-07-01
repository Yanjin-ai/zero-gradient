"""Track 1 -- current ZeroBP 4B backbone on REAL SST-2 (binary sentiment). RESEARCH BRANCH (uses autograd
for Mixed-BP; never the submission path). Places the ZeroBP stack on a standard modern-LLM sentiment
benchmark, mirroring phasee_nli_4b.py but with the sentiment task (NCLS=2) and real SST-2 data.

DIAGNOSTIC version: the first run returned a bit-identical 49.08% across zero-shot / emb / emb+attn, which
smelled like a readout-collapse artifact rather than a true chance result. So for each variant this now
reports BOTH readouts side by side and the PREDICTION-CLASS DISTRIBUTION:
  - closed-form 2-layer MLP head (PN.closed_head), and
  - a standard BP linear probe (nn.Linear + AdamW on the frozen rep).
If BOTH heads collapse to ~one class -> the frozen ZeroBP rep genuinely doesn't separate real SST-2
sentiment (true chance). If the BP probe recovers signal the closed head missed -> it was a readout bug.

Run on Kaggle (T4 + internet for SST-2 + 4B checkpoint via kernel_sources). Env: ZG_E_STEPS, ZG_E_LR.
Writes runs/track1_sst2_run_summary.json.
"""
import os, json, random, torch
import torch.nn as nn
from dataclasses import fields
from pathlib import Path
import kaggle_zerograd_moe as Z
import phase_e_4b as P4
import phasee_nli_4b as PN

SEED = Z.SEED
STEPS = int(os.environ.get("ZG_E_STEPS", 1000)); LR = float(os.environ.get("ZG_E_LR", 0.1))
NCLS = 2


def gen_sst2(enc, seq_len, split, n, seed):                      # real GLUE/SST-2, encoded with the BPE encoder
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2")[split]
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx); idx = idx[:min(n, len(idx))]
    X, Y = [], []
    for i in idx:
        r = ds[i]
        if int(r["label"]) < 0: continue                         # test split has hidden (-1) labels; use train/validation
        ids = enc(r["sentence"].strip())[-seq_len:]; ids = [0]*(seq_len-len(ids)) + ids
        X.append(torch.tensor(ids, dtype=torch.long)); Y.append(int(r["label"]))
    return torch.stack(X), torch.tensor(Y)


@torch.no_grad()
def feats(base, X, bs=64):                                        # frozen last-position rep [N,d] (float)
    out = []
    for i in range(0, len(X), bs):
        h, _, _ = base.forward(X[i:i+bs].to(Z.DEVICE)); out.append(h.float())
    return torch.cat(out)


@torch.no_grad()
def closed_preds(base, hp, X, bs=64):                            # argmax preds of the closed-form 2-layer MLP head
    W1, b1, W2, b2 = hp; out = []
    for i in range(0, len(X), bs):
        h, _, _ = base.forward(X[i:i+bs].to(Z.DEVICE))
        out.append((torch.relu(h.float() @ W1 + b1) @ W2 + b2).argmax(-1).cpu())
    return torch.cat(out)


def bp_linear_probe(Htr, Ytr, Hv, ncls, steps=2000, lr=1e-2):    # standard BP linear head (AdamW) on frozen rep
    d = Htr.shape[1]; head = nn.Linear(d, ncls).to(Htr.device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4); lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED); Ytr_d = Ytr.to(Htr.device)
    with torch.enable_grad():                                    # submission Z disables grad globally -> re-enable for the probe
        for step in range(steps):
            ix = torch.randint(0, len(Htr), (128,), generator=g).to(Htr.device)
            loss = lossf(head(Htr[ix]), Ytr_d[ix]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return head(Hv).argmax(-1).cpu()


def dist(preds, ncls):
    return [int((preds == c).sum()) for c in range(ncls)]


def main():
    ck = P4.C.find_ckpt(); print(f"loading {ck}")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    fld = {f.name for f in fields(Z.Config)}; cfg = Z.Config(**{k: v for k, v in blob["cfg"].items() if k in fld})
    base = Z.ZeroGradMoE(cfg, cfg.vocab); base.load_state_dict(blob["state"]); del blob
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    baseA = base.state_dict()                                     # read-only golden checkpoint (cloned on load)
    d = cfg.d_model; data = Z.build_data(cfg); enc = data["_encode"]
    Xlm, Ylm = (data["Xtest"], data["Ytest"]) if "Xtest" in data else (data["Xval"], data["Yval"])
    Xtr, Ytr = gen_sst2(enc, cfg.seq_len, "train", 8000, SEED+1)
    Xv, Yv = gen_sst2(enc, cfg.seq_len, "validation", 1500, SEED+2)
    maj = max(float((Yv == c).float().mean()) for c in range(NCLS)); true_dist = dist(Yv, NCLS)
    ppl0 = Z.evaluate(base, Xlm, Ylm, cfg, batches=10**9)

    variants = []                                                # (name, bp_attn or None)
    variants = [("zeroshot", None), ("mixedbp_emb", False), ("mixedbp_emb_attn", True)]
    rows = {}
    for name, ba in variants:
        if ba is None: base.load_state_dict(baseA)
        else: PN.bp_adapt(base, baseA, Xtr, Ytr, cfg, STEPS, LR, ba, NCLS)
        # closed-form head: acc + prediction distribution
        ch = PN.closed_head(base, Xtr, Ytr, d, NCLS); cp = closed_preds(base, ch, Xv)
        c_acc = float((cp == Yv).float().mean()); c_dist = dist(cp, NCLS)
        # standard BP linear probe on the (adapted) frozen rep: acc + distribution
        Htr, Hv = feats(base, Xtr), feats(base, Xv)
        bp = bp_linear_probe(Htr, Ytr, Hv, NCLS); b_acc = float((bp == Yv).float().mean()); b_dist = dist(bp, NCLS)
        ppl = ppl0 if ba is None else Z.evaluate(base, Xlm, Ylm, cfg, batches=10**9)
        rows[name] = dict(closed_acc=round(c_acc, 4), closed_pred_dist=c_dist,
                          bp_linear_acc=round(b_acc, 4), bp_linear_pred_dist=b_dist, wiki_ppl=round(ppl, 3))

    summary = dict(checkpoint=str(ck), param_gigaparams=round(cfg.param_count()/1e9, 3), task="SST2-sentiment",
                   branch="Track1 ZeroBP 4B + Mixed-BP (research, DIAGNOSTIC)", dataset="glue/sst2",
                   majority_baseline=round(maj, 3), val_size=len(Xv), true_class_dist=true_dist,
                   bp_steps=STEPS, bp_lr=LR, wiki_ppl_zeroshot=round(ppl0, 3), variants=rows, uses_autograd=True)
    out = Path("/kaggle/working") if Z.ON_KAGGLE else Path(__file__).parent/"runs"; out.mkdir(parents=True, exist_ok=True)
    (out/"track1_sst2_run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n==== Track 1: ZeroBP 4B on real SST-2 (binary)  majority {maj*100:.1f}%  true_dist {true_dist} ====")
    print(f"  {'variant':20} {'closed_acc':>11} {'closed_dist':>14} {'bp_lin_acc':>11} {'bp_lin_dist':>14} {'wiki_ppl':>9}")
    for name, r in rows.items():
        print(f"  {name:20} {r['closed_acc']*100:>10.1f}% {str(r['closed_pred_dist']):>14} "
              f"{r['bp_linear_acc']*100:>10.1f}% {str(r['bp_linear_pred_dist']):>14} {r['wiki_ppl']:>9.1f}")
    return summary


if __name__ == "__main__":
    main()
