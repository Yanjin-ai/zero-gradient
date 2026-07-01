"""Track 1 -- current ZeroBP 4B backbone on REAL SST-2 (binary sentiment). RESEARCH BRANCH (uses autograd
for Mixed-BP; never the submission path). Places the ZeroBP stack on a standard modern-LLM sentiment
benchmark, mirroring phasee_nli_4b.py exactly but with the sentiment task (NCLS=2) and real SST-2 data.

Reloads the 4.16B checkpoint, then reports zero-shot / Mixed-BP(emb) / Mixed-BP(emb+attn) with a FAIR
closed-form readout, plus WikiText test-ppl drift. Goal (Track 1): show the ZeroBP backbone + a little
embedding BP reaches a reasonable modern-small-LLM sentiment level (contrast: sentiment is the "bag" task
ZeroBP CAN adapt, unlike the relational/multi-step tasks -> Phase H).

Run on Kaggle (T4 + internet for SST-2 + 4B checkpoint via kernel_sources). Env: ZG_E_STEPS, ZG_E_LR.
Reuses phasee_nli_4b helpers (closed_head / head_acc / bp_adapt). Writes runs/track1_sst2_run_summary.json.
"""
import os, json, random, torch
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
    maj = max(float((Yv == c).float().mean()) for c in range(NCLS))
    ppl0 = Z.evaluate(base, Xlm, Ylm, cfg, batches=10**9)
    acc_zs = PN.head_acc(base, PN.closed_head(base, Xtr, Ytr, d, NCLS), Xv, Yv)
    rows = {"zeroshot": (acc_zs, ppl0)}
    for name, ba in [("mixedbp_emb", False), ("mixedbp_emb_attn", True)]:
        PN.bp_adapt(base, baseA, Xtr, Ytr, cfg, STEPS, LR, ba, NCLS)
        rows[name] = (PN.head_acc(base, PN.closed_head(base, Xtr, Ytr, d, NCLS), Xv, Yv),
                      Z.evaluate(base, Xlm, Ylm, cfg, batches=10**9))

    summary = dict(checkpoint=str(ck), param_gigaparams=round(cfg.param_count()/1e9, 3), task="SST2-sentiment",
                   branch="Track1 ZeroBP 4B + Mixed-BP (research)", dataset="glue/sst2", majority_baseline=round(maj, 3),
                   bp_steps=STEPS, bp_lr=LR, val_size=len(Xv), zeroshot_acc=round(acc_zs, 4),
                   mixedbp_emb_acc=round(rows["mixedbp_emb"][0], 4), mixedbp_emb_attn_acc=round(rows["mixedbp_emb_attn"][0], 4),
                   wiki_ppl_zeroshot=round(ppl0, 3), wiki_ppl_emb=round(rows["mixedbp_emb"][1], 3),
                   forget_emb=round(rows["mixedbp_emb"][1]-ppl0, 3), uses_autograd=True)
    out = Path("/kaggle/working") if Z.ON_KAGGLE else Path(__file__).parent/"runs"; out.mkdir(parents=True, exist_ok=True)
    (out/"track1_sst2_run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n==== Track 1: ZeroBP 4B on real SST-2 (binary) ====")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\n  zero-shot {acc_zs*100:.1f}% -> emb {rows['mixedbp_emb'][0]*100:.1f}% -> emb+attn {rows['mixedbp_emb_attn'][0]*100:.1f}%  (majority {maj*100:.1f}%)")
    return summary


if __name__ == "__main__":
    main()
