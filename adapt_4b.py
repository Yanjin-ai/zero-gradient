"""v1.1 stage 2.2 — 4B in-domain adaptation + C.1 (zero autograd), with forgetting mitigation.

On the reloaded 4B checkpoint (WikiText, ppl ~1355):
  0. baseline: WikiText test ppl + zero-shot head-only MLP sentiment acc (= the ~60% result).
  1. ADAPT: short in-domain zero-BP LM adaptation on the sentiment-template domain, with mitigation:
       small lr (ZG_ADAPT_LR), few steps near the local knee (ZG_ADAPT_STEPS), routing frozen,
       and WikiText REPLAY mixed in (ZG_ADAPT_REPLAY fraction) to anchor the original domain.
  2. re-measure: WikiText test ppl (forgetting) + head-only MLP sentiment acc (gain).

Writes adapt_run_summary.json comparing 4B-ZeroShot vs 4B-Adapt. Same code path runs small + 4B.
Run:  python3 adapt_4b.py    Env: ZG_ADAPT_STEPS=150 ZG_ADAPT_LR=0.005 ZG_ADAPT_REPLAY=0.5 ZG_C1_STEPS=1000
"""
import os, json, math, random, torch
from dataclasses import fields, replace
from pathlib import Path
import kaggle_zerograd_moe as Z
import c1_4b as C

SEED = Z.SEED
ADAPT_STEPS = int(os.environ.get("ZG_ADAPT_STEPS", 150))
ADAPT_LR = float(os.environ.get("ZG_ADAPT_LR", 0.005))
REPLAY = float(os.environ.get("ZG_ADAPT_REPLAY", 0.5))         # fraction of adaptation tokens drawn from WikiText
FREEZE_HEADS = os.environ.get("ZG_ADAPT_FREEZE_HEADS", "0") == "1"  # 2.1 mitigation: protect the LM head
BB_SCALE = float(os.environ.get("ZG_ADAPT_BB_SCALE", "1.0"))   # 2.1 mitigation: importance-weight the backbone
HEAD_STEPS = int(os.environ.get("ZG_C1_STEPS", 1000))

def sent_stream(enc, n, seed):                                 # unlabeled sentiment sentences -> LM token stream
    rng = random.Random(seed); out = []
    for _ in range(n):
        s = rng.choice(C.SUBJ); pos = rng.random() < 0.5; w = rng.choice(C.POS if pos else C.NEG)
        out += enc(f"the {s} is {'not ' if rng.random() < 0.5 else ''}{w}")
    return out

def windows(ids, seq_len, cap=200000):
    ids = torch.tensor(ids, dtype=torch.long); Xs, Ys = [], []
    for i in range(0, min(len(ids)-seq_len-1, cap)): Xs.append(ids[i:i+seq_len]); Ys.append(ids[i+seq_len])
    return torch.stack(Xs), torch.stack(Ys)

def head_acc(model, Xtr, Ytr, Xv, Yv, d):
    hp = C.train_head_mlp(model, Xtr, Ytr, d); return C.acc(model, hp, Xv, Yv)

def main():
    ck = C.find_ckpt(); print(f"loading {ck}")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    fld = {f.name for f in fields(Z.Config)}; cfg = Z.Config(**{k: v for k, v in blob["cfg"].items() if k in fld})
    model = Z.ZeroGradMoE(cfg, cfg.vocab); model.load_state_dict(blob["state"]); del blob
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  reloaded {cfg.name}: {cfg.param_count()/1e9:.3f}B  vocab={cfg.vocab}  seq_len={cfg.seq_len}")

    data = Z.build_data(cfg); enc = data["_encode"]
    Xlm, Ylm = (data["Xtest"], data["Ytest"]) if "Xtest" in data else (data["Xval"], data["Yval"])
    Xtr_c, Ytr_c, _ = C.gen(enc, cfg.seq_len, 6000, SEED+1); Xv_c, Yv_c, NG = C.gen(enc, cfg.seq_len, 1500, SEED+2)
    maj = max(float((Ytr_c == 0).float().mean()), float((Ytr_c == 1).float().mean()))

    ppl_before = Z.evaluate(model, Xlm, Ylm, cfg, batches=10**9)
    acc_zeroshot = head_acc(model, Xtr_c, Ytr_c, Xv_c, Yv_c, cfg.d_model)
    print(f"  [zero-shot] wiki_ppl={ppl_before:.1f}  sentiment_acc={acc_zeroshot*100:.1f}% (majority {maj*100:.1f}%)")

    # --- in-domain adaptation corpus: sentiment LM stream + WikiText replay (mitigation) ---
    Xs, Ys = windows(sent_stream(enc, 8000, SEED), cfg.seq_len)
    Xs, Ys = Xs.to(Z.DEVICE), Ys.to(Z.DEVICE)
    if REPLAY > 0 and REPLAY < 1:
        nrep = int(len(Xs)*REPLAY/(1-REPLAY)); idx = torch.randperm(len(data["Xtr"]))[:nrep]
        Xs = torch.cat([Xs, data["Xtr"][idx]]); Ys = torch.cat([Ys, data["Ytr"][idx]])
    nval = max(64, len(Xs)//20)
    adapt = dict(corpus="sent-adapt", Xtr=Xs[nval:], Ytr=Ys[nval:], Y2tr=Ys[nval:], Xval=Xs[:nval], Yval=Ys[:nval])
    acfg = replace(cfg, steps=ADAPT_STEPS, lr=ADAPT_LR, lr_min=ADAPT_LR, warmup_steps=min(10, max(1, ADAPT_STEPS//4)),
                   eval_every=max(1, ADAPT_STEPS), patience=10**9, freeze_routing_step=0, save_ckpt=False, time_limit_s=3600,
                   freeze_heads=FREEZE_HEADS, backbone_lr_scale=BB_SCALE)
    print(f"  [adapt] {ADAPT_STEPS} steps, lr={ADAPT_LR}, replay={REPLAY}, routing frozen, freeze_heads={FREEZE_HEADS}, bb_scale={BB_SCALE}")
    Z.train(model, adapt, acfg)                                # in-domain zero-BP LM adaptation (in place)

    ppl_after = Z.evaluate(model, Xlm, Ylm, cfg, batches=10**9)
    acc_adapted = head_acc(model, Xtr_c, Ytr_c, Xv_c, Yv_c, cfg.d_model)
    acc_neg = C.acc(model, C.train_head_mlp(model, Xtr_c, Ytr_c, cfg.d_model), Xv_c, Yv_c, NG.bool())

    summary = dict(checkpoint=str(ck), config=cfg.name, param_gigaparams=round(cfg.param_count()/1e9, 3),
                   corpus=data["corpus"], task="compositional-sentiment", majority_baseline=round(maj, 3),
                   adapt_steps=ADAPT_STEPS, adapt_lr=ADAPT_LR, replay_frac=REPLAY, freeze_heads=FREEZE_HEADS,
                   backbone_lr_scale=BB_SCALE, head_train_steps=HEAD_STEPS,
                   zeroshot_acc=round(acc_zeroshot, 4), adapted_acc=round(acc_adapted, 4), adapted_acc_negated=round(acc_neg, 4),
                   wiki_ppl_before=round(ppl_before, 3), wiki_ppl_after=round(ppl_after, 3),
                   forgetting_dppl=round(ppl_after-ppl_before, 3),
                   acc_gain=round(acc_adapted-acc_zeroshot, 4), zero_autograd=not torch.is_grad_enabled())
    out = Path("/kaggle/working") if Z.ON_KAGGLE else Path(__file__).parent/"runs"; out.mkdir(parents=True, exist_ok=True)
    (out/"adapt_run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n==== 4B-ADAPT (in-domain adaptation + C.1, with mitigation) ====")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\n  zero-shot {acc_zeroshot*100:.1f}% -> adapted {acc_adapted*100:.1f}%  "
          f"(acc +{(acc_adapted-acc_zeroshot)*100:.1f}pp, forgetting dPPL {ppl_after-ppl_before:+.1f})")
    return summary

if __name__ == "__main__":
    main()
