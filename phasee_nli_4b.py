"""Phase E task #2 at 4B -- NLI (3-class relational entailment), hybrid BP (RESEARCH BRANCH).

On the reloaded 4.16B checkpoint, real-word NLI: "the {s1} is {r1} than the {s2} . the {s3} is {r2}
than the {s4}" -> entail / contra / neutral (needs cross-clause comparison of BOTH entity pairs and
relations). Tests at 4B/WikiText what the small config showed: Mixed-BP(embedding) helps, and the
relational task additionally benefits from Mixed-BP(embedding+ATTENTION) -- because alignment lives in
the frozen attention that embedding-BP can't reach.

Reports zero-shot / Mixed-BP(emb) / Mixed-BP(emb+attn), each with a FAIR closed-form readout on the
adapted rep, plus WikiText test ppl drift. 4B single-shot, no shared state; baseA is read-only (cloned
on load). Run: python3 phasee_nli_4b.py   Env: ZG_E_STEPS (default 1000), ZG_E_LR (default 0.1).
"""
import os, json, math, random, torch
from dataclasses import fields
from pathlib import Path
import kaggle_zerograd_moe as Z
import phase_e_4b as P4

SEED = Z.SEED
STEPS = int(os.environ.get("ZG_E_STEPS", 1000)); LR = float(os.environ.get("ZG_E_LR", 0.1))
SUBJ = ["man", "dog", "car", "book", "city", "king", "river", "house"]
REL = {"bigger": "smaller", "smaller": "bigger"}; RELS = list(REL); NCLS = 3

def gen_nli(enc, seq_len, n, seed):
    rng = random.Random(seed); X, Y = [], []
    for _ in range(n):
        s1, s2 = rng.sample(SUBJ, 2); r1 = rng.choice(RELS); lab = rng.randrange(3)
        if lab == 0: s3, s4, r2 = s2, s1, REL[r1]              # entailment
        elif lab == 1: s3, s4, r2 = s2, s1, r1                 # contradiction
        else:
            while True:
                s3, s4 = rng.sample(SUBJ, 2)
                if {s3, s4} != {s1, s2}: break                 # neutral (different pair)
            r2 = rng.choice(RELS)
        text = f"the {s1} is {r1} than the {s2} . the {s3} is {r2} than the {s4}"
        ids = enc(text)[-seq_len:]; ids = [0]*(seq_len-len(ids)) + ids
        X.append(torch.tensor(ids, dtype=torch.long)); Y.append(lab)
    return torch.stack(X), torch.tensor(Y)

def closed_head(base, Xc, Yc, d, ncls, steps=1000, lr=0.2):   # closed-form n-class MLP head on frozen base (zero autograd)
    g = torch.Generator().manual_seed(SEED)
    W1 = (torch.randn(d, d, generator=g)/math.sqrt(d)).to(Z.DEVICE); b1 = torch.zeros(d, device=Z.DEVICE)
    W2 = (torch.randn(d, ncls, generator=g)/math.sqrt(d)).to(Z.DEVICE); b2 = torch.zeros(ncls, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xc), (64,), generator=gi)
        xb, yb = Xc[ix].to(Z.DEVICE), Yc[ix].to(Z.DEVICE); B = xb.shape[0]
        h, _, _ = base.forward(xb); h = h.float(); z1 = h @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        W2 = W2 - lr*(a1.T @ p); b2 = b2 - lr*p.sum(0); dz1 = (p @ W2.T)*(z1 > 0).float()
        W1 = W1 - lr*(h.T @ dz1); b1 = b1 - lr*dz1.sum(0)
    return (W1, b1, W2, b2)

def head_acc(base, hp, Xc, Yc):
    W1, b1, W2, b2 = hp; c = 0
    for i in range(0, len(Xc), 64):
        xb = Xc[i:i+64].to(Z.DEVICE); h, _, _ = base.forward(xb)
        c += int((torch.relu(h.float() @ W1 + b1) @ W2 + b2).argmax(-1).cpu().eq(Yc[i:i+64]).sum())
    return c/len(Xc)

def bp_adapt(base, baseA, Xtr, Ytr, cfg, steps, lr, bp_attn, ncls):   # BP embedding (+attn) + head; base mutated in place
    base.load_state_dict(baseA); d = cfg.d_model                      # reset from read-only golden (clones)
    E_p = base.E.detach().float().clone().requires_grad_(True)
    Wq_p = base.Wq.detach().float().clone().requires_grad_(True) if bp_attn else None
    Wk_p = base.Wk.detach().float().clone().requires_grad_(True) if bp_attn else None
    g = torch.Generator().manual_seed(SEED); ht = 64
    W1 = (torch.randn(d, ht, generator=g)/math.sqrt(d)).to(Z.DEVICE).requires_grad_(True); b1 = torch.zeros(ht, device=Z.DEVICE, requires_grad=True)
    W2 = (torch.randn(ht, ncls, generator=g)/math.sqrt(ht)).to(Z.DEVICE).requires_grad_(True); b2 = torch.zeros(ncls, device=Z.DEVICE, requires_grad=True)
    params = [E_p] + ([Wq_p, Wk_p] if bp_attn else []) + [W1, b1, W2, b2]
    for step in range(steps):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xtr), (64,), generator=gi)
        xb, yb = Xtr[ix].to(Z.DEVICE), Ytr[ix].to(Z.DEVICE)
        with torch.no_grad():
            _, rrep = base.context(xb); al = [base.route(rrep, base.C[l])[0] for l in range(cfg.n_layers)]
        with torch.enable_grad():
            h = P4.ctx_fwd(base, xb, E_p, Wq_p, Wk_p)
            for l in range(cfg.n_layers): h = P4.block_live(base, h, al[l], l, cfg.k_route)
            lg = torch.relu(h @ W1 + b1) @ W2 + b2; loss = torch.nn.functional.cross_entropy(lg, yb)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p, gr in zip(params, grads):
                if gr is not None: p -= lr*gr
    base.E = E_p.detach().to(cfg.td)
    if bp_attn: base.Wq = Wq_p.detach().to(cfg.td); base.Wk = Wk_p.detach().to(cfg.td)

def main():
    ck = P4.C.find_ckpt(); print(f"loading {ck}")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    fld = {f.name for f in fields(Z.Config)}; cfg = Z.Config(**{k: v for k, v in blob["cfg"].items() if k in fld})
    base = Z.ZeroGradMoE(cfg, cfg.vocab); base.load_state_dict(blob["state"]); del blob
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    baseA = base.state_dict()                                  # read-only golden checkpoint (cloned)
    d = cfg.d_model; data = Z.build_data(cfg); enc = data["_encode"]
    Xlm, Ylm = (data["Xtest"], data["Ytest"]) if "Xtest" in data else (data["Xval"], data["Yval"])
    Xtr, Ytr = gen_nli(enc, cfg.seq_len, 6000, SEED+1); Xv, Yv = gen_nli(enc, cfg.seq_len, 1500, SEED+2)
    maj = max(float((Ytr == c).float().mean()) for c in range(NCLS))
    ppl0 = Z.evaluate(base, Xlm, Ylm, cfg, batches=10**9)
    acc_zs = head_acc(base, closed_head(base, Xtr, Ytr, d, NCLS), Xv, Yv)
    rows = {"zeroshot": (acc_zs, ppl0)}
    for name, ba in [("mixedbp_emb", False), ("mixedbp_emb_attn", True)]:
        bp_adapt(base, baseA, Xtr, Ytr, cfg, STEPS, LR, ba, NCLS)
        rows[name] = (head_acc(base, closed_head(base, Xtr, Ytr, d, NCLS), Xv, Yv),
                      Z.evaluate(base, Xlm, Ylm, cfg, batches=10**9))

    summary = dict(checkpoint=str(ck), param_gigaparams=round(cfg.param_count()/1e9, 3), task="NLI-3class",
                   branch="Phase E hybrid BP (research)", majority_baseline=round(maj, 3), bp_steps=STEPS, bp_lr=LR,
                   zeroshot_acc=round(acc_zs, 4), mixedbp_emb_acc=round(rows["mixedbp_emb"][0], 4),
                   mixedbp_emb_attn_acc=round(rows["mixedbp_emb_attn"][0], 4), wiki_ppl_zeroshot=round(ppl0, 3),
                   wiki_ppl_emb=round(rows["mixedbp_emb"][1], 3), wiki_ppl_emb_attn=round(rows["mixedbp_emb_attn"][1], 3),
                   forget_emb=round(rows["mixedbp_emb"][1]-ppl0, 3), forget_emb_attn=round(rows["mixedbp_emb_attn"][1]-ppl0, 3),
                   uses_autograd=True)
    out = Path("/kaggle/working") if Z.ON_KAGGLE else Path(__file__).parent/"runs"; out.mkdir(parents=True, exist_ok=True)
    (out/"phasee_nli_run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n==== Phase E 4B NLI (3-class) ====")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\n  zero-shot {acc_zs*100:.1f}% -> emb {rows['mixedbp_emb'][0]*100:.1f}% -> emb+attn {rows['mixedbp_emb_attn'][0]*100:.1f}%")
    return summary

if __name__ == "__main__":
    main()
