"""Phase E at 4B -- hybrid BP (RESEARCH BRANCH; uses autograd, never the zero-BP submission).

On the reloaded 4.16B checkpoint: keep everything frozen and zero-BP EXCEPT a little real BP in the
adaptation stage through the EMBEDDING + task head (small-config ablation showed the embedding is the
whole lever -- BP reaches it through the frozen blocks; the top block is not needed). Measures the
zero-shot head baseline, then the BP-adapted sentiment acc, and WikiText test ppl before/after (forgetting).

Memory-safe at 4B: the autograd graph is just the embedding leaf (~130MB fp32) + head; frozen experts are
cast fp16->fp32 on the fly for only the ROUTED experts per step (a handful). Run: python3 phase_e_4b.py
Env: ZG_E_STEPS (BP steps, default 400), ZG_E_LR (default 0.05).
"""
import os, json, math, torch
from dataclasses import fields
from pathlib import Path
import kaggle_zerograd_moe as Z
import c1_4b as C

SEED = Z.SEED
E_STEPS = int(os.environ.get("ZG_E_STEPS", 400)); E_LR = float(os.environ.get("ZG_E_LR", 0.05))

def ctx_fwd(base, xb, E_p):                                    # differentiable context (embedding trainable; attn frozen, cast fp32)
    T = xb.shape[1]; emb = E_p[xb] + base.pos[:T].float().unsqueeze(0)
    q = emb @ base.Wq.float(); k = emb @ base.Wk.float()
    sc = (q @ k.transpose(1, 2))/math.sqrt(emb.shape[-1])
    m = torch.triu(torch.ones(T, T, device=xb.device), 1).bool()
    att = torch.softmax(sc.masked_fill(m, float("-inf")), -1)
    return (emb + att @ emb)[:, -1]

def block_live(base, hin, a, l, k):                           # functional block; frozen experts cast fp16->fp32 on the fly (routed only)
    out = torch.zeros_like(hin)
    for e in torch.unique(a).tolist():
        idx = (a == e).any(1).nonzero(as_tuple=True)[0]
        if idx.numel() == 0: continue
        out = out.index_add(0, idx, torch.relu(hin[idx] @ base.We[l][e].float() + base.be[l][e].float()))
    return hin + out/k

def main():
    ck = C.find_ckpt(); print(f"loading {ck}")
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    fld = {f.name for f in fields(Z.Config)}; cfg = Z.Config(**{k: v for k, v in blob["cfg"].items() if k in fld})
    base = Z.ZeroGradMoE(cfg, cfg.vocab); base.load_state_dict(blob["state"]); del blob
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    d = cfg.d_model; print(f"  reloaded {cfg.param_count()/1e9:.3f}B  vocab={cfg.vocab}  seq_len={cfg.seq_len}")

    data = Z.build_data(cfg); enc = data["_encode"]
    Xlm, Ylm = (data["Xtest"], data["Ytest"]) if "Xtest" in data else (data["Xval"], data["Yval"])
    Xtr, Ytr, _ = C.gen(enc, cfg.seq_len, 6000, SEED+1); Xv, Yv, NG = C.gen(enc, cfg.seq_len, 1500, SEED+2)
    maj = max(float((Ytr == 0).float().mean()), float((Ytr == 1).float().mean()))
    ppl_before = Z.evaluate(base, Xlm, Ylm, cfg, batches=10**9)
    acc_zeroshot = C.acc(base, C.train_head_mlp(base, Xtr, Ytr, d, partition=False), Xv, Yv)
    print(f"  [zero-shot] wiki_ppl={ppl_before:.1f}  head-only acc={acc_zeroshot*100:.1f}% (majority {maj*100:.1f}%)")

    # --- Phase E: BP through embedding + task head (everything else frozen) ---
    E_p = base.E.detach().float().clone().requires_grad_(True)
    g = torch.Generator().manual_seed(SEED); ht = 64
    W1 = (torch.randn(d, ht, generator=g)/math.sqrt(d)).to(Z.DEVICE).requires_grad_(True); b1 = torch.zeros(ht, device=Z.DEVICE, requires_grad=True)
    W2 = (torch.randn(ht, 2, generator=g)/math.sqrt(ht)).to(Z.DEVICE).requires_grad_(True); b2 = torch.zeros(2, device=Z.DEVICE, requires_grad=True)
    params = [E_p, W1, b1, W2, b2]
    print(f"  [Phase E BP] embedding + head, {E_STEPS} steps, lr={E_LR}")
    for step in range(E_STEPS):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xtr), (64,), generator=gi)
        xb, yb = Xtr[ix].to(Z.DEVICE), Ytr[ix].to(Z.DEVICE)
        with torch.no_grad():
            _, rrep = base.context(xb); al = [base.route(rrep, base.C[l])[0] for l in range(cfg.n_layers)]
        with torch.enable_grad():
            h = ctx_fwd(base, xb, E_p)
            for l in range(cfg.n_layers): h = block_live(base, h, al[l], l, cfg.k_route)
            lg = torch.relu(h @ W1 + b1) @ W2 + b2; loss = torch.nn.functional.cross_entropy(lg, yb)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p, gr in zip(params, grads):
                if gr is not None: p -= E_LR*gr
    base.E = E_p.detach().to(cfg.td)                           # write adapted embedding back
    head = ("mlp", W1.detach(), b1.detach(), W2.detach(), b2.detach(), False)
    acc_bp = C.acc(base, head, Xv, Yv); acc_bp_neg = C.acc(base, head, Xv, Yv, NG.bool())
    ppl_after = Z.evaluate(base, Xlm, Ylm, cfg, batches=10**9)

    summary = dict(checkpoint=str(ck), config=cfg.name, param_gigaparams=round(cfg.param_count()/1e9, 3),
                   branch="Phase E hybrid BP (embedding+head, research)", majority_baseline=round(maj, 3),
                   bp_steps=E_STEPS, bp_lr=E_LR, zeroshot_acc=round(acc_zeroshot, 4),
                   mixed_bp_acc=round(acc_bp, 4), mixed_bp_acc_negated=round(acc_bp_neg, 4),
                   wiki_ppl_before=round(ppl_before, 3), wiki_ppl_after=round(ppl_after, 3),
                   forgetting_dppl=round(ppl_after-ppl_before, 3), acc_gain=round(acc_bp-acc_zeroshot, 4),
                   uses_autograd=True)
    out = Path("/kaggle/working") if Z.ON_KAGGLE else Path(__file__).parent/"runs"; out.mkdir(parents=True, exist_ok=True)
    (out/"phase_e_run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\n==== Phase E 4B (hybrid BP: embedding + head) ====")
    for k, v in summary.items(): print(f"  {k}: {v}")
    print(f"\n  zero-shot {acc_zeroshot*100:.1f}% -> Mixed-BP {acc_bp*100:.1f}%  (+{(acc_bp-acc_zeroshot)*100:.1f}pp, forgetting {ppl_after-ppl_before:+.1f})")
    return summary

if __name__ == "__main__":
    main()
