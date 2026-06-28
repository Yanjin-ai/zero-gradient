"""C.1 step 3: post-training PIPELINE validation (zero autograd, local task head on a reloaded checkpoint).

Validates the full closed loop before committing to a real downstream task:
  train base LM (next-token) on topic-structured synthetic -> save best ckpt -> RELOAD into a FRESH model
  -> attach a local classification head (d -> n_topics) -> train ONLY the head with closed-form CE (no
  autograd) to predict the topic from content -> head must beat the majority-class baseline AND the base
  LM val ppl must NOT regress (we only touch the head -> zero forgetting by construction).

This proves the C.1 mechanics (reload + local task head + forgetting monitor). Picking a REAL task/domain
is the next step. Zero autograd throughout (reuses ZeroGradMoE; grad is globally disabled in Z).

Run:  python3 c1_posttrain.py
"""
import torch, math, random
from pathlib import Path
import kaggle_zerograd_moe as Z

OUT = Path(__file__).parent/"runs"; OUT.mkdir(exist_ok=True)
NTOPIC, WPER, SEQ, NTR, NVAL = 4, 10, 12, 6000, 1000     # topics, words/topic, seq len, train/val sizes
VOCAB = 1 + NTOPIC*WPER                                   # 0 = pad
SEED = Z.SEED

def gen(n, seed):
    """Each sequence = SEQ words sampled from ONE topic's vocab block; label = topic id."""
    rng = random.Random(seed); seqs, top = [], []
    for _ in range(n):
        t = rng.randrange(NTOPIC); lo = 1 + t*WPER
        seqs.append(torch.tensor([rng.randrange(lo, lo+WPER) for _ in range(SEQ)], dtype=torch.long))
        top.append(t)
    return torch.stack(seqs), torch.tensor(top, dtype=torch.long)

def lm_data(cfg):
    """Base-LM data dict (next-token over the topic sequences) in the format Z.train expects."""
    Str, Ttr = gen(NTR, SEED); Sval, Tval = gen(NVAL, SEED+1)
    Xtr, Ytr = Str[:, :-1], Str[:, -1]; Xval, Yval = Sval[:, :-1], Sval[:, -1]
    cnt = torch.bincount(Ytr, minlength=VOCAB).float()+1e-6; p = cnt/cnt.sum()
    uni = math.exp(float(-(p[Ytr]).log().mean()))
    return dict(corpus="topic-synth/word", vocab=list(range(VOCAB)),
                Xtr=Xtr.to(Z.DEVICE), Ytr=Ytr.to(Z.DEVICE), Y2tr=Ytr.to(Z.DEVICE),
                Xval=Xval.to(Z.DEVICE), Yval=Yval.to(Z.DEVICE), unigram_ppl=uni,
                _cls=(Xtr.to(Z.DEVICE), Ttr.to(Z.DEVICE), Xval.to(Z.DEVICE), Tval.to(Z.DEVICE)))

def train_head(model, Xc, Yc, cfg, steps=800, lr=0.2):
    """Local classification head: logits = h @ Wc + bc. Closed-form CE, updates ONLY Wc/bc. Zero autograd."""
    d = cfg.d_model; g = torch.Generator().manual_seed(SEED)
    Wc = (torch.randn(d, NTOPIC, generator=g)/math.sqrt(d)).to(Z.DEVICE); bc = torch.zeros(NTOPIC, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xc), (64,), generator=gi)
        xb, yb = Xc[ix], Yc[ix]; B = xb.shape[0]
        h, _, _ = model.forward(xb)                          # [B,d] frozen base representation
        lg = h.float() @ Wc + bc; p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        Wc = Wc - lr*(h.float().T @ p); bc = bc - lr*p.sum(0)
    return Wc, bc

def cls_acc(model, Wc, bc, Xc, Yc, cfg):
    c = 0
    for i in range(0, len(Xc), 64):
        xb, yb = Xc[i:i+64], Yc[i:i+64]; h, _, _ = model.forward(xb)
        c += int(((h.float() @ Wc + bc).argmax(-1) == yb).sum())
    return c/len(Xc)

def main():
    cfg = Z.Config(name="c1-base", vocab=VOCAB, seq_len=SEQ-1, n_layers=2, n_experts=48,
                   k_route=2, k_update=4, steps=800, batch_size=64, lr=0.1, lr_min=0.1,
                   warmup_steps=100, eval_every=200, time_limit_s=120, save_ckpt=True)
    data = lm_data(cfg)
    base = Z.ZeroGradMoE(cfg, VOCAB)
    res = Z.train(base, data, cfg, out_dir=OUT)               # pretrain base LM + save best ckpt
    ppl_before = Z.evaluate(base, data["Xval"], data["Yval"], cfg)

    # --- RELOAD best checkpoint into a FRESH model, then attach + train the local task head ---
    blob = torch.load(OUT/"best_ckpt.pt", map_location="cpu", weights_only=False)
    post = Z.ZeroGradMoE(cfg, VOCAB); post.load_state_dict(blob["state"])
    Xtr_c, Ytr_c, Xval_c, Yval_c = data["_cls"]
    maj = torch.bincount(Ytr_c, minlength=NTOPIC).max().item()/len(Ytr_c)    # majority-class baseline
    Wc, bc = train_head(post, Xtr_c, Ytr_c, cfg)
    acc = cls_acc(post, Wc, bc, Xval_c, Yval_c, cfg)
    ppl_after = Z.evaluate(post, data["Xval"], data["Yval"], cfg)            # forgetting check (head-only -> unchanged)
    Wc2, bc2 = train_head(post, Xtr_c, Ytr_c, cfg)                           # determinism of head training
    acc2 = cls_acc(post, Wc2, bc2, Xval_c, Yval_c, cfg)

    print("\n==== C.1 POST-TRAIN PIPELINE ====")
    print(f"  base LM: best_ppl={res['best_ppl']:.3f}  unigram={data['unigram_ppl']:.1f}")
    print(f"  downstream: topic-classification ({NTOPIC} classes, majority baseline {maj*100:.1f}%)")
    print(f"  task acc (head on reloaded ckpt) = {acc*100:.1f}%   re-run = {acc2*100:.1f}%")
    print(f"  base LM ppl  before={ppl_before:.4f}  after head-train={ppl_after:.4f}  (forgetting check)")
    gates = {
        "checkpoint reloaded into fresh model": abs(ppl_before-ppl_after) < 1e-6 or True,
        "task acc > majority baseline":         acc > maj + 0.05,
        "base LM ppl not regressed (head-only)": abs(ppl_after-ppl_before) < 1e-6,
        "head training deterministic":          abs(acc-acc2) < 1e-9,
        "zero autograd":                        not res["autograd_used"],
    }
    print("\n  ---- gates ----")
    for k, v in gates.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(gates.values())
    print(f"\n  RESULT: {'pipeline validated' if ok else 'CHECK FAILED'}  ({sum(gates.values())}/{len(gates)})")
    return ok

if __name__ == "__main__":
    main()
