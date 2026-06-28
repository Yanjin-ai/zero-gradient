"""v1.1 mitigation 2.2: structural partition (new task pathway, original model fully frozen).

The original model is left BYTE-FROZEN (experts, LM head, routing, attention) -> original-domain PPL is
unchanged BY CONSTRUCTION (zero forgetting). We add a new task pathway = a trainable head reading frozen
base features, and crucially also the frozen EMBEDDINGS pooled over the sequence (which the frozen weak
random attention discarded when it collapsed the sequence to one vector). All new params train by
closed-form local CE (zero autograd). This tests: can a new partition recover the task at zero forgetting,
without adapting the backbone (the thing that caused catastrophic forgetting in 2.1 / 4B-Adapt)?

Variants of the task feature fed to a 2-layer MLP task head:
  frozen-h          : base.forward last-position rep only       (= the zero-shot head-only baseline)
  pooled-emb        : mean-pooled frozen embeddings only         (new pathway, bypasses frozen attention)
  concat[h, pooled] : both                                       (structural partition)
For all: original-domain PPL == Base A (frozen) -> forgetting = 0.

Run:  python3 adapt_partition.py
"""
import math, torch
import kaggle_zerograd_moe as Z
import adapt_sentiment as A

def feats(base, X):                                            # frozen base features: last-pos h + pooled embeddings
    H, P = [], []
    for i in range(0, len(X), 64):
        xb = X[i:i+64].to(Z.DEVICE); h, _, _ = base.forward(xb)
        emb = base.E[xb].float(); pm = (xb != 0).float().unsqueeze(-1)
        H.append(h.float()); P.append((emb*pm).sum(1)/(pm.sum(1)+1e-6))
    return torch.cat(H), torch.cat(P)

def train_mlp(feat, Y, steps=1000, lr=0.2):                    # 2-layer MLP head on precomputed frozen features
    din = feat.shape[1]; g = torch.Generator().manual_seed(Z.SEED)
    W1 = (torch.randn(din, din, generator=g)/math.sqrt(din)).to(Z.DEVICE); b1 = torch.zeros(din, device=Z.DEVICE)
    W2 = (torch.randn(din, 2, generator=g)/math.sqrt(din)).to(Z.DEVICE); b2 = torch.zeros(2, device=Z.DEVICE)
    for step in range(steps):
        gi = torch.Generator().manual_seed(Z.SEED+step); ix = torch.randint(0, len(feat), (64,), generator=gi)
        f, y = feat[ix], Y[ix].to(Z.DEVICE); B = f.shape[0]
        z1 = f @ W1 + b1; a1 = torch.relu(z1); lg = a1 @ W2 + b2
        p = torch.softmax(lg, -1); p[torch.arange(B), y] -= 1.0; p /= B
        W2 = W2 - lr*(a1.T @ p); b2 = b2 - lr*p.sum(0); dz1 = (p @ W2.T)*(z1 > 0).float()
        W1 = W1 - lr*(f.T @ dz1); b1 = b1 - lr*dz1.sum(0)
    return (W1, b1, W2, b2)

def acc(hp, feat, Y):
    W1, b1, W2, b2 = hp; return float((torch.relu(feat @ W1 + b1) @ W2 + b2).argmax(-1).cpu().eq(Y).float().mean())

def main():
    cfg = Z.Config(name="part", vocab=A.VOCAB, seq_len=A.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Odata = A.lm_data(A.gen_O(8000, Z.SEED), cfg)
    Xtr_c, Ytr_c = A.gen_S_cls(6000, Z.SEED+2); Xv_c, Yv_c = A.gen_S_cls(1500, Z.SEED+3)
    maj = max(float((Ytr_c == 0).float().mean()), float((Ytr_c == 1).float().mean()))
    base = Z.ZeroGradMoE(cfg, A.VOCAB); Z.train(base, Odata, cfg)          # Base A on O, then FROZEN
    oppl = Z.evaluate(base, Odata["Xval"], Odata["Yval"], cfg)             # same for every variant (base frozen)

    Htr, Ptr = feats(base, Xtr_c); Hv, Pv = feats(base, Xv_c)
    variants = {
        "frozen-h (zero-shot)": (Htr, Hv),
        "pooled-emb (new path)": (Ptr, Pv),
        "concat[h,pooled] (partition)": (torch.cat([Htr, Ptr], -1), torch.cat([Hv, Pv], -1)),
    }
    print(f"\n==== v1.1 STRUCTURAL PARTITION (original FROZEN -> forgetting=0)  majority {maj*100:.1f}%  O-ppl={oppl:.2f} ====")
    print(f"  {'task feature':30} {'sent_acc':>9} {'O_ppl':>8} {'forget':>8}")
    rows = []
    for name, (ftr, fv) in variants.items():
        hp = train_mlp(ftr, Ytr_c); a = acc(hp, fv, Yv_c); rows.append((name, a))
        print(f"  {name:30} {a*100:>8.1f}% {oppl:>8.2f} {'+0.00':>8}")
    zs = rows[0][1]; best = max(rows[1:], key=lambda r: r[1])
    print(f"\n  zero-shot (frozen-h) {zs*100:.1f}%  ->  best partition '{best[0]}' {best[1]*100:.1f}%  at ZERO forgetting")
    print(f"  [{'PASS' if best[1] > zs + 0.15 else 'WEAK'}] structural partition beats zero-shot by >15pp with 0 forgetting")
    return rows

if __name__ == "__main__":
    main()
