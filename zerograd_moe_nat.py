"""
Zero-Gradient — Phase A / Stage 5 (3.2): natural corpus
=======================================================
Swap the clean synthetic grammar for a REAL, noisier distribution (Pride & Prejudice +
Shakespeare, data/natural.txt) WITHOUT changing the mechanism. Question: under a natural
distribution with NO topic markers, does (a) EMA-prototype content routing still find
COHERENT specialization, and (b) does the controller's importance>random advantage survive
the noise?  Same model (StackedMoE), routing rep = context MEAN (no markers), v4 coverage
auto-scaling. NO autograd anywhere.

Label-free specialization metric (no ground-truth topics in real text):
  routing COHERENCE = mean cos(key, own prototype) - mean cos(key, random prototype).
  > 0  => similar contexts land on the same expert (content routing works / experts specialize).
"""
from __future__ import annotations
import json, math, time, re
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from zerograd_nano import SEED, DEVICE
from zerograd_moe_scale import ScaleCfg, StackedMoE, train, evaluate

torch.set_grad_enabled(False)
HERE = Path(__file__).parent


def build_natural_corpus(seq_len=16, vocab_cap=4000, max_tokens=70000):
    text = (HERE/"data"/"natural.txt").read_text(encoding="utf-8")
    toks = re.findall(r"[a-zA-Z']+|[.,!?;:]", text.lower())[:max_tokens]
    vocab = ["<pad>", "<unk>"] + [w for w, _ in Counter(toks).most_common(vocab_cap-2)]
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = torch.tensor([stoi.get(w, 1) for w in toks], dtype=torch.long)
    X, Y = [], []
    for i in range(0, len(ids)-seq_len-1, 3):
        X.append(ids[i:i+seq_len]); Y.append(ids[i+seq_len])
    X = torch.stack(X); Y = torch.stack(Y)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED)); X, Y = X[perm], Y[perm]
    nval = max(256, len(X)//10); V = len(vocab)
    d = dict(Xtr=X[nval:].to(DEVICE), Ytr=Y[nval:].to(DEVICE), Xval=X[:nval].to(DEVICE), Yval=Y[:nval].to(DEVICE), vocab=vocab)
    cnt = torch.bincount(d["Ytr"], minlength=V).float()+1e-6; p = cnt/cnt.sum()
    d["unigram_ppl"] = float(torch.exp(-(p[d["Yval"]].log()).mean()))
    bg = torch.ones(V, V)*1e-3
    for a, b in zip(d["Xtr"][:, -1].tolist(), d["Ytr"].tolist()): bg[a, b] += 1.0
    bg = bg/bg.sum(1, keepdim=True)
    d["bigram_ppl"] = float(torch.exp(-(bg[d["Xval"][:, -1], d["Yval"]].log()).mean()))
    return d


def coherence(model, data, cfg, layer=0, batches=24):
    own, rnd, counts = [], [], torch.zeros(cfg.n_experts); g = torch.Generator().manual_seed(SEED)
    Cn = model.C[layer]/(model.C[layer].norm(dim=-1, keepdim=True)+1e-6)
    for i in range(0, min(len(data["Xval"]), batches*cfg.batch_size), cfg.batch_size):
        xb = data["Xval"][i:i+cfg.batch_size]; _, _, rr = model.context(xb); a, kk = model.route(rr, model.C[layer])
        for j in range(cfg.k_route):
            e = a[:, j]; own.append((kk*Cn[e]).sum(-1))
            re_ = torch.randint(0, cfg.n_experts, (len(e),), generator=g); rnd.append((kk*Cn[re_]).sum(-1))
            for ee in e.tolist(): counts[ee] += 1
    o = float(torch.cat(own).mean()); r = float(torch.cat(rnd).mean())
    pe = counts/counts.sum(); pe = pe[pe > 0]; ent = float(-(pe*pe.log()).sum())
    return round(o-r, 4), round(ent, 3), round(math.log(cfg.n_experts), 3)


def run():
    data = build_natural_corpus()
    print(f"natural corpus: vocab={len(data['vocab'])} train={len(data['Xtr'])} val={len(data['Xval'])} "
          f"unigram={data['unigram_ppl']:.1f} bigram={data['bigram_ppl']:.1f}")
    base = dict(d_model=128, n_layers=2, n_experts=32, k_update=6, seq_len=16, route_rep="mean", steps=1200)
    seeds = 4; modes = ["uniform", "random", "importance"]; ppl = {m: [] for m in modes}; mdl = None
    for sd in range(seeds):
        for m in modes:
            r = train(ScaleCfg(routing_mode=m, seed=sd, **base), data); ppl[m].append(r["ppl"])
            if m == "importance" and sd == 0: mdl = r["model"]
    coh, ent, maxent = coherence(mdl, data, ScaleCfg(**base), layer=0)
    g = np.array(ppl["random"]) - np.array(ppl["importance"])
    out = dict(stage="A-scale-3.2-natural", corpus="Pride&Prejudice+Shakespeare", vocab=len(data["vocab"]),
               unigram=round(data["unigram_ppl"], 2), bigram=round(data["bigram_ppl"], 2),
               importance=round(float(np.mean(ppl["importance"])), 3), importance_std=round(float(np.std(ppl["importance"])), 3),
               random=round(float(np.mean(ppl["random"])), 3), uniform=round(float(np.mean(ppl["uniform"])), 3),
               gap=round(float(g.mean()), 3), gap_std=round(float(g.std()), 3), pos=int((g > 0).sum()), n=seeds,
               coherence=coh, route_entropy=ent, max_entropy=maxent, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    (HERE/"runs").mkdir(exist_ok=True); (HERE/"runs"/"natural.json").write_text(json.dumps(out, indent=2))
    (HERE/"dashboard").mkdir(exist_ok=True); (HERE/"dashboard"/"dataN.js").write_text("window.STAGE_N="+json.dumps(out)+";")
    print(f"  uniform={out['uniform']} importance={out['importance']}±{out['importance_std']} random={out['random']}  "
          f"(unigram {out['unigram']}, bigram {out['bigram']})")
    print(f"  routing coherence = {coh:+.4f} (>0 = content-coherent specialization), entropy {ent}/{maxent}")
    print(f"  importance vs random gap = +{out['gap']:.3f} ± {out['gap_std']:.3f}  ({out['pos']}/{seeds} seeds positive)")
    return out


if __name__ == "__main__":
    run()
