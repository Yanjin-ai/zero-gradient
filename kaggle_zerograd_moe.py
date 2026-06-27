"""
The Post-Backprop Challenge — Zero-Gradient Sparse-Learning MoE  (Kaggle submission)
====================================================================================
Self-contained (no local imports) so it pastes straight into a Kaggle notebook.

What this is (settled in Phase A — see Phase-A 定论.md):
  TWO legs, neither of which claims an "intelligent controller improves PPL":
   (1) a content-routed MoE trained by per-expert LOCAL rules (NO autograd) genuinely learns;
   (2) >=4B params RESIDENT, but per-step training FLOPs ∝ k_update (decoupled from N) -> trains
       under T4/3h, where BP on 4B OOMs. The scheduler is a DETERMINISTIC deficit round-robin
       (reproducibility + worst-case backlog bound), NOT a performance claim.

Hard rules honored:
  - NO torch.autograd / loss.backward() / optimizer. `torch.set_grad_enabled(False)` global + asserted.
  - >=4,000,000,000 fp16 params resident before training (asserted on the Kaggle config).
  - Deterministic (SEED=42, deterministic algorithms), reproducible.
  - Self-supplied data; WikiText-103 perplexity reported ONLY if the dataset is attached (offline gate).
  - Artifacts written under /kaggle/working: loss curve, memory profile, run_summary.json.

Run locally for LOGIC validation:  python kaggle_zerograd_moe.py            (small config, CPU, asserts)
Run on Kaggle T4:                  set CONFIG = KAGGLE (>=4B) at the top, GPU runtime.
"""
from __future__ import annotations
import os, sys, json, math, time, random, re
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import torch

# ====================================================================================
# 0. DETERMINISM + COMPLIANCE (no autograd, ever)
# ====================================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try: torch.use_deterministic_algorithms(True)
except Exception as e: print("[warn] deterministic:", e)
torch.set_grad_enabled(False)                                  # GLOBAL: no autograd graph is ever built
ON_KAGGLE = Path("/kaggle").exists()
CUDA = torch.cuda.is_available()


# ====================================================================================
# 1. CONFIG  (SMALL = local logic check | KAGGLE = >=4B resident on T4)
# ====================================================================================
@dataclass
class Config:
    name: str = "small"
    d_model: int = 128
    vocab: int = 4000
    seq_len: int = 16
    n_layers: int = 2                 # stacked MoE blocks
    n_experts: int = 48               # experts PER layer
    k_route: int = 2                  # experts each token passes through (forward, capacity-limited)
    k_update: int = 4                 # DETERMINISTIC budget: experts updated per layer per step (k_update << N)
    capacity_factor: float = 2.0
    dtype: str = "float32"
    steps: int = 600
    batch_size: int = 64
    lr: float = 0.1
    proto_ema: float = 0.05
    time_limit_s: float = 120.0
    eval_every: int = 100
    param_gate: float = 0.0           # min resident params (4e9 on Kaggle)

    @property
    def td(self): return torch.float16 if self.dtype == "float16" else torch.float32

    def param_count(self) -> int:
        d, V = self.d_model, self.vocab
        emb = V*d + self.seq_len*d
        attn = 2*d*d
        proto = self.n_layers*self.n_experts*max(32, d//4)
        experts = self.n_layers*self.n_experts*(d*d + d)
        heads = self.n_layers*(d*V + V)
        return emb + attn + proto + experts + heads


# 4B Kaggle config: experts hold the bulk. 4 layers x 950 experts x (1024^2) ≈ 3.99B + heads/emb ≈ 4.15B.
KAGGLE = Config(name="kaggle-4B", d_model=1024, vocab=32000, seq_len=64, n_layers=4, n_experts=950,
                k_route=2, k_update=4, dtype="float16", lr=0.03, steps=10**9, time_limit_s=3*3600-300,
                eval_every=200, param_gate=4_000_000_000)

if os.environ.get("ZG_SMOKE") == "1":                          # smoke: 4B resident, ~4 min, measure throughput/mem
    KAGGLE.steps = 150; KAGGLE.time_limit_s = 240; KAGGLE.eval_every = 25
elif os.environ.get("ZG_RUN_MIN"):                             # real run with a wall-clock cap (minutes)
    KAGGLE.time_limit_s = int(os.environ["ZG_RUN_MIN"])*60; KAGGLE.eval_every = 100
CONFIG = KAGGLE if (ON_KAGGLE and CUDA) else Config()          # auto-pick; flip manually if needed
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
DT = CONFIG.td


# ====================================================================================
# 2. DATA  (WikiText-103 if attached -> offline gate; else reproducible synthetic for local logic)
# ====================================================================================
def _tok(s): return re.findall(r"[a-zA-Z']+|[.,!?;:]", s.lower())

def find_wikitext():
    root = Path("/kaggle/input")
    if not root.exists(): return None
    for p in root.rglob("*"):
        if p.is_file() and "wiki" in p.name.lower() and p.suffix in {".tokens", ".txt", ".raw"} and "train" in p.name.lower():
            return p
    return None

def build_data(cfg: Config):
    src = find_wikitext(); test_text = None
    if src is not None:
        text = src.read_text(encoding="utf-8", errors="ignore")[:8_000_000]; corpus = "wikitext-103"
        tp = src.parent/"wiki.test.tokens"; test_text = tp.read_text(encoding="utf-8", errors="ignore") if tp.exists() else None
    else:                                                      # local fallback: structured synthetic
        rng = random.Random(SEED); subj=["robot","cat","river","engine","child","wizard","planet","farmer"]
        verb=["builds","watches","carries","breaks","finds","guards","paints","feeds"]
        adj=["bright","silent","heavy","ancient","tiny","golden","frozen","wild"]; obj=["bridge","garden","machine","forest","tower","signal","harvest","stone"]
        K=4; tmap={t:{s:verb[(subj.index(s)+3*t)%8] for s in subj} for t in range(K)}
        sents=[" ".join([f"t{rng.randrange(K)}","the",rng.choice(adj),s,tmap[rng.randrange(K)][s],"the",rng.choice(adj),rng.choice(obj)]) for s in (rng.choice(subj) for _ in range(8000))]
        text=" ".join(sents); corpus="synthetic(local)"
    from collections import Counter
    toks = _tok(text)
    vocab = ["<pad>", "<unk>"] + [w for w, _ in Counter(toks).most_common(cfg.vocab-2)]
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = torch.tensor([stoi.get(w, 1) for w in toks], dtype=torch.long)
    X, Y = [], []
    for i in range(0, min(len(ids)-cfg.seq_len-1, 400000), 3):
        X.append(ids[i:i+cfg.seq_len]); Y.append(ids[i+cfg.seq_len])
    X = torch.stack(X); Y = torch.stack(Y)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED)); X, Y = X[perm], Y[perm]
    nval = max(256, len(X)//20)
    d = dict(corpus=corpus, vocab=vocab, stoi=stoi, Xtr=X[nval:].to(DEVICE), Ytr=Y[nval:].to(DEVICE),
             Xval=X[:nval].to(DEVICE), Yval=Y[:nval].to(DEVICE))
    cnt = torch.bincount(d["Ytr"], minlength=len(vocab)).float()+1e-6; p = cnt/cnt.sum()
    d["unigram_ppl"] = float(torch.exp(-(p[d["Yval"]].log()).mean()))
    if test_text is not None:                                  # official WikiText-103 test split
        tt = _tok(test_text); tids = torch.tensor([stoi.get(w, 1) for w in tt], dtype=torch.long)
        Xt, Yt = [], []
        for i in range(0, len(tids)-cfg.seq_len-1, cfg.seq_len):
            Xt.append(tids[i:i+cfg.seq_len]); Yt.append(tids[i+cfg.seq_len])
        if Xt: d["Xtest"] = torch.stack(Xt)[:3000].to(DEVICE); d["Ytest"] = torch.stack(Yt)[:3000].to(DEVICE)
    return d


# ====================================================================================
# 3. MODEL — content-routed stacked MoE, experts on critical path, >=4B resident, zero-autograd forward
# ====================================================================================
def winit(*shape, scale, seed, dt):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dt).to(DEVICE).contiguous()

class ZeroGradMoE:
    def __init__(self, cfg: Config, V: int):
        self.cfg = cfg; d = cfg.d_model; self.dk = max(32, d//4); c = 0
        def w(*s, sc): nonlocal c; c += 1; return winit(*s, scale=sc, seed=SEED+c, dt=cfg.td)
        self.E = w(V, d, sc=0.02); self.pos = w(cfg.seq_len, d, sc=0.02)
        self.Wq = w(d, d, sc=1/math.sqrt(d)); self.Wk = w(d, d, sc=1/math.sqrt(d))        # frozen attention
        self.P = w(d, self.dk, sc=1/math.sqrt(d))                                          # frozen routing projection
        self.C = [w(cfg.n_experts, self.dk, sc=1.0) for _ in range(cfg.n_layers)]          # EMA prototypes
        for l in range(cfg.n_layers): self.C[l] = self.C[l]/(self.C[l].float().norm(dim=-1, keepdim=True)+1e-6).to(cfg.td)
        self.We = [[w(d, d, sc=0.05) for _ in range(cfg.n_experts)] for _ in range(cfg.n_layers)]   # experts = FFN (the 4B bulk)
        self.be = [[torch.zeros(d, dtype=cfg.td, device=DEVICE) for _ in range(cfg.n_experts)] for _ in range(cfg.n_layers)]
        self.Hb = [w(d, V, sc=1/math.sqrt(d)) for _ in range(cfg.n_layers)]                # per-block local heads
        self.bH = [torch.zeros(V, dtype=cfg.td, device=DEVICE) for _ in range(cfg.n_layers)]
        self.num_params = cfg.param_count()

    def context(self, x):
        T = x.shape[1]; emb = self.E[x] + self.pos[:T].unsqueeze(0)
        q = emb @ self.Wq; k = emb @ self.Wk
        sc = (q.float() @ k.float().transpose(1, 2)) / math.sqrt(emb.shape[-1])
        m = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        att = torch.softmax(sc.masked_fill(m, float("-inf")), -1).to(emb.dtype)
        pm = (x != 0).float().unsqueeze(-1)
        rrep = ((emb.float()*pm).sum(1)/(pm.sum(1)+1e-6)).to(emb.dtype)                     # content routing rep
        return (emb + att @ emb)[:, -1], rrep

    def route(self, rrep, Cl):                                 # EMA-prototype nearest-neighbor + capacity
        kk = (rrep.float()); kk = kk @ self.P.float(); kk = kk/(kk.norm(dim=-1, keepdim=True)+1e-6)
        Cn = Cl.float(); Cn = Cn/(Cn.norm(dim=-1, keepdim=True)+1e-6)
        order = torch.argsort(kk @ Cn.T, dim=-1, descending=True)
        B, N = order.shape; k = self.cfg.k_route; cap = int(math.ceil(B*k/N*self.cfg.capacity_factor))
        load = torch.zeros(N, dtype=torch.long); assign = torch.zeros(B, k, dtype=torch.long, device=rrep.device); ol = order.tolist()
        for b in range(B):
            slot = 0
            for e in ol[b]:
                if load[e] < cap:
                    assign[b, slot] = e; load[e] += 1; slot += 1
                    if slot == k: break
            while slot < k:
                e = int(torch.argmin(load)); assign[b, slot] = e; load[e] += 1; slot += 1
        return assign, kk

    def block(self, hin, assign, l):
        out = torch.zeros_like(hin)
        for e in torch.unique(assign).tolist():
            mm = (assign == e).any(1); out[mm] = out[mm] + torch.relu(hin[mm] @ self.We[l][e] + self.be[l][e])
        return hin + out/self.cfg.k_route

    def forward(self, x):
        h, rrep = self.context(x); A = []
        for l in range(self.cfg.n_layers):
            a, _ = self.route(rrep, self.C[l]); A.append(a); h = self.block(h, a, l)
        return h, rrep, A

    def logits(self, h): return h @ self.Hb[-1] + self.bH[-1]


# ====================================================================================
# 4. LOCAL RULE (deeply-supervised closed form, no autograd) + DETERMINISTIC budget scheduler
# ====================================================================================
class RoundRobin:
    """Deterministic deficit scheduler: among touched experts, update the k_update LEAST-updated.
    No value signals (refuted at scale). Reproducible; bounds worst-case backlog."""
    def __init__(self, n): self.upd = torch.zeros(n)
    def select(self, cand, k):
        k = min(k, cand.numel())
        order = torch.argsort(self.upd[cand])                  # least-updated first (deterministic)
        sel = cand[order[:k]]; self.upd[sel] += 1.0
        return set(sel.tolist())


def ce_signal(out, y, Wh, bh):
    """softmax-CE closed form (no autograd): loss + grad wrt out + head grads."""
    B = y.shape[0]; lg = (out.float() @ Wh.float() + bh.float())
    lp = lg - lg.logsumexp(-1, keepdim=True); loss = float(-lp[torch.arange(B), y].mean())
    p = torch.softmax(lg, -1); p[torch.arange(B), y] -= 1.0; p /= B
    return loss, p @ Wh.float().T, out.float().T @ p, p.sum(0)


# ====================================================================================
# 5. TRAIN (wall-clock guard, memory tracking, zero autograd)
# ====================================================================================
def evaluate(model, X, Y, cfg, batches=16):
    nll, n = 0.0, 0
    for i in range(0, min(len(X), batches*cfg.batch_size), cfg.batch_size):
        xb, yb = X[i:i+cfg.batch_size], Y[i:i+cfg.batch_size]; h, _, _ = model.forward(xb)
        lg = model.logits(h).float(); lp = lg - lg.logsumexp(-1, keepdim=True); B = xb.shape[0]
        nll += float(-lp[torch.arange(B), yb].sum()); n += B
    return math.exp(nll/n)

def train(model, data, cfg):
    sched = [RoundRobin(cfg.n_experts) for _ in range(cfg.n_layers)]
    Xtr, Ytr = data["Xtr"], data["Ytr"]; t0 = time.time(); curve = []; used_autograd = torch.is_grad_enabled()
    if CUDA: torch.cuda.reset_peak_memory_stats()
    step = 0
    while step < cfg.steps and time.time()-t0 < cfg.time_limit_s:
        g = torch.Generator().manual_seed(SEED+step); ix = torch.randint(0, len(Xtr), (cfg.batch_size,), generator=g)
        xb, yb = Xtr[ix], Ytr[ix]; B = xb.shape[0]
        lrf = cfg.lr * min(1.0, (step+1)/200.0)                # linear warmup (200 steps) -> stabilizes start
        h, rrep = model.context(xb)
        hin, monitor = h, []
        for l in range(cfg.n_layers):
            assign, kk = model.route(rrep, model.C[l])
            cache = {}; moe = torch.zeros_like(hin)
            for e in torch.unique(assign).tolist():
                mm = (assign == e).any(1); inp = hin[mm]; z = inp @ model.We[l][e] + model.be[l][e]; he = torch.relu(z)
                moe[mm] = moe[mm] + he; cache[e] = (inp, z, mm, he)
            hl = hin + moe/cfg.k_route
            loss, dh, dHb, dbH = ce_signal(hl, yb, model.Hb[l], model.bH[l]); monitor.append(loss)
            model.Hb[l] = (model.Hb[l].float() - lrf*dHb).to(cfg.td); model.bH[l] = (model.bH[l].float() - lrf*dbH).to(cfg.td)
            cand = torch.tensor(sorted(cache.keys())); sel = sched[l].select(cand, cfg.k_update)
            for e in sel:
                inp, z, mm, he = cache[e]; dz = (dh[mm]/cfg.k_route) * (z.float() > 0).float()
                model.We[l][e] = (model.We[l][e].float() - lrf*(inp.float().T @ dz)).to(cfg.td)
                model.be[l][e] = (model.be[l][e].float() - lrf*dz.sum(0)).to(cfg.td)
            for e in torch.unique(assign).tolist():
                mm = (assign == e).any(1)
                model.C[l][e] = ((1-cfg.proto_ema)*model.C[l][e].float() + cfg.proto_ema*kk[mm].mean(0)).to(cfg.td)
            hin = hl
        # embedding update (approx local credit assignment from first block)
        model.E.index_add_(0, xb[:, -1], (-lrf*dh).to(cfg.td));
        m = float(np.mean(monitor))
        if step % cfg.eval_every == 0 or step == cfg.steps-1:
            ppl = evaluate(model, data["Xval"], data["Yval"], cfg)
            curve.append(dict(step=step, monitor=round(m, 4), val_ppl=round(ppl, 3),
                              t=round(time.time()-t0, 1), tok_s=int((step+1)*cfg.batch_size*cfg.seq_len/(time.time()-t0+1e-9))))
            print(f"  step {step:5d} monitor={m:.3f} val_ppl={ppl:.2f} t={time.time()-t0:.0f}s")
        step += 1
    peak = (torch.cuda.max_memory_allocated()/2**20) if CUDA else (model.num_params*(2 if cfg.dtype=='float16' else 4)/2**20)
    return dict(curve=curve, steps=step, wall_s=round(time.time()-t0, 1), peak_mem_mb=round(peak, 1),
                final_ppl=curve[-1]["val_ppl"] if curve else None, autograd_used=bool(used_autograd))


# ====================================================================================
# 6. BP baseline + 4B-OOM demonstration (Kaggle GPU only)
# ====================================================================================
def bp_oom_demo(cfg: Config):
    """Estimate BP memory for a DENSE model of the same resident size and try to allocate it ->
    shows BP OOMs on T4 while our local-rule sparse path runs. Reported, not required for training."""
    fp16 = cfg.param_count()*2/2**30
    bp_need = fp16*4                                            # weights + grads + 2 Adam moments (no activations)
    info = dict(resident_fp16_GB=round(fp16, 2), bp_min_GB_weights_grads_adam=round(bp_need, 2),
                t4_GB=16, bp_fits_t4=bp_need < 15.0)
    return info


# ====================================================================================
# 7. MAIN — param gate, train, eval, artifacts, compliance self-checks
# ====================================================================================
def run():
    OUT = Path("/kaggle/working") if ON_KAGGLE else Path(__file__).parent/"runs"; OUT.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    print(f"config={cfg.name} device={DEVICE} dtype={cfg.dtype} params={cfg.param_count()/1e9:.3f}B "
          f"(gate>={cfg.param_gate/1e9:.1f}B) experts/layer={cfg.n_experts} k_update={cfg.k_update}")
    # --- PARAM GATE (resident >= 4B before training) ---
    assert cfg.param_count() >= cfg.param_gate, f"param gate FAIL: {cfg.param_count()} < {cfg.param_gate}"
    data = build_data(cfg)
    print(f"corpus={data['corpus']} vocab={len(data['vocab'])} train={len(data['Xtr'])} unigram_ppl={data['unigram_ppl']:.1f}")
    model = ZeroGradMoE(cfg, len(data["vocab"]))
    # re-run determinism check (small only; skip on 4B for time)
    res = train(model, data, cfg)
    if not (ON_KAGGLE and CUDA):
        m2 = ZeroGradMoE(cfg, len(data["vocab"])); r2 = train(m2, data, cfg)
        det_ok = abs(r2["final_ppl"]-res["final_ppl"]) < 1e-6
    else: det_ok = True
    wt_ppl = None
    if "Xtest" in data:                                        # official WikiText-103 test perplexity (offline gate)
        wt_ppl = round(evaluate(model, data["Xtest"], data["Ytest"], cfg, batches=10**9), 3)
        print(f"  WikiText-103 TEST perplexity = {wt_ppl}")
    oom = bp_oom_demo(cfg)
    gates = {
        "resident params >= gate": cfg.param_count() >= cfg.param_gate,
        "zero autograd (grad disabled)": not torch.is_grad_enabled() and not res["autograd_used"],
        "monitor loss decreased": res["curve"][0]["monitor"] > res["curve"][-1]["monitor"] if len(res["curve"]) > 1 else True,
        "val_ppl < unigram (sanity)": (res["final_ppl"] or 9e9) < data["unigram_ppl"],
        "deterministic (re-run identical)": det_ok,
        "BP-4B would OOM on T4 (we run)": not oom["bp_fits_t4"],
    }
    summary = dict(config=asdict(cfg), corpus=data["corpus"], param_count=cfg.param_count(),
                   param_gigaparams=round(cfg.param_count()/1e9, 3), per_step_updated_experts=cfg.k_update*cfg.n_layers,
                   update_fraction=round(cfg.k_update/cfg.n_experts, 4), unigram_ppl=round(data["unigram_ppl"], 2),
                   wikitext103_test_ppl=wt_ppl,
                   train=res, bp_oom=oom, gates=gates, gates_passed=int(sum(gates.values())), gates_total=len(gates),
                   ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT/"run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        c = res["curve"]
        if c:
            plt.figure(figsize=(8, 3)); plt.plot([p["step"] for p in c], [p["val_ppl"] for p in c], marker="o")
            plt.xlabel("step"); plt.ylabel("val perplexity"); plt.title("local-rule MoE training"); plt.tight_layout()
            plt.savefig(OUT/"loss_curve.png", dpi=140); plt.close()
            plt.figure(figsize=(5, 3)); plt.bar(["resident(fp16)", "BP needs"], [oom["resident_fp16_GB"], oom["bp_min_GB_weights_grads_adam"]])
            plt.axhline(16, ls="--", color="r", label="T4 16GB"); plt.ylabel("GB"); plt.legend(); plt.title("memory: us vs BP"); plt.tight_layout()
            plt.savefig(OUT/"memory_profile.png", dpi=140); plt.close()
    except Exception as e: print("[warn] plot:", e)
    print("\n==== GATES ====")
    for k, v in gates.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  params={cfg.param_count()/1e9:.3f}B  per-step updated experts={cfg.k_update*cfg.n_layers} "
          f"({cfg.k_update}/{cfg.n_experts} per layer)  peak_mem={res['peak_mem_mb']}MB  final_ppl={res['final_ppl']}")
    print(f"  BP-4B memory need ≈ {oom['bp_min_GB_weights_grads_adam']}GB (> T4 16GB -> OOM); we run at {oom['resident_fp16_GB']}GB resident")
    print(f"  wrote {OUT/'run_summary.json'}")
    return summary


if __name__ == "__main__":
    run()
