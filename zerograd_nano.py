"""
Zero-Gradient Learning — nano vertical slice
============================================

A SINGLE-FILE, runnable validation of the project's core algorithmic bet:

    "A deterministic, gradient-FREE importance controller that allocates training
     resources across layers/experts can beat random allocation."

Everything here obeys the competition's hard rule: NO autograd, NO loss.backward(),
NO torch optimizer. All updates are hand-written closed-form local rules.
`torch.set_grad_enabled(False)` is set globally and asserted.

This file is the structure-preserving COMPRESSION of the 4B design (see CONFIG
section). nano runs on CPU in seconds; the SAME code scales to 4B on a Kaggle T4
by changing only the numbers in `Config`.

Outputs (for the dashboard):
    runs/run.json          consolidated metrics + gate results (dashboard reads this)
    runs/metrics.jsonl     fine-grained per-step log (one JSON object per line)
"""

from __future__ import annotations
import os, json, math, time, random, hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import torch

# --------------------------------------------------------------------------------------
# 0. DETERMINISM + COMPLIANCE  (gate: reproducible, zero-autograd)
# --------------------------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print("[warn] deterministic algorithms not fully enabled:", e)
torch.set_grad_enabled(False)                       # global: no autograd graph, ever
DEVICE = torch.device("cpu")                         # CPU for bit-exact determinism on this box
DTYPE = torch.float32                                # nano uses fp32; 4B preset uses fp16


# ======================================================================================
# 1. CONFIG  —  4B target design, then the structure-preserving nano compression
# ======================================================================================
# The 4B model is a "small real backbone (carries quality) + large sparse expert bank
# (carries the >=4B param gate, is the stage for the importance controller)".
#
#   4B PRESET (conceptual target, runs only on a T4):
#       vocab V=32000, d=1024, backbone L_b=8 blocks, experts N_e≈3700 (each d×d≈1.05M),
#       forward top-k_fwd=4 experts/token, shared next-token head H (d×V, one copy).
#       params ≈ embed 32.8M + backbone 8.4M + shared head 32.8M + experts 3700×1.05M≈3.88B
#             ≈ 3.95B fp16 ≈ 7.9 GB  -> fits T4 16GB; BP on the same would OOM.
#
#   STRUCTURE-PRESERVING RATIOS kept when shrinking to nano:
#       depth/width, experts:backbone (sparsity), k_fwd:N_e, head:hidden, residual stream.

@dataclass
class Config:
    name: str = "nano"
    # --- shapes (nano numbers; 4B numbers in comments above) ---
    d_model: int = 64
    n_backbone: int = 3            # backbone blocks (real-ish LM quality core)
    n_experts: int = 16            # sparse bank (carries params + hosts the controller)
    k_fwd: int = 4                 # experts each sample is routed through (forward)
    k_update: int = 6              # experts updated per step (the BUDGET the controller allocates)
    seq_len: int = 12              # context window length
    # --- data ---
    vocab_cap: int = 256
    n_sentences: int = 6000
    # --- training ---
    steps: int = 600
    batch_size: int = 64
    lr_backbone: float = 0.05
    lr_expert: float = 0.05
    lr_head: float = 0.05
    lr_embed: float = 0.02
    eval_every: int = 25
    eval_batches: int = 16
    time_limit_s: float = 120.0
    # --- importance controller (gradient-FREE) ---
    ema_rho: float = 0.9           # smoothing of importance score
    alpha_learn: float = 1.0       # weight on "learnability"  (still improving?)
    beta_util: float = 1.0         # weight on "utility"       (competent / useful now?)
    gamma_cost: float = 0.3        # weight on "cost"          (FLOPs / state)
    load_balance: float = 0.5      # penalty on over-used experts (keeps bank from collapsing)
    soft_floor: float = 0.15       # hybrid: non-selected experts still get this fraction of lr
    routing_mode: str = "importance"   # {"importance","random"} -> the headline ablation

    def param_count(self) -> int:
        V = self.vocab_cap
        embed = V * self.d_model
        pos = self.seq_len * self.d_model
        backbone = self.n_backbone * (self.d_model * self.d_model + self.d_model)
        experts = self.n_experts * (self.d_model * self.d_model + self.d_model)
        head = self.d_model * V
        return embed + pos + backbone + experts + head


# ======================================================================================
# 2. DATA  —  deterministic structured synthetic corpus (real learnable next-token
#    structure, offline, reproducible). Kaggle swaps this for real WikiText-103.
# ======================================================================================
def build_corpus(cfg: Config):
    rng = random.Random(SEED)
    # a tiny grammar with bigram-ish structure so next-token is genuinely predictable
    subj = ["robot", "cat", "river", "engine", "child", "wizard", "planet", "farmer"]
    verb = ["builds", "watches", "carries", "breaks", "finds", "guards", "paints", "feeds"]
    adj  = ["bright", "silent", "heavy", "ancient", "tiny", "golden", "frozen", "wild"]
    obj  = ["bridge", "garden", "machine", "forest", "tower", "signal", "harvest", "stone"]
    conj = ["and then", "because", "while", "so"]
    # strong structure: each subject prefers a subset of verbs (learnable association)
    pref = {s: rng.sample(verb, 3) for s in subj}
    sentences = []
    for _ in range(cfg.n_sentences):
        s = rng.choice(subj)
        v = rng.choice(pref[s])                       # subject->verb association (the signal)
        parts = ["the", rng.choice(adj), s, v, "the", rng.choice(adj), rng.choice(obj)]
        if rng.random() < 0.5:
            parts += [rng.choice(conj), "the", rng.choice(adj), rng.choice(obj)]
        sentences.append(" ".join(parts))
    text = " <eos> ".join(sentences)
    toks = text.split()
    # word-level tokenizer (the encode/window interface matches the BPE we use on Kaggle)
    from collections import Counter
    vocab = ["<pad>", "<unk>", "<eos>"] + [w for w, _ in Counter(toks).most_common(cfg.vocab_cap - 3)]
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = torch.tensor([stoi.get(w, 1) for w in toks], dtype=torch.long)
    # windows: context -> next token
    X, Y = [], []
    for i in range(0, len(ids) - cfg.seq_len - 1, 3):
        X.append(ids[i:i + cfg.seq_len]); Y.append(ids[i + cfg.seq_len])
    X = torch.stack(X); Y = torch.stack(Y)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))
    X, Y = X[perm], Y[perm]
    n_val = max(256, len(X) // 10)
    data = dict(Xtr=X[n_val:].to(DEVICE), Ytr=Y[n_val:].to(DEVICE),
                Xval=X[:n_val].to(DEVICE), Yval=Y[:n_val].to(DEVICE),
                vocab=vocab, stoi=stoi)
    # unigram baseline perplexity (the bar the model must beat)
    counts = torch.bincount(data["Ytr"], minlength=len(vocab)).float() + 1e-6
    p = counts / counts.sum()
    data["unigram_ppl"] = float(torch.exp(-(p[data["Yval"]].log()).mean()))
    return data


# ======================================================================================
# 3. MODEL  —  hand-written forward that EXPOSES every unit's state for local updates.
#    backbone block: [fixed reservoir attention] -> trainable linear+relu (+residual)
#    expert:         trainable linear+relu, routed top-k by a FIXED key (forward routing)
#    shared head H:  one next-token readout reused by every local objective (keeps params low)
# ======================================================================================
def init_(*shape, scale):
    return (torch.randn(*shape, generator=torch.Generator().manual_seed(SEED + sum(shape)),
                        dtype=DTYPE, device=DEVICE) * scale).contiguous()

class ZeroGradLM:
    def __init__(self, cfg: Config, V: int):
        self.cfg = cfg; self.V = V; d = cfg.d_model
        self.E = init_(V, d, scale=0.02)
        self.pos = init_(cfg.seq_len, d, scale=0.02)
        # fixed reservoir attention projection (NOT trained -> no gradient needed through it)
        self.Wq = init_(d, d, scale=1/math.sqrt(d)); self.Wk = init_(d, d, scale=1/math.sqrt(d))
        # trainable backbone blocks
        self.Wb = [init_(d, d, scale=1/math.sqrt(d)) for _ in range(cfg.n_backbone)]
        self.bb = [torch.zeros(d, dtype=DTYPE, device=DEVICE) for _ in range(cfg.n_backbone)]
        # trainable experts + their FIXED routing keys
        self.We = [init_(d, d, scale=1/math.sqrt(d)) for _ in range(cfg.n_experts)]
        self.be = [torch.zeros(d, dtype=DTYPE, device=DEVICE) for _ in range(cfg.n_experts)]
        self.Ekey = init_(cfg.n_experts, d, scale=1/math.sqrt(d))   # fixed forward router
        # per-backbone-block heads (deeply-supervised, avoid head-fighting) + final readout head
        self.Hb = [init_(d, V, scale=1/math.sqrt(d)) for _ in range(cfg.n_backbone)]
        self.bHb = [torch.zeros(V, dtype=DTYPE, device=DEVICE) for _ in range(cfg.n_backbone)]
        self.Hf = init_(d, V, scale=1/math.sqrt(d))            # eval path goes through this
        self.bf = torch.zeros(V, dtype=DTYPE, device=DEVICE)

    # ---- pieces ----
    def context(self, x):
        T = x.shape[1]
        h = self.E[x] + self.pos[:T].unsqueeze(0)                    # [B,T,d]
        q = h @ self.Wq; k = h @ self.Wk
        scores = (q @ k.transpose(1, 2)) / math.sqrt(h.shape[-1])
        mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))            # causal
        att = torch.softmax(scores, dim=-1)
        mixed = h + att @ h                                         # residual around fixed attention
        return mixed[:, -1], att[:, -1, :]                          # context vec, last-row attention

    def logits(self, h):                                            # final readout head
        return h @ self.Hf + self.bf

    def route(self, h):                                            # forward top-k_fwd experts/sample
        score = h @ self.Ekey.T                                     # [B,Ne]
        topk = torch.topk(score, self.cfg.k_fwd, dim=-1).indices    # [B,k_fwd]
        return topk

    def forward_states(self, x):
        """Return all per-unit states needed for local updates AND the final prediction."""
        h0, att_last = self.context(x); states = {"h0": h0, "att_last": att_last, "x": x}
        h = h0; bb_in, bb_z, bb_out = [], [], []
        for l in range(self.cfg.n_backbone):
            z = h @ self.Wb[l] + self.bb[l]; a = torch.relu(z)
            bb_in.append(h); bb_z.append(z); bb_out.append(a + h)   # residual
            h = a + h
        states.update(bb_in=bb_in, bb_z=bb_z, bb_out=bb_out, h_back=h)
        # experts operate on the backbone output
        topk = self.route(h)                                        # [B,k_fwd]
        states["topk"] = topk
        return states


# ======================================================================================
# 4. LOCAL LEARNING RULE  —  deeply-supervised, closed-form, NO autograd.
#    For a unit producing output `out` from input `inp` via z=inp@W+b, a=relu(z):
#       local target = next token y, via shared head H
#       p = softmax(H·out);  dlogits = (p - onehot(y)) / B
#       dH    = out^T dlogits                      (shared head)
#       dout  = dlogits @ H^T
#       dz    = dout * relu'(z)
#       dW    = inp^T dz ;  db = dz.sum(0)
#    Returns local cross-entropy loss (a gradient-free competence signal) + the deltas.
# ======================================================================================
def local_signals(out, inp, z, y, Whead, bhead):
    """Closed-form local update for one block + its readout head. Returns dout too."""
    B = y.shape[0]
    logits = out @ Whead + bhead
    logp = logits - logits.logsumexp(-1, keepdim=True)
    loss = float(-logp[torch.arange(B), y].mean())
    p = torch.softmax(logits, -1); p[torch.arange(B), y] -= 1.0; p /= B
    dWhead = out.T @ p; dbhead = p.sum(0)
    dout = p @ Whead.T
    dz = dout * (z > 0).to(DTYPE)
    dW = inp.T @ dz; db = dz.sum(0)
    return loss, dW, db, dWhead, dbhead, dout


# ======================================================================================
# 5. IMPORTANCE CONTROLLER  —  deterministic, gradient-FREE resource allocator.
#    Per expert it tracks (all from cheap forward statistics, NO gradients):
#       learnability = recent drop rate of its local loss   (still has room to learn)
#       utility      = competence = -local_loss (z-scored)  (useful to the task now)
#       cost         = static state/FLOPs proxy
#       I = α·learn + β·util - γ·cost - load_balance·usage_ema
#    EMA-smoothed. Allocation = top-k_update get full lr (hard) + soft floor for the rest.
#    routing_mode="random" -> select the SAME number of experts at random (the ablation).
# ======================================================================================
class Controller:
    def __init__(self, cfg: Config):
        self.cfg = cfg; n = cfg.n_experts
        self.loss_ema = torch.full((n,), float("nan"))
        self.loss_prev = torch.full((n,), float("nan"))
        self.learn = torch.zeros(n); self.util = torch.zeros(n)
        self.usage_ema = torch.zeros(n)
        self.I = torch.zeros(n); self.I_ema = torch.zeros(n)
        self.cost = torch.ones(n)                                   # uniform in nano (all experts same size)

    @staticmethod
    def _z(x):
        m = x[torch.isfinite(x)]
        if m.numel() < 2: return torch.zeros_like(x)
        return (x - m.mean()) / (m.std() + 1e-6)

    def observe(self, expert_loss: dict, touched: torch.Tensor):
        """expert_loss: {eid: local_loss}; touched: bool mask of experts seen this step."""
        for e, L in expert_loss.items():
            prev = self.loss_ema[e]
            self.loss_ema[e] = L if torch.isnan(prev) else float(0.7 * prev + 0.3 * L)
            if not torch.isnan(prev):
                self.learn[e] = max(0.0, float(prev - self.loss_ema[e]))   # drop rate
            self.util[e] = -self.loss_ema[e]
        self.usage_ema = 0.95 * self.usage_ema + 0.05 * touched.float()

    def score_and_select(self, candidates: torch.Tensor, step: int):
        c = self.cfg
        I = (c.alpha_learn * self._z(self.learn) + c.beta_util * self._z(self.util)
             - c.gamma_cost * self._z(self.cost) - c.load_balance * self._z(self.usage_ema))
        self.I = I
        self.I_ema = c.ema_rho * self.I_ema + (1 - c.ema_rho) * I
        k = min(c.k_update, candidates.numel())
        if c.routing_mode == "random":
            g = torch.Generator().manual_seed(SEED + step)
            sel = candidates[torch.randperm(candidates.numel(), generator=g)[:k]]
        else:
            order = torch.argsort(self.I_ema[candidates], descending=True)
            sel = candidates[order[:k]]
        return set(sel.tolist())


# ======================================================================================
# 6. TRAIN + EVAL  +  fine-grained logging
# ======================================================================================
def evaluate(model: ZeroGradLM, X, Y, cfg: Config):
    nll, correct, n = 0.0, 0, 0
    for i in range(0, min(len(X), cfg.eval_batches * cfg.batch_size), cfg.batch_size):
        xb, yb = X[i:i+cfg.batch_size], Y[i:i+cfg.batch_size]
        st = model.forward_states(xb)
        h = st["h_back"]; topk = st["topk"]; B = xb.shape[0]
        # final prediction = backbone output + mean of routed experts (experts contribute to quality)
        cand = torch.unique(topk).flatten()
        exp_mix = torch.zeros_like(h)
        for e in cand.tolist():
            m = (topk == e).any(dim=1)
            exp_mix[m] += torch.relu(h[m] @ model.We[e] + model.be[e])
        h_final = h + exp_mix / cfg.k_fwd
        logits = model.logits(h_final)
        logp = logits - logits.logsumexp(-1, keepdim=True)
        nll += float(-logp[torch.arange(B), yb].sum()); n += B
        correct += int((logits.argmax(-1) == yb).sum())
    return math.exp(nll / n), correct / n


def train(cfg: Config, data, log_rows: list):
    model = ZeroGradLM(cfg, len(data["vocab"]))
    ctrl = Controller(cfg)
    Xtr, Ytr = data["Xtr"], data["Ytr"]
    t0 = time.time(); anomalies = []
    unit_hist = {f"E{e}": [] for e in range(cfg.n_experts)}   # importance EMA timeline
    sel_hist = {f"E{e}": [] for e in range(cfg.n_experts)}
    curve = []
    for step in range(cfg.steps):
        if time.time() - t0 > cfg.time_limit_s:
            print(f"[time] stop at step {step}"); break
        g = torch.Generator().manual_seed(SEED + 1000 + step)
        idx = torch.randint(0, len(Xtr), (cfg.batch_size,), generator=g)
        xb, yb = Xtr[idx], Ytr[idx]; B = xb.shape[0]
        st = model.forward_states(xb)

        # ---- backbone: deeply-supervised local update with per-block heads (quality core) ----
        bb_loss = []
        for l in range(cfg.n_backbone):
            L, dW, db, dH, dbH, _ = local_signals(st["bb_out"][l], st["bb_in"][l], st["bb_z"][l],
                                                  yb, model.Hb[l], model.bHb[l])
            model.Wb[l] -= cfg.lr_backbone * dW; model.bb[l] -= cfg.lr_backbone * db
            model.Hb[l] -= cfg.lr_head * dH; model.bHb[l] -= cfg.lr_head * dbH
            bb_loss.append(L)

        # ---- experts: forward-routed; combined into h_final; final head drives updates ----
        h = st["h_back"]; topk = st["topk"]
        cand = torch.unique(topk).flatten()
        he_cache, route_mask = {}, {}
        exp_mix = torch.zeros_like(h)
        for e in cand.tolist():
            m = (topk == e).any(dim=1); route_mask[e] = m
            inp = h[m]; z = inp @ model.We[e] + model.be[e]; he = torch.relu(z)
            he_cache[e] = (inp, z, he); exp_mix[m] += he
        h_final = h + exp_mix / cfg.k_fwd

        # final readout head: closed-form local update + one-step signal to experts (no autograd graph)
        logits_f = h_final @ model.Hf + model.bf
        logp = logits_f - logits_f.logsumexp(-1, keepdim=True)
        final_loss = float(-logp[torch.arange(B), yb].mean())
        pf = torch.softmax(logits_f, -1); pf[torch.arange(B), yb] -= 1.0; pf /= B
        model.Hf -= cfg.lr_head * (h_final.T @ pf); model.bf -= cfg.lr_head * pf.sum(0)
        dh_final = pf @ model.Hf.T

        # controller signals (gradient-free) + final-objective-aligned local delta per expert
        touched = torch.zeros(cfg.n_experts); expert_loss, expert_state = {}, {}
        for e in cand.tolist():
            inp, z, he = he_cache[e]; m = route_mask[e]; mm = int(m.sum())
            comp = (h[m] + he) @ model.Hf + model.bf            # competence of this expert's contribution
            clp = comp - comp.logsumexp(-1, keepdim=True)
            expert_loss[e] = float(-clp[torch.arange(mm), yb[m]].mean())
            dz = (dh_final[m] / cfg.k_fwd) * (z > 0).to(DTYPE)  # one-step local signal from final head
            expert_state[e] = (inp.T @ dz, dz.sum(0)); touched[e] = 1.0
        ctrl.observe(expert_loss, touched)
        selected = ctrl.score_and_select(cand, step)

        # ---- hybrid hard/soft allocation: selected -> full lr; others -> soft floor ----
        upd_norm = {}
        for e in cand.tolist():
            dWe, dbe = expert_state[e]
            scale = cfg.lr_expert if e in selected else cfg.lr_expert * cfg.soft_floor
            model.We[e] -= scale * dWe; model.be[e] -= scale * dbe
            upd_norm[e] = float(dWe.norm()) * scale

        # ---- embedding update: approximate local credit assignment through fixed attention ----
        weighted = st["att_last"].unsqueeze(-1) * dh_final.unsqueeze(1)   # [B,T,d] attention path
        model.E.index_add_(0, xb.reshape(-1), (-cfg.lr_embed * weighted).reshape(-1, cfg.d_model))
        model.pos[:xb.shape[1]] -= cfg.lr_embed * weighted.sum(0)
        model.E.index_add_(0, xb[:, -1], -cfg.lr_embed * dh_final)        # direct residual path (last token)
        model.pos[xb.shape[1] - 1] -= cfg.lr_embed * dh_final.sum(0)

        monitor = float(np.mean(bb_loss + [final_loss]))
        if not math.isfinite(monitor):
            anomalies.append({"step": step, "kind": "nan_loss", "detail": "monitor loss not finite"})

        # ---- fine-grained log row ----
        if step % 5 == 0 or step == cfg.steps - 1:
            for e in range(cfg.n_experts):
                unit_hist[f"E{e}"].append(round(float(ctrl.I_ema[e]), 4))
                sel_hist[f"E{e}"].append(1 if e in selected else 0)
            row = dict(step=step, t=round(time.time()-t0, 3), monitor_loss=round(monitor, 4),
                       backbone_loss=round(float(np.mean(bb_loss)), 4),
                       n_candidates=int(cand.numel()), n_selected=len(selected),
                       selected=sorted(selected),
                       max_update_norm=round(max(upd_norm.values()) if upd_norm else 0.0, 4),
                       route_entropy=round(float(_entropy(touched)), 4))
            log_rows.append(row)

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            ppl, acc = evaluate(model, data["Xval"], data["Yval"], cfg)
            curve.append(dict(step=step, val_ppl=round(ppl, 3), val_acc=round(acc, 4),
                              monitor_loss=round(monitor, 4)))
            print(f"  step {step:4d}  monitor={monitor:.3f}  val_ppl={ppl:.2f}  acc={acc:.3f}")

    return dict(curve=curve, unit_hist=unit_hist, sel_hist=sel_hist, anomalies=anomalies,
                wall_s=round(time.time()-t0, 2), final=curve[-1] if curve else None)


def _entropy(counts):
    p = counts / (counts.sum() + 1e-9); p = p[p > 0]
    return -(p * p.log()).sum() if p.numel() else torch.tensor(0.0)


# ======================================================================================
# 7. MAIN  —  run the headline ablation (importance vs random) + gates + dashboard json
# ======================================================================================
def run():
    OUT = Path(__file__).parent / "runs"; OUT.mkdir(exist_ok=True)
    base = Config()
    data = build_corpus(base)
    print(f"corpus: vocab={len(data['vocab'])} train={len(data['Xtr'])} val={len(data['Xval'])} "
          f"unigram_ppl={data['unigram_ppl']:.2f}")
    p4b = Config(name="4B", d_model=1024, n_backbone=8, n_experts=3700,
                 seq_len=512, vocab_cap=32000).param_count()
    print(f"4B-preset param count (info): {p4b/1e9:.2f}B")

    results, jsonl = {}, []
    for mode in ["importance", "random"]:
        print(f"\n=== routing_mode = {mode} ===")
        cfg = Config(routing_mode=mode)
        rows = []
        res = train(cfg, data, rows)
        results[mode] = res
        for r in rows: r["mode"] = mode; jsonl.append(r)

    # ---- determinism check: re-run importance, compare final ppl ----
    rerun = train(Config(routing_mode="importance"), data, [])
    det_ok = abs(rerun["final"]["val_ppl"] - results["importance"]["final"]["val_ppl"]) < 1e-6

    imp_ppl = results["importance"]["final"]["val_ppl"]
    rnd_ppl = results["random"]["final"]["val_ppl"]
    gates = {
        "deterministic (re-run identical)": det_ok,
        "zero autograd (grad globally disabled)": not torch.is_grad_enabled(),
        "monitor loss decreased": results["importance"]["curve"][0]["monitor_loss"]
                                   > results["importance"]["curve"][-1]["monitor_loss"],
        "val_ppl < unigram baseline": imp_ppl < data["unigram_ppl"],
        "importance routing < random routing (HEADLINE)": imp_ppl < rnd_ppl,
        "no NaN/Inf anomalies": len(results["importance"]["anomalies"]) == 0,
    }

    summary = dict(
        config=asdict(Config()),
        param_count_nano=Config().param_count(),
        param_count_4B=Config(name="4B", d_model=1024, n_backbone=8, n_experts=3700,
                              seq_len=512, vocab_cap=32000).param_count(),
        unigram_ppl=round(data["unigram_ppl"], 3),
        importance=results["importance"], random=results["random"],
        determinism_ok=det_ok, gates=gates,
        gates_passed=int(sum(gates.values())), gates_total=len(gates),
    )
    (OUT / "run.json").write_text(json.dumps(summary, indent=2))
    (OUT / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in jsonl))
    dash = Path(__file__).parent / "dashboard"; dash.mkdir(exist_ok=True)
    (dash / "data.js").write_text("window.RUN_DATA = " + json.dumps(summary) + ";")

    print("\n================  GATES  ================")
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    print(f"\n  importance val_ppl = {imp_ppl:.3f}   random val_ppl = {rnd_ppl:.3f}   "
          f"unigram = {data['unigram_ppl']:.3f}")
    print(f"  wrote {OUT/'run.json'} and {OUT/'metrics.jsonl'}")
    print(f"  open the dashboard: dashboard/index.html")
    return summary


if __name__ == "__main__":
    run()
