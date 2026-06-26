"""
Zero-Gradient — Phase A, Stage 3: minimal nano-MoE with content routing (controller OFF)
=========================================================================================

Goal of this stage (per 阶段A设计.md): build a MoE where experts are on the CRITICAL PATH
and routing is CONTENT-CORRELATED, then check whether experts SPECIALIZE. The controller is
OFF: every routed expert is updated with the SAME local rule. Only the local learner runs.

Design (locked):
  - thin backbone : embedding + FROZEN reservoir causal attention (NO trainable backbone FFN)
                    -> the experts ARE the only FFN capacity (so experts are necessary)
  - MoE FFN       : N experts (small FFN each); token MUST pass its routed experts
  - router        : EMA-prototype nearest-neighbor (content-correlated, zero-gradient):
                    key = (h @ P) normalized ; top-k nearest expert prototypes (cosine);
                    prototypes c_e updated by EMA toward assigned keys (online k-means)
  - local learner : closed-form deeply-supervised update from the readout head (no autograd)

Specialization monitors (the point of this stage):
  - per-expert TOPIC distribution + purity  (do experts capture distinct topics?)
  - routing entropy over time               (collapse check)
  - per-expert local loss / activation
  - prototype drift trajectory + expert usage

Reuses corpus / utilities from zerograd_nano. NO autograd anywhere (grad globally disabled).
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from zerograd_nano import SEED, DEVICE, DTYPE, init_, build_corpus, Config as DataCfg

torch.set_grad_enabled(False)
HERE = Path(__file__).parent


@dataclass
class MoECfg:
    d_model: int = 64
    d_key: int = 32
    n_experts: int = 16
    k_route: int = 2                 # top-k experts per token (forward)
    seq_len: int = 12
    steps: int = 1200
    batch_size: int = 64
    lr_head: float = 0.1
    lr_expert: float = 0.1
    lr_embed: float = 0.05
    proto_ema: float = 0.05          # eta for online-k-means prototype update
    capacity_factor: float = 2.0     # per-expert per-batch capacity = factor * B*k/N (load mgmt, anti-collapse)
    eval_every: int = 50
    eval_batches: int = 16
    seed: int = 0                    # run seed (varies init + batch order -> across-run variance)
    # ---- Stage 4: controller v3 (budget allocation over touched experts) ----
    routing_mode: str = "off"        # off | uniform | random | fixed_topk | importance | v5
    k_update: int = 8                # update budget: how many touched experts get FULL update per step
    soft_floor: float = 0.15         # non-selected experts get this fraction of lr
    ema_rho: float = 0.9
    lam_cov: float = 1.0             # coverage (deficit = routed-traffic minus update-count)
    lam_lev: float = 0.8             # leverage (||dW||)
    lam_learn: float = 0.8           # learnability (local-error drop)
    lam_act: float = 0.2
    lam_cost: float = 0.2
    # ---- v5 layered scheduler: deficit-primary (zero-noise) + bounded value tie-breaker ----
    lam_cov_base: float = 1.0        # effective lam_cov(N) = base*log(N) -> coverage gets HARDER with N
    lam_val: float = 0.3             # value tie-breaker is tanh-bounded to [-lam_val, lam_val] -> never dominates


class ZeroGradMoE:
    def __init__(self, cfg: MoECfg, V: int):
        self.cfg = cfg; self.V = V; d = cfg.d_model
        sd = SEED + cfg.seed*100003; ctr = [0]
        def w(*shape, scale):
            ctr[0] += 1; g = torch.Generator().manual_seed(sd + ctr[0])
            return (torch.randn(*shape, generator=g, dtype=DTYPE, device=DEVICE) * scale).contiguous()
        self.E = w(V, d, scale=0.02); self.pos = w(cfg.seq_len, d, scale=0.02)
        self.Wq = w(d, d, scale=1/math.sqrt(d)); self.Wk = w(d, d, scale=1/math.sqrt(d))  # frozen
        self.P = w(d, cfg.d_key, scale=1/math.sqrt(d))                                      # frozen routing proj
        self.C = w(cfg.n_experts, cfg.d_key, scale=1.0)                                     # prototypes (EMA)
        self.C = self.C / (self.C.norm(dim=-1, keepdim=True) + 1e-6)
        self.We = [w(d, d, scale=0.05) for _ in range(cfg.n_experts)]          # experts = the FFN
        self.be = [torch.zeros(d, dtype=DTYPE, device=DEVICE) for _ in range(cfg.n_experts)]
        self.Hf = w(d, V, scale=1/math.sqrt(d)); self.bf = torch.zeros(V, dtype=DTYPE, device=DEVICE)
        self.num_params = V*d + cfg.seq_len*d + 2*d*d + d*cfg.d_key + cfg.n_experts*(d*d+d) + d*V

    def context(self, x):
        T = x.shape[1]; emb = self.E[x] + self.pos[:T].unsqueeze(0)
        q = emb @ self.Wq; k = emb @ self.Wk; sc = (q @ k.transpose(1, 2)) / math.sqrt(emb.shape[-1])
        mask = torch.triu(torch.ones(T, T, device=emb.device), diagonal=1).bool()
        att = torch.softmax(sc.masked_fill(mask, float("-inf")), -1)
        h_last = (emb + att @ emb)[:, -1]                   # thin backbone: attention only, no FFN
        # routing rep: first non-pad token embedding (the topic-bearing context token; left-padded)
        first = (x == 0).long().sum(1).clamp(max=T-1)
        rrep = emb[torch.arange(x.shape[0]), first]
        return h_last, att[:, -1, :], rrep

    def key(self, r):
        k = r @ self.P; return k / (k.norm(dim=-1, keepdim=True) + 1e-6)

    def route(self, rrep):
        kk = self.key(rrep); Cn = self.C / (self.C.norm(dim=-1, keepdim=True) + 1e-6)
        sim = kk @ Cn.T                                     # cosine [B,N]
        order = torch.argsort(sim, dim=-1, descending=True) # preference order per token
        B, N = sim.shape; k = self.cfg.k_route
        cap = int(math.ceil(B * k / N * self.cfg.capacity_factor))   # capacity-limited (anti-collapse + reroute)
        load = torch.zeros(N, dtype=torch.long); assign = torch.full((B, k), 0, dtype=torch.long, device=rrep.device)
        ordl = order.tolist()
        for b in range(B):
            slot = 0
            for e in ordl[b]:
                if load[e] < cap:
                    assign[b, slot] = e; load[e] += 1; slot += 1
                    if slot == k: break
            if slot < k:                                    # all preferred full -> least loaded
                for e in torch.argsort(load).tolist():
                    assign[b, slot] = e; load[e] += 1; slot += 1
                    if slot == k: break
        return assign, kk

    def moe(self, h, topk):
        out = torch.zeros_like(h)
        for e in torch.unique(topk).tolist():
            m = (topk == e).any(dim=1)
            out[m] = out[m] + torch.relu(h[m] @ self.We[e] + self.be[e])
        return h + out / self.cfg.k_route                   # experts on the critical path (thin residual h)

    def logits(self, hm): return hm @ self.Hf + self.bf


class Ctrl:
    """Stage-4 controller v3: allocates the per-step UPDATE budget over touched experts."""
    def __init__(self, cfg: MoECfg):
        n = cfg.n_experts; self.cfg = cfg
        self.m_err = torch.full((n,), float("nan")); self.m_lev = torch.zeros(n); self.m_act = torch.zeros(n)
        self.learn = torch.zeros(n); self.usage = torch.zeros(n); self.upd = torch.zeros(n); self.s_ema = torch.zeros(n)
        self.upd_count = torch.zeros(n); self.touch_count = torch.zeros(n)   # cumulative coverage counters (zero-noise)
    @staticmethod
    def _z(x, mask):
        v = x[mask]
        if v.numel() < 2 or float(v.std()) < 1e-6: return torch.zeros_like(x)
        return ((x - v.mean()) / (v.std() + 1e-6)).clamp(-4, 4)
    def observe(self, err, lev, act, touched):
        b = self.cfg.ema_rho
        for e, val in err.items():
            prev = self.m_err[e]; self.m_err[e] = val if torch.isnan(prev) else b*float(prev)+(1-b)*val
            if not torch.isnan(prev): self.learn[e] = max(0.0, float(prev)-float(self.m_err[e]))
        for e, val in lev.items(): self.m_lev[e] = b*self.m_lev[e]+(1-b)*val
        for e, val in act.items(): self.m_act[e] = b*self.m_act[e]+(1-b)*val
        self.usage = 0.95*self.usage + 0.05*touched
    def select(self, cand, step):
        c = self.cfg; mask = torch.zeros(c.n_experts, dtype=torch.bool); mask[cand] = True
        deficit = self.usage - self.upd; cost = torch.ones(c.n_experts)
        s = (c.lam_cov*self._z(deficit, mask) + c.lam_lev*self._z(self.m_lev, mask)
             + c.lam_learn*self._z(self.learn, mask) + c.lam_act*self._z(self.m_act, mask)
             - c.lam_cost*self._z(cost, mask))
        self.s_ema = c.ema_rho*self.s_ema + (1-c.ema_rho)*s
        k = min(c.k_update, int(mask.sum()))
        self.touch_count[cand] += 1.0
        if c.routing_mode in ("off", "uniform"): sel = set(cand.tolist())
        elif c.routing_mode == "random":
            g = torch.Generator().manual_seed(SEED + c.seed*131 + step)
            sel = set(cand[torch.randperm(cand.numel(), generator=g)[:k]].tolist())
        elif c.routing_mode == "fixed_topk": sel = set(cand[:k].tolist())
        elif c.routing_mode == "v5":
            # LAYERED: deficit-primary (exact counting -> zero noise) + tanh-bounded value tie-breaker.
            # value can NEVER overpower coverage because it is clamped to [-lam_val, lam_val] while the
            # coverage term scales as lam_cov_base*log(N).
            deficit = self.touch_count - self.upd_count                  # owed training, exact
            dv = deficit[mask]; dnorm = (deficit - dv.min())/((dv.max()-dv.min()).clamp_min(1e-6))
            value = torch.tanh(self._z(self.m_lev, mask) + self._z(self.learn, mask)) * c.lam_val
            sv = (c.lam_cov_base * math.log(max(2, c.n_experts))) * dnorm + value
            order = torch.argsort(sv[cand], descending=True); sel = set(cand[order[:k]].tolist())
        else:                                                            # "importance" = v4 multi-signal sum
            order = torch.argsort(self.s_ema[cand], descending=True); sel = set(cand[order[:k]].tolist())
        selm = torch.zeros(c.n_experts)
        if sel: selm[list(sel)] = 1.0
        self.upd = 0.95*self.upd + 0.05*selm
        if sel: self.upd_count[list(sel)] += 1.0                         # cumulative update counts (coverage metric)
        return sel


def topic_report(model: ZeroGradMoE, data, n_topics, batches=24):
    """Expert x topic assignment matrix + purity + routing entropy (on val)."""
    cfg = model.cfg; M = torch.zeros(cfg.n_experts, n_topics)
    Xv, Tv = data["Xval"], data["Tval"]
    for i in range(0, min(len(Xv), batches*cfg.batch_size), cfg.batch_size):
        xb = Xv[i:i+cfg.batch_size]; tb = Tv[i:i+cfg.batch_size]
        h, _, rr = model.context(xb); topk, _ = model.route(rr)
        for j in range(cfg.k_route):
            for e in range(cfg.n_experts):
                m = topk[:, j] == e
                if m.any():
                    for t in range(n_topics): M[e, t] += int((tb[m] == t).sum())
    usage = M.sum(1)
    purity = float((M.max(1).values[usage > 0] / usage[usage > 0]).mean()) if (usage > 0).any() else 0.0
    pe = usage / (usage.sum() + 1e-9); pe = pe[pe > 0]
    route_entropy = float(-(pe * pe.log()).sum())           # nats; max = ln(N_active)
    return M, purity, route_entropy, usage


def evaluate(model, X, Y, cfg):
    nll, corr, n = 0.0, 0, 0
    for i in range(0, min(len(X), cfg.eval_batches*cfg.batch_size), cfg.batch_size):
        xb, yb = X[i:i+cfg.batch_size], Y[i:i+cfg.batch_size]
        h, _, rr = model.context(xb); topk, _ = model.route(rr); hm = model.moe(h, topk)
        lg = model.logits(hm); lp = lg - lg.logsumexp(-1, keepdim=True); B = xb.shape[0]
        nll += float(-lp[torch.arange(B), yb].sum()); n += B; corr += int((lg.argmax(-1) == yb).sum())
    return math.exp(nll/n), corr/n


def train(cfg: MoECfg, data):
    model = ZeroGradMoE(cfg, len(data["vocab"]))
    Xtr, Ytr = data["Xtr"], data["Ytr"]; K = data["n_topics"]
    # prototype init from real data keys (spread -> better clustering, avoids cold-start collapse)
    _, _, rr0 = model.context(Xtr[:cfg.batch_size]); k0 = model.key(rr0)
    g = torch.Generator().manual_seed(SEED); idx = torch.randperm(k0.shape[0], generator=g)[:cfg.n_experts]
    model.C = k0[idx].clone()
    ctrl = Ctrl(cfg)
    t0 = time.time(); curve = []; ent_curve = []; drift_curve = []; anomalies = []
    expert_loss_ema = torch.zeros(cfg.n_experts); expert_act_ema = torch.zeros(cfg.n_experts)
    for step in range(cfg.steps):
        gg = torch.Generator().manual_seed(SEED+1000+step+cfg.seed*9176); ix = torch.randint(0, len(Xtr), (cfg.batch_size,), generator=gg)
        xb, yb = Xtr[ix], Ytr[ix]; B = xb.shape[0]
        h, att, rr = model.context(xb); topk, kk = model.route(rr)
        # forward MoE
        cache = {}; moe = torch.zeros_like(h)
        for e in torch.unique(topk).tolist():
            m = (topk == e).any(dim=1); inp = h[m]; z = inp @ model.We[e] + model.be[e]; he = torch.relu(z)
            moe[m] = moe[m] + he; cache[e] = (inp, z, m, he)
        hm = h + moe / cfg.k_route
        # readout head closed-form update + one-step signal to experts
        lg = hm @ model.Hf + model.bf; lp = lg - lg.logsumexp(-1, keepdim=True)
        loss = float(-lp[torch.arange(B), yb].mean())
        if not math.isfinite(loss): anomalies.append({"step": step, "kind": "nan_loss"})
        p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
        model.Hf -= cfg.lr_head * (hm.T @ p); model.bf -= cfg.lr_head * p.sum(0)
        dh = p @ model.Hf.T
        # experts: controller v3 allocates the UPDATE budget over touched experts (Stage 4).
        # mode "off" = Stage-3 behavior (all updated equally). Signals computed for ALL touched.
        cand = torch.tensor(sorted(cache.keys())); err, lev, act = {}, {}, {}; delta = {}
        touched = torch.zeros(cfg.n_experts)
        for e, (inp, z, m, he) in cache.items():
            mm = int(m.sum()); dz = (dh[m] / cfg.k_route) * (z > 0).to(DTYPE)
            dWe = inp.T @ dz; dbe = dz.sum(0); delta[e] = (dWe, dbe)
            comp = (he @ model.Hf + model.bf); clp = comp - comp.logsumexp(-1, keepdim=True)
            err[e] = float(-clp[torch.arange(mm), yb[m]].mean()); lev[e] = float(dWe.norm())
            act[e] = float(he.norm()/math.sqrt(max(1, mm))); touched[e] = 1.0
            expert_loss_ema[e] = 0.9*expert_loss_ema[e] + 0.1*err[e]
            expert_act_ema[e] = 0.9*expert_act_ema[e] + 0.1*act[e]
        ctrl.observe(err, lev, act, touched); sel = ctrl.select(cand, step)
        for e, (dWe, dbe) in delta.items():
            scale = cfg.lr_expert if (cfg.routing_mode in ("off", "uniform") or e in sel) else cfg.lr_expert*cfg.soft_floor
            model.We[e] -= scale * dWe; model.be[e] -= scale * dbe
        # embedding update (approx local credit assignment through frozen attention)
        w = att.unsqueeze(-1) * dh.unsqueeze(1)
        model.E.index_add_(0, xb.reshape(-1), (-cfg.lr_embed*w).reshape(-1, cfg.d_model))
        model.pos[:xb.shape[1]] -= cfg.lr_embed * w.sum(0)
        model.E.index_add_(0, xb[:, -1], -cfg.lr_embed*dh); model.pos[xb.shape[1]-1] -= cfg.lr_embed*dh.sum(0)
        # prototype EMA update (online k-means) -> induces specialization
        drift = 0.0
        for e in torch.unique(topk).tolist():
            m = (topk == e).any(dim=1); new = (1-cfg.proto_ema)*model.C[e] + cfg.proto_ema*kk[m].mean(0)
            drift += float((new - model.C[e]).norm()); model.C[e] = new
        if step % cfg.eval_every == 0 or step == cfg.steps-1:
            ppl, acc = evaluate(model, data["Xval"], data["Yval"], cfg)
            _, purity, ent, _ = topic_report(model, data, K, batches=8)
            curve.append(dict(step=step, val_ppl=round(ppl, 3), val_acc=round(acc, 4),
                              monitor_loss=round(loss, 4), topic_purity=round(purity, 4)))
            ent_curve.append(dict(step=step, route_entropy=round(ent, 4)))
            drift_curve.append(dict(step=step, drift=round(drift, 4)))
            print(f"  step {step:4d} ppl={ppl:6.3f} acc={acc:.3f} purity={purity:.3f} route_H={ent:.3f}/{math.log(cfg.n_experts):.2f}")
    M, purity, ent, usage = topic_report(model, data, K)
    return dict(model=model, curve=curve, ent_curve=ent_curve, drift_curve=drift_curve,
                expert_topic=M.tolist(), final_purity=round(purity, 4), final_route_entropy=round(ent, 4),
                max_entropy=round(math.log(cfg.n_experts), 4), usage=usage.tolist(),
                expert_loss=[round(x, 4) for x in expert_loss_ema.tolist()],
                expert_act=[round(x, 4) for x in expert_act_ema.tolist()],
                anomalies=anomalies, wall_s=round(time.time()-t0, 2), final=curve[-1])


def run():
    cfg = MoECfg(); data = build_corpus(DataCfg())
    print(f"corpus vocab={len(data['vocab'])} topics={data['n_topics']} "
          f"unigram={data['unigram_ppl']:.2f} bigram={data['bigram_ppl']:.2f}")
    print(f"nano-MoE: d={cfg.d_model} experts={cfg.n_experts} top-{cfg.k_route} (controller OFF, equal updates)")
    r = train(cfg, data)
    # specialization verdict
    purity_random = 1.0/data["n_topics"]                    # purity if routing were topic-blind
    specialized = r["final_purity"] > purity_random + 0.15 and r["final_route_entropy"] > 0.5*r["max_entropy"]
    out = dict(stage="A-stage3", config=asdict(cfg), n_topics=data["n_topics"],
               unigram_ppl=round(data["unigram_ppl"], 3), bigram_ppl=round(data["bigram_ppl"], 3),
               curve=r["curve"], ent_curve=r["ent_curve"], drift_curve=r["drift_curve"],
               expert_topic=r["expert_topic"], usage=r["usage"], expert_loss=r["expert_loss"],
               expert_act=r["expert_act"], final_purity=r["final_purity"],
               final_route_entropy=r["final_route_entropy"], max_entropy=r["max_entropy"],
               purity_random_baseline=round(purity_random, 4), specialized=bool(specialized),
               final=r["final"], wall_s=r["wall_s"], anomalies=r["anomalies"],
               param_count=r["model"].num_params, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    (HERE/"runs").mkdir(exist_ok=True); (HERE/"runs"/"stageA.json").write_text(json.dumps(out, indent=2))
    (HERE/"dashboard").mkdir(exist_ok=True); (HERE/"dashboard"/"dataA.js").write_text("window.STAGE_A="+json.dumps(out)+";")
    print(f"\n  final: ppl={r['final']['val_ppl']} acc={r['final']['val_acc']} "
          f"purity={r['final_purity']} (random={purity_random:.3f}) route_H={r['final_route_entropy']}/{r['max_entropy']}")
    print(f"  SPECIALIZED = {specialized}  (purity>{purity_random+0.15:.2f} and entropy not collapsed)")
    return out


def run4(seeds=5, k_bar=4, k_grid=(2, 3, 4, 6, 10)):
    """Stage 4: controller v3 ablation on the specialized MoE (variance over seeds)."""
    data = build_corpus(DataCfg()); modes = ["uniform", "random", "fixed_topk", "importance"]
    bar = {m: [] for m in modes}
    for sd in range(seeds):
        for m in modes:
            bar[m].append(train(MoECfg(routing_mode=m, k_update=k_bar, steps=1000, eval_every=1000, seed=sd), data)["final"]["val_ppl"])
    sweep = []
    for k in k_grid:
        ri = [train(MoECfg(routing_mode="importance", k_update=k, steps=1000, eval_every=1000, seed=sd), data)["final"]["val_ppl"] for sd in range(seeds)]
        rr = [train(MoECfg(routing_mode="random", k_update=k, steps=1000, eval_every=1000, seed=sd), data)["final"]["val_ppl"] for sd in range(seeds)]
        g = np.array(rr) - np.array(ri)
        sweep.append(dict(k=k, imp=round(float(np.mean(ri)), 3), rnd=round(float(np.mean(rr)), 3),
                          gap=round(float(g.mean()), 3), gap_std=round(float(g.std()), 3), pos=int((g > 0).sum()), n=seeds))
    gp = np.array(bar["random"]) - np.array(bar["importance"])
    out = dict(stage="A-stage4", seeds=seeds, k_bar=k_bar, n_experts=MoECfg().n_experts, bigram=round(data["bigram_ppl"], 3),
               bar={m: dict(mean=round(float(np.mean(bar[m])), 3), std=round(float(np.std(bar[m])), 3),
                            vals=[round(x, 3) for x in bar[m]]) for m in modes},
               headline_gap=round(float(gp.mean()), 3), headline_gap_std=round(float(gp.std()), 3),
               headline_pos=int((gp > 0).sum()), sweep=sweep, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    (HERE/"runs").mkdir(exist_ok=True); (HERE/"runs"/"stage4.json").write_text(json.dumps(out, indent=2))
    (HERE/"dashboard").mkdir(exist_ok=True); (HERE/"dashboard"/"data4.js").write_text("window.STAGE_4="+json.dumps(out)+";")
    print("\n==== STAGE 4 (controller v3 on specialized MoE) ====")
    for m in modes: print(f"  {m:11s} ppl = {out['bar'][m]['mean']:.3f} ± {out['bar'][m]['std']:.3f}")
    print(f"  headline gap (random - importance) @k={k_bar}: +{out['headline_gap']:.3f} ± {out['headline_gap_std']:.3f}  ({out['headline_pos']}/{seeds} seeds positive)")
    print("  gap vs budget (tighter budget -> bigger controller advantage, = the 4B regime):")
    for s in sweep: print(f"    k={s['k']:2d}: gap=+{s['gap']:.3f} ± {s['gap_std']:.3f}  ({s['pos']}/{s['n']})")
    return out


if __name__ == "__main__":
    import sys
    run4() if (len(sys.argv) > 1 and sys.argv[1] == "stage4") else run()
