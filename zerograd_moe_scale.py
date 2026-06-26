"""
Zero-Gradient — Phase A / Stage 5 (scale step 3.1): nano -> small, STACKED MoE
==============================================================================
Same mechanisms as the validated single-layer MoE (EMA-prototype content routing,
experts-as-FFN on critical path, controller v3 with coverage/deficit, deeply-supervised
LOCAL updates, NO autograd), now scaled along the structure-preserving axes:
  - wider hidden dim
  - multiple stacked MoE blocks (each: route -> experts -> local head; own controller)
  - more experts per block, small top-k

Question: do (a) expert specialization and (b) the importance>random gap HOLD / GROW as the
"same kind of system" gets bigger? Same corpus + gates as stage 3/4.

Each MoE block is trained by its OWN local next-token head (deeply-supervised) -> no signal
crosses blocks (stays local / zero-gradient). Routing rep = topic-bearing context (first
non-pad token embedding), shared across blocks; each block has its OWN prototypes + experts +
controller, so each block clusters independently.
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from zerograd_nano import SEED, DEVICE, DTYPE, build_corpus, Config as DataCfg
from zerograd_moe import Ctrl

torch.set_grad_enabled(False)
HERE = Path(__file__).parent


@dataclass
class ScaleCfg:
    tag: str = "nano"
    d_model: int = 64
    d_key: int = 32
    n_layers: int = 1                # number of stacked MoE blocks
    n_experts: int = 16              # experts per block
    k_route: int = 2
    k_update: int = 4                # update budget per block per step
    capacity_factor: float = 2.0
    seq_len: int = 12
    steps: int = 1000
    batch_size: int = 64
    lr_head: float = 0.1
    lr_expert: float = 0.1
    lr_embed: float = 0.05
    proto_ema: float = 0.05
    seed: int = 0
    route_rep: str = "first"          # "first" = topic-marker (synthetic) | "mean" = context mean (natural text)
    routing_mode: str = "importance"
    soft_floor: float = 0.15
    ema_rho: float = 0.9
    lam_cov: float = 1.0; lam_lev: float = 0.8; lam_learn: float = 0.8; lam_act: float = 0.2; lam_cost: float = 0.2
    eval_batches: int = 16


class StackedMoE:
    def __init__(self, cfg: ScaleCfg, V: int):
        self.cfg = cfg; self.V = V; d = cfg.d_model; sd = SEED + cfg.seed*100003; ctr = [0]
        def w(*shape, scale):
            ctr[0] += 1; g = torch.Generator().manual_seed(sd+ctr[0])
            return (torch.randn(*shape, generator=g, dtype=DTYPE, device=DEVICE)*scale).contiguous()
        self.E = w(V, d, scale=0.02); self.pos = w(cfg.seq_len, d, scale=0.02)
        self.Wq = w(d, d, scale=1/math.sqrt(d)); self.Wk = w(d, d, scale=1/math.sqrt(d))
        self.P = w(d, cfg.d_key, scale=1/math.sqrt(d))
        self.C = [(lambda c: c/(c.norm(dim=-1, keepdim=True)+1e-6))(w(cfg.n_experts, cfg.d_key, scale=1.0)) for _ in range(cfg.n_layers)]
        self.We = [[w(d, d, scale=0.05) for _ in range(cfg.n_experts)] for _ in range(cfg.n_layers)]
        self.be = [[torch.zeros(d, dtype=DTYPE, device=DEVICE) for _ in range(cfg.n_experts)] for _ in range(cfg.n_layers)]
        self.Hb = [w(d, V, scale=1/math.sqrt(d)) for _ in range(cfg.n_layers)]
        self.bH = [torch.zeros(V, dtype=DTYPE, device=DEVICE) for _ in range(cfg.n_layers)]
        self.num_params = V*d*(1+cfg.n_layers) + 2*d*d + d*cfg.d_key + cfg.n_layers*cfg.n_experts*(d*d+d)

    def context(self, x):
        T = x.shape[1]; emb = self.E[x] + self.pos[:T].unsqueeze(0)
        q = emb @ self.Wq; k = emb @ self.Wk; sc = (q @ k.transpose(1, 2))/math.sqrt(emb.shape[-1])
        m = torch.triu(torch.ones(T, T, device=emb.device), diagonal=1).bool()
        att = torch.softmax(sc.masked_fill(m, float("-inf")), -1)
        if self.cfg.route_rep == "mean":                         # natural text: no markers -> context mean
            pm = (x != 0).float().unsqueeze(-1); rrep = (emb*pm).sum(1)/(pm.sum(1)+1e-6)
        else:                                                    # synthetic: topic-marker (first non-pad) token
            first = (x == 0).long().sum(1).clamp(max=T-1); rrep = emb[torch.arange(x.shape[0]), first]
        return (emb + att @ emb)[:, -1], att[:, -1, :], rrep

    def route(self, rrep, Cl):
        kk = rrep @ self.P; kk = kk/(kk.norm(dim=-1, keepdim=True)+1e-6)
        Cn = Cl/(Cl.norm(dim=-1, keepdim=True)+1e-6); sim = kk @ Cn.T
        order = torch.argsort(sim, dim=-1, descending=True); B, N = sim.shape; k = self.cfg.k_route
        cap = int(math.ceil(B*k/N*self.cfg.capacity_factor)); load = torch.zeros(N, dtype=torch.long)
        assign = torch.zeros(B, k, dtype=torch.long, device=rrep.device); ordl = order.tolist()
        for b in range(B):
            slot = 0
            for e in ordl[b]:
                if load[e] < cap:
                    assign[b, slot] = e; load[e] += 1; slot += 1
                    if slot == k: break
            if slot < k:
                for e in torch.argsort(load).tolist():
                    assign[b, slot] = e; load[e] += 1; slot += 1
                    if slot == k: break
        return assign, kk

    def block(self, hin, assign, l):
        out = torch.zeros_like(hin)
        for e in torch.unique(assign).tolist():
            mm = (assign == e).any(dim=1); out[mm] = out[mm] + torch.relu(hin[mm] @ self.We[l][e] + self.be[l][e])
        return hin + out/self.cfg.k_route

    def forward(self, x):
        h, att, rrep = self.context(x); assigns = []
        for l in range(self.cfg.n_layers):
            a, _ = self.route(rrep, self.C[l]); assigns.append(a); h = self.block(h, a, l)
        return h, att, rrep, assigns

    def logits(self, h): return h @ self.Hb[-1] + self.bH[-1]


def purity_entropy(model, data, layer, batches=10):
    cfg = model.cfg; M = torch.zeros(cfg.n_experts, data["n_topics"]); Xv, Tv = data["Xval"], data["Tval"]
    for i in range(0, min(len(Xv), batches*cfg.batch_size), cfg.batch_size):
        xb = Xv[i:i+cfg.batch_size]; tb = Tv[i:i+cfg.batch_size]
        _, _, rr = model.context(xb); a, _ = model.route(rr, model.C[layer])
        for j in range(cfg.k_route):
            for e in range(cfg.n_experts):
                mm = a[:, j] == e
                if mm.any():
                    for t in range(data["n_topics"]): M[e, t] += int((tb[mm] == t).sum())
    u = M.sum(1); pur = float((M.max(1).values[u > 0]/u[u > 0]).mean()) if (u > 0).any() else 0.0
    pe = u/(u.sum()+1e-9); pe = pe[pe > 0]; return pur, float(-(pe*pe.log()).sum())


def evaluate(model, X, Y, cfg):
    nll, n = 0.0, 0
    for i in range(0, min(len(X), cfg.eval_batches*cfg.batch_size), cfg.batch_size):
        xb, yb = X[i:i+cfg.batch_size], Y[i:i+cfg.batch_size]; h, _, _, _ = model.forward(xb)
        lg = model.logits(h); lp = lg - lg.logsumexp(-1, keepdim=True); B = xb.shape[0]
        nll += float(-lp[torch.arange(B), yb].sum()); n += B
    return math.exp(nll/n)


def train(cfg: ScaleCfg, data):
    model = StackedMoE(cfg, len(data["vocab"])); Xtr, Ytr = data["Xtr"], data["Ytr"]
    _, _, rr0 = model.context(Xtr[:cfg.batch_size]); kk0 = rr0 @ model.P; kk0 = kk0/(kk0.norm(dim=-1, keepdim=True)+1e-6)
    g = torch.Generator().manual_seed(SEED); idx = torch.randperm(kk0.shape[0], generator=g)[:cfg.n_experts]
    for l in range(cfg.n_layers): model.C[l] = kk0[idx].clone()
    ctrls = [Ctrl(_as_moecfg(cfg)) for _ in range(cfg.n_layers)]
    t0 = time.time()
    for step in range(cfg.steps):
        gg = torch.Generator().manual_seed(SEED+1000+step+cfg.seed*9176); ix = torch.randint(0, len(Xtr), (cfg.batch_size,), generator=gg)
        xb, yb = Xtr[ix], Ytr[ix]; B = xb.shape[0]
        h, att, rrep = model.context(xb); hin = h
        for l in range(cfg.n_layers):
            assign, kk = model.route(rrep, model.C[l])
            cache = {}; moe = torch.zeros_like(hin)
            for e in torch.unique(assign).tolist():
                mm = (assign == e).any(dim=1); inp = hin[mm]; z = inp @ model.We[l][e] + model.be[l][e]; he = torch.relu(z)
                moe[mm] = moe[mm] + he; cache[e] = (inp, z, mm, he)
            hl = hin + moe/cfg.k_route
            lg = hl @ model.Hb[l] + model.bH[l]; p = torch.softmax(lg, -1); p[torch.arange(B), yb] -= 1.0; p /= B
            model.Hb[l] -= cfg.lr_head*(hl.T @ p); model.bH[l] -= cfg.lr_head*p.sum(0); dh = p @ model.Hb[l].T
            cand = torch.tensor(sorted(cache.keys())); err, lev, act, delta = {}, {}, {}, {}; touched = torch.zeros(cfg.n_experts)
            for e, (inp, z, mm, he) in cache.items():
                m_ = int(mm.sum()); dz = (dh[mm]/cfg.k_route)*(z > 0).to(DTYPE); dWe = inp.T @ dz
                delta[e] = (dWe, dz.sum(0)); comp = he @ model.Hb[l] + model.bH[l]; clp = comp - comp.logsumexp(-1, keepdim=True)
                err[e] = float(-clp[torch.arange(m_), yb[mm]].mean()); lev[e] = float(dWe.norm()); act[e] = float(he.norm()/math.sqrt(max(1, m_))); touched[e] = 1.0
            ctrls[l].observe(err, lev, act, touched); sel = ctrls[l].select(cand, step)
            for e, (dWe, dbe) in delta.items():
                scale = cfg.lr_expert if (cfg.routing_mode in ("off", "uniform") or e in sel) else cfg.lr_expert*cfg.soft_floor
                model.We[l][e] -= scale*dWe; model.be[l][e] -= scale*dbe
            for e in torch.unique(assign).tolist():
                mm = (assign == e).any(dim=1); model.C[l][e] = (1-cfg.proto_ema)*model.C[l][e] + cfg.proto_ema*kk[mm].mean(0)
            if l == 0:                                           # embedding update from first block (local, approx)
                w_ = att.unsqueeze(-1)*dh.unsqueeze(1)
                model.E.index_add_(0, xb.reshape(-1), (-cfg.lr_embed*w_).reshape(-1, cfg.d_model)); model.pos[:xb.shape[1]] -= cfg.lr_embed*w_.sum(0)
                model.E.index_add_(0, xb[:, -1], -cfg.lr_embed*dh); model.pos[xb.shape[1]-1] -= cfg.lr_embed*dh.sum(0)  # direct last-token path
            hin = hl
    ppl = evaluate(model, data["Xval"], data["Yval"], cfg)
    purs = [purity_entropy(model, data, l) for l in range(cfg.n_layers)] if "Tval" in data else []
    return dict(model=model, ppl=round(ppl, 3),
                purity=[round(p, 3) for p, _ in purs], entropy=[round(e, 3) for _, e in purs],
                max_entropy=round(math.log(cfg.n_experts), 3), wall_s=round(time.time()-t0, 2), params=model.num_params,
                upd_count=[c.upd_count.tolist() for c in ctrls], touch_count=[c.touch_count.tolist() for c in ctrls])


def _as_moecfg(cfg: ScaleCfg):
    from zerograd_moe import MoECfg
    # v4: coverage weight auto-scales with expert count (more experts -> coverage need dominates).
    # Validated at 3L/64e: lam_cov ~ (N/16)^2 restores importance>random (gap -0.22 -> +0.04, 3/3).
    lam_cov_eff = cfg.lam_cov * (cfg.n_experts / 16.0) ** 2
    return MoECfg(n_experts=cfg.n_experts, k_update=cfg.k_update, routing_mode=cfg.routing_mode,
                  soft_floor=cfg.soft_floor, ema_rho=cfg.ema_rho, lam_cov=lam_cov_eff, lam_lev=cfg.lam_lev,
                  lam_learn=cfg.lam_learn, lam_act=cfg.lam_act, lam_cost=cfg.lam_cost, seed=cfg.seed)


def gap(cfg_kw, data, seeds=4):
    ri, rr = [], []
    for sd in range(seeds):
        ri.append(train(ScaleCfg(routing_mode="importance", seed=sd, **cfg_kw), data)["ppl"])
        rr.append(train(ScaleCfg(routing_mode="random", seed=sd, **cfg_kw), data)["ppl"])
    g = np.array(rr)-np.array(ri)
    return dict(importance=round(float(np.mean(ri)), 3), random=round(float(np.mean(rr)), 3),
                gap=round(float(g.mean()), 3), gap_std=round(float(g.std()), 3), pos=int((g > 0).sum()), n=seeds)


def run():
    data = build_corpus(DataCfg())
    print(f"corpus topics={data['n_topics']} bigram={data['bigram_ppl']:.2f}")
    configs = {
        "nano  (1L, d64,  16e)": dict(d_model=64, n_layers=1, n_experts=16, k_update=4),
        "small (2L, d128, 32e)": dict(d_model=128, n_layers=2, n_experts=32, k_update=6),
        "small (3L, d128, 64e)": dict(d_model=128, n_layers=3, n_experts=64, k_update=10),
    }
    rows = []
    for name, kw in configs.items():
        r0 = train(ScaleCfg(routing_mode="importance", seed=0, **kw), data)
        g = gap(kw, data, seeds=4)
        row = dict(name=name, params=r0["params"], ppl=g["importance"], purity_l0=r0["purity"][0],
                   purity_mean=round(float(np.mean(r0["purity"])), 3), entropy_l0=r0["entropy"][0],
                   max_entropy=r0["max_entropy"], gap=g["gap"], gap_std=g["gap_std"], pos=g["pos"], n=g["n"], wall_s=r0["wall_s"])
        rows.append(row)
        print(f"  {name}: params={r0['params']/1e3:.0f}K ppl={g['importance']:.2f} purity(L0)={r0['purity'][0]:.3f} "
              f"entropy(L0)={r0['entropy'][0]:.2f}/{r0['max_entropy']:.2f} | gap=+{g['gap']:.3f}±{g['gap_std']:.3f} ({g['pos']}/{g['n']})")
    out = dict(stage="A-scale-3.1", bigram=round(data["bigram_ppl"], 3), rows=rows, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    (HERE/"runs").mkdir(exist_ok=True); (HERE/"runs"/"scale.json").write_text(json.dumps(out, indent=2))
    (HERE/"dashboard").mkdir(exist_ok=True); (HERE/"dashboard"/"dataS.js").write_text("window.STAGE_S="+json.dumps(out)+";")
    return out


if __name__ == "__main__":
    run()
