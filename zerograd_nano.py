"""
Zero-Gradient Learning — nano vertical slice  (v2: locked controller + experiment system)
==========================================================================================

Validates the project's core bet under the competition's hard rule (NO autograd, NO
loss.backward(), NO optimizer): a DETERMINISTIC, GRADIENT-FREE importance controller that
allocates training resources across modules can beat random / uniform allocation.

This file is the structure-preserving COMPRESSION of the 4B design (see CONFIG). The SAME
code scales nano -> 4B by changing only numbers. It runs all baselines, computes the three
metric families (model-quality / algorithm / system), writes per-experiment metadata + a
cross-run HISTORY, and emits the dashboard data.

--------------------------------------------------------------------------------------------
LOCKED v1 SPEC  (frozen for the "prove importance > random, then scale" phase)
--------------------------------------------------------------------------------------------
MODEL (frozen / trainable / controller separation):
  frozen structure   : token embedding shape, positional table shape, fixed reservoir
                       causal attention (Wq,Wk are NOT trained -> no gradient needed through them)
  trainable modules  : backbone block linears (per-block deeply-supervised heads),
                       sparse expert linears, final readout head, embedding rows (approx update)
  deterministic ctrl : reads per-module scalar stats -> importance score -> resource budget
                       (it does NOT update theta; it updates "who gets trained how hard")

CONTROLLER MATH (locked):
  online stats per module l:  a_l=activation strength, e_l=local error, v_l=leverage(||dW||),
                              c_l=cost
  EMA:        m_x_l = beta*m_x_l + (1-beta)*x_l        (x in {a,e,v})
  learnab.:   learn_l = max(0, m_e_l[prev] - m_e_l)     (local error still dropping?)
  normalize:  phi = z-score across the CANDIDATE modules this step
  score:      s_l = lam_v*phi(m_v) + lam_learn*phi(learn) + lam_a*phi(m_a)
                    - lam_c*phi(c) - load_balance*phi(usage)
  EMA score:  s_ema = rho*s_ema + (1-rho)*s
  budget:     b_l = softmax(s_ema/tau) over candidates           (logged)
  select:     top-k by s_ema (hard)                              (the routing decision)

RESOURCE INTERFACE (locked — score controls ALL of):
  selected (top-k) : update_scale = lr_full ; local_iters = iter_high ; cache_tier = 'full'
  non-selected     : update_scale = lr_full*soft_floor ; local_iters = 1 ; cache_tier = 'summary'
  (baselines: random=k random; uniform=all at budget/n; fixed_topk=first k; importance=top-k by score)

GATES (must hold before scaling): deterministic; zero-autograd; ppl drops; ppl<unigram AND
  <bigram; no NaN / no majority-collapse; importance < random (HEADLINE); importance <= uniform.
"""

from __future__ import annotations
import os, json, math, time, random
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch

# ---- determinism + zero-autograd (gate) ----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try: torch.use_deterministic_algorithms(True)
except Exception as e: print("[warn] deterministic:", e)
torch.set_grad_enabled(False)
DEVICE = torch.device("cpu"); DTYPE = torch.float32
HERE = Path(__file__).parent

# Human-readable design descriptor (the 7 slots) — shipped to the dashboard each run.
DESIGN = {
    "input":       "batch of (context[T], next-token y); carried controller state S",
    "state":       "per-expert EMA of {local error, activation, leverage}, importance s_ema, usage, selection mask",
    "per_step":    "fwd states -> backbone local update -> route experts -> final head update -> controller score/select -> budgeted expert updates -> embedding update",
    "signal":      "deeply-supervised closed-form CE (delta=p-onehot); experts: final-head one-step local delta. NO autograd.",
    "update":      "theta -= scale * hand-derived local delta; scale set by controller budget",
    "resource":    "controller score -> {update_scale, local_iters, cache_tier} per module (top-k full / rest summary)",
    "eval":        "val/train perplexity, next-token acc, vs unigram & bigram; importance-vs-{random,uniform,fixed_topk}",
}


@dataclass
class Config:
    name: str = "nano"
    d_model: int = 64
    n_backbone: int = 3
    n_experts: int = 32                # sparsity ratio toward 4B (sparse:backbone ~ 10:1)
    k_fwd: int = 4
    k_update: int = 10
    iter_high: int = 2                 # local iterations granted to selected experts
    seq_len: int = 12
    vocab_cap: int = 256
    n_sentences: int = 6000
    steps: int = 1000
    batch_size: int = 64
    lr_backbone: float = 0.3
    lr_expert: float = 0.3
    lr_head: float = 0.3
    lr_embed: float = 0.15
    expert_init: float = 0.01          # small init -> experts start ~neutral, grow only where useful
    eval_every: int = 25               # shortened review interval
    eval_batches: int = 16
    time_limit_s: float = 90.0
    # controller (locked math)
    ema_rho: float = 0.9               # beta for stats AND rho for score EMA
    tau: float = 1.0
    # v3 controller: coverage(deficit) is a first-class term (matters for sparse experts), but B
    # showed the decisive importance>random win needs expert SPECIALIZATION -> arrives in phase A.
    lam_cov: float = 1.0               # coverage = routed-traffic minus update-count (under-served experts)
    lam_lev: float = 0.8               # leverage  (||dW|| ~ expected final-loss reduction)
    lam_learn: float = 0.8             # learnability (local error still dropping)
    lam_act: float = 0.2               # activation strength
    lam_cost: float = 0.2              # cost penalty
    load_balance: float = 0.0          # (legacy; coverage now handled by lam_cov)
    soft_floor: float = 0.15
    routing_mode: str = "importance"   # importance | random | uniform | fixed_topk

    def param_count(self) -> int:
        V, d = self.vocab_cap, self.d_model
        return (V*d + self.seq_len*d + self.n_backbone*(d*d+d) + self.n_experts*(d*d+d)
                + self.n_backbone*(d*V+V))      # readout = last block's deeply-supervised head


P4B = Config(name="4B", d_model=1024, n_backbone=8, n_experts=3700, seq_len=512, vocab_cap=32000)


# ======================================================================================
# DATA — structured synthetic corpus (strong learnable structure, offline, reproducible)
# ======================================================================================
def build_corpus(cfg: Config):
    # CONTEXT-DEPENDENT structure: K topics; the same subject maps to DIFFERENT verbs per topic.
    # The topic marker sits at the sentence start -> bigram (last token only) cannot predict the
    # verb (it averages over topics); a context model + topic-specialized experts can. This leaves
    # real "beyond-bigram" residual for the expert bank / router to capture.
    rng=random.Random(SEED); K=4
    topics=[f"<t{t}>" for t in range(K)]
    subj=["robot","cat","river","engine","child","wizard","planet","farmer"]
    verb=["builds","watches","carries","breaks","finds","guards","paints","feeds"]
    adj =["bright","silent","heavy","ancient","tiny","golden","frozen","wild"]
    obj =["bridge","garden","machine","forest","tower","signal","harvest","stone"]
    # per-topic subject->verb map (distinct across topics) => verb needs topic AND subject
    tmap={t:{s:verb[(subj.index(s)+3*t)%len(verb)] for s in subj} for t in range(K)}
    sents=[]
    for _ in range(cfg.n_sentences):
        t=rng.randrange(K); s=rng.choice(subj)
        sents.append([topics[t],"the",rng.choice(adj),s,tmap[t][s],"the",rng.choice(adj),rng.choice(obj)])
    from collections import Counter
    allw=[w for s in sents for w in s]
    vocab=["<pad>","<unk>"]+[w for w,_ in Counter(allw).most_common(cfg.vocab_cap-2)]
    stoi={w:i for i,w in enumerate(vocab)}; PAD=stoi["<pad>"]
    enc=[[stoi.get(w,1) for w in s] for s in sents]
    X,Y=[],[]
    for s in enc:                                   # per-sentence left-padded windows (topic always in context)
        for i in range(2,len(s)):
            ctx=s[max(0,i-cfg.seq_len):i]; ctx=[PAD]*(cfg.seq_len-len(ctx))+ctx
            X.append(ctx); Y.append(s[i])
    X=torch.tensor(X,dtype=torch.long); Y=torch.tensor(Y,dtype=torch.long)
    perm=torch.randperm(len(X),generator=torch.Generator().manual_seed(SEED)); X,Y=X[perm],Y[perm]
    nval=max(256,len(X)//10)
    d=dict(Xtr=X[nval:].to(DEVICE),Ytr=Y[nval:].to(DEVICE),Xval=X[:nval].to(DEVICE),
           Yval=Y[:nval].to(DEVICE),vocab=vocab,stoi=stoi)
    V=len(vocab)
    # unigram baseline
    cnt=torch.bincount(d["Ytr"],minlength=V).float()+1e-6; p=cnt/cnt.sum()
    d["unigram_ppl"]=float(torch.exp(-(p[d["Yval"]].log()).mean()))
    d["majority_freq"]=float(cnt.max()/cnt.sum())
    # bigram baseline (last context token -> next), unigram backoff
    bg=torch.ones(V,V)*1e-3
    last=d["Xtr"][:,-1]
    for a,b in zip(last.tolist(),d["Ytr"].tolist()): bg[a,b]+=1.0
    bg=bg/bg.sum(1,keepdim=True)
    lp=bg[d["Xval"][:,-1],d["Yval"]].log()
    d["bigram_ppl"]=float(torch.exp(-lp.mean()))
    return d


# ======================================================================================
# MODEL — frozen attention reservoir + trainable backbone/experts/heads; exposes states
# ======================================================================================
def init_(*shape, scale):
    return (torch.randn(*shape,generator=torch.Generator().manual_seed(SEED+sum(shape)),
                        dtype=DTYPE,device=DEVICE)*scale).contiguous()

class ZeroGradLM:
    def __init__(self, cfg: Config, V: int):
        self.cfg=cfg; self.V=V; d=cfg.d_model
        self.E=init_(V,d,scale=0.02); self.pos=init_(cfg.seq_len,d,scale=0.02)
        self.Wq=init_(d,d,scale=1/math.sqrt(d)); self.Wk=init_(d,d,scale=1/math.sqrt(d))   # frozen
        self.Wb=[init_(d,d,scale=1/math.sqrt(d)) for _ in range(cfg.n_backbone)]
        self.bb=[torch.zeros(d,dtype=DTYPE,device=DEVICE) for _ in range(cfg.n_backbone)]
        self.We=[init_(d,d,scale=cfg.expert_init) for _ in range(cfg.n_experts)]   # start ~neutral
        self.be=[torch.zeros(d,dtype=DTYPE,device=DEVICE) for _ in range(cfg.n_experts)]
        self.Ekey=init_(cfg.n_experts,d,scale=1/math.sqrt(d))                                # frozen router
        self.Hb=[init_(d,V,scale=1/math.sqrt(d)) for _ in range(cfg.n_backbone)]
        self.bHb=[torch.zeros(V,dtype=DTYPE,device=DEVICE) for _ in range(cfg.n_backbone)]
        self.num_params=cfg.param_count()

    def context(self,x):
        T=x.shape[1]; h=self.E[x]+self.pos[:T].unsqueeze(0)
        q=h@self.Wq; k=h@self.Wk; sc=(q@k.transpose(1,2))/math.sqrt(h.shape[-1])
        mask=torch.triu(torch.ones(T,T,device=h.device),diagonal=1).bool()
        att=torch.softmax(sc.masked_fill(mask,float("-inf")),-1)
        mixed=h+att@h                                            # residual around frozen attention
        return mixed[:,-1], att[:,-1,:]

    def logits(self,h): return h@self.Hb[-1]+self.bHb[-1]   # unified readout = last block's head
    def route(self,h): return torch.topk(h@self.Ekey.T,self.cfg.k_fwd,dim=-1).indices

    def forward_states(self,x):
        h0,att=self.context(x); st={"att_last":att,"x":x}
        h=h0; bbin,bbz,bbout=[],[],[]
        for l in range(self.cfg.n_backbone):
            z=h@self.Wb[l]+self.bb[l]; a=torch.relu(z)
            bbin.append(h); bbz.append(z); bbout.append(a+h); h=a+h
        st.update(bb_in=bbin,bb_z=bbz,bb_out=bbout,h_back=h,topk=self.route(h)); return st


def ce_delta(out, y, Whead, bhead):
    """Closed-form softmax-CE: returns loss, dout(grad wrt out), dWhead, dbhead. No autograd."""
    B=y.shape[0]; logits=out@Whead+bhead
    logp=logits-logits.logsumexp(-1,keepdim=True); loss=float(-logp[torch.arange(B),y].mean())
    p=torch.softmax(logits,-1); p[torch.arange(B),y]-=1.0; p/=B
    return loss, p@Whead.T, out.T@p, p.sum(0)


# ======================================================================================
# CONTROLLER — locked deterministic gradient-free resource allocator
# ======================================================================================
class Controller:
    def __init__(self, cfg: Config):
        n=cfg.n_experts; self.cfg=cfg
        self.m_err=torch.full((n,),float("nan")); self.m_act=torch.zeros(n); self.m_lev=torch.zeros(n)
        self.learn=torch.zeros(n); self.usage=torch.zeros(n); self.update_ema=torch.zeros(n)
        self.s=torch.zeros(n); self.s_ema=torch.zeros(n); self.budget=torch.zeros(n)

    @staticmethod
    def _z(x,mask):
        v=x[mask]
        if v.numel()<2 or float(v.std())<1e-6: return torch.zeros_like(x)
        return ((x-v.mean())/(v.std()+1e-6)).clamp(-4,4)         # clamp -> numerically stable score

    def observe(self, err, act, lev, touched):
        b=self.cfg.ema_rho
        for e,val in err.items():
            prev=self.m_err[e]
            self.m_err[e]=val if torch.isnan(prev) else b*float(prev)+(1-b)*val
            if not torch.isnan(prev): self.learn[e]=max(0.0,float(prev)-float(self.m_err[e]))
        for e,val in act.items(): self.m_act[e]=b*self.m_act[e]+(1-b)*val
        for e,val in lev.items(): self.m_lev[e]=b*self.m_lev[e]+(1-b)*val
        self.usage=0.95*self.usage+0.05*touched

    def score_select(self, cand, step):
        c=self.cfg; mask=torch.zeros(c.n_experts,dtype=torch.bool); mask[cand]=True
        cost=torch.ones(c.n_experts)
        deficit=self.usage-self.update_ema                       # routed-but-under-updated -> needs coverage
        s=(c.lam_cov*self._z(deficit,mask)+c.lam_lev*self._z(self.m_lev,mask)
           +c.lam_learn*self._z(self.learn,mask)+c.lam_act*self._z(self.m_act,mask)
           -c.lam_cost*self._z(cost,mask)-c.load_balance*self._z(self.usage,mask))
        self.s=s; self.s_ema=c.ema_rho*self.s_ema+(1-c.ema_rho)*s
        sc=self.s_ema.clone(); sc[~mask]=float("-inf"); self.budget=torch.softmax(sc/c.tau,0)
        k=min(c.k_update,int(mask.sum()))
        if c.routing_mode=="random":
            g=torch.Generator().manual_seed(SEED+step)
            sel=set(cand[torch.randperm(cand.numel(),generator=g)[:k]].tolist())
        elif c.routing_mode=="uniform":   sel=set(cand.tolist())
        elif c.routing_mode=="fixed_topk":sel=set(cand[:k].tolist())
        else:
            order=torch.argsort(self.s_ema[cand],descending=True); sel=set(cand[order[:k]].tolist())
        selm=torch.zeros(c.n_experts);
        if sel: selm[list(sel)]=1.0
        self.update_ema=0.95*self.update_ema+0.05*selm           # track coverage
        return sel


# ======================================================================================
# TRAIN + EVAL
# ======================================================================================
def evaluate(model, X, Y, cfg, use_experts=True):
    nll,corr,n=0.0,0,0
    for i in range(0,min(len(X),cfg.eval_batches*cfg.batch_size),cfg.batch_size):
        xb,yb=X[i:i+cfg.batch_size],Y[i:i+cfg.batch_size]; st=model.forward_states(xb)
        h=st["h_back"]; topk=st["topk"]; B=xb.shape[0]
        rep=h
        if use_experts:
            cand=torch.unique(topk).flatten(); mix=torch.zeros_like(h)
            for e in cand.tolist():
                m=(topk==e).any(dim=1); mix[m]+=torch.relu(h[m]@model.We[e]+model.be[e])
            rep=h+mix/cfg.k_fwd
        logits=model.logits(rep); logp=logits-logits.logsumexp(-1,keepdim=True)
        nll+=float(-logp[torch.arange(B),yb].sum()); n+=B; corr+=int((logits.argmax(-1)==yb).sum())
    return math.exp(nll/n), corr/n


def train(cfg: Config, data):
    model=ZeroGradLM(cfg,len(data["vocab"])); ctrl=Controller(cfg)
    Xtr,Ytr=data["Xtr"],data["Ytr"]; t0=time.time(); anomalies=[]
    unit_hist={f"E{e}":[] for e in range(cfg.n_experts)}; sel_hist={f"E{e}":[] for e in range(cfg.n_experts)}
    curve=[]; cycles=0
    for step in range(cfg.steps):
        if time.time()-t0>cfg.time_limit_s: break
        cycles+=1
        g=torch.Generator().manual_seed(SEED+1000+step); idx=torch.randint(0,len(Xtr),(cfg.batch_size,),generator=g)
        xb,yb=Xtr[idx],Ytr[idx]; B=xb.shape[0]; st=model.forward_states(xb)

        # backbone deeply-supervised (per-block heads) — always trained (quality core)
        bb_loss=[]
        for l in range(cfg.n_backbone):
            L,dout,dH,dbH=ce_delta(st["bb_out"][l],yb,model.Hb[l],model.bHb[l])
            dz=dout*(st["bb_z"][l]>0).to(DTYPE)
            model.Wb[l]-=cfg.lr_backbone*(st["bb_in"][l].T@dz); model.bb[l]-=cfg.lr_backbone*dz.sum(0)
            model.Hb[l]-=cfg.lr_head*dH; model.bHb[l]-=cfg.lr_head*dbH; bb_loss.append(L)

        # experts: route, combine, final-head update, controller-budgeted local updates
        h=st["h_back"]; topk=st["topk"]; cand=torch.unique(topk).flatten()
        route_mask={e:(topk==e).any(dim=1) for e in cand.tolist()}
        mix=torch.zeros_like(h)
        for e in cand.tolist(): mix[route_mask[e]]+=torch.relu(h[route_mask[e]]@model.We[e]+model.be[e])
        h_final=h+mix/cfg.k_fwd
        fL,dhf,dHf,dbf=ce_delta(h_final,yb,model.Hb[-1],model.bHb[-1])
        model.Hb[-1]-=cfg.lr_head*dHf; model.bHb[-1]-=cfg.lr_head*dbf

        err,act,lev,state={},{},{},{}; touched=torch.zeros(cfg.n_experts)
        for e in cand.tolist():
            m=route_mask[e]; inp=h[m]; z=inp@model.We[e]+model.be[e]; he=torch.relu(z); mm=int(m.sum())
            comp=(h[m]+he)@model.Hb[-1]+model.bHb[-1]; clp=comp-comp.logsumexp(-1,keepdim=True)
            err[e]=float(-clp[torch.arange(mm),yb[m]].mean()); act[e]=float(he.norm()/math.sqrt(mm))
            dz=(dhf[m]/cfg.k_fwd)*(z>0).to(DTYPE); dWe=inp.T@dz
            lev[e]=float(dWe.norm()); state[e]=(inp,z,m); touched[e]=1.0
        ctrl.observe(err,act,lev,touched); selected=ctrl.score_select(cand,step)

        upd_norm={}
        for e in cand.tolist():
            inp,z,m=state[e]
            if cfg.routing_mode=="uniform": scale=cfg.lr_expert*(cfg.k_update/max(1,cand.numel())); iters=1
            elif e in selected: scale=cfg.lr_expert; iters=cfg.iter_high
            else: scale=cfg.lr_expert*cfg.soft_floor; iters=1
            tot=0.0
            for _ in range(iters):                                  # selected -> more local iterations
                z=inp@model.We[e]+model.be[e]; he=torch.relu(z)
                comp=(h[m]+he)@model.Hb[-1]+model.bHb[-1]
                p=torch.softmax(comp,-1); p[torch.arange(int(m.sum())),yb[m]]-=1.0; p/=int(m.sum())
                dz=(p@model.Hb[-1].T)*(z>0).to(DTYPE); dWe=inp.T@dz
                model.We[e]-=scale*dWe; model.be[e]-=scale*dz.sum(0); tot+=float(dWe.norm())*scale
            upd_norm[e]=tot

        # embedding update (approx local credit assignment through frozen attention)
        w=st["att_last"].unsqueeze(-1)*dhf.unsqueeze(1)
        model.E.index_add_(0,xb.reshape(-1),(-cfg.lr_embed*w).reshape(-1,cfg.d_model))
        model.pos[:xb.shape[1]]-=cfg.lr_embed*w.sum(0)
        model.E.index_add_(0,xb[:,-1],-cfg.lr_embed*dhf); model.pos[xb.shape[1]-1]-=cfg.lr_embed*dhf.sum(0)

        monitor=float(np.mean(bb_loss+[fL]))
        if not math.isfinite(monitor): anomalies.append({"step":step,"kind":"nan_loss","detail":"monitor not finite"})
        if step%5==0 or step==cfg.steps-1:
            for e in range(cfg.n_experts):
                unit_hist[f"E{e}"].append(round(float(ctrl.s_ema[e]),4)); sel_hist[f"E{e}"].append(1 if e in selected else 0)
        if step%cfg.eval_every==0 or step==cfg.steps-1:
            ppl,acc=evaluate(model,data["Xval"],data["Yval"],cfg)
            tppl,_=evaluate(model,data["Xtr"][:cfg.eval_batches*cfg.batch_size],data["Ytr"][:cfg.eval_batches*cfg.batch_size],cfg)
            curve.append(dict(step=step,val_ppl=round(ppl,3),val_acc=round(acc,4),train_ppl=round(tppl,3),monitor_loss=round(monitor,4)))

    bb_ppl,_=evaluate(model,data["Xval"],data["Yval"],cfg,use_experts=False)   # backbone-only diagnostic
    # score stability = mean over experts of std of s_ema over time (lower = more stable)
    stab=float(np.mean([np.std(v[-20:]) for v in unit_hist.values() if len(v)>=2])) if curve else float("nan")
    return dict(curve=curve,unit_hist=unit_hist,sel_hist=sel_hist,anomalies=anomalies,
                wall_s=round(time.time()-t0,2),final=curve[-1] if curve else None,
                backbone_only_ppl=round(bb_ppl,3),
                cycles=cycles,score_stability=round(stab,4),
                peak_mem_mb=(round(torch.cuda.max_memory_allocated()/2**20,1) if torch.cuda.is_available() else None))


# add tiny helper referenced above (kept simple)
ZeroGradLM._eval_h=lambda self,inp,h,m: h[m]


# ======================================================================================
# MAIN — run all baselines, compute metrics, gates, history, dashboard data
# ======================================================================================
def run():
    OUT=HERE/"runs"; OUT.mkdir(exist_ok=True); data=build_corpus(Config())
    print(f"corpus vocab={len(data['vocab'])} train={len(data['Xtr'])} val={len(data['Xval'])} "
          f"unigram={data['unigram_ppl']:.2f} bigram={data['bigram_ppl']:.2f} 4B_params={P4B.param_count()/1e9:.2f}B")
    modes=["importance","random","uniform","fixed_topk"]; res={}
    for m in modes:
        print(f"--- {m} ---"); res[m]=train(Config(routing_mode=m),data)
        f=res[m]["final"]; print(f"    val_ppl={f['val_ppl']}  train_ppl={f['train_ppl']}  acc={f['val_acc']}  wall={res[m]['wall_s']}s  stab={res[m]['score_stability']}")
    rerun=train(Config(routing_mode="importance"),data)
    det_ok=abs(rerun["final"]["val_ppl"]-res["importance"]["final"]["val_ppl"])<1e-6

    imp,rnd,uni,ftk=(res[m]["final"]["val_ppl"] for m in modes)
    acc=res["importance"]["final"]["val_acc"]
    gates={
        "deterministic (re-run identical)":det_ok,
        "zero autograd (grad disabled)":not torch.is_grad_enabled(),
        "ppl decreased":res["importance"]["curve"][0]["val_ppl"]>imp,
        "ppl < unigram":imp<data["unigram_ppl"],
        "ppl < bigram":imp<data["bigram_ppl"],
        "no majority-class collapse":acc>data["majority_freq"]+0.02,
        "no NaN/Inf":len(res["importance"]["anomalies"])==0,
        "importance < random (HEADLINE)":imp<rnd,
        "importance <= uniform":imp<=uni+1e-9,
    }
    cfg=Config()
    # rough FLOPs/step proxy (matmuls touched)
    d,V=cfg.d_model,len(data["vocab"])
    flops=2*cfg.batch_size*(d*d*(cfg.n_backbone+cfg.k_fwd)+d*V*(cfg.n_backbone+1))
    latest=dict(
        config=asdict(cfg), design=DESIGN,
        param_count_nano=cfg.param_count(), param_count_4B=P4B.param_count(),
        unigram_ppl=round(data["unigram_ppl"],3), bigram_ppl=round(data["bigram_ppl"],3),
        majority_freq=round(data["majority_freq"],3),
        approx_flops_per_step=int(flops),
        results={m:res[m] for m in modes},
        metrics=dict(importance_ppl=imp,random_ppl=rnd,uniform_ppl=uni,fixed_topk_ppl=ftk,
                     val_acc=acc, importance_vs_random_gap=round(rnd-imp,3),
                     score_stability=res["importance"]["score_stability"],
                     wall_s=res["importance"]["wall_s"], peak_mem_mb=res["importance"]["peak_mem_mb"]),
        gates=gates, gates_passed=int(sum(gates.values())), gates_total=len(gates),
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (OUT/"run.json").write_text(json.dumps(latest,indent=2))

    # cross-run HISTORY (compact record per invocation)
    hist_path=OUT/"history.json"; history=json.loads(hist_path.read_text()) if hist_path.exists() else []
    history.append(dict(ts=latest["ts"], run=len(history)+1, mode_config=cfg.name,
                        importance_ppl=imp, random_ppl=rnd, uniform_ppl=uni, fixed_topk_ppl=ftk,
                        gap=round(rnd-imp,3), unigram=latest["unigram_ppl"], bigram=latest["bigram_ppl"],
                        acc=acc, gates=f"{latest['gates_passed']}/{latest['gates_total']}",
                        headline_pass=gates["importance < random (HEADLINE)"],
                        score_stability=res["importance"]["score_stability"], wall_s=res["importance"]["wall_s"]))
    hist_path.write_text(json.dumps(history,indent=2))

    dash=HERE/"dashboard"; dash.mkdir(exist_ok=True)
    (dash/"data.js").write_text("window.RUN_DATA="+json.dumps({"latest":latest,"history":history})+";")

    print("\n==== GATES ====")
    for k,v in gates.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  importance={imp}  random={rnd}  uniform={uni}  fixed_topk={ftk}  unigram={latest['unigram_ppl']}  bigram={latest['bigram_ppl']}")
    print(f"  gap(random-importance)={rnd-imp:.3f}  run#{len(history)} logged to history.json")
    return latest


if __name__=="__main__":
    run()
