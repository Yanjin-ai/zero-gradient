"""Phase F / F1-data (strict ZeroBP): does a richer pretraining DISTRIBUTION make relational structure
appear in the representation? Compares NLI zero-shot for a base pretrained on the current RANDOM corpus
vs one pretrained on a RICHER corpus (random LM + CONSISTENT relation-pairs + QA-style) -- all unlabeled
LM, strict ZeroBP (no BP anywhere). Key metric: NLI zero-shot off its ~49% floor?

The only change is the pretraining text distribution (data prep), NOT the algorithm. Reuses task_nli's
vocab + zero-shot head. selfcheck-covered (load_state_dict clones; single base per corpus).

Run:  python3 f1_data.py
"""
import random, torch
import kaggle_zerograd_moe as Z
import task_nli as T

def gen_random(n, seed):                                      # baseline corpus: random well-formed (clauses unrelated)
    return T.gen_O(n, seed)

def gen_richer(n, seed):                                      # F1: random LM + CONSISTENT relation-pairs + QA-style
    rng = random.Random(seed); out = []
    for _ in range(n):
        b = rng.random()
        if b < 0.35:                                          # (a) random LM base (keep current distribution)
            e1, e2 = rng.sample(T.ENT, 2); e3, e4 = rng.sample(T.ENT, 2)
            out += [e1, rng.choice([T.GT, T.LT]), T.THAN, e2, T.SEP, e3, rng.choice([T.GT, T.LT]), T.THAN, e4]
        elif b < 0.8:                                         # (b) CONSISTENT relation pair (premise + logical hypothesis)
            e1, e2 = rng.sample(T.ENT, 2); r1 = rng.choice([T.GT, T.LT]); lab = rng.randrange(3)
            if lab == 0: e3, e4, r2 = e2, e1, T.INV[r1]       # entailment: B inv(r) A
            elif lab == 1: e3, e4, r2 = e2, e1, r1            # contradiction: B r A
            else:
                while True:
                    e3, e4 = rng.sample(T.ENT, 2)
                    if {e3, e4} != {e1, e2}: break            # neutral
                r2 = rng.choice([T.GT, T.LT])
            out += [e1, r1, T.THAN, e2, T.SEP, e3, r2, T.THAN, e4]
        else:                                                 # (c) QA-style: question then a consistent answer (same tokens)
            e1, e2 = rng.sample(T.ENT, 2); r1 = rng.choice([T.GT, T.LT])
            out += [e1, r1, T.THAN, e2, T.SEP, e2, T.INV[r1], T.THAN, e1]   # "A>B ?  B<A"
    return out

def main():
    cfg = Z.Config(name="f1", vocab=T.VOCAB, seq_len=T.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Xv, Yv = T.gen_cls(1500, Z.SEED+3); d = cfg.d_model
    maj = max(float((Yv == c).float().mean()) for c in range(T.NCLS))
    print(f"\n==== Phase F / F1-data: NLI zero-shot vs pretraining distribution  (majority {maj*100:.1f}%) ====")
    print(f"  {'pretraining corpus':34} {'NLI zero-shot':>14}")
    rows = []
    for name, gen in [("random (current baseline)", gen_random), ("richer (consistent pairs + QA)", gen_richer)]:
        base = Z.ZeroGradMoE(cfg, T.VOCAB); Z.train(base, T.lm_data(gen(8000, Z.SEED), cfg), cfg)
        zs = T.acc(base, T.train_head(base, *T.gen_cls(6000, Z.SEED+2), d, T.NCLS), Xv, Yv)
        rows.append((name, zs)); print(f"  {name:34} {zs*100:>13.1f}%")
    base_zs, rich_zs = rows[0][1], rows[1][1]
    print(f"\n  random {base_zs*100:.1f}%  ->  richer {rich_zs*100:.1f}%  (gain {(rich_zs-base_zs)*100:+.1f}pp)")
    print(f"  [{'SIGNAL: richer data lifts NLI zero-shot' if rich_zs > base_zs + 0.05 else 'NO: data alone insufficient -> need F2/F3/H1'}]")
    return rows

if __name__ == "__main__":
    main()
