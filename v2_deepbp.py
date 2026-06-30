"""Phase G / v2.0 (RESEARCH-ONLY, never touches the submission) -- deeper-BP probe (EXPLORATION, weak prior).

Context: the relational/multi-step wall is locked (ADR-002). v2.0's two most on-point structural levers are
already REFUTED: non-collapsing readout (`v2_readout.py`) and trainable attention in isolation (`v2_attn.py`).
The Phase E task-method matrix only ever ran Mixed-BP at the SHALLOWEST depth (embedding[+attn]+head); it
NEVER let the MoE blocks themselves be BP-trained on NLI/arithmetic. This probe tests the last untested v2.0
rung -- giving BP more DEPTH (top block, then ALL blocks) -- so the path is measured to its end for the paper.

Weak prior (state it honestly): the blocks operate on the ALREADY-collapsed single vector h=(emb+att@emb)[:,-1]
(last-position collapse happens UPSTREAM of every block), and we proved relational structure is not in the
frozen rep. So deeper block-BP = a deeper MLP on a collapsed vector; it cannot recover cross-position alignment.
The honest question is only: does it show ANY signal above the shallow-BP / floor numbers?

Ladder (each measured with a FRESH closed-form head on the shaped frozen rep -- fair "is structure in the rep"):
  zero-shot floor | emb (shallow ref) | emb+top-block | emb+ALL-blocks (deep) | emb+all+attn (max BP)
Run both NLI (3-cls) and 2-step arithmetic (5-cls). Outcome locks ADR either way:
  no rung escapes floor -> "BP depth ALSO insufficient under this backbone"  | any rung lifts small -> "signal small-only, won't transfer 4B".

Strict research branch; does NOT import or modify the submission path. Run: python3 v2_deepbp.py
"""
import torch
import kaggle_zerograd_moe as Z
import phase_e as PE
import task_nli as NLI
import task_arith as AR


def run_task(M, tag):
    cfg = Z.Config(name="v2deep", vocab=M.VOCAB, seq_len=M.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=1000, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=100, eval_every=250, time_limit_s=120)
    Odata = M.lm_data(M.gen_O(8000, Z.SEED), cfg)
    Xtr, Ytr = M.gen_cls(6000, Z.SEED+2); Xv, Yv = M.gen_cls(1500, Z.SEED+3); d = cfg.d_model
    maj = max(float((Yv == c).float().mean()) for c in range(M.NCLS))
    base = Z.ZeroGradMoE(cfg, M.VOCAB); Z.train(base, Odata, cfg); baseA = base.state_dict()
    def fresh(): return M.acc(base, M.train_head(base, Xtr, Ytr, d, M.NCLS), Xv, Yv)   # FRESH head on shaped frozen rep

    # (label, kwargs to run_phase_e); None = no BP (zero-shot floor)
    arms = [
        ("zero-shot floor",        None),
        ("BP emb (shallow ref)",   dict(bp_emb=True, bp_top=False, bp_attn=False, bp_deep=False)),
        ("BP emb+top-block",       dict(bp_emb=True, bp_top=True,  bp_attn=False, bp_deep=False)),
        ("BP emb+ALL-blocks",      dict(bp_emb=True, bp_top=True,  bp_attn=False, bp_deep=True)),
        ("BP emb+all+attn (max)",  dict(bp_emb=True, bp_top=True,  bp_attn=True,  bp_deep=True)),
    ]
    print(f"\n==== Phase G / v2.0 deeper-BP probe: {tag}  (majority {maj*100:.1f}%, chance {100/M.NCLS:.1f}%) ====")
    print(f"  {'BP depth (NLI/arith label CE)':28} {'acc (fresh head)':>18}")
    rows = []
    for label, kw in arms:
        if kw is None:
            base.load_state_dict(baseA); a = fresh()
        else:
            PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 1000, lr=0.1, ncls=M.NCLS, **kw); a = fresh()
        rows.append((label, a)); print(f"  {label:28} {a*100:>17.1f}%")
    floor = rows[0][1]; best = max(rows[1:], key=lambda r: r[1])
    print(f"\n  floor {floor*100:.1f}%  ->  best-depth '{best[0]}' {best[1]*100:.1f}%  (gain {(best[1]-floor)*100:+.1f}pp)")
    return rows


def main():
    nli = run_task(NLI, "NLI (3-class, relational)")
    ar = run_task(AR, "2-step arithmetic (5-class, multi-step)")
    print("\n==== verdict (EXPLORATION, weak prior) ====")
    for tag, rows in [("NLI", nli), ("arith", ar)]:
        floor = rows[0][1]; best = max(rows[1:], key=lambda r: r[1])
        signal = best[1] > floor + 0.05
        print(f"  {tag:6}: floor {floor*100:.1f}% -> deepest-best {best[1]*100:.1f}% "
              f"[{'SMALL SIGNAL (record; check 4B-transfer prior = weak)' if signal else 'NO signal -> BP depth also insufficient under this backbone'}]")
    return nli, ar


if __name__ == "__main__":
    main()
