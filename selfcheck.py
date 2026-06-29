"""Reset self-check for multi-reset small-config experiments (ENGINEERING.md norm #3).

Guards against the load_state_dict aliasing / in-place corruption class of bug: runs the SAME
short Phase E adaptation twice -- once on a freshly-trained base, once AFTER an intervening
zero-BP adapt + reset. With a correct (cloning) reset and no golden-checkpoint corruption, the
two must be bit-identical. Fast (small config, few steps).

Run:  python3 selfcheck.py     (exit 0 = OK)
"""
import sys
from dataclasses import replace
import kaggle_zerograd_moe as Z
import task_nli as T
import phase_e as PE

def main():
    cfg = Z.Config(name="selfcheck", vocab=T.VOCAB, seq_len=T.L, n_layers=2, n_experts=48, k_route=2, k_update=4,
                   steps=400, batch_size=64, lr=0.1, lr_min=0.1, warmup_steps=40, eval_every=10**9, time_limit_s=120)
    Odata = T.lm_data(T.gen_O(4000, Z.SEED), cfg); Sdata = T.lm_data(T.gen_O(4000, Z.SEED+1), cfg)
    Xtr, Ytr = T.gen_cls(3000, Z.SEED+2); Xv, Yv = T.gen_cls(800, Z.SEED+3); d = cfg.d_model
    base = Z.ZeroGradMoE(cfg, T.VOCAB); Z.train(base, Odata, cfg); baseA = base.state_dict()

    PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 200, lr=0.1, bp_top=False, ncls=T.NCLS)
    a1 = T.acc(base, T.train_head(base, Xtr, Ytr, d, T.NCLS), Xv, Yv)
    base.load_state_dict(baseA); Z.train(base, Sdata, replace(cfg, steps=400, freeze_routing_step=0))   # intervening adapt
    PE.run_phase_e(base, baseA, Xtr, Ytr, Odata, cfg, 200, lr=0.1, bp_top=False, ncls=T.NCLS)
    a2 = T.acc(base, T.train_head(base, Xtr, Ytr, d, T.NCLS), Xv, Yv)

    ok = abs(a1 - a2) < 1e-9
    print(f"  fresh={a1*100:.4f}%  after-intervening={a2*100:.4f}%  -> [{'OK' if ok else 'FAIL: golden corrupted / reset leaks'}]")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
