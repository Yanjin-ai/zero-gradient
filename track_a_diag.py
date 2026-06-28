"""Track A diagnostic: is A1 DEAD or SAVABLE?

A1 (train Wq/Wk from block-0 readout signal dh0) failed the assoc-recall probe: attention barely
moved (dW~1e-4), acc stayed at chance. Question: is the attention signal STRUCTURALLY starved
(attention contributes ~nothing to h, so no gradient can reach it) or is it a WEAK-READOUT problem
(dh0 too small but rescuable with a stronger derivation / dedicated attention head)?

Instruments attn_update over a real assoc-recall run and reports:
  ||dh0||            : block-0 readout signal magnitude (per-token mean L2)
  r_attn            : ||(att@emb)_L|| / ||emb_L||  -> how much of h_L comes from attention vs the
                      direct embedding term (if <<1, emb_L dominates -> attention is a small perturbation)
  ||dWq||+||dWk||    : raw attention gradient norm
  rel_step          : lr*||dW|| / ||Wq||  -> relative weight movement per step (if ~1e-4, ~frozen)

Run:  python3 track_a_diag.py
"""
import torch, math
import kaggle_zerograd_moe as Z
import track_a_probe as P

logs = []
_orig = Z.ZeroGradMoE.attn_update

def _patched(self, cache, dh0, lr):
    emb = cache["emb"].float()
    embL = emb[:, -1, :]
    attoutL = (cache["att"].float() @ emb)[:, -1, :]              # attention contribution at last pos
    r_attn = float((attoutL.norm(dim=-1) / (embL.norm(dim=-1) + 1e-9)).mean())
    dh_mag = float(dh0.float().norm(dim=-1).mean())
    Wq_norm = float(self.Wq.float().norm())
    dnorm = _orig(self, cache, dh0, lr)                            # applies the real update, returns ||dWq||(+||dWk||)
    logs.append((dh_mag, r_attn, dnorm, lr, Wq_norm))
    return dnorm

Z.ZeroGradMoE.attn_update = _patched

if __name__ == "__main__":
    print(f"task=assoc-recall vocab={P.VOCAB} seq_len={P.SEQ} pairs={P.NPAIR}")
    res, acc, uni = P.run(True, attn_lr_scale=1.0)               # main lr 0.1 * 1.0
    n = len(logs)
    def col(i): return [x[i] for x in logs]
    import statistics as st
    dh = col(0); r = col(1); dW = col(2); lr = col(3); Wq = col(4)
    rel = [lr[i]*dW[i]/(Wq[i]+1e-9) for i in range(n)]
    print(f"  steps logged={n}  final val_ppl={res['best_ppl']:.3f}  val_acc={acc*100:.1f}%  unigram={uni:.1f}")
    print(f"  ||dh0||      mean={st.mean(dh):.4g}  min={min(dh):.4g}  max={max(dh):.4g}")
    print(f"  r_attn       mean={st.mean(r):.4g}  min={min(r):.4g}  max={max(r):.4g}   (||att@emb||/||emb_L|| at L)")
    print(f"  ||dWq+dWk||  mean={st.mean(dW):.4g}  max={max(dW):.4g}")
    print(f"  rel_step     mean={st.mean(rel):.4g}  max={max(rel):.4g}   (lr*||dW||/||Wq|| per step)")
    print(f"  ||Wq||       start={Wq[0]:.4g}  end={Wq[-1]:.4g}  (drift={abs(Wq[-1]-Wq[0])/Wq[0]*100:.2f}%)")
