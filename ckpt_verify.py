"""C.1 prep verification: does best_ckpt.pt save + reload reproduce the model EXACTLY?

This is the hard prerequisite for C.1 post-training (a stable best checkpoint must seed it). We verify
the save/reload path on the small local config (logic transfers to 4B). Train with save_ckpt -> a fresh
model loads best_ckpt.pt -> its eval ppl must match the best checkpoint's recorded ppl bit-for-bit, and
re-running forward must be deterministic.

Run:  python3 ckpt_verify.py
"""
import torch
from pathlib import Path
import kaggle_zerograd_moe as Z

OUT = Path(__file__).parent/"runs"; OUT.mkdir(exist_ok=True)

def main():
    cfg = Z.Config(save_ckpt=True)                              # small config + checkpoint on
    data = Z.build_data(cfg); V = len(data["vocab"])
    model = Z.ZeroGradMoE(cfg, V)
    res = Z.train(model, data, cfg, out_dir=OUT)
    ck = OUT/"best_ckpt.pt"
    assert ck.exists(), "best_ckpt.pt was not written"
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    saved_step, saved_ppl = blob["step"], blob["val_ppl"]

    # reload into a FRESH model (different random init) and re-evaluate
    fresh = Z.ZeroGradMoE(cfg, V)
    pre = Z.evaluate(fresh, data["Xval"], data["Yval"], cfg)    # fresh (random) -> should differ
    fresh.load_state_dict(blob["state"])
    post = Z.evaluate(fresh, data["Xval"], data["Yval"], cfg)   # after load -> should equal saved best
    post2 = Z.evaluate(fresh, data["Xval"], data["Yval"], cfg)  # determinism of reloaded model

    print(f"\n==== C.1 CHECKPOINT VERIFY ====")
    print(f"  saved best:  step={saved_step}  val_ppl={saved_ppl}")
    print(f"  fresh model (random) val_ppl   = {pre:.4f}   (sanity: should be high/unrelated)")
    print(f"  reloaded model val_ppl         = {post:.6f}")
    print(f"  reloaded re-run val_ppl        = {post2:.6f}   (determinism)")
    match = abs(post - saved_ppl) < 1e-3
    det = abs(post - post2) < 1e-9
    print(f"\n  [{'PASS' if match else 'FAIL'}] reload reproduces best checkpoint (|{post:.4f}-{saved_ppl}|<1e-3)")
    print(f"  [{'PASS' if det else 'FAIL'}] reloaded model is deterministic")
    print(f"  [{'PASS' if pre>post else 'WARN'}] reload actually changed the model (random {pre:.2f} -> loaded {post:.2f})")
    print(f"  ckpt file size = {ck.stat().st_size/1024:.1f} KB")
    return match and det

if __name__ == "__main__":
    ok = main()
    print(f"\n  RESULT: {'all checks passed' if ok else 'CHECK FAILED'}")
