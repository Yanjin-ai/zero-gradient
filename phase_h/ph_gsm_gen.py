"""Phase H / v3.0 -- G2b (STRETCH): GENERATIVE multi-step reasoning with a causal LM (ph_base.PhCausalLM).

Unlike G2 (classification of the final answer), this GENERATES the answer token-by-token and scores exact
match -- the honest shape of real GSM8K. RESEARCH-ONLY, isolated (no ZeroBP import).

  --source synth : char-level k-step arithmetic "d op d op d = result\\n" (CPU smoke; learnable).
  --source gsm8k : real GSM8K via HuggingFace `datasets` (char-level). HONEST NOTE: a small from-scratch
                   char LM is a weak fit for NL GSM8K -- expect low exact-match without scale/pretraining.
                   This scaffolds the machinery + contrast (ZeroBP installs multi-step at NO depth); it is
                   NOT a claim that a tiny model solves GSM8K.

Writes runs/ph_gsmgen_run_summary.json. Run (smoke): python3 phase_h/ph_gsm_gen.py --source synth --n_steps 2
"""
import os, sys, json, random, argparse, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ph_base import PhConfig, PhCausalLM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CharTok:                                                    # id 0 = PAD; '\n' = EOS
    def __init__(self, alphabet):
        self.itos = ["<pad>"] + list(alphabet); self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.eos = self.stoi["\n"]
    def enc(self, s): return [self.stoi[c] for c in s if c in self.stoi]
    def dec(self, ids): return "".join(self.itos[i] for i in ids if 0 < i < len(self.itos))
    @property
    def V(self): return len(self.itos)


def synth(n, seed, n_steps):                                     # returns list[(prompt_str, answer_str, full_str)]
    rng = random.Random(seed); out = []
    for _ in range(n):
        ds = [rng.randrange(10) for _ in range(n_steps + 1)]; ops = [rng.choice("+-") for _ in range(n_steps)]
        expr = str(ds[0]); v = ds[0]
        for i in range(n_steps):
            expr += ops[i] + str(ds[i+1]); v = v + ds[i+1] if ops[i] == "+" else v - ds[i+1]
        out.append((expr + "=", str(v), f"{expr}={v}\n"))
    return out


def load_gsm8k(split, limit):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")[split]; out = []
    for i, r in enumerate(ds):
        if i >= limit: break
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        prompt = r["question"].strip() + "\n#### "
        out.append((prompt, ans, prompt + ans + "\n"))
    return out


def batchify(tok, examples, seq_len):
    X = []
    for _, _, full in examples:
        ids = tok.enc(full)[:seq_len]; ids += [0] * (seq_len - len(ids)); X.append(ids)
    return torch.tensor(X, dtype=torch.long)


@torch.no_grad()
def exact_match(model, tok, examples, seq_len, max_new=8):
    model.eval(); c = 0; n = min(len(examples), 500)
    for prompt, ans, _ in examples[:n]:
        pid = tok.enc(prompt)[:seq_len]
        idx = torch.tensor([pid], dtype=torch.long, device=DEVICE)
        out = model.generate(idx, max_new=max_new, eos=tok.eos)[0].tolist()[len(pid):]
        gen = tok.dec(out).split("\n")[0].strip()
        c += int(gen == ans)
    return c / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synth", "gsm8k"], default="synth")
    ap.add_argument("--n_steps", type=int, default=2); ap.add_argument("--seq_len", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=6000); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "ph_gsmgen_run_summary.json"))
    a = ap.parse_args(); torch.manual_seed(a.seed); t0 = time.time()

    if a.source == "synth":
        tok = CharTok("0123456789+-=\n"); tr = synth(a.limit, a.seed+2, a.n_steps); va = synth(1500, a.seed+3, a.n_steps)
        seq = a.seq_len or (max(len(f) for _, _, f in tr) + 1)
    else:
        tr = load_gsm8k("train", a.limit); va = load_gsm8k("test", 500)
        alpha = "".join(sorted(set(ch for _, _, f in tr for ch in f)))
        if "\n" not in alpha: alpha += "\n"
        tok = CharTok(alpha); seq = a.seq_len or 320
    Xtr = batchify(tok, tr, seq).to(DEVICE)
    cfg = PhConfig(vocab=tok.V, seq_len=seq, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, dropout=0.1)
    model = PhCausalLM(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"\n==== Phase H G2b: generative {a.source}  {cfg.n_layers}L x {cfg.n_heads}H d={cfg.d_model}  "
          f"params={model.n_params()/1e6:.2f}M  V={tok.V} seq={seq} train={len(tr)} dev={DEVICE.type} ====")
    g = torch.Generator().manual_seed(a.seed)
    for step in range(1, a.train_steps+1):
        model.train(); ix = torch.randint(0, len(Xtr), (a.batch,), generator=g); xb = Xtr[ix]
        inp, tgt = xb[:, :-1], xb[:, 1:]
        _, loss = model(inp, tgt); opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, a.train_steps//8) == 0 or step == 1:
            print(f"  step {step:>5}/{a.train_steps}  loss {loss.item():.4f}  EM {exact_match(model, tok, va, seq)*100:.1f}%")
    em = exact_match(model, tok, va, seq)
    summ = {"phase": "H", "gate": "G2b", "task": "generative-reasoning", "source": a.source,
            "n_steps": (a.n_steps if a.source == "synth" else None), "params_M": round(model.n_params()/1e6, 3),
            "config": {"layers": a.layers, "heads": a.heads, "d_model": a.d_model, "seq_len": seq, "train_steps": a.train_steps},
            "exact_match": round(em, 4), "zerobp_contrast": "ZeroBP installs multi-step at NO BP depth",
            "note": ("real GSM8K is a stretch for a small char LM" if a.source == "gsm8k" else "synthetic generative arithmetic"),
            "wall_s": round(time.time()-t0, 1), "device": DEVICE.type}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f: json.dump(summ, f, indent=2)
    print(f"\n  Phase H G2b exact-match = {em*100:.1f}%  (source={a.source})\n  summary -> {a.out}")
    return summ


if __name__ == "__main__":
    main()
