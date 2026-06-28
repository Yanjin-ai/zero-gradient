"""Unified experiment ledger + monitor for local and remote (Kaggle) runs.

Ledger = runs/experiments.jsonl (one JSON object per line). The orchestrator appends Kaggle events;
local scripts (or `seed`) append local results. `show` prints a summary of both for analysis.

  python3 experiments.py seed     # record the local results obtained so far (idempotent-ish; appends)
  python3 experiments.py show     # print a summary table of local + remote runs
  python3 experiments.py log '<json>'   # append an arbitrary entry
"""
import json, sys
from pathlib import Path
from datetime import datetime, timezone

LEDGER = Path(__file__).parent/"runs"/"experiments.jsonl"
LEDGER.parent.mkdir(exist_ok=True)

def log(entry):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **entry}
    with open(LEDGER, "a") as f: f.write(json.dumps(entry, default=float) + "\n")
    return entry

LOCAL_SEED = [
    {"site": "local", "track": "A", "exp": "A1 attention probe (assoc-recall)", "status": "refuted",
     "metrics": {"frozen_acc": 0.027, "train_acc_best": 0.028, "verdict": "chance; attention not learned"}},
    {"site": "local", "track": "A", "exp": "A1 dead-or-savable diagnostic", "status": "refuted",
     "metrics": {"r_attn": 0.38, "rel_step": 7e-7, "acc_at_lr1e5": 0.025, "verdict": "direction insufficient, not scale"}},
    {"site": "local", "track": "C.1", "exp": "checkpoint save/reload (ckpt_verify)", "status": "pass",
     "metrics": {"reload_ppl": 6.2506, "saved_best": 6.251, "deterministic": True}},
    {"site": "local", "track": "C.1", "exp": "post-train pipeline (c1_posttrain, topic)", "status": "pass 5/5",
     "metrics": {"task_acc": 1.0, "majority": 0.25, "zero_forgetting": True}},
    {"site": "local", "track": "C.1", "exp": "compositional sentiment (c1_sentiment, in-domain)", "status": "pass 5/5",
     "metrics": {"linear_acc": 0.528, "mlp_acc": 1.0, "atomic_not": 1.0, "atomic_polarity": 0.999,
                 "zero_forgetting": True, "note": "XOR needs MLP head"}},
    {"site": "local", "track": "C.1", "exp": "c1_4b plumbing (small BPE, off-domain)", "status": "plumbing-ok",
     "metrics": {"linear_acc": 0.493, "mlp_acc": 0.57, "majority": 0.503, "zero_forgetting": True,
                 "note": "off-domain base -> ~chance; confirms 4B must use real WikiText words"}},
    {"site": "local", "track": "submission", "exp": "default path sanity", "status": "pass 6/7",
     "metrics": {"final_ppl": 6.251, "deterministic": True, "zero_autograd": True}},
]

def seed():
    for e in LOCAL_SEED: log(e)
    print(f"seeded {len(LOCAL_SEED)} local entries -> {LEDGER}")

def show():
    if not LEDGER.exists(): print("no ledger yet"); return
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    print(f"\n  {'SITE':6} {'TRACK':10} {'STATUS':14} EXP / EVENT")
    print("  " + "-"*78)
    for r in rows:
        site = r.get("site", "?"); track = r.get("track") or r.get("stage", "-")
        st = r.get("status") or r.get("event", "-")
        what = r.get("exp") or r.get("event", "")
        m = r.get("metrics", {})
        key = ""
        if isinstance(m, dict):
            for k in ("mlp_acc", "mlp_head_acc", "task_acc", "final_ppl", "wikitext_test_ppl", "lm_ppl_after", "verdict"):
                if k in m: key = f"{k}={m[k]}"; break
        print(f"  {site:6} {str(track):10} {str(st):14} {str(what)[:40]:40} {key}")
    print(f"\n  {len(rows)} entries in {LEDGER}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "seed": seed()
    elif cmd == "log": log(json.loads(sys.argv[2]))
    else: show()
