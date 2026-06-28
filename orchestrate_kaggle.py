"""Kaggle orchestrator for the C.1 4B pipeline: push -> poll -> pull -> record.

Stages (run sequentially; each is resumable):
  ckpt : push kaggle_ckpt  -> 4B pretrain with ZG_CKPT=1 -> best_ckpt.pt (8GB) in the kernel's output
  c1   : push kaggle_c1    -> reloads best_ckpt.pt (via kernel_sources) -> head-only MLP sentiment

Designed to run in the background (long poll loops). Writes status + metrics to runs/experiments.jsonl
and a human log to runs/orchestrator.log. Does NOT download the 8GB checkpoint locally (the C.1 kernel
consumes it server-side); only pulls the small C.1 summary.

Usage:  python3 orchestrate_kaggle.py [ckpt] [c1]      (default: both)
Requires Kaggle API creds (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).
"""
import subprocess, time, json, sys, re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent
(ROOT/"runs").mkdir(exist_ok=True)
LEDGER = ROOT/"runs"/"experiments.jsonl"; LOG = ROOT/"runs"/"orchestrator.log"
CKPT_DIR, C1_DIR, ADAPT_DIR = ROOT/"kaggle_ckpt", ROOT/"kaggle_c1", ROOT/"kaggle_adapt"
CKPT_REF = "yanjinli2001/post-backprop-zerograd-4b-checkpoint"
C1_REF = "yanjinli2001/post-backprop-zerograd-c1"
ADAPT_REF = "yanjinli2001/post-backprop-zerograd-adapt"
DONE = ("complete", "error", "cancelacknowledged", "cancelrequested")
ACTIVE = ("running", "queued")
FORCE = False                                                 # set from argv in __main__ ("force" -> always re-push)

def now(): return datetime.now(timezone.utc).isoformat()
def log(m):
    line = f"[{now()}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")
def record(e):
    e = {"ts": now(), "site": "kaggle", **e}
    with open(LEDGER, "a") as f: f.write(json.dumps(e, default=float) + "\n")
def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True); return r.returncode, (r.stdout or "") + (r.stderr or "")

def check_creds():
    rc, out = sh(["kaggle", "config", "view"])
    if rc != 0 or "username" not in out.lower():
        rc2, _ = sh(["kaggle", "kernels", "list", "-m", "--page-size", "1"])
        if rc2 != 0: return False
    return True

def push(d):
    rc, out = sh(["kaggle", "kernels", "push", "-p", str(d)]); log(f"push {d.name}: rc={rc} {out.strip()[:240]}"); return rc == 0
def status(ref):
    rc, out = sh(["kaggle", "kernels", "status", ref]); m = re.search(r'status\s+"?([\w.]+)"?', out)
    return (m.group(1).split(".")[-1].lower() if m else "unknown"), out   # "KernelWorkerStatus.COMPLETE" -> "complete"
def wait(ref, label, timeout_s, interval=60):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st, out = status(ref); log(f"{label} status={st} ({int(time.time()-t0)}s)")
        if st in DONE: return st
        time.sleep(interval)
    return "timeout"
def pull(ref, d):
    d.mkdir(exist_ok=True); rc, out = sh(["kaggle", "kernels", "output", ref, "-p", str(d)]); log(f"pull {ref} rc={rc}"); return rc == 0

def stage_ckpt():
    st, _ = status(CKPT_REF)                                    # idempotent: don't re-push a run already in flight
    if st == "complete": log("ckpt already complete; skip push"); return True
    if st in ACTIVE: log(f"ckpt already {st}; skip push, just wait")
    else:
        record({"stage": "ckpt", "event": "push", "ref": CKPT_REF})
        if not push(CKPT_DIR): record({"stage": "ckpt", "event": "push_failed"}); return False
    st = wait(CKPT_REF, "ckpt", timeout_s=13000); record({"stage": "ckpt", "event": "finished", "status": st})
    return st == "complete"                                     # don't pull the 8GB .pt; C.1 reads it server-side

def stage_c1():
    st, _ = status(C1_REF)
    if st == "complete": log("c1 already complete; pulling")
    elif st in ACTIVE: log(f"c1 already {st}; skip push, just wait")
    else:
        record({"stage": "c1", "event": "push", "ref": C1_REF})
        if not push(C1_DIR): record({"stage": "c1", "event": "push_failed"}); return False
    st = wait(C1_REF, "c1", timeout_s=6000); record({"stage": "c1", "event": "finished", "status": st})
    if st != "complete": return False
    out = C1_DIR/"out"
    if pull(C1_REF, out) and (out/"c1_run_summary.json").exists():
        d = json.loads((out/"c1_run_summary.json").read_text())
        record({"stage": "c1", "event": "metrics", "metrics": d}); log("C1 RESULT: " + json.dumps(d, default=float))
    return True

def stage_adapt():
    st, _ = status(ADAPT_REF)
    if not FORCE and st == "complete": log("adapt already complete; pulling")
    elif not FORCE and st in ACTIVE: log(f"adapt already {st}; skip push, just wait")
    else:
        record({"stage": "adapt", "event": "push", "ref": ADAPT_REF, "force": FORCE})
        if not push(ADAPT_DIR): record({"stage": "adapt", "event": "push_failed"}); return False
    st = wait(ADAPT_REF, "adapt", timeout_s=6000); record({"stage": "adapt", "event": "finished", "status": st})
    if st != "complete": return False
    out = ADAPT_DIR/"out"
    if pull(ADAPT_REF, out) and (out/"adapt_run_summary.json").exists():
        d = json.loads((out/"adapt_run_summary.json").read_text())
        record({"stage": "adapt", "event": "metrics", "metrics": d}); log("ADAPT RESULT: " + json.dumps(d, default=float))
    return True

if __name__ == "__main__":
    args = sys.argv[1:]; FORCE = "force" in args                # force -> always (re)push even if complete
    stages = [s for s in args if s != "force"] or ["ckpt", "c1"]
    log(f"orchestrator start: stages={stages} force={FORCE}")
    if not check_creds():
        log("NO KAGGLE CREDENTIALS — put kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY"); sys.exit(2)
    if "ckpt" in stages and not stage_ckpt(): log("ckpt stage did not complete; stopping"); sys.exit(1)
    if "c1" in stages and not stage_c1(): log("c1 stage did not complete"); sys.exit(1)
    if "adapt" in stages and not stage_adapt(): log("adapt stage did not complete"); sys.exit(1)
    log("orchestrator done")
