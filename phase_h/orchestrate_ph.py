"""Phase H Kaggle orchestrator: push -> poll -> pull -> record (self-contained; mirrors ../orchestrate_kaggle.py).

Kept inside phase_h/ so the new stack stays isolated. Monitors the G1 (real NLI) kernel and records the
run summary to runs/experiments.jsonl + a human log to runs/ph_orchestrator.log.

Usage:  python3 phase_h/orchestrate_ph.py [phnli] [phgsm]  (default: both)
        python3 phase_h/orchestrate_ph.py phnli force      (always re-push)
Requires Kaggle API creds (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY). Run build_ph_kernels.py first.
"""
import subprocess, time, json, sys, re
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).parent; ROOT = HERE.parent
(ROOT/"runs").mkdir(exist_ok=True)
LEDGER = ROOT/"runs"/"experiments.jsonl"; LOG = ROOT/"runs"/"ph_orchestrator.log"
PHNLI_DIR = HERE/"kaggle_ph_nli"; PHNLI_REF = "yanjinli2001/post-backprop-phase-h-nli"
PHGSM_DIR = HERE/"kaggle_ph_gsm"; PHGSM_REF = "yanjinli2001/post-backprop-phase-h-multi-step"
PHGEN_DIR = HERE/"kaggle_ph_gsm_gen"; PHGEN_REF = "yanjinli2001/post-backprop-phaseh-gsmgen-v2"
DONE = ("complete", "error", "cancelacknowledged", "cancelrequested"); ACTIVE = ("running", "queued")
FORCE = False

def now(): return datetime.now(timezone.utc).isoformat()
def log(m):
    line = f"[{now()}] {m}"; print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")
def record(e):
    with open(LEDGER, "a") as f: f.write(json.dumps({"ts": now(), "site": "kaggle", "stack": "phase_h", **e}, default=float) + "\n")
def sh(c): r = subprocess.run(c, capture_output=True, text=True); return r.returncode, (r.stdout or "") + (r.stderr or "")
def check_creds():
    rc, out = sh(["kaggle", "config", "view"])
    if rc != 0 or "username" not in out.lower():
        rc2, _ = sh(["kaggle", "kernels", "list", "-m", "--page-size", "1"]); return rc2 == 0
    return True
def push(d): rc, out = sh(["kaggle", "kernels", "push", "-p", str(d)]); log(f"push {d.name}: rc={rc} {out.strip()[:240]}"); return rc == 0
def status(ref):
    rc, out = sh(["kaggle", "kernels", "status", ref]); m = re.search(r'status\s+"?([\w.]+)"?', out)
    return (m.group(1).split(".")[-1].lower() if m else "unknown"), out
def wait(ref, label, timeout_s, interval=60):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st, _ = status(ref); log(f"{label} status={st} ({int(time.time()-t0)}s)")
        if st in DONE: return st
        time.sleep(interval)
    return "timeout"
def pull(ref, d): d.mkdir(exist_ok=True); rc, _ = sh(["kaggle", "kernels", "output", ref, "-p", str(d)]); log(f"pull {ref} rc={rc}"); return rc == 0

def stage(name, ref, d, summ, label, timeout_s=13000):
    st, _ = status(ref)
    if not FORCE and st == "complete": log(f"{name} already complete; pulling")
    elif not FORCE and st in ACTIVE: log(f"{name} already {st}; skip push, just wait")
    else:
        record({"stage": name, "event": "push", "ref": ref})
        if not push(d): record({"stage": name, "event": "push_failed"}); return False
    st = wait(ref, name, timeout_s=timeout_s); record({"stage": name, "event": "finished", "status": st})
    if st != "complete": return False
    out = d/"out"
    if pull(ref, out) and (out/summ).exists():
        m = json.loads((out/summ).read_text())
        record({"stage": name, "event": "metrics", "metrics": m}); log(f"{label}: " + json.dumps(m, default=float))
    return True

STAGES = {
    "phnli": lambda: stage("phnli", PHNLI_REF, PHNLI_DIR, "ph_nli_run_summary.json", "PHASE H G1 RESULT"),
    "phgsm": lambda: stage("phgsm", PHGSM_REF, PHGSM_DIR, "ph_gsm_run_summary.json", "PHASE H G2 RESULT"),
    "phgsmgen": lambda: stage("phgsmgen", PHGEN_REF, PHGEN_DIR, "ph_gsmgen_run_summary.json", "PHASE H G2b RESULT"),
}

if __name__ == "__main__":
    args = sys.argv[1:]; FORCE = "force" in args
    stages = [s for s in args if s != "force"] or ["phnli", "phgsm", "phgsmgen"]
    log(f"ph orchestrator start: stages={stages} force={FORCE}")
    if not check_creds(): log("NO KAGGLE CREDENTIALS — ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY"); sys.exit(2)
    for s in stages:
        if s in STAGES and not STAGES[s](): log(f"{s} stage did not complete"); sys.exit(1)
    log("ph orchestrator done")
