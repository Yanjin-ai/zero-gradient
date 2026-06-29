"""Generate the self-contained Kaggle notebooks for the C.1 4B pipeline (keeps them in sync with the .py).

  kaggle_ckpt/ckpt_kernel.ipynb : run the 4B submission with ZG_CKPT=1 -> best_ckpt.pt (8GB) in output
  kaggle_c1/c1_kernel.ipynb     : reload best_ckpt.pt (via kernel_sources) -> head-only MLP sentiment

Both embed kaggle_zerograd_moe.py (+ c1_4b.py for C.1) via %%writefile so no Kaggle utility-script setup
is needed. Run:  python3 build_kaggle_kernels.py
"""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).parent
# Kaggle/papermill require a kernelspec or execution fails with "No kernel name found in notebook".
KSPEC = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
         "language_info": {"name": "python", "pygments_lexer": "ipython3"}}
def wf(fname): return nbf.v4.new_code_cell(f"%%writefile {fname}\n" + (root/fname).read_text())

# ---- checkpoint kernel: 4B pretrain with ZG_CKPT=1 -> best_ckpt.pt ----
ckpt = nbf.v4.new_notebook()
ckpt.cells = [
    nbf.v4.new_markdown_cell(
        "# Produce the 4B best checkpoint (ZG_CKPT=1)\n"
        "Runs the 4.160B BPE+MLP submission unchanged except for `ZG_CKPT=1`, which writes the best model "
        "to `/kaggle/working/best_ckpt.pt` (~8GB) once, at the end. Attach the WikiText-103 dataset. "
        "The C.1 kernel consumes this kernel's output via `kernel_sources`."),
    wf("kaggle_zerograd_moe.py"),
    nbf.v4.new_code_cell(
        "import os, runpy\n"
        "os.environ['ZG_CKPT'] = '1'                       # persist best checkpoint (save-once at end)\n"
        "runpy.run_path('kaggle_zerograd_moe.py', run_name='__main__')   # full 4B pretrain + WikiText eval + best_ckpt.pt"),
]
ckpt.metadata.update(KSPEC)
(root/"kaggle_ckpt").mkdir(exist_ok=True)
nbf.write(ckpt, str(root/"kaggle_ckpt"/"ckpt_kernel.ipynb"))

# ---- C.1 kernel: head-only MLP sentiment on the reloaded checkpoint ----
c1 = nbf.v4.new_notebook()
c1.cells = [
    nbf.v4.new_markdown_cell(
        "# C.1 — head-only post-training on the 4B zero-gradient MoE\n"
        "Reloads `best_ckpt.pt` (mounted from the checkpoint kernel via `kernel_sources`) and trains ONLY "
        "a 2-layer MLP sentiment head (zero autograd, closed-form local CE) on a vocab-overlapping "
        "compositional sentiment task (sentiment = polarity XOR negation, real words via the model's own "
        "tokenizer). Monitors task acc + WikiText LM ppl (forgetting). Attach the WikiText-103 dataset."),
    wf("kaggle_zerograd_moe.py"),
    wf("c1_4b.py"),
    nbf.v4.new_code_cell(
        "import os\n"
        "os.environ.setdefault('ZG_C1_STEPS', '1000')      # head-train steps\n"
        "import c1_4b, json\n"
        "summary = c1_4b.main()\n"
        "print(json.dumps(summary, indent=2, default=float))"),
]
c1.metadata.update(KSPEC)
(root/"kaggle_c1").mkdir(exist_ok=True)
nbf.write(c1, str(root/"kaggle_c1"/"c1_kernel.ipynb"))

# ---- v1.1 4B-Adapt kernel: in-domain adaptation + C.1, with forgetting mitigation ----
ad = nbf.v4.new_notebook()
ad.cells = [
    nbf.v4.new_markdown_cell(
        "# v1.1 4B-Adapt — in-domain adaptation + C.1 (zero autograd, mitigated)\n"
        "Reloads `best_ckpt.pt` (via `kernel_sources`), does a SHORT in-domain zero-BP LM adaptation on the "
        "sentiment domain (small lr + WikiText replay + frozen routing to limit forgetting), then head-only "
        "MLP sentiment. Reports 4B-ZeroShot vs 4B-Adapt acc + WikiText ppl drift. Attach WikiText-103."),
    wf("kaggle_zerograd_moe.py"), wf("c1_4b.py"), wf("adapt_4b.py"),
    nbf.v4.new_code_cell(
        "import os, json\n"
        "os.environ.setdefault('ZG_ADAPT_STEPS', '300')         # moderate adaptation budget\n"
        "os.environ.setdefault('ZG_ADAPT_LR', '0.02')           # < pretrain 0.03\n"
        "os.environ.setdefault('ZG_ADAPT_REPLAY', '0.3')        # 30% WikiText replay\n"
        "os.environ.setdefault('ZG_ADAPT_FREEZE_HEADS', '1')    # 2.1 mitigation: protect the LM head (cut forgetting)\n"
        "os.environ.setdefault('ZG_C1_STEPS', '1000')\n"
        "import adapt_4b\n"
        "print(json.dumps(adapt_4b.main(), indent=2, default=float))"),
]
ad.metadata.update(KSPEC)
(root/"kaggle_adapt").mkdir(exist_ok=True)
nbf.write(ad, str(root/"kaggle_adapt"/"adapt_kernel.ipynb"))

# ---- Phase E (hybrid BP) kernel: research branch, autograd on embedding + task head ----
pe = nbf.v4.new_notebook()
pe.cells = [
    nbf.v4.new_markdown_cell(
        "# Phase E -- hybrid BP at 4B (RESEARCH BRANCH, uses autograd)\n"
        "Reloads `best_ckpt.pt` and does a little REAL BP through the embedding + task head only "
        "(everything else frozen, pretraining stays zero-BP) to adapt the compositional sentiment task. "
        "Reports zero-shot vs Mixed-BP acc + WikiText ppl drift. NOT the zero-BP submission. Attach WikiText-103."),
    wf("kaggle_zerograd_moe.py"), wf("c1_4b.py"), wf("phase_e_4b.py"),
    nbf.v4.new_code_cell(
        "import os, json\n"
        "os.environ.setdefault('ZG_E_STEPS', '1000')   # match the small-config setting that hit 100%\n"
        "os.environ.setdefault('ZG_E_LR', '0.1')\n"
        "import phase_e_4b\n"
        "print(json.dumps(phase_e_4b.main(), indent=2, default=float))"),
]
pe.metadata.update(KSPEC)
(root/"kaggle_phase_e").mkdir(exist_ok=True)
nbf.write(pe, str(root/"kaggle_phase_e"/"phase_e_kernel.ipynb"))

for p in ["kaggle_ckpt/ckpt_kernel.ipynb", "kaggle_c1/c1_kernel.ipynb", "kaggle_adapt/adapt_kernel.ipynb",
          "kaggle_phase_e/phase_e_kernel.ipynb"]:
    print(f"wrote {p} ({(root/p).stat().st_size//1024} KB)")
