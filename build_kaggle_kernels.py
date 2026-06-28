"""Generate the self-contained Kaggle notebooks for the C.1 4B pipeline (keeps them in sync with the .py).

  kaggle_ckpt/ckpt_kernel.ipynb : run the 4B submission with ZG_CKPT=1 -> best_ckpt.pt (8GB) in output
  kaggle_c1/c1_kernel.ipynb     : reload best_ckpt.pt (via kernel_sources) -> head-only MLP sentiment

Both embed kaggle_zerograd_moe.py (+ c1_4b.py for C.1) via %%writefile so no Kaggle utility-script setup
is needed. Run:  python3 build_kaggle_kernels.py
"""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).parent
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
(root/"kaggle_c1").mkdir(exist_ok=True)
nbf.write(c1, str(root/"kaggle_c1"/"c1_kernel.ipynb"))

for p in ["kaggle_ckpt/ckpt_kernel.ipynb", "kaggle_c1/c1_kernel.ipynb"]:
    print(f"wrote {p} ({(root/p).stat().st_size//1024} KB)")
