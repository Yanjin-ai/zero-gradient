"""Generate the self-contained Kaggle C.1 notebook (kaggle_c1/c1_kernel.ipynb).

Embeds kaggle_zerograd_moe.py + c1_4b.py via %%writefile, then runs c1_4b. The C.1 run reloads the
4B best_ckpt.pt (attach as a dataset), rebuilds the same BPE tokenizer from WikiText (attach that
dataset), and trains a head-only MLP sentiment head. Self-contained -> no Kaggle utility-script setup.

Run:  python3 build_c1_kernel.py   ->  writes kaggle_c1/c1_kernel.ipynb (metadata already in kaggle_c1/)
"""
import json
from pathlib import Path
import nbformat as nbf

root = Path(__file__).parent
out = root/"kaggle_c1"; out.mkdir(exist_ok=True)

def writefile_cell(fname):
    body = (root/fname).read_text()
    return nbf.v4.new_code_cell(f"%%writefile {fname}\n" + body)

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# C.1 — head-only post-training on the 4B zero-gradient MoE\n"
        "Reloads the 4.160B BPE+MLP best checkpoint and trains ONLY a 2-layer MLP sentiment head "
        "(zero autograd, closed-form local CE) on a vocab-overlapping compositional sentiment task "
        "(sentiment = polarity XOR negation). Monitors task acc + WikiText LM ppl (forgetting).\n\n"
        "**Attach two datasets:** the WikiText-103 dataset, and the dataset holding `best_ckpt.pt` "
        "(produced by a `ZG_CKPT=1` run of the main submission)."),
    writefile_cell("kaggle_zerograd_moe.py"),
    writefile_cell("c1_4b.py"),
    nbf.v4.new_code_cell(
        "import os\n"
        "os.environ.setdefault('ZG_C1_STEPS', '1000')   # head-train steps (raise for longer/convergence)\n"
        "# c1_4b auto-finds /kaggle/input/**/best_ckpt.pt and the WikiText dataset\n"
        "import c1_4b; summary = c1_4b.main()\n"
        "import json; print(json.dumps(summary, indent=2, default=float))"),
]
nbf.write(nb, str(out/"c1_kernel.ipynb"))
print(f"wrote {out/'c1_kernel.ipynb'} ({(out/'c1_kernel.ipynb').stat().st_size//1024} KB)")
