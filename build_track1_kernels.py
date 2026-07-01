"""Generate the Track 1 Kaggle notebook: ZeroBP 4B on real SST-2 (mirrors the phasee_nli kernel).

Embeds the 4B stack (kaggle_zerograd_moe + c1_4b + phase_e_4b + phasee_nli_4b) + track1_sst2_4b via
%%writefile, reloads the 4B checkpoint (kernel_sources), and adapts on real SST-2 (internet for GLUE).
RESEARCH branch (Mixed-BP), never the submission path. Run:  python3 build_track1_kernels.py
"""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).parent
KSPEC = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
         "language_info": {"name": "python", "pygments_lexer": "ipython3"}}
def wf(f): return nbf.v4.new_code_cell(f"%%writefile {f}\n" + (root/f).read_text())

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Track 1 — ZeroBP 4B on real SST-2 (binary sentiment)\n"
        "RESEARCH branch (Mixed-BP embedding[+attn], autograd) — NOT the pure-ZeroBP submission. Reloads the "
        "4.16B checkpoint via `kernel_sources`, adapts on real GLUE/SST-2 (internet ON), reports zero-shot / "
        "Mixed-BP(emb) / Mixed-BP(emb+attn) with a fair closed-form head + WikiText ppl drift. Attach "
        "WikiText-103 + the 4B checkpoint kernel."),
    wf("kaggle_zerograd_moe.py"), wf("c1_4b.py"), wf("phase_e_4b.py"), wf("phasee_nli_4b.py"), wf("track1_sst2_4b.py"),
    nbf.v4.new_code_cell("import runpy\nrunpy.run_path('track1_sst2_4b.py', run_name='__main__')"),
]
nb.metadata.update(KSPEC)
kdir = root/"kaggle_track1_sst2"; kdir.mkdir(exist_ok=True)
nbf.write(nb, str(kdir/"track1_sst2_kernel.ipynb"))
(kdir/"kernel-metadata.json").write_text(
    '{\n'
    '  "id": "yanjinli2001/post-backprop-track1-sst2-v2",\n'
    '  "title": "Post Backprop Track1 SST2 V2",\n'
    '  "code_file": "track1_sst2_kernel.ipynb",\n'
    '  "language": "python",\n'
    '  "kernel_type": "notebook",\n'
    '  "is_private": true,\n'
    '  "enable_gpu": true,\n'
    '  "enable_internet": true,\n'
    '  "machine_shape": "NvidiaTeslaT4",\n'
    '  "dataset_sources": [\n    "vadimkurochkin/wikitext-103"\n  ],\n'
    '  "competition_sources": [],\n'
    '  "kernel_sources": [\n    "yanjinli2001/post-backprop-zerograd-4b-checkpoint"\n  ]\n'
    '}\n')
print(f"wrote {kdir/'track1_sst2_kernel.ipynb'} + kernel-metadata.json")
