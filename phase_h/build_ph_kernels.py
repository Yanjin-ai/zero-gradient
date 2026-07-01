"""Generate the Phase H Kaggle notebook (keeps it in sync with the .py; mirrors ../build_kaggle_kernels.py).

Embeds ph_base.py + ph_nli_gpu.py via %%writefile (self-contained, no Kaggle utility-script setup), then
runs real SNLI on a T4. RESEARCH-ONLY / isolated: this kernel has ZERO relation to the ZeroBP submission.
Run:  python3 phase_h/build_ph_kernels.py
"""
from pathlib import Path
import nbformat as nbf

here = Path(__file__).parent
KSPEC = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
         "language_info": {"name": "python", "pygments_lexer": "ipython3"}}
def wf(fname): return nbf.v4.new_code_cell(f"%%writefile {fname}\n" + (here/fname).read_text())

def meta(kid, title, code_file, internet=True):
    return ('{\n'
            f'  "id": "yanjinli2001/{kid}",\n'
            f'  "title": "{title}",\n'
            f'  "code_file": "{code_file}",\n'
            '  "language": "python",\n'
            '  "kernel_type": "notebook",\n'
            '  "is_private": true,\n'
            '  "enable_gpu": true,\n'
            f'  "enable_internet": {"true" if internet else "false"},\n'
            '  "machine_shape": "NvidiaTeslaT4",\n'
            '  "dataset_sources": [],\n'
            '  "competition_sources": [],\n'
            '  "kernel_sources": []\n'
            '}\n')

def build(kid, title, code_file, kdirname, md, script, argv):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(md), wf("ph_base.py"), wf(script),
                nbf.v4.new_code_cell(f"import sys, runpy\nsys.argv = {argv!r}\nrunpy.run_path('{script}', run_name='__main__')")]
    nb.metadata.update(KSPEC)
    kdir = here/kdirname; kdir.mkdir(exist_ok=True)
    nbf.write(nb, str(kdir/code_file)); (kdir/"kernel-metadata.json").write_text(meta(kid, title, code_file))
    print(f"wrote {kdir/code_file} + kernel-metadata.json")

# ---- G1: real NLI (SNLI) ----
build("post-backprop-phase-h-nli", "Post Backprop Phase H NLI", "ph_nli_kernel.ipynb", "kaggle_ph_nli",
      "# Phase H / v3.0 — G1: real NLI (SNLI) on a standard multi-layer trainable-attention base\n"
      "RESEARCH-ONLY, fully isolated from the ZeroBP 4B submission. 6L x 8H d=256 + full backprop on SNLI. "
      "Contrast: ZeroBP 4B NLI = chance (33.4%). Internet ON for `datasets.load_dataset('snli')`, or attach "
      "an NLI dataset and use `--source jsonl --data_dir /kaggle/input/<slug>`.",
      "ph_nli_gpu.py",
      ['ph_nli_gpu.py', '--source', 'hf', '--dataset', 'snli', '--layers', '6', '--heads', '8',
       '--d_model', '256', '--max_len', '64', '--epochs', '3', '--out', '/kaggle/working/ph_nli_run_summary.json'])

# ---- G2: multi-step arithmetic depth sweep (GSM proxy; real GSM8K = G2b stretch) ----
build("post-backprop-phase-h-multi-step", "Post Backprop Phase H Multi-step", "ph_gsm_kernel.ipynb", "kaggle_ph_gsm",
      "# Phase H / v3.0 — G2: multi-step arithmetic depth sweep (standard trainable-attn base + full BP)\n"
      "RESEARCH-ONLY, isolated. Sweeps n_steps to map how deep multi-step reasoning installs. Contrast: "
      "ZeroBP is chance at k=2 already (uninstallable at any BP depth). Real NL GSM8K = G2b stretch "
      "(generative; needs a causal-LM variant + scale) — not this kernel.",
      "ph_gsm_gpu.py",
      ['ph_gsm_gpu.py', '--steps_list', '2,4,6,8', '--train_steps', '6000', '--d_model', '256',
       '--layers', '6', '--heads', '8', '--out', '/kaggle/working/ph_gsm_run_summary.json'])

# ---- G2b: GENERATIVE multi-step (causal LM) — synthetic + real GSM8K (honest stretch) ----
build("post-backprop-phase-h-gsm-gen", "Post Backprop Phase H GSM Gen", "ph_gsmgen_kernel.ipynb", "kaggle_ph_gsm_gen",
      "# Phase H / v3.0 — G2b: GENERATIVE multi-step reasoning (causal LM, exact-match)\n"
      "RESEARCH-ONLY, isolated. Generates the answer token-by-token (real GSM8K shape). Runs synthetic "
      "generative arithmetic AND real GSM8K (HF datasets, internet ON). HONEST: a small char LM is a weak "
      "fit for NL GSM8K — this scaffolds the machinery + contrast (ZeroBP installs multi-step at NO depth), "
      "not a claim a tiny model solves GSM8K.",
      "ph_gsm_gen.py",
      ['ph_gsm_gen.py', '--source', 'gsm8k', '--layers', '6', '--heads', '8', '--d_model', '256',
       '--seq_len', '320', '--train_steps', '8000', '--limit', '7000', '--out', '/kaggle/working/ph_gsmgen_run_summary.json'])
