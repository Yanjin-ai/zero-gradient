# arXiv submission package

Self-contained LaTeX source for the technical report *"ZeroBP at Scale and the Architecture of Capability."*
Everything needed to build the PDF and to upload to arXiv is in **this folder**.

## Contents

```
arxiv/
├── main.tex          # the full paper (single source)
├── figures/          # all figures used by the paper (PNG)
│   ├── research_arc.png
│   ├── capability_comparison.png
│   ├── reasoning_depth_wall.png
│   └── scorecard.png
├── Makefile          # convenience build targets
└── README.md         # this file
```

No custom `.sty`/`.cls` files and **no `.bib`/`.bbl`** are required: the paper uses only standard TeX Live
packages, and the bibliography is embedded with `thebibliography` (so no BibTeX pass is needed).

**Packages used** (all in a stock TeX Live / arXiv install): `inputenc`, `fontenc`, `lmodern`, `geometry`,
`graphicx`, `booktabs`, `amsmath`, `amssymb`, `xcolor`, `caption`, `array`, `microtype`, `verbatim`,
`hyperref`.

## Build locally

```bash
make            # -> main.pdf (runs pdflatex twice for cross-references/hyperref)
# or:
pdflatex main.tex && pdflatex main.tex
# or, with a self-contained engine that fetches packages automatically:
tectonic main.tex
```

The source is pure ASCII and uses only pdflatex-safe constructs, so it builds with `pdflatex` (no
`xelatex`/`lualatex` needed).

## Submit to arXiv

1. Build once locally to confirm it compiles (`make`).
2. Create the upload tarball (source + figures only — **not** the PDF/aux):
   ```bash
   make dist        # -> arxiv-submission.tar.gz  (main.tex + figures/)
   ```
3. On arXiv, choose *Submit* → category **cs.LG** (cross-list **cs.CL**), upload `arxiv-submission.tar.gz`.
   arXiv runs `pdflatex` on `main.tex`; since the bibliography is embedded, no BibTeX step is needed.
4. Fill in title/abstract (copy from `main.tex`), authors, and license.

> Tip: arXiv accepts PNG figures. If a reviewer prefers vector figures, regenerate them as PDF from the
> repository root with a one-line change to `make_figures.py` (`savefig(... .pdf)`), then drop the PDFs into
> `figures/` and rebuild.
