# Build the arXiv paper. Bibliography is embedded (no bibtex pass needed).
MAIN = main

.PHONY: all dist clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex figures/*.png
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

# Tarball for arXiv upload: source + figures only (no build artifacts).
dist: $(MAIN).tex
	tar czf arxiv-submission.tar.gz $(MAIN).tex figures/*.png

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).pdf arxiv-submission.tar.gz
