#!/bin/bash
# Regenerate every number, table, and figure in uec_theory.tex, then build the PDF.
# Run from the repository root. Overwrites anc/*.csv, tables/, and figures/.
set -euo pipefail

export PYTHONPATH=src

echo "Python: $(python --version 2>&1)"
echo "NumPy:  $(python -c 'import numpy; print(numpy.__version__)')"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Date:   $(date)"

echo "=== 1. Experiments (anc/*.csv) ==="
python scripts/paper_experiments.py

echo "=== 2. Tables and summary statistics ==="
python scripts/summarize.py

echo "=== 3. Figures ==="
python scripts/plot_results.py

echo "=== 4. Paper ==="
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -quiet uec_theory.tex
  latexmk -c -quiet uec_theory.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error uec_theory.tex >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error uec_theory.tex >/dev/null
  rm -f uec_theory.aux uec_theory.log uec_theory.out
else
  echo "No LaTeX found; skipping PDF build."
fi

echo "Done."
