#!/bin/bash
# Build an arXiv source bundle: uec_theory_arxiv.tar.gz in the repository root.
# The paper's ancillary data and code go under anc/, as arXiv expects.
set -euo pipefail

OUT=arxiv_submission
rm -rf "$OUT"
mkdir -p "$OUT/figures" "$OUT/tables" "$OUT/anc"

cp uec_theory.tex "$OUT/"
cp figures/markov_convergence.pdf figures/calibration.pdf figures/hmm_order_sweep.pdf figures/seq_*.pdf "$OUT/figures/"
cp tables/*.tex "$OUT/tables/"
cp anc/*.csv "$OUT/anc/"
cp -r src/uec "$OUT/anc/uec"
cp scripts/paper_experiments.py scripts/summarize.py scripts/plot_results.py scripts/sequential_experiments.py scripts/sequential_tables.py scripts/plot_sequential.py "$OUT/anc/"
find "$OUT" -name '__pycache__' -type d -prune -exec rm -rf {} +

tar -czf uec_theory_arxiv.tar.gz -C "$OUT" .
rm -rf "$OUT"

echo "Wrote uec_theory_arxiv.tar.gz"
echo "Code: https://github.com/jbwinters/Evidence-Holonomy"
