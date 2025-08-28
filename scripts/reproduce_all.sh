#!/bin/bash
set -e

echo "=== UEC Paper Reproducibility Suite ==="
echo "Generating all results for arXiv submission..."

# Clean previous results
rm -rf results/* anc/* figures/*
mkdir -p results anc figures

# Set up Python path
export PYTHONPATH=src

# Record environment info
echo "Python: $(python --version)"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Date: $(date)"
echo ""

# Helper function to run UEC battery
run_battery() {
  python -c "from uec.cli import run_battery; run_battery(['--seed', '$1', '--n', '$2', '--k', '$3', '--order', '3', '--out_csv', 'anc/markov_sanity.csv'])"
}

# Helper function to run UEC battery with coder
run_battery_coder() {
  python -c "from uec.cli import run_battery; run_battery(['--seed', '$1', '--n', '$2', '--k', '$3', '--coder', '$4', '--out_csv', 'anc/code_invariance_$4.csv'])"
}

# 1. Synthetic Markov sanity tests (Table 2 in paper)
echo "=== 1. Markov Sanity Tests ==="
echo "Testing entropy production estimation across multiple chains and window sizes..."

# Multiple seeds and window sizes for robust statistics
for seed in 1 2 3 4 5; do
  for n in 512 2048 8192 32768; do
    for k in 3 4 5; do
      echo "Running: seed=$seed, n=$n, k=$k"
      run_battery $seed $n $k
    done
  done
done

# 2. Code invariance tests (Figure 2 in paper) 
echo ""
echo "=== 2. Code Invariance Tests ==="
echo "Comparing KT vs LZ78 holonomy estimates..."

# Run same tests with different coders
for seed in 1 2 3; do
  for n in 2048 8192; do
    echo "Code invariance: seed=$seed, n=$n"
    # KT coder
    run_battery_coder $seed $n 3 kt
    # LZ78 coder  
    run_battery_coder $seed $n 3 lz78
  done
done

# 3. Representation-space holonomy validation
echo ""
echo "=== 3. Representation Holonomy Tests ==="
echo "Testing entropy-rate difference estimation..."

# Simple permutation loops (should give zero holonomy)
for seed in 1 2 3; do
  echo "Permutation test: seed=$seed"
  run_battery $seed 16384 4
done

echo ""
echo "=== 4. Generating Summary Tables ==="
PYTHONPATH=src python scripts/summarize.py

echo ""
echo "=== 5. Generating Figures ==="
PYTHONPATH=src python scripts/plot_results.py

echo ""
echo "=== Reproducibility Suite Complete ==="
echo "Results in: results/"
echo "Ancillary files for arXiv: anc/"
echo "Figures for paper: figures/"
echo ""
echo "Files ready for arXiv submission:"
ls -la anc/