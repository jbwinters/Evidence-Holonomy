#!/bin/bash
set -e

echo "=== Packaging for arXiv Submission ==="

# Create arXiv submission directory
mkdir -p arxiv_submission
cd arxiv_submission

# Copy main tex file
cp ../uec_theory.tex .

# Copy figures
mkdir -p figures
cp ../figures/*.pdf figures/

# Copy ancillary files  
mkdir -p anc
cp ../anc/*.csv anc/
cp ../anc/README_anc.md anc/

# Copy essential scripts for reproducibility
mkdir -p scripts
cp ../scripts/reproduce_all.sh scripts/
cp ../scripts/summarize.py scripts/
cp ../scripts/plot_results.py scripts/

# Copy source code
mkdir -p src
cp -r ../src/uec src/

# Create submission tarball
tar -czf ../uec_theory_arxiv_submission.tar.gz .

cd ..

echo ""
echo "=== arXiv Submission Package Ready ==="
echo ""
echo "Main files:"
echo "  uec_theory.tex - Main paper"
echo "  figures/ - PDF figures for paper"
echo "  anc/ - Ancillary CSV data files"
echo "  src/ - Complete UEC library source"
echo "  scripts/ - Reproducibility scripts"
echo ""
echo "Submission file: uec_theory_arxiv_submission.tar.gz"
echo ""
echo "To submit:"
echo "1. Upload the .tar.gz file to arXiv"
echo "2. Use category: cs.IT (Information Theory)"
echo "3. Cross-list: cond-mat.stat-mech"
echo "4. Comments: 'Code: https://github.com/josh-winters/holonomy'"
echo "5. License: CC BY 4.0"