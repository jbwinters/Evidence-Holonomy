#!/usr/bin/env python3
"""
Generate summary tables and statistics from CSV results for the paper.
Produces LaTeX tables ready for inclusion in uec_theory.tex.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def load_csv_safe(path):
    """Load CSV if it exists, return empty DataFrame if not."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Warning: {path} not found, skipping...")
        return pd.DataFrame()

def summarize_markov_sanity():
    """Generate Table 2: Markov sanity check results."""
    df = load_csv_safe('anc/markov_sanity.csv')
    if df.empty:
        return
    
    print("=== Table 2: Markov Sanity Check ===")
    
    # Group by n and k, compute statistics
    summary = df.groupby(['n', 'k']).agg({
        'sigma_true': 'mean',
        'hol_rate': 'mean',
        'abs_err': ['mean', 'std'],
        'rel_err': ['mean', 'std', 'median']
    }).round(6)
    
    print("Chain (states) | n | σ_true | σ_hat | Abs Error | Rel Error")
    print("-" * 60)
    
    for (n, k), row in summary.iterrows():
        sigma_true = row[('sigma_true', 'mean')]
        sigma_hat = row[('hol_rate', 'mean')]
        abs_err = row[('abs_err', 'mean')]
        rel_err_pct = row[('rel_err', 'median')] * 100
        
        print(f"{k}-state | {n:5d} | {sigma_true:.4f} | {sigma_hat:.4f} | {abs_err:.1e} | {rel_err_pct:.1f}%")
    
    # Generate LaTeX table
    latex_table = """
\\begin{table}[t]
\\centering
\\caption{Entropy production rate $\\sigma$ (bits/step): ground truth vs. estimate.}
\\label{tab:markov}
\\begin{tabular}{lcccc}
\\toprule
Chain (states) & $n$ & $\\sigma_{\\text{true}}$ & $\\sigma_{\\text{hat}}$ & Rel. error \\\\
\\midrule
"""
    
    for (n, k), row in summary.iterrows():
        sigma_true = row[('sigma_true', 'mean')]
        sigma_hat = row[('hol_rate', 'mean')]
        rel_err_pct = row[('rel_err', 'median')] * 100
        
        latex_table += f"{k}-state & $2^{{{int(np.log2(n))}}}$ & {sigma_true:.3f} & {sigma_hat:.3f} & {rel_err_pct:.1f}\\% \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open('results/table_markov_sanity.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to: results/table_markov_sanity.tex")

def summarize_code_invariance():
    """Analyze code invariance between KT and LZ78."""
    df_kt = load_csv_safe('anc/code_invariance_kt.csv')
    df_lz78 = load_csv_safe('anc/code_invariance_lz78.csv')
    
    if df_kt.empty or df_lz78.empty:
        return
    
    print("\n=== Code Invariance Analysis ===")
    
    # Merge on matching conditions (seed, n, k)
    merged = pd.merge(df_kt, df_lz78, on=['seed', 'n', 'k'], suffixes=('_kt', '_lz78'))
    merged['delta_hol'] = abs(merged['hol_rate_kt'] - merged['hol_rate_lz78'])
    
    # Statistics
    mean_delta = merged['delta_hol'].mean()
    median_delta = merged['delta_hol'].median()
    max_delta = merged['delta_hol'].max()
    pct_small = (merged['delta_hol'] <= 0.05).mean() * 100
    
    print(f"Mean |hol_KT - hol_LZ78|: {mean_delta:.4f} bits/step")
    print(f"Median |hol_KT - hol_LZ78|: {median_delta:.4f} bits/step")
    print(f"Max |hol_KT - hol_LZ78|: {max_delta:.4f} bits/step")
    print(f"Windows with |delta| ≤ 0.05: {pct_small:.1f}%")
    
    # Save for plotting
    merged[['seed', 'n', 'k', 'hol_rate_kt', 'hol_rate_lz78', 'delta_hol']].to_csv(
        'anc/code_invariance_merged.csv', index=False)
    
    print("Code invariance data saved to: anc/code_invariance_merged.csv")

def generate_paper_stats():
    """Generate key statistics mentioned in the paper."""
    print("\n=== Paper Statistics Summary ===")
    
    # Markov sanity stats
    df = load_csv_safe('anc/markov_sanity.csv')
    if not df.empty:
        # Filter for n >= 2048 (convergence regime)
        df_converged = df[df['n'] >= 2048]
        median_rel_err = df_converged['rel_err'].median() * 100
        pct_good = (df_converged['rel_err'] <= 0.10).mean() * 100
        
        print(f"Markov convergence (n≥2048):")
        print(f"  - Median relative error: {median_rel_err:.1f}%")
        print(f"  - Runs with <10% error: {pct_good:.0f}%")
    
    # Code invariance stats  
    df_inv = load_csv_safe('anc/code_invariance_merged.csv')
    if not df_inv.empty:
        pct_small = (df_inv['delta_hol'] <= 0.05).mean() * 100
        print(f"Code invariance:")
        print(f"  - Windows with |KT-LZ78| ≤ 0.05: {pct_small:.0f}%")

if __name__ == "__main__":
    print("Summarizing experimental results...")
    print("=" * 50)
    
    summarize_markov_sanity()
    summarize_code_invariance() 
    generate_paper_stats()
    
    print("\n" + "=" * 50)
    print("Summary complete. Check results/ for LaTeX tables.")