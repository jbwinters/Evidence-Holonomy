#!/usr/bin/env python3
"""
Generate figures for the paper from experimental results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

def plot_markov_convergence():
    """Figure 1: KL-holonomy convergence to σ."""
    try:
        df = pd.read_csv('anc/markov_sanity.csv')
    except FileNotFoundError:
        print("Warning: markov_sanity.csv not found, skipping convergence plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Group by n, compute statistics
    stats = df.groupby('n').agg({
        'sigma_true': 'mean',
        'hol_rate': ['mean', 'std'],
        'abs_err': ['mean', 'std']
    })
    
    n_values = stats.index
    sigma_true_mean = stats[('sigma_true', 'mean')]
    hol_mean = stats[('hol_rate', 'mean')]
    hol_std = stats[('hol_rate', 'std')]
    
    # Plot estimated holonomy with error bars
    ax.errorbar(n_values, hol_mean, yerr=hol_std, 
                marker='o', capsize=5, capthick=2, 
                label='KL-holonomy estimate', color='blue')
    
    # Plot true entropy production  
    ax.plot(n_values, sigma_true_mean, 
            'r--', linewidth=2, label=r'True $\sigma$')
    
    ax.set_xlabel('Window length $n$')
    ax.set_ylabel('Entropy production (bits/step)')
    ax.set_xscale('log', base=2)
    ax.set_title('Convergence of KL-holonomy to entropy production rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/markov_convergence.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/markov_convergence.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/markov_convergence.pdf")
    plt.close()

def plot_code_invariance():
    """Figure 2: Code invariance scatter plot."""
    try:
        df = pd.read_csv('anc/code_invariance_merged.csv')
    except FileNotFoundError:
        print("Warning: code_invariance_merged.csv not found, skipping invariance plot")
        return
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Scatter plot
    ax.scatter(df['hol_rate_kt'], df['hol_rate_lz78'], 
               alpha=0.6, s=30, color='blue')
    
    # y=x reference line
    min_val = min(df['hol_rate_kt'].min(), df['hol_rate_lz78'].min())
    max_val = max(df['hol_rate_kt'].max(), df['hol_rate_lz78'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, alpha=0.8, label='y = x')
    
    # Statistics
    r = np.corrcoef(df['hol_rate_kt'], df['hol_rate_lz78'])[0, 1]
    mean_delta = df['delta_hol'].mean()
    
    ax.set_xlabel('KT holonomy rate (bits/step)')
    ax.set_ylabel('LZ78 holonomy rate (bits/step)')
    ax.set_title(f'Code invariance: KT vs LZ78\n(r={r:.3f}, mean |Δ|={mean_delta:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('figures/code_invariance_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/code_invariance_scatter.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/code_invariance_scatter.pdf")
    plt.close()

def plot_error_vs_n():
    """Additional plot: Relative error vs window size."""
    try:
        df = pd.read_csv('anc/markov_sanity.csv')
    except FileNotFoundError:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Box plots of relative error by n
    n_values = sorted(df['n'].unique())
    rel_errs = [df[df['n'] == n]['rel_err'] for n in n_values]
    
    bp = ax.boxplot(rel_errs, positions=range(len(n_values)), 
                    patch_artist=True, boxprops=dict(facecolor='lightblue'))
    
    ax.set_xlabel('Window length $n$')
    ax.set_ylabel('Relative error')
    ax.set_yscale('log')
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels([f'$2^{{{int(np.log2(n))}}}$' for n in n_values])
    ax.set_title('Relative error vs window size')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/error_vs_n.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/error_vs_n.png', dpi=300, bbox_inches='tight')
    print("Saved: figures/error_vs_n.pdf")
    plt.close()

if __name__ == "__main__":
    print("Generating figures...")
    
    # Create figures directory
    Path('figures').mkdir(exist_ok=True)
    
    plot_markov_convergence()
    plot_code_invariance() 
    plot_error_vs_n()
    
    print("Figure generation complete.")