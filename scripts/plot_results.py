#!/usr/bin/env python3
"""
Generate the paper's figures from the ancillary CSVs in anc/.

Every plotted point comes from anc/*.csv, produced by scripts/paper_experiments.py.
Outputs PDF (for the paper) and PNG (for the README) under figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9})
COLORS = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c"}


def _save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/{name}.pdf")


def plot_markov_convergence() -> None:
    """Figure 1: estimate / sigma for each chain as n grows."""
    df = pd.read_csv("anc/markov_sanity.csv")
    df["ratio"] = df["estimate"] / df["sigma_true"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (seed, k), g in df.groupby(["seed", "k"]):
        g = g.sort_values("n")
        ax.plot(g["n"], g["ratio"], "-o", color=COLORS[k], alpha=0.45, ms=3, lw=1)
    med = df.groupby("n")["ratio"].median()
    ax.plot(med.index, med.values, "k-", lw=2.5, label="median over 15 chains")
    for k, c in COLORS.items():
        ax.plot([], [], "-o", color=c, ms=3, label=f"{k}-state chains")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sample length $n$")
    ax.set_ylabel(r"estimate / $\sigma_{\mathrm{true}}$")
    ax.set_ylim(0, max(2.5, float(df["ratio"].max()) * 1.05))
    ax.legend(loc="upper right")
    ax.set_title("Time-reversal estimate relative to analytic entropy production")
    _save(fig, "markov_convergence")


def plot_calibration() -> None:
    """Figure 2: estimate vs sigma for 60 random chains at n = 2^15."""
    df = pd.read_csv("anc/calibration.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for k, g in df.groupby("k"):
        ax.scatter(g["sigma_true"], g["estimate"], s=18, color=COLORS[k], label=f"{k}-state", alpha=0.8)
    hi = float(max(df["sigma_true"].max(), df["estimate"].max())) * 1.05
    ax.plot([0, hi], [0, hi], "k--", lw=1, label="$y=x$")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(r"analytic $\sigma$ (bits/step)")
    ax.set_ylabel(r"estimate, $R=3$, $n=2^{15}$ (bits/step)")
    ax.legend(loc="upper left")
    ax.set_title("Calibration across 60 random chains")
    _save(fig, "calibration")


def plot_hmm_order_sweep() -> None:
    """Figure 3: observed-record estimates vs Markov order R for a hidden ring."""
    df = pd.read_csv("anc/hmm_order_sweep.csv")
    sigma_hidden = float(df["sigma_hidden"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, color in (("noisy", "#9467bd"), ("lumped", "#8c564b")):
        g = df[df["channel"] == name].groupby("order")["estimate"]
        mean = g.mean()
        ax.errorbar(mean.index, mean.values, yerr=[mean - g.min(), g.max() - mean],
                    marker="o", capsize=3, color=color, label=f"observed record: {name} channel")
    ax.axhline(sigma_hidden, color="k", ls="--", lw=1, label=r"hidden-chain $\sigma$")
    ax.set_xlabel("KT mixture order $R$")
    ax.set_ylabel("estimate (bits/step)")
    ax.set_ylim(bottom=min(0.0, float(df["estimate"].min()) * 1.1))
    ax.legend(loc="center right")
    ax.set_title("Observed irreversibility of a hidden ring ($n=2^{17}$, 3 seeds)")
    _save(fig, "hmm_order_sweep")


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    plot_markov_convergence()
    plot_calibration()
    plot_hmm_order_sweep()
