#!/usr/bin/env python3
"""
Figures for the anytime-valid reversibility test, from anc/seq_*.csv.

Writes figures/seq_validity, seq_detection, seq_growth, and seq_rates (PDF and PNG).
Run after scripts/sequential_experiments.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9})
ALPHA = 0.05


def _save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/{name}.pdf")


def plot_validity():
    v = pd.read_csv("anc/seq_validity.csv").assign(group="Markov")
    m = pd.read_csv("anc/seq_misspec.csv").assign(group="hidden (not Markov)")
    df = pd.concat([v, m], ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(df))[::-1]
    h = 0.16
    series = [("loop_anytime", "loop test, checked after every step", "#264653"),
              ("eprocess_anytime", "likelihood e-process, every checkpoint", "#2a9d8f"),
              ("lrt_fixed_n", "LRT at the planned n only", "#8d99ae"),
              ("lrt_peek_10_looks", "LRT, checked 10 times", "#e9c46a"),
              ("lrt_peek_every_checkpoint", "LRT, checked at every checkpoint", "#e76f51")]
    for j, (col, label, color) in enumerate(series):
        vals = df[col].to_numpy(dtype=float)
        ax.barh(y + (2 - j) * h, np.nan_to_num(vals), height=h, color=color, label=label)
    for col, off in (("loop", 2), ("eprocess", 1)):
        val = df[f"{col}_anytime"] if col == "loop" else df["eprocess_anytime"]
        ax.errorbar(val, y + off * h, xerr=[val - df[f"{col}_lo"], df[f"{col}_hi"] - val], fmt="none",
                    ecolor="black", capsize=2, lw=1)
    ax.axvline(ALPHA, color="black", ls="--", lw=1)
    ax.text(ALPHA, y.max() + 0.55, " α = 0.05", fontsize=9, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  [{g}]" for c, g in zip(df["chain"], df["group"])], fontsize=8.5)
    ax.set_xlabel("fraction of reversible trajectories flagged as irreversible (n = 20,000)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("False alarms when the process is reversible")
    _save(fig, "seq_validity")


def plot_detection():
    d = pd.read_csv("anc/seq_detection.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(d["lrt_fixed_n80"], d["e_tau_80"], color="#2a9d8f", zorder=3, label="likelihood e-process")
    ax.scatter(d["lrt_fixed_n80"], d["loop_tau_80"], color="#264653", marker="s", zorder=3,
               label="loop test")
    for _, r in d.iterrows():
        ax.plot([r["lrt_fixed_n80"]] * 2, [r["loop_tau_80"], r["e_tau_80"]], color="#bbbbbb", lw=1, zorder=1)
        ax.annotate(r["chain"], (r["lrt_fixed_n80"], r["loop_tau_80"]), fontsize=7, xytext=(4, -8),
                    textcoords="offset points")
    lo = min(d["lrt_fixed_n80"].min(), d["loop_tau_80"].min()) * 0.7
    hi = max(d["lrt_fixed_n80"].max(), d["e_tau_80"].max()) * 1.4
    xs = np.array([lo, hi])
    ax.plot(xs, xs, "k--", lw=1, label="equal sample size")
    ax.plot(xs, 2 * xs, ":", color="gray", lw=1, label="2× and 4×")
    ax.plot(xs, 4 * xs, ":", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("fixed-n LRT: n for 80% power")
    ax.set_ylabel("anytime test: steps until 80% have rejected")
    ax.legend(loc="upper left")
    ax.set_title("Price of anytime validity (α = 0.05)")
    _save(fig, "seq_detection")


def plot_growth():
    g = pd.read_csv("anc/seq_growth.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = ["#e76f51", "#2a9d8f", "#264653", "#e9c46a"]
    for (chain, sub), c in zip(g.groupby("chain", sort=False), colors):
        ax.plot(sub["t"], sub["rate_mean"], color=c, lw=2, label=chain)
        ax.plot(sub["t"], sub["loop_rate_mean"], color=c, lw=1.2, ls="-.")
        ax.fill_between(sub["t"], sub["rate_lo"], sub["rate_hi"], color=c, alpha=0.15)
        ax.axhline(sub["rho"].iloc[0], color=c, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("steps t")
    ax.set_ylabel("log₂(E_t) / t   (bits/step)")
    ax.set_title("Evidence rate: likelihood e-process (solid) → ρ (dashed);\nloop test (dash-dot)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "seq_growth")


def plot_rates():
    r = pd.read_csv("anc/seq_rates.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(r["sigma"], r["rho"], s=10, alpha=0.7, color="#264653")
    xs = np.linspace(0, r["sigma"].max() * 1.05, 50)
    ax.plot(xs, xs / 2, "k--", lw=1, label="σ/2 (upper bound)")
    ax.plot(xs, xs / 4, ":", color="gray", lw=1.2, label="σ/4 (near equilibrium)")
    ax.set_xlabel("entropy production σ (bits/step)")
    ax.set_ylabel("ρ, distance to reversibility (bits/step)")
    ax.legend(loc="upper left")
    ax.set_title("ρ vs σ for 300 random chains")
    _save(fig, "seq_rates")


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    plot_rates()
    plot_growth()
    plot_validity()
    plot_detection()
