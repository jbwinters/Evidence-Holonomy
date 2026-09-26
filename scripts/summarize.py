#!/usr/bin/env python3
"""
Summarize the ancillary CSVs in anc/ and write the paper's LaTeX tables.

Writes tables/table_markov_sanity.tex (included by uec_theory.tex) and prints
the statistics quoted in the paper's numerical section.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def table_markov_sanity() -> None:
    df = pd.read_csv("anc/markov_sanity.csv")
    df["abs_rel"] = (df["estimate"] - df["sigma_true"]).abs() / df["sigma_true"]
    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Absolute relative error $|\hat\sigma-\sigma|/\sigma$ of the time-reversal estimate"
        r" ($R=3$) on 15 random biased chains (five each with 3, 4, and 5 states), by sample length $n$."
        r" Every row uses the same 15 chains.}",
        r"\label{tab:markov}",
        r"\begin{tabular}{rccc}",
        r"\toprule",
        r"$n$ & median & interquartile range & max \\",
        r"\midrule",
    ]
    for n, g in df.groupby("n"):
        q1, med, q3 = np.quantile(g["abs_rel"], [0.25, 0.5, 0.75])
        lines.append(
            f"$2^{{{int(np.log2(n))}}}$ & {100 * med:.1f}\\% & [{100 * q1:.1f}\\%, {100 * q3:.1f}\\%]"
            f" & {100 * g['abs_rel'].max():.0f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path("tables").mkdir(exist_ok=True)
    Path("tables/table_markov_sanity.tex").write_text("\n".join(lines))
    print("Wrote tables/table_markov_sanity.tex")
    print(df.groupby("n")["abs_rel"].describe()[["25%", "50%", "75%", "max"]].to_string())
    print("sigma range:", df["sigma_true"].min(), df["sigma_true"].max())


def calibration_stats() -> None:
    df = pd.read_csv("anc/calibration.csv")
    err = df["estimate"] - df["sigma_true"]
    rel = err.abs() / df["sigma_true"]
    r = np.corrcoef(df["sigma_true"], df["estimate"])[0, 1]
    print(f"calibration: r={r:.4f}  median |err|={err.abs().median():.4f}  max |err|={err.abs().max():.4f}  "
          f"median rel={rel.median():.3%}  max rel={rel.max():.1%}  "
          f"sigma range=[{df['sigma_true'].min():.4f}, {df['sigma_true'].max():.3f}]")
    big = df[df["sigma_true"] >= 0.05]
    big_rel = (big["estimate"] - big["sigma_true"]).abs() / big["sigma_true"]
    print(f"  chains with sigma>=0.05: {len(big)}, median rel={big_rel.median():.3%}, max rel={big_rel.max():.1%}")


def reversible_stats() -> None:
    df = pd.read_csv("anc/reversible_bias.csv")
    print("reversible chains (sigma=0): estimate by k, R, n")
    print(df.groupby(["k", "order", "n"])["estimate"].agg(["mean", "max"]).to_string())


def hmm_stats() -> None:
    df = pd.read_csv("anc/hmm_order_sweep.csv")
    print(f"hidden sigma = {df['sigma_hidden'].iloc[0]:.4f}")
    print(df.groupby(["channel", "order"])["estimate"].agg(["mean", "min", "max"]).to_string())


if __name__ == "__main__":
    table_markov_sanity()
    calibration_stats()
    reversible_stats()
    hmm_stats()
