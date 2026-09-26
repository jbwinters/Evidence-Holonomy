#!/usr/bin/env python3
"""
LaTeX tables for the paper's sequential-testing results, from anc/seq_*.csv.

Writes tables/seq_validity.tex, tables/seq_detection.tex, tables/seq_efficiency.tex.
Run after scripts/sequential_experiments.py.
"""

from pathlib import Path

import pandas as pd


def tex(name: str) -> str:
    return str(name).replace("#", r"\#").replace("_", r"\_")


def pct(x):
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}\\%"


def validity():
    v = pd.read_csv("anc/seq_validity.csv").assign(kind="Markov")
    m = pd.read_csv("anc/seq_misspec.csv").assign(kind="hidden")
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             r"Reversible process & Type & Loop test & Likelihood & LRT, fixed $n$ & LRT, 10 looks & LRT, 221 checkpoints \\",
             r"\midrule"]
    for _, r in pd.concat([v, m]).iterrows():
        loop = f"{100 * r.loop_anytime:.1f}\\% [{100 * r.loop_lo:.1f}, {100 * r.loop_hi:.1f}]"
        lines.append(f"{tex(r.chain)} & {r.kind} & {loop} & {pct(r.eprocess_anytime)} & {pct(r.lrt_fixed_n)} & "
                     f"{pct(r.lrt_peek_10_looks)} & {pct(r.lrt_peek_every_checkpoint)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path("tables/seq_validity.tex").write_text("\n".join(lines))


def detection():
    d = pd.read_csv("anc/seq_detection.csv").sort_values("rho")
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Chain & $\rho$ & $\log_2(20)/\rho$ & Fixed-$n$ LRT & Loop test (median / 80\%) & Likelihood (80\%) \\",
             r"\midrule"]
    for _, r in d.iterrows():
        lines.append(
            f"{tex(r.chain)} & {r.rho:.4f} & {r['theory_log2(1/a)/rho']:.0f} & {r.lrt_fixed_n80:.0f} & "
            f"{r.loop_tau_median:.0f} / {r.loop_tau_80:.0f} ({r.ratio_loop80_over_lrt80:.1f}$\\times$) & "
            f"{r.e_tau_80:.0f} ({r.ratio_e80_over_lrt80:.1f}$\\times$) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path("tables/seq_detection.tex").write_text("\n".join(lines))


def efficiency():
    e = pd.read_csv("anc/seq_loop_efficiency.csv").sort_values("rho")
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Chain & $\rho$ (bits/step) & Loop-test rate & Efficiency \\", r"\midrule"]
    for _, r in e.iterrows():
        lines.append(f"{tex(r.chain)} & {r.rho:.4f} & {r.loop_rate:.4f} $\\pm$ {r.loop_rate_se:.4f} & "
                     f"{r.efficiency:.3f} $\\pm$ {r.efficiency_se:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    Path("tables/seq_efficiency.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    Path("tables").mkdir(exist_ok=True)
    validity()
    detection()
    efficiency()
    print("Wrote tables/seq_validity.tex, tables/seq_detection.tex, tables/seq_efficiency.tex")
