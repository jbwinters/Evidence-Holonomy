#!/usr/bin/env python3
"""
UEC quantum analysis helper.

Loads CSVs produced by the sweep scripts and prints summary stats that can
feed into the REPORT.md. Designed to work without plotting libraries.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from statistics import mean


def load_csv(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            out.append(row)
    return out


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def summarize_channel_map(paths: list[str]) -> None:
    rows: list[dict] = []
    for p in paths:
        rows.extend(load_csv(p))
    if not rows:
        print("No channel-space CSVs found.")
        return
    # Group by dim
    for dim in sorted(set(int(r["dim"]) for r in rows)):
        subset = [r for r in rows if int(r["dim"]) == dim]
        eff = [to_float(r["efficiency"]) for r in subset]
        hol = [to_float(r["hol_rate_bits_per_step"]) for r in subset]
        accum = [to_float(r["avg_accum_bits_per_step"]) for r in subset]
        print(f"[ChannelMap dim={dim}] n={len(subset)} eff_mean={mean(eff):.3g} hol_mean={mean(hol):.3g} accum_mean={mean(accum):.3g}")
        # Extremes
        top = max(subset, key=lambda r: to_float(r["efficiency"]))
        bot = min(subset, key=lambda r: to_float(r["efficiency"]))
        print(f"  best eff: gamma={top['gamma']} sigma={top['sigma']} eff={to_float(top['efficiency']):.3g} hol={to_float(top['hol_rate_bits_per_step']):.3g}")
        print(f"  worst eff: gamma={bot['gamma']} sigma={bot['sigma']} eff={to_float(bot['efficiency']):.3g} hol={to_float(bot['hol_rate_bits_per_step']):.3g}")


def summarize_basis(paths: list[str]) -> None:
    rows: list[dict] = []
    for p in paths:
        rows.extend(load_csv(p))
    if not rows:
        print("No basis-selection CSVs found.")
        return
    from collections import defaultdict
    by_basis = defaultdict(list)
    for r in rows:
        by_basis[r["basis"]].append(r)
    for b, lst in by_basis.items():
        eff = mean(to_float(r["efficiency"]) for r in lst)
        hol = mean(to_float(r["hol_rate_bits_per_step"]) for r in lst)
        print(f"[Basis {b}] n={len(lst)} eff_mean={eff:.3g} hol_mean={hol:.3g}")


def main() -> None:
    ap = argparse.ArgumentParser(description="UEC quantum analysis helper")
    ap.add_argument("--channel_maps_glob", type=str, default="results/channel_space_*.csv")
    ap.add_argument("--basis_glob", type=str, default="results/basis_selection*.csv")
    args = ap.parse_args()

    ch_paths = glob.glob(args.channel_maps_glob)
    bs_paths = glob.glob(args.basis_glob)

    summarize_channel_map(ch_paths)
    summarize_basis(bs_paths)


if __name__ == "__main__":
    main()

