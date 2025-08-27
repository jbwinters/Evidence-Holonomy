#!/usr/bin/env python3
"""
Sensitivity grid for discretization (k_r, k_v) and KT order R.

Computes UEC (counts or KT) over BTC returns×volume and reports median/mean/std
per configuration. Outputs CSV for downstream plotting.

Example:
  python scripts/uec_sensitivity.py --tail 50000 --k_list 6,8,12,16 --r_list 1,2,3 --method counts
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List, Tuple

import numpy as np

import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

try:
    import uec_battery as uec
except Exception:
    uec = None

from btc_uec_analysis import (
    load_btc_csv,
    feature_block,
    symbolize_joint,
    compute_uec_stream,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_config(ts, o, h, l, c, v, k_r: int, k_v: int, R: int, W: int, method: str) -> Tuple[float, float, float, float]:
    r, body, wick, rv, skew, v_z = feature_block(ts, o, h, l, c, v)
    sym, _, _ = symbolize_joint(r, v, k_r, k_v)
    k = int(k_r * k_v)
    t0 = time.time()
    uec_vals, uec_z = compute_uec_stream(sym, k=k, W=W, R=R, method=method, kt_stride=8)
    dt = time.time() - t0
    vals = uec_vals[np.isfinite(uec_vals)]
    return float(np.median(vals)), float(np.mean(vals)), float(np.std(vals)), float(dt)


def main():
    ap = argparse.ArgumentParser(description="UEC discretization/order sensitivity grid")
    ap.add_argument("--csv", default="data/kaggle/btcusd_1min/btcusd_1-min_data.csv")
    ap.add_argument("--tail", type=int, default=50000)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--k_list", default="6,8,12,16")
    ap.add_argument("--r_list", default="1,2,3")
    ap.add_argument("--method", choices=["counts", "kt"], default="counts")
    ap.add_argument("--out", default="results/uec_sensitivity.csv")
    args = ap.parse_args()

    ensure_dir("results")
    ts, o, h, l, c, v = load_btc_csv(args.csv, tail=args.tail, limit=None)
    ks = [int(x) for x in args.k_list.split(",") if x]
    rs = [int(x) for x in args.r_list.split(",") if x]

    rows: List[List[object]] = []
    for k_r in ks:
        for k_v in ks:
            for R in rs:
                med, mean, std, dt = run_config(ts, o, h, l, c, v, k_r, k_v, R, args.W, args.method)
                rows.append([k_r, k_v, R, med, mean, std, dt])
                print(f"k_r={k_r} k_v={k_v} R={R} → med={med:.4g} std={std:.4g} time={dt:.2f}s")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k_r", "k_v", "R", "median_uec", "mean_uec", "std_uec", "seconds"])
        w.writerows(rows)
    print(f"Saved sensitivity grid to {args.out}")


if __name__ == "__main__":
    main()

