#!/usr/bin/env python3
"""
UEC performance benchmark: counts vs KT across (W, k, R).

Outputs results/uec_bench.csv with timing per configuration.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import List

import numpy as np

import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

try:
    import uec_battery as uec
except Exception:
    uec = None

from btc_uec_analysis import load_btc_csv, feature_block, symbolize_joint, compute_uec_stream


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_once(sym: np.ndarray, k: int, W: int, R: int, method: str) -> float:
    t0 = time.time()
    _vals, _z = compute_uec_stream(sym, k=k, W=W, R=R, method=method, kt_stride=8)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="UEC benchmark counts vs KT")
    ap.add_argument("--csv", default="data/kaggle/btcusd_1min/btcusd_1-min_data.csv")
    ap.add_argument("--tail", type=int, default=20000)
    ap.add_argument("--k_r", type=int, default=8)
    ap.add_argument("--k_v", type=int, default=6)
    ap.add_argument("--W_list", default="128,256")
    ap.add_argument("--R_list", default="1,2,3")
    ap.add_argument("--out", default="results/uec_bench.csv")
    args = ap.parse_args()

    ensure_dir("results")
    ts, o, h, l, c, v = load_btc_csv(args.csv, tail=args.tail, limit=None)
    r, body, wick, rv, skew, v_z = feature_block(ts, o, h, l, c, v)
    sym, _, _ = symbolize_joint(r, v, args.k_r, args.k_v)
    k = int(args.k_r * args.k_v)

    Ws = [int(x) for x in args.W_list.split(",") if x]
    Rs = [int(x) for x in args.R_list.split(",") if x]
    rows: List[List[object]] = []
    for W in Ws:
        for R in Rs:
            dt_counts = run_once(sym, k, W, R, method="counts")
            if uec is not None:
                dt_kt = run_once(sym, k, W, R, method="kt")
            else:
                dt_kt = float('nan')
            rows.append([k, W, R, dt_counts, dt_kt])
            print(f"k={k} W={W} R={R} → counts={dt_counts:.2f}s kt={dt_kt:.2f}s")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "W", "R", "seconds_counts", "seconds_kt"])
        w.writerows(rows)
    print(f"Saved benchmark to {args.out}")


if __name__ == "__main__":
    main()

