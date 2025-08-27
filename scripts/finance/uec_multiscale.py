#!/usr/bin/env python3
"""
Multi-scale curvature spectrum and attribution.

Computes:
- Spectrum: UEC (counts) after downsampling by s ∈ {1,2,4,8} on BTC minute bars.
- Coarse-grain loop holonomy (Downsample→Upsample) via uec_battery general holonomy.
- Attribution: joint (r×v) vs returns-only vs volume-only at base scale.

Outputs CSVs in results/: uec_multiscale.csv, uec_attribution.csv
"""

from __future__ import annotations

import argparse
import csv
import os
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
    quantile_edges,
    bin_index,
    compute_uec_stream,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def downsample_series(r: np.ndarray, v: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    n = (len(r) // step) * step
    r2 = r[:n].reshape(-1, step).sum(axis=1)
    v2 = v[:n].reshape(-1, step).sum(axis=1)
    return r2, v2


def symbolize_rv(r: np.ndarray, v: np.ndarray, k_r: int, k_v: int) -> Tuple[np.ndarray, int]:
    er = quantile_edges(r, k_r)
    ev = quantile_edges(v, k_v)
    rb = bin_index(r, er)
    vb = bin_index(v, ev)
    sym = rb * k_v + vb
    return sym.astype(int), int(k_r * k_v)


def holonomy_coarse_grain_loop(symbols: np.ndarray, k: int, step: int, R: int) -> float:
    if uec is None:
        return float('nan')
    alphabet = list(range(k))
    loop = [uec.Downsample(step=step), uec.UpsampleRepeat(step=step)]
    return float(uec.klrate_holonomy_general(symbols.tolist(), alphabet, loop, k=k, R=R, align="head"))


def main():
    ap = argparse.ArgumentParser(description="UEC multi-scale spectrum and attribution")
    ap.add_argument("--csv", default="data/kaggle/btcusd_1min/btcusd_1-min_data.csv")
    ap.add_argument("--tail", type=int, default=50000)
    ap.add_argument("--k_r", type=int, default=12)
    ap.add_argument("--k_v", type=int, default=8)
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--scales", default="1,2,4,8")
    ap.add_argument("--out_spec", default="results/uec_multiscale.csv")
    ap.add_argument("--out_attr", default="results/uec_attribution.csv")
    args = ap.parse_args()

    ensure_dir("results")
    ts, o, h, l, c, vraw = load_btc_csv(args.csv, tail=args.tail, limit=None)
    r, body, wick, rv, skew, vz = feature_block(ts, o, h, l, c, vraw)

    # Spectrum across downsample factors
    scales = [int(s) for s in args.scales.split(",") if s]
    spec_rows: List[List[object]] = []
    for s in scales:
        r2, v2 = downsample_series(r[1:], vraw[1:], s)  # align after first return
        sym, k = symbolize_rv(r2, v2, args.k_r, args.k_v)
        uec_vals, _ = compute_uec_stream(sym, k=k, W=max(8, args.W // max(1, s)), R=args.R, method="counts", kt_stride=8)
        vals = uec_vals[np.isfinite(uec_vals)]
        med = float(np.median(vals)) if vals.size else float('nan')
        # Coarse-grain loop holonomy (Downsample→Upsample)
        cg = holonomy_coarse_grain_loop(sym, k, step=s, R=args.R) if s > 1 else 0.0
        spec_rows.append([s, k, med, cg])
        print(f"scale={s} k={k} → median_uec={med:.4g} coarse_grain={cg:.4g}")

    with open(args.out_spec, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scale", "k", "median_uec", "coarse_grain_holonomy"])
        w.writerows(spec_rows)
    print(f"Saved spectrum to {args.out_spec}")

    # Attribution at base scale s=1
    sym_joint, k_joint = symbolize_rv(r[1:], vraw[1:], args.k_r, args.k_v)
    ej = float(np.nanmedian(compute_uec_stream(sym_joint, k=k_joint, W=args.W, R=args.R, method="counts", kt_stride=8)[0]))
    # returns-only
    er = quantile_edges(r[1:], args.k_r)
    rb = bin_index(r[1:], er)
    kr = int(args.k_r)
    ej_r = float(np.nanmedian(compute_uec_stream(rb.astype(int), k=kr, W=args.W, R=args.R, method="counts", kt_stride=8)[0]))
    # volume-only
    ev = quantile_edges(vraw[1:], args.k_v)
    vb = bin_index(vraw[1:], ev)
    kv = int(args.k_v)
    ej_v = float(np.nanmedian(compute_uec_stream(vb.astype(int), k=kv, W=args.W, R=args.R, method="counts", kt_stride=8)[0]))

    with open(args.out_attr, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "k", "median_uec"])
        w.writerow(["joint_r_x_v", k_joint, ej])
        w.writerow(["returns_only", kr, ej_r])
        w.writerow(["volume_only", kv, ej_v])
    print(f"Saved attribution to {args.out_attr}")


if __name__ == "__main__":
    main()

