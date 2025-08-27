#!/usr/bin/env python3
"""
Diagnostics suite for UEC/KL-rate holonomy assumptions and sanity checks.

Includes:
- Gauge test: bijection → inverse ⇒ holonomy ≈ 0.
- Surrogate test: time-shuffle ⇒ holonomy ≈ 0.
- Markov EP test: synthetic chain with known EP σ; holonomy ≈ σ.

Usage examples:
  python scripts/uec_diagnostics.py --gauge --surrogate --markov
  python scripts/uec_diagnostics.py --surrogate --tail 50000
"""

from __future__ import annotations

import argparse
import math
import os
import random
from typing import List

import numpy as np

# Ensure repo root on path for local module imports
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

try:
    import uec_battery as uec
except Exception as e:  # pragma: no cover
    raise SystemExit(f"uec_battery import required for diagnostics: {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_btc_symbols(path: str, tail: int, k_r: int, k_v: int, jitter_std: float = 0.0):
    import csv
    # Load close and volume
    ts: List[float] = []
    c: List[float] = []
    v: List[float] = []
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        _ = next(rdr, None)
        for row in rdr:
            try:
                ts.append(float(row[0]))
                c.append(float(row[4]))
                v.append(float(row[5]))
            except Exception:
                continue
    order = np.argsort(ts)
    ts = np.array(ts, dtype=float)[order]
    c = np.array(c, dtype=float)[order]
    v = np.array(v, dtype=float)[order]
    if tail and len(ts) > tail:
        ts, c, v = ts[-tail:], c[-tail:], v[-tail:]
    # Returns
    r = np.zeros_like(c)
    r[1:] = np.log(np.maximum(c[1:], 1e-12)) - np.log(np.maximum(c[:-1], 1e-12))
    if jitter_std and jitter_std > 0:
        rng = np.random.default_rng(0)
        r = r + rng.normal(0.0, np.std(r) * float(jitter_std), size=r.shape)
    # Quantile bins
    def qedges(x, k):
        x = x[np.isfinite(x)]
        qs = np.linspace(0, 1, k + 1)
        edges = np.quantile(x, qs)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        return edges
    er = qedges(r, k_r)
    ev = qedges(v, k_v)
    def bindex(x, e):
        idx = np.searchsorted(e, x, side="right") - 1
        return np.clip(idx, 0, len(e) - 2)
    rb = bindex(r, er)
    vb = bindex(v, ev)
    sym = rb * k_v + vb
    return sym.astype(int), int(k_r * k_v)


def gauge_test(symbols: np.ndarray, k: int, R: int) -> float:
    # Construct random bijection P on alphabet [0..k-1]
    perm = list(range(k))
    random.shuffle(perm)
    inv = [0] * k
    for i, p in enumerate(perm):
        inv[p] = i
    class Recode(uec.Transform):
        def __init__(self, mapping: List[int]):
            self.map = mapping
        def apply(self, seq, alphabet):
            return [int(self.map[int(s)]) for s in seq], [int(a) for a in alphabet]
    P = Recode(perm)
    Pinv = Recode(inv)
    loop = [P, Pinv]
    # Apply loop closure at general holonomy
    alphabet = list(range(k))
    return float(uec.klrate_holonomy_general(symbols.tolist(), alphabet, loop, k=k, R=R, align="head"))


def surrogate_test(symbols: np.ndarray, k: int, R: int) -> float:
    rng = np.random.default_rng(0)
    shuf = symbols.copy()
    rng.shuffle(shuf)
    return float(uec.klrate_holonomy_time_reversal_markov(shuf.tolist(), k=k, R=R))


def markov_ep_test(k: int, delta: float, n: int, R: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    T = uec.random_markov_biased(k=k, delta=delta, rng=rng)
    x = uec.sample_markov(T, n=n, rng=rng)
    sigma = uec.entropy_production_rate_bits(T)
    hol = uec.klrate_holonomy_time_reversal_markov(x.tolist(), k=k, R=R)
    return float(sigma), float(hol), float(abs(hol - sigma))


def main():
    ap = argparse.ArgumentParser(description="UEC diagnostics suite")
    ap.add_argument("--csv", default="data/kaggle/btcusd_1min/btcusd_1-min_data.csv")
    ap.add_argument("--tail", type=int, default=50000)
    ap.add_argument("--k_r", type=int, default=8)
    ap.add_argument("--k_v", type=int, default=6)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--gauge", action="store_true")
    ap.add_argument("--surrogate", action="store_true")
    ap.add_argument("--markov", action="store_true")
    ap.add_argument("--ergodic_segments", type=int, default=0, help="Ergodicity probe: segments count (0=skip)")
    ap.add_argument("--jitter_std", type=float, default=0.0, help="Jitter std as multiple of return std (0=off)")
    ap.add_argument("--out", default="results/diagnostics.txt")
    args = ap.parse_args()

    ensure_dir("results")
    logs: List[str] = []

    sym, k = load_btc_symbols(args.csv, args.tail, args.k_r, args.k_v, args.jitter_std)
    logs.append(f"Loaded symbols: n={len(sym)}, k={k}, order={args.order}, jitter_std={args.jitter_std}")

    if args.gauge:
        val = gauge_test(sym, k, args.order)
        logs.append(f"[Gauge] bijection→inverse holonomy≈ {val:.6g} (expect ≈ 0)")

    if args.surrogate:
        val = surrogate_test(sym, k, args.order)
        logs.append(f"[Surrogate] time-shuffle holonomy≈ {val:.6g} (expect ≈ 0)")

    if args.markov:
        sigma, hol, diff = markov_ep_test(k=3, delta=0.6, n=150000, R=args.order)
        logs.append(f"[Markov] analytic EP={sigma:.6g}, holonomy={hol:.6g}, |diff|={diff:.3g}")

    if args.ergodic_segments and args.ergodic_segments > 1:
        segs = int(args.ergodic_segments)
        n = len(sym)
        seg_len = n // segs
        vals = []
        for sidx in range(segs):
            a = sidx * seg_len
            b = (sidx + 1) * seg_len if sidx < segs - 1 else n
            if b - a < 1000:
                continue
            v = gauge_test(sym[a:b], k, args.order)
            vals.append(v)
        if vals:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            logs.append(f"[Ergodicity] gauge across {len(vals)} segments: mean={mean_v:.3g}, std={std_v:.3g}")

    print("\n".join(logs))
    with open(args.out, "w") as f:
        f.write("\n".join(logs) + "\n")
    print(f"Saved diagnostics to {args.out}")


if __name__ == "__main__":
    main()
