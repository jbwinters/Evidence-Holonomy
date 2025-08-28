#!/usr/bin/env python3
"""
Basis selection test for density-matrix holonomy.

Runs the density-matrix simulation under different measurement bases and
compares classical record holonomy (KL-rate), quantum accumulators, and
efficiency. Bases:
- computational (I)
- fourier (DFT)
- random_unitary (Haar-approx via QR of complex Ginibre)

Writes a small CSV summary and prints a concise comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict

import numpy as np

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from scripts import quantum_density_matrix_holonomy as dm
    from uec.holonomy import klrate_holonomy_time_reversal_markov
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Imports failed: {e}")


def fourier_unitary(d: int) -> np.ndarray:
    j, k = np.meshgrid(np.arange(d), np.arange(d))
    W = np.exp(2j * np.pi * j * k / d) / np.sqrt(d)
    return W.astype(np.complex128)


def random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # Make Q unitary with positive diag of R
    ph = np.diag(R)
    ph = ph / np.abs(ph)
    U = Q @ np.diag(ph.conj())
    return U.astype(np.complex128)


def main() -> None:
    ap = argparse.ArgumentParser(description="Basis selection test")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--method", type=str, default="relative_entropy_to_decohered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/basis_selection.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bases: Dict[str, np.ndarray | None] = {
        "computational": None,
        "fourier": fourier_unitary(args.dim),
        "random_unitary": random_unitary(args.dim, rng),
    }

    rows = []
    for name, U in bases.items():
        sys = dm.DensityMatrixSystem(dim=args.dim, dt=args.dt, seed=args.seed, omega=1.0)
        record, times, S_series, C_series, Hacc_series, Srate_series, collapses = sys.evolve_with_collapse(
            n_steps=args.steps,
            threshold_bits=args.threshold,
            gamma=args.gamma,
            sigma=args.sigma,
            holonomy_method=args.method,
            record_mode="state",
            measure_U=U,
        )
        hol_rate = klrate_holonomy_time_reversal_markov(record, k=args.dim, R=3)
        avg_accum = sys.accum_total_bits / max(1, args.steps)
        eff = 0.0 if avg_accum <= 0 else hol_rate / avg_accum
        rows.append({
            "basis": name,
            "dim": args.dim,
            "gamma": args.gamma,
            "sigma": args.sigma,
            "hol_rate_bits_per_step": float(hol_rate),
            "avg_accum": float(avg_accum),
            "efficiency": float(eff),
            "collapses": int(collapses),
        })
        print(f"[Basis {name}] hol={hol_rate:.6g} eff={eff:.6g} collapses={collapses}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"Saved basis selection results to {args.out}")


if __name__ == "__main__":
    main()

