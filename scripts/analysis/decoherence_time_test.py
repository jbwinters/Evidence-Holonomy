#!/usr/bin/env python3
"""
Decoherence time prediction test.

Predict t_dec ≈ threshold_bits / <dH/dt> using the Lindblad entropy production
rate (bits/time), and compare against the simulated time to first collapse.
"""

from __future__ import annotations

import argparse
import numpy as np
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts import quantum_density_matrix_holonomy as dm


def main() -> None:
    ap = argparse.ArgumentParser(description="Decoherence time prediction test")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sysdm = dm.DensityMatrixSystem(dim=args.dim, dt=args.dt, seed=args.seed, omega=1.0)
    # Estimate average rate at t=0 from initial state
    srate0 = sysdm.lindblad_entropy_production_rate_bits(sysdm.rho, args.gamma, args.sigma)
    pred_t = (args.threshold / max(1e-12, srate0))

    # Simulate until first collapse
    record, times, S_series, C_series, Hacc_series, Srate_series, collapses = sysdm.evolve_with_collapse(
        n_steps=args.steps, threshold_bits=args.threshold, gamma=args.gamma, sigma=args.sigma,
        holonomy_method="relative_entropy_to_decohered", record_mode="state", measure_U=None,
    )
    # detect first collapse time as first time accumulator exceeds threshold
    t_first = None
    for i, h in enumerate(Hacc_series):
        if h >= args.threshold:
            t_first = times[i]
            break
    print(f"pred_t={pred_t:.6g}  sim_first_collapse_t={t_first if t_first is not None else float('nan')}")


if __name__ == "__main__":
    main()
