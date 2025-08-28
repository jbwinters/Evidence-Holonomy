#!/usr/bin/env python3
"""
Channel space mapping for density-matrix holonomy.

Sweeps gamma (damping) and sigma (dephasing) on a grid for dims in {2,3,4}
using the relative-entropy-to-decohered accumulator and records in the state
basis. Writes a tidy CSV with:

- hol_rate_bits_per_step (classical record holonomy)
- avg_accum_bits_per_step (quantum decoherence rate proxy)
- efficiency (ratio)
- collapses, steady_coherence_l1, avg_lindblad_srate_bits_per_time

By default uses steps=20000 for reasonable runtime; adjust as needed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace


_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    from scripts import quantum_density_matrix_holonomy as dm
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Unable to import density-matrix core: {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Channel space mapping")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--dims", type=str, default="2,3,4")
    ap.add_argument("--gammas", type=int, default=11, help="Grid points for gamma in [0,0.05]")
    ap.add_argument("--sigmas", type=int, default=11, help="Grid points for sigma in [0,0.5]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/channel_space_map.csv")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(',') if x.strip()]

    import numpy as np
    gamma_grid = np.linspace(0.0, 0.05, int(args.gammas))
    sigma_grid = np.linspace(0.0, 0.5, int(args.sigmas))

    base = {
        "steps": int(args.steps),
        "dt": float(args.dt),
        "omega": 1.0,
        "threshold_bits": float(args.threshold),
        "method": "relative_entropy_to_decohered",
        "seed": int(args.seed),
        "out_prefix": "results/dm_map",
    }

    rows = []
    total = len(dims) * len(gamma_grid) * len(sigma_grid)
    done = 0
    for dim in dims:
        for gamma in gamma_grid:
            for sigma in sigma_grid:
                res = dm.run_single(
                    dim=dim,
                    steps=base["steps"],
                    dt=base["dt"],
                    omega=base["omega"],
                    gamma=float(gamma),
                    sigma=float(sigma),
                    threshold_bits=base["threshold_bits"],
                    method=base["method"],
                    seed=base["seed"],
                    out_prefix=base["out_prefix"],
                )
                rows.append({
                    **res,
                })
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"[Map] {done}/{total} completed…")

    ensure_dir(os.path.dirname(args.out) or ".")
    # Order fields
    fields = [
        "dim", "steps", "dt", "threshold_bits", "gamma", "sigma",
        "hol_rate_bits_per_step", "avg_accum_bits_per_step", "efficiency",
        "collapses", "steady_coherence_l1", "avg_lindblad_srate_bits_per_time",
        "csv",
    ]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in fields})
    print(f"Saved channel space map to {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

