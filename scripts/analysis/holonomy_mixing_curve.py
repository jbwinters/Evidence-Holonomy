#!/usr/bin/env python3
"""
Efficiency vs mixing-angle sweep between amplitude damping and pure dephasing.

Parameterization:
  gamma(theta)  = amp_scale * cos(theta)
  sigma(theta)  = dephase_scale * sin(theta)

Notes:
- Units differ (gamma is a probability per step, sigma is phase std in radians);
  this curve is exploratory and intended to map qualitative trends, not an
  absolute channel equivalence.
- Uses record_mode=state by default on dim≥3 so record holonomy can be non-zero.

Outputs a CSV with columns: theta, gamma, sigma, hol_rate, avg_accum_rate,
efficiency, collapses, etc.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import replace

# Import simulator
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
try:
    from scripts import quantum_decoherence_holonomy as qd
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Unable to import simulation core: {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Mixing-angle curve between damping and dephasing")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--n_points", type=int, default=17)
    ap.add_argument("--amp_scale", type=float, default=0.02, help="Max amplitude-damping rate per step")
    ap.add_argument("--dephase_scale", type=float, default=0.2, help="Max dephasing std per step (radians)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--coupling", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--record_mode", type=str, default="state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/mixing_curve.csv")
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.out) or ".")

    base = qd.Config(
        dim=int(args.dim),
        steps=int(args.steps),
        dt=float(args.dt),
        omega=1.0,
        coupling=float(args.coupling),
        temperature=float(args.temperature),
        threshold_bits=float(args.threshold),
        seed=int(args.seed),
        R_order=int(args.order),
        out_prefix="results/qdecoh_mix",
        init_superposition=True,
        include_quiet_symbol=False,
        record_mode=str(args.record_mode),
        amp_damp_gamma=0.0,
        amp_damp_target=0,
        dephase_sigma=0.0,
        ramp_window=200,
    )

    rows = []
    # theta ∈ [0, π/2] from pure damping (theta=0) to pure dephasing (theta=π/2)
    for i in range(max(2, int(args.n_points))):
        theta = (math.pi / 2.0) * (i / max(1, args.n_points - 1))
        gamma = float(args.amp_scale) * math.cos(theta)
        sigma = float(args.dephase_scale) * math.sin(theta)
        cfg = replace(base, amp_damp_gamma=gamma, dephase_sigma=sigma)
        summary, *_ = qd.simulate(cfg)
        rows.append({
            "theta": theta,
            "gamma": gamma,
            "sigma": sigma,
            "dim": summary["dim"],
            "record_mode": summary["record_mode"],
            "hol_rate_bits_per_step": summary["klrate_holonomy_bits_per_step"],
            "avg_accum_bits_per_step": summary["avg_accum_bits_per_step"],
            "irreversibility_efficiency": summary["irreversibility_efficiency"],
            "collapses": summary["collapses"],
            "steps": summary["steps"],
            "dt": summary["dt"],
        })
        print(f"theta={theta:.3f} gamma={gamma:.4g} sigma={sigma:.4g} hol={rows[-1]['hol_rate_bits_per_step']:.4g} eff={rows[-1]['irreversibility_efficiency']:.4g}")

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"Saved mixing curve to {args.out} ({len(rows)} points)")


if __name__ == "__main__":
    main()

