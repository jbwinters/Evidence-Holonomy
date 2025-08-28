#!/usr/bin/env python3
"""
Sweep channel parameters (amplitude damping, dephasing), dimensions, and record
encodings to map classical KL-rate holonomy vs the accumulator rate and their
ratio (irreversibility efficiency).

Writes a tidy CSV summary under results/ with one row per run.

Examples:
  # Quick default sweep (small grid)
  python scripts/holonomy_channel_sweep.py

  # Custom grid
  python scripts/holonomy_channel_sweep.py \
    --dims 2,3 --gammas 0,0.01,0.02 --sigmas 0,0.1,0.2 \
    --record_modes state,state_quiet,events --steps 40000 --threshold 1.0

Notes:
- Uses the simulation core in scripts/quantum_decoherence_holonomy.py.
- Does not write per-run traces; only a summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from itertools import product
from typing import List

import numpy as np


# Ensure repo root and src/ are on path, and import the simulation module
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_src_root = os.path.join(_repo_root, "src")
for _p in (_repo_root, _src_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scripts import quantum_decoherence_holonomy as qd
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Unable to import simulation core: {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_list_floats(s: str) -> List[float]:
    if not s:
        return []
    out: List[float] = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def parse_list_ints(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(','):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def parse_list_strs(s: str) -> List[str]:
    if not s:
        return []
    return [tok.strip() for tok in s.split(',') if tok.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Holonomy channel sweep")
    ap.add_argument("--dims", type=str, default="2,3")
    ap.add_argument("--gammas", type=str, default="0,0.02")
    ap.add_argument("--sigmas", type=str, default="0,0.2")
    ap.add_argument("--record_modes", type=str, default="state,events")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--coupling", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--out", type=str, default="results/channel_sweep.csv")
    args = ap.parse_args()

    dims = parse_list_ints(args.dims)
    gammas = parse_list_floats(args.gammas)
    sigmas = parse_list_floats(args.sigmas)
    modes = parse_list_strs(args.record_modes)
    seeds = parse_list_ints(args.seeds)
    if not dims or not gammas or not sigmas or not modes or not seeds:
        raise SystemExit("All lists (dims, gammas, sigmas, record_modes, seeds) must be non-empty")

    ensure_dir(os.path.dirname(args.out) or ".")

    # Base configuration
    base_cfg = qd.Config(
        dim=dims[0],
        steps=int(args.steps),
        dt=float(args.dt),
        omega=1.0,
        coupling=float(args.coupling),
        temperature=float(args.temperature),
        threshold_bits=float(args.threshold),
        seed=0,
        R_order=int(args.order),
        out_prefix="results/qdecoh_sweep",
        init_superposition=True,
        include_quiet_symbol=False,
        record_mode="state",
        amp_damp_gamma=0.0,
        amp_damp_target=0,
        dephase_sigma=0.0,
        ramp_window=200,
    )

    rows = []
    total = len(dims) * len(gammas) * len(sigmas) * len(modes) * len(seeds)
    done = 0
    for dim, gamma, sigma, mode, seed in product(dims, gammas, sigmas, modes, seeds):
        cfg = replace(
            base_cfg,
            dim=int(dim),
            amp_damp_gamma=float(gamma),
            dephase_sigma=float(sigma),
            record_mode=str(mode),
            seed=int(seed),
        )
        summary, _, _, _, _ = qd.simulate(cfg)
        rows.append({
            "dim": summary["dim"],
            "record_mode": summary["record_mode"],
            "gamma_amp_damp": float(summary["amp_damp_gamma"]),
            "sigma_dephase": float(sigma),
            "steps": summary["steps"],
            "dt": summary["dt"],
            "threshold": summary["threshold_bits"],
            "coupling": summary["coupling"],
            "temperature": summary["temperature"],
            "seed": summary["seed"],
            "collapses": summary["collapses"],
            "record_k": summary["record_alphabet_k"],
            "hol_rate_bits_per_step": summary["klrate_holonomy_bits_per_step"],
            "avg_accum_bits_per_step": summary["avg_accum_bits_per_step"],
            "irreversibility_efficiency": summary["irreversibility_efficiency"],
        })
        done += 1
        if done % 10 == 0 or done == total:
            print(f"[Sweep] {done}/{total} completed…")

    # Write CSV
    fieldnames = [
        "dim", "record_mode", "gamma_amp_damp", "sigma_dephase",
        "steps", "dt", "threshold", "coupling", "temperature", "seed",
        "collapses", "record_k", "hol_rate_bits_per_step",
        "avg_accum_bits_per_step", "irreversibility_efficiency",
    ]
    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"Saved sweep CSV to {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

