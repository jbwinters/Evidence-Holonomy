"""
Command-line interfaces:

- uec-battery: quick demonstration that KL-rate holonomy ≈ entropy production
  on finite-state Markov chains using the canonical time-reversal loop.

- uec-aot: Arrow-of-Time demos for CSV/WAV time-series. Supports single-file
  runs and folder scoreboards; exposes preprocessing flags and random seeds.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np
from .markov import random_markov_biased, sample_markov, entropy_production_rate_bits
from .holonomy import klrate_holonomy_time_reversal_markov
from .aot import aot_from_series, load_csv_column, load_wav_mono


def run_battery(argv: list[str] | None = None) -> None:
    """Minimal EP ≈ KL-holonomy demonstration with adjustable sample size."""
    p = argparse.ArgumentParser(prog="uec-battery", description="UEC battery: tests and validations")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--n", type=int, default=150_000)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args(argv)

    if args.fast:
        args.n = max(60_000, args.n // 3)

    print("\n=== UEC Battery (minimal): starting ===")
    # Time-reversal ≈ EP
    T = random_markov_biased(k=args.k, delta=0.6)
    sigma_bits = entropy_production_rate_bits(T)
    x = sample_markov(T, n=args.n)
    hol_rate = klrate_holonomy_time_reversal_markov(x, k=args.k, R=args.order)
    diff = abs(hol_rate - sigma_bits)
    rel = diff / max(1e-8, abs(sigma_bits))
    print(
        f"[Time reversal] EP analytic={sigma_bits:.6g}  KL-rate hol={hol_rate:.6g}  "
        f"abs diff={diff:.3g}  rel diff={rel:.3%}"
    )
    print("=== UEC Battery (minimal): done ===")


def run_aot(argv: list[str] | None = None) -> None:
    """Run AoT demos on WAV/CSV or build a scoreboard from a glob of files."""
    p = argparse.ArgumentParser(prog="uec-aot", description="UEC Arrow-of-Time demos")
    p.add_argument("--aot_csv", type=str)
    p.add_argument("--aot_csv_col", type=str, default="0")
    p.add_argument("--aot_wav", type=str)
    p.add_argument("--aot_bins", type=int, default=8)
    p.add_argument("--aot_win", type=int, default=4096)
    p.add_argument("--aot_stride", type=int, default=2048)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--aot_diff", action="store_true")
    p.add_argument("--aot_logreturn", action="store_true")
    p.add_argument("--aot_rate", type=float)
    p.add_argument("--scoreboard_glob", type=str)
    p.add_argument("--seed", type=int, help="RNG seed for reproducible AoT CIs")
    args = p.parse_args(argv)

    if args.aot_csv:
        col = args.aot_csv_col
        try:
            col = int(col)
        except Exception:
            pass
        x = load_csv_column(args.aot_csv, column=col)
        rng = np.random.default_rng(args.seed) if args.seed is not None else None
        res = aot_from_series(
            x,
            k=args.aot_bins,
            R=args.order,
            sr=int(args.aot_rate) if args.aot_rate else None,
            win=args.aot_win,
            stride=args.aot_stride,
            use_diff=bool(args.aot_diff),
            use_logreturn=bool(args.aot_logreturn),
            rng=rng,
        )
        print(
            f"[AoT CSV] AUC={res['auc']:.3f}  bits/step={res['bits_per_step']:.6g}  "
            f"CI=[{res['hol_ci_lo']:.6g},{res['hol_ci_hi']:.6g}]"
        )
        print(json.dumps({"file": args.aot_csv, **res}))
        return

    if args.aot_wav:
        x, sr = load_wav_mono(args.aot_wav)
        rng = np.random.default_rng(args.seed) if args.seed is not None else None
        res = aot_from_series(
            x,
            k=args.aot_bins,
            R=args.order,
            sr=sr,
            win=args.aot_win,
            stride=args.aot_stride,
            use_diff=bool(args.aot_diff),
            use_logreturn=False,
            rng=rng,
        )
        bps = res["bits_per_second"]
        print(
            f"[AoT WAV] AUC={res['auc']:.3f}  bits/step={res['bits_per_step']:.6g}  "
            f"bits/s={bps if bps is not None else 'NA'}  "
            f"CI=[{res['hol_ci_lo']:.6g},{res['hol_ci_hi']:.6g}]  sr={sr}Hz"
        )
        print(json.dumps({"file": args.aot_wav, **res}))
        return

    if args.scoreboard_glob:
        import glob
        rows = []
        base_rng = np.random.default_rng(args.seed) if args.seed is not None else None
        for idx, path in enumerate(glob.glob(args.scoreboard_glob)):
            try:
                if path.lower().endswith(".wav"):
                    x, sr = load_wav_mono(path)
                    rng = (None if base_rng is None else np.random.default_rng(base_rng.integers(0, 2**32)))
                    res = aot_from_series(
                        x,
                        k=args.aot_bins,
                        R=args.order,
                        sr=sr,
                        win=args.aot_win,
                        stride=args.aot_stride,
                        use_diff=bool(args.aot_diff),
                        use_logreturn=False,
                        rng=rng,
                    )
                else:
                    x = load_csv_column(path, column=args.aot_csv_col)
                    rng = (None if base_rng is None else np.random.default_rng(base_rng.integers(0, 2**32)))
                    res = aot_from_series(
                        x,
                        k=args.aot_bins,
                        R=args.order,
                        sr=int(args.aot_rate) if args.aot_rate else None,
                        win=args.aot_win,
                        stride=args.aot_stride,
                        use_diff=bool(args.aot_diff),
                        use_logreturn=bool(args.aot_logreturn),
                        rng=rng,
                    )
                rows.append({
                    "file": path,
                    "auc": res["auc"],
                    "bits_per_step": res["bits_per_step"],
                    "bits_per_second": res["bits_per_second"],
                    "ci_lo": res["hol_ci_lo"],
                    "ci_hi": res["hol_ci_hi"],
                })
                print(f"[Scoreboard] {os.path.basename(path)} AUC={res['auc']:.3f} b/step={res['bits_per_step']:.4g}")
            except Exception as e:
                print(f"[Scoreboard] Skipped {path}: {e}")
        if rows:
            print(json.dumps(rows))
        return

    # if nothing provided, show help
    p.print_help(sys.stdout)
