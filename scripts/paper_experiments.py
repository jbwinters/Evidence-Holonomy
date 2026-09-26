#!/usr/bin/env python3
"""
Generate the ancillary data behind the paper's tables and figures.

Writes, under anc/:
- markov_sanity.csv    Random biased chains (k = 3, 4, 5; five seeds each) at
                       n = 2^9 ... 2^15: analytic EP vs the time-reversal estimate.
- calibration.csv      60 random chains at n = 2^15 with varied bias: estimate vs EP.
- reversible_bias.csv  Detailed-balance chains (EP = 0) across n: the in-sample
                       plug-in bias of the estimator.
- hmm_order_sweep.csv  Hidden 3-state ring observed through two channels: the
                       estimate on the observed record for R = 1..6, compared with
                       the hidden chain's EP (an upper bound on the observed rate).

Usage: PYTHONPATH=src python scripts/paper_experiments.py [--only NAME ...]
All runs are seeded; rerunning reproduces the files exactly.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np

from uec.holonomy import klrate_holonomy_time_reversal_markov
from uec.markov import (
    entropy_production_rate_bits,
    random_markov_biased,
    ring_chain_3,
    ring_ep_bits,
    sample_HMM,
    sample_markov,
)

OUT = "anc"
R_DEFAULT = 3


def _write(name: str, rows: List[Dict]) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def markov_sanity() -> None:
    rows = []
    for seed in range(1, 6):
        for k in (3, 4, 5):
            for n in (2**9, 2**11, 2**13, 2**15):
                # Same chain for a given (seed, k) at every n; only the sample length changes.
                rng = np.random.default_rng(seed)
                T = random_markov_biased(k=k, delta=0.6, rng=rng)
                sigma = entropy_production_rate_bits(T, strict=True)
                x = sample_markov(T, n=n, rng=rng)
                est = klrate_holonomy_time_reversal_markov(x, k=k, R=R_DEFAULT)
                rows.append({"seed": seed, "k": k, "n": n, "order": R_DEFAULT, "sigma_true": sigma,
                             "estimate": est, "rel_err": (est - sigma) / sigma})
    _write("markov_sanity.csv", rows)


def calibration() -> None:
    rng = np.random.default_rng(20250827)
    rows = []
    for i in range(60):
        k = int(rng.choice([3, 4, 5]))
        delta = float(rng.uniform(0.0, 0.9))
        T = random_markov_biased(k=k, delta=delta, rng=rng)
        sigma = entropy_production_rate_bits(T, strict=True)
        x = sample_markov(T, n=2**15, rng=rng)
        est = klrate_holonomy_time_reversal_markov(x, k=k, R=R_DEFAULT)
        rows.append({"chain": i, "k": k, "delta": delta, "n": 2**15, "order": R_DEFAULT,
                     "sigma_true": sigma, "estimate": est})
    _write("calibration.csv", rows)


def _reversible_chain(k: int, rng: np.random.Generator) -> np.ndarray:
    W = rng.gamma(1.0, 1.0, size=(k, k))
    W = W + W.T  # symmetric weights => detailed balance with pi proportional to row sums
    return W / W.sum(axis=1, keepdims=True)


def reversible_bias() -> None:
    rng = np.random.default_rng(314159)
    rows = []
    for rep in range(10):
        for k in (3, 5):
            T = _reversible_chain(k, rng)
            sigma = entropy_production_rate_bits(T, strict=True)
            for n in (2**9, 2**11, 2**13, 2**15):
                x = sample_markov(T, n=n, rng=rng)
                for R in (1, 3):
                    est = klrate_holonomy_time_reversal_markov(x, k=k, R=R)
                    rows.append({"rep": rep, "k": k, "n": n, "order": R, "sigma_true": sigma, "estimate": est})
    _write("reversible_bias.csv", rows)


def hmm_order_sweep() -> None:
    p, q = 0.3, 0.05
    A = ring_chain_3(p, q)
    sigma_hidden = ring_ep_bits(p, q)
    channels = {
        # Each hidden state is reported correctly with probability 0.8.
        "noisy": np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
        # States 1 and 2 are merged. This observation is symmetric under 1<->2,
        # which maps the ring to its time reversal, so the observed process is reversible.
        "lumped": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
    }
    rows = []
    for name, B in channels.items():
        for seed in range(3):
            rng = np.random.default_rng(1000 + seed)
            obs, _, m = sample_HMM(A, B, n=2**17, rng=rng)
            for R in range(1, 7):
                est = klrate_holonomy_time_reversal_markov(obs, k=m, R=R)
                rows.append({"channel": name, "seed": seed, "n": 2**17, "order": R,
                             "sigma_hidden": sigma_hidden, "estimate": est})
            print(f"  {name} seed {seed} done")
    _write("hmm_order_sweep.csv", rows)


EXPERIMENTS = {
    "markov_sanity": markov_sanity,
    "calibration": calibration,
    "reversible_bias": reversible_bias,
    "hmm_order_sweep": hmm_order_sweep,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS), help="Run a subset")
    args = ap.parse_args()
    for name in args.only or EXPERIMENTS:
        print(f"== {name}")
        EXPERIMENTS[name]()
