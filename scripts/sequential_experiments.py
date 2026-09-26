#!/usr/bin/env python3
"""
Experiments for the anytime-valid reversibility test (uec.sequential).

Writes CSVs under anc/:
- seq_validity.csv      False-alarm rates on reversible chains: the two e-processes
                        (universal inference and loop orientation) monitored at
                        every checkpoint vs the classical
                        likelihood-ratio test (LRT) at a fixed n and with peeking.
- seq_misspec.csv       The same on records of hidden reversible chains, which
                        are reversible but not first-order Markov.
- seq_detection.csv     Detection times on irreversible chains: e-process stopping
                        times vs the fixed-n LRT sample size for 80% power.
- seq_growth.csv        log2(E_t)/t on long runs vs the predicted rate rho.
- seq_rates.csv         rho vs sigma across random chains.
- seq_loop_efficiency.csv  Oracle rate of the loop-orientation test vs rho.

All randomness is seeded. Usage:
  PYTHONPATH=src python scripts/sequential_experiments.py [--only NAME ...]
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from scipy.stats import beta, chi2

from uec.markov import (
    entropy_production_rate_bits,
    random_markov_biased,
    ring_chain_3,
    stationary_distribution,
)
from uec.sequential import LOG2, _log_ratio, reversible_mle, reversible_projection_rate, unconstrained_loglik

OUT = "anc"
ALPHA = 0.05
THRESH = np.log2(1 / ALPHA)


# ------------------------------------------------------------------
# Vectorized simulation and monitoring
# ------------------------------------------------------------------


def checkpoint_grid(n: int, dense_until: int = 64, points: int = 240) -> np.ndarray:
    """Every transition up to dense_until, then a geometric grid up to n."""
    head = np.arange(1, min(dense_until, n) + 1)
    tail = np.unique(np.geomspace(dense_until, n, points).astype(int))
    return np.unique(np.concatenate([head, tail]))


def simulate(T: np.ndarray, n: int, reps: int, rng: np.random.Generator,
             emit: np.ndarray | None = None) -> np.ndarray:
    """reps stationary trajectories of length n+1 (shape reps x (n+1)).

    With emit (k_hidden x k_obs), returns observations of the hidden chain.
    """
    k = T.shape[0]
    pi = stationary_distribution(T)
    cum = np.cumsum(T, axis=1)
    cum[:, -1] = 1.0
    z = np.empty((reps, n + 1), dtype=np.int64)
    z[:, 0] = rng.choice(k, size=reps, p=pi)
    for t in range(1, n + 1):
        u = rng.random(reps)
        z[:, t] = (u[:, None] > cum[z[:, t - 1]]).sum(axis=1)
    if emit is None:
        return z
    ecum = np.cumsum(emit, axis=1)
    ecum[:, -1] = 1.0
    u = rng.random(z.shape)
    return (u[..., None] > ecum[z]).sum(axis=-1)


WORKERS = 8
BURN_IN_PER_STATE = 10


def _monitor_chunk(x: np.ndarray, k: int, cps: np.ndarray, kt: float = 0.5,
                   with_mle: bool = True) -> Dict[str, np.ndarray]:
    reps, n1 = x.shape
    counts = np.zeros((reps, k, k))
    rows = np.zeros((reps, k))
    log_num = np.zeros(reps)
    idx = np.arange(reps)
    logE = np.full((reps, len(cps)), np.nan)
    logL = np.empty((reps, len(cps)))
    lrt = np.full((reps, len(cps)), np.nan)
    # loop-orientation test: one reference state per trajectory, fixed after a burn-in
    burn = min(BURN_IN_PER_STATE * k, n1 - 1)
    ref = np.array([np.argmax(np.bincount(row[: burn + 1], minlength=k)) for row in x])
    loopE = np.zeros(reps)
    loop_max = np.zeros(reps)                  # running max over every step
    loop_tau = np.full(reps, np.inf)           # first step at which log2 E >= THRESH
    acc = np.zeros(reps)
    started = (x[:, 0] == ref) & (burn == 0)
    A = np.broadcast_to(_log_ratio(counts[0], kt), (reps, k, k)).copy()
    ci = 0
    X = None
    for t in range(1, n1):
        a, b = x[:, t - 1], x[:, t]
        log_num += np.log((counts[idx, a, b] + kt) / (rows[idx, a] + kt * k))
        acc += np.where(started, A[idx, a, b], 0.0)
        counts[idx, a, b] += 1.0
        rows[idx, a] += 1.0
        back = (b == ref) & (t >= burn)
        loopE += np.where(back & started, 1.0 - np.log2(1.0 + 2.0 ** (-acc)), 0.0)
        loop_max = np.maximum(loop_max, loopE)
        loop_tau = np.where(np.isinf(loop_tau) & (loopE >= THRESH), t, loop_tau)
        if back.any():
            started = started | back
            acc = np.where(back, 0.0, acc)
            A[back] = _log_ratio(counts[back], kt)
        if ci < len(cps) and cps[ci] == t:
            logL[:, ci] = loopE
            if with_mle:
                _, ll_rev, X = reversible_mle(counts, X0=X, return_flows=True)
                logE[:, ci] = (log_num - ll_rev) / LOG2
                lrt[:, ci] = 2.0 * (unconstrained_loglik(counts) - ll_rev)
            ci += 1
    return {"logE": logE, "logL": logL, "lrt": lrt, "loop_max": loop_max, "loop_tau": loop_tau}


def monitor(x: np.ndarray, k: int, cps: np.ndarray, kt: float = 0.5,
            with_mle: bool = True) -> Dict[str, np.ndarray]:
    """For a batch of trajectories, at each checkpoint: log2 E_t of the
    universal-inference e-process ("logE"), log2 E_t of the loop-orientation
    test ("logL"), and the LRT statistic ("lrt"). Splits the batch across
    WORKERS processes; results do not depend on the split."""
    chunks = [c for c in np.array_split(x, min(WORKERS, len(x))) if len(c)]
    if len(chunks) == 1:
        return _monitor_chunk(x, k, cps, kt, with_mle)
    from multiprocessing import get_context
    with get_context("fork").Pool(len(chunks)) as pool:
        parts = pool.starmap(_monitor_chunk, [(c, k, cps, kt, with_mle) for c in chunks])
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def cp_interval(hits: int, n: int, level: float = 0.95):
    a = (1 - level) / 2
    lo = beta.ppf(a, hits, n - hits + 1) if hits > 0 else 0.0
    hi = beta.ppf(1 - a, hits + 1, n - hits) if hits < n else 1.0
    return float(lo), float(hi)


def _write(name: str, rows: List[Dict]) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def one_way_ring(p: float) -> np.ndarray:
    """Three-state ring that steps forward with probability p and never backward (sigma infinite)."""
    return np.array([[1 - p, p, 0.0], [0.0, 1 - p, p], [p, 0.0, 1 - p]])


def reversible_chain(k: int, rng: np.random.Generator, sparse: bool = False) -> np.ndarray:
    W = rng.gamma(1.0, 1.0, size=(k, k))
    W = W + W.T
    if sparse:  # birth-death: only neighbours and self-loops
        mask = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) <= 1
        W = W * mask
    return W / W.sum(axis=1, keepdims=True)


def cycle_df(T: np.ndarray) -> int:
    """Independent cycle constraints of detailed balance on the support graph."""
    k = T.shape[0]
    E = sum(1 for i in range(k) for j in range(i + 1, k) if T[i, j] > 0 or T[j, i] > 0)
    return max(E - (k - 1), 0)


# ------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------


def _null_rows(label, x, k, df, cps, n):
    m = monitor(x, k, cps)
    reps = x.shape[0]
    e_hits = int((m["logE"].max(axis=1) >= THRESH).sum())
    l_hits = int((m["loop_max"] >= THRESH).sum())  # checked after every single step
    row = {"chain": label, "k": k, "n": n, "reps": reps, "df": df,
           "eprocess_anytime": e_hits / reps, "loop_anytime": l_hits / reps}
    row["eprocess_lo"], row["eprocess_hi"] = cp_interval(e_hits, reps)
    row["loop_lo"], row["loop_hi"] = cp_interval(l_hits, reps)
    if df > 0:
        crit = chi2.ppf(1 - ALPHA, df)
        fixed = int((m["lrt"][:, -1] > crit).sum())
        peek = int((m["lrt"][:, cps >= 100] > crit).any(axis=1).sum())
        looks10 = np.unique(np.searchsorted(cps, np.geomspace(100, n, 10).astype(int)))
        peek10 = int((m["lrt"][:, looks10] > crit).any(axis=1).sum())
        row.update(lrt_fixed_n=fixed / reps, lrt_peek_every_checkpoint=peek / reps,
                   lrt_peek_10_looks=peek10 / reps)
    else:
        row.update(lrt_fixed_n=float("nan"), lrt_peek_every_checkpoint=float("nan"),
                   lrt_peek_10_looks=float("nan"))
    row["max_log2E_median"] = float(np.median(m["logE"].max(axis=1)))
    return row


def validity(reps: int = 1000, n: int = 20000) -> None:
    rng = np.random.default_rng(101)
    cps = checkpoint_grid(n)
    rows = []
    cases = [("random k=3", reversible_chain(3, rng)), ("random k=4", reversible_chain(4, rng)),
             ("random k=5", reversible_chain(5, rng)), ("birth-death k=5", reversible_chain(5, rng, sparse=True)),
             ("unbiased ring k=3", ring_chain_3(0.2, 0.2))]
    for label, T in cases:
        assert entropy_production_rate_bits(T) < 1e-9
        x = simulate(T, n, reps, rng)
        rows.append(_null_rows(label, x, T.shape[0], cycle_df(T), cps, n))
        print(f"  {label}: {rows[-1]}")
    _write("seq_validity.csv", rows)


def misspecification(reps: int = 1000, n: int = 20000) -> None:
    """Hidden reversible chains observed through channels: reversible, not Markov."""
    rng = np.random.default_rng(202)
    cps = checkpoint_grid(n)
    rows = []
    Th = reversible_chain(4, rng)
    noisy4 = 0.7 * np.eye(4) + 0.1 * (np.ones((4, 4)) - np.eye(4))
    lump = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    ring = ring_chain_3(0.2, 0.2)
    noisy3 = 0.8 * np.eye(3) + 0.1 * (np.ones((3, 3)) - np.eye(3))
    for label, T, B in [("reversible k=4, noisy view", Th, noisy4),
                        ("reversible k=4, states 2,3 merged", Th, lump),
                        ("unbiased ring, noisy view", ring, noisy3)]:
        x = simulate(T, n, reps, rng, emit=B)
        k_obs = B.shape[1]
        df = (k_obs - 1) * (k_obs - 2) // 2
        rows.append(_null_rows(label, x, k_obs, df, cps, n))
        print(f"  {label}: {rows[-1]}")
    _write("seq_misspec.csv", rows)


def _first_cross(values: np.ndarray, cps: np.ndarray, thresh: float) -> np.ndarray:
    hit = values >= thresh
    first = np.where(hit.any(axis=1), cps[np.argmax(hit, axis=1)], np.inf)
    return first


NULL_REPS = 1000


def detection(reps: int = 400) -> None:
    rng = np.random.default_rng(303)
    cases = []
    for p, q in [(0.30, 0.25), (0.30, 0.20), (0.30, 0.12), (0.30, 0.05)]:
        cases.append((f"ring p={p} q={q}", ring_chain_3(p, q)))
    cases.append(("one-way ring p=0.3", one_way_ring(0.3)))
    for i in range(3):
        cases.append((f"random k=4 #{i}", random_markov_biased(4, delta=0.5, rng=rng)))
    rows = []
    for label, T in cases:
        k = T.shape[0]
        rho = reversible_projection_rate(T)
        sigma = entropy_production_rate_bits(T) if np.all((T > 0) == (T.T > 0)) else float("inf")
        n = int(min(200000, max(2000, 12 * THRESH / max(rho, 1e-6) + 40 * k * k / max(rho, 1e-6))))
        cps = checkpoint_grid(n, points=300)
        x = simulate(T, n, reps, rng)
        m = monitor(x, k, cps)
        tau = _first_cross(m["logE"], cps, THRESH)
        tau_loop = m["loop_tau"]  # exact first crossing, every step
        df = cycle_df(T)
        crit_asym = chi2.ppf(1 - ALPHA, df)
        # Calibrate the fixed-n LRT under the closest reversible chain M = (T + T*)/2,
        # the least favourable point of the null, so its size is 5% at every n.
        pi = stationary_distribution(T)
        M = 0.5 * (T + (pi[None, :] * T.T) / pi[:, None])
        m0 = monitor(simulate(M, n, NULL_REPS, rng), k, cps)
        crit = np.quantile(m0["lrt"], 1 - ALPHA, axis=0)
        power = (m["lrt"] > crit).mean(axis=0)
        n80_lrt = float(cps[np.argmax(power >= 0.8)]) if (power >= 0.8).any() else float("inf")
        power_asym = (m["lrt"] > crit_asym).mean(axis=0)
        n80_asym = float(cps[np.argmax(power_asym >= 0.8)]) if (power_asym >= 0.8).any() else float("inf")
        stopped = np.isfinite(tau)
        rows.append({
            "chain": label, "k": k, "sigma": sigma, "rho": rho, "n_max": n, "reps": reps,
            "e_detect_frac": float(stopped.mean()),
            "e_tau_median": float(np.median(tau)), "e_tau_80": float(np.quantile(tau, 0.8)),
            "theory_log2(1/a)/rho": THRESH / rho, "lrt_fixed_n80": n80_lrt,
            "lrt_fixed_n80_asymptotic_crit": n80_asym,
            "ratio_e80_over_lrt80": float(np.quantile(tau, 0.8)) / n80_lrt,
            "loop_detect_frac": float(np.isfinite(tau_loop).mean()),
            "loop_tau_median": float(np.median(tau_loop)), "loop_tau_80": float(np.quantile(tau_loop, 0.8)),
            "ratio_loop80_over_lrt80": float(np.quantile(tau_loop, 0.8)) / n80_lrt,
        })
        print(f"  {label}: {rows[-1]}")
    _write("seq_detection.csv", rows)


def growth(reps: int = 20, n: int = 200000) -> None:
    rng = np.random.default_rng(404)
    cps = np.unique(np.geomspace(100, n, 60).astype(int))
    rows = []
    for label, T in [("ring p=0.3 q=0.05", ring_chain_3(0.3, 0.05)),
                     ("ring p=0.3 q=0.2", ring_chain_3(0.3, 0.2)),
                     ("one-way ring p=0.3", one_way_ring(0.3)),
                     ("random k=4", random_markov_biased(4, delta=0.5, rng=rng))]:
        x = simulate(T, n, reps, rng)
        m = monitor(x, T.shape[0], cps)
        rate = m["logE"] / cps[None, :]
        lrate = m["logL"] / cps[None, :]
        rho = reversible_projection_rate(T)
        for j, t in enumerate(cps):
            rows.append({"chain": label, "t": int(t), "rho": rho, "rate_mean": float(rate[:, j].mean()),
                         "rate_lo": float(np.quantile(rate[:, j], 0.1)), "rate_hi": float(np.quantile(rate[:, j], 0.9)),
                         "loop_rate_mean": float(lrate[:, j].mean())})
        print(f"  {label}: rho={rho:.4f}  final mean rate={rate[:, -1].mean():.4f}  loop={lrate[:, -1].mean():.4f}")
    _write("seq_growth.csv", rows)


def rates() -> None:
    rng = np.random.default_rng(505)
    rows = []
    for i in range(300):
        k = int(rng.choice([3, 4, 5]))
        T = random_markov_biased(k, delta=float(rng.uniform(0, 0.95)), rng=rng)
        rows.append({"k": k, "sigma": entropy_production_rate_bits(T, strict=True),
                     "rho": reversible_projection_rate(T)})
    _write("seq_rates.csv", rows)


def _loop_rate(T: np.ndarray, r: int, n_loops: int, rng: np.random.Generator):
    """Oracle evidence per step (bits) of the loop-orientation test with reference
    state r, and its standard error (ratio estimator, delta method)."""
    cum = np.cumsum(T, axis=1)
    cum[:, -1] = 1.0
    with np.errstate(divide="ignore"):
        lT = np.log2(T)
    vals = np.empty(n_loops)
    lens = np.empty(n_loops)
    for i in range(n_loops):
        s, d, length = r, 0.0, 0
        while True:
            y = int((rng.random() > cum[s]).sum())
            d += lT[s, y] - lT[y, s]
            length += 1
            s = y
            if s == r:
                break
        lens[i] = length
        vals[i] = 1.0 if d == np.inf else 1.0 - np.log2(1.0 + 2.0 ** (-d))
    rate = vals.sum() / lens.sum()
    resid = vals - rate * lens
    se = np.sqrt(np.sum(resid ** 2)) / lens.sum()
    return rate, se


def loop_efficiency(base_loops: int = 20000, max_loops: int = 1_000_000) -> None:
    """Oracle loop-orientation rate relative to the optimal rate rho, for the most
    visited reference state (the one the test selects asymptotically). The number
    of simulated loops grows as 1/rho so that near-equilibrium ratios are resolved."""
    rng = np.random.default_rng(606)
    cases = [(f"ring p=0.3 q={q}", ring_chain_3(0.3, q)) for q in (0.25, 0.2, 0.12, 0.05)]
    cases.append(("one-way ring p=0.3", one_way_ring(0.3)))
    rng_c = np.random.default_rng(303)
    cases += [(f"random k=4 #{i}", random_markov_biased(4, delta=0.5, rng=rng_c)) for i in range(3)]
    rows = []
    for label, T in cases:
        rho = reversible_projection_rate(T)
        r_used = int(np.argmax(stationary_distribution(T)))
        n_loops = int(min(max_loops, base_loops * max(1.0, 0.1 / rho)))
        rate, se = _loop_rate(T, r_used, n_loops, rng)
        rows.append({"chain": label, "rho": rho, "loops": n_loops, "loop_rate": rate, "loop_rate_se": se,
                     "efficiency": rate / rho, "efficiency_se": se / rho})
        print(f"  {label}: rho={rho:.4f} loop={rate:.4f}±{se:.4f} efficiency={rate / rho:.2f}±{se / rho:.2f}")
    _write("seq_loop_efficiency.csv", rows)


EXPERIMENTS = {"validity": validity, "misspecification": misspecification, "detection": detection,
               "growth": growth, "rates": rates, "loop_efficiency": loop_efficiency}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS))
    args = ap.parse_args()
    for name in args.only or EXPERIMENTS:
        print(f"== {name}")
        EXPERIMENTS[name]()
