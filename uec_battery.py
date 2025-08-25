#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UEC Battery: Universal Evidence Curvature (UEC) tests and validations.

This script implements:
- Robust finite-state Markov utilities and analytic entropy production (EP) in bits/step.
- Universal coders:
    * KTMarkovMixture with 'frozen' evaluation (cross-entropy on held sequences).
    * LZ78 (for observer-independence sanity).
- Representation transforms and loop machinery.
- KL-rate holonomy estimators (the correct functional matching D(P||Q)).
- A test battery with assertions:
    1) Gauge invariance under bijective recoding (≈ 0).
    2) Coarse-graining/refinement loop (≥ 0).
    3) Time-reversal loop on Markov chains (≈ EP).
    4) Observer-independence sanity (KT vs LZ difference per symbol shrinks).
    5) Random-chain sweep: EP vs KL-rate holonomy with CI and error stats.
    6) Optional demos (HMM, measurement-record).
Usage:
    python uec_battery.py --help
"""

import argparse
import math
import random
import statistics
from collections import defaultdict
from typing import List, Tuple, Sequence, Dict

import numpy as np

# ------------------------------
# Utilities and reproducibility
# ------------------------------


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ------------------------------
# Robust Markov utilities
# ------------------------------


def _row_stochastic(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return T / rs


def stationary_distribution(
    T: np.ndarray, tol: float = 1e-12, max_iter: int = 200000
) -> np.ndarray:
    """Power-iteration stationary distribution with guards (no NaNs)."""
    T = _row_stochastic(T)
    k = T.shape[0]
    pi = np.ones(k, dtype=float) / k
    for _ in range(max_iter):
        pi_new = pi @ T
        if np.linalg.norm(pi_new - pi, 1) < tol:
            pi = pi_new
            break
        pi = pi_new
    s = float(pi.sum())
    if not np.isfinite(s) or s == 0:
        pi = np.ones(k, dtype=float) / k
    else:
        pi = pi / s
    return pi


def sample_markov(
    T: np.ndarray,
    n: int,
    init: np.ndarray = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Safe sampler that normalizes T and uses robust stationary distribution by default."""
    T = _row_stochastic(T)
    if rng is None:
        rng = np.random.default_rng()
    k = T.shape[0]
    if init is None:
        pi = stationary_distribution(T)
    else:
        pi = np.asarray(init, dtype=float)
        s = float(pi.sum())
        pi = (pi / s) if s != 0 else np.ones(k) / k
    x = np.zeros(n, dtype=int)
    x[0] = rng.choice(np.arange(k), p=pi)
    for t in range(1, n):
        x[t] = rng.choice(np.arange(k), p=T[x[t - 1]])
    return x


def entropy_production_rate_bits(T: np.ndarray) -> float:
    """
    EP σ = Σ_{i,j} π_i T_ij log2((π_i T_ij)/(π_j T_ji)).
    We floor one-way edges with a large finite penalty to avoid +∞ in numerics.
    """
    T = _row_stochastic(T)
    pi = stationary_distribution(T)
    k = T.shape[0]
    s = 0.0
    for i in range(k):
        for j in range(k):
            if T[i, j] > 0 and T[j, i] > 0:
                num = pi[i] * T[i, j]
                den = pi[j] * T[j, i]
                s += num * (math.log(num / den, 2.0))
            elif T[i, j] > 0 and T[j, i] == 0:
                s += (pi[i] * T[i, j]) * 1000.0  # finite surrogate for +∞
    return float(s)


def random_markov_biased(
    k: int,
    delta: float = 0.4,
    min_prob: float = 1e-6,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Construct a kxk stochastic matrix with mild directional bias (irreversibility)
    but strictly positive entries (finite EP).
    """
    if rng is None:
        rng = np.random.default_rng()
    G = rng.gamma(shape=1.0, scale=1.0, size=(k, k))
    T = G / G.sum(axis=1, keepdims=True)
    # Apply a simple cycle bias: favor i->i+1 and disfavor i->i-1
    for i in range(k):
        T[i, (i + 1) % k] *= 1.0 + delta
        T[i, (i - 1) % k] *= 1.0 - 0.5 * delta
    T = T / T.sum(axis=1, keepdims=True)
    # Floor all entries to min_prob and renormalize
    T = np.maximum(T, min_prob)
    T = T / T.sum(axis=1, keepdims=True)
    return T


def counts_from_sequence(seq: Sequence[int], k: int) -> np.ndarray:
    """Compute kxk transition counts from a state sequence."""
    C = np.zeros((k, k), dtype=np.int64)
    for t in range(len(seq) - 1):
        i = int(seq[t])
        j = int(seq[t + 1])
        C[i, j] += 1
    return C


def ep_bits_from_counts_smoothed(counts: np.ndarray, alpha: float = 0.5) -> float:
    """
    Estimate entropy production (bits/step) from a single path via smoothed counts.
    - counts: (k x k) transition counts
    - alpha: KT-style smoothing (0.5) for robustness to zeros
    """
    k = counts.shape[0]
    # Smoothed transition matrix
    N_i = counts.sum(axis=1, keepdims=True)
    T_hat = (counts + alpha) / (N_i + alpha * k)
    # Smoothed stationary frequency proxy
    occ = counts.sum(axis=1) + alpha
    pi_hat = occ / occ.sum()
    s = 0.0
    for i in range(k):
        for j in range(k):
            if T_hat[i, j] > 0 and T_hat[j, i] > 0:
                num = pi_hat[i] * T_hat[i, j]
                den = pi_hat[j] * T_hat[j, i]
                s += num * (math.log(num / den, 2.0))
    return float(s)


# ------------------------------
# Universal coders
# ------------------------------


class KTMarkovMixture:
    """
    KT (Krichevsky–Trofimov) Bayesian mixture over Markov orders r=0..R for k-ary alphabets.
    Online training; supports 'frozen' evaluation (cross-entropy on a fixed model).
    """

    def __init__(self, alphabet_size: int, R: int = 3, prior_decay: float = 2.0):
        self.k = int(alphabet_size)
        self.R = int(R)
        priors = np.array([prior_decay ** (-r) for r in range(self.R + 1)], dtype=float)
        self.alpha = priors / priors.sum()
        self.tables: List[Dict[Tuple[int, ...], np.ndarray]] = [
            defaultdict(lambda: np.zeros(self.k, dtype=float))
            for _ in range(self.R + 1)
        ]
        self.history: List[int] = []

    @staticmethod
    def _kt_predict(counts: np.ndarray, k: int) -> np.ndarray:
        n = float(counts.sum())
        return (counts + 0.5) / (n + 0.5 * k)

    def update_and_codelen(self, sym: int) -> float:
        sym = int(sym)
        preds = []
        for r in range(self.R + 1):
            ctx = tuple(self.history[-r:]) if r > 0 else ()
            counts = self.tables[r][ctx]
            p = self._kt_predict(counts, self.k)
            preds.append(p)
        mix = sum(a * p for a, p in zip(self.alpha, preds))
        p_sym = max(float(mix[sym]), 1e-300)
        codelen = -math.log(p_sym, 2.0)

        # Update mixture weights
        post = np.array(
            [a * max(float(p[sym]), 1e-300) for a, p in zip(self.alpha, preds)],
            dtype=float,
        )
        s = float(post.sum())
        post = (np.ones_like(post) / len(post)) if s <= 0 else (post / s)
        self.alpha = post

        # Update counts
        for r in range(self.R + 1):
            ctx = tuple(self.history[-r:]) if r > 0 else ()
            self.tables[r][ctx][sym] += 1.0

        self.history.append(sym)
        return codelen

    def fit(self, sequence: Sequence[int]) -> float:
        self.reset()
        total = 0.0
        for s in sequence:
            total += self.update_and_codelen(int(s))
        return total

    def reset(self) -> None:
        self.alpha = self.alpha * 0 + (1.0 / (self.R + 1))
        self.tables = [
            defaultdict(lambda: np.zeros(self.k, dtype=float))
            for _ in range(self.R + 1)
        ]
        self.history = []

    def snapshot_frozen(self) -> "KTFrozenPredictor":
        # deep copy minimal state for frozen evaluation
        tables_copy: List[Dict[Tuple[int, ...], np.ndarray]] = []
        for d in self.tables:
            newd: Dict[Tuple[int, ...], np.ndarray] = {}
            for ctx, arr in d.items():
                newd[ctx] = arr.copy()
            tables_copy.append(newd)
        return KTFrozenPredictor(tables_copy, self.alpha.copy(), self.k, self.R)


class KTFrozenPredictor:
    """Evaluate code length of a test sequence with fixed counts/mixture from training (no updates)."""

    def __init__(
        self,
        tables: List[Dict[Tuple[int, ...], np.ndarray]],
        alpha: np.ndarray,
        k: int,
        R: int,
    ):
        self.tables = tables
        self.alpha = alpha
        self.k = int(k)
        self.R = int(R)
        self.history: List[int] = []

    @staticmethod
    def _kt_predict(counts: np.ndarray, k: int) -> np.ndarray:
        n = float(counts.sum())
        return (counts + 0.5) / (n + 0.5 * k)

    def codelen_sequence(self, sequence: Sequence[int]) -> float:
        self.history = []
        total = 0.0
        for sym in sequence:
            sym = int(sym)
            preds = []
            for r in range(self.R + 1):
                ctx = tuple(self.history[-r:]) if r > 0 else ()
                counts = self.tables[r].get(ctx)
                if counts is None:
                    counts = np.zeros(self.k, dtype=float)
                p = self._kt_predict(counts, self.k)
                preds.append(p)
            mix = sum(a * p for a, p in zip(self.alpha, preds))
            p_sym = max(float(mix[sym]), 1e-300)
            total += -math.log(p_sym, 2.0)
            self.history.append(sym)
        return total


class LZ78Coder:
    """Simple LZ78 code length estimator (for sanity checks and observer-independence trends)."""

    def __init__(self, alphabet_size: int):
        self.k = int(alphabet_size)

    def total_codelen(self, sequence: Sequence[int]) -> float:
        dict_trie: Dict[Tuple[int, ...], Dict[int, dict]] = {(): {}}
        curr: Tuple[int, ...] = ()
        phrases = 0
        bits = 0.0
        for sym in sequence:
            sym = int(sym)
            if curr not in dict_trie:
                dict_trie[curr] = {}
            if sym in dict_trie[curr]:
                curr = curr + (sym,)
            else:
                phrases += 1
                index_bits = math.log2(max(1, phrases))
                symbol_bits = math.log2(self.k)
                bits += index_bits + symbol_bits
                dict_trie[curr][sym] = {}
                curr = ()
        if len(curr) > 0:
            phrases += 1
            index_bits = math.log2(max(1, phrases))
            symbol_bits = math.log2(self.k)
            bits += index_bits + symbol_bits
        return bits


# ------------------------------
# Transforms & Loops
# ------------------------------


class Transform:
    def apply(
        self, seq: Sequence[int], alphabet: Sequence[int]
    ) -> Tuple[List[int], List[int]]:
        raise NotImplementedError


class Permute(Transform):
    """Bijective recoding via a permutation of alphabet indices [0..k-1]."""

    def __init__(self, perm: Sequence[int]):
        self.perm = list(map(int, perm))

    def apply(self, seq, alphabet):
        mapped = [self.perm[int(s)] for s in seq]
        return mapped, list(alphabet)


class MergeSymbols(Transform):
    """Coarse-grain by merging symbols using a mapping old->new indices. Many-to-one allowed."""

    def __init__(self, mapping: Dict[int, int], new_k: int = None):
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.new_k = (
            int(new_k) if new_k is not None else (max(self.mapping.values()) + 1)
        )

    def apply(self, seq, alphabet):
        mapped = [self.mapping[int(s)] for s in seq]
        return mapped, list(range(self.new_k))


class TimeReverse(Transform):
    def apply(self, seq, alphabet):
        return list(reversed(seq)), list(alphabet)


class TransitionEncode(Transform):
    """
    Map a sequence x_0..x_{n-1} over k symbols to transitions y_0..y_{n-2} over k^2 symbols:
    y_t = (x_t, x_{t+1}) encoded as index i*k + j.
    (Drops one boundary symbol.)
    """

    def __init__(self, k: int):
        self.k = int(k)

    def apply(self, seq, alphabet):
        x = list(seq)
        y = []
        for t in range(0, len(x) - 1):
            y.append(int(x[t]) * self.k + int(x[t + 1]))
        return y, list(range(self.k * self.k))


class TransitionDecodeTakeSecond(Transform):
    """
    Decode transitions back to a state sequence by taking the SECOND coordinate of each pair.
    If input are pairs (i,j), output is [j_0, j_1, ..., j_{m-1}] in the original k-ary alphabet.
    """

    def __init__(self, k: int):
        self.k = int(k)

    def apply(self, seq, alphabet):
        out = []
        for z in seq:
            j = int(z % self.k)
            out.append(j)
        return out, list(range(self.k))


def apply_loop(
    seq: Sequence[int], alphabet: Sequence[int], transforms: List[Transform]
) -> Tuple[List[int], List[int]]:
    s, a = list(seq), list(alphabet)
    for T in transforms:
        s, a = T.apply(s, a)
    return s, a


class Downsample(Transform):
    def __init__(self, step: int = 2):
        self.step = int(step)

    def apply(self, seq, alphabet):
        return list(seq)[:: self.step], list(alphabet)


class UpsampleRepeat(Transform):
    def __init__(self, step: int = 2):
        self.step = int(step)

    def apply(self, seq, alphabet):
        out: List[int] = []
        for s in seq:
            out.extend([int(s)] * self.step)
        return out, list(alphabet)


# ------------------------------
# KL-rate holonomy estimators
# ------------------------------


def klrate_between_sequences(
    p_seq: Sequence[int], q_seq: Sequence[int], k: int, R: int = 3
) -> float:
    """
    Estimate D(P || Q) by training KT(R) models on p_seq and q_seq separately,
    then computing H(P,Q) - H(P) using frozen predictors, on the SAME p_seq.
    Returns bits/symbol.
    """
    Pcoder = KTMarkovMixture(k, R=R)
    Pcoder.fit(p_seq)
    Pfrozen = Pcoder.snapshot_frozen()
    Qcoder = KTMarkovMixture(k, R=R)
    Qcoder.fit(q_seq)
    Qfrozen = Qcoder.snapshot_frozen()
    H_P = Pfrozen.codelen_sequence(p_seq) / max(1, len(p_seq))
    H_PQ = Qfrozen.codelen_sequence(p_seq) / max(1, len(p_seq))
    return float(H_PQ - H_P)


def klrate_holonomy_general(
    seq: Sequence[int],
    alphabet: Sequence[int],
    loop: List[Transform],
    k: int,
    R: int = 3,
    align: str = "tail",
) -> float:
    """
    KL-rate holonomy for a general loop that returns to the original alphabet (k),
    possibly with a length change.
    We align the evaluation sequence to the q_seq length (tail by default).
    """
    q_seq, aQ = apply_loop(seq, alphabet, loop)
    if len(aQ) != k:
        raise ValueError("Final loop output must be in the original alphabet.")
    nQ = len(q_seq)
    if nQ <= 0:
        raise ValueError("Loop produced empty sequence; cannot evaluate KL-rate.")
    # Align lengths
    if align == "tail":
        p_eval = list(seq)[-nQ:]
    elif align == "head":
        p_eval = list(seq)[:nQ]
    else:
        raise ValueError("align must be 'tail' or 'head'")
    return klrate_between_sequences(p_eval, q_seq, k, R=R)


def klrate_holonomy_time_reversal_markov(
    seq: Sequence[int], k: int, R: int = 3
) -> float:
    """
    Special KL-rate holonomy for time reversal on Markov chains via:
    TransitionEncode(k) -> TimeReverse() -> TransitionDecodeTakeSecond(k).
    We then compare p_seq = seq[1:] vs q_seq (length n-1), both in k-ary alphabet.
    """
    E = TransitionEncode(k)
    Rv = TimeReverse()
    D2 = TransitionDecodeTakeSecond(k)
    q_seq, aQ = apply_loop(seq, list(range(k)), [E, Rv, D2])
    p_eval = list(seq)[1:]  # align lengths to transitions (drop the boundary)
    return klrate_between_sequences(p_eval, q_seq, k, R=R)


# ------------------------------
# Tests
# ------------------------------


def test_gauge_invariance(
    seed: int = 123, n: int = 60000, k: int = 4, R: int = 3
) -> None:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.0, rng=rng)
    x = sample_markov(T, n=n, rng=rng)
    alphabet = list(range(k))
    perm = list(rng.permutation(k))
    P = Permute(perm)
    Pinv = Permute([perm.index(i) for i in range(k)])
    loop = [P, Pinv]
    hol_rate = klrate_holonomy_general(x, alphabet, loop, k=k, R=R, align="head")
    print(f"[Gauge invariance] KL-rate holonomy (bits/step) ≈ 0: {hol_rate:.6g}")
    assert abs(hol_rate) < 5e-3, "Gauge invariance violated beyond tolerance."


def test_coarse_grain_nonneg(
    seed: int = 456, n: int = 80000, k: int = 4, R: int = 3
) -> None:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.25, rng=rng)
    x = sample_markov(T, n=n, rng=rng)
    alphabet = list(range(k))
    M = MergeSymbols({0: 0, 1: 0, 2: 1, 3: 1}, new_k=2)

    class LiftBinaryTo4(Transform):
        def apply(self, seq, alphabet):
            return [0 if int(s) == 0 else 2 for s in seq], list(range(4))

    L = LiftBinaryTo4()
    loop = [M, L]
    hol_rate = klrate_holonomy_general(x, alphabet, loop, k=4, R=R, align="head")
    print(f"[Coarse-grain] KL-rate holonomy (bits/step) >= 0: {hol_rate:.6g}")
    assert (
        hol_rate > -3e-3
    ), "Coarse-grain KL-rate holonomy should be non-negative (allowing tiny numerical slack)."


def test_time_reversal_equals_EP(
    seed: int = 789, n: int = 150000, k: int = 3, R: int = 3, delta: float = 0.6
) -> None:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=delta, rng=rng)
    sigma_bits = entropy_production_rate_bits(T)
    x = sample_markov(T, n=n, rng=rng)
    hol_rate = klrate_holonomy_time_reversal_markov(x, k=k, R=R)
    diff = abs(hol_rate - sigma_bits)
    rel = diff / max(1e-8, abs(sigma_bits))
    print(
        f"[Time reversal] EP analytic={sigma_bits:.6g}  KL-rate hol={hol_rate:.6g}  "
        f"abs diff={diff:.3g}  rel diff={rel:.3%}"
    )
    # tolerances: large n should give tight agreement; relax if you lower n
    assert (
        (diff < 5e-3)
        or (rel < 5e-2)
        or (abs(sigma_bits) < 1e-5 and diff < 5e-4)
    ), "Time-reversal KL-rate holonomy does not match EP within tolerance."


def test_observer_independence_trend(seed: int = 135, k: int = 3) -> None:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.4, rng=rng)
    Ns = [4000, 8000, 16000, 32000, 64000]
    per_symbol_diffs = []
    for n in Ns:
        x = sample_markov(T, n=n, rng=rng)
        kt = KTMarkovMixture(k, R=3)
        lz = LZ78Coder(k)
        eKT = kt.fit(x)
        eLZ = lz.total_codelen(x)
        per_symbol_diffs.append((eKT - eLZ) / n)
    print(
        "[Observer-independence sanity] KT-LZ per-symbol evidence difference:",
        per_symbol_diffs,
    )
    # Trend should approach a constant or shrink in magnitude; we check the last is closer to 0 than the first
    assert (
        abs(per_symbol_diffs[-1]) <= abs(per_symbol_diffs[0]) + 1e-3
    ), "Per-symbol KT-LZ difference did not shrink or stabilize."


def sweep_random_chains_EP_vs_KL(
    seed: int = 2468,
    reps: int = 8,
    n: int = 120000,
    k: int = 3,
    delta: float = 0.6,
    R: int = 3,
) -> None:
    rng = np.random.default_rng(seed)
    diffs = []
    rels = []
    sigmas = []
    hols = []
    for r in range(reps):
        T = random_markov_biased(k=k, delta=delta, rng=rng)
        sigma_bits = entropy_production_rate_bits(T)
        x = sample_markov(T, n=n, rng=rng)
        hol_rate = klrate_holonomy_time_reversal_markov(x, k=k, R=R)
        diffs.append(hol_rate - sigma_bits)
        rels.append((hol_rate - sigma_bits) / max(1e-8, abs(sigma_bits)))
        sigmas.append(sigma_bits)
        hols.append(hol_rate)
        print(
            f"  Rep {r+1}/{reps}: EP={sigma_bits:.6g}  Hol={hol_rate:.6g}  "
            f"diff={hol_rate - sigma_bits:.3g}  rel={rels[-1]:.2%}"
        )
    mean_diff = statistics.mean(diffs)
    mean_abs_diff = statistics.mean(abs(d) for d in diffs)
    sd_diff = statistics.pstdev(diffs)
    q25, q50, q75 = np.quantile(diffs, [0.25, 0.5, 0.75])
    valid_rels = [r for r, s in zip(rels, sigmas) if abs(s) >= 1e-4] or rels
    mean_rel = statistics.mean(valid_rels)
    print(
        f"[Sweep] mean |diff|={mean_abs_diff:.3g}  sd(diff)={sd_diff:.3g}  mean rel={mean_rel:.2%}  "
        f"median={q50:.3g}  IQR=[{q25:.3g},{q75:.3g}]"
    )
    # Soft assertion: average relative error within 5%
    assert (
        abs(mean_rel) < 0.05
    ), "Average relative error across random chains is too large."


def test_ep_consistency_three_way(
    seed: int = 202, n: int = 150000, k: int = 3, R: int = 3, delta: float = 0.6
) -> None:
    """Cross-validate EP via (i) analytic T, (ii) KL-holonomy, (iii) smoothed counts MLE."""
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=delta, rng=rng)
    sigma_true = entropy_production_rate_bits(T)
    x = sample_markov(T, n=n, rng=rng)
    hol = klrate_holonomy_time_reversal_markov(x, k=k, R=R)
    C = counts_from_sequence(x, k)
    sigma_mle = ep_bits_from_counts_smoothed(C, alpha=0.5)
    d_true_hol = abs(sigma_true - hol)
    d_true_mle = abs(sigma_true - sigma_mle)
    d_hol_mle = abs(hol - sigma_mle)
    rel = d_true_hol / max(1e-8, abs(sigma_true))
    print(
        f"[EP 3-way] true={sigma_true:.6g}  hol={hol:.6g}  mle={sigma_mle:.6g}  "
        f"|t-h|={d_true_hol:.3g}  |t-m|={d_true_mle:.3g}  |h-m|={d_hol_mle:.3g}  rel={rel:.2%}"
    )
    assert (
        (d_true_hol < 5e-3 or rel < 5e-2) and d_hol_mle < 6e-3
    ), "Three-way EP consistency failed (increase n or check RNG)."


def test_temporal_coarse_grain(
    seed: int = 642, n: int = 80000, k: int = 3, R: int = 3
) -> None:
    """Temporal coarse-graining via downsample+upsample should yield non-negative holonomy."""
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.5, rng=rng)
    x = sample_markov(T, n=n, rng=rng)
    alphabet = list(range(k))
    loop = [Downsample(step=2), UpsampleRepeat(step=2)]
    hol_rate = klrate_holonomy_general(x, alphabet, loop, k=k, R=R, align="head")
    print(
        f"[Temporal coarse-grain] KL-rate holonomy (bits/step) >= 0: {hol_rate:.6g}"
    )
    assert hol_rate > -3e-3


# ------------------------------
# Optional demos (off by default)
# ------------------------------


def sample_HMM(
    A: np.ndarray,
    B: np.ndarray,
    n: int,
    rng: np.random.Generator = None,
) -> Tuple[List[int], int, int]:
    """Hidden Markov Model sampler for observed symbols; uses robust stationary init for hidden chain."""
    A = _row_stochastic(np.array(A, dtype=float))
    B = _row_stochastic(np.array(B, dtype=float))
    k = A.shape[0]
    m = B.shape[1]
    if rng is None:
        rng = np.random.default_rng()
    pi = stationary_distribution(A)
    z = int(rng.choice(np.arange(k), p=pi))
    obs: List[int] = []
    for _ in range(n):
        o = int(rng.choice(np.arange(m), p=B[z]))
        obs.append(o)
        z = int(rng.choice(np.arange(k), p=A[z]))
    return obs, k, m


def demo_hmm_coarse_grain(seed: int = 8642, n: int = 60000) -> None:
    rng = np.random.default_rng(seed)
    A = random_markov_biased(2, delta=0.5, rng=rng)
    B = np.array([[0.8, 0.15, 0.05], [0.2, 0.5, 0.3]], dtype=float)
    obs, k_hidden, m_obs = sample_HMM(A, B, n=n, rng=rng)
    alphabet = list(range(m_obs))
    M = MergeSymbols({0: 0, 1: 1, 2: 1}, new_k=2)

    class Lift2to3(Transform):
        def apply(self, seq, alphabet):
            return [0 if int(s) == 0 else 2 for s in seq], [0, 1, 2]

    L = Lift2to3()
    loop = [M, L]
    hol_rate = klrate_holonomy_general(obs, alphabet, loop, k=3, R=3, align="head")
    print(f"[HMM demo] Coarse-grain KL-rate holonomy (bits/step) >= 0: {hol_rate:.6g}")


def demo_measurement_record(seed: int = 9753, n: int = 80000) -> None:
    rng = np.random.default_rng(seed)
    # Two-state Markov (measurement record): add slight excitation p01 and decay gamma
    p01 = 0.05
    gamma = 0.2
    T = np.array([[1 - p01, p01], [gamma, 1 - gamma]], dtype=float)
    sigma_bits = entropy_production_rate_bits(T)
    x = sample_markov(T, n=n, rng=rng)
    hol_rate = klrate_holonomy_time_reversal_markov(x, k=2, R=3)
    print(
        f"[Measurement record] EP analytic={sigma_bits:.6g}  KL-rate hol={hol_rate:.6g}  "
        f"abs diff={abs(hol_rate - sigma_bits):.3g}"
    )


def demo_measurement_record_3state(seed: int = 97531, n: int = 80000) -> None:
    rng = np.random.default_rng(seed)
    # 3-state cycle with bias: 0->1, 1->2, 2->0 favored; ensures irreversibility
    base = np.array(
        [
            [0.90, 0.09, 0.01],
            [0.01, 0.90, 0.09],
            [0.09, 0.01, 0.90],
        ],
        dtype=float,
    )
    T = _row_stochastic(base)
    sigma_bits = entropy_production_rate_bits(T)
    x = sample_markov(T, n=n, rng=rng)
    hol_rate = klrate_holonomy_time_reversal_markov(x, k=3, R=3)
    print(
        f"[3-state record] EP analytic={sigma_bits:.6g}  KL-rate hol={hol_rate:.6g}  "
        f"diff={abs(hol_rate - sigma_bits):.3g}"
    )


# ------------------------------
# Main driver
# ------------------------------


def main():
    parser = argparse.ArgumentParser(description="UEC Battery: tests and validations.")
    parser.add_argument("--seed", type=int, default=12345, help="Base RNG seed.")
    parser.add_argument(
        "--n", type=int, default=150000, help="Sequence length for main Markov test."
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Alphabet size for main Markov test."
    )
    parser.add_argument("--order", type=int, default=3, help="KT Markov order R.")
    parser.add_argument(
        "--sweep_reps", type=int, default=6, help="Random-chain sweep repetitions."
    )
    parser.add_argument(
        "--run_optional", action="store_true", help="Run HMM and measurement demos."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: shorter sequences and fewer reps for quick sanity checks.",
    )
    args = parser.parse_args()

    set_seeds(args.seed)

    print("\n=== UEC Battery: starting ===")
    if args.fast:
        # Reduce workloads for quicker dev cycles
        args.n = max(60000, args.n // 3)
        args.sweep_reps = max(3, args.sweep_reps // 2)
    # 1) Gauge invariance (bijective loop ⇒ 0)
    test_gauge_invariance(
        seed=args.seed + 1, n=max(60000, args.n // 2), k=max(3, args.k), R=args.order
    )

    # 2) Coarse-graining (≥ 0)
    test_coarse_grain_nonneg(
        seed=args.seed + 2,
        n=max(80000, args.n // 2),
        k=max(4, args.k + 1),
        R=args.order,
    )

    # 3) Time reversal (≈ EP)
    test_time_reversal_equals_EP(
        seed=args.seed + 3, n=args.n, k=args.k, R=args.order, delta=0.6
    )
    test_ep_consistency_three_way(
        seed=args.seed + 30, n=args.n, k=args.k, R=args.order, delta=0.6
    )

    # 4) Observer-independence sanity (KT vs LZ evidence difference per symbol shrinks)
    test_observer_independence_trend(seed=args.seed + 4, k=args.k)

    # 5) Random-chain sweep aggregate statistics
    sweep_random_chains_EP_vs_KL(
        seed=args.seed + 5,
        reps=args.sweep_reps,
        n=max(120000, args.n // 2),
        k=args.k,
        delta=0.6,
        R=args.order,
    )

    # Optional demos
    if args.run_optional:
        demo_hmm_coarse_grain(seed=args.seed + 6, n=max(60000, args.n // 2))
        demo_measurement_record(seed=args.seed + 7, n=max(80000, args.n // 2))
        demo_measurement_record_3state(
            seed=args.seed + 8, n=max(80000, args.n // 2)
        )
        test_temporal_coarse_grain(
            seed=args.seed + 9, n=max(80000, args.n // 2), k=args.k, R=args.order
        )

    print("=== UEC Battery: all tests completed successfully ===")


if __name__ == "__main__":
    main()
