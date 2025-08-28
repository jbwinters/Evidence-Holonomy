"""
Markov utilities and analytic entropy production (EP) in bits/step.

This module provides:
- Row-stochastic normalization and stationary distribution via power iteration.
- A safe Markov sampler with explicit RNG threading.
- Analytic entropy production rate for finite-state Markov chains in bits/step:
  sigma = sum_{i,j} pi_i T_ij log2((pi_i T_ij)/(pi_j T_ji)).
  Note: If T has one-way edges (T_ij>0 while T_ji==0), the KL rate is infinite;
  we add a large finite surrogate contribution so the function returns a number,
  and the tests/documentation call out the absolute continuity caveat.
- Random biased transition matrices (irreversible, strictly positive entries).
- A tiny HMM sampler for demos, and a 3-state ring generator with closed-form EP.

See uec_theory.tex for the theory, reductions, and references.
"""

from __future__ import annotations
import math
from typing import List, Sequence, Tuple
import numpy as np


def _row_stochastic(T: np.ndarray) -> np.ndarray:
    """Return a row-stochastic copy of T (rows sum to 1, guarding zero rows)."""
    T = np.asarray(T, dtype=float)
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return T / rs


def stationary_distribution(T: np.ndarray, tol: float = 1e-12, max_iter: int = 200_000) -> np.ndarray:
    """Compute the stationary distribution of a finite-state Markov chain.

    Uses power iteration on the row-stochastic transition matrix T. The result
    is normalized and finite even for numerically ill-conditioned inputs.
    """
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
    init: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a length-n path from a finite-state Markov chain.

    - T is row-stochastic; if not, it is normalized.
    - init is an optional initial distribution; otherwise the stationary pi.
    - rng threads randomness for reproducibility across demos/tests.
    """
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


def entropy_production_rate_bits(T: np.ndarray, strict: bool = False) -> float:
    """Analytic EP for a Markov chain in bits/step.

    sigma = sum_{i,j} pi_i T_ij log2((pi_i T_ij)/(pi_j T_ji)).
    If detailed balance holds (pi_i T_ij == pi_j T_ji for all i,j), sigma == 0.
    With one-way edges, the exact EP diverges; we add a large surrogate term
    unless strict=True, in which case we raise ValueError.
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
                if strict:
                    raise ValueError("EP diverges: one-way edge present (violates absolute continuity).")
                s += (pi[i] * T[i, j]) * 1000.0
    return float(s)


def random_markov_biased(
    k: int,
    delta: float = 0.4,
    min_prob: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Construct an irreducible, mildly biased kxk transition matrix.

    - Draws a random row-stochastic matrix, then biases i->i+1 over i->i-1.
    - Floors all entries to min_prob and renormalizes to avoid zeros.
    """
    if rng is None:
        rng = np.random.default_rng()
    G = rng.gamma(shape=1.0, scale=1.0, size=(k, k))
    T = G / G.sum(axis=1, keepdims=True)
    for i in range(k):
        T[i, (i + 1) % k] *= 1.0 + delta
        T[i, (i - 1) % k] *= 1.0 - 0.5 * delta
    T = T / T.sum(axis=1, keepdims=True)
    T = np.maximum(T, min_prob)
    T = T / T.sum(axis=1, keepdims=True)
    return T


def sample_HMM(
    A: np.ndarray,
    B: np.ndarray,
    n: int,
    rng: np.random.Generator | None = None,
) -> Tuple[List[int], int, int]:
    """Sample observed symbols from a simple HMM with transitions A and emissions B."""
    if rng is None:
        rng = np.random.default_rng()
    A = _row_stochastic(np.array(A, dtype=float))
    B = _row_stochastic(np.array(B, dtype=float))
    k = A.shape[0]
    m = B.shape[1]
    pi = stationary_distribution(A)
    z = int(rng.choice(np.arange(k), p=pi))
    obs: List[int] = []
    for _ in range(n):
        o = int(rng.choice(np.arange(m), p=B[z]))
        obs.append(o)
        z = int(rng.choice(np.arange(k), p=A[z]))
    return obs, k, m


def ring_chain_3(p: float, q: float) -> np.ndarray:
    """3-state ring with forward p, backward q, stay 1-p-q (mod 3)."""
    assert 0 < p < 1 and 0 < q < 1 and p + q < 1
    T = np.array(
        [
            [1 - p - q, p, q],
            [q, 1 - p - q, p],
            [p, q, 1 - p - q],
        ],
        dtype=float,
    )
    return _row_stochastic(T)


def ring_ep_bits(p: float, q: float) -> float:
    """Closed-form EP for the 3-state ring in bits/step (uniform stationary)."""
    return p * math.log(p / q, 2.0) + q * math.log(q / p, 2.0)
