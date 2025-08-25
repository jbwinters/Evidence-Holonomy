from __future__ import annotations
import math
from typing import List, Sequence, Tuple
import numpy as np


def _row_stochastic(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0.0] = 1.0
    return T / rs


def stationary_distribution(T: np.ndarray, tol: float = 1e-12, max_iter: int = 200_000) -> np.ndarray:
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
                s += (pi[i] * T[i, j]) * 1000.0
    return float(s)


def random_markov_biased(
    k: int,
    delta: float = 0.4,
    min_prob: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
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
    return p * math.log(p / q, 2.0) + q * math.log(q / p, 2.0)

