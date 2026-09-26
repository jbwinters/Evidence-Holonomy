"""
Anytime-valid test of time-reversal symmetry (detailed balance).

Null hypothesis H0: the sequence is a stationary first-order Markov chain that
satisfies detailed balance, pi_i T_ij = pi_j T_ji. The test statistic

    E_t = prod_{s=1..t} q_{s-1}(x_s | x_{s-1}) / sup_{T reversible} prod_{s=1..t} T(x_{s-1}, x_s)

uses a predictable numerator (a Krichevsky-Trofimov predictor fitted to
x_0..x_{s-1} only) and the reversible maximum likelihood in the denominator.
Under H0 with true matrix T0, the denominator is at least the likelihood under
T0, so E_t is bounded by a nonnegative martingale with mean 1 (conditional on
x_0). By Ville's inequality, P(E_t >= 1/alpha for some t) <= alpha: the test
may be checked after every step and stopped at any time. This is the running-MLE
"universal inference" construction of Wasserman, Ramdas and Balakrishnan (2020).

Under an irreversible stationary chain, log2(E_t)/t converges to

    rho = sum_ij f_ij log2( 2 f_ij / (f_ij + f_ji) ),   f_ij = pi_i T_ij,

the KL rate from T to the nearest reversible chain, attained at the additive
reversiblization (T + T*)/2. rho <= sigma/2, rho ~ sigma/4 near equilibrium,
and rho stays finite when sigma is infinite (one-way transitions).

The null is first-order reversible Markov. Records of hidden reversible chains
are reversible but not Markov; for them validity is not guaranteed by this
argument (see the accompanying experiments). loop_eprocess_reversibility is a
second, simpler test with exact validity and much lower overhead near
equilibrium.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

LOG2 = np.log(2.0)


def transition_counts(seq: Sequence[int], k: int) -> np.ndarray:
    """k x k matrix of transition counts C[i, j] = #{t : x_t = i, x_{t+1} = j}."""
    x = np.asarray(seq, dtype=np.int64)
    C = np.zeros((k, k), dtype=float)
    np.add.at(C, (x[:-1], x[1:]), 1.0)
    return C


def reversible_mle(C: np.ndarray, tol: float = 1e-12, max_iter: int = 100_000,
                   X0: np.ndarray | None = None, return_flows: bool = False):
    """Maximum-likelihood reversible transition matrix for counts C.

    Works on a batch: C has shape (..., k, k). Returns (T_hat, loglik), where
    loglik = sum_ij C_ij log T_hat_ij in nats, plus the symmetric flow matrix X
    if return_flows is True. Uses the fixed-point iteration
    X_ij <- (C_ij + C_ji) / (N_i/x_i + N_j/x_j), T_ij = X_ij / x_i, the standard
    self-consistent equations of the reversible maximum likelihood (the problem
    has a convex reformulation; Trendelkamp-Schroer et al., J. Chem. Phys. 143,
    174101, 2015). Convergence is linear; the stopping rule is not a certified
    optimality gap. X0 warm-starts the
    iteration (e.g. from a previous checkpoint). Stops when every log-likelihood
    in the batch changes by less than tol (relative) over 10 iterations.
    """
    C = np.asarray(C, dtype=float)
    S = C + np.swapaxes(C, -1, -2)
    N = C.sum(axis=-1)
    X = (S / 2.0 if X0 is None else np.where(S > 0, np.maximum(X0, 1e-300), 0.0)) + 1e-300
    prev = _loglik(C, X)
    for it in range(1, max_iter + 1):
        x = X.sum(axis=-1)
        ratio = np.where(N > 0, N / np.maximum(x, 1e-300), 0.0)
        denom = ratio[..., :, None] + ratio[..., None, :]
        X = np.where(S > 0, S / np.maximum(denom, 1e-300), 0.0)
        if it % 10 == 0:
            ll = _loglik(C, X)
            if np.all(np.abs(ll - prev) <= tol * np.maximum(1.0, np.abs(ll))):
                break
            prev = ll
    x = X.sum(axis=-1, keepdims=True)
    T = np.where(x > 0, X / np.maximum(x, 1e-300), 0.0)
    ll = _loglik(C, X)
    return (T, ll, X) if return_flows else (T, ll)


def _loglik(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    x = X.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logT = np.where(C > 0, np.log(np.where(C > 0, X, 1.0) / np.where(C > 0, x, 1.0)), 0.0)
    return (C * logT).sum(axis=(-1, -2))


def unconstrained_loglik(C: np.ndarray) -> np.ndarray:
    """Maximized first-order Markov log-likelihood (nats) for counts C (batched)."""
    C = np.asarray(C, dtype=float)
    N = C.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logT = np.where(C > 0, np.log(np.where(C > 0, C, 1.0) / np.where(N > 0, N, 1.0)), 0.0)
    return (C * logT).sum(axis=(-1, -2))


def eprocess_reversibility(
    seq: Sequence[int], k: int, checkpoints: Sequence[int] | None = None, kt: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """log2 E_t of the reversibility e-process at the given checkpoints.

    checkpoints are numbers of transitions t (1 <= t <= len(seq) - 1); by default
    every transition. Returns (checkpoints, log2_E). Reject H0 at level alpha the
    first time log2_E >= log2(1/alpha).
    """
    x = np.asarray(seq, dtype=np.int64)
    n = len(x) - 1
    if checkpoints is None:
        checkpoints = np.arange(1, n + 1)
    cps = np.asarray(sorted(set(int(c) for c in checkpoints if 1 <= c <= n)), dtype=np.int64)
    counts = np.zeros((k, k))
    row = np.zeros(k)
    log_num = 0.0
    out = np.empty(len(cps))
    ci = 0
    X = None
    for t in range(1, n + 1):
        a, b = x[t - 1], x[t]
        log_num += np.log((counts[a, b] + kt) / (row[a] + kt * k))
        counts[a, b] += 1.0
        row[a] += 1.0
        while ci < len(cps) and cps[ci] == t:
            _, ll_rev, X = reversible_mle(counts, X0=X, return_flows=True)
            out[ci] = (log_num - ll_rev) / LOG2
            ci += 1
    return cps, out


def reversible_projection_rate(T: np.ndarray, pi: np.ndarray | None = None) -> float:
    """rho = min over reversible Q of the KL rate D(T || Q), in bits/step.

    Equals sum_ij f_ij log2(2 f_ij / (f_ij + f_ji)) with f_ij = pi_i T_ij: a
    Jensen-Shannon-type divergence between forward and backward edge flows.
    """
    T = np.asarray(T, dtype=float)
    if pi is None:
        from .markov import stationary_distribution

        pi = stationary_distribution(T)
    f = pi[:, None] * T
    s = f + f.T
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(f > 0, f * np.log2(2.0 * f / np.where(s > 0, s, 1.0)), 0.0)
    return float(terms.sum())


# ------------------------------------------------------------------
# Loop-orientation e-process
# ------------------------------------------------------------------


def loop_eprocess_reversibility(
    seq: Sequence[int], k: int, checkpoints: Sequence[int] | None = None, kt: float = 0.5,
    burn_in: int | None = None, ref_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """log2 E_t of the loop-orientation test at the given checkpoints.

    After a burn-in, the reference state r is fixed: the most visited state in
    x_0..x_burn_in (ties go to the lowest index), unless ref_state is given in
    advance. From the first visit to r at or after the burn-in, the path splits
    into loops that leave r and return to it. Loops are independent (strong
    Markov property), and under detailed balance a loop and its reversal have
    equal probability (Kolmogorov's criterion). So, given the unordered pair
    {loop, reversed loop}, the observed orientation is a fair coin. When a loop
    e closes, E is multiplied by 2 q(e) / (q(e) + q(reversed e)), where q is a
    Krichevsky-Trofimov transition estimate frozen at the loop's start.

    Validity: indexed by completed loops, E is a nonnegative martingale with
    mean 1 under H0, and it is constant between loop completions. By Ville's
    inequality, P(E_t >= 1/alpha for some t) <= alpha, so the test may be
    checked after every step. E_t is not an e-process for arbitrary stopping
    times at the step level: a stopping rule that looks inside an open loop can
    make E[E_tau] exceed 1. Only the level-alpha test is guaranteed.

    checkpoints are numbers of transitions t (1 <= t <= len(seq) - 1). burn_in
    defaults to 10 k. Returns (checkpoints, log2_E).
    """
    x = np.asarray(seq, dtype=np.int64)
    n = len(x) - 1
    if checkpoints is None:
        checkpoints = np.arange(1, n + 1)
    cps = np.asarray(sorted(set(int(c) for c in checkpoints if 1 <= c <= n)), dtype=np.int64)
    burn = 10 * k if burn_in is None else int(burn_in)
    burn = min(burn, n)
    r = int(ref_state) if ref_state is not None else int(np.argmax(np.bincount(x[: burn + 1], minlength=k)))
    counts = np.zeros((k, k))
    logE = 0.0
    acc = 0.0
    started = bool(burn == 0 and x[0] == r)
    A = _log_ratio(counts, kt) if started else None
    out = np.empty(len(cps))
    ci = 0
    for t in range(1, n + 1):
        a, b = x[t - 1], x[t]
        if started:
            acc += A[a, b]
        counts[a, b] += 1.0
        if b == r and t >= burn:
            if started:
                logE += 1.0 - np.log2(1.0 + 2.0 ** (-acc))
            started = True
            acc = 0.0
            A = _log_ratio(counts, kt)
        while ci < len(cps) and cps[ci] == t:
            out[ci] = logE
            ci += 1
    return cps, out


def _log_ratio(counts: np.ndarray, kt: float) -> np.ndarray:
    k = counts.shape[-1]
    T = (counts + kt) / (counts.sum(axis=-1, keepdims=True) + kt * k)
    L = np.log2(T)
    return L - np.swapaxes(L, -1, -2)
