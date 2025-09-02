"""
Biological AoT utilities: conditional bins, lead–lag holonomy, and surrogates.

Implements:
- Conditional tokenization by driver bins (e.g., PAR, T), and day/night splits.
- Lead–lag holonomy (τ-hour) using KL-rate estimators from holonomy.
- IAAFT and circular-shift surrogates for rigorous nulls.
- Counts-based sanity estimator for time-reversal holonomy.
- Sufficiency, support asymmetry, and stationarity checks.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import math
import numpy as np

from .aot import discretize_series, quantile_bins
from .holonomy import (
    klrate_between_sequences,
    klrate_holonomy_time_reversal_markov,
    klrate_holonomy_leadlag_markov,
)
from .transforms import TransitionEncode, TransitionDecodeTakeSecond, TimeReverse, apply_loop
from .aot import train_forward_and_reverse_models, window_iter, auc_from_scores


# ----------------------
# Surrogate generators
# ----------------------

def iaaft_surrogate(x: np.ndarray, iters: int = 100, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate.

    Preserves the marginal distribution (by rank) and approximates the spectrum.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x.copy()
    # Sorted target distribution
    xs = np.sort(x)
    # Initial Gaussianized sequence with x's rank distribution
    y = rng.standard_normal(n)
    y = np.sort(y)
    # Random permutation to start
    perm = rng.permutation(n)
    y = y[perm]
    # Target amplitude spectrum
    X = np.fft.rfft(x)
    amp = np.abs(X)
    for _ in range(max(1, iters)):
        # Match spectrum
        Y = np.fft.rfft(y)
        Y = amp * np.exp(1j * np.angle(Y))
        y = np.fft.irfft(Y, n=n)
        # Match ranks (marginal)
        ranks = np.argsort(np.argsort(y))
        y = xs[ranks]
    return y


def circular_shift_surrogate(x: np.ndarray, shift: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n = len(x)
    if n == 0:
        return np.asarray(x, dtype=float)
    if shift is None:
        shift = int(rng.integers(0, n))
    return np.roll(np.asarray(x, dtype=float), shift)


def make_surrogates(x: np.ndarray, n: int, kind: str = "iaaft", iters: int = 100, rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for _ in range(max(0, int(n))):
        if kind == "iaaft":
            out.append(iaaft_surrogate(x, iters=iters, rng=rng))
        elif kind == "shift":
            out.append(circular_shift_surrogate(x, rng=rng))
        else:
            raise ValueError("unknown surrogate kind")
    return out


# ----------------------
# Counts-based estimator (sanity)
# ----------------------

def counts_from_sequence(seq: Sequence[int], k: int) -> np.ndarray:
    C = np.zeros((k, k), dtype=np.int64)
    for t in range(len(seq) - 1):
        i = int(seq[t])
        j = int(seq[t + 1])
        C[i, j] += 1
    return C


def klrate_time_reversal_from_counts(seq: Sequence[int], k: int, alpha: float = 0.5) -> float:
    """Approximate KL-rate holonomy using smoothed transition counts (R=1 analogue)."""
    C = counts_from_sequence(seq, k)
    N_i = C.sum(axis=1, keepdims=True)
    T_hat = (C + alpha) / (N_i + alpha * k)
    # Encode transitions -> reverse -> decode second, compare to seq[1:]
    E = TransitionEncode(k)
    Rv = TimeReverse()
    D2 = TransitionDecodeTakeSecond(k)
    q_seq, _ = apply_loop(seq, list(range(k)), [E, Rv, D2])
    p_eval = list(seq)[1:]
    # Empirical per-symbol KL using T_hat as model for p_eval and q_seq via context-free proxy
    # Compute NLL under 0th-order proxy from T_hat's marginal next-state freq
    next_freq = (C.sum(axis=0) + alpha) / (C.sum() + alpha * k)
    def nll_zeroth(x: Sequence[int]) -> float:
        eps = 1e-12
        return float(-np.sum(np.log2(np.maximum(next_freq[np.asarray(x, dtype=int)], eps))))
    HP = nll_zeroth(p_eval)
    HQ = nll_zeroth(q_seq)
    n = max(1, len(p_eval))
    return float((HQ - HP) / n)


# ----------------------
# Conditional tokenization and splits
# ----------------------

def joint_tokens_from_series(series: List[np.ndarray], k_list: List[int]) -> Tuple[np.ndarray, int]:
    """Quantile-discretize each series to k_i and return joint-token sequence.

    token = s0 + k0*(s1 + k1*(s2 + ...)).
    """
    assert len(series) == len(k_list)
    discs: List[np.ndarray] = []
    for x, k in zip(series, k_list):
        discs.append(discretize_series(np.asarray(x, dtype=float), int(k)))
    # Joint index
    token = discs[0].astype(int)
    mult = int(k_list[0])
    for i in range(1, len(discs)):
        token = token + mult * discs[i].astype(int)
        mult *= int(k_list[i])
    return token.astype(int), int(mult)


def boolean_mask_by_quantile_bin(x: np.ndarray, nbins: int, which: int) -> np.ndarray:
    edges = quantile_bins(np.asarray(x, dtype=float), nbins)
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, nbins - 1)
    return (idx == int(which))


# ----------------------
# Diagnostics
# ----------------------

def check_sufficiency(n_train: int, n_eval: int, k: int, R: int) -> Dict[str, bool]:
    need_train = 50 * (k ** R)
    need_eval = 10 * (k ** R)
    return {
        "ok_train": n_train >= need_train,
        "ok_eval": n_eval >= need_eval,
        "need_train": need_train,
        "need_eval": need_eval,
    }


def support_asymmetry(seq: Sequence[int], k: int) -> float:
    C = counts_from_sequence(seq, k)
    forward_edges = (C > 0)
    reverse_edges = (C.T > 0)
    asym = np.logical_and(forward_edges, np.logical_not(reverse_edges))
    total = forward_edges.sum()
    return 0.0 if total == 0 else float(asym.sum() / total)


def stationarity_delta_bits(seq: Sequence[int], k: int, R: int = 1, coder: str = "kt") -> float:
    n = len(seq)
    if n < 4:
        return 0.0
    a = seq[: n // 2]
    b = seq[n // 2 :]
    # Compare time-reversal holonomy on halves
    ha = klrate_holonomy_time_reversal_markov(a, k=k, R=R, coder=coder)
    hb = klrate_holonomy_time_reversal_markov(b, k=k, R=R, coder=coder)
    return float(abs(hb - ha))


# ----------------------
# Evaluation helpers
# ----------------------

def holonomy_day_night(
    x: np.ndarray,
    par: np.ndarray,
    k_x: int = 4,
    R: int = 1,
    coder: str = "kt",
    day_thresh_quantile: float = 0.5,
) -> Dict[str, float]:
    """Compute time-reversal holonomy on Δx in day vs night splits by PAR quantile.
    Returns dict with hol_day, hol_night, auc_star-like separation proxy.
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x, prepend=x[:1])
    s = discretize_series(dx, k=k_x).astype(int)
    # Day/night mask by PAR threshold quantile
    q = np.quantile(par, day_thresh_quantile)
    day_mask = (par >= q)
    night_mask = (par < q)
    sd = s[day_mask]
    sn = s[night_mask]
    hol_day = klrate_holonomy_time_reversal_markov(sd.tolist(), k=k_x, R=R, coder=coder) if len(sd) > 100 else 0.0
    hol_night = klrate_holonomy_time_reversal_markov(sn.tolist(), k=k_x, R=R, coder=coder) if len(sn) > 100 else 0.0
    sep = float(hol_day - hol_night)
    return {"hol_day": hol_day, "hol_night": hol_night, "day_night_delta": sep, "n_day": int(len(sd)), "n_night": int(len(sn))}


def holonomy_leadlag(
    x: np.ndarray,
    k_x: int = 4,
    R: int = 1,
    tau_hours: int = 1,
    coder: str = "kt",
) -> float:
    dx = np.diff(np.asarray(x, dtype=float), prepend=x[:1])
    s = discretize_series(dx, k=k_x).astype(int)
    return klrate_holonomy_leadlag_markov(s.tolist(), k=k_x, tau=max(1, int(tau_hours)), R=R, coder=coder)


def surrogate_pvalue(stat_obs: float, stats_null: List[float], tail: str = "right") -> float:
    arr = np.asarray(stats_null, dtype=float)
    if tail == "right":
        return float((np.sum(arr >= stat_obs) + 1) / (len(arr) + 1))
    elif tail == "left":
        return float((np.sum(arr <= stat_obs) + 1) / (len(arr) + 1))
    else:  # two-sided
        return float((np.sum(np.abs(arr) >= abs(stat_obs)) + 1) / (len(arr) + 1))


def coder_agreement(seq: Sequence[int], k: int, R: int = 1, alpha: float = 0.5) -> Tuple[float, float]:
    """Return (delta_bits, r_like) comparing KT vs counts across two halves.
    delta_bits: |KT - counts| on full sequence.
    r_like: correlation-like agreement across halves (2 points only -> degenerate, return 1-diff_norm).
    """
    kt = klrate_holonomy_time_reversal_markov(seq, k=k, R=R, coder="kt")
    ct = klrate_time_reversal_from_counts(seq, k=k, alpha=alpha)
    delta = float(abs(kt - ct))
    # crude two-point agreement proxy
    n = len(seq)
    a = seq[: n // 2]
    b = seq[n // 2 :]
    kta = klrate_holonomy_time_reversal_markov(a, k=k, R=R, coder="kt")
    cta = klrate_time_reversal_from_counts(a, k=k, alpha=alpha)
    ktb = klrate_holonomy_time_reversal_markov(b, k=k, R=R, coder="kt")
    ctb = klrate_time_reversal_from_counts(b, k=k, alpha=alpha)
    v1 = np.array([kta, ktb])
    v2 = np.array([cta, ctb])
    if np.allclose(v1.std(), 0) or np.allclose(v2.std(), 0):
        r = 0.0
    else:
        r = float(np.corrcoef(v1, v2)[0, 1])
    return delta, r


# ----------------------
# Token-level AoT (scores and AUC*)
# ----------------------

def aot_on_tokens(
    seq: Sequence[int],
    k: int,
    R: int = 1,
    win: int = 256,
    stride: int = 128,
    train_frac: float = 0.5,
    coder: str = "kt",
    rng: Optional[np.random.Generator] = None,
) -> dict:
    s = list(map(int, seq))
    n = len(s)
    ntr = int(n * train_frac)
    train, test = s[:ntr], s[ntr:]
    if len(test) < max(4, win * 2):
        raise ValueError(f"Not enough test data for AoT windows (len(test)={len(test)}, win={win}).")
    Pf, Qf = train_forward_and_reverse_models(train, k, R, coder)
    wins_fwd = window_iter(test, win, stride)
    E_win, Rv_win, D2_win = TransitionEncode(k), TimeReverse(), TransitionDecodeTakeSecond(k)
    wins_q = [apply_loop(w, list(range(k)), [E_win, Rv_win, D2_win])[0] for w in wins_fwd]
    from .aot import signed_lr_score  # lazy import to avoid cycles in type checking
    scores_fwd = np.array([signed_lr_score(w, Pf, Qf, coder) for w in wins_fwd], dtype=float)
    scores_rev = np.array([signed_lr_score(wq, Pf, Qf, coder) for wq in wins_q], dtype=float)
    auc = auc_from_scores(scores_fwd, scores_rev)
    # Per-window holonomy estimate via KL between w[1:] and loop(w)
    vals: List[float] = []
    for w, qw in zip(wins_fwd, wins_q):
        if len(w) > 1 and len(qw) > 0:
            vals.append(klrate_between_sequences(w[1:], qw, k, R, coder))
    hol_mean = float(np.mean(vals)) if vals else 0.0
    return {
        "auc": float(auc),
        "scores_forward": scores_fwd.tolist(),
        "scores_reversed": scores_rev.tolist(),
        "holonomy_bits": hol_mean,
        "n_windows": len(wins_fwd),
    }


def benjamini_hochberg(pvals: List[float], alpha: float = 0.05) -> List[bool]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    thresh = [(i + 1) * alpha / m for i in range(m)]
    keep = [False] * m
    max_k = -1
    for rank, idx in enumerate(order):
        if pvals[idx] <= thresh[rank]:
            max_k = rank
    if max_k >= 0:
        for rank, idx in enumerate(order):
            if rank <= max_k:
                keep[idx] = True
    return keep
