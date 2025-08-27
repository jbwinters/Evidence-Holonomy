#!/usr/bin/env python3
"""
One-off BTC analysis using UEC (KL-rate holonomy) as a regime detector.

Inputs:
  - CSV at data/kaggle/btcusd_1min/btcusd_1-min_data.csv (Timestamp,Open,High,Low,Close,Volume)

Methodology (adapted from the blueprint):
  - Symbolize joint (return x volume) via global quantile bins.
  - Compute sliding-window irreversibility score (UEC) in bits/step.
    * Default fast estimator uses smoothed transition-counts EPR per step.
    * Optional 'kt' estimator calls uec_battery.klrate_holonomy_time_reversal_markov
      (slower but truer to AoT holonomy; use on shorter spans).
  - Direction proxy p_up via simple momentum ensemble (no external deps).
  - Risk sizing via (half-)Kelly with volatility targeting and regime amplification.
  - Simple state machine: trend entries only when UEC_z > z_hi and |edge| > edge_th; flat otherwise.

Outputs:
  - results/btc_uec_analysis.csv: per-bar metrics (uec, uec_z, p_up, edge, size, pos, pnl, equity,...)
  - Console summary with basic performance stats and ablation pointers.

Notes:
  - Uses only stdlib + numpy; pandas/sklearn not required.
  - For very long streams, use --tail or --limit to bound compute.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Ensure repo root on sys.path for local module imports
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

# Local import (repo root) for the tested KL-holonomy implementation and counts-based EP.
try:  # prefer local module
    import uec_battery as uec
except Exception:
    uec = None  # counts-based fallback still works


# ------------------------------
# Utilities
# ------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ewm(series: np.ndarray, span: int, init: Optional[float] = None) -> np.ndarray:
    """Simple exponential weighted mean with span -> alpha mapping (alpha = 2/(span+1))."""
    if span <= 1:
        return series.astype(float).copy()
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0] if init is None else init
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def ewm_mean_std(series: np.ndarray, span: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return EWM mean and std via first/second moment recursion."""
    if span <= 1:
        return series.astype(float), np.zeros_like(series, dtype=float)
    alpha = 2.0 / (span + 1.0)
    m = np.empty_like(series, dtype=float)
    v = np.empty_like(series, dtype=float)
    m[0] = series[0]
    v[0] = 0.0
    for i in range(1, len(series)):
        dx = series[i] - m[i - 1]
        m[i] = m[i - 1] + alpha * dx
        v[i] = (1.0 - alpha) * (v[i - 1] + alpha * dx * dx)
    std = np.sqrt(np.maximum(v, 0.0))
    return m, std


def zscore_fast(x: np.ndarray, span: int) -> np.ndarray:
    m, s = ewm_mean_std(x, span)
    s = np.where(s <= 1e-12, 1.0, s)
    return (x - m) / s


def quantile_edges(x: np.ndarray, k: int) -> np.ndarray:
    """Compute k-quantile bin edges as an array of length k+1 (min..max)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 1.0])
    qs = np.linspace(0.0, 1.0, num=k + 1)
    try:
        edges = np.nanquantile(x, qs)
    except Exception:
        edges = np.quantile(x, qs)
    # Ensure strictly increasing by adding tiny jitter if needed
    eps = 1e-12
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges


def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values x into integer bins [0..k-1] according to edges of length k+1."""
    # searchsorted returns index in [1..k] for the right bin; clamp to [0..k-1]
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx.astype(np.int64)


def rolling_counts_window(sym: np.ndarray, t_end: int, W: int, k: int) -> np.ndarray:
    """Compute kxk transition counts on sym[t_end-W .. t_end] inclusive transitions."""
    start = t_end - W
    if start < 0:
        start = 0
    # transitions from sym[start..t_end-1] -> sym[start+1..t_end]
    a = sym[start : t_end]
    b = sym[start + 1 : t_end + 1]
    C = np.zeros((k, k), dtype=np.int64)
    # vectorized bincount into 2D
    flat = a * k + b
    counts = np.bincount(flat, minlength=k * k)
    C[:, :] = counts.reshape(k, k)
    return C


def epr_bits_from_counts_smoothed(counts: np.ndarray, alpha: float = 0.5) -> float:
    """Entropy production rate (bits/step) from smoothed counts (compatible with uec_battery)."""
    k = counts.shape[0]
    N_i = counts.sum(axis=1, keepdims=True)
    T_hat = (counts + alpha) / (N_i + alpha * k)
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


def compute_uec_stream(
    symbols: np.ndarray,
    k: int,
    W: int = 256,
    R: int = 3,
    method: str = "counts",
    kt_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sliding-window UEC rate (bits/step) and its fast z-score.

    method:
      - 'counts': fast counts/EPR approximation (default)
      - 'kt'    : KT holonomy via uec_battery (slow; use kt_stride>1 for speed)
    """
    n = len(symbols)
    uec_vals = np.full(n, np.nan, dtype=float)
    if method == "kt" and uec is None:
        method = "counts"
    for t in range(W, n - 1):
        if method == "counts":
            C = rolling_counts_window(symbols, t, W=W, k=k)
            hol = epr_bits_from_counts_smoothed(C, alpha=0.5)
        else:
            # Evaluate only on strides to reduce cost; interpolate in between.
            if (t - W) % max(1, kt_stride) != 0:
                continue
            win = symbols[t - W : t + 1]
            hol = float(uec.klrate_holonomy_time_reversal_markov(win.tolist(), k=k, R=R))
        uec_vals[t] = hol
    # If kt_stride>1, fill NaNs by forward/backward fill then linear interpolate
    if method == "kt" and kt_stride > 1:
        idx = np.arange(n)
        mask = np.isfinite(uec_vals)
        if mask.any():
            uec_vals = np.interp(idx, idx[mask], uec_vals[mask])
    # Smooth and z-score
    span = max(3 * W, 16)
    uec_vals[np.isnan(uec_vals)] = np.nanmean(uec_vals)
    uec_smooth = ewm(uec_vals, span=span)
    uec_z = zscore_fast(uec_smooth, span=span)
    return uec_vals, uec_z


def bootstrap_ci_counts_for_window(
    symbols: np.ndarray, t: int, W: int, k: int, B: int, block: int
) -> Tuple[float, float]:
    """Moving-block bootstrap CI for counts-based holonomy in one window."""
    start = t - W
    if start < 0:
        start = 0
    win = symbols[start : t + 1]
    if len(win) < 2 or B <= 0:
        return float("nan"), float("nan")
    # bootstrap samples
    n = len(win)
    if block <= 0:
        block = max(2, n // 8)
    rng = np.random.default_rng(123)
    vals = []
    for _ in range(B):
        idx = []
        while len(idx) < n:
            s = int(rng.integers(0, max(1, n - block)))
            idx.extend(list(range(s, min(s + block, n))))
        idx = idx[:n]
        resamp = win[idx]
        # counts on resampled sequence
        Cb = np.zeros((k, k), dtype=np.int64)
        a = resamp[:-1]
        b = resamp[1:]
        flat = a * k + b
        counts = np.bincount(flat, minlength=k * k)
        Cb[:, :] = counts.reshape(k, k)
        vals.append(epr_bits_from_counts_smoothed(Cb))
    vals = np.array(vals, dtype=float)
    lo = float(np.nanquantile(vals, 0.025))
    hi = float(np.nanquantile(vals, 0.975))
    return lo, hi


# ------------------------------
# Direction and sizing modules
# ------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def direction_proxy(
    r: np.ndarray,
    v_z: np.ndarray,
    body: np.ndarray,
    span_fast: int = 12,
    span_slow: int = 48,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return p_up, edge, exp_move proxy.
    - p_up: logistic of MACD + body + v_z
    - edge: p_up - 0.5
    - exp_move: EWM of |r| as magnitude proxy
    """
    r = r.astype(float)
    ema_fast = ewm(r, span_fast)
    ema_slow = ewm(r, span_slow)
    macd = ema_fast - ema_slow
    # Normalize features by their EW stds
    macd_n = zscore_fast(macd, span=10 * span_slow)
    body_n = zscore_fast(body, span=10 * span_slow)
    vzn = np.clip(v_z, -4.0, 4.0)
    # Heuristic weights (no training; bounded to avoid overflow)
    lin = 0.9 * macd_n + 0.2 * body_n + 0.1 * vzn
    lin = np.clip(lin, -6.0, 6.0)
    p_up = sigmoid(lin)
    edge = p_up - 0.5
    exp_move = ewm(np.abs(r), span=span_slow)
    return p_up, edge, exp_move


def position_size(
    edge: np.ndarray,
    exp_move: np.ndarray,
    realized_vol: np.ndarray,
    uec_z: np.ndarray,
    target_vol_per_bar: float,
    kelly_cap: float = 0.5,
    max_leverage: float = 2.0,
) -> np.ndarray:
    mu = edge * exp_move
    var = np.maximum(realized_vol ** 2, 1e-10)
    f_kelly = np.clip(mu / var, -1.0, 1.0)
    f_kelly *= kelly_cap
    vol_scalar = target_vol_per_bar / np.maximum(realized_vol, 1e-8)
    regime_amp = 1.0 + 0.5 * np.tanh(uec_z / 2.0)
    size = f_kelly * vol_scalar * regime_amp
    return np.clip(size, -max_leverage, max_leverage)


# ------------------------------
# Data loading and feature engineering
# ------------------------------


@dataclass
class MarketRow:
    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_btc_csv(
    path: str,
    tail: Optional[int] = None,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV with header: Timestamp,Open,High,Low,Close,Volume. Returns arrays sorted by ts."""
    buf_ts: Deque[float] = deque(maxlen=tail or 10_000_000)
    buf_o: Deque[float] = deque(maxlen=tail or 10_000_000)
    buf_h: Deque[float] = deque(maxlen=tail or 10_000_000)
    buf_l: Deque[float] = deque(maxlen=tail or 10_000_000)
    buf_c: Deque[float] = deque(maxlen=tail or 10_000_000)
    buf_v: Deque[float] = deque(maxlen=tail or 10_000_000)
    n = 0
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        for row in rdr:
            try:
                ts, o, h, l, c, v = row[:6]
                buf_ts.append(float(ts))
                buf_o.append(float(o))
                buf_h.append(float(h))
                buf_l.append(float(l))
                buf_c.append(float(c))
                buf_v.append(float(v))
                n += 1
                if limit and n >= limit:
                    break
            except Exception:
                continue
    ts = np.array(buf_ts, dtype=float)
    order = np.argsort(ts)
    return (
        ts[order],
        np.array(buf_o, dtype=float)[order],
        np.array(buf_h, dtype=float)[order],
        np.array(buf_l, dtype=float)[order],
        np.array(buf_c, dtype=float)[order],
        np.array(buf_v, dtype=float)[order],
    )


def feature_block(ts, o, h, l, c, v):
    # log return
    r = np.zeros_like(c, dtype=float)
    r[1:] = np.log(np.maximum(c[1:], 1e-12)) - np.log(np.maximum(c[:-1], 1e-12))
    # candle body, wick
    with np.errstate(divide="ignore", invalid="ignore"):
        body = (c - o) / np.where(o != 0, o, 1.0)
        wick = (h - l) / np.where(c != 0, c, 1.0)
    # realized vol proxy
    rv = ewm(r ** 2, span=64)
    # skew proxy via EWM third central moment
    m1 = ewm(r, span=64)
    m2 = ewm((r - m1) ** 2, span=64)
    m3 = ewm((r - m1) ** 3, span=64)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = m3 / np.maximum(m2 ** 1.5, 1e-12)
    # volume z-score (long span)
    v_mean, v_std = ewm_mean_std(v, span=512)
    v_z = (v - v_mean) / np.where(v_std > 1e-12, v_std, 1.0)
    return r, body, wick, rv, skew, v_z


def symbolize_joint(r: np.ndarray, v: np.ndarray, k_r: int, k_v: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Exclude the first NaN return from quantiles
    r_valid = r[np.isfinite(r)]
    v_valid = v[np.isfinite(v)]
    edges_r = quantile_edges(r_valid, k_r)
    edges_v = quantile_edges(v_valid, k_v)
    r_bin = bin_index(r, edges_r)
    v_bin = bin_index(v, edges_v)
    sym = r_bin * k_v + v_bin
    return sym.astype(np.int64), edges_r, edges_v


def check_loop_closure(symbols: np.ndarray, k: int, W: int) -> Tuple[bool, str]:
    """Verify encode→reverse→decode returns same alphabet and n-1 length (if uec available)."""
    if uec is None:
        return True, "uec_battery not available; skipped (counts mode)."
    try:
        E = uec.TransitionEncode(k)
        Rv = uec.TimeReverse()
        D2 = uec.TransitionDecodeTakeSecond(k)
        # take a short window in the middle
        mid = len(symbols) // 2
        t = max(W, mid)
        win = symbols[t - W : t + 1]
        q_seq, a = uec.apply_loop(win.tolist(), list(range(k)), [E, Rv, D2])
        ok_alpha = (a == list(range(k)))
        ok_len = (len(q_seq) == len(win) - 1)
        if ok_alpha and ok_len:
            return True, f"loop ok (alpha=k={k}, len→{len(win)-1})."
        return False, f"loop mismatch: alpha_ok={ok_alpha}, len_ok={ok_len}"
    except Exception as e:
        return False, f"loop check error: {e}"


def check_support_overlap(symbols: np.ndarray, k: int, W: int, samples: int = 10) -> Tuple[int, int]:
    """Count (i->j) seen while (j->i) never seen across sampled windows (pre‑smoothing)."""
    n = len(symbols)
    if n <= W + 2:
        return 0, 0
    idx = np.linspace(W, n - 2, num=max(1, samples), dtype=int)
    bad = 0
    total = 0
    for t in idx:
        C = rolling_counts_window(symbols, int(t), W=W, k=k)
        total += (C > 0).sum()
        # find asymmetric supports
        nz = np.argwhere(C > 0)
        for i, j in nz:
            if C[j, i] == 0:
                bad += 1
    return bad, total


def check_stationarity_split(uec_vals: np.ndarray) -> Tuple[float, float, float]:
    """Compare mean holonomy first vs second half; return (mean1, mean2, delta)."""
    finite = np.isfinite(uec_vals)
    if not finite.any():
        return float("nan"), float("nan"), float("nan")
    vals = uec_vals[finite]
    n = len(vals)
    half = n // 2
    m1 = float(np.nanmean(vals[:half]))
    m2 = float(np.nanmean(vals[half:]))
    return m1, m2, (m2 - m1)


# ------------------------------
# Backtest loop
# ------------------------------


def backtest(
    ts: np.ndarray,
    close: np.ndarray,
    r: np.ndarray,
    uec_z: np.ndarray,
    p_up: np.ndarray,
    edge: np.ndarray,
    size: np.ndarray,
    z_hi: float,
    z_lo: float,
    edge_th: float,
    fee_bps: float = 1.0,  # per unit turnover
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(ts)
    pos = np.zeros(n, dtype=float)
    pnl = np.zeros(n, dtype=float)
    equity = np.ones(n, dtype=float)
    # Simple trend-only state: enter when high regime + edge; exit otherwise
    for t in range(1, n):
        if np.isfinite(uec_z[t]) and abs(edge[t]) > edge_th and uec_z[t] > z_hi:
            pos[t] = math.copysign(abs(size[t]), 1.0 if p_up[t] > 0.5 else -1.0)
        else:
            pos[t] = 0.0
        # Transaction cost on turnover
        turnover = abs(pos[t] - pos[t - 1])
        fee = (fee_bps * 1e-4) * turnover
        ret = pos[t - 1] * r[t] - fee
        pnl[t] = ret
        equity[t] = equity[t - 1] * math.exp(ret)
    return pos, pnl, equity, (pnl.cumsum())


# ------------------------------
# CLI and main
# ------------------------------


def main():
    parser = argparse.ArgumentParser(description="BTC UEC regime analysis and simple backtest")
    parser.add_argument(
        "--csv",
        default="data/kaggle/btcusd_1min/btcusd_1-min_data.csv",
        help="Path to BTC 1-min CSV (Timestamp,Open,High,Low,Close,Volume)",
    )
    parser.add_argument("--tail", type=int, default=250_000, help="Load only the last N rows (approx)")
    parser.add_argument("--limit", type=int, default=None, help="Hard cap total rows read (from start)")
    parser.add_argument("--k_r", type=int, default=12, help="Return bins")
    parser.add_argument("--k_v", type=int, default=8, help="Volume bins")
    parser.add_argument("--order", type=int, default=3, help="KT Markov order R")
    parser.add_argument("--window", type=int, default=256, help="Sliding window length W (bars)")
    parser.add_argument(
        "--uec_method",
        choices=["counts", "kt"],
        default="counts",
        help="UEC estimator: fast smoothed-counts vs KT holonomy (slow)",
    )
    parser.add_argument("--kt_stride", type=int, default=8, help="Stride for KT windows to speed up")
    parser.add_argument("--z_hi", type=float, default=1.5, help="UEC z-score high threshold")
    parser.add_argument("--z_lo", type=float, default=0.0, help="UEC z-score low threshold")
    parser.add_argument("--edge_th", type=float, default=0.03, help="Min |edge| to act")
    parser.add_argument("--fee_bps", type=float, default=1.0, help="Fee per unit turnover (bps)")
    parser.add_argument("--out", default="results/btc_uec_analysis.csv", help="Output CSV path")
    parser.add_argument("--summary", default="results/btc_uec_summary.json", help="Summary JSON path")
    # CI and change-point options
    parser.add_argument("--ci_b", type=int, default=0, help="Bootstrap samples per CI (0=off)")
    parser.add_argument("--ci_block", type=int, default=0, help="Bootstrap block size (0=auto)")
    parser.add_argument("--ci_stride", type=int, default=16, help="Compute CI every N bars (interpolate)")
    parser.add_argument("--cp_window", type=int, default=512, help="Change-point local window for z-test")
    parser.add_argument("--cp_z", type=float, default=2.0, help="Z-threshold for change-point candidates")
    parser.add_argument("--cp_min_dist", type=int, default=512, help="Minimum distance between change-points")
    args = parser.parse_args()

    ts, o, h, l, c, v = load_btc_csv(args.csv, tail=args.tail, limit=args.limit)
    n = len(ts)
    if n < args.window + 2:
        raise SystemExit(f"Not enough rows loaded: {n} < window {args.window}")

    r, body, wick, rv, skew, v_z = feature_block(ts, o, h, l, c, v)

    sym, edges_r, edges_v = symbolize_joint(r, v, args.k_r, args.k_v)
    k = int(args.k_r * args.k_v)

    print(f"Loaded {n} bars. k={k} (r x v = {args.k_r} x {args.k_v}), W={args.window}, method={args.uec_method}")

    # Core assumption checks (fast)
    contexts = float(k) ** float(args.order)
    n_train = float(args.window)
    n_eval = float(args.window)
    warn_train = n_train < 50.0 * contexts
    warn_eval = n_eval < 10.0 * contexts
    if warn_train or warn_eval:
        print(
            f"[Sufficiency] k^R={contexts:.3g}, train≈{n_train:.0f}, eval≈{n_eval:.0f} → "
            f"{'train ' if warn_train else ''}{'eval ' if warn_eval else ''}may be data‑starved."
        )

    ok_loop, msg_loop = check_loop_closure(sym, k, args.window)
    print(f"[Loop] {msg_loop}")

    bad, total = check_support_overlap(sym, k, args.window, samples=10)
    if total > 0:
        ratio = bad / total
        if ratio > 0.0:
            print(
                f"[Support] Asymmetric transitions in samples: {bad}/{total} ({ratio:.3%}). Consider smoothing/coarsening."
            )

    uec_vals, uec_z = compute_uec_stream(
        sym, k=k, W=args.window, R=args.order, method=args.uec_method, kt_stride=args.kt_stride
    )

    m1, m2, d = check_stationarity_split(uec_vals)
    if np.isfinite(d):
        print(f"[Stationarity] mean(1st half)={m1:.4g}, mean(2nd)={m2:.4g}, delta={d:.4g}")

    # Optional bootstrap CIs (counts only)
    uec_lo = np.full_like(uec_vals, np.nan)
    uec_hi = np.full_like(uec_vals, np.nan)
    if args.ci_b and args.uec_method == "counts":
        for t in range(args.window, n - 1, max(1, args.ci_stride)):
            lo, hi = bootstrap_ci_counts_for_window(sym, t, args.window, k, args.ci_b, args.ci_block)
            uec_lo[t] = lo
            uec_hi[t] = hi
        # interpolate CI across bars
        idx = np.arange(n)
        for arr in (uec_lo, uec_hi):
            mask = np.isfinite(arr)
            if mask.any():
                arr[:] = np.interp(idx, idx[mask], arr[mask])

    # Change-point detection on uec_z via rolling first-difference z-score
    cps: List[int] = []
    try:
        win = max(8, args.cp_window)
        zth = float(args.cp_z)
        mind = int(args.cp_min_dist)
        dz = np.zeros_like(uec_z)
        dz[1:] = np.abs(uec_z[1:] - uec_z[:-1])
        mz, sz = ewm_mean_std(dz, span=win)
        z = (dz - mz) / np.where(sz > 1e-12, sz, 1.0)
        cand = np.where(z > zth)[0]
        last = -10**9
        for t in cand:
            if t - last >= mind:
                cps.append(int(t))
                last = t
    except Exception:
        cps = []

    # Direction and sizing
    p_up, edge, exp_move = direction_proxy(r, v_z, body)
    # Realized vol proxy per bar
    realized_vol = np.sqrt(ewm(r ** 2, span=64))

    # Target per-bar vol from daily target (2% default hypothesis -> scaled by sqrt(time))
    # Infer bar seconds from median delta
    dt_sec = float(np.median(np.diff(ts))) if n > 2 else 60.0
    bars_per_day = max(1.0, 86400.0 / max(dt_sec, 1.0))
    target_vol_day = 0.02
    target_vol_per_bar = target_vol_day / math.sqrt(bars_per_day)

    size = position_size(edge, exp_move, realized_vol, uec_z, target_vol_per_bar)

    # Backtest
    pos, pnl, equity, pnl_cum = backtest(
        ts, c, r, uec_z, p_up, edge, size, args.z_hi, args.z_lo, args.edge_th, fee_bps=args.fee_bps
    )

    # Summary stats
    valid = np.isfinite(pnl)
    ret = pnl[valid]
    mean = float(np.mean(ret))
    std = float(np.std(ret) + 1e-12)
    sharpe = mean / std * math.sqrt(bars_per_day)
    mdd = 0.0
    peak = -1e9
    eq = equity[valid]
    peak = eq[0]
    for x in eq:
        peak = max(peak, x)
        mdd = max(mdd, (peak - x) / max(peak, 1e-12))

    print(
        f"Summary: bars={n} mean_ret={mean:.3e}/bar std={std:.3e} Sharpe≈{sharpe:.2f} MaxDD≈{mdd:.1%} FinalEq={eq[-1]:.3f}"
    )

    # Persist per-bar outputs
    ensure_dir(os.path.dirname(args.out) or ".")
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "close",
                "r",
                "volume",
                "uec_bits",
                "uec_z",
                "uec_ci_lo",
                "uec_ci_hi",
                "p_up",
                "edge",
                "size",
                "pos",
                "pnl",
                "equity",
            ]
        )
        for i in range(n):
            w.writerow(
                [
                    ts[i],
                    c[i],
                    r[i],
                    v[i],
                    uec_vals[i] if i < len(uec_vals) else float("nan"),
                    uec_z[i] if i < len(uec_z) else float("nan"),
                    uec_lo[i] if i < len(uec_lo) else float("nan"),
                    uec_hi[i] if i < len(uec_hi) else float("nan"),
                    p_up[i],
                    edge[i],
                    size[i],
                    pos[i],
                    pnl[i],
                    equity[i],
                ]
            )

    print(f"Saved per-bar analysis to {args.out}")
    print("Hints: Try --uec_method kt --kt_stride 16 on ~50k bars for fidelity checks.")

    # Summary JSON
    import json
    uec_median = float(np.nanmedian(uec_vals))
    summary = {
        "bars": int(n),
        "k": int(k),
        "W": int(args.window),
        "order": int(args.order),
        "method": args.uec_method,
        "uec_median": uec_median,
        "mean_ret": mean,
        "std_ret": std,
        "sharpe_daily": sharpe,
        "max_drawdown": mdd,
        "final_equity": float(eq[-1]),
        "stationarity_mean1": m1,
        "stationarity_mean2": m2,
        "stationarity_delta": d,
        "change_points": cps,
    }
    ensure_dir(os.path.dirname(args.summary) or ".")
    with open(args.summary, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()
