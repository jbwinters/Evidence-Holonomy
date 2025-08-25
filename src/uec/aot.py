"""
Arrow-of-Time (AoT) utilities and demos.

Provides:
- Quantile-based discretization for real-valued series.
- CSV/WAV loaders (SciPy optional; robust stdlib fallback for PCM WAVs).
- Signed likelihood-ratio scoring using KT frozen models (HQ-HP per window).
- AUC calculation (tie-aware) and windowed KL-rate with simple block bootstrap.

The AoT pipeline mirrors the time-reversal loop used by KL holonomy, so AUC
and bits/step align conceptually. Preprocessing flags (use_diff, use_logreturn)
help amplify irreversibility in practice for audio/sensors and finance.
"""

from __future__ import annotations
import numpy as np
from typing import List, Sequence, Tuple
from .coders import KTMarkovMixture, KTFrozenPredictor
from .holonomy import klrate_between_sequences, klrate_holonomy_time_reversal_markov
from .transforms import TransitionDecodeTakeSecond, TransitionEncode, TimeReverse, apply_loop


def quantile_bins(x: np.ndarray, k: int) -> np.ndarray:
    """Return monotone bin edges for k-quantile discretization of x."""
    qs = np.linspace(0.0, 1.0, k + 1)
    edges = np.quantile(x, qs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges


def discretize_series(x: np.ndarray, k: int) -> np.ndarray:
    """Discretize a real-valued array into k quantile bins (0..k-1)."""
    edges = quantile_bins(x, k)
    s = np.clip(np.digitize(x, edges[1:-1], right=False), 0, k - 1)
    return s.astype(int)


def load_csv_column(path: str, column: str | int = 0, skip_header: bool = True) -> np.ndarray:
    """Load a single numeric column from a CSV file (by name or index)."""
    if skip_header:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if isinstance(column, str):
            if column not in data.dtype.names:
                raise ValueError(f"Column '{column}' not found in CSV header.")
            x = np.asarray(data[column], dtype=float)
        else:
            cols = list(data.dtype.names)
            x = np.asarray(data[cols[int(column)]], dtype=float)
    else:
        arr = np.genfromtxt(path, delimiter=",", dtype=float)
        x = arr[:, int(column)] if arr.ndim == 2 else arr
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x


def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """Load mono audio and sampling rate from a WAV file.

    Tries SciPy's wavfile; falls back to robust stdlib parsing for 8/16/24/32-bit
    PCM, averaging channels and normalizing to unit variance.
    """
    try:
        from scipy.io import wavfile  # type: ignore

        sr, wav = wavfile.read(path)
        wav = np.asarray(wav)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        x = wav.astype(float)
        x = (x - x.mean()) / (x.std() + 1e-12)
        return x, int(sr)
    except Exception:
        import wave
        import struct

        with wave.open(path, "rb") as wf:
            nchan = wf.getnchannels()
            fr = wf.getframerate()
            nframes = wf.getnframes()
            sampwidth = wf.getsampwidth()  # bytes per sample
            raw = wf.readframes(nframes)

        # Convert raw PCM to int array depending on sample width
        if sampwidth == 1:
            buf = np.frombuffer(raw, dtype=np.uint8)
            data = buf.astype(np.int16) - 128  # center
        elif sampwidth == 2:
            data = np.frombuffer(raw, dtype='<i2').astype(np.int32)
        elif sampwidth == 3:
            # 24-bit little-endian
            buf = np.frombuffer(raw, dtype=np.uint8)
            if len(buf) % 3 != 0:
                buf = buf[: len(buf) - (len(buf) % 3)]
            a = buf.reshape(-1, 3)
            sign = (a[:, 2] & 0x80) != 0
            b = np.zeros((a.shape[0], 4), dtype=np.uint8)
            b[:, :3] = a
            b[:, 3] = np.where(sign, 0xFF, 0x00)
            data = b.view('<i4').ravel()
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype='<i4')
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        # Reshape by channels
        if nchan > 1:
            total_samples = (data.size // nchan) * nchan
            data = data[:total_samples]
            data = data.reshape(-1, nchan).mean(axis=1)
        vals = data.astype(float)
        x = (vals - vals.mean()) / (vals.std() + 1e-12)
        return x, int(fr)


def auc_from_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute AUC via Mann–Whitney U with average ranks for ties."""
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos, dtype=bool), np.zeros_like(neg, dtype=bool)])
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]
    ranks = np.empty_like(x_sorted, dtype=float)
    i, N = 0, len(x_sorted)
    while i < N:
        j = i + 1
        while j < N and x_sorted[j] == x_sorted[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + 1 + j)
        i = j
    R1 = ranks[y_sorted].sum()
    n1 = float(y.sum())
    n0 = float(N - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((R1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def window_iter(seq: Sequence[int], win: int, stride: int) -> List[List[int]]:
    """Return overlapping windows of integer sequence with given win/stride."""
    out: List[List[int]] = []
    n = len(seq)
    for start in range(0, max(1, n - win + 1), stride):
        out.append(list(seq[start : start + win]))
    return out


def train_forward_and_reverse_models(
    train_seq: Sequence[int], k: int, R: int
) -> Tuple[KTFrozenPredictor, KTFrozenPredictor]:
    """Train KT models for forward P and looped-reverse Q on the training split.

    Returns frozen evaluators Pf and Qf so held-out windows can be scored as
    HQ-HP consistently with the KL holonomy time-reversal loop.
    """
    P = KTMarkovMixture(k, R=R)
    P.fit(train_seq)
    Pf = P.snapshot_frozen()
    E = TransitionEncode(k)
    Rv = TimeReverse()
    D2 = TransitionDecodeTakeSecond(k)
    q_train, _ = apply_loop(train_seq, list(range(k)), [E, Rv, D2])
    Q = KTMarkovMixture(k, R=R)
    Q.fit(q_train)
    Qf = Q.snapshot_frozen()
    return Pf, Qf


def signed_lr_score(seq: Sequence[int], Pf: KTFrozenPredictor, Qf: KTFrozenPredictor) -> float:
    """Return per-symbol signed LR score: H_Q(seq) - H_P(seq)."""
    n = max(1, len(seq))
    HQ = Qf.codelen_sequence(seq) / n
    HP = Pf.codelen_sequence(seq) / n
    return float(HQ - HP)


def aot_from_series(
    x_raw: np.ndarray,
    k: int = 8,
    R: int = 3,
    sr: int | None = None,
    train_frac: float = 0.5,
    win: int = 4096,
    stride: int = 2048,
    B: int = 200,
    block_wins: int = 10,
    rng: np.random.Generator | None = None,
    use_diff: bool = False,
    use_logreturn: bool = False,
) -> dict:
    """Compute AoT metrics for a real-valued series.

    - Discretizes the series (optionally after first-diff or log-returns).
    - Trains KT models on forward train split and looped-reverse of train.
    - Computes signed LR scores HQ-HP on held-out windows; reports AUC.
    - Estimates KL-rate holonomy on held-out test and a simple block-bootstrap CI.
    - If sr is provided, also reports bits/second (bits/step * sr).
    """
    if rng is None:
        rng = np.random.default_rng()
    x_arr = np.asarray(x_raw, dtype=float)
    if use_logreturn:
        x_arr = x_arr[np.isfinite(x_arr)]
        min_pos = np.nanmin(x_arr)
        shift = 0.0 if min_pos > 0 else (1e-12 - min_pos + 1e-12)
        lx = np.log(x_arr + shift)
        x_arr = np.diff(lx, prepend=lx[:1])
    elif use_diff:
        x_arr = np.diff(x_arr, prepend=x_arr[:1])
    s = discretize_series(x_arr, k)
    n = len(s)
    ntr = int(n * train_frac)
    train, test = s[:ntr], s[ntr:]
    if len(test) < win * 4:
        raise ValueError(
            f"Not enough test data for the chosen window size (len(test)={len(test)}, win={win})."
        )
    Pf, Qf = train_forward_and_reverse_models(train, k, R)
    wins_fwd = window_iter(test, win, stride)
    E_win, Rv_win, D2_win = TransitionEncode(k), TimeReverse(), TransitionDecodeTakeSecond(k)
    wins_q = [apply_loop(w, list(range(k)), [E_win, Rv_win, D2_win])[0] for w in wins_fwd]
    scores_fwd = np.array([signed_lr_score(w, Pf, Qf) for w in wins_fwd], dtype=float)
    scores_rev = np.array([signed_lr_score(wq, Pf, Qf) for wq in wins_q], dtype=float)
    auc = auc_from_scores(scores_fwd, scores_rev)
    hol_rate = klrate_holonomy_time_reversal_markov(test, k=k, R=R)
    vals: List[float] = []
    for w in wins_fwd:
        qw, _ = apply_loop(w, list(range(k)), [E_win, Rv_win, D2_win])
        if len(w) > 1 and len(qw) > 0:
            vals.append(klrate_between_sequences(w[1:], qw, k, R))
    if len(vals) == 0:
        mean = hol_rate
        lo = hol_rate
        hi = hol_rate
    else:
        arr = np.array(vals, dtype=float)
        block = max(1, int(block_wins))
        blocks = [arr[i : i + block].mean() for i in range(0, len(arr) - block + 1, block)]
        if len(blocks) < 2:
            mean = float(np.mean(arr))
            lo = float(np.min(arr))
            hi = float(np.max(arr))
        else:
            boot: List[float] = []
            for _ in range(B):
                smp = [blocks[int(rng.integers(0, len(blocks)))] for __ in range(len(blocks))]
                boot.append(float(np.mean(smp)))
            mean = float(np.mean(boot))
            lo = float(np.percentile(boot, 2.5))
            hi = float(np.percentile(boot, 97.5))
    bits_per_step = float(hol_rate)
    bits_per_second = (bits_per_step * sr) if sr is not None else None
    return {
        "k": int(k),
        "R": int(R),
        "win": int(win),
        "stride": int(stride),
        "auc": float(auc),
        "scores_forward": scores_fwd.tolist(),
        "scores_reversed": scores_rev.tolist(),
        "bits_per_step": bits_per_step,
        "bits_per_second": (float(bits_per_second) if bits_per_second is not None else None),
        "hol_ci_mean": float(mean),
        "hol_ci_lo": float(lo),
        "hol_ci_hi": float(hi),
        "n_train": int(ntr),
        "n_test": int(len(test)),
    }
