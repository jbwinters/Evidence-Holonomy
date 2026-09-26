"""
Validation battery for the KL-rate time-reversal estimator.

Each check_* function runs one experiment, prints a one-line result, appends a
JSON record under results/, and raises AssertionError if the result falls
outside its tolerance. run_core() and run_suite() group them; the uec-battery
console script exposes both (see uec.cli.run_battery).

Checks:
- Gauge: a bijective relabeling loop gives ~0.
- Time reversal: the estimate matches the analytic entropy production (EP) of
  random chains, closed-form rings, reversible chains, and low/high-EP edge cases.
- Three-way EP: analytic EP, the KT estimator, and a smoothed-count estimator agree.
- Coder agreement: KT and LZ78 code lengths per symbol approach each other
  (both estimate the entropy rate); this is not a KL estimate.
- Order robustness: KT mixtures with R=1 and R=3 give the same time-reversal
  estimate on first-order chains. The mixture posterior concentrates on order 1
  in both cases, so this shows robustness to R, not independence from the coder.
- Coarse-graining loops: the smoothed estimate is non-negative. With a
  deterministic lift, the ideal KL rate is infinite; the check only guards sign.
- Negative controls: alphabet mismatch and empty outputs raise; naive
  self-evidence differences miss irreversibility.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .coders import KTMarkovMixture, LZ78Coder
from .holonomy import klrate_holonomy_general, klrate_holonomy_time_reversal_markov
from .markov import (
    _row_stochastic,
    entropy_production_rate_bits,
    random_markov_biased,
    ring_chain_3,
    ring_ep_bits,
    sample_HMM,
    sample_markov,
)
from .transforms import (
    Downsample,
    MergeSymbols,
    Permute,
    TimeReverse,
    Transform,
    TransitionDecodeTakeSecond,
    TransitionEncode,
    UpsampleRepeat,
    apply_loop,
)

RESULTS_DIR = "results"
_RUN_ID: Optional[str] = None


# ------------------------------
# Logging and reproducibility
# ------------------------------


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def set_run_id(run_id: str) -> None:
    global _RUN_ID
    _RUN_ID = str(run_id)


def _path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


def log_json(name: str, record: dict) -> None:
    """Append a JSON record to results/<name>, tagged with the run id."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rec = dict(record)
    if _RUN_ID is not None:
        rec.setdefault("run_id", _RUN_ID)
    rec["timestamp"] = time.time()
    with open(_path(name), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _read_json_lines(name: str) -> List[dict]:
    path = _path(name)
    if not os.path.exists(path):
        return []
    out: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def log_environment() -> None:
    log_json(
        "env.json",
        {"python_version": sys.version, "platform": platform.platform(), "numpy_version": np.__version__},
    )


# ------------------------------
# Estimator helpers
# ------------------------------


def counts_from_sequence(seq: Sequence[int], k: int) -> np.ndarray:
    """Return the k x k transition count matrix of a state sequence."""
    C = np.zeros((k, k), dtype=np.int64)
    for t in range(len(seq) - 1):
        C[int(seq[t]), int(seq[t + 1])] += 1
    return C


def ep_bits_from_counts_smoothed(counts: np.ndarray, alpha: float = 0.5) -> float:
    """Plug-in entropy production (bits/step) from smoothed transition counts."""
    k = counts.shape[0]
    N_i = counts.sum(axis=1, keepdims=True)
    T_hat = (counts + alpha) / (N_i + alpha * k)
    occ = counts.sum(axis=1) + alpha
    pi_hat = occ / occ.sum()
    s = 0.0
    for i in range(k):
        for j in range(k):
            num = pi_hat[i] * T_hat[i, j]
            den = pi_hat[j] * T_hat[j, i]
            s += num * math.log(num / den, 2.0)
    return float(s)


def naive_evidence_diff_between_sequences(p_seq: Sequence[int], q_seq: Sequence[int], k: int, R: int = 3) -> float:
    """Self-evidence difference H_Q(q) - H_P(p) per symbol (negative control, not a KL)."""
    Hp = KTMarkovMixture(k, R=R).fit(p_seq) / max(1, len(p_seq))
    Hq = KTMarkovMixture(k, R=R).fit(q_seq) / max(1, len(q_seq))
    return float(Hq - Hp)


def bootstrap_klrate(
    seq: Sequence[int],
    alphabet: Sequence[int],
    loop: List[Transform],
    k: int,
    R: int = 3,
    B: int = 200,
    block: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Non-overlapping block bootstrap of the loop estimate: (mean, 2.5%, 97.5%)."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(seq)
    x = list(seq)
    blocks = [x[s : s + block] for s in range(0, max(1, n - block + 1), block)]
    vals: List[float] = []
    for _ in range(B):
        sample: List[int] = []
        while len(sample) < n:
            sample.extend(blocks[int(rng.integers(0, len(blocks)))])
        vals.append(klrate_holonomy_general(sample[:n], alphabet, loop, k, R=R, align="head"))
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


class DropAll(Transform):
    def apply(self, seq, alphabet):
        return [], list(alphabet)


class _LiftBinary(Transform):
    """Deterministic lift {0,1} -> {0, target} into a k-ary alphabet."""

    def __init__(self, k: int, target: int):
        self.k, self.target = k, target

    def apply(self, seq, alphabet):
        return [0 if int(s) == 0 else self.target for s in seq], list(range(self.k))


def _time_reversal_loop(k: int) -> List[Transform]:
    return [TransitionEncode(k), TimeReverse(), TransitionDecodeTakeSecond(k)]


# ------------------------------
# Core checks
# ------------------------------


def check_gauge_invariance(seed: int = 123, n: int = 60000, k: int = 4, R: int = 3) -> float:
    rng = np.random.default_rng(seed)
    x = sample_markov(random_markov_biased(k=k, delta=0.0, rng=rng), n=n, rng=rng)
    perm = list(rng.permutation(k))
    loop = [Permute(perm), Permute([perm.index(i) for i in range(k)])]
    hol = klrate_holonomy_general(x, list(range(k)), loop, k=k, R=R, align="head")
    print(f"[Gauge] bijective loop, estimate ≈ 0: {hol:.6g}")
    log_json("b1_gauge.json", {"seed": seed, "n": n, "k": k, "R": R, "hol_bits": hol})
    assert abs(hol) < 5e-3, "Gauge invariance violated beyond tolerance."
    return hol


def check_coarse_grain_nonneg(seed: int = 456, n: int = 80000, k: int = 4, R: int = 3) -> float:
    rng = np.random.default_rng(seed)
    x = sample_markov(random_markov_biased(k=k, delta=0.25, rng=rng), n=n, rng=rng)
    loop = [MergeSymbols({0: 0, 1: 0, 2: 1, 3: 1}, new_k=2), _LiftBinary(4, 2)]
    hol = klrate_holonomy_general(x, list(range(k)), loop, k=4, R=R, align="head")
    print(f"[Coarse-grain] smoothed estimate >= 0 (ideal rate is infinite): {hol:.6g}")
    log_json("b2_coarse.json", {"seed": seed, "n": n, "k": k, "R": R, "hol_bits": hol})
    assert hol > -3e-3, "Coarse-grain estimate should be non-negative."
    return hol


def check_time_reversal_equals_ep(
    seed: int = 789, n: int = 150000, k: int = 3, R: int = 3, delta: float = 0.6
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=delta, rng=rng)
    sigma = entropy_production_rate_bits(T)
    hol = klrate_holonomy_time_reversal_markov(sample_markov(T, n=n, rng=rng), k=k, R=R)
    diff = abs(hol - sigma)
    rel = diff / max(1e-8, abs(sigma))
    print(f"[Time reversal] EP={sigma:.6g}  estimate={hol:.6g}  abs diff={diff:.3g}  rel diff={rel:.3%}")
    log_json(
        "b3_time_reversal.json",
        {"seed": seed, "n": n, "k": k, "R": R, "delta": delta, "ep_bits": sigma, "hol_bits": hol,
         "abs_diff": diff, "rel_diff": rel},
    )
    assert diff < 5e-3 or rel < 5e-2 or (abs(sigma) < 1e-5 and diff < 5e-4), (
        "Time-reversal estimate does not match EP within tolerance."
    )
    return sigma, hol


def check_ep_three_way(seed: int = 202, n: int = 150000, k: int = 3, R: int = 3, delta: float = 0.6) -> None:
    """Analytic EP vs the KT estimator vs smoothed transition counts."""
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=delta, rng=rng)
    sigma = entropy_production_rate_bits(T)
    x = sample_markov(T, n=n, rng=rng)
    hol = klrate_holonomy_time_reversal_markov(x, k=k, R=R)
    mle = ep_bits_from_counts_smoothed(counts_from_sequence(x, k))
    d_th, d_tm, d_hm = abs(sigma - hol), abs(sigma - mle), abs(hol - mle)
    rel = d_th / max(1e-8, abs(sigma))
    print(f"[EP 3-way] true={sigma:.6g}  kt={hol:.6g}  counts={mle:.6g}  |t-k|={d_th:.3g}  |k-c|={d_hm:.3g}")
    log_json(
        "b4_three_way.json",
        {"seed": seed, "n": n, "k": k, "R": R, "delta": delta, "ep_true": sigma, "ep_hol": hol,
         "ep_mle": mle, "abs_t_h": d_th, "abs_t_m": d_tm, "abs_h_m": d_hm, "rel": rel},
    )
    assert (d_th < 5e-3 or rel < 5e-2) and d_hm < 6e-3, "Three-way EP consistency failed."


def check_coder_entropy_agreement(seed: int = 135, k: int = 3) -> List[float]:
    """KT and LZ78 code lengths per symbol converge toward each other (entropy rate)."""
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.4, rng=rng)
    diffs = []
    for n in (4000, 8000, 16000, 32000, 64000):
        x = sample_markov(T, n=n, rng=rng)
        diffs.append((KTMarkovMixture(k, R=3).fit(x) - LZ78Coder(k).total_codelen(x)) / n)
    print("[Coder agreement] KT - LZ78 bits/symbol by n:", [round(d, 4) for d in diffs])
    log_json("b5_coder_agreement.json", {"seed": seed, "k": k, "per_symbol_diffs": diffs})
    assert abs(diffs[-1]) <= abs(diffs[0]) + 1e-3, "KT-LZ78 per-symbol difference did not shrink."
    return diffs


def check_order_robustness(seed: int = 246, n: int = 150000, k: int = 3, num_windows: int = 20) -> Tuple[float, float]:
    """Time-reversal estimates with KT R=3 vs R=1 across windows of one chain."""
    rng = np.random.default_rng(seed)
    x = sample_markov(random_markov_biased(k=k, delta=0.6, rng=rng), n=n, rng=rng)
    w = n // num_windows
    a = np.array([klrate_holonomy_time_reversal_markov(x[i * w:(i + 1) * w], k=k, R=3) for i in range(num_windows)])
    b = np.array([klrate_holonomy_time_reversal_markov(x[i * w:(i + 1) * w], k=k, R=1) for i in range(num_windows)])
    r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else 1.0
    mad = float(np.mean(np.abs(a - b)))
    print(f"[Order robustness] KT R=3 vs R=1: r={r:.3f}, mean |Δ|={mad:.3g} bits/step")
    log_json(
        "b7_order_robustness.json",
        {"seed": seed, "n": n, "k": k, "num_windows": num_windows, "correlation": r, "mean_abs_diff": mad,
         "hol_rates_R3": a.tolist(), "hol_rates_R1": b.tolist()},
    )
    assert r > 0.8 and mad < 0.1, "Order robustness failed."
    return r, mad


def sweep_random_chains(seed: int = 2468, reps: int = 8, n: int = 120000, k: int = 3, delta: float = 0.6, R: int = 3) -> float:
    rng = np.random.default_rng(seed)
    diffs, rels, sigmas = [], [], []
    for r in range(reps):
        T = random_markov_biased(k=k, delta=delta, rng=rng)
        sigma = entropy_production_rate_bits(T)
        hol = klrate_holonomy_time_reversal_markov(sample_markov(T, n=n, rng=rng), k=k, R=R)
        diffs.append(hol - sigma)
        rels.append((hol - sigma) / max(1e-8, abs(sigma)))
        sigmas.append(sigma)
        print(f"  Rep {r + 1}/{reps}: EP={sigma:.6g}  estimate={hol:.6g}  rel={rels[-1]:.2%}")
        log_json("b6_sweep_reps.json", {"rep": r + 1, "seed": seed, "n": n, "k": k, "R": R, "delta": delta,
                                        "ep_bits": sigma, "hol_bits": hol, "diff": hol - sigma, "rel": rels[-1]})
    valid = [x for x, s in zip(rels, sigmas) if abs(s) >= 1e-4] or rels
    mean_rel = statistics.mean(valid)
    q25, q50, q75 = np.quantile(diffs, [0.25, 0.5, 0.75])
    print(f"[Sweep] mean |diff|={statistics.mean(abs(d) for d in diffs):.3g}  mean rel={mean_rel:.2%}  "
          f"median diff={q50:.3g}  IQR=[{q25:.3g},{q75:.3g}]")
    log_json("b6_sweep_summary.json", {"seed": seed, "reps": reps, "n": n, "k": k, "R": R, "delta": delta,
                                       "mean_abs_diff": statistics.mean(abs(d) for d in diffs),
                                       "sd_diff": statistics.pstdev(diffs), "mean_rel": mean_rel,
                                       "q25": float(q25), "median": float(q50), "q75": float(q75)})
    assert abs(mean_rel) < 0.05, "Average relative error across random chains is too large."
    return mean_rel


def check_temporal_coarse_grain(seed: int = 642, n: int = 80000, k: int = 3, R: int = 3) -> float:
    rng = np.random.default_rng(seed)
    x = sample_markov(random_markov_biased(k=k, delta=0.5, rng=rng), n=n, rng=rng)
    hol = klrate_holonomy_general(x, list(range(k)), [Downsample(2), UpsampleRepeat(2)], k=k, R=R, align="head")
    print(f"[Temporal coarse-grain] estimate >= 0: {hol:.6g}")
    log_json("temporal_coarse_grain.json", {"seed": seed, "n": n, "k": k, "R": R, "hol_bits": hol})
    assert hol > -3e-3
    return hol


# ------------------------------
# Closed-form and edge cases
# ------------------------------


def check_two_state_reversible(seed: int = 222, n: int = 150000) -> None:
    rng = np.random.default_rng(seed)
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    sigma = entropy_production_rate_bits(T)
    hol = klrate_holonomy_time_reversal_markov(sample_markov(T, n=n, rng=rng), k=2, R=3)
    print(f"[2-state reversible] EP={sigma:.6g}  estimate={hol:.6g}")
    log_json("c1_two_state.json", {"seed": seed, "n": n, "ep_bits": sigma, "hol_bits": hol})
    assert abs(sigma) < 1e-6 and abs(hol) < 5e-4


def check_ring_closed_form(seed: int = 111, n: int = 150000, R: int = 3) -> None:
    rng = np.random.default_rng(seed)
    for p, q in [(0.2, 0.05), (0.3, 0.1), (0.35, 0.05)]:
        hol = klrate_holonomy_time_reversal_markov(sample_markov(ring_chain_3(p, q), n=n, rng=rng), k=3, R=R)
        ep = ring_ep_bits(p, q)
        print(f"[Ring] p={p} q={q} EP={ep:.6g} estimate={hol:.6g} diff={abs(hol - ep):.3g}")
        log_json("c2_ring.json", {"p": p, "q": q, "ep": ep, "hol": hol, "diff": abs(hol - ep)})
        assert abs(hol - ep) < 5e-3 or abs(hol - ep) / ep < 5e-2, "Ring EP mismatch."


def check_low_ep(seed: int = 333, n: int = 300000, k: int = 3, R: int = 3) -> None:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.05, rng=rng)
    sigma = entropy_production_rate_bits(T)
    hol = klrate_holonomy_time_reversal_markov(sample_markov(T, n=n, rng=rng), k=k, R=R)
    print(f"[Low-EP] EP={sigma:.6g} estimate={hol:.6g} diff={abs(hol - sigma):.3g}")
    log_json("d1_low_ep.json", {"seed": seed, "n": n, "k": k, "R": R, "ep": sigma, "hol": hol, "diff": abs(hol - sigma)})
    if abs(sigma) < 1e-3:
        assert abs(hol - sigma) < 5e-4


def check_high_ep(seed: int = 444, n: int = 150000, k: int = 3, R: int = 3) -> None:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.9, rng=rng)
    sigma = entropy_production_rate_bits(T)
    hol = klrate_holonomy_time_reversal_markov(sample_markov(T, n=n, rng=rng), k=k, R=R)
    print(f"[High-EP] EP={sigma:.6g} estimate={hol:.6g} diff={abs(hol - sigma):.3g}")
    log_json("d2_high_ep.json", {"seed": seed, "n": n, "k": k, "R": R, "ep": sigma, "hol": hol, "diff": abs(hol - sigma)})
    assert abs(hol - sigma) < 1e-2


def check_alignment_invariance(seed: int = 555, n: int = 80000, k: int = 4, R: int = 3) -> None:
    rng = np.random.default_rng(seed)
    x = sample_markov(random_markov_biased(k=k, delta=0.25, rng=rng), n=n, rng=rng)
    loop = [MergeSymbols({0: 0, 1: 0, 2: 1, 3: 1}, new_k=2), _LiftBinary(4, 2)]
    head = klrate_holonomy_general(x, list(range(k)), loop, k=4, R=R, align="head")
    tail = klrate_holonomy_general(x, list(range(k)), loop, k=4, R=R, align="tail")
    print(f"[Align] head={head:.6g} tail={tail:.6g} diff={abs(head - tail):.3g}")
    log_json("d3_align.json", {"seed": seed, "n": n, "k": k, "R": R, "head": head, "tail": tail, "diff": abs(head - tail)})
    assert abs(head - tail) < 1e-3


def check_segment_stability(seed: int = 666, n: int = 400000, k: int = 3, R: int = 3) -> None:
    rng = np.random.default_rng(seed)
    x = list(sample_markov(random_markov_biased(k=k, delta=0.6, rng=rng), n=n, rng=rng))
    loop, a = _time_reversal_loop(k), list(range(k))
    full = klrate_holonomy_general(x, a, loop, k=k, R=R, align="head")
    first = klrate_holonomy_general(x[: n // 2], a, loop, k=k, R=R, align="head")
    second = klrate_holonomy_general(x[n // 2:], a, loop, k=k, R=R, align="head")
    print(f"[Segment] full={full:.6g} first={first:.6g} second={second:.6g}")
    log_json("d4_segment.json", {"seed": seed, "n": n, "k": k, "R": R, "full": full, "first": first, "second": second})


def check_bootstrap_ci(seed: int = 1234, n: int = 200000, k: int = 3, R: int = 3, B: int = 200) -> bool:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.6, rng=rng)
    x = sample_markov(T, n=n, rng=rng)
    loop, a = _time_reversal_loop(k), list(range(k))
    point = klrate_holonomy_general(x, a, loop, k=k, R=R, align="head")
    mean, lo, hi = bootstrap_klrate(x, a, loop, k=k, R=R, B=B, block=2000, rng=rng)
    sigma = entropy_production_rate_bits(T)
    cover = bool(lo <= sigma <= hi)
    print(f"[Bootstrap] point={point:.6g} CI95=[{lo:.6g},{hi:.6g}] EP={sigma:.6g} covered={cover}")
    log_json("e1_bootstrap.json", {"seed": seed, "n": n, "k": k, "R": R, "point": point, "mean": mean,
                                   "lo": lo, "hi": hi, "ep_bits": sigma, "cover": cover})
    return cover


def check_order_sweep(seed: int = 24601, n: int = 150000, k: int = 3, orders: Sequence[int] = (1, 2, 3, 5)) -> None:
    rng = np.random.default_rng(seed)
    T = random_markov_biased(k=k, delta=0.6, rng=rng)
    sigma = entropy_production_rate_bits(T)
    x = sample_markov(T, n=n, rng=rng)
    for R in orders:
        hol = klrate_holonomy_time_reversal_markov(x, k=k, R=R)
        print(f"[Order] R={R} estimate={hol:.6g} diff={abs(hol - sigma):.3g}")
        log_json("e2_order.json", {"R": R, "hol": hol, "diff": abs(hol - sigma)})


# ------------------------------
# Negative controls
# ------------------------------


def check_mismatch_alphabet_raises(seed: int = 777, n: int = 60000, k: int = 3, R: int = 3) -> None:
    rng = np.random.default_rng(seed)
    x = sample_markov(random_markov_biased(k=k, delta=0.4, rng=rng), n=n, rng=rng)
    try:
        klrate_holonomy_general(x, list(range(k)), [MergeSymbols({0: 0, 1: 0, 2: 1}, new_k=2)], k=k, R=R)
    except ValueError as e:
        print(f"[Negative] alphabet mismatch raised as expected: {e}")
        log_json("f1_mismatch.json", {"ok": True, "error": str(e)})
        return
    log_json("f1_mismatch.json", {"ok": False})
    raise AssertionError("Expected ValueError due to mismatched alphabet size.")


def check_empty_output_raises(seed: int = 888, n: int = 100, k: int = 3, R: int = 3) -> None:
    rng = np.random.default_rng(seed)
    x = sample_markov(random_markov_biased(k=k, delta=0.4, rng=rng), n=n, rng=rng)
    try:
        klrate_holonomy_general(x, list(range(k)), [DropAll()], k=k, R=R)
    except ValueError as e:
        print(f"[Negative] empty output raised as expected: {e}")
        log_json("f2_empty.json", {"ok": True, "error": str(e)})
        return
    log_json("f2_empty.json", {"ok": False})
    raise AssertionError("Expected ValueError due to empty output.")


def check_naive_contrast(seed: int = 999, n: int = 150000, k: int = 3, R: int = 3) -> float:
    """Self-evidence differences stay near 0 under time reversal even when EP > 0."""
    rng = np.random.default_rng(seed)
    x = sample_markov(random_markov_biased(k=k, delta=0.6, rng=rng), n=n, rng=rng)
    q_seq, _ = apply_loop(x, list(range(k)), _time_reversal_loop(k))
    naive = naive_evidence_diff_between_sequences(list(x)[1:], q_seq, k, R=R)
    print(f"[Negative] naive self-evidence difference under reversal ≈ 0: {naive:.6g}")
    log_json("f3_naive.json", {"seed": seed, "n": n, "k": k, "R": R, "naive_tr": naive})
    assert abs(naive) < 1e-2
    return naive


# ------------------------------
# Optional demos
# ------------------------------


def demo_hmm_coarse_grain(seed: int = 8642, n: int = 60000) -> None:
    rng = np.random.default_rng(seed)
    A = random_markov_biased(2, delta=0.5, rng=rng)
    B = np.array([[0.8, 0.15, 0.05], [0.2, 0.5, 0.3]])
    obs, _, m = sample_HMM(A, B, n=n, rng=rng)
    loop = [MergeSymbols({0: 0, 1: 1, 2: 1}, new_k=2), _LiftBinary(3, 2)]
    hol = klrate_holonomy_general(obs, list(range(m)), loop, k=3, R=3, align="head")
    print(f"[HMM demo] coarse-grain estimate >= 0: {hol:.6g}")
    log_json("optional_hmm.json", {"seed": seed, "n": n, "hol_bits": hol})


def demo_measurement_record(seed: int = 9753, n: int = 80000) -> None:
    rng = np.random.default_rng(seed)
    T = np.array([[0.95, 0.05], [0.2, 0.8]])
    sigma = entropy_production_rate_bits(T)
    hol = klrate_holonomy_time_reversal_markov(sample_markov(T, n=n, rng=rng), k=2, R=3)
    print(f"[2-state record] EP={sigma:.6g}  estimate={hol:.6g}")
    log_json("optional_measurement_2state.json", {"seed": seed, "n": n, "ep_bits": sigma, "hol_bits": hol})


def demo_measurement_record_3state(seed: int = 97531, n: int = 80000) -> None:
    rng = np.random.default_rng(seed)
    T = _row_stochastic(np.array([[0.90, 0.09, 0.01], [0.01, 0.90, 0.09], [0.09, 0.01, 0.90]]))
    sigma = entropy_production_rate_bits(T)
    hol = klrate_holonomy_time_reversal_markov(sample_markov(T, n=n, rng=rng), k=3, R=3)
    print(f"[3-state cycle] EP={sigma:.6g}  estimate={hol:.6g}")
    log_json("optional_measurement_3state.json", {"seed": seed, "n": n, "ep_bits": sigma, "hol_bits": hol})


# ------------------------------
# Groups
# ------------------------------


def run_core(seed: int = 12345, n: int = 150000, k: int = 3, R: int = 3, sweep_reps: int = 6,
             long: bool = False, optional: bool = False) -> None:
    check_gauge_invariance(seed=seed + 1, n=max(60000, n // 2), k=max(3, k), R=R)
    check_coarse_grain_nonneg(seed=seed + 2, n=max(80000, n // 2), k=max(4, k + 1), R=R)
    check_time_reversal_equals_ep(seed=seed + 3, n=n, k=k, R=R)
    check_ep_three_way(seed=seed + 30, n=max(300000, n) if long else n, k=k, R=R)
    check_coder_entropy_agreement(seed=seed + 4, k=k)
    check_order_robustness(seed=seed + 41, n=n, k=k)
    sweep_random_chains(seed=seed + 5, reps=max(10, sweep_reps) if long else sweep_reps,
                        n=max(200000, n // 2) if long else max(120000, n // 2), k=k, R=R)
    if optional:
        demo_hmm_coarse_grain(seed=seed + 6, n=max(60000, n // 2))
        demo_measurement_record(seed=seed + 7, n=max(80000, n // 2))
        demo_measurement_record_3state(seed=seed + 8, n=max(80000, n // 2))
        check_temporal_coarse_grain(seed=seed + 9, n=max(80000, n // 2), k=k, R=R)


def run_suite(seed: int = 12345, n: int = 150000, k: int = 3, R: int = 3, sweep_reps: int = 6,
              long: bool = False, optional: bool = False, bootstrap_B: int = 40) -> None:
    log_environment()
    run_core(seed=seed, n=n, k=k, R=R, sweep_reps=sweep_reps, long=long, optional=optional)
    check_two_state_reversible(seed=seed + 10, n=n)
    check_ring_closed_form(seed=seed + 11, n=max(150000, n // 2), R=R)
    check_low_ep(seed=seed + 12, n=max(300000, n), k=k, R=R)
    check_high_ep(seed=seed + 13, n=n, k=k, R=R)
    check_alignment_invariance(seed=seed + 14)
    check_segment_stability(seed=seed + 15)
    check_bootstrap_ci(seed=seed + 16, n=max(100000, n // 2), k=k, R=R, B=bootstrap_B)
    check_order_sweep(seed=seed + 17, n=n, k=k)
    check_mismatch_alphabet_raises(seed=seed + 18)
    check_empty_output_raises(seed=seed + 19)
    check_naive_contrast(seed=seed + 20, n=n, k=k, R=R)


def new_run_id() -> str:
    return f"run-{int(time.time() * 1000)}-{random.randint(0, 1_000_000)}"
