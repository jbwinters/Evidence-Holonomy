"""
KL-rate holonomy estimators and the time-reversal loop.

Semantics:
- klrate_between_sequences: trains universal codes on P and Q sequences,
  then evaluates cross-entropy on P to estimate D(P||Q) per symbol.
- klrate_holonomy_general: applies a loop of transforms, aligns lengths,
  and computes KL-rate between original and looped sequences.
- klrate_holonomy_time_reversal_markov: time-reversal estimator. The loop
  encode→reverse→decode-second returns exactly reversed(seq[1:]); the result is
  the order-R plug-in estimate of D(P||P_rev), which converges to the entropy
  production rate for finite-state Markov chains when R >= 1.
- klrate_holonomy_leadlag_markov: the same estimator applied to the tau-step
  skeleton seq[o::tau], averaged over phases.

Note: both models are evaluated on the sequence they were trained on (in-sample
plug-in), so estimates carry a small positive bias of order k**(R+1)/n.
"""

from __future__ import annotations
from typing import Sequence
from .coders import KTMarkovMixture
from .transforms import (
    TransitionDecodeTakeSecond,
    TransitionEncode,
    TimeReverse,
    apply_loop,
    Transform,
)


def klrate_between_sequences(p_seq: Sequence[int], q_seq: Sequence[int], k: int, R: int = 3, coder: str = "kt") -> float:
    """Estimate the per-symbol KL divergence D(P||Q) using universal coding.
    This is an estimator for the (ideal) KL-holonomy rate defined in the paper.

    Trains coders on p_seq and q_seq separately; evaluates H_P(p) and H_Q(p) on
    the same p_seq using frozen predictors; returns H_Q(p)-H_P(p) in bits/symbol.
    """
    if coder == "kt":
        Pcoder = KTMarkovMixture(alphabet_size=k, R=R)
        Pcoder.fit(p_seq)
        Pfrozen = Pcoder.snapshot_frozen()
        Qcoder = KTMarkovMixture(alphabet_size=k, R=R)
        Qcoder.fit(q_seq)
        Qfrozen = Qcoder.snapshot_frozen()
        H_P = Pfrozen.codelen_sequence(p_seq) / max(1, len(p_seq))
        H_PQ = Qfrozen.codelen_sequence(p_seq) / max(1, len(p_seq))
        return float(H_PQ - H_P)
    elif coder == "lz78":
        from .coders import LZ78Coder
        # LZ78 baseline: compare compression lengths directly
        # NOTE: This is NOT a true KL-holonomy estimate! LZ78 is not universal
        # in the pointwise sense. This serves only as a sanity check baseline.
        P_len = LZ78Coder(k).total_codelen(p_seq) / max(1, len(p_seq))
        Q_len = LZ78Coder(k).total_codelen(q_seq) / max(1, len(q_seq))
        # Return difference in per-symbol compression
        return float(Q_len - P_len)
    else:
        raise ValueError("coder must be 'kt' or 'lz78'")


def klrate_holonomy_general(
    seq: Sequence[int],
    alphabet: Sequence[int],
    loop: Sequence[Transform],
    k: int,
    R: int = 3,
    align: str = "tail",
    coder: str = "kt",
) -> float:
    """KL-rate holonomy for a general loop returning to the original alphabet.

    - Applies loop transforms to seq, aligns p_eval to q_seq length by tail/head.
    - Computes KL-rate between original and looped sequences using KT.
    - Guards: final alphabet size must be k; loop output must be non-empty.
    """
    q_seq, aQ = apply_loop(seq, alphabet, loop)
    m = len(aQ)
    if m != k:
        raise ValueError(f"Final loop output alphabet size ({m}) must equal k ({k}); "
                        f"got final_alphabet={aQ[:10]}{'...' if len(aQ) > 10 else ''}")
    nQ = len(q_seq)
    if nQ <= 0:
        raise ValueError("Loop produced empty sequence; cannot evaluate KL-rate.")
    if align == "tail":
        p_eval = list(seq)[-nQ:]
    elif align == "head":
        p_eval = list(seq)[:nQ]
    elif align == "auto":
        # Choose alignment based on length change
        if nQ < len(seq):
            p_eval = list(seq)[-nQ:]  # tail for shortening
        elif nQ > len(seq):
            p_eval = list(seq)[:nQ]   # head for lengthening (pad with zeros if needed)
            if len(p_eval) < nQ:
                p_eval = p_eval + [0] * (nQ - len(p_eval))
        else:
            p_eval = list(seq)  # same length
    else:
        raise ValueError("align must be 'tail', 'head', or 'auto'")
    return klrate_between_sequences(p_eval, q_seq, k, R=R, coder=coder)


def klrate_holonomy_time_reversal_markov(seq: Sequence[int], k: int, R: int = 3, coder: str = "kt") -> float:
    """Time-reversal KL-rate estimate D(P||P_rev) in bits/step.

    Loop: TransitionEncode(k) → TimeReverse() → TransitionDecodeTakeSecond(k),
    which equals reversed(seq[1:]). p_eval = seq[1:] is scored against it.
    """
    E = TransitionEncode(k)
    Rv = TimeReverse()
    D2 = TransitionDecodeTakeSecond(k)
    q_seq, _ = apply_loop(seq, list(range(k)), [E, Rv, D2])
    p_eval = list(seq)[1:]
    return klrate_between_sequences(p_eval, q_seq, k, R=R, coder=coder)


def klrate_holonomy_leadlag_markov(
    seq: Sequence[int], k: int, tau: int = 1, R: int = 3, coder: str = "kt"
) -> float:
    """Time-reversal KL-rate at time scale tau (bits per tau-step).

    Estimates the irreversibility of the tau-step skeleton: for each phase
    o in 0..tau-1 the subsequence seq[o::tau] is scored with the time-reversal
    estimator, and the phase estimates are averaged (weighted by length).
    tau=1 reduces to klrate_holonomy_time_reversal_markov. Phases shorter than
    three symbols are skipped; returns 0.0 if none remain.
    """
    tau = int(tau)
    if tau <= 0:
        raise ValueError("tau must be a positive integer")
    x = list(seq)
    total, weight = 0.0, 0
    for o in range(tau):
        sub = x[o::tau]
        if len(sub) < 3:
            continue
        total += klrate_holonomy_time_reversal_markov(sub, k=k, R=R, coder=coder) * (len(sub) - 1)
        weight += len(sub) - 1
    return float(total / weight) if weight else 0.0
