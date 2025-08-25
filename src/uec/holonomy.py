"""
KL-rate holonomy estimators and the time-reversal loop.

Semantics:
- klrate_between_sequences: trains universal codes on P and Q sequences,
  then evaluates cross-entropy on P to estimate D(P||Q) per symbol.
- klrate_holonomy_general: applies a loop of transforms, aligns lengths,
  and computes KL-rate between original and looped sequences.
- klrate_holonomy_time_reversal_markov: canonical Markov loop (encode→reverse→
  decode-second) with length alignment to seq[1:], matching the EP identity.
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


def klrate_between_sequences(p_seq: Sequence[int], q_seq: Sequence[int], k: int, R: int = 3) -> float:
    """Estimate D(P||Q) by cross-entropy difference on the same evaluation sequence.

    Trains KT(k,R) on p_seq and q_seq separately; evaluates H_P(p) and H_Q(p) on
    the same p_seq using frozen predictors; returns H_Q(p)-H_P(p) in bits/symbol.
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
    loop: Sequence[Transform],
    k: int,
    R: int = 3,
    align: str = "tail",
) -> float:
    """KL-rate holonomy for a general loop returning to the original alphabet.

    - Applies loop transforms to seq, aligns p_eval to q_seq length by tail/head.
    - Computes KL-rate between original and looped sequences using KT.
    - Guards: final alphabet size must be k; loop output must be non-empty.
    """
    q_seq, aQ = apply_loop(seq, alphabet, loop)
    if len(aQ) != k:
        raise ValueError("Final loop output must be in the original alphabet.")
    nQ = len(q_seq)
    if nQ <= 0:
        raise ValueError("Loop produced empty sequence; cannot evaluate KL-rate.")
    if align == "tail":
        p_eval = list(seq)[-nQ:]
    elif align == "head":
        p_eval = list(seq)[:nQ]
    else:
        raise ValueError("align must be 'tail' or 'head'")
    return klrate_between_sequences(p_eval, q_seq, k, R=R)


def klrate_holonomy_time_reversal_markov(seq: Sequence[int], k: int, R: int = 3) -> float:
    """KL-rate holonomy for the canonical Markov time-reversal loop.

    Loop: TransitionEncode(k) → TimeReverse() → TransitionDecodeTakeSecond(k).
    Align p_eval = seq[1:] to the transition sequence length. Returns bits/step.
    """
    E = TransitionEncode(k)
    Rv = TimeReverse()
    D2 = TransitionDecodeTakeSecond(k)
    q_seq, _ = apply_loop(seq, list(range(k)), [E, Rv, D2])
    p_eval = list(seq)[1:]
    return klrate_between_sequences(p_eval, q_seq, k, R=R)
