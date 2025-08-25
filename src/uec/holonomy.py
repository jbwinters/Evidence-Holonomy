from __future__ import annotations
from typing import Sequence
import numpy as np
from .coders import KTMarkovMixture
from .transforms import TransitionDecodeTakeSecond, TransitionEncode, TimeReverse, apply_loop


def klrate_between_sequences(p_seq: Sequence[int], q_seq: Sequence[int], k: int, R: int = 3) -> float:
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
    loop,
    k: int,
    R: int = 3,
    align: str = "tail",
) -> float:
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
    E = TransitionEncode(k)
    Rv = TimeReverse()
    D2 = TransitionDecodeTakeSecond(k)
    q_seq, _ = apply_loop(seq, list(range(k)), [E, Rv, D2])
    p_eval = list(seq)[1:]
    return klrate_between_sequences(p_eval, q_seq, k, R=R)

