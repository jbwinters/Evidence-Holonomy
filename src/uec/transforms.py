"""
Representation transforms and loop machinery.

These pure transforms act on integer-valued sequences and alphabets:
- Permute: bijective recoding of symbols by a permutation (gauge transform).
- MergeSymbols: coarse-grain by mapping many symbols to fewer.
- TimeReverse: reverse the sequence order.
- TransitionEncode/Decode: map a state sequence to transitions and back.
- Downsample/UpsampleRepeat: simple temporal scaling transforms.

Loops are composed by apply_loop(), returning the transformed sequence and
its final alphabet. Holonomy estimators build on these to define loops.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple


class Transform:
    def apply(self, seq: Sequence[int], alphabet: Sequence[int]) -> Tuple[List[int], List[int]]:
        raise NotImplementedError


class Permute(Transform):
    def __init__(self, perm: Sequence[int]):
        self.perm = list(map(int, perm))

    def apply(self, seq, alphabet):
        mapped = [self.perm[int(s)] for s in seq]
        return mapped, list(alphabet)


class MergeSymbols(Transform):
    def __init__(self, mapping: Dict[int, int], new_k: int | None = None):
        self.mapping = {int(k): int(v) for k, v in mapping.items()}
        self.new_k = int(new_k) if new_k is not None else (max(self.mapping.values()) + 1)

    def apply(self, seq, alphabet):
        mapped = [self.mapping[int(s)] for s in seq]
        return mapped, list(range(self.new_k))


class TimeReverse(Transform):
    def apply(self, seq, alphabet):
        return list(reversed(seq)), list(alphabet)


class TransitionEncode(Transform):
    def __init__(self, k: int):
        self.k = int(k)

    def apply(self, seq, alphabet):
        x = list(seq)
        y = []
        for t in range(0, len(x) - 1):
            y.append(int(x[t]) * self.k + int(x[t + 1]))
        return y, list(range(self.k * self.k))


class TransitionDecodeTakeSecond(Transform):
    def __init__(self, k: int):
        self.k = int(k)

    def apply(self, seq, alphabet):
        out = []
        for z in seq:
            j = int(z % self.k)
            out.append(j)
        return out, list(range(self.k))


class TransitionEncodeLag(Transform):
    def __init__(self, k: int, tau: int = 1):
        self.k = int(k)
        self.tau = int(tau)

    def apply(self, seq, alphabet):
        x = list(seq)
        y = []
        n = len(x)
        if self.tau <= 0:
            return [], list(range(self.k * self.k))
        for t in range(0, max(0, n - self.tau)):
            y.append(int(x[t]) * self.k + int(x[t + self.tau]))
        return y, list(range(self.k * self.k))


class Downsample(Transform):
    def __init__(self, step: int = 2):
        self.step = int(step)

    def apply(self, seq, alphabet):
        return list(seq)[:: self.step], list(alphabet)


class UpsampleRepeat(Transform):
    def __init__(self, step: int = 2):
        self.step = int(step)

    def apply(self, seq, alphabet):
        out: List[int] = []
        for s in seq:
            out.extend([int(s)] * self.step)
        return out, list(alphabet)


def apply_loop(seq: Sequence[int], alphabet: Sequence[int], transforms: List[Transform]) -> Tuple[List[int], List[int]]:
    """Apply a list of transforms in order, returning (sequence, alphabet)."""
    s, a = list(seq), list(alphabet)
    for T in transforms:
        s, a = T.apply(s, a)
    return s, a
