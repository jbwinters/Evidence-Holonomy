from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
import numpy as np


class KTMarkovMixture:
    def __init__(self, alphabet_size: int, R: int = 3, prior_decay: float = 2.0):
        self.k = int(alphabet_size)
        self.R = int(R)
        priors = np.array([prior_decay ** (-r) for r in range(self.R + 1)], dtype=float)
        self.alpha = priors / priors.sum()
        self.tables: List[Dict[Tuple[int, ...], np.ndarray]] = [
            defaultdict(lambda: np.zeros(self.k, dtype=float))
            for _ in range(self.R + 1)
        ]
        self.history: List[int] = []

    @staticmethod
    def _kt_predict(counts: np.ndarray, k: int) -> np.ndarray:
        n = float(counts.sum())
        return (counts + 0.5) / (n + 0.5 * k)

    def update_and_codelen(self, sym: int) -> float:
        sym = int(sym)
        preds = []
        for r in range(self.R + 1):
            ctx = tuple(self.history[-r:]) if r > 0 else ()
            counts = self.tables[r][ctx]
            p = self._kt_predict(counts, self.k)
            preds.append(p)
        mix = sum(a * p for a, p in zip(self.alpha, preds))
        p_sym = max(float(mix[sym]), 1e-300)
        codelen = -math.log(p_sym, 2.0)
        post = np.array(
            [a * max(float(p[sym]), 1e-300) for a, p in zip(self.alpha, preds)], dtype=float
        )
        s = float(post.sum())
        post = (np.ones_like(post) / len(post)) if s <= 0 else (post / s)
        self.alpha = post
        for r in range(self.R + 1):
            ctx = tuple(self.history[-r:]) if r > 0 else ()
            self.tables[r][ctx][sym] += 1.0
        self.history.append(sym)
        return codelen

    def fit(self, sequence: Sequence[int]) -> float:
        self.reset()
        total = 0.0
        for s in sequence:
            total += self.update_and_codelen(int(s))
        return total

    def reset(self) -> None:
        self.alpha = self.alpha * 0 + (1.0 / (self.R + 1))
        self.tables = [
            defaultdict(lambda: np.zeros(self.k, dtype=float))
            for _ in range(self.R + 1)
        ]
        self.history = []

    def snapshot_frozen(self) -> "KTFrozenPredictor":
        tables_copy: List[Dict[Tuple[int, ...], np.ndarray]] = []
        for d in self.tables:
            newd: Dict[Tuple[int, ...], np.ndarray] = {}
            for ctx, arr in d.items():
                newd[ctx] = arr.copy()
            tables_copy.append(newd)
        return KTFrozenPredictor(tables_copy, self.alpha.copy(), self.k, self.R)


class KTFrozenPredictor:
    def __init__(
        self,
        tables: List[Dict[Tuple[int, ...], np.ndarray]],
        alpha: np.ndarray,
        k: int,
        R: int,
    ):
        self.tables = tables
        self.alpha = alpha
        self.k = int(k)
        self.R = int(R)
        self.history: List[int] = []

    @staticmethod
    def _kt_predict(counts: np.ndarray, k: int) -> np.ndarray:
        n = float(counts.sum())
        return (counts + 0.5) / (n + 0.5 * k)

    def codelen_sequence(self, sequence: Sequence[int]) -> float:
        self.history = []
        total = 0.0
        for sym in sequence:
            sym = int(sym)
            preds = []
            for r in range(self.R + 1):
                ctx = tuple(self.history[-r:]) if r > 0 else ()
                counts = self.tables[r].get(ctx)
                if counts is None:
                    counts = np.zeros(self.k, dtype=float)
                p = self._kt_predict(counts, self.k)
                preds.append(p)
            mix = sum(a * p for a, p in zip(self.alpha, preds))
            p_sym = max(float(mix[sym]), 1e-300)
            total += -math.log(p_sym, 2.0)
            self.history.append(sym)
        return total


class LZ78Coder:
    def __init__(self, alphabet_size: int):
        self.k = int(alphabet_size)

    def total_codelen(self, sequence: Sequence[int]) -> float:
        dict_trie: Dict[Tuple[int, ...], Dict[int, dict]] = {(): {}}
        curr: Tuple[int, ...] = ()
        phrases = 0
        bits = 0.0
        for sym in sequence:
            sym = int(sym)
            if curr not in dict_trie:
                dict_trie[curr] = {}
            if sym in dict_trie[curr]:
                curr = curr + (sym,)
            else:
                phrases += 1
                index_bits = math.log2(max(1, phrases))
                symbol_bits = math.log2(self.k)
                bits += index_bits + symbol_bits
                dict_trie[curr][sym] = {}
                curr = ()
        if len(curr) > 0:
            phrases += 1
            index_bits = math.log2(max(1, phrases))
            symbol_bits = math.log2(self.k)
            bits += index_bits + symbol_bits
        return bits

