import numpy as np

from uec.transforms import Permute, TimeReverse, TransitionEncode, TransitionDecodeTakeSecond, Downsample, UpsampleRepeat, apply_loop
from uec.holonomy import klrate_holonomy_general, klrate_holonomy_time_reversal_markov


def test_gauge_holonomy_near_zero():
    rng = np.random.default_rng(0)
    k = 7
    x = rng.integers(0, k, size=20000).tolist()
    perm = np.arange(k)
    rng.shuffle(perm)
    inv = np.zeros_like(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    loop = [Permute(perm.tolist()), Permute(inv.tolist())]
    val = klrate_holonomy_general(x, list(range(k)), loop, k=k, R=3, align="head")
    assert abs(val) < 1e-3


def test_surrogate_time_shuffle_near_zero():
    rng = np.random.default_rng(1)
    k = 5
    x = rng.integers(0, k, size=30000)
    rng.shuffle(x)
    val = klrate_holonomy_time_reversal_markov(x.tolist(), k=k, R=3)
    # i.i.d. sequence should be (nearly) reversible
    assert abs(val) < 5e-3


def test_coarse_grain_loop_non_negative():
    rng = np.random.default_rng(2)
    k = 4
    # Weakly biased random walk on k symbols
    x = rng.integers(0, k, size=50000).tolist()
    loop = [Downsample(step=2), UpsampleRepeat(step=2)]
    val = klrate_holonomy_general(x, list(range(k)), loop, k=k, R=3, align="head")
    # Allow tiny numerical slack but expect non-negative
    assert val > -1e-6


def test_time_reversal_alignment_head_tail_ok():
    rng = np.random.default_rng(3)
    k = 3
    x = rng.integers(0, k, size=10000).tolist()
    E = TransitionEncode(k)
    Rv = TimeReverse()
    D2 = TransitionDecodeTakeSecond(k)
    q_head, _ = apply_loop(x, list(range(k)), [E, Rv, D2])
    # Ensure head and tail alignments both run (no exceptions); values finite
    v_head = klrate_holonomy_general(x, list(range(k)), [E, Rv, D2], k=k, R=3, align="head")
    v_tail = klrate_holonomy_general(x, list(range(k)), [E, Rv, D2], k=k, R=3, align="tail")
    assert np.isfinite(v_head) and np.isfinite(v_tail)

