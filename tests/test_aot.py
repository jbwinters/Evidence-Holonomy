import numpy as np
from uec.aot import aot_from_series


def test_aot_on_irreversible_series():
    # Synthetic: sawtooth-like signal with trend breaks (irreversible)
    n = 20000
    t = np.arange(n)
    x = (t % 200) + 0.1 * np.random.default_rng(0).standard_normal(n)
    res = aot_from_series(x, k=8, R=3, win=1024, stride=512, use_diff=True)
    assert res["auc"] >= 0.6

