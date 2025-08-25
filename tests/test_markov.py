import numpy as np
from uec.markov import random_markov_biased, sample_markov, entropy_production_rate_bits
from uec.holonomy import klrate_holonomy_time_reversal_markov


def test_time_reversal_equals_ep_small_n():
    rng = np.random.default_rng(123)
    T = random_markov_biased(k=3, delta=0.6, rng=rng)
    sigma = entropy_production_rate_bits(T)
    x = sample_markov(T, n=60000, rng=rng)
    hol = klrate_holonomy_time_reversal_markov(x, k=3, R=3)
    diff = abs(hol - sigma)
    rel = diff / max(1e-8, abs(sigma))
    assert (diff < 1e-2) or (rel < 0.05)

