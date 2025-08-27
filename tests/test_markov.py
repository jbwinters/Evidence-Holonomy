import numpy as np
import pytest
from uec.markov import (
    _row_stochastic,
    stationary_distribution,
    sample_markov,
    entropy_production_rate_bits,
    random_markov_biased,
    sample_HMM,
    ring_chain_3,
    ring_ep_bits,
)
from uec.holonomy import klrate_holonomy_time_reversal_markov


def test_row_stochastic():
    # Test that rows sum to 1
    T = np.array([[0.7, 0.3], [0.4, 0.6]])
    T_stoch = _row_stochastic(T)
    np.testing.assert_allclose(T_stoch.sum(axis=1), 1.0)
    
    # Test with already stochastic matrix
    np.testing.assert_allclose(T_stoch, T)
    
    # Test with non-stochastic matrix
    T_bad = np.array([[2.0, 1.0], [1.0, 3.0]])
    T_fixed = _row_stochastic(T_bad)
    np.testing.assert_allclose(T_fixed.sum(axis=1), 1.0)


def test_stationary_distribution():
    # Test with known stationary distribution
    T = np.array([[0.7, 0.3], [0.4, 0.6]])
    pi = stationary_distribution(T)
    
    # Check pi @ T = pi (stationary property)
    np.testing.assert_allclose(pi @ T, pi, atol=1e-10)
    
    # Check pi sums to 1
    np.testing.assert_allclose(pi.sum(), 1.0)
    
    # Test with identity matrix (uniform stationary)
    T_uniform = np.ones((3, 3)) / 3
    pi_uniform = stationary_distribution(T_uniform)
    np.testing.assert_allclose(pi_uniform, [1/3, 1/3, 1/3], atol=1e-10)


def test_sample_markov():
    rng = np.random.default_rng(42)
    T = np.array([[0.8, 0.2], [0.3, 0.7]])
    
    # Test basic sampling
    seq = sample_markov(T, n=1000, rng=rng)
    assert len(seq) == 1000
    assert all(s in [0, 1] for s in seq)
    
    # Test with custom initial distribution
    init_dist = np.array([0.0, 1.0])  # start from state 1
    seq_init = sample_markov(T, n=100, init=init_dist, rng=rng)
    assert seq_init[0] == 1
    
    # Test reproducibility with same seed
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    seq1 = sample_markov(T, n=50, rng=rng1)
    seq2 = sample_markov(T, n=50, rng=rng2)
    np.testing.assert_array_equal(seq1, seq2)


def test_entropy_production_rate_bits():
    # Test reversible chain (zero entropy production)
    T_rev = np.array([[0.5, 0.5], [0.5, 0.5]])
    ep = entropy_production_rate_bits(T_rev)
    assert abs(ep) < 1e-10
    
    # Test irreversible chain (positive entropy production)
    T_irrev = np.array([[0.9, 0.1], [0.2, 0.8]])
    ep_irrev = entropy_production_rate_bits(T_irrev)
    assert ep_irrev > 0


def test_random_markov_biased():
    rng = np.random.default_rng(42)
    
    # Test basic generation
    T = random_markov_biased(k=3, delta=0.5, rng=rng)
    assert T.shape == (3, 3)
    np.testing.assert_allclose(T.sum(axis=1), 1.0)
    assert np.all(T >= 0)
    
    # Test that delta=0 gives closer to uniform matrix (but with random variation)
    T_uniform = random_markov_biased(k=2, delta=0.0, rng=rng)
    # With delta=0, should have less directional bias, but still some randomness
    # Just check that it's a valid stochastic matrix
    np.testing.assert_allclose(T_uniform.sum(axis=1), [1.0, 1.0])
    assert np.all(T_uniform > 0)  # All entries should be positive


def test_sample_HMM():
    rng = np.random.default_rng(42)
    
    # Simple 2-state HMM
    T = np.array([[0.7, 0.3], [0.4, 0.6]])
    E = np.array([[0.8, 0.2], [0.1, 0.9]])  # emission matrix
    
    obs, k_hidden, m_obs = sample_HMM(T, E, n=100, rng=rng)
    
    assert len(obs) == 100
    assert k_hidden == 2
    assert m_obs == 2
    assert all(o in [0, 1] for o in obs)


def test_ring_chain_3():
    # Test 3-state ring chain (p+q must be < 1)
    T = ring_chain_3(p=0.6, q=0.3)
    assert T.shape == (3, 3)
    np.testing.assert_allclose(T.sum(axis=1), 1.0)
    
    # Check specific structure: T[i, (i+1)%3] = p, T[i, (i+2)%3] = q
    assert abs(T[0, 1] - 0.6) < 1e-10
    assert abs(T[0, 2] - 0.3) < 1e-10
    assert abs(T[1, 2] - 0.6) < 1e-10
    assert abs(T[1, 0] - 0.3) < 1e-10


def test_ring_ep_bits():
    # Test entropy production calculation for ring
    ep = ring_ep_bits(p=0.6, q=0.4)
    assert ep >= 0  # Should be non-negative
    
    # Test reversible case p=q=0.5
    ep_rev = ring_ep_bits(p=0.5, q=0.5)
    assert abs(ep_rev) < 1e-10


def test_time_reversal_equals_ep_small_n():
    rng = np.random.default_rng(123)
    T = random_markov_biased(k=3, delta=0.6, rng=rng)
    sigma = entropy_production_rate_bits(T)
    x = sample_markov(T, n=60000, rng=rng)
    hol = klrate_holonomy_time_reversal_markov(x, k=3, R=3)
    diff = abs(hol - sigma)
    rel = diff / max(1e-8, abs(sigma))
    assert (diff < 1e-2) or (rel < 0.05)

