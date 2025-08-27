import numpy as np
import pytest
from uec.holonomy import (
    klrate_between_sequences,
    klrate_holonomy_general,
    klrate_holonomy_time_reversal_markov,
)
from uec.transforms import Permute, TimeReverse, TransitionEncode, TransitionDecodeTakeSecond


def test_klrate_between_sequences():
    rng = np.random.default_rng(42)
    
    # Test identical sequences should have ~0 KL rate
    seq = rng.integers(0, 4, size=10000).tolist()
    kl_rate = klrate_between_sequences(seq, seq, k=4, R=3)
    assert abs(kl_rate) < 1e-6
    
    # Test different sequences should have positive KL rate
    seq1 = rng.integers(0, 3, size=5000).tolist()
    seq2 = rng.integers(0, 3, size=5000).tolist()
    kl_rate = klrate_between_sequences(seq1, seq2, k=3, R=2)
    assert kl_rate >= 0  # KL divergence is non-negative
    
    # Test with deterministic sequences
    seq_det1 = [0, 1, 0, 1] * 1000
    seq_det2 = [1, 0, 1, 0] * 1000
    kl_rate = klrate_between_sequences(seq_det1, seq_det2, k=2, R=2)
    assert kl_rate > 0


def test_klrate_holonomy_general_identity_loop():
    rng = np.random.default_rng(42)
    seq = rng.integers(0, 5, size=8000).tolist()
    alphabet = list(range(5))
    
    # Empty loop should give ~0 holonomy
    kl_rate = klrate_holonomy_general(seq, alphabet, [], k=5, R=3)
    assert abs(kl_rate) < 1e-6
    
    # Gauge transformation (permutation + inverse) should give ~0 holonomy
    perm = [4, 2, 0, 3, 1]  # some permutation
    inv_perm = [2, 4, 1, 3, 0]  # inverse permutation
    loop = [Permute(perm), Permute(inv_perm)]
    kl_rate = klrate_holonomy_general(seq, alphabet, loop, k=5, R=3)
    assert abs(kl_rate) < 1e-3


def test_klrate_holonomy_general_time_reversal():
    # Create a biased sequence (irreversible)
    rng = np.random.default_rng(42)
    # Markov-like: state i tends to go to (i+1) % k
    seq = []
    state = 0
    k = 4
    for _ in range(20000):
        seq.append(state)
        if rng.random() < 0.8:  # strong bias
            state = (state + 1) % k
        else:
            state = rng.integers(0, k)
    
    alphabet = list(range(k))
    
    # Time reversal should give positive holonomy for irreversible sequence
    loop = [TimeReverse()]
    kl_rate = klrate_holonomy_general(seq, alphabet, loop, k=k, R=3)
    assert kl_rate > 0
    
    # Test alignment options
    kl_head = klrate_holonomy_general(seq, alphabet, loop, k=k, R=3, align="head")
    kl_tail = klrate_holonomy_general(seq, alphabet, loop, k=k, R=3, align="tail")
    assert np.isfinite(kl_head)
    assert np.isfinite(kl_tail)


def test_klrate_holonomy_general_transition_encoding():
    rng = np.random.default_rng(42)
    k = 3
    seq = rng.integers(0, k, size=15000).tolist()
    alphabet = list(range(k))
    
    # Transition encode -> time reverse -> decode should be well-defined
    E = TransitionEncode(k)
    R = TimeReverse()
    D = TransitionDecodeTakeSecond(k)
    loop = [E, R, D]
    
    kl_rate = klrate_holonomy_general(seq, alphabet, loop, k=k, R=3)
    assert np.isfinite(kl_rate)
    assert kl_rate >= -1e-6  # Should be non-negative (up to numerical precision)


def test_klrate_holonomy_time_reversal_markov():
    rng = np.random.default_rng(42)
    
    # Test on i.i.d. sequence (should be nearly reversible)
    seq_iid = rng.integers(0, 4, size=30000)
    rng.shuffle(seq_iid)  # ensure it's shuffled
    kl_rate = klrate_holonomy_time_reversal_markov(seq_iid.tolist(), k=4, R=3)
    assert abs(kl_rate) < 0.01  # i.i.d. should be nearly reversible
    
    # Test on highly biased sequence (should be irreversible)
    seq_biased = [0, 1, 2, 3] * 5000  # deterministic cycle
    kl_rate = klrate_holonomy_time_reversal_markov(seq_biased, k=4, R=3)
    assert kl_rate > 0.1  # should detect irreversibility


def test_holonomy_functions_consistency():
    """Test that different holonomy functions give consistent results for time reversal."""
    rng = np.random.default_rng(123)
    k = 3
    seq = rng.integers(0, k, size=10000).tolist()
    
    # Compare specific time reversal implementations
    kl1 = klrate_holonomy_time_reversal_markov(seq, k=k, R=3)
    
    alphabet = list(range(k))
    loop = [TimeReverse()]
    kl2 = klrate_holonomy_general(seq, alphabet, loop, k=k, R=3)
    
    # They should be similar (within numerical tolerance)
    assert abs(kl1 - kl2) < 0.1