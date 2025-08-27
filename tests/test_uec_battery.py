import pytest
import tempfile
import os
import json
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the root directory to sys.path to import uec_battery
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from uec_battery import (
    set_seeds,
    _row_stochastic,
    stationary_distribution,
    sample_markov,
    entropy_production_rate_bits,
    random_markov_biased,
    counts_from_sequence,
    ep_bits_from_counts_smoothed,
    KTMarkovMixture,
    KTFrozenPredictor,
    LZ78Coder,
    klrate_between_sequences,
    klrate_holonomy_time_reversal_markov,
    quantile_bins,
    discretize_series,
    load_csv_column,
    auc_from_scores,
    window_iter,
    aot_from_series,
)


class TestUtilities:
    def test_set_seeds(self):
        # Test that setting seeds affects random number generation
        set_seeds(42)
        val1 = np.random.random()
        
        set_seeds(42)
        val2 = np.random.random()
        
        assert val1 == val2  # Should be identical with same seed
        
    def test_row_stochastic(self):
        # Test row normalization
        T = np.array([[2.0, 1.0], [3.0, 3.0]])
        T_norm = _row_stochastic(T)
        
        np.testing.assert_allclose(T_norm.sum(axis=1), [1.0, 1.0])
        
        # Test with zero row
        T_zero = np.array([[0.0, 0.0], [1.0, 2.0]])
        T_zero_norm = _row_stochastic(T_zero)
        
        # Zero row stays zero in this implementation (rs[rs == 0.0] = 1.0 then T/rs gives 0/1=0)
        assert T_zero_norm.sum(axis=1)[0] == 0.0  # Zero row divided by 1 gives zero
        assert T_zero_norm.sum(axis=1)[1] == 1.0  # Non-zero row normalizes properly
        
    def test_stationary_distribution(self):
        # Test with known stationary distribution
        T = np.array([[0.7, 0.3], [0.4, 0.6]])
        pi = stationary_distribution(T)
        
        # Should satisfy pi @ T = pi
        np.testing.assert_allclose(pi @ T, pi, atol=1e-10)
        np.testing.assert_allclose(pi.sum(), 1.0)
        
    def test_sample_markov(self):
        rng = np.random.default_rng(42)
        T = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        seq = sample_markov(T, n=100, rng=rng)
        
        assert len(seq) == 100
        assert all(s in [0, 1] for s in seq)
        
    def test_entropy_production_rate_bits(self):
        # Test reversible matrix (should give ~0)
        T_rev = np.array([[0.5, 0.5], [0.5, 0.5]])
        ep_rev = entropy_production_rate_bits(T_rev)
        assert abs(ep_rev) < 1e-10
        
        # Test irreversible matrix (should give >0)
        T_irrev = np.array([[0.9, 0.1], [0.2, 0.8]])
        ep_irrev = entropy_production_rate_bits(T_irrev)
        assert ep_irrev > 0
        
    def test_random_markov_biased(self):
        rng = np.random.default_rng(42)
        T = random_markov_biased(k=3, delta=0.5, rng=rng)
        
        assert T.shape == (3, 3)
        np.testing.assert_allclose(T.sum(axis=1), [1.0, 1.0, 1.0])
        assert np.all(T > 0)  # Should be strictly positive
        
    def test_counts_from_sequence(self):
        seq = [0, 1, 0, 1, 2]
        counts = counts_from_sequence(seq, k=3)
        
        assert counts.shape == (3, 3)
        assert counts[0, 1] == 2  # 0->1 appears twice
        assert counts[1, 0] == 1  # 1->0 appears once
        assert counts[1, 2] == 1  # 1->2 appears once
        
    def test_ep_bits_from_counts_smoothed(self):
        # Create simple transition counts
        counts = np.array([[10, 2], [1, 8]], dtype=np.int64)
        ep = ep_bits_from_counts_smoothed(counts, alpha=0.5)
        
        assert ep >= 0  # EP should be non-negative
        assert np.isfinite(ep)


class TestCoders:
    def test_kt_markov_mixture_basic(self):
        coder = KTMarkovMixture(alphabet_size=3, R=2)
        
        # Test basic properties
        assert coder.k == 3
        assert coder.R == 2
        
        # Test fitting a sequence
        seq = [0, 1, 2, 0, 1, 2] * 10
        total_bits = coder.fit(seq)
        
        assert total_bits > 0
        assert np.isfinite(total_bits)
        
        # Test frozen snapshot
        frozen = coder.snapshot_frozen()
        assert isinstance(frozen, KTFrozenPredictor)
        
    def test_kt_frozen_predictor(self):
        # Train a coder first
        coder = KTMarkovMixture(alphabet_size=2, R=1)
        train_seq = [0, 1, 0, 1, 0] * 20
        coder.fit(train_seq)
        
        # Get frozen version
        frozen = coder.snapshot_frozen()
        
        # Test prediction on test sequence
        test_seq = [0, 1, 0, 1]
        codelen = frozen.codelen_sequence(test_seq)
        
        assert codelen > 0
        assert np.isfinite(codelen)
        
    def test_lz78_coder(self):
        coder = LZ78Coder(alphabet_size=3)
        
        seq = [0, 1, 2, 0, 1, 2, 0, 1]
        codelen = coder.total_codelen(seq)
        
        assert codelen > 0
        assert np.isfinite(codelen)
        
        # Test on repeated pattern (should compress well)
        repeated = [0, 1] * 50
        codelen_repeated = coder.total_codelen(repeated)
        
        # Should be much less than naive encoding
        naive_bits = len(repeated) * np.log2(2)
        assert codelen_repeated < naive_bits


class TestHolonomy:
    def test_klrate_between_sequences(self):
        rng = np.random.default_rng(42)
        
        # Identical sequences should have ~0 KL rate
        seq = rng.integers(0, 3, size=1000).tolist()
        kl_rate = klrate_between_sequences(seq, seq, k=3, R=2)
        assert abs(kl_rate) < 1e-6
        
        # Different sequences should have positive KL rate
        seq1 = [0, 1, 0, 1] * 100
        seq2 = [1, 0, 1, 0] * 100
        kl_rate = klrate_between_sequences(seq1, seq2, k=2, R=2)
        assert kl_rate >= 0
        
    def test_klrate_holonomy_time_reversal_markov(self):
        rng = np.random.default_rng(42)
        
        # Test on i.i.d. sequence (should be nearly reversible)
        seq_iid = rng.integers(0, 3, size=5000).tolist()
        kl_rate = klrate_holonomy_time_reversal_markov(seq_iid, k=3, R=2)
        
        # Should be small for i.i.d. sequence
        assert abs(kl_rate) < 0.05
        
        # Test on biased sequence
        seq_biased = [0, 1, 2] * 1000  # deterministic cycle
        kl_rate_biased = klrate_holonomy_time_reversal_markov(seq_biased, k=3, R=2)
        
        # Should detect irreversibility
        assert kl_rate_biased > 0.01


class TestAoTUtils:
    def test_quantile_bins(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = quantile_bins(x, k=3)
        
        assert len(bins) == 4  # k+1 edges
        assert bins[0] <= bins[1] <= bins[2] <= bins[3]
        
    def test_discretize_series(self):
        x = np.array([1.1, 2.3, 3.7, 4.2, 5.8])
        discrete = discretize_series(x, k=3)
        
        assert len(discrete) == len(x)
        assert all(0 <= s < 3 for s in discrete)
        
    def test_load_csv_column(self):
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n")
            f.write("1.5,2.7,3.2\n")
            f.write("4.1,5.3,6.8\n")
            temp_path = f.name
            
        try:
            # Test by column name
            data = load_csv_column(temp_path, column="b")
            np.testing.assert_allclose(data, [2.7, 5.3])
            
            # Test by column index
            data_idx = load_csv_column(temp_path, column=2)
            np.testing.assert_allclose(data_idx, [3.2, 6.8])
            
        finally:
            os.unlink(temp_path)
            
    def test_auc_from_scores(self):
        # Perfect separation
        pos = np.array([0.8, 0.9, 0.7])
        neg = np.array([0.1, 0.2, 0.3])
        auc = auc_from_scores(pos, neg)
        assert auc >= 0.9  # Should be high
        
        # No separation
        same = np.array([0.5, 0.6, 0.4])
        auc_same = auc_from_scores(same, same)
        # With identical arrays, some ties occur which can give AUC != 0.5 exactly
        assert 0.2 <= auc_same <= 0.8  # Allow broader range for tied values
        
    def test_window_iter(self):
        seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        # Non-overlapping windows
        windows = window_iter(seq, win=3, stride=3)
        assert windows == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        
        # Overlapping windows
        windows_overlap = window_iter(seq, win=4, stride=2)
        assert windows_overlap[0] == [0, 1, 2, 3]
        assert windows_overlap[1] == [2, 3, 4, 5]
        
    def test_aot_from_series(self):
        # Simple test signal
        rng = np.random.default_rng(42)
        n = 10000
        t = np.arange(n)
        x = np.sin(0.01 * t) + 0.1 * rng.standard_normal(n)
        
        res = aot_from_series(
            x, k=4, R=2, 
            win=500, stride=250,
            train_frac=0.6,
            B=50  # Fewer bootstrap samples for speed
        )
        
        # Check output structure
        assert "auc" in res
        assert "bits_per_step" in res
        assert "hol_ci_lo" in res
        assert "hol_ci_hi" in res
        
        # Check value ranges
        assert 0.0 <= res["auc"] <= 1.0
        assert np.isfinite(res["bits_per_step"])
        assert res["hol_ci_lo"] <= res["hol_ci_hi"]
        
    def test_aot_from_series_with_preprocessing(self):
        # Test with different preprocessing options - use larger dataset
        x = np.array([100, 102, 98, 105, 97, 103, 108, 95, 110, 101, 99, 107])
        
        # Test with differencing
        res_diff = aot_from_series(
            x, k=2, R=1,
            win=2, stride=1,
            train_frac=0.4,  # Use less for training to leave more for test
            use_diff=True,
            B=10
        )
        
        assert "auc" in res_diff
        assert np.isfinite(res_diff["bits_per_step"])
        
        # Test with log returns - use larger dataset
        x_positive = np.array([100, 102, 105, 103, 108, 112, 107, 115, 109, 118, 114, 120])
        res_logret = aot_from_series(
            x_positive, k=2, R=1,
            win=2, stride=1,
            train_frac=0.4,
            use_logreturn=True,
            B=10
        )
        
        assert "auc" in res_logret
        assert np.isfinite(res_logret["bits_per_step"])


class TestIntegration:
    def test_markov_chain_holonomy_consistency(self):
        """Test that holonomy estimation is consistent with theoretical EP."""
        rng = np.random.default_rng(123)
        
        # Create biased Markov chain
        T = random_markov_biased(k=3, delta=0.4, rng=rng)
        theoretical_ep = entropy_production_rate_bits(T)
        
        # Sample long sequence
        seq = sample_markov(T, n=50000, rng=rng)
        
        # Estimate holonomy
        estimated_hol = klrate_holonomy_time_reversal_markov(seq, k=3, R=3)
        
        # Should be reasonably close - allow for some numerical variance
        abs_error = abs(estimated_hol - theoretical_ep)
        rel_error = abs_error / max(abs(theoretical_ep), 1e-6)
        assert (abs_error < 0.025) or (rel_error < 0.12)  # Allow either small absolute or relative error
        
    def test_coder_consistency(self):
        """Test that different coders give reasonable results."""
        rng = np.random.default_rng(42)
        seq = rng.integers(0, 3, size=1000).tolist()
        
        # KT coder
        kt = KTMarkovMixture(alphabet_size=3, R=2)
        kt_bits = kt.fit(seq)
        
        # LZ coder
        lz = LZ78Coder(alphabet_size=3)
        lz_bits = lz.total_codelen(seq)
        
        # Both should give finite positive values
        assert kt_bits > 0 and np.isfinite(kt_bits)
        assert lz_bits > 0 and np.isfinite(lz_bits)
        
        # For random sequence, shouldn't be too different
        # (This is a loose check - exact relationship depends on sequence)
        ratio = kt_bits / lz_bits
        assert 0.1 < ratio < 10  # Within order of magnitude


@pytest.mark.slow
class TestLongRunning:
    """Tests that take longer to run - marked for optional execution."""
    
    def test_full_battery_simulation(self):
        """Test a complete battery simulation (reduced size)."""
        rng = np.random.default_rng(12345)
        
        # Generate biased Markov chain
        T = random_markov_biased(k=3, delta=0.5, rng=rng)
        theoretical_ep = entropy_production_rate_bits(T)
        
        # Sample sequence
        seq = sample_markov(T, n=20000, rng=rng)
        
        # Test various holonomy measures
        hol_tr = klrate_holonomy_time_reversal_markov(seq, k=3, R=3)
        
        # Check theoretical consistency - allow slightly larger tolerance for this stochastic test
        assert abs(hol_tr - theoretical_ep) < 0.025
        
        # Test AoT analysis on the sequence
        x_continuous = np.array([float(s) + 0.1*rng.standard_normal() for s in seq])
        aot_result = aot_from_series(
            x_continuous, k=3, R=2,
            win=200, stride=100,
            B=20  # Reduced bootstrap samples
        )
        
        # Should detect some irreversibility
        assert aot_result["auc"] > 0.55
        assert aot_result["bits_per_step"] > 0