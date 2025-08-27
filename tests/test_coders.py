import numpy as np
import pytest
from uec.coders import KTMarkovMixture, KTFrozenPredictor, LZ78Coder


class TestKTMarkovMixture:
    def test_initialization(self):
        coder = KTMarkovMixture(alphabet_size=4, R=2)
        assert coder.k == 4
        assert coder.R == 2
        assert len(coder.tables) == 3  # R+1 orders (0, 1, 2)
        
    def test_update_and_codelen(self):
        coder = KTMarkovMixture(alphabet_size=3, R=2)
        
        # Test incremental updates
        codelen1 = coder.update_and_codelen(0)
        codelen2 = coder.update_and_codelen(1)
        
        assert codelen1 > 0
        assert codelen2 > 0
        assert np.isfinite(codelen1)
        assert np.isfinite(codelen2)
        
    def test_fit_sequence(self):
        coder = KTMarkovMixture(alphabet_size=3, R=1)
        seq = [0, 1, 2, 0, 1]
        
        # Fit sequence and get total code length
        total_bits = coder.fit(seq)
        assert np.isfinite(total_bits)
        assert total_bits > 0  # Should require some bits
        
        # Empty sequence should give 0 bits
        coder_empty = KTMarkovMixture(alphabet_size=3, R=1)
        assert coder_empty.fit([]) == 0.0
        
    def test_sequence_training(self):
        coder = KTMarkovMixture(alphabet_size=2, R=2)
        seq = [0, 1, 0, 1, 0, 1] * 100  # periodic sequence
        
        # Train on sequence and get code length
        total_bits = coder.fit(seq)
        
        # Should give finite positive code length
        assert total_bits > 0
        assert np.isfinite(total_bits)
        
        # Test that the coder has learned some patterns
        frozen = coder.snapshot_frozen()
        test_seq = [0, 1, 0, 1]
        test_bits = frozen.codelen_sequence(test_seq)
        
        # Should be reasonable code length for test sequence
        assert test_bits > 0
        assert np.isfinite(test_bits)


class TestKTFrozenPredictor:
    def test_initialization(self):
        # Create a base coder to freeze
        base_coder = KTMarkovMixture(alphabet_size=3, R=1)
        base_coder.fit([0, 1, 2, 0, 1])
        
        frozen = base_coder.snapshot_frozen()
        assert frozen.k == 3
        assert frozen.R == 1
        
    def test_codelen_sequence(self):
        base_coder = KTMarkovMixture(alphabet_size=2, R=1)
        train_seq = [0, 1, 0, 1, 0] * 20
        
        # Train base coder
        base_coder.fit(train_seq)
        frozen = base_coder.snapshot_frozen()
        
        # Test evaluation on test sequence
        test_seq = [0, 1, 0, 1]
        codelen = frozen.codelen_sequence(test_seq)
        
        assert codelen > 0
        assert np.isfinite(codelen)
        
    def test_consistency_with_base_training(self):
        # Train two identical coders
        base_coder1 = KTMarkovMixture(alphabet_size=3, R=2)
        base_coder2 = KTMarkovMixture(alphabet_size=3, R=2)
        
        train_seq = [0, 1, 2, 0, 1, 2] * 10
        
        # Train both the same way
        base_coder1.fit(train_seq)
        base_coder2.fit(train_seq)
        
        frozen1 = base_coder1.snapshot_frozen()
        frozen2 = base_coder2.snapshot_frozen()
        
        test_seq = [0, 1, 2, 0]
        codelen1 = frozen1.codelen_sequence(test_seq)
        codelen2 = frozen2.codelen_sequence(test_seq)
        
        # Should give same results
        np.testing.assert_allclose(codelen1, codelen2, rtol=1e-10)


class TestLZ78Coder:
    def test_initialization(self):
        coder = LZ78Coder(alphabet_size=4)
        assert coder.k == 4
        
    def test_total_codelen_basic(self):
        coder = LZ78Coder(alphabet_size=3)
        seq = [0, 1, 2, 0, 1]
        
        # Get total code length
        codelen = coder.total_codelen(seq)
        assert codelen > 0
        assert np.isfinite(codelen)
        
    def test_compression_on_repeated_patterns(self):
        coder = LZ78Coder(alphabet_size=2)
        
        # Highly compressible sequence
        seq_compressible = [0, 1] * 50
        codelen_comp = coder.total_codelen(seq_compressible)
        
        # Random sequence (less compressible)
        rng = np.random.default_rng(42)
        seq_random = rng.integers(0, 2, size=100).tolist()
        coder2 = LZ78Coder(alphabet_size=2)
        codelen_rand = coder2.total_codelen(seq_random)
        
        # Per-symbol code length should be better for repeated patterns
        per_symbol_comp = codelen_comp / len(seq_compressible)
        per_symbol_rand = codelen_rand / len(seq_random)
        
        # This is a loose check - LZ78 should do reasonably well on repetitive patterns
        assert per_symbol_comp <= per_symbol_rand * 1.5  # Allow some tolerance
        
    def test_empty_sequence(self):
        coder = LZ78Coder(alphabet_size=3)
        
        # Empty sequence should give 0 bits
        codelen_empty = coder.total_codelen([])
        assert codelen_empty == 0.0
        
    def test_single_symbol_sequence(self):
        coder = LZ78Coder(alphabet_size=2)
        
        # Single repeated symbol
        seq_single = [0] * 20
        codelen = coder.total_codelen(seq_single)
        
        assert codelen > 0
        assert np.isfinite(codelen)
        
        # Should compress better than naive encoding
        naive_bits = len(seq_single) * np.log2(2)
        assert codelen < naive_bits


def test_coders_integration():
    """Test that different coders can work on the same sequence."""
    rng = np.random.default_rng(42)
    k = 3
    seq = rng.integers(0, k, size=200).tolist()
    
    # KT Markov
    kt_coder = KTMarkovMixture(alphabet_size=k, R=2)
    kt_bits = kt_coder.fit(seq)
    
    # LZ78
    lz_coder = LZ78Coder(alphabet_size=k)
    lz_bits = lz_coder.total_codelen(seq)
    
    # Both should give finite positive code lengths
    assert np.isfinite(kt_bits)
    assert np.isfinite(lz_bits)
    assert kt_bits > 0
    assert lz_bits > 0
    
    # Should be in reasonable range for this sequence length
    max_naive_bits = len(seq) * np.log2(k)  # naive encoding
    assert kt_bits <= max_naive_bits * 2  # allow some overhead
    assert lz_bits <= max_naive_bits * 2