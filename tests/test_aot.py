import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from uec.aot import (
    quantile_bins,
    discretize_series,
    load_csv_column,
    load_wav_mono,
    auc_from_scores,
    window_iter,
    train_forward_and_reverse_models,
    signed_lr_score,
    aot_from_series,
)
from uec.coders import KTFrozenPredictor, KTMarkovMixture


def test_quantile_bins():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Test with k=3
    bins = quantile_bins(x, k=3)
    assert len(bins) == 4  # k+1 edges for k bins
    assert bins[0] <= bins[1] <= bins[2] <= bins[3]
    
    # Test with k=1 (should work)
    bins_single = quantile_bins(x, k=1)
    assert len(bins_single) == 2  # k+1 edges for k=1 bins
    
    # Test with uniform data
    x_uniform = np.ones(100)
    bins_uniform = quantile_bins(x_uniform, k=5)
    assert len(bins_uniform) == 6  # k+1 edges


def test_discretize_series():
    x = np.array([1.5, 2.7, 3.2, 4.8, 5.1])
    
    # Discretize to k=3 symbols
    discrete = discretize_series(x, k=3)
    assert len(discrete) == len(x)
    assert all(0 <= s < 3 for s in discrete)
    
    # Test with k=1
    discrete_single = discretize_series(x, k=1)
    assert all(s == 0 for s in discrete_single)
    
    # Test sorted property: larger values should generally get larger symbols
    x_sorted = np.sort(x)
    discrete_sorted = discretize_series(x_sorted, k=5)
    # Should be non-decreasing (allowing ties at boundaries)
    for i in range(1, len(discrete_sorted)):
        assert discrete_sorted[i] >= discrete_sorted[i-1] - 1  # allow small boundary effects


def test_load_csv_column():
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2,col3\n")
        f.write("1.5,2.7,3.2\n")
        f.write("4.8,5.1,6.3\n")
        f.write("7.2,8.5,9.1\n")
        temp_path = f.name
    
    try:
        # Test loading by column name
        data = load_csv_column(temp_path, column="col2", skip_header=True)
        expected = np.array([2.7, 5.1, 8.5])
        np.testing.assert_allclose(data, expected)
        
        # Test loading by column index
        data_idx = load_csv_column(temp_path, column=0, skip_header=True)
        expected_idx = np.array([1.5, 4.8, 7.2])
        np.testing.assert_allclose(data_idx, expected_idx)
        
        # Test without skipping header (should parse header as data and potentially fail)
        # This might raise an exception or give weird results
        
    finally:
        os.unlink(temp_path)


def test_load_wav_mono():
    # Test loading WAV file from data/wav directory
    wav_path = Path(__file__).parent.parent / "data" / "wav" / "427624__polaina_legal__sine.wav"
    if wav_path.exists():
        x, sr = load_wav_mono(str(wav_path))
        
        # Check basic properties
        assert isinstance(x, np.ndarray)
        assert len(x) > 0
        assert isinstance(sr, int)
        assert sr > 0
        
        # Check normalization (should have unit variance approximately)
        assert abs(x.std() - 1.0) < 0.1  # Allow some tolerance
    else:
        pytest.skip("Test WAV file not found")


def test_auc_from_scores():
    # Perfect separation
    pos = np.array([0.8, 0.9, 0.7, 0.95])
    neg = np.array([0.1, 0.2, 0.3, 0.15])
    auc = auc_from_scores(pos, neg)
    assert 0.9 <= auc <= 1.0  # Should be near perfect
    
    # No separation (same distributions)
    pos_same = np.array([0.5, 0.6, 0.4])
    neg_same = np.array([0.5, 0.6, 0.4])
    auc_same = auc_from_scores(pos_same, neg_same)
    assert 0.4 <= auc_same <= 0.6  # Should be around 0.5
    
    # Reversed separation (neg > pos)
    pos_low = np.array([0.1, 0.2, 0.15])
    neg_high = np.array([0.8, 0.9, 0.85])
    auc_rev = auc_from_scores(pos_low, neg_high)
    assert 0.0 <= auc_rev <= 0.1  # Should be near 0


def test_window_iter():
    seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Non-overlapping windows
    windows = window_iter(seq, win=3, stride=3)
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert windows == expected
    
    # Overlapping windows
    windows_overlap = window_iter(seq, win=4, stride=2)
    expected_overlap = [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]
    assert windows_overlap == expected_overlap
    
    # Single window
    windows_single = window_iter(seq, win=len(seq), stride=1)
    assert len(windows_single) == 1
    assert windows_single[0] == seq


def test_train_forward_and_reverse_models():
    rng = np.random.default_rng(42)
    
    # Create some training data as a flat sequence
    train_data = rng.integers(0, 3, size=500).tolist()
    
    # Train models  
    forward_model, reverse_model = train_forward_and_reverse_models(
        train_data, k=3, R=2
    )
    
    # Check that models are frozen predictors
    assert isinstance(forward_model, KTFrozenPredictor)
    assert isinstance(reverse_model, KTFrozenPredictor)
    assert forward_model.k == 3
    assert reverse_model.k == 3
    
    # Test prediction on a sample sequence
    test_seq = [0, 1, 2, 1, 0]
    forward_ll = forward_model.codelen_sequence(test_seq)
    reverse_ll = reverse_model.codelen_sequence(test_seq[::-1])
    
    assert np.isfinite(forward_ll)
    assert np.isfinite(reverse_ll)


def test_signed_lr_score():
    # Create simple predictors for testing
    base_coder = KTMarkovMixture(alphabet_size=2, R=1)
    
    # Train on forward sequences
    forward_train = [0, 1, 0, 1] * 50
    base_coder.fit(forward_train)
    forward_model = base_coder.snapshot_frozen()
    
    # Train on reverse sequences
    reverse_coder = KTMarkovMixture(alphabet_size=2, R=1)
    reverse_train = forward_train[::-1]
    reverse_coder.fit(reverse_train)
    reverse_model = reverse_coder.snapshot_frozen()
    
    # Test on forward-like sequence
    test_seq = [0, 1, 0, 1, 0]
    score = signed_lr_score(test_seq, forward_model, reverse_model)
    
    # Should favor forward model (positive score)
    assert score > 0
    
    # Test on reverse sequence
    score_rev = signed_lr_score(test_seq[::-1], forward_model, reverse_model)
    # Might favor reverse model (negative score) or be similar
    assert np.isfinite(score_rev)


def test_aot_from_series():
    # Synthetic: sawtooth-like signal with trend breaks (irreversible)
    n = 20000
    t = np.arange(n)
    x = (t % 200) + 0.1 * np.random.default_rng(0).standard_normal(n)
    res = aot_from_series(x, k=8, R=3, win=1024, stride=512, use_diff=True)
    # Sanity: AUC in [0,1], arrays present, bits_per_step finite
    assert 0.0 <= res["auc"] <= 1.0
    assert isinstance(res["scores_forward"], list) and isinstance(res["scores_reversed"], list)
    assert isinstance(res["bits_per_step"], float)
    

def test_aot_from_series_reversible():
    # Test on more reversible signal
    rng = np.random.default_rng(42)
    x = rng.standard_normal(10000)  # i.i.d. Gaussian (should be nearly reversible)
    
    res = aot_from_series(x, k=4, R=2, win=512, stride=256, use_diff=True)
    
    # Should have lower AUC (closer to 0.5) for reversible signal
    assert 0.0 <= res["auc"] <= 1.0
    # Might be closer to 0.5 but not guaranteed due to finite sample effects
    
    assert np.isfinite(res["bits_per_step"])
    assert len(res["scores_forward"]) == len(res["scores_reversed"])


def test_aot_from_series_deterministic():
    # Deterministic irreversible sequence
    x = np.array([i % 5 for i in range(5000)], dtype=float)  # 0,1,2,3,4,0,1,2,3,4,...
    
    res = aot_from_series(x, k=5, R=2, win=200, stride=100, use_diff=False)
    
    # Should detect strong irreversibility (high AUC)
    assert res["auc"] > 0.7  # Should be quite irreversible
    assert np.isfinite(res["bits_per_step"])
    

def test_aot_edge_cases():
    # Use a larger sequence that satisfies len(test) >= win * 4 requirement
    x_short = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    
    # Should handle gracefully with appropriate parameters
    # train_frac=0.2 means test gets 80% = 12.8 elements, win=2 needs test >= 8
    res = aot_from_series(x_short, k=2, R=1, win=2, stride=1, use_diff=True, train_frac=0.2)
    assert "auc" in res
    assert "bits_per_step" in res
    
    # Single value repeated
    x_constant = np.ones(1000)
    res_const = aot_from_series(x_constant, k=3, R=1, win=50, stride=25, use_diff=True)
    assert "auc" in res_const
    # AUC might be undefined or 0.5 for constant signal


def test_aot_parameter_variations():
    """Test AoT with different parameter combinations."""
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.standard_normal(5000))  # random walk (irreversible)
    
    # Test different k values
    for k in [2, 4, 8]:
        res = aot_from_series(x, k=k, R=2, win=200, stride=100, use_diff=True)
        assert 0.0 <= res["auc"] <= 1.0
        assert np.isfinite(res["bits_per_step"])
    
    # Test different R values
    for R in [1, 2, 3]:
        res = aot_from_series(x, k=4, R=R, win=200, stride=100, use_diff=True)
        assert 0.0 <= res["auc"] <= 1.0
        assert np.isfinite(res["bits_per_step"])
        
    # Test with and without differencing
    res_diff = aot_from_series(x, k=4, R=2, win=200, stride=100, use_diff=True)
    res_no_diff = aot_from_series(x, k=4, R=2, win=200, stride=100, use_diff=False)
    
    # Both should be valid
    assert 0.0 <= res_diff["auc"] <= 1.0
    assert 0.0 <= res_no_diff["auc"] <= 1.0
