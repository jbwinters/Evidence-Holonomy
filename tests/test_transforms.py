import numpy as np
import pytest
from uec.transforms import (
    Transform,
    Permute,
    MergeSymbols,
    TimeReverse,
    TransitionEncode,
    TransitionDecodeTakeSecond,
    Downsample,
    UpsampleRepeat,
    apply_loop,
)


class TestTransform:
    def test_base_transform_interface(self):
        # Transform is abstract, but we can test its interface
        transform = Transform()
        
        # Should have apply method (even if not implemented)
        with pytest.raises(NotImplementedError):
            transform.apply([1, 2, 3], [0, 1, 2])


class TestPermute:
    def test_permute_basic(self):
        perm = [2, 0, 1]  # 0->2, 1->0, 2->1
        transform = Permute(perm)
        
        seq = [0, 1, 2, 0, 1, 2]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        expected_seq = [2, 0, 1, 2, 0, 1]
        assert result_seq == expected_seq
        assert result_alphabet == [0, 1, 2]  # alphabet unchanged
        
    def test_permute_identity(self):
        identity_perm = [0, 1, 2]
        transform = Permute(identity_perm)
        
        seq = [0, 1, 2, 1, 0]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        assert result_seq == seq
        assert result_alphabet == alphabet
        
    def test_permute_inverse_property(self):
        perm = [1, 2, 0]
        inv_perm = [2, 0, 1]  # inverse of perm
        
        transform1 = Permute(perm)
        transform2 = Permute(inv_perm)
        
        seq = [0, 1, 2, 0, 1]
        alphabet = [0, 1, 2]
        
        # Apply permutation then inverse
        temp_seq, temp_alphabet = transform1.apply(seq, alphabet)
        result_seq, result_alphabet = transform2.apply(temp_seq, temp_alphabet)
        
        # Should get back original sequence
        assert result_seq == seq
        assert result_alphabet == alphabet


class TestMergeSymbols:
    def test_merge_basic(self):
        merge_map = {0: 0, 1: 0, 2: 1}  # merge 0,1 -> 0; 2 -> 1
        transform = MergeSymbols(merge_map)
        
        seq = [0, 1, 2, 0, 1, 2]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        expected_seq = [0, 0, 1, 0, 0, 1]
        expected_alphabet = [0, 1]
        
        assert result_seq == expected_seq
        assert set(result_alphabet) == set(expected_alphabet)
        
    def test_merge_identity(self):
        identity_map = {0: 0, 1: 1, 2: 2}
        transform = MergeSymbols(identity_map)
        
        seq = [0, 1, 2, 1, 0]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        assert result_seq == seq
        assert set(result_alphabet) == set(alphabet)


class TestTimeReverse:
    def test_time_reverse_basic(self):
        transform = TimeReverse()
        
        seq = [0, 1, 2, 1, 0]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        expected_seq = [0, 1, 2, 1, 0]  # reversed
        assert result_seq == expected_seq
        assert result_alphabet == alphabet
        
    def test_time_reverse_double_application(self):
        transform = TimeReverse()
        
        seq = [0, 1, 2, 3, 4]
        alphabet = [0, 1, 2, 3, 4]
        
        # Apply twice should give original sequence
        temp_seq, temp_alphabet = transform.apply(seq, alphabet)
        result_seq, result_alphabet = transform.apply(temp_seq, temp_alphabet)
        
        assert result_seq == seq
        assert result_alphabet == alphabet


class TestTransitionEncode:
    def test_transition_encode_basic(self):
        transform = TransitionEncode(k=3)
        
        seq = [0, 1, 2, 1, 0]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        # Should encode transitions as symbols 0..k²-1
        assert len(result_seq) == len(seq) - 1  # one less transition than symbols
        assert all(0 <= s < 9 for s in result_seq)  # transitions in [0, k²-1]
        assert len(result_alphabet) == 9  # k² transitions
        
        # Check specific transitions
        # 0->1 should be 0*3+1=1, 1->2 should be 1*3+2=5, etc.
        expected_seq = [1, 5, 7, 3]  # (0,1), (1,2), (2,1), (1,0)
        assert result_seq == expected_seq


class TestTransitionDecodeTakeSecond:
    def test_transition_decode_basic(self):
        transform = TransitionDecodeTakeSecond(k=3)
        
        # Encoded transitions: (0,1), (1,2), (2,0)
        seq = [1, 5, 6]  # 0*3+1, 1*3+2, 2*3+0
        alphabet = list(range(9))  # k² transitions
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        # Should decode to second element of each transition
        expected_seq = [1, 2, 0]
        assert result_seq == expected_seq
        assert result_alphabet == [0, 1, 2]
        
    def test_encode_decode_roundtrip(self):
        encoder = TransitionEncode(k=2)
        decoder = TransitionDecodeTakeSecond(k=2)
        
        seq = [0, 1, 0, 1, 0]
        alphabet = [0, 1]
        
        # Encode then decode
        encoded_seq, encoded_alphabet = encoder.apply(seq, alphabet)
        decoded_seq, decoded_alphabet = decoder.apply(encoded_seq, encoded_alphabet)
        
        # Should get back all but first symbol
        assert decoded_seq == seq[1:]
        assert decoded_alphabet == alphabet


class TestDownsample:
    def test_downsample_basic(self):
        transform = Downsample(step=2)
        
        seq = [0, 1, 2, 3, 4, 5]
        alphabet = [0, 1, 2, 3, 4, 5]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        # Should take every 2nd element starting from 0
        expected_seq = [0, 2, 4]
        assert result_seq == expected_seq
        assert result_alphabet == alphabet  # alphabet unchanged
        
    def test_downsample_step_three(self):
        transform = Downsample(step=3)
        
        seq = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        expected_seq = [0, 3, 6]
        assert result_seq == expected_seq


class TestUpsampleRepeat:
    def test_upsample_basic(self):
        transform = UpsampleRepeat(step=2)
        
        seq = [0, 1, 2]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        # Should repeat each element step times
        expected_seq = [0, 0, 1, 1, 2, 2]
        assert result_seq == expected_seq
        assert result_alphabet == alphabet
        
    def test_upsample_step_three(self):
        transform = UpsampleRepeat(step=3)
        
        seq = [1, 2]
        alphabet = [0, 1, 2]
        
        result_seq, result_alphabet = transform.apply(seq, alphabet)
        
        expected_seq = [1, 1, 1, 2, 2, 2]
        assert result_seq == expected_seq
        
    def test_downsample_upsample_roundtrip(self):
        down = Downsample(step=2)
        up = UpsampleRepeat(step=2)
        
        # Start with even-length sequence
        seq = [0, 1, 2, 3, 4, 5]
        alphabet = [0, 1, 2, 3, 4, 5]
        
        # Downsample then upsample
        down_seq, down_alphabet = down.apply(seq, alphabet)
        up_seq, up_alphabet = up.apply(down_seq, down_alphabet)
        
        # Should get back every element twice
        expected_seq = [0, 0, 2, 2, 4, 4]
        assert up_seq == expected_seq


class TestApplyLoop:
    def test_apply_loop_empty(self):
        seq = [0, 1, 2, 1, 0]
        alphabet = [0, 1, 2]
        transforms = []
        
        result_seq, result_alphabet = apply_loop(seq, alphabet, transforms)
        
        assert result_seq == seq
        assert result_alphabet == alphabet
        
    def test_apply_loop_single_transform(self):
        seq = [0, 1, 2, 1, 0]
        alphabet = [0, 1, 2]
        transforms = [TimeReverse()]
        
        result_seq, result_alphabet = apply_loop(seq, alphabet, transforms)
        
        # Should be equivalent to applying TimeReverse directly
        expected_seq = [0, 1, 2, 1, 0]
        assert result_seq == expected_seq
        
    def test_apply_loop_multiple_transforms(self):
        seq = [0, 1, 2, 0, 1]
        alphabet = [0, 1, 2]
        
        # Permute then time reverse
        perm = [2, 0, 1]  # 0->2, 1->0, 2->1
        transforms = [Permute(perm), TimeReverse()]
        
        result_seq, result_alphabet = apply_loop(seq, alphabet, transforms)
        
        # First permute: [2, 0, 1, 2, 0]
        # Then reverse: [0, 2, 1, 0, 2]
        expected_seq = [0, 2, 1, 0, 2]
        assert result_seq == expected_seq
        
    def test_apply_loop_gauge_transformation(self):
        seq = [0, 1, 2, 0, 1]
        alphabet = [0, 1, 2]
        
        # Permutation followed by its inverse should be identity
        perm = [1, 2, 0]
        inv_perm = [2, 0, 1]
        transforms = [Permute(perm), Permute(inv_perm)]
        
        result_seq, result_alphabet = apply_loop(seq, alphabet, transforms)
        
        # Should get back original sequence (gauge transformation)
        assert result_seq == seq
        assert result_alphabet == alphabet
        
    def test_apply_loop_coarse_grain(self):
        seq = [0, 1, 0, 1, 0, 1, 0, 1]
        alphabet = [0, 1]
        
        # Downsample by 2, then upsample by 2
        transforms = [Downsample(step=2), UpsampleRepeat(step=2)]
        
        result_seq, result_alphabet = apply_loop(seq, alphabet, transforms)
        
        # Downsample: [0, 0, 0, 0]
        # Upsample: [0, 0, 0, 0, 0, 0, 0, 0]
        expected_seq = [0, 0, 0, 0, 0, 0, 0, 0]
        assert result_seq == expected_seq
        
    def test_apply_loop_transition_encoding_loop(self):
        seq = [0, 1, 0, 1, 0]
        alphabet = [0, 1]
        k = 2
        
        # Encode transitions, reverse, then decode
        transforms = [
            TransitionEncode(k),
            TimeReverse(),
            TransitionDecodeTakeSecond(k)
        ]
        
        result_seq, result_alphabet = apply_loop(seq, alphabet, transforms)
        
        # Should be well-defined and finite
        assert len(result_seq) > 0
        assert all(s in [0, 1] for s in result_seq)
        assert result_alphabet == alphabet


def test_transforms_composition():
    """Test that transforms can be composed in various ways."""
    seq = np.random.default_rng(42).integers(0, 3, size=100).tolist()
    alphabet = [0, 1, 2]
    
    # Test various combinations
    transform_sets = [
        [Permute([1, 2, 0])],
        [TimeReverse()],
        [MergeSymbols({0: 0, 1: 0, 2: 1})],
        [Downsample(2), UpsampleRepeat(2)],
        [Permute([2, 0, 1]), Permute([1, 2, 0])],  # composition
    ]
    
    for transforms in transform_sets:
        try:
            result_seq, result_alphabet = apply_loop(seq, alphabet, transforms)
            # Should produce valid results
            assert isinstance(result_seq, list)
            assert isinstance(result_alphabet, list)
            assert all(isinstance(s, int) for s in result_seq)
        except Exception as e:
            pytest.fail(f"Transform composition failed: {transforms}, error: {e}")