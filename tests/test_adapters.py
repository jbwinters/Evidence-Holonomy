"""
Tests for image and video adapters.
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from src.uec.adapters import (
    load_image_gray, image_to_tokens_raster, image_patches,
    fit_codebook_kmeans, assign_codebook, image_to_tokens_patch_vq,
    load_video_frames, frames_to_tokens_vq, video_to_tokens_vq,
    save_codebook, load_codebook
)

# Optional dependency handling
PIL = pytest.importorskip("PIL", reason="PIL/Pillow not available")
imageio = pytest.importorskip("imageio", reason="imageio not available")


def create_test_image(width=64, height=64):
    """Create a simple test image with PIL."""
    from PIL import Image
    # Create a simple gradient image
    img_array = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            img_array[i, j] = (i + j) % 256
    return Image.fromarray(img_array, mode='L')


def create_test_video(width=32, height=32, frames=10, fps=10):
    """Create a simple test video with imageio."""
    video_data = []
    for t in range(frames):
        # Create a moving pattern
        frame = np.zeros((height, width), dtype=np.uint8)
        # Moving diagonal stripe
        for i in range(height):
            for j in range(width):
                if (i + j + t) % 8 < 4:
                    frame[i, j] = 255
                else:
                    frame[i, j] = 100
        video_data.append(frame)
    return video_data, fps


class TestImageAdapters:
    
    def test_load_image_gray(self):
        """Test loading an image and converting to grayscale."""
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img = create_test_image(32, 32)
            img.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            gray = load_image_gray(tmp_path)
            assert gray.shape == (32, 32)
            assert gray.dtype == float
            assert 0.0 <= gray.min() <= gray.max() <= 1.0
        finally:
            os.unlink(tmp_path)
    
    def test_image_to_tokens_raster(self):
        """Test raster scan tokenization."""
        # Create simple 4x4 image
        img = np.array([[0.0, 0.25, 0.5, 0.75],
                       [0.1, 0.35, 0.6, 0.85], 
                       [0.2, 0.45, 0.7, 0.95],
                       [0.3, 0.55, 0.8, 1.0]])
        
        tokens, k = image_to_tokens_raster(img, k=4)
        
        assert len(tokens) == 16  # 4x4 = 16 pixels
        assert k == 4
        assert all(0 <= t < k for t in tokens)
        assert isinstance(tokens, list)
        assert all(isinstance(t, (int, np.integer)) for t in tokens)
    
    def test_image_patches(self):
        """Test patch extraction."""
        # 8x8 image, 4x4 patches -> 4 patches
        img = np.random.rand(8, 8)
        patches = image_patches(img, patch=4)
        
        assert patches.shape == (4, 16)  # 4 patches, each 4x4=16 elements
        
        # Test with non-divisible size
        img = np.random.rand(10, 10)
        patches = image_patches(img, patch=4)
        assert patches.shape == (4, 16)  # Should trim to 8x8 -> 4 patches
    
    def test_codebook_operations(self):
        """Test k-means codebook fitting and assignment."""
        # Generate some test data
        np.random.seed(42)
        X = np.random.randn(100, 8)  # 100 samples, 8 dimensions
        k_codebook = 16
        
        # Test fitting
        codebook = fit_codebook_kmeans(X, k_codebook, seed=42)
        assert codebook.shape == (k_codebook, 8)
        
        # Test assignment
        assignments = assign_codebook(X, codebook)
        assert assignments.shape == (100,)
        assert assignments.min() >= 0
        assert assignments.max() < k_codebook
        assert assignments.dtype in [int, np.int32, np.int64]
    
    def test_image_patch_vq_tokenization(self):
        """Test patch-based VQ tokenization."""
        # Create test image
        img = np.random.rand(16, 16)
        
        tokens, k, codebook = image_to_tokens_patch_vq(
            img, k_codebook=8, patch=4, seed=42
        )
        
        assert len(tokens) == 16  # 16x16 with 4x4 patches = 4x4 = 16 patches
        assert k == 8
        assert all(0 <= t < k for t in tokens)
        assert codebook.shape == (8, 16)  # 8 centroids, 16 features (4x4)
        
        # Test with existing codebook
        tokens2, k2, _ = image_to_tokens_patch_vq(
            img, k_codebook=8, patch=4, codebook=codebook
        )
        assert k2 == k
        assert len(tokens2) == len(tokens)
    
    def test_codebook_save_load(self):
        """Test saving and loading codebooks."""
        codebook = np.random.rand(16, 8)
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            save_codebook(codebook, tmp_path)
            loaded = load_codebook(tmp_path)
            np.testing.assert_array_equal(codebook, loaded)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestVideoAdapters:
    
    def test_frames_to_tokens_vq(self):
        """Test frame-level VQ tokenization."""
        # Create simple test frames
        frames = []
        for i in range(5):
            frame = np.ones((16, 16)) * (i / 4.0)  # Varying intensity
            frames.append(frame)
        
        tokens, k, codebook = frames_to_tokens_vq(
            frames, k_codebook=8, down=4, seed=42
        )
        
        assert len(tokens) == 5  # 5 frames
        assert k == 8
        assert all(0 <= t < k for t in tokens)
        assert codebook.shape == (8, 16)  # 8 centroids, 16 features (4x4)
    
    def test_video_pipeline_synthetic(self):
        """Test complete video pipeline with synthetic data."""
        # Create synthetic video
        video_data, fps = create_test_video(16, 16, 10, 15)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save test video
            imageio.mimsave(tmp_path, video_data, fps=fps)
            
            # Test loading
            frames, loaded_fps = load_video_frames(tmp_path, stride=1, max_frames=5)
            assert len(frames) == 5
            assert abs(loaded_fps - fps) < 1.0  # Allow some tolerance
            assert all(f.shape == (16, 16) for f in frames)
            assert all(f.dtype == float for f in frames)
            assert all(0.0 <= f.min() <= f.max() <= 1.0 for f in frames)
            
            # Test complete pipeline
            tokens, k, fps_ret, codebook = video_to_tokens_vq(
                tmp_path, k_codebook=8, down=4, stride=2, max_frames=5, seed=42
            )
            
            assert len(tokens) == 5  # max_frames=5, so 5 frames loaded
            assert k == 8
            assert all(0 <= t < k for t in tokens)
            assert abs(fps_ret - fps) < 1.0
            assert codebook.shape == (8, 16)  # 8 centroids, 4x4=16 features
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestEdgeCases:
    
    def test_small_images(self):
        """Test with very small images."""
        # 2x2 image
        img = np.array([[0.0, 0.5], [0.25, 0.75]])
        tokens, k = image_to_tokens_raster(img, k=2)
        assert len(tokens) == 4
        assert k == 2
    
    def test_single_frame_video(self):
        """Test with single frame 'video'."""
        frame = np.random.rand(8, 8)
        frames = [frame]
        
        tokens, k, codebook = frames_to_tokens_vq(frames, k_codebook=4, down=2)
        assert len(tokens) == 1
        assert k == 4
        assert codebook.shape == (4, 16)  # down=2 means we get more features than expected
    
    def test_deterministic_behavior(self):
        """Test that same inputs with same seeds give same outputs."""
        img = np.random.RandomState(123).rand(16, 16)
        
        tokens1, _, cb1 = image_to_tokens_patch_vq(img, k_codebook=8, patch=4, seed=42)
        tokens2, _, cb2 = image_to_tokens_patch_vq(img, k_codebook=8, patch=4, seed=42)
        
        assert tokens1 == tokens2
        np.testing.assert_array_equal(cb1, cb2)


if __name__ == "__main__":
    pytest.main([__file__])