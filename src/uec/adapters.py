"""
Image and video tokenization adapters for AoT analysis.

Provides tokenizers that convert images and video into discrete sequences
for use with the existing AoT/holonomy pipeline. All functions return
(tokens, k, sr) tuples compatible with aot_from_series().

Images support:
- Raster scan tokenization (grayscale intensities)
- Patch-based vector quantization

Video support:
- Frame-level vector quantization
- Optional optical flow tokenization
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .aot import discretize_series

# Optional imports with graceful fallbacks
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import imageio.v3 as iio
except ImportError:
    try:
        import imageio as iio
    except ImportError:
        iio = None

try:
    from sklearn.cluster import KMeans
except (ImportError, ValueError):  # ValueError can occur with numpy version mismatches
    KMeans = None


def load_image_gray(path: str) -> np.ndarray:
    """Load image and convert to grayscale float array in [0,1]."""
    if Image is None:
        raise ImportError("Pillow not installed: pip install pillow")
    img = Image.open(path).convert("L")  # grayscale
    x = np.asarray(img, dtype=float) / 255.0
    return x  # shape (H, W)


def image_to_tokens_raster(x2d: np.ndarray, k: int) -> Tuple[List[int], int]:
    """Convert 2D image to 1D token sequence via raster scan (row-major order)."""
    x = x2d.ravel(order="C")  # row-major flatten
    s = discretize_series(x, k=k).astype(int)
    return s.tolist(), int(k)


def image_patches(x2d: np.ndarray, patch: int = 8) -> np.ndarray:
    """Extract non-overlapping patches from 2D image.
    
    Args:
        x2d: 2D image array
        patch: Patch size (patch x patch)
        
    Returns:
        Array of shape (num_patches, patch*patch)
    """
    H, W = x2d.shape
    # Trim to ensure divisibility by patch size
    Hc, Wc = H - (H % patch), W - (W % patch)
    x = x2d[:Hc, :Wc]
    # Reshape to patches
    x = x.reshape(Hc//patch, patch, Wc//patch, patch).swapaxes(1, 2).reshape(-1, patch*patch)
    return x


def fit_codebook_kmeans(X: np.ndarray, k_codebook: int, seed: int = 0) -> np.ndarray:
    """Fit k-means codebook on patch/frame vectors.
    
    Args:
        X: Array of vectors, shape (n_samples, n_features)
        k_codebook: Number of codebook entries
        seed: Random seed
        
    Returns:
        Codebook array of shape (k_codebook, n_features)
    """
    if KMeans is None:
        # Fallback to simple random initialization if sklearn not available
        rng = np.random.default_rng(seed)
        # Simple k-means++ style initialization
        n_samples, n_features = X.shape
        codebook = np.zeros((k_codebook, n_features))
        
        # First centroid: random sample
        idx = rng.integers(0, n_samples)
        codebook[0] = X[idx]
        
        # Subsequent centroids: farthest from existing
        for i in range(1, k_codebook):
            distances = np.inf * np.ones(n_samples)
            for j in range(n_samples):
                for k in range(i):
                    dist = np.sum((X[j] - codebook[k])**2)
                    distances[j] = min(distances[j], dist)
            
            # Sample proportional to squared distance
            total = distances.sum()
            if total > 0:
                probs = distances / total
                idx = rng.choice(n_samples, p=probs)
            else:
                idx = rng.integers(0, n_samples)
            codebook[i] = X[idx]
        
        # Simple Lloyd's algorithm iterations
        for _ in range(10):
            assignments = assign_codebook(X, codebook)
            new_codebook = np.zeros_like(codebook)
            for k in range(k_codebook):
                mask = assignments == k
                if mask.sum() > 0:
                    new_codebook[k] = X[mask].mean(axis=0)
                else:
                    new_codebook[k] = codebook[k]
            
            # Check for convergence
            if np.allclose(codebook, new_codebook, rtol=1e-6):
                break
            codebook = new_codebook
        
        return codebook
    else:
        km = KMeans(n_clusters=k_codebook, random_state=seed, n_init="auto")
        km.fit(X)
        return km.cluster_centers_


def assign_codebook(X: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Assign vectors to nearest codebook entries.
    
    Args:
        X: Array of vectors, shape (n_samples, n_features)
        codebook: Codebook array, shape (k_codebook, n_features)
        
    Returns:
        Integer assignments, shape (n_samples,)
    """
    # Compute squared distances using broadcasting
    dists = ((X[:, None, :] - codebook[None, :, :])**2).sum(axis=2)
    return dists.argmin(axis=1).astype(int)


def image_to_tokens_patch_vq(x2d: np.ndarray, k_codebook: int, patch: int = 8, 
                            codebook: Optional[np.ndarray] = None, seed: int = 0) -> Tuple[List[int], int, np.ndarray]:
    """Convert image to tokens using patch-based vector quantization.
    
    Args:
        x2d: 2D image array
        k_codebook: Number of codebook entries
        patch: Patch size (patch x patch)
        codebook: Pre-fitted codebook (if None, will fit on this image)
        seed: Random seed for codebook fitting
        
    Returns:
        (tokens, k, codebook) tuple
    """
    X = image_patches(x2d, patch=patch)
    
    if codebook is None:
        codebook = fit_codebook_kmeans(X, k_codebook, seed=seed)
    
    tokens = assign_codebook(X, codebook)
    return tokens.tolist(), int(k_codebook), codebook


def load_video_frames(path: str, stride: int = 1, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    """Load video frames as grayscale arrays.
    
    Args:
        path: Path to video file
        stride: Frame sampling stride
        max_frames: Maximum number of frames to load
        
    Returns:
        (frames, fps) tuple where frames is list of 2D arrays in [0,1]
    """
    if iio is None:
        raise ImportError("imageio not installed: pip install imageio")
    
    # Get metadata
    meta = iio.immeta(path)
    fps = float(meta.get("fps", 30.0))
    
    frames = []
    idx = 0
    
    for frame in iio.imiter(path):
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
        if max_frames and len(frames) >= max_frames:
            break
    
    # Convert to grayscale float in [0,1]
    processed_frames = []
    for f in frames:
        if f.ndim == 3:
            # RGB -> grayscale using standard weights
            fr = 0.2989*f[:,:,0] + 0.5870*f[:,:,1] + 0.1140*f[:,:,2]
        else:
            fr = f
        processed_frames.append(np.asarray(fr, dtype=float) / 255.0)
    
    return processed_frames, fps


def frames_to_tokens_vq(frames: List[np.ndarray], k_codebook: int, down: int = 32,
                       codebook: Optional[np.ndarray] = None, seed: int = 0) -> Tuple[List[int], int, np.ndarray]:
    """Convert video frames to tokens using frame-level vector quantization.
    
    Args:
        frames: List of 2D frame arrays
        k_codebook: Number of codebook entries  
        down: Downscale frames to down x down before vectorization
        codebook: Pre-fitted codebook (if None, will fit on these frames)
        seed: Random seed for codebook fitting
        
    Returns:
        (tokens, k, codebook) tuple
    """
    # Downscale frames and flatten
    frame_vectors = []
    for fr in frames:
        H, W = fr.shape
        # Simple downsampling by averaging non-overlapping blocks
        dH, dW = H // down, W // down
        if dH == 0 or dW == 0:
            # Frame too small, use simple resize
            try:
                from scipy.ndimage import zoom
                ds = zoom(fr, (down/H, down/W))
            except (ImportError, ValueError):
                # Fallback: simple nearest neighbor
                ds = fr[::max(1, H//down), ::max(1, W//down)]
                if ds.shape[0] < down:
                    ds = np.repeat(ds, (down + ds.shape[0] - 1) // ds.shape[0], axis=0)[:down]
                if ds.shape[1] < down:
                    ds = np.repeat(ds, (down + ds.shape[1] - 1) // ds.shape[1], axis=1)[:, :down]
        else:
            # Block averaging
            fr_trim = fr[:dH*down, :dW*down]
            ds = fr_trim.reshape(dH, down, dW, down).mean(axis=(1, 3))
        
        frame_vectors.append(ds.ravel())
    
    X = np.vstack(frame_vectors)
    
    if codebook is None:
        codebook = fit_codebook_kmeans(X, k_codebook, seed=seed)
    
    tokens = assign_codebook(X, codebook)
    return tokens.tolist(), int(k_codebook), codebook


def video_to_tokens_vq(path: str, k_codebook: int = 256, down: int = 32, stride: int = 1,
                      max_frames: Optional[int] = None, codebook: Optional[np.ndarray] = None, 
                      seed: int = 0) -> Tuple[List[int], int, float, np.ndarray]:
    """Complete pipeline: load video and convert to VQ tokens.
    
    Args:
        path: Path to video file
        k_codebook: Number of codebook entries
        down: Downscale frames to down x down
        stride: Frame sampling stride
        max_frames: Maximum number of frames
        codebook: Pre-fitted codebook
        seed: Random seed
        
    Returns:
        (tokens, k, fps, codebook) tuple
    """
    frames, fps = load_video_frames(path, stride=stride, max_frames=max_frames)
    tokens, k, codebook = frames_to_tokens_vq(frames, k_codebook, down=down, codebook=codebook, seed=seed)
    return tokens, k, fps, codebook


def save_codebook(codebook: np.ndarray, path: str) -> None:
    """Save codebook to file."""
    np.savez_compressed(path, codebook=codebook)


def load_codebook(path: str) -> np.ndarray:
    """Load codebook from file."""
    data = np.load(path)
    return data['codebook']