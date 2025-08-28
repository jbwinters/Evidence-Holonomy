#!/usr/bin/env python3
"""
Generate synthetic test videos ranging from reversible to irreversible
for testing the AoT video pipeline.

Creates several test videos demonstrating different levels of temporal irreversibility:
1. Reversible: Static pattern (AUC ≈ 0.5)
2. Semi-reversible: Periodic motion (AUC slightly > 0.5) 
3. Moderately irreversible: Biased random walk (AUC > 0.5)
4. Highly irreversible: Expanding/diffusing pattern (AUC >> 0.5)
5. Temporal gradient: Monotonic brightness change (very high AUC)
"""

import numpy as np
import os
from pathlib import Path

# Check for imageio
try:
    import imageio.v3 as iio
except ImportError:
    try:
        import imageio as iio
    except ImportError:
        print("Error: imageio not installed. Please install with: pip install imageio[ffmpeg]")
        exit(1)


def create_static_pattern_video(width=64, height=64, frames=60, fps=30):
    """Completely reversible: static checkerboard pattern."""
    video_data = []
    for t in range(frames):
        frame = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if (i // 4 + j // 4) % 2:
                    frame[i, j] = 255
                else:
                    frame[i, j] = 128
        video_data.append(frame)
    return video_data


def create_periodic_motion_video(width=64, height=64, frames=60, fps=30):
    """Semi-reversible: periodic motion (oscillating stripe)."""
    video_data = []
    for t in range(frames):
        frame = np.zeros((height, width), dtype=np.uint8)
        # Sine wave motion
        phase = 2 * np.pi * t / 20  # Period of 20 frames
        offset = int(10 * np.sin(phase))
        
        for i in range(height):
            for j in range(width):
                if (j + offset) % 12 < 6:
                    frame[i, j] = 255
                else:
                    frame[i, j] = 100
        video_data.append(frame)
    return video_data


def create_biased_random_walk_video(width=64, height=64, frames=60, fps=30, bias=0.6):
    """Moderately irreversible: biased random walk pattern."""
    rng = np.random.RandomState(42)
    
    # Start with random dot positions
    n_dots = 20
    positions = rng.rand(n_dots, 2) * np.array([width, height])
    velocities = rng.randn(n_dots, 2) * 0.1
    
    video_data = []
    for t in range(frames):
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Update positions with bias (drift to the right)
        bias_force = np.array([bias, 0]) * 0.5
        velocities += rng.randn(n_dots, 2) * 0.3 + bias_force
        positions += velocities
        
        # Bounce off walls
        for i in range(n_dots):
            if positions[i, 0] < 0 or positions[i, 0] >= width:
                velocities[i, 0] *= -0.8
                positions[i, 0] = np.clip(positions[i, 0], 0, width-1)
            if positions[i, 1] < 0 or positions[i, 1] >= height:
                velocities[i, 1] *= -0.8
                positions[i, 1] = np.clip(positions[i, 1], 0, height-1)
        
        # Draw dots with some spread
        for i in range(n_dots):
            x, y = int(positions[i, 0]), int(positions[i, 1])
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        intensity = 255 - 30 * (abs(dx) + abs(dy))
                        frame[ny, nx] = max(frame[ny, nx], intensity)
        
        video_data.append(frame)
    
    return video_data


def create_diffusion_video(width=64, height=64, frames=60, fps=30):
    """Highly irreversible: expanding/diffusing pattern."""
    video_data = []
    
    # Start with a central bright spot
    for t in range(frames):
        frame = np.zeros((height, width), dtype=np.float32)
        
        # Multiple expanding circles from center
        cx, cy = width // 2, height // 2
        radius = t * 1.5  # Expanding radius
        
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((j - cx)**2 + (i - cy)**2)
                # Create expanding ring pattern
                if radius - 5 < dist < radius + 5:
                    intensity = 255 * np.exp(-((dist - radius) / 3)**2)
                    frame[i, j] = max(frame[i, j], intensity)
                
                # Add some noise diffusion
                if dist < radius:
                    noise = np.random.RandomState(i*width + j + t).rand() * 50
                    frame[i, j] = max(frame[i, j], noise * (1 - dist/radius))
        
        video_data.append(frame.astype(np.uint8))
    
    return video_data


def create_temporal_gradient_video(width=64, height=64, frames=60, fps=30):
    """Very irreversible: monotonic temporal change."""
    video_data = []
    
    for t in range(frames):
        frame = np.zeros((height, width), dtype=np.uint8)
        
        # Monotonic brightness increase over time
        base_brightness = int(255 * t / (frames - 1))
        
        # Add some spatial structure
        for i in range(height):
            for j in range(width):
                # Spatial pattern
                spatial = int(50 * np.sin(2 * np.pi * i / 16) * np.cos(2 * np.pi * j / 16))
                
                # Temporal gradient
                temporal = base_brightness
                
                # Combine
                final = np.clip(temporal + spatial, 0, 255)
                frame[i, j] = final
        
        video_data.append(frame)
    
    return video_data


def save_videos(output_dir="test_videos"):
    """Generate and save all test videos."""
    Path(output_dir).mkdir(exist_ok=True)
    
    videos = {
        "static_reversible": create_static_pattern_video,
        "periodic_semi_reversible": create_periodic_motion_video,
        "biased_walk_moderate": create_biased_random_walk_video,
        "diffusion_irreversible": create_diffusion_video,
        "gradient_highly_irreversible": create_temporal_gradient_video,
    }
    
    print(f"Generating test videos in {output_dir}/...")
    
    for name, func in videos.items():
        print(f"Creating {name}.mp4...")
        frames = func()
        output_path = Path(output_dir) / f"{name}.mp4"
        
        try:
            iio.imwrite(output_path, frames, fps=30)
            print(f"  ✓ Saved {output_path} ({len(frames)} frames)")
        except Exception as e:
            print(f"  ✗ Error saving {output_path}: {e}")
    
    print("\nTest videos created. Expected AoT behavior:")
    print("- static_reversible: AUC ≈ 0.50 (perfectly reversible)")
    print("- periodic_semi_reversible: AUC ≈ 0.51-0.55 (weakly irreversible)")
    print("- biased_walk_moderate: AUC ≈ 0.60-0.70 (moderately irreversible)")
    print("- diffusion_irreversible: AUC ≈ 0.70-0.80 (highly irreversible)")
    print("- gradient_highly_irreversible: AUC ≈ 0.80+ (very irreversible)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate test videos for AoT analysis")
    parser.add_argument("--output_dir", "-o", default="test_videos", help="Output directory")
    args = parser.parse_args()
    
    save_videos(args.output_dir)