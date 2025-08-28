#!/usr/bin/env python3
"""Generate test WAV files for AoT analysis validation."""

import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

def generate_white_noise_wav(duration_seconds=10.0, sample_rate=44100, amplitude=0.5, seed=42):
    """Generate white noise WAV file."""
    n_samples = int(duration_seconds * sample_rate)
    
    # Generate white noise
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_samples) * amplitude
    
    # Convert to 16-bit PCM
    noise_int16 = (noise * 32767).astype(np.int16)
    
    return noise_int16, sample_rate

def generate_sine_wave_wav(duration_seconds=10.0, sample_rate=44100, frequency=440.0, amplitude=0.5):
    """Generate sine wave WAV file."""
    n_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, n_samples, endpoint=False)
    
    # Generate sine wave
    sine = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    sine_int16 = (sine * 32767).astype(np.int16)
    
    return sine_int16, sample_rate

def generate_chirp_wav(duration_seconds=10.0, sample_rate=44100, f0=100, f1=2000, amplitude=0.5):
    """Generate chirp (frequency sweep) WAV file."""
    n_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, n_samples, endpoint=False)
    
    # Generate chirp: frequency increases linearly from f0 to f1
    chirp = amplitude * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration_seconds) * t)
    
    # Convert to 16-bit PCM
    chirp_int16 = (chirp * 32767).astype(np.int16)
    
    return chirp_int16, sample_rate

def main():
    """Generate test WAV files for model validation."""
    output_dir = Path("data/wav")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating test WAV files for Arrow-of-Time validation...")
    
    # Generate white noise (should be reversible, AUC ≈ 0.5)
    print("  • White noise (20s, 44.1kHz)")
    noise, sr = generate_white_noise_wav(duration_seconds=20.0)
    noise_path = output_dir / "test_white_noise_20sec.wav"
    wavfile.write(noise_path, sr, noise)
    
    # Generate sine wave (should be reversible)
    print("  • Sine wave 440Hz (20s, 44.1kHz)")
    sine, sr = generate_sine_wave_wav(duration_seconds=20.0, frequency=440.0)
    sine_path = output_dir / "test_sine_440hz_20sec.wav"
    wavfile.write(sine_path, sr, sine)
    
    # Generate chirp (frequency sweep - should be irreversible)
    print("  • Chirp 100-2000Hz (20s, 44.1kHz)")
    chirp, sr = generate_chirp_wav(duration_seconds=20.0, f0=100, f1=2000)
    chirp_path = output_dir / "test_chirp_100_2000hz_20sec.wav"
    wavfile.write(chirp_path, sr, chirp)
    
    # Generate short test file for quick validation
    print("  • White noise (2s, 8kHz) for quick tests")
    quick_noise, quick_sr = generate_white_noise_wav(duration_seconds=2.0, sample_rate=8000)
    quick_path = output_dir / "test_white_noise_2sec_8khz.wav"
    wavfile.write(quick_path, quick_sr, quick_noise)
    
    print(f"\nGenerated 4 test WAV files in {output_dir}/")
    print("\nThese files validate the holonomy-based Arrow-of-Time analysis:")
    print("  • White noise:    AUC ≈ 0.5 (reversible)")
    print("  • Sine wave:      AUC ≈ 0.5 (periodic → reversible)")
    print("  • Chirp:          AUC < 0.5 (directional → irreversible)")
    print("\nTo test: python -m uec.cli run_aot --aot_wav data/wav/test_white_noise_20sec.wav")

if __name__ == "__main__":
    main()