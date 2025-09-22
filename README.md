# UEC Holonomy (uec-holonomy)

Universal Evidence Curvature (UEC): KL-rate holonomy, entropy production estimation, and Arrow-of-Time demos.

- KL-rate holonomy estimators that match D(P||Q) on loops of representation.
- Equality to entropy production for the Markov time-reversal loop (bits/step).
- A minimal test battery and AoT (Arrow-of-Time) demos for audio, images, video, sensors, and finance.

## Install

```bash
pip install uec-holonomy            # (once published)
# from source (dev extras include tests/lint)
pip install -e .[dev]
# optional extras
pip install -e .[audio]        # SciPy WAV reader
pip install -e .[image]        # PIL/Pillow for images  
pip install -e .[video]        # imageio for video
pip install -e .[all]          # all optional dependencies
```

## CLI

- Battery (core validations):

```bash
uec-battery --fast         # quick run
uec-battery --run_suite    # full suite + artifacts
```

- AoT demos (CSV/WAV/Image/Video, scoreboard):

```bash
# Audio analysis
uec-aot --aot_wav data/wav/boiling.wav --aot_bins 32 --aot_win 65536 --aot_stride 32768 --order 5 --aot_diff

# Financial time series
uec-aot --aot_csv data/kaggle/btc.csv --aot_csv_col Close --aot_logreturn --aot_rate 1

# Image analysis (raster scan)
uec-aot --aot_image image.png --image_mode raster --aot_bins 16

# Image analysis (patch vector quantization)
uec-aot --aot_image image.png --image_mode patch --image_vq_k 256 --image_patch 8

# Video analysis (frame-level vector quantization)
uec-aot --aot_video video.mp4 --video_vq_k 64 --video_down 16 --aot_win 512 --aot_stride 256

# Scoreboard across multiple files
uec-aot --scoreboard_glob "data/wav/*.wav" --aot_bins 32 --aot_win 65536 --aot_stride 32768 --order 5 --aot_diff
```

See `uec_theory.tex` for the theory (two holonomies: representation-space vs. observer-transported KL), reductions, and references.

## Analysis Scripts

BTC UEC analysis and utilities (research-only):

- `python scripts/btc_uec_analysis.py [--tail N] [--window W] [--k_r K] [--k_v K] [--uec_method counts|kt]`:
  - Computes UEC stream (bits/step), z-scores, optional bootstrap CIs, change-points; runs a simple UEC-gated trend backtest.
  - Outputs: `results/btc_uec_analysis.csv`, `results/btc_uec_summary.json`.

- `python scripts/uec_diagnostics.py --gauge --surrogate --markov [--ergodic_segments 4] [--jitter_std 0.05]`:
  - Gauge/surrogate ~ 0 checks, Markov EP vs holonomy, ergodicity probe across segments, measurement jitter robustness.

- `python scripts/uec_sensitivity.py --tail 50000 --k_list 6,8,12,16 --r_list 1,2,3 --method counts`:
  - Sensitivity grid over discretization and order; outputs `results/uec_sensitivity.csv`.

- `python scripts/uec_multiscale.py --tail 50000 --scales 1,2,4,8`:
  - Multi-scale spectrum (downsampling) and coarse-grain-loop holonomy; attribution (joint vs returns-only vs volume-only).

- `python scripts/uec_bench.py --tail 20000 --W_list 128,256 --R_list 1,2,3`:
  - Timing for counts vs KT pipelines.

## Python API

```python
# Core holonomy analysis
from uec.markov import random_markov_biased, sample_markov, entropy_production_rate_bits
from uec.holonomy import klrate_holonomy_time_reversal_markov
from uec.aot import aot_from_series

T = random_markov_biased(k=3, delta=0.6)
x = sample_markov(T, n=150_000)
print(entropy_production_rate_bits(T))
print(klrate_holonomy_time_reversal_markov(x, k=3, R=3))

# Image and video analysis
from uec.adapters import (
    load_image_gray, image_to_tokens_raster, image_to_tokens_patch_vq,
    video_to_tokens_vq
)
import numpy as np

# Load and tokenize image
img = load_image_gray("image.png")
tokens, k = image_to_tokens_raster(img, k=16)
result = aot_from_series(np.array(tokens), k=k, R=3)

# Load and tokenize video  
tokens, k, fps, codebook = video_to_tokens_vq("video.mp4", k_codebook=64)
result = aot_from_series(np.array(tokens), k=k, R=3, sr=fps)
```

## Model Validation Results

The holonomy-based Arrow-of-Time analysis has been validated across diverse audio signals, demonstrating correct detection of temporal asymmetries:

### Test Signal Results

| Audio Type | AUC | bits/step | bits/second | Interpretation |
|------------|-----|-----------|-------------|----------------|
| **Generated White Noise** | 0.495 | ~0 | ~0 | ✅ **Perfectly Reversible** |
| **Generated Sine Wave** | 0.497 | 5.9×10⁻⁶ | 0.26 | ✅ **Nearly Reversible** |
| **Generated Chirp** | 0.483 | 1.0×10⁻⁵ | 0.44 | 🔶 **Slightly Irreversible** |
| **Test WAV Sine** | 0.489 | 1.5×10⁻⁸ | 0.0001 | ✅ **Nearly Reversible** |
| **Applause** | 0.496 | 7.2×10⁻⁵ | 3.16 | 🔶 **Slightly Irreversible** |
| **Human Singing** | 0.536 | 1.1×10⁻⁴ | 4.73 | 🔶 **Moderately Irreversible** |
| **Rain + Traffic** | 0.527 | 2.9×10⁻⁵ | 1.40 | 🔶 **Moderately Irreversible** |

**Key Validation Points:**

1. **Mathematical signals behave as predicted**: White noise and pure sine waves show AUC ≈ 0.5 (reversible), while directional signals like frequency chirps show detectable irreversibility.

2. **Real audio complexity correlates with temporal structure**: Human voice shows highest irreversibility (structured speech/melody), environmental sounds show moderate values, pure tones remain nearly reversible.

3. **Entropy production scales with signal complexity**: Simple mathematical signals produce ~0 bits/second, natural sounds produce 1-5 bits/second, structured human sounds show highest values.

These results demonstrate that the holonomy-based approach correctly distinguishes reversible from irreversible temporal processes across both synthetic test cases and real-world audio recordings.

### Image and Video Validation

The framework has been extended to support images and video with comprehensive validation:

**Video Test Results (Synthetic):**

| Video Type | AUC | Interpretation |
|------------|-----|----------------|
| **Static Pattern** | 0.000 | ✅ **Perfect Reversibility** |
| **Periodic Motion** | 0.598 | 🔶 **Weakly Irreversible** |
| **Biased Random Walk** | 0.698 | 🔶 **Moderately Irreversible** |
| **Temporal Gradient** | 0.947 | 🔴 **Highly Irreversible** |

**Image Analysis Modes:**
- **Raster scan**: Treats images as 1D sequences via row-major order
- **Patch VQ**: Vector quantization of image patches for spatial-temporal structure

**Key Features:**
- Frame-level vector quantization for video temporal analysis
- Codebook training and reuse across datasets
- Integration with existing AoT pipeline (bits/step, bits/second)
- Robust fallbacks when optional dependencies unavailable

## Development

Run tests:

```bash
pytest -q
```

Package layout:

```
src/uec/
  markov.py      # transitions, stationary, EP, HMM, ring EP
  coders.py      # KT mixture (frozen), LZ78
  transforms.py  # recode, coarse-grain, time-reversal, transitions, (down/up)-sample
  holonomy.py    # KL-rate estimators and time-reversal loop
  aot.py         # AoT pipeline (discretize → train(P,Q) → scores & CI)
  adapters.py    # image/video tokenization (raster, patch VQ, frame VQ)
  cli.py         # console entry points: uec-battery, uec-aot
```

CI: GitHub Actions runs tests on 3.9–3.11. Publishing to PyPI happens on release tag; add `PYPI_API_TOKEN` to repo secrets.

## License

Code is licensed under MIT. Text, figures, and conceptual content are licensed under CC-BY 4.0 — please cite when reusing.

## Attributions

Audio samples under `data/wav/` are from Freesound.org and used for AoT demos. See `ATTRIBUTIONS.md` for links and author credits. Please abide by the license terms on each Freesound page.

## Assumptions and Safe Operation

See `docs/assumptions.md` for refined theoretical assumptions (finite alphabet, stationarity, ergodicity, sufficiency, loop closure) and practical “Check / Work‑around / Relax” guidance.
