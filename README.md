# UEC Holonomy (uec-holonomy)

Universal Evidence Curvature (UEC): KL-rate holonomy, entropy production estimation, and Arrow-of-Time demos.

- KL-rate holonomy estimators that match D(P||Q) on loops of representation.
- Equality to entropy production for the Markov time-reversal loop (bits/step).
- A minimal test battery and AoT (Arrow-of-Time) demos for audio, sensors, finance.

## Install

```bash
pip install uec-holonomy            # (once published)
# from source (dev extras include tests/lint)
pip install -e .[dev]
# audio extras (optional, for SciPy WAV reader)
pip install -e .[audio]
```

## CLI

- Battery (core validations):

```bash
uec-battery --fast         # quick run
uec-battery --run_suite    # full suite + artifacts
```

- AoT demos (CSV/WAV, scoreboard):

```bash
uec-aot --aot_wav data/wav/boiling.wav --aot_bins 32 --aot_win 65536 --aot_stride 32768 --order 5 --aot_diff
uec-aot --aot_csv data/kaggle/btc.csv --aot_csv_col Close --aot_logreturn --aot_rate 1
uec-aot --scoreboard_glob "data/wav/*.wav" --aot_bins 32 --aot_win 65536 --aot_stride 32768 --order 5 --aot_diff
```

See `uec_theory.tex` for the theory (two holonomies: representation-space vs. observer-transported KL), reductions, and references.

## Python API

```python
from uec.markov import random_markov_biased, sample_markov, entropy_production_rate_bits
from uec.holonomy import klrate_holonomy_time_reversal_markov

T = random_markov_biased(k=3, delta=0.6)
x = sample_markov(T, n=150_000)
print(entropy_production_rate_bits(T))
print(klrate_holonomy_time_reversal_markov(x, k=3, R=3))
```

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
  cli.py         # console entry points: uec-battery, uec-aot
```

CI: GitHub Actions runs tests on 3.9–3.11. Publishing to PyPI happens on release tag; add `PYPI_API_TOKEN` to repo secrets.

## License

Code is licensed under MIT. Text, figures, and conceptual content are licensed under CC-BY 4.0 — please cite when reusing.

## Attributions

Audio samples under `data/wav/` are from Freesound.org and used for AoT demos. See `ATTRIBUTIONS.md` for links and author credits. Please abide by the license terms on each Freesound page.
