# UEC Holonomy (uec-holonomy)

Universal Evidence Curvature (UEC): KL-rate holonomy, entropy production estimation, and Arrow-of-Time demos.

- KL-rate holonomy estimators that match D(P||Q) on loops of representation.
- Equality to entropy production for the Markov time-reversal loop (bits/step).
- Robust test battery and AoT (Arrow-of-Time) demos for audio, sensors, finance.

## Install

```bash
pip install uec-holonomy
# or from source
pip install -e .[dev]
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

See `uec_theory.tex` for the theory and references.

## Python API

```python
from uec.markov import random_markov_biased, sample_markov, entropy_production_rate_bits
from uec.holonomy import klrate_holonomy_time_reversal_markov

T = random_markov_biased(k=3, delta=0.6)
x = sample_markov(T, n=150_000)
print(entropy_production_rate_bits(T))
print(klrate_holonomy_time_reversal_markov(x, k=3, R=3))
```

## Tests

```bash
pytest -q
```

## License

MIT (see LICENSE). If you need a different license, adjust `pyproject.toml`.

