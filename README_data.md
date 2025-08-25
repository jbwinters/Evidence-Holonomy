# Data Guide (Audio, Sensors, Finance)

This repo includes small audio samples under `data/wav/` for Arrow‑of‑Time (AoT) demos, but it does not ship large datasets. The directory `data/kaggle/` is ignored by Git on purpose to keep the repo lean.

Use this guide to fetch your own data and run the AoT meter consistently.

## Audio (WAV)

- Included: a few short WAVs under `data/wav/` from Freesound.org (see `ATTRIBUTIONS.md`).
- Add more: drop additional WAVs under `data/wav/` (ideally PCM 16/24/32‑bit). The loader also supports 8‑bit PCM; if SciPy is installed (`pip install .[audio]`), compressed formats handled by SciPy may also work.
- Recommended CLI flags:

```bash
uec-aot --aot_wav data/wav/boiling.wav \
        --aot_bins 32 --aot_win 65536 --aot_stride 32768 --order 5 \
        --aot_diff --seed 123
```

Notes
- `--aot_diff` (first‑difference) improves forward/backward asymmetry for many real‑world audio signals.
- `--seed` makes the bootstrap CI reproducible.

## Finance (CSV)

Place your CSVs under `data/kaggle/` (ignored by Git). Any time series column (e.g., Close) can be used.

- Example paths (adjust to your files):
  - `data/kaggle/btcusd_1min/btcusd_1-min_data.csv`
  - `data/kaggle/dowjones/dowjones.csv`
- Recommended run (BTC 1‑min close prices):

```bash
uec-aot --aot_csv data/kaggle/btcusd_1min/btcusd_1-min_data.csv \
        --aot_csv_col Close \
        --aot_logreturn \
        --aot_rate 1 \
        --aot_bins 12 --aot_win 2048 --aot_stride 1024 \
        --seed 123
```

Notes
- `--aot_logreturn` uses log‑returns (with a positive shift guard) to expose irreversibility at financial sampling rates.
- `--aot_rate` reports bits/second (for 1‑minute data: 1 Hz; for 1‑second data: 1 Hz means 1 sample/second).
- You can also specify a different column with `--aot_csv_col` by name or index.

## Motion Sensors (e.g., MotionSense)

If you have zipped datasets, unzip them under `data/kaggle/`:

```bash
unzip -n data/kaggle/motionsense.zip -d data/kaggle/motionsense
```

Find column names:

```bash
head -n 1 data/kaggle/motionsense/**/*.csv
```

Run AoT on a chosen column (e.g., `acc_phone_z`) with a typical sampling rate (e.g., 50 Hz):

```bash
uec-aot --scoreboard_glob "data/kaggle/motionsense/**/*.csv" \
        --aot_csv_col acc_phone_z \
        --aot_rate 50 \
        --aot_bins 12 --aot_win 2048 --aot_stride 1024 \
        --aot_diff --seed 123
```

## Keeping the Repo Clean

- `data/kaggle/**` is ignored by `.gitignore`. Keep large datasets out of Git.
- If you accidentally committed large files in the past, rewrite history using [`git-filter-repo`](https://github.com/newren/git-filter-repo) or BFG, then force‑push (see notes in the PR comments or your shell history).
- If you need to version large assets, consider Git LFS, but be mindful of bandwidth quotas.

## Reproducibility Tips

- Use `--seed` to make AoT bootstraps deterministic.
- Document your CLI runs in `results/summary.json` (the package’s test battery uses per‑run JSON logs if you run the original battery script).

## Licensing

- Audio under `data/wav/` includes credits and links in `ATTRIBUTIONS.md`. Abide by the Freesound page licenses (CC‑BY / CC0, etc.).
- Large datasets you download (Kaggle, etc.) come with their own licenses/terms; please respect them.

