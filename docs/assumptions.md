# Assumptions, Checks, and Practical Work‑arounds

This document refines the core assumptions behind UEC/KL‑rate holonomy and turns them into actionable checks, work‑arounds, and relaxations you can apply to real data.

## A. Non‑negotiable (as used in the theorems)

1) Finite alphabet (discretization)
- Check: chosen `k` is finite; loop returns to the same alphabet (loop closure).
- Work‑around (continuous data): fix `k` and report sensitivity over `k∈{6,8,12,16}`; or use ordinal patterns (permutation symbols).

2) Stationarity
- Check: compare code‑length rate on first vs. second half; big shifts ⇒ nonstationary.
- Work‑around: local holonomy in sliding windows with bootstrap CIs; add change‑point detection; report per‑regime estimates.

3) Ergodicity
- Check: run gauge tests (bijective loop ≈ 0) on multiple disjoint segments; instability suggests weak ergodicity or nonstationarity.
- Work‑around: restrict to long, homogeneous segments; avoid stitching across regimes.

4) Sufficient data
- Rule of thumb: ensure effective counts per context are not tiny: `n_train ≳ 50·k^R` and `n_eval ≳ 10·k^R`.
- If violated, reduce `k` and/or `R`.

5) Loop closure + shift‑compatibility
- Check: the composed loop maps `X^n→X^(n−1)` on the same alphabet; evaluate on an aligned tail or head consistently; confirm stationarity of the pushforward.
- Work‑around: if length changes, align head/tail consistently and document.

## B. Semi‑negotiable / context‑specific

6) Absolute continuity (support overlap)
- Check: if forward observes `i→j` but reverse `j→i` is never seen, raw reversed model has zeros (KL → ∞). KT smoothing mitigates but signals a problem.
- Work‑around: smooth counts (KT 0.5); coarsen bins until both directions appear; add tiny dithering before binning.

7) Markov property (only for analytic EP formula)
- You do not need Markovity to measure KL‑rate holonomy; only to equate it to the analytic EP rate `Σ_ij π_i T_ij log((π_i T_ij)/(π_j T_ji))`.
- Check: on synthetic Markov chains, holonomy ≈ analytic EP.
- Work‑around (non‑Markov): interpret holonomy as EP lower bound under partial observation.

## C. Implicit clarifications

8) Computational tractability
- Check: per‑symbol time and memory; if slow, reduce `R`, `k`, or window; or use `counts` estimator for long runs and KT for audits.

9) Measurement precision
- Check: small jitter shouldn’t wildly change bins; if it does, prefer ordinal patterns or return transforms before discretization.

10) Observational completeness (replaces “no hidden variables”)
- Equality (holonomy = physical EP) needs observational completeness; otherwise measured holonomy is a certified lower bound (data processing inequality).

## Practical limitations → remedies

1) Non‑stationarity
- Local holonomy with moving‑block bootstrap CIs.
- Change‑point detection (CUSUM/KL‑drift) on code‑rate; report per‑regime.

2) Continuous data / binning
- Stable defaults: log‑returns + quantile binning.
- Sensitivity panel: `k∈{6,8,12,16}`; show stability vs `k`.
- MDL bin search (upgrade): choose `k` minimizing held‑out codelength.
- Ordinal patterns for robustness.

3) Partial observation
- Interpretation: lower bound on EP.
- Tighten bound: add channels (e.g., volume with returns), or raise `R`.
- Ablations: measure drop when removing a channel.

4) Causal structure / attribution
- Compute holonomy on `(X,Y)`, then `X` and `Y` alone; attribute via inclusion–exclusion.
- Directional tests: `X|lag(Y)` vs `Y|lag(X)`.

5) Multi‑scale formalism
- Curvature spectrum: holonomy after downsample/coarse‑grain factors `s∈{1,2,4,8}`.
- Emergence scale: first scale with persistent non‑zero holonomy (CI‑aware).

## Diagnostics to ship
1) Gauge test: permutation → inverse permutation ⇒ holonomy ≈ 0.
2) Surrogate test: time‑shuffle ⇒ holonomy ≈ 0 (within CI).
3) Markov EP test (synthetic): known `σ` ⇒ holonomy ≈ `σ`.
4) Support check: min reverse count across observed forward transitions ≥ 1 after smoothing; otherwise warn.
5) Sensitivity grid: holonomy over `k∈{6,8,12,16}`, `R∈{1,2,3}`; choose a stable region.

