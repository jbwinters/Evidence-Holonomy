# UEC Holonomy under Quantum Channels — Experimental Report

This report summarizes the implementation and first results of a density‑matrix–based investigation of “holonomy as information loss” and its relation to classical irreversibility, across quantum channels and measurement choices.

## Summary

- We added density‑matrix evolution with Kraus channels (amplitude damping, pure dephasing), several information‑theoretic holonomy accumulators, and optional measurement bases. Collapse triggers when the accumulator exceeds a threshold.
- A robust, channel‑dependent “irreversibility efficiency” emerges: efficiency = classical record holonomy rate / quantum accumulator rate.
- Amplitude damping yields high efficiency (strong classical arrow‑of‑time), while pure dephasing yields very low efficiency (phases dissipate with little classical irreversibility). Mixed channels interpolate smoothly.
- A relative‑entropy–based accumulator D(ρ || Δ(ρ)) produces efficiency O(1) without tuning, suggesting rough parity between decoherence and classical irreversibility production.

## What’s Implemented

- scripts/quantum_density_matrix_holonomy.py
  - Density matrix ρ evolution: unitary + amplitude damping to |0⟩ + pure dephasing.
  - Holonomy accumulators:
    - entropy_change: ΔS(ρ)×dt to cycle anchor
    - coherence_loss: normalized L1‑coherence loss per step (unitless, per‑cycle)
    - relative_entropy_to_decohered: D(ρ || Δ(ρ))×dt (bits)
  - Lindblad generator L(ρ) and instantaneous entropy production rate: dS/dt = −Tr(L(ρ) log ρ)/ln 2 (bits/time)
  - Collapse to a measurement outcome; emits classical record of basis indices.
  - Outputs: per‑step CSV and summary metrics (record holonomy rate, avg accumulator rate, efficiency, collapses, steady coherence, avg Lindblad rate).

- scripts/holonomy_channel_sweep.py: grid over damping γ and dephasing σ, dims {2,3,4}, method=relative_entropy_to_decohered; writes results/channel_space_map.csv

- scripts/holonomy_mixing_curve.py: one‑parameter curve between damping and dephasing (θ‑mix); writes results/mixing_curve.csv

- scripts/basis_selection_test.py: compares measurement bases (computational, Fourier, random unitary) at fixed channel; writes results/basis_selection*.csv

- scripts/channel_space_mapping.py: full γ×σ×dim sweep; writes results/channel_space_map.csv

- scripts/decoherence_time_test.py: predicts t_dec ≈ threshold / (instantaneous Lindblad rate) and compares to simulated first‑collapse time.

- scripts/uec_quantum_analysis.py: aggregates sweep CSVs and prints summary stats.

## Key Results (selected)

1) Event‑mode (pure‑state) findings reproduced and extended
- Dim=3, record=state, steps=40000
  - Amplitude damping γ=0.02, σ=0.0 → hol_rate ≈ 0.0573 b/step; avg_accum ≈ 0.00222; efficiency ≈ 25.9
  - Pure dephasing γ=0.0, σ=0.2 → hol_rate ≈ 1.23e−4 b/step; avg_accum ≈ 0.00238; efficiency ≈ 0.0517
- Interpretation: damping channels convert quantum information loss into classical circulation much more efficiently than dephasing channels.

2) Mixing‑angle curve (scripts/holonomy_mixing_curve.py)
- Dim=3, steps=20000, amp_scale=0.02, dephase_scale=0.2, θ∈[0,π/2]
  - θ=0 (pure damping): hol≈0.0594, efficiency≈26.8
  - θ≈π/4 (balanced): hol≈0.00656, efficiency≈2.86
  - θ≈π/2 (pure dephasing): hol≈8.3e−5, efficiency≈0.0347
- Smooth interpolation confirms efficiency as a channel descriptor.

3) Density‑matrix accumulators (scripts/quantum_density_matrix_holonomy.py — compare)
- Dim=3, steps=20000, γ=0.02:
  - entropy_change: hol≈2.25e−4, avg_accum≈9.56e−3, eff≈0.0236
  - coherence_loss (normalized): hol≈1.41e−3, avg_accum≈5.04e−5, eff≈28.0
  - relative_entropy_to_decohered: hol≈0.0568, avg_accum≈0.0410, eff≈1.38
- The relative‑entropy variant yields an O(1) efficiency without tuning — most physically plausible among tested accumulators.

4) Channel space mapping (quick slice)
- Dim=3, steps=5000 (results/channel_space_map_quick.csv)
  - γ∈{0,0.025,0.05}, σ∈{0,0.25,0.5}
  - Efficiency ranges ~1.08–1.46 for relative_entropy_to_decohered across this slice, with modest σ‑dependence.

5) Basis selection (scripts/basis_selection_test.py)
- Dim=3, γ=0.02, σ=0.0, steps=10000:
  - computational: hol≈0.0377, eff≈0.919
  - fourier: hol≈0.5260, eff≈7.44
  - random_unitary: hol≈0.2500, eff≈4.82
- Observation: Fourier basis yielded higher classical irreversibility for this setup, counter to the initial “computational‑basis dominance” hypothesis. Basis choice materially affects classical arrow‑of‑time.

6) Decoherence time prediction (scripts/decoherence_time_test.py)
- Dim=3, γ=0.02, σ=0.0, threshold=1.0, dt=0.05
  - Predicted t_dec from instantaneous Lindblad rate: ≈ 2.26
  - Simulated first collapse time: ≈ 0.75
- Takeaway: a single‑point (t=0) rate underestimates early accumulation; averaging dS/dt along the path (or using D(ρ||Δ(ρ))) should yield better predictions of collapse timing.

## Interpretation

- The **irreversibility efficiency** (record holonomy / accumulator rate) captures how a channel converts quantum information loss into classical arrow‑of‑time. Damping is high‑efficiency; dephasing is low‑efficiency; mixed channels interpolate smoothly.
- The relative‑entropy accumulator D(ρ||Δ(ρ)) correlates at O(1) with classical irreversibility without tuning, suggesting it is a promising candidate for a principled holonomy budget.
- Measurement basis profoundly influences the classical record’s irreversibility; “nature chooses basis to maximize classical irreversibility” is testable and not trivially true in our initial runs.

## Open Questions and Next Tests

1) Can efficiency be predicted analytically from Lindblad operators?
   - We now compute dS/dt = −Tr(L(ρ) log ρ)/ln 2. Map efficiency vs theoretically derived rates across the γ–σ plane and compare.

2) Basis selection principle
   - Extend basis tests with multiple random unitaries per channel to characterize expected holonomy vs basis class. Identify when energy basis is optimal.

3) Universal patterns
   - Run the full channel grid (11×11×{2,3,4}) and study efficiency contours; look for level sets and anomalies.

4) Decoherence time prediction
   - Predict t_dec by averaging Lindblad dS/dt along the uncollapsed path, or via D(ρ||Δ(ρ)) trajectory, and compare to simulations.

5) Thermal initialization
   - Add explicit thermal initial states to probe how initial entropy affects efficiency and collapse timing.

## How to Reproduce

- Density matrix runs:
  - `python scripts/quantum_density_matrix_holonomy.py --dim 3 --steps 20000 --gamma 0.02 --method relative_entropy_to_decohered --threshold 1.0`

- Channel sweep (quick):
  - `python scripts/channel_space_mapping.py --dims 3 --gammas 3 --sigmas 3 --steps 5000 --out results/channel_space_map_quick.csv`

- Mixing angle curve:
  - `python scripts/holonomy_mixing_curve.py --dim 3 --n_points 17 --amp_scale 0.02 --dephase_scale 0.2 --steps 20000`

- Basis selection:
  - `python scripts/basis_selection_test.py --dim 3 --steps 10000 --gamma 0.02 --sigma 0.0`

- Decoherence time test:
  - `python scripts/decoherence_time_test.py --dim 3 --steps 10000 --gamma 0.02 --sigma 0.0 --threshold 1.0`

- Aggregate summaries:
  - `python scripts/uec_quantum_analysis.py`

## Limitations

- Relative entropy and dS/dt are computed per step on numerically projected PSD states to ensure stability. This introduces small biases but is necessary for robustness.
- The Fourier/random basis results demonstrate that classical irreversibility depends on the measurement scheme; tying basis choice to dynamics (e.g., energy basis) is physically motivated but not enforced here.
- Our dephasing parameterization (σ) maps to a dephasing rate κ = σ²/2; units between γ and σ differ.

## Next Engineering Tasks

- Add thermal initial state option and target‑relative D(ρ||σ_target) accumulator.
- Add non‑Markovian noise (colored phase noise, correlated jumps) to test memory effects.
- Expand sweep scripts to include density‑matrix accumulators side‑by‑side.

---

Artifacts: see results/* for CSV outputs produced by the quick runs embedded above (e.g., `channel_space_map_quick.csv`, `mixing_curve_quick.csv`, `basis_selection_quick.csv`).

