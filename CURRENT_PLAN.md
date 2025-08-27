Scope
- Codify assumptions in docs and instrument code to check, warn, or adapt.
- Add diagnostics (gauge/surrogate/Markov EP), local holonomy + CIs, change‑points.
- Provide sensitivity and multi‑scale tools; produce reproducible reports.

Deliverables
- docs/assumptions.md: refined assumptions with “Check / Work‑around / Relax” actions.
- Diagnostics suite: gauge, surrogate, Markov EP, support overlap, stationarity splits.
- Local holonomy: sliding windows, block‑bootstrap CIs, change‑point flags.
- Sensitivity grid: k × R panel with stability summary; MDL bin search (optional).
- Multi‑scale + attribution: downsample/coarse‑grain spectrum; channel ablations.
- Reports: CSVs + JSON summaries + figures and README links.

Phase 1: Documentation
- Assumptions doc: non‑negotiables (finite alphabet, stationarity, ergodicity, sufficiency, loop closure), semi‑negotiables, clarifications (hidden variables → lower bound).
- “Assumptions → Actions” cheat‑sheet embedded in README with links.
- Protocol notes for local holonomy and CI method (moving‑block bootstrap).

Phase 2: Core Checks (fast, always‑on)
- Finite alphabet: assert k finite; record chosen k, bin edges, and loop closure alignment.
- Loop closure: verify encode→reverse→decode returns same alphabet and n−1 length; align head/tail explicitly.
- Sufficiency: compute k^R and show n_train, n_eval margins; warn when n << 50·k^R (train) or 10·k^R (eval).
- Support overlap: count reverse zeros post‑smoothing; warn and suggest coarsen/dither.
- Stationarity split: first/second‑half code‑rate comparison with threshold and p‑value.

Phase 3: Local Holonomy + Regime Detection
- Sliding‑window UEC stream with W, stride; produce bits/step and z‑score.
- Moving‑block bootstrap CIs; emit median + CI bands.
- Change‑point detector (CUSUM/KL‑drift) on code‑rate and/or uec_z; emit time stamps and segments.

Phase 4: Diagnostics Suite (reproducible)
- Gauge test: random permutation → inverse → holonomy ≈ 0 within CI.
- Surrogate test: time‑shuffle / circular shift → holonomy ≈ 0.
- Markov EP test: synthetic chains with known σ; holonomy ≈ σ across R.
- Ergodicity probe: gauge test across disjoint segments; instability → flag nonstationarity.
- Measurement stability: jitter data pre‑binning; holonomy variance vs. jitter amplitude.

Phase 5: Sensitivity + Model Selection
- Grid panel: k ∈ {6,8,12,16}, R ∈ {1,2,3}; summarize stability region and runtime.
- MDL‑style selection (optional): pick k minimizing held‑out codelength; report chosen k.
- Ordinal patterns option: enable rank‑pattern symbols as robust discretization.

Phase 6: Multi‑Scale + Attribution
- Curvature spectrum: downsample/coarse‑grain factors s ∈ {1,2,4,8}; holonomy vs. scale.
- Emergence scale: mark first scale with persistent non‑zero holonomy (CI‑aware).
- Attribution: holonomy on (return×volume) vs returns‑only vs volume‑only; inclusion–exclusion summary.

Phase 7: Performance + Fallbacks
- Benchmark: per‑symbol time/memory for counts vs KT across W,k,R; plot scaling.
- Fallback logic: choose counts for long runs; KT for audits (configurable stride/interp).
- CTW hook (future): optional deeper contexts if KT too shallow.

Phase 8: Reporting
- Outputs: per‑bar CSV (signals, states, CIs), JSON summaries (assumption checks, flags).
- Figures: holonomy stream + CI, change‑points, sensitivity heatmaps, spectra.
- Example notebooks/Markdown in results/ with “How to read” guidance.

Configuration
- CLI flags and JSON config for: k, R, W, method (counts/kt), stride, bootstrap params, jitter amplitude, grid values, downsample factors.
- Seeds and environment logging (Python, NumPy) in every run record.

Acceptance Criteria
- Loop closure verified and logged; alignment documented.
- Stationarity split delta below threshold or flagged with segments and reruns by regime.
- Support overlap warnings resolved (coarsen/dither) or acknowledged in report.
- Diagnostics pass: gauge/surrogate ≈ 0 within CI; synthetic Markov holonomy ≈ σ at tolerance.
- Sensitivity grid identifies a stable region; chosen k,R justified.
- Multi‑scale and attribution plots generated; emergence scale identified (if present).
- README updated with “Assumptions → Actions” and run recipes.

