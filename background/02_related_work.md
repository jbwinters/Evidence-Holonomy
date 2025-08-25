# Related Work: Irreversibility, KL Rates, and Entropy Production

This review connects evidence holonomy to established results in stochastic thermodynamics and information theory, and summarizes model‑free estimation trends.

## 1) Arrow of time and entropy production

- **Second Law / EP:** Macroscopic irreversibility is captured by **entropy production (EP)**. In Markovian steady states, a standard formula is
  \[
  \sigma = \sum_{i,j} \pi_i T_{ij}\,\ln\frac{\pi_i T_{ij}}{\pi_j T_{ji}}\ \ (\text{nats/step}),
  \]
  vanishing iff detailed balance holds (Schnakenberg, 1976).

- **Path‑space KL:** Fluctuation theorems relate EP to **pathwise KL divergence** between forward and reverse processes (Crooks, 1999; Kawai–Parrondo–Van den Broeck, 2007). In steady‑state settings with full observation,
  \[
  \Delta S_{\text{tot}} = D_{\mathrm{KL}}\big(P[\omega]\ \|\ P[\tilde\omega]\big).
  \]
  Our KL‑holonomy implements exactly this distinguishability in bits.

- **Experimental confirmations:** Microscopic experiments (e.g., Batalhão et al., 2015) measured EP and verified equality with path KL in driven quantum systems.

## 2) Information‑theoretic time asymmetry

- A process is **reversible** if its statistics are invariant under time reversal. Deviations can be measured by
  - KL divergence between forward and reversed path laws;
  - tests based on permutation patterns / visibility graphs (e.g., Porta et al., 2008; Zanin et al., 2020).

- **Key point:** With **partial observation** or coarse‑graining, the **observed** KL is a **lower bound** on true EP (data‑processing inequality). Our framework makes this operational through loops and observer transport.

## 3) Estimating EP and irreversibility from data

Three broad families:

1. **Model‑based** (known dynamics): compute EP from rates/fluxes or trajectory work/heat integrals.
2. **Variational / TUR‑based bounds:** learn auxiliary currents/observables that lower‑bound EP (neural implementations exist).
3. **Model‑free time‑series estimators:** directly estimate the **forward vs. reverse** distinguishability (e.g., KL divergence or well‑calibrated classification proxies). Recent approaches provide **windowed** and **bias‑corrected** estimators with surrogates/bootstraps and finite‑sample extrapolations.

Our **KL holonomy** belongs to class (3): a universal‑coding route to path KL. For canonical time‑reversal loops on Markov chains, it matches analytic EP; in partially observed systems, it certifies a **lower bound**.

## 4) Holonomy analogies

- **Geometric phase / holonomy:** parallel transport around loops accumulates a path‑dependent residue when the space has curvature. Evidence holonomy mirrors this: **lossless** steps yield zero residue; **irreversible** steps (quotients/coarse‑grainings) produce positive curvature.

---

## Selected references

- Schnakenberg, J. (1976). *Network theory of master equation*. **Rev. Mod. Phys.** 48, 571–585.
- Crooks, G. E. (1999). *Entropy production fluctuation theorem and the nonequilibrium work relation*. **Phys. Rev. E** 60, 2721–2726.
- Kawai, R., Parrondo, J. M. R., & Van den Broeck, C. (2007). *Dissipation: The phase‑space perspective*. **Phys. Rev. Lett.** 98, 080602.
- Seifert, U. (2012). *Stochastic thermodynamics, fluctuation theorems and molecular machines*. **Rep. Prog. Phys.** 75, 126001.
- Batalhão, T. B. et al. (2015). *Irreversibility and the arrow of time in a quenched quantum system*. **Phys. Rev. Lett.** 115, 190601.
- Roldán, É., & Parrondo, J. M. R. (2010). *Estimating dissipation from single stationary trajectories*. **Phys. Rev. Lett.** 105, 150607.
- Porta, A. et al. (2008). *Temporal asymmetries of short‑term heart period variability*. **Am. J. Physiol.–Regul.** 295, R550–R557.
- Zanin, M. et al. (2020). *Time irreversibility in brain dynamics*. **Front. Physiol.** 10:1619.
- Wei, D. et al. (2018). *Learning and Using the Arrow of Time*. **CVPR**.
- (Plus recent preprints on direct irreversibility estimates / windowed EP; see your project bibliography.)

