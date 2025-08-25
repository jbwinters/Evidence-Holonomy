# Universal Evidence Curvature (UEC): Overview & Minimal Formalism

*An information‑geometric way to state the arrow of time—and irreversibility—without referring to a specific physics, coordinate system, or observer.*

## Core idea

Take any world that produces data (a trajectory of observations). Consider loops of **representation transforms**—recodings, coarse‑grainings, time reversals. If you carry your *evidence* (e.g., description length under a universal code) around such a closed loop and return to the starting representation, you generally do **not** return with the same total evidence. The path‑dependent excess is **evidence holonomy**.

- If each step is **lossless and bijective**, holonomy is **zero**.
- If some step is **many‑to‑one** (information‑losing), the **expected** holonomy is **non‑negative** and (under our assumptions) equals a **KL‑rate**; for canonical time‑reversal loops on Markov chains it equals the **entropy‑production rate** (bits/step).

> **UEC (informal):** Irreversibility = curvature of inference. Positive curvature ↔ entropy production.

---

## Minimal formalism

- **World:** a stationary ergodic process \(P\) on a finite alphabet \(\mathcal X\); sample path \(X_{0:n-1}\sim P\).
- **Transforms:** \(F_i\) map finite strings to finite strings. A loop is \(L = F_m\circ\cdots\circ F_1\) with output back in \(\mathcal X^n\).
- **Evidence:** use a **universal code** \(\mathcal E\) (KT/CTW/LZ/PPM). For many stationary ergodic sources,
  \[
  \mathcal E(x_{0:n-1}) \approx -\log_2 P_n(x_{0:n-1})\quad\text{(up to }o(n)\text{)}.
  \]
- **Holonomy on a path \(x\):**
  \[
  \mathrm{Hol}_n^\gamma(x) \;=\; \sum_{i=1}^m \big[\mathcal E(x^{(i)}) - \mathcal E(x^{(i-1)})\big]
  \;=\; \mathcal E(L(x)) - \mathcal E(x).
  \]

Two operational regimes:

1. **Representation‑space holonomy** (evaluate evidence *in the final representation*). In expectation,
   \[
   \frac{1}{n}\,\mathbb E[\mathrm{Hol}] \to h(Q)-h(P),
   \]
   where \(Q\) is the stationary pushforward of \(P\) along \(L\).

2. **Observer‑transported (KL) holonomy** (evaluate both codes on the **same** original coordinates, one universal for \(P\) and one for \(Q\)). Then
   \[
   \frac{1}{n}\,\mathbb E[\mathrm{Hol}] \to \mathsf d(P\|Q)\;\ge 0.
   \]
   For time‑reversal loops on finite‑state Markov chains, this equals the **entropy‑production rate**.

---

## Tiny worked example: two‑state Markov chain

Let \(X_t\in\{0,1\}\) be stationary Markov with
\[
T=\begin{pmatrix}1-a & a\\ b & 1-b\end{pmatrix},\quad \pi\ \text{stationary}.
\]
Loop: **encode transitions** \(\to\) **reverse** \(\to\) **decode second symbol**. The KL‑holonomy rate equals
\[
\sigma \;=\; \sum_{i,j\in\{0,1\}} \pi_i T_{ij}\,\log_2\frac{\pi_i T_{ij}}{\pi_j T_{ji}}\;\ge 0,
\]
vanishing iff **detailed balance** holds (\(\pi_i T_{ij}=\pi_j T_{ji}\)). This is the standard entropy‑production rate in bits/step.

---

## Estimation recipe: the **Holonomy Meter**

**Inputs**
- A time series \(x\); optional transforms (permutations, symbol merges, down/up‑sampling).
- Choice of universal coder (KT/CTW/LZ/PPM).

**Procedure**
1. **Preprocess/quantize** (if needed) to a finite alphabet; keep a consistent mapping.
2. **Construct a loop** \(L\) (e.g., Transition‑Encode \(\to\) Reverse \(\to\) Decode‑Second).
3. **Observer‑transported KL holonomy:**
   - Train a universal code \(\mathcal E_P\) on forward data; snapshot frozen predictor \(P_f\).
   - Apply the loop to training data; train \(\mathcal E_Q\) on looped data; snapshot \(Q_f\).
   - For each evaluation segment \(w\), compute
     \[
     \widehat{\mathrm{KL}}(w) \;=\; \frac{1}{|w|}\big[\,Q_f\text{-codelen}(w) - P_f\text{-codelen}(w)\,\big].
     \]
   - Average across windows; bootstrap for CIs.
4. **Diagnostics**
   - Bijective recodings \(\Rightarrow\) near‑zero holonomy (gauge invariance).
   - Time‑reversal loop on Markov data \(\Rightarrow\) holonomy \(\approx\) analytic EP.
   - Coarse‑grain / refine loops \(\Rightarrow\) non‑negative KL‑holonomy (data‑processing).

**Outputs**
- Holonomy (bits/step), CI from bootstrap, and a scale map vs. window/quantization.

---

## Why this matters

- **Gauge invariance:** bijective relabelings contribute zero.
- **Irreversibility:** many‑to‑one steps contribute a non‑negative KL‑rate.
- **Physical link:** for canonical time‑reversal loops on Markov chains, the KL‑holonomy equals **entropy production**.

*Takeaway:* Irreversibility emerges as **curvature of inference**—a loop integral of evidence under representation changes.

