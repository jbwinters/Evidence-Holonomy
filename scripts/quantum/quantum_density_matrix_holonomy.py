#!/usr/bin/env python3
"""
Density-matrix holonomy simulation with information-theoretic accumulators.

Implements:
- DensityMatrixSystem evolving ρ via unitary + Kraus channels
- Holonomy accumulators based on von Neumann entropy and coherence loss
- Collapse when accumulated holonomy exceeds threshold; classical record emitted
- CSV trace per run and summary metrics, including efficiency ratio

Usage examples:
  python scripts/quantum_density_matrix_holonomy.py --dim 3 --steps 40000 \
    --dt 0.05 --gamma 0.02 --sigma 0.0 --method entropy_change --threshold 1.0

  # Compare multiple methods on same parameters
  python scripts/quantum_density_matrix_holonomy.py --compare \
    --methods entropy_change,coherence_loss --dim 3 --gamma 0.02 --steps 40000
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Import UEC holonomy estimator from local src/
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_src_root = _os.path.join(_repo_root, "src")
if _src_root not in _sys.path:
    _sys.path.insert(0, _src_root)

try:
    from uec.holonomy import klrate_holonomy_time_reversal_markov
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Could not import uec.holonomy from src/: {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def matrix_log_psd(rho: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    # Hermitian PSD matrix logarithm via eigendecomposition; returns log_e(rho)
    w, V = np.linalg.eigh((rho + rho.conj().T) / 2.0)
    w = np.clip(np.real(w), eps, None)
    logw = np.log(w)
    return (V @ np.diag(logw) @ V.conj().T).astype(np.complex128, copy=False)


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-15) -> float:
    # S(ρ) in bits
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    w = np.clip(np.real(w), 0.0, 1.0)
    w = w[w > eps]
    if w.size == 0:
        return 0.0
    return float(-np.sum(w * (np.log(w) / np.log(2.0))))


def coherence_l1_norm(rho: np.ndarray) -> float:
    # Sum of absolute off-diagonal elements (L1 coherence, without diagonal)
    a = np.abs(rho)
    return float(np.sum(a) - np.sum(np.abs(np.diag(np.diag(rho)))))


def project_positive_trace_one(rho: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    # Enforce Hermiticity, PSD, trace=1 with eigenvalue clipping
    rhoH = (rho + rho.conj().T) / 2.0
    w, V = np.linalg.eigh(rhoH)
    w = np.clip(np.real(w), 0.0, None)
    rhoP = (V @ np.diag(w) @ V.conj().T).astype(np.complex128, copy=False)
    tr = float(np.real(np.trace(rhoP)))
    if tr <= eps:
        # fallback to maximally mixed
        d = rhoP.shape[0]
        rhoP = np.eye(d, dtype=np.complex128) / float(d)
    else:
        rhoP = rhoP / tr
    return rhoP


def unitary_from_hamiltonian(H: np.ndarray, dt: float) -> np.ndarray:
    w, V = np.linalg.eigh((H + H.conj().T) / 2.0)
    phase = np.exp(-1j * np.real(w) * dt)
    return (V @ np.diag(phase) @ np.linalg.inv(V)).astype(np.complex128, copy=False)


def qubit_rotation_x(omega: float, dt: float) -> np.ndarray:
    theta = omega * dt
    c = math.cos(theta / 2.0)
    s = -1j * math.sin(theta / 2.0)
    I = np.eye(2, dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return c * I + s * sx


@dataclass
class DensityMatrixSystem:
    dim: int
    dt: float
    seed: int
    omega: float = 1.0
    use_random_h: bool = False

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        psi = np.ones(self.dim, dtype=np.complex128) / math.sqrt(self.dim)
        self.rho = np.outer(psi, psi.conj())
        self.rho_initial = self.rho.copy()
        self.C_initial = coherence_l1_norm(self.rho_initial)
        self.holonomy_bits = 0.0
        self.accum_total_bits = 0.0
        self.C_prev: float | None = None
        # Build a fixed unitary propagator
        if self.dim == 2 and not self.use_random_h:
            self.U = qubit_rotation_x(self.omega, self.dt)
        else:
            # Random Hermitian with bounded spectrum
            A = self.rng.normal(size=(self.dim, self.dim)) + 1j * self.rng.normal(size=(self.dim, self.dim))
            H = (A + A.conj().T) / 2.0
            w = np.linalg.eigvalsh(H)
            rad = float(np.max(np.abs(w))) if w.size else 1.0
            if rad > 0:
                H = H / rad
            self.U = unitary_from_hamiltonian(H, self.dt)
        # Store Hamiltonian proxy for thermal init if needed
        self.H_for_init = None

    # Channels
    def apply_amplitude_damping(self, gamma: float) -> None:
        gamma = float(np.clip(gamma, 0.0, 1.0))
        K0 = np.eye(self.dim, dtype=np.complex128)
        K0[0, 0] = 1.0 + 0.0j
        for i in range(1, self.dim):
            K0[i, i] = math.sqrt(max(0.0, 1.0 - gamma))
        rho_new = K0 @ self.rho @ K0.conj().T
        for i in range(1, self.dim):
            Ki = np.zeros((self.dim, self.dim), dtype=np.complex128)
            Ki[0, i] = math.sqrt(max(0.0, gamma))
            rho_new += Ki @ self.rho @ Ki.conj().T
        self.rho = project_positive_trace_one(rho_new)

    def apply_dephasing(self, sigma: float) -> None:
        # Phase diffusion parameter sigma (std of phase); off-diagonals decay by exp(-sigma^2/2)
        decay = math.exp(-0.5 * float(sigma) * float(sigma))
        R = self.rho.copy()
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    R[i, j] = R[i, j] * decay
        self.rho = project_positive_trace_one(R)

    # Info measures
    def compute_holonomy_increment(self, method: str = "entropy_change") -> float:
        if method == "entropy_change":
            S_cur = von_neumann_entropy(self.rho)
            S0 = von_neumann_entropy(self.rho_initial)
            dH = max(0.0, S_cur - S0) * self.dt
        elif method == "coherence_loss":
            # Fractional coherence loss relative to the cycle's initial coherence.
            C = coherence_l1_norm(self.rho)
            if self.C_prev is None:
                dH = 0.0
            else:
                C0 = float(self.C_initial) if self.C_initial is not None else 0.0
                if C0 > 0:
                    frac_loss = (self.C_prev - C) / C0
                    dH = float(max(0.0, frac_loss))  # no dt scaling
                else:
                    dH = 0.0
            self.C_prev = C
        elif method == "relative_entropy_to_decohered":
            # D(ρ || Δ(ρ)), Δ: dephasing map (drop off-diagonals). Use matrix logs.
            rho = self.rho
            diag = np.diag(np.diag(rho)).astype(np.complex128)
            # Avoid singularities with small diagonal floor
            eps = 1e-12
            diag = project_positive_trace_one(diag, eps)
            log_rho = matrix_log_psd(project_positive_trace_one(rho))
            log_diag = matrix_log_psd(diag)
            # natural logs; convert to bits
            D_nat = np.real(np.trace(rho @ (log_rho - log_diag)))
            D_bits = float(D_nat / np.log(2.0))
            dH = max(0.0, D_bits) * self.dt
        else:
            raise ValueError(f"Unknown holonomy method: {method}")
        # accumulate running total (never resets); separate from threshold accumulator
        self.accum_total_bits += dH
        return float(dH)

    # Lindblad generator (per unit time) for amplitude damping to |0> with rate gamma
    # and pure dephasing with rate kappa = sigma^2/2 (on off-diagonals)
    def lindblad_generator(self, rho: np.ndarray, gamma: float, sigma: float) -> np.ndarray:
        d = self.dim
        L = np.zeros_like(rho, dtype=np.complex128)
        g = float(max(0.0, gamma))
        if g > 0.0:
            # Amplitude damping: L_i = sqrt(g) |0><i| for i>=1
            for i in range(1, d):
                Ki = np.zeros((d, d), dtype=np.complex128)
                Ki[0, i] = math.sqrt(g)
                L += Ki @ rho @ Ki.conj().T
                M = np.zeros((d, d), dtype=np.complex128)
                M[i, i] = g
                L -= 0.5 * (M @ rho + rho @ M)
        if sigma and sigma > 0.0:
            kappa = 0.5 * float(sigma) * float(sigma)
            # Dephasing with projectors P_i; L_dephase = sum_i (P_i ρ P_i - 1/2{P_i,ρ}) * 2*kappa? Use kappa to match off-diagonal rate
            # For off-diagonals, the sum over i of -1/2{P_i,ρ} yields -ρ_ij; so coefficient should be kappa to get -kappa ρ_ij.
            for i in range(d):
                P = np.zeros((d, d), dtype=np.complex128)
                P[i, i] = 1.0
                L += kappa * (P @ rho @ P - 0.5 * (P @ rho + rho @ P))
        return L

    def lindblad_entropy_production_rate_bits(self, rho: np.ndarray, gamma: float, sigma: float) -> float:
        # dS/dt = -Tr(L(rho) log rho) in nats; convert to bits
        L = self.lindblad_generator(rho, gamma, sigma)
        log_rho = matrix_log_psd(project_positive_trace_one(rho))
        rate_nat = -float(np.real(np.trace(L @ log_rho)))
        return float(rate_nat / np.log(2.0))

    # Evolution
    def evolve_with_collapse(
        self,
        n_steps: int,
        threshold_bits: float,
        gamma: float,
        sigma: float,
        holonomy_method: str = "entropy_change",
        record_mode: str = "state",
        measure_U: np.ndarray | None = None,
    ) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        record: List[int] = []
        times = np.zeros(n_steps, dtype=float)
        S_series = np.zeros(n_steps, dtype=float)
        C_series = np.zeros(n_steps, dtype=float)
        Hacc_series = np.zeros(n_steps, dtype=float)
        Srate_series = np.zeros(n_steps, dtype=float)

        self.holonomy_bits = 0.0
        self.accum_total_bits = 0.0
        self.C_prev = None
        collapses = 0

        for t in range(n_steps):
            # Unitary
            self.rho = self.U @ self.rho @ self.U.conj().T
            self.rho = project_positive_trace_one(self.rho)

            # Channels per-step (scaled by dt)
            if gamma and gamma > 0:
                self.apply_amplitude_damping(float(gamma) * self.dt)
            if sigma and sigma > 0:
                self.apply_dephasing(float(sigma) * self.dt)

            # Measures and accumulation
            S_series[t] = von_neumann_entropy(self.rho)
            C_series[t] = coherence_l1_norm(self.rho)
            # Instantaneous Lindblad entropy rate (bits/time)
            Srate_series[t] = self.lindblad_entropy_production_rate_bits(self.rho, gamma, sigma)
            dH = self.compute_holonomy_increment(method=holonomy_method)
            self.holonomy_bits += dH
            Hacc_series[t] = self.holonomy_bits
            times[t] = (t + 1) * self.dt

            # Collapse if threshold exceeded
            if self.holonomy_bits >= threshold_bits:
                # Measurement in chosen basis (U): probabilities from diag(U ρ U†)
                if measure_U is None:
                    probs = np.real(np.diag(self.rho))
                else:
                    rho_meas = measure_U @ self.rho @ measure_U.conj().T
                    probs = np.real(np.diag(rho_meas))
                probs = probs / max(1e-32, float(np.sum(probs)))
                outcome = int(self.rng.choice(self.dim, p=probs))
                # Project onto outcome
                if measure_U is None:
                    self.rho = np.zeros_like(self.rho)
                    self.rho[outcome, outcome] = 1.0
                else:
                    Pk = np.zeros_like(self.rho)
                    Pk[outcome, outcome] = 1.0
                    self.rho = measure_U.conj().T @ Pk @ measure_U
                self.rho_initial = self.rho.copy()
                self.C_initial = coherence_l1_norm(self.rho_initial)
                self.holonomy_bits = 0.0
                self.C_prev = None
                rec_sym = outcome
                collapses += 1
            else:
                # No collapse: emit argmax state for record
                rec_sym = int(np.argmax(np.real(np.diag(self.rho))))

            record.append(rec_sym)

        return record, times, S_series, C_series, Hacc_series, Srate_series, collapses


def parse_methods(s: str) -> List[str]:
    return [tok.strip() for tok in s.split(',') if tok.strip()]


def run_single(
    dim: int,
    steps: int,
    dt: float,
    omega: float,
    gamma: float,
    sigma: float,
    threshold_bits: float,
    method: str,
    seed: int,
    out_prefix: str,
) -> dict:
    sys = DensityMatrixSystem(dim=dim, dt=dt, seed=seed, omega=omega)
    record, times, S_series, C_series, Hacc_series, Srate_series, collapses = sys.evolve_with_collapse(
        n_steps=steps,
        threshold_bits=threshold_bits,
        gamma=gamma,
        sigma=sigma,
        holonomy_method=method,
        record_mode="state",
        measure_U=None,
    )

    # Holonomy on classical record
    hol_rate = klrate_holonomy_time_reversal_markov(record, k=dim, R=3)
    avg_accum = sys.accum_total_bits / max(1, steps)
    eff = 0.0 if avg_accum <= 0 else (hol_rate / avg_accum)

    # Save CSV trace
    ensure_dir("results")
    tag = f"dm_dim{dim}_m{method}_g{gamma}_s{sigma}_dt{dt}_n{steps}"
    csv_path = f"{out_prefix}_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "von_neumann_entropy_bits", "coherence_l1", "holonomy_accum_bits", "lindblad_srate_bits_per_time", "record_symbol"])
        for i in range(len(times)):
            wr.writerow([f"{times[i]:.9g}", f"{S_series[i]:.9g}", f"{C_series[i]:.9g}", f"{Hacc_series[i]:.9g}", f"{Srate_series[i]:.9g}", int(record[i])])

    return {
        "dim": dim,
        "steps": steps,
        "dt": dt,
        "omega": omega,
        "gamma": gamma,
        "sigma": sigma,
        "threshold_bits": threshold_bits,
        "method": method,
        "seed": seed,
        "hol_rate_bits_per_step": float(hol_rate),
        "avg_accum_bits_per_step": float(avg_accum),
        "efficiency": float(eff),
        "collapses": int(collapses),
        "steady_coherence_l1": float(C_series[-1]) if len(C_series) else float('nan'),
        "avg_lindblad_srate_bits_per_time": float(np.mean(Srate_series)) if len(Srate_series) else 0.0,
        "csv": csv_path,
    }


def validations_quick() -> List[str]:
    logs: List[str] = []
    # 1) Pure state entropy ~ 0
    sys = DensityMatrixSystem(dim=3, dt=0.05, seed=0)
    S0 = von_neumann_entropy(sys.rho)
    logs.append(f"[Pure] S(|ψ><ψ|)≈{S0:.3g} (expect ~0)")
    # 2) Unitary invariance
    S1 = von_neumann_entropy(sys.U @ sys.rho @ sys.U.conj().T)
    logs.append(f"[Unitary] S(UρU†)-S(ρ)≈{(S1-S0):.3g} (expect ~0)")
    # 3) Channel composition (qubit damping): g_eff≈1-(1-g1)(1-g2)
    sys2 = DensityMatrixSystem(dim=2, dt=0.05, seed=0)
    g1, g2 = 0.1, 0.2
    rho_a = sys2.rho.copy()
    sys2.apply_amplitude_damping(g1)
    sys2.apply_amplitude_damping(g2)
    rho_b = sys2.rho.copy()
    sys3 = DensityMatrixSystem(dim=2, dt=0.05, seed=0)
    g_eff = 1.0 - (1.0 - g1) * (1.0 - g2)
    sys3.apply_amplitude_damping(g_eff)
    rho_c = sys3.rho.copy()
    diff = np.linalg.norm(rho_b - rho_c)
    logs.append(f"[Compose] ||ρ(g1∘g2)-ρ(g_eff)||≈{diff:.3g} (small)")
    # 4) Equilibrium (damping→|0><0|), late dS≈0
    sys4 = DensityMatrixSystem(dim=2, dt=0.05, seed=0)
    for _ in range(2000):
        sys4.apply_amplitude_damping(0.02*sys4.dt)
    S_eq = von_neumann_entropy(sys4.rho)
    logs.append(f"[Equil] S≈{S_eq:.3g} (near 0); ρ00≈{np.real(sys4.rho[0,0]):.4f}")
    return logs


def main() -> None:
    ap = argparse.ArgumentParser(description="Density-matrix holonomy simulation")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.0, help="Amplitude-damping rate per unit time")
    ap.add_argument("--sigma", type=float, default=0.0, help="Dephasing phase-noise std per unit time (radians)")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--method", type=str, default="entropy_change")
    ap.add_argument("--methods", type=str, default="entropy_change,coherence_loss,relative_entropy_to_decohered")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix", type=str, default="results/dm_holonomy")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    if args.validate:
        for line in validations_quick():
            print(line)

    if args.compare:
        methods = parse_methods(args.methods)
        rows = []
        for m in methods:
            res = run_single(
                dim=args.dim,
                steps=args.steps,
                dt=args.dt,
                omega=args.omega,
                gamma=args.gamma,
                sigma=args.sigma,
                threshold_bits=args.threshold,
                method=m,
                seed=args.seed,
                out_prefix=args.out_prefix,
            )
            rows.append(res)
            print(
                f"[Method {m}] hol_rate={res['hol_rate_bits_per_step']:.6g} avg_accum={res['avg_accum_bits_per_step']:.6g} "
                f"eff={res['efficiency']:.6g} csv={res['csv']}"
            )
        # Summary CSV
        ensure_dir("results")
        sum_path = f"{args.out_prefix}_summary.csv"
        keys = list(rows[0].keys())
        with open(sum_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
        print(f"Saved comparison summary to {sum_path}")
        return

    # Single run
    res = run_single(
        dim=args.dim,
        steps=args.steps,
        dt=args.dt,
        omega=args.omega,
        gamma=args.gamma,
        sigma=args.sigma,
        threshold_bits=args.threshold,
        method=args.method,
        seed=args.seed,
        out_prefix=args.out_prefix,
    )
    print(
        f"[DensityMatrix] dim={res['dim']} steps={res['steps']} method={res['method']} "
        f"hol_rate={res['hol_rate_bits_per_step']:.6g} avg_accum={res['avg_accum_bits_per_step']:.6g} "
        f"eff={res['efficiency']:.6g}"
    )


if __name__ == "__main__":
    main()
