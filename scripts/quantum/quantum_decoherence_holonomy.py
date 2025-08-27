#!/usr/bin/env python3
"""
Holonomy-triggered quantum decoherence (collapse) toy simulation.

Idea: quantum = compressed description; decoherence = decompression when
irreversibility (holonomy) accumulates. We simulate a small quantum system
with unitary evolution, accumulate a holonomy budget proportional to the
state's measurement entropy in a preferred basis, and trigger a projective
measurement (collapse) when the accumulated holonomy exceeds a threshold
in bits. We then evaluate the classical measurement record's KL-rate
holonomy using the repo's UEC estimators.

This is a pedagogical model, not a physical solver. It is designed to
probe the "holonomy-as-irreversibility budget" hypothesis and how a
threshold produces quantum→classical transitions.

Outputs:
- Prints a concise summary to stdout.
- Writes a CSV trace and a summary TXT under results/.
- Writes an events CSV with per-collapse analytics (pre-gap, dH, overshoot).
- Writes a ramp-average CSV showing mean holonomy build-up before collapses.

Usage examples:
  python scripts/quantum_decoherence_holonomy.py --steps 50000 --threshold 2.0 \
      --coupling 0.04 --omega 1.0 --dim 2 --seed 0

  # Compare regimes (higher threshold ⇒ rarer collapses)
  python scripts/quantum_decoherence_holonomy.py --threshold 0.5
  python scripts/quantum_decoherence_holonomy.py --threshold 2.0
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Ensure local src/ on sys.path for `import uec` without installing
import sys as _sys, os as _os
_repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
_src_root = _os.path.join(_repo_root, "src")
if _src_root not in _sys.path:
    _sys.path.insert(0, _src_root)

try:
    # KL-rate holonomy estimator (Markov time-reversal loop)
    from uec.holonomy import klrate_holonomy_time_reversal_markov
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Could not import uec.holonomy from src/: {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def shannon_bits(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    mask = p > eps
    if not np.any(mask):
        return 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def random_hermitian(dim: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    H = (A + A.conj().T) / 2.0
    # Normalize spectral radius for reasonable time scales
    w = np.linalg.eigvalsh(H)
    rad = float(np.max(np.abs(w))) if w.size else 1.0
    if rad > 0:
        H = H / rad
    return (H * scale).astype(np.complex128, copy=False)


def unitary_from_hamiltonian(H: np.ndarray, dt: float) -> np.ndarray:
    # U = exp(-i H dt) via eigen decomposition (dim is small in this toy)
    w, V = np.linalg.eigh(H)
    phase = np.exp(-1j * w * dt)
    Vinv = np.linalg.inv(V)
    U = (V @ np.diag(phase) @ Vinv).astype(np.complex128, copy=False)
    # Ensure numerical unitarity (symmetrize small errors)
    return U


def qubit_rotation_x(omega: float, dt: float) -> np.ndarray:
    # U = exp(-i (omega/2) sigma_x dt) = cos(theta/2) I - i sin(theta/2) sigma_x
    theta = omega * dt
    c = math.cos(theta / 2.0)
    s = -1j * math.sin(theta / 2.0)
    I = np.eye(2, dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return c * I + s * sx


@dataclass
class Config:
    dim: int = 2
    steps: int = 50_000
    dt: float = 0.05
    omega: float = 1.0
    coupling: float = 0.04
    temperature: float = 1.0
    threshold_bits: float = 1.0
    seed: int = 0
    R_order: int = 3
    out_prefix: str = "results/qdecoh"
    init_superposition: bool = True
    # If True, emission alphabet includes a 0 symbol for "no collapse"
    include_quiet_symbol: bool = True
    # Record mode: 'auto' (use include_quiet_symbol), 'state', 'state_quiet', 'events'
    record_mode: str = "auto"
    # Optional irreversible bath: amplitude damping rate and target basis index
    amp_damp_gamma: float = 0.0
    amp_damp_target: int = 0
    # Pure dephasing (phase damping) strength per step (std of phase noise)
    dephase_sigma: float = 0.0
    # Window (steps) for averaging the pre-collapse holonomy ramp
    ramp_window: int = 200


def simulate(cfg: Config) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    dim = int(cfg.dim)

    # Build unitary per-step propagator
    if dim == 2:
        U = qubit_rotation_x(cfg.omega, cfg.dt)
    else:
        # Generic bounded-spectrum Hamiltonian with nearest-neighbor ring structure
        H = random_hermitian(dim, rng, scale=1.0)
        # Add a small circulant coupling to encourage mixing
        J = 0.5
        circ = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            circ[i, (i + 1) % dim] = J
            circ[(i + 1) % dim, i] = J
        H = (H + circ + circ.conj().T) / 2.0
        U = unitary_from_hamiltonian(H, cfg.dt)

    # Initial state: equal superposition or basis state |0>
    if cfg.init_superposition:
        psi = np.ones(dim, dtype=np.complex128)
        psi /= np.linalg.norm(psi)
    else:
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0 + 0.0j

    # Holonomy budget accumulator (bits)
    H_accum = 0.0
    threshold = float(cfg.threshold_bits)

    # Measurement/record alphabet depends on record_mode
    mode = getattr(cfg, "record_mode", "auto")
    if mode == "auto":
        mode = "state_quiet" if cfg.include_quiet_symbol else "state"
    if mode == "state_quiet":
        # 0 = no-event; 1..dim = collapse→basis index
        k_record = dim + 1
    elif mode == "state":
        # argmax basis index each step (length-dim alphabet)
        k_record = dim
    elif mode == "events":
        # 0 = no-event; 1..dim = collapse→basis index; dim+1 = amplitude-damping jump
        k_record = dim + 2
    else:
        raise SystemExit(f"Unknown record_mode: {mode}")
    records: list[int] = []

    # Time series for analysis
    ts = np.zeros(cfg.steps, dtype=float)
    hol = np.zeros(cfg.steps, dtype=float)
    H_meas = np.zeros(cfg.steps, dtype=float)
    collapses = 0
    intervals: list[int] = []
    last_collapse_step = -1

    # Per-collapse analytics
    events = []  # list of dicts
    collapse_indices: list[int] = []

    for t in range(cfg.steps):
        # Unitary, reversible step
        psi = U @ psi
        # Numerical renormalization in case of drift
        psi = psi / max(1e-32, np.linalg.norm(psi))

        # Optional amplitude damping (quantum jump unraveling)
        g = max(0.0, min(1.0, float(getattr(cfg, "amp_damp_gamma", 0.0))))
        jumped = False
        if g > 0.0:
            if dim == 2:
                p1 = float(np.abs(psi[1]) ** 2)
                p_jump = g * p1
                if rng.random() < p_jump:
                    psi[:] = 0.0 + 0.0j
                    psi[0] = 1.0 + 0.0j
                    jumped = True
                else:
                    psi[1] *= math.sqrt(max(0.0, 1.0 - g))
                    psi = psi / max(1e-32, np.linalg.norm(psi))
            else:
                tgt = int(getattr(cfg, "amp_damp_target", 0)) % dim
                mask = np.ones(dim, dtype=bool)
                mask[tgt] = False
                p_exc = float(np.sum(np.abs(psi[mask]) ** 2))
                p_jump = g * p_exc
                if rng.random() < p_jump:
                    psi[:] = 0.0 + 0.0j
                    psi[tgt] = 1.0 + 0.0j
                    jumped = True
                else:
                    psi[mask] *= math.sqrt(max(0.0, 1.0 - g))
                    psi = psi / max(1e-32, np.linalg.norm(psi))

        # Optional pure dephasing: independent random phase diffusion on components
        sig = float(getattr(cfg, "dephase_sigma", 0.0))
        if sig > 0.0:
            phases = rng.normal(loc=0.0, scale=sig, size=dim)
            psi = psi * np.exp(1j * phases)
            # populations unchanged; no renorm needed beyond numerical safety
            psi = psi / max(1e-32, np.linalg.norm(psi))

        # Measurement probabilities in preferred (computational) basis
        p = np.abs(psi) ** 2
        p = p / max(1e-32, np.sum(p))
        H_bits = shannon_bits(p)

        # Environmental irreversibility proxy (bits per unit time)
        dH = cfg.coupling * cfg.temperature * H_bits * cfg.dt
        pre_hol = H_accum
        H_accum = pre_hol + dH

        # Collapse condition: accumulated holonomy budget exceeds threshold
        if H_accum >= threshold:
            # Projective measurement in the preferred basis
            idx = int(rng.choice(dim, p=p))
            # Log event analytics
            events.append({
                "step": int(t),
                "time": float((t + 1) * cfg.dt),
                "pre_hol": float(pre_hol),
                "dH": float(dH),
                "overshoot": float(H_accum - threshold),
                "threshold": float(threshold),
                "H_bits": float(H_bits),
                "collapse_index": int(idx),
            })
            collapse_indices.append(t)

            psi[:] = 0.0 + 0.0j
            psi[idx] = 1.0 + 0.0j
            H_accum = 0.0
            collapses += 1
            if last_collapse_step >= 0:
                intervals.append(t - last_collapse_step)
            last_collapse_step = t
            if mode == "state_quiet":
                symbol = idx + 1
            elif mode == "state":
                symbol = idx  # post-collapse argmax equals idx
            elif mode == "events":
                symbol = idx + 1  # collapse symbol bucket
            else:
                raise AssertionError
        else:
            if mode == "state_quiet":
                symbol = 0
            elif mode == "state":
                symbol = int(np.argmax(p))
            elif mode == "events":
                symbol = (dim + 1) if jumped else 0
            else:
                raise AssertionError

        records.append(int(symbol))
        ts[t] = (t + 1) * cfg.dt
        hol[t] = H_accum
        H_meas[t] = H_bits

    # Evaluate KL-rate holonomy on the classical record
    k = int(k_record)
    hol_rate = klrate_holonomy_time_reversal_markov(records, k=k, R=cfg.R_order)

    # Proxy for quantum holonomy accumulation rate (bits/step):
    # average per-step dH used by the accumulator
    avg_accum_bits_per_step = float(cfg.coupling * cfg.temperature * cfg.dt * float(np.mean(H_meas)))
    irr_efficiency = float(0.0 if avg_accum_bits_per_step <= 0 else hol_rate / avg_accum_bits_per_step)

    # Summaries
    mean_interval = float(np.mean(intervals)) if intervals else float("nan")
    out = {
        "dim": dim,
        "steps": int(cfg.steps),
        "dt": float(cfg.dt),
        "omega": float(cfg.omega),
        "coupling": float(cfg.coupling),
        "temperature": float(cfg.temperature),
        "threshold_bits": float(cfg.threshold_bits),
        "seed": int(cfg.seed),
        "R_order": int(cfg.R_order),
        "include_quiet_symbol": bool(cfg.include_quiet_symbol),
        "record_mode": mode,
        "amp_damp_gamma": float(getattr(cfg, "amp_damp_gamma", 0.0)),
        "amp_damp_target": int(getattr(cfg, "amp_damp_target", 0)),
        "collapses": int(collapses),
        "mean_steps_between_collapses": mean_interval,
        "record_length": int(len(records)),
        "record_alphabet_k": k,
        "klrate_holonomy_bits_per_step": float(hol_rate),
        "avg_accum_bits_per_step": avg_accum_bits_per_step,
        "irreversibility_efficiency": irr_efficiency,
    }

    # Attach detailed event info
    out["events"] = events
    out["collapse_indices"] = collapse_indices

    return out, ts, hol, H_meas, np.array(records, dtype=int)


def write_outputs(cfg: Config, summary: dict, ts: np.ndarray, hol: np.ndarray, H_meas: np.ndarray, records: np.ndarray) -> str:
    ensure_dir("results")
    tag = (
        f"dim{summary['dim']}_thr{summary['threshold_bits']:.3g}_"
        f"c{summary['coupling']:.3g}_T{summary['temperature']:.3g}_"
        f"w{summary['omega']:.3g}_dt{summary['dt']:.3g}_n{summary['steps']}"
    )
    base = f"{cfg.out_prefix}_{tag}"

    # CSV trace
    csv_path = f"{base}.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "holonomy_accum_bits", "meas_entropy_bits", "record_symbol"])
        for i in range(len(ts)):
            wr.writerow([f"{ts[i]:.9g}", f"{hol[i]:.9g}", f"{H_meas[i]:.9g}", int(records[i])])

    # Events CSV (per-collapse analytics)
    events = summary.get("events", [])
    ev_path = f"{base}_events.csv"
    with open(ev_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["step", "time", "pre_hol", "dH", "overshoot", "threshold", "H_bits", "collapse_index"])
        for e in events:
            wr.writerow([
                e["step"], f"{e['time']:.9g}", f"{e['pre_hol']:.9g}", f"{e['dH']:.9g}",
                f"{e['overshoot']:.9g}", f"{e['threshold']:.9g}", f"{e['H_bits']:.9g}", e["collapse_index"]
            ])

    # Ramp-average CSV (mean build-up before collapse over a fixed window)
    W = int(max(1, getattr(cfg, "ramp_window", 200)))
    ramp_sum = np.zeros(W, dtype=float)
    ramp_cnt = 0
    collapse_idx = summary.get("collapse_indices", [])
    hol_arr = np.asarray(hol, dtype=float)
    for t in collapse_idx:
        a = t - W
        b = t
        if a < 0:
            continue
        window = hol_arr[a:b]
        # Skip if a reset occurs inside window (aside from at the collapse step b)
        if np.any(window[:-1] == 0.0):
            continue
        ramp_sum += window
        ramp_cnt += 1
    ramp_mean = ramp_sum / max(1, ramp_cnt)
    ramp_path = f"{base}_ramp.csv"
    with open(ramp_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["offset_steps", "mean_holonomy_bits"])
        for i in range(W):
            wr.writerow([int(i - W), f"{ramp_mean[i]:.9g}"])

    # TXT summary
    txt_path = f"{base}.txt"
    lines = []
    lines.append("[Quantum→Classical via Holonomy] Simulation summary")
    lines.append(
        f"dim={summary['dim']} steps={summary['steps']} dt={summary['dt']} omega={summary['omega']}\n"
        f"coupling={summary['coupling']} T={summary['temperature']} threshold_bits={summary['threshold_bits']}\n"
        f"record_mode={summary['record_mode']}\n"
        f"amp_damp_gamma={summary['amp_damp_gamma']} amp_damp_target={summary['amp_damp_target']}\n"
        f"collapses={summary['collapses']} mean_steps_between_collapses={summary['mean_steps_between_collapses']}\n"
        f"record_k={summary['record_alphabet_k']} klrate_holonomy_bits/step={summary['klrate_holonomy_bits_per_step']:.6g}\n"
        f"avg_accum_bits/step={summary['avg_accum_bits_per_step']:.6g} irreversibility_efficiency={summary['irreversibility_efficiency']:.6g}"
    )
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return txt_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Holonomy-triggered quantum decoherence toy model")
    ap.add_argument("--dim", type=int, default=2, help="Hilbert space dimension (default 2)")
    ap.add_argument("--steps", type=int, default=50_000, help="Simulation steps")
    ap.add_argument("--dt", type=float, default=0.05, help="Time step")
    ap.add_argument("--omega", type=float, default=1.0, help="Unitary frequency (qubit rotation around X)")
    ap.add_argument("--coupling", type=float, default=0.04, help="Environment coupling strength multiplier")
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling for holonomy rate")
    ap.add_argument("--threshold", type=float, default=1.0, help="Collapse threshold in bits (holonomy budget)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--order", type=int, default=3, help="KT Markov order for KL-rate holonomy")
    ap.add_argument("--out_prefix", type=str, default="results/qdecoh", help="Output file prefix in results/")
    ap.add_argument("--no_quiet_symbol", action="store_true", help="Do not include a 'no-collapse' symbol in record (deprecated by --record_mode)")
    ap.add_argument("--record_mode", type=str, default="auto", choices=["auto","state","state_quiet","events"],
                    help="Record encoding: auto=compat, state=argmax, state_quiet=no-event+collapse_i, events=no-event+collapse_i+jump")
    ap.add_argument("--basis_superposition", action="store_true", help="Start in equal superposition (default)")
    ap.add_argument("--basis_ground", action="store_true", help="Start in |0> instead of superposition")
    ap.add_argument("--amp_damp", type=float, default=0.0, help="Amplitude-damping rate per step (0..1)")
    ap.add_argument("--amp_damp_target", type=int, default=0, help="Basis index to relax to (default 0)")
    ap.add_argument("--dephase", type=float, default=0.0, help="Pure dephasing phase-noise std per step (radians)")
    ap.add_argument("--ramp_window", type=int, default=200, help="Steps for ramp-average before collapses")
    args = ap.parse_args()

    cfg = Config(
        dim=int(args.dim),
        steps=int(args.steps),
        dt=float(args.dt),
        omega=float(args.omega),
        coupling=float(args.coupling),
        temperature=float(args.temperature),
        threshold_bits=float(args.threshold),
        seed=int(args.seed),
        R_order=int(args.order),
        out_prefix=str(args.out_prefix),
        include_quiet_symbol=(not bool(args.no_quiet_symbol)),
        record_mode=str(args.record_mode),
        init_superposition=(False if args.basis_ground else True),
        amp_damp_gamma=float(args.amp_damp),
        amp_damp_target=int(args.amp_damp_target),
        dephase_sigma=float(args.dephase),
        ramp_window=int(args.ramp_window),
    )

    summary, ts, hol, H_meas, records = simulate(cfg)

    # Print concise summary to stdout
    print(
        f"[Holonomy Decoherence] dim={summary['dim']} steps={summary['steps']} "
        f"thr_bits={summary['threshold_bits']} coupling={summary['coupling']} T={summary['temperature']} "
        f"collapses={summary['collapses']} hol_rate(record)={summary['klrate_holonomy_bits_per_step']:.6g} b/step"
    )

    path = write_outputs(cfg, summary, ts, hol, H_meas, records)
    print(f"Saved summary and trace under {os.path.dirname(path)}/ with prefix {os.path.basename(path).split('.txt')[0]}")


if __name__ == "__main__":
    main()
