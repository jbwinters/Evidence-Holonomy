#!/usr/bin/env python3
"""
Explainer animations for the README (animated GIFs under figures/).

- forward_vs_reverse.gif: a biased three-state ring and an unbiased one, each
  played forward and backward, with the time-reversal estimate as it accumulates.
- evidence_accumulation.gif: the log-likelihood ratio log2 P(x)/P_rev(x) of one
  recording growing step by step; its slope is the entropy production rate.
- hidden_observation.gif: one hidden ring seen fully, through a noisy channel,
  and with two states merged, and the irreversibility each record reveals.

All trajectories come from uec.markov samplers with fixed seeds, and all
estimates from uec.holonomy.klrate_holonomy_time_reversal_markov.

Usage: PYTHONPATH=src python scripts/make_animations.py [--only NAME ...]
Needs matplotlib and Pillow (pip install -e ".[paper,image]").
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from uec.holonomy import klrate_holonomy_time_reversal_markov  # noqa: E402
from uec.markov import ring_chain_3, ring_ep_bits, sample_markov  # noqa: E402

OUT = Path("figures")
DPI = 80
FPS = 12
STATE_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
P_FWD, Q_BWD = 0.3, 0.05  # biased ring: forward 0->1->2->0 with p, backward with q
SIGMA = ring_ep_bits(P_FWD, Q_BWD)
NODE_XY = np.array([[math.cos(a), math.sin(a)] for a in (math.pi / 2, math.pi / 2 - 2 * math.pi / 3,
                                                          math.pi / 2 + 2 * math.pi / 3)])

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5})


def _save(anim: FuncAnimation, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(anim._fig)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB)")


def _direction(a: int, b: int) -> int:
    """+1 for a forward ring step (a -> a+1 mod 3), -1 for backward, 0 for staying."""
    d = (b - a) % 3
    return 1 if d == 1 else (-1 if d == 2 else 0)


# ------------------------------------------------------------------
# 1. Forward vs reverse
# ------------------------------------------------------------------


def _draw_ring(ax, title: str):
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5)
    for i in range(3):
        j = (i + 1) % 3
        ax.annotate("", xy=NODE_XY[j] * 0.8 + NODE_XY[i] * 0.2, xytext=NODE_XY[i] * 0.8 + NODE_XY[j] * 0.2,
                    arrowprops=dict(arrowstyle="-", color="#cccccc", lw=1.5))
    for i, (x, y) in enumerate(NODE_XY):
        ax.add_patch(plt.Circle((x, y), 0.22, color=STATE_COLORS[i], alpha=0.25, zorder=1))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=9, color="#333333", zorder=2)
    trail, = ax.plot([], [], "-", color="#555555", lw=1.2, alpha=0.5, zorder=3)
    dot, = ax.plot([], [], "o", color="black", ms=10, zorder=4)
    counter = ax.text(0, -1.32, "", ha="center", va="bottom", fontsize=8.5, linespacing=1.3)
    return trail, dot, counter


def forward_vs_reverse(steps: int = 960, per_frame: int = 4) -> None:
    rng = np.random.default_rng(11)
    biased = [int(s) for s in sample_markov(ring_chain_3(P_FWD, Q_BWD), n=steps + 1, rng=rng)]
    fair_p = (P_FWD + Q_BWD) / 2
    unbiased = [int(s) for s in sample_markov(ring_chain_3(fair_p, fair_p), n=steps + 1, rng=rng)]
    recordings = [
        ("biased ring\nplayed forward", biased),
        ("biased ring\nplayed backward", biased[::-1]),
        ("unbiased ring\nplayed forward", unbiased),
        ("unbiased ring\nplayed backward", unbiased[::-1]),
    ]
    ts = np.arange(16, steps + 1, per_frame)
    est_b = [klrate_holonomy_time_reversal_markov(biased[: t + 1], k=3, R=1) for t in ts]
    est_u = [klrate_holonomy_time_reversal_markov(unbiased[: t + 1], k=3, R=1) for t in ts]

    fig = plt.figure(figsize=(9.6, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.3, wspace=0.12)
    rings = [_draw_ring(fig.add_subplot(gs[0, i]), title) for i, (title, _) in enumerate(recordings)]
    ax = fig.add_subplot(gs[1, :])
    ax.axhline(SIGMA, color="#C44E52", ls="--", lw=1)
    ax.axhline(0, color="#4C72B0", ls="--", lw=1)
    ax.text(steps, SIGMA - 0.04, f"biased ring: σ = {SIGMA:.2f} bits/step", ha="right", va="top",
            color="#C44E52", fontsize=9)
    ax.text(steps, 0.03, "unbiased ring: σ = 0", ha="right", va="bottom", color="#4C72B0", fontsize=9)
    line_b, = ax.plot([], [], color="#C44E52", lw=2)
    line_u, = ax.plot([], [], color="#4C72B0", lw=2)
    ax.set_xlim(0, steps)
    ax.set_ylim(-0.1, max(1.0, max(est_b) * 1.05))
    ax.set_xlabel("steps observed")
    ax.set_ylabel("estimate (bits/step)")
    ax.set_title("How distinguishable is each recording from its own reversal?")
    fig.suptitle("Irreversibility = telling forward from backward", fontsize=12, y=0.99)

    def update(frame):
        t = min(steps, frame * per_frame)
        artists = []
        for (title, seq), (trail, dot, counter) in zip(recordings, rings):
            lo = max(0, t - per_frame)
            pts = NODE_XY[seq[lo: t + 1]]
            trail.set_data(pts[:, 0], pts[:, 1])
            dot.set_data([NODE_XY[seq[t]][0]], [NODE_XY[seq[t]][1]])
            dirs = [_direction(a, b) for a, b in zip(seq[:t], seq[1: t + 1])]
            counter.set_text(f"steps 0→1→2: {dirs.count(1)}\nsteps 2→1→0: {dirs.count(-1)}")
            artists += [trail, dot, counter]
        k = int(np.searchsorted(ts, t, side="right"))
        line_b.set_data(ts[:k], est_b[:k])
        line_u.set_data(ts[:k], est_u[:k])
        return artists + [line_b, line_u]

    frames = list(range(0, steps // per_frame + 1)) + [steps // per_frame] * FPS * 2
    _save(FuncAnimation(fig, update, frames=frames, blit=False), "forward_vs_reverse.gif")


# ------------------------------------------------------------------
# 2. Evidence accumulating
# ------------------------------------------------------------------


def evidence_accumulation(steps: int = 600, per_frame: int = 4) -> None:
    T = ring_chain_3(P_FWD, Q_BWD)
    x = [int(s) for s in sample_markov(T, n=steps + 1, rng=np.random.default_rng(5))]
    inc = np.array([math.log2(T[a, b] / T[b, a]) for a, b in zip(x[:-1], x[1:])])
    cum = np.concatenate([[0.0], np.cumsum(inc)])
    t_axis = np.arange(steps + 1)

    fig, (ax_s, ax) = plt.subplots(2, 1, figsize=(9.6, 5.4), gridspec_kw={"height_ratios": [1, 3.2]})
    window = 40
    ax_s.set_xlim(-0.5, window - 0.5)
    ax_s.set_ylim(-0.6, 0.6)
    ax_s.axis("off")
    cells = [ax_s.add_patch(plt.Rectangle((i - 0.45, -0.45), 0.9, 0.9, color="white")) for i in range(window)]
    labels = [ax_s.text(i, 0, "", ha="center", va="center", fontsize=8, color="white") for i in range(window)]
    ax_s.set_title("the recording (latest steps on the right; each step moves 0→1→2→0 forward, back, or stays)",
                   fontsize=9.5)

    ax.plot(t_axis, SIGMA * t_axis, "--", color="#C44E52", lw=1.2, label=f"σ · t   (σ = {SIGMA:.2f} bits/step)")
    line, = ax.plot([], [], color="black", lw=2, label="log₂ P(recording) / P(recording reversed)")
    head, = ax.plot([], [], "o", color="black", ms=6)
    note = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlim(0, steps)
    ax.set_ylim(min(-5.0, cum.min() * 1.1), max(cum.max(), SIGMA * steps) * 1.08)
    ax.set_xlabel("step t")
    ax.set_ylabel("evidence for 'forward' (bits)")
    ax.legend(loc="lower right")
    ax.set_title("Each forward step adds log₂(0.30/0.05) = +2.58 bits; each backward step subtracts 2.58; "
                 "staying adds 0", fontsize=9.5)

    def update(frame):
        t = min(steps, frame * per_frame)
        seg = x[max(0, t - window + 1): t + 1]
        off = window - len(seg)
        for i in range(window):
            j = i - off
            if j < 0:
                cells[i].set_color("white")
                labels[i].set_text("")
            else:
                cells[i].set_color(STATE_COLORS[seg[j]])
                labels[i].set_text(str(seg[j]))
        line.set_data(t_axis[: t + 1], cum[: t + 1])
        head.set_data([t], [cum[t]])
        if t > 0:
            d = _direction(x[t - 1], x[t])
            what = {1: "forward step", -1: "backward step", 0: "stay"}[d]
            note.set_text(f"t = {t}:  {what}, {inc[t - 1]:+.2f} bits\n"
                          f"total {cum[t]:.1f} bits  →  {cum[t] / t:.2f} bits/step so far")
        return cells + labels + [line, head, note]

    frames = list(range(0, steps // per_frame + 1)) + [steps // per_frame] * FPS * 2
    _save(FuncAnimation(fig, update, frames=frames, blit=False), "evidence_accumulation.gif")


# ------------------------------------------------------------------
# 3. What you observe limits what you can see
# ------------------------------------------------------------------


def hidden_observation(n_max: int = 2**15) -> None:
    A = ring_chain_3(P_FWD, Q_BWD)
    noisy = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    rng = np.random.default_rng(21)
    hidden = [int(s) for s in sample_markov(A, n=n_max, rng=rng)]
    rng_obs = np.random.default_rng(22)
    obs_noisy = [int(rng_obs.choice(3, p=noisy[z])) for z in hidden]
    obs_lumped = [0 if z == 0 else 1 for z in hidden]
    records = [
        ("full view of the ring", hidden, 3, "#C44E52"),
        ("noisy view: right state 80% of the time", obs_noisy, 3, "#8172B2"),
        ("merged view: states 1 and 2 look the same", obs_lumped, 2, "#937860"),
    ]
    ns = np.unique(np.logspace(7, 15, 50, base=2).astype(int))
    curves = [[klrate_holonomy_time_reversal_markov(seq[:n], k=k, R=3) for n in ns] for _, seq, k, _ in records]
    for (label, _, _, _), c in zip(records, curves):
        print(f"  {label}: final estimate {c[-1]:.3f}")

    fig = plt.figure(figsize=(9.6, 5.8))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.42, 0.42, 0.42, 2.6], hspace=0.55)
    window = 48
    strips = []
    for r, (label, seq, k, color) in enumerate(records):
        a = fig.add_subplot(gs[r])
        a.set_xlim(-0.5, window - 0.5)
        a.set_ylim(-0.5, 0.5)
        a.axis("off")
        a.set_title(label, fontsize=9, loc="left", color=color, pad=2)
        strips.append([a.add_patch(plt.Rectangle((i - 0.45, -0.45), 0.9, 0.9, color="white")) for i in range(window)])
    lumped_colors = [STATE_COLORS[0], "#999999"]

    ax = fig.add_subplot(gs[3])
    ax.axhline(SIGMA, color="black", ls="--", lw=1)
    ax.text(ns[0], SIGMA + 0.02, f"hidden ring's entropy production σ = {SIGMA:.2f} bits/step", va="bottom",
            fontsize=9)
    lines = [ax.plot([], [], color=color, lw=2, marker="o", ms=3, label=label)[0]
             for label, _, _, color in records]
    ax.set_xscale("log", base=2)
    ax.set_xlim(ns[0], ns[-1])
    ax.set_ylim(-0.05, SIGMA * 1.25)
    ax.set_xlabel("steps observed")
    ax.set_ylabel("estimate (bits/step)")
    ax.legend(loc="center right", fontsize=8.5)
    fig.suptitle("What you observe limits the irreversibility you can see", fontsize=12, y=0.995)

    n_frames = len(ns)
    hold = FPS * 2

    def update(frame):
        f = min(frame, n_frames - 1)
        start = frame * 3
        for (label, seq, k, _), cells in zip(records, strips):
            for i, cell in enumerate(cells):
                s = seq[start + i]
                cell.set_color(lumped_colors[s] if k == 2 else STATE_COLORS[s])
        for line, c in zip(lines, curves):
            line.set_data(ns[: f + 1], c[: f + 1])
        return [c for cells in strips for c in cells] + lines

    _save(FuncAnimation(fig, update, frames=list(range(n_frames + hold)), blit=False), "hidden_observation.gif")


ANIMATIONS = {
    "forward_vs_reverse": forward_vs_reverse,
    "evidence_accumulation": evidence_accumulation,
    "hidden_observation": hidden_observation,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=sorted(ANIMATIONS))
    args = ap.parse_args()
    for name in args.only or ANIMATIONS:
        print(f"== {name}")
        ANIMATIONS[name]()
