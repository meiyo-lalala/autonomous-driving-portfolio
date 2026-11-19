"""Figures for the synthetic sweep."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

CASES = [("garage_r20", "Garage, 20 m lidar"),
         ("corridor_r20", "Plain corridor, 20 m"),
         ("glass_r8", "Glass wall, 8 m lidar")]


def load(case):
    p = os.path.join(RES, case, "results.npy")
    return np.load(p, allow_pickle=True).item() if os.path.exists(p) else None


def draw_world(ax, segments, color="#c8ced6"):
    for a, b in segments:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.7, zorder=1)


def fig_trajectories():
    fig, axes = plt.subplots(1, len(CASES), figsize=(16, 4.2))
    for ax, (case, title) in zip(axes, CASES):
        d = load(case)
        if d is None:
            ax.set_axis_off()
            continue
        draw_world(ax, d["segments"])
        gt = d["gt"][:, :2]
        ax.plot(gt[:, 0], gt[:, 1], color="#111", lw=2.0, label="ground truth", zorder=3)
        ax.plot(d["odom"][:, 0], d["odom"][:, 1], color="#e08b2b", lw=1.2,
                ls="--", label="odometry only", zorder=2)
        ax.plot(d["naive"][:, 0], d["naive"][:, 1], color="#2f6fdb", lw=1.3,
                label="scan matching (naive)", zorder=4)
        ax.plot(d["deg_aware"][:, 0], d["deg_aware"][:, 1], color="#c0392b", lw=1.3,
                ls=":", label="degeneracy-aware", zorder=5)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_trajectories.png"), dpi=150)
    plt.close(fig)


def fig_cond():
    fig, axes = plt.subplots(1, len(CASES), figsize=(16, 3.6), sharey=True)
    for ax, (case, title) in zip(axes, CASES):
        d = load(case)
        if d is None:
            continue
        rows = [r for r in d["frames"] if r["mode"] == "naive"]
        cond = np.array([r["cond"] for r in rows])
        ax.semilogy(cond, color="#2f6fdb", lw=0.9)
        ax.axhline(40.0, color="#c0392b", lw=1.0, ls="--", label="threshold (40)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("frame")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("condition number $\\lambda_{max}/\\lambda_{min}$")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_condition_number.png"), dpi=150)
    plt.close(fig)


def fig_direction_error():
    fig, axes = plt.subplots(2, len(CASES), figsize=(16, 6), sharex=True)
    for col, (case, title) in enumerate(CASES):
        d = load(case)
        if d is None:
            continue
        rows = [r for r in d["frames"] if r["mode"] == "naive"]
        gt = d["gt"]
        naive = d["naive"]
        n = min(len(rows), len(gt), len(naive))
        ax = axes[0, col]
        ax.plot([r["direction_deg"] for r in rows[:n]], color="#6a4c93", lw=0.9)
        ax.set_ylim(-10, 190)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("degenerate dir [deg]")
        ax.grid(alpha=0.15)

        ax2 = axes[1, col]
        err_n = np.linalg.norm(naive[:n, :2] - gt[:n, :2], axis=1)
        err_o = np.linalg.norm(d["odom"][:n, :2] - gt[:n, :2], axis=1)
        ax2.plot(err_o, color="#e08b2b", lw=1.0, ls="--", label="odom")
        ax2.plot(err_n, color="#2f6fdb", lw=1.0, label="naive")
        ax2.set_xlabel("frame")
        ax2.set_ylabel("position error [m]")
        ax2.grid(alpha=0.15)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_direction_and_error.png"), dpi=150)
    plt.close(fig)


def table():
    rows = []
    for case, title in CASES:
        d = load(case)
        if d is None:
            continue
        for s in d["summary"]:
            rows.append(dict(case=case, scenario=title, **s))
    import csv
    with open(os.path.join(RES, "summary_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'case':<14}{'mode':<12}{'ATE':>8}{'RPE_t':>8}{'deg%':>7}{'trig':>6}")
    for r in rows:
        print(f"{r['case']:<14}{r['mode']:<12}{r['ate_aligned']:>8.3f}"
              f"{r['rpe_trans']:>8.4f}{r['degenerate_fraction']*100:>7.1f}"
              f"{r['triggered_frames']:>6}")


if __name__ == "__main__":
    fig_trajectories()
    fig_cond()
    fig_direction_error()
    table()
    print("figures ->", FIG)
