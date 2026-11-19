"""Overlay Cartographer's occupancy grid with the ground-truth trajectory."""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def read_pgm(path):
    from PIL import Image
    return np.array(Image.open(path).convert("L"))


def read_yaml(path):
    out = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def overlay(case, ax, title):
    meta = read_yaml(os.path.join(RES, "cartographer", case, "map.yaml"))
    img = read_pgm(os.path.join(RES, "cartographer", case, "map.pgm"))
    res = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"].strip("[]").split(",")]
    x0, y0 = origin[0], origin[1]
    extent = [x0, x0 + img.shape[1] * res, y0, y0 + img.shape[0] * res]
    ax.imshow(img, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=255,
              interpolation="nearest")
    gt_path = os.path.join(RES, case, "results.npy")
    if os.path.exists(gt_path):
        d = np.load(gt_path, allow_pickle=True).item()
        gt = d["gt"]
        # Cartographer starts the map at the robot's first pose (0,0), so shift
        # the ground truth into the map frame before overlaying it.
        gt = gt[:, :2] - gt[0, :2]
        ax.plot(gt[:, 0], gt[:, 1], color="#d62728", lw=1.6, label="ground truth")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main():
    cases = [("garage_r20", "Cartographer map — garage, odometry-fed")]
    fig, axes = plt.subplots(1, len(cases), figsize=(7, 5))
    if len(cases) == 1:
        axes = [axes]
    for ax, (case, title) in zip(axes, cases):
        if os.path.exists(os.path.join(RES, "cartographer", case, "map.pgm")):
            overlay(case, ax, title)
        else:
            ax.set_axis_off()
    fig.tight_layout()
    out = os.path.join(RES, "figures", "fig4_cartographer_map.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
