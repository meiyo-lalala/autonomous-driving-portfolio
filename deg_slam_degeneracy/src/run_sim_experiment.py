"""Experiment 1: synthetic parking garage / plain corridor.

Three pipelines on an identical lidar stream:
  odom_only : wheel odometry only (how far the prior alone gets you)
  naive     : scan-to-map registration, no degeneracy handling
  deg_aware : same registration, plus a soft odometry prior along the
              eigen-direction of weakest constraint (degslam.fusion)
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degslam.scan_sim import GarageWorld, make_truth_trajectory      # noqa: E402
from degslam.odom import Odometer, integrate                          # noqa: E402
from degslam.scan_matcher import MapModel, align                      # noqa: E402
from degslam.fusion import refine_with_prior, weak_translation_dirs   # noqa: E402
from degslam.metrics import ate, rpe, path_length                    # noqa: E402
from degslam.geometry import se2, to_pose, transform_points          # noqa: E402


def run(mode, scans, odom_deltas, start_pose, cond_thresh=40.0,
        prior_weight=0.05, prior_ratio=0.03, local_radius=0.0, failsafe=True):
    mapping = MapModel(voxel=0.06)
    poses = [np.array(start_pose, float)]
    mapping.add(transform_points(se2(*poses[0]), scans[0]))

    rows = []
    for i in range(1, len(scans)):
        init = Odometer.compose(poses[-1], odom_deltas[i - 1])
        t0 = time.time()
        if mode == "odom_only":
            pose = init
            info = None
            inl = rmse = float("nan")
            triggered = 0
        else:
            res = align(scans[i], init, mapping, max_corr_dist=0.7,
                        iters_coarse=12, iters_fine=6)
            pose = res["pose"]
            info = res["info"]
            inl, rmse = res["inlier_ratio"], res["rmse"]
            triggered = 0
            if failsafe and (inl < 0.35 or rmse > 0.25):
                # the scan does not explain the map: trust the odometry prior
                # rather than let a bad match drag the map into the wrong place
                pose = init
            if mode == "deg_aware":
                dirs, lam_w, lam_tt = weak_translation_dirs(res["Hmat"], ratio=prior_ratio)
                if dirs:
                    pose = refine_with_prior(
                        scans[i], init, mapping, np.column_stack(dirs),
                        lam_prior=prior_weight * lam_tt,
                        max_corr_dist=0.7, iters=8)
                    triggered = len(dirs)
        poses.append(pose)
        mapping.add(transform_points(se2(*pose), scans[i]))
        if local_radius > 0:
            mapping.trim(pose[:2], radius=local_radius)

        rows.append({
            "mode": mode, "frame": i,
            "cond": float(info["cond"]) if info else float("nan"),
            "lam_min": float(info["lam_min"]) if info else float("nan"),
            "lam_max": float(info["lam_max"]) if info else float("nan"),
            "direction_deg": float(info["direction_deg"]) if info else float("nan"),
            "inlier_ratio": float(inl), "rmse": float(rmse),
            "triggered": triggered,
            "ipt_ms": (time.time() - t0) * 1e3,
        })
        if i % 40 == 0:
            print(f"  [{mode}] frame {i}/{len(scans)-1}", flush=True)
    return poses, rows, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["garage", "corridor", "glass"], default="garage")
    ap.add_argument("--out", default="results")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--laps", type=float, default=2.0)
    ap.add_argument("--cond-thresh", type=float, default=40.0)
    ap.add_argument("--prior-weight", type=float, default=0.05)
    ap.add_argument("--prior-ratio", type=float, default=0.03)
    ap.add_argument("--max-range", type=float, default=20.0)
    ap.add_argument("--local-map", type=float, default=0.0,
                    help="keep a local map window of this radius (0 = cumulative map)")
    ap.add_argument("--no-failsafe", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, f"{args.scenario}_r{int(args.max_range)}")
    os.makedirs(out_dir, exist_ok=True)

    if args.scenario == "garage":
        world = GarageWorld(with_pillars=True, with_stalls=True)
    elif args.scenario == "corridor":
        world = GarageWorld(with_pillars=False, with_stalls=False)
    else:
        # one wall is glass / dark paint -> no echoes from that side
        world = GarageWorld(with_pillars=False, with_stalls=False, glass_side=True)

    truth = make_truth_trajectory(rate=args.rate, speed=args.speed, laps=args.laps)
    if args.frames:
        truth = truth[: args.frames]

    rng = np.random.default_rng(7)
    print(f"[{args.scenario}] simulating {len(truth)} frames ...", flush=True)
    scans = [world.to_scan(p, n_beams=720, max_range=args.max_range, noise_std=0.02, rng=rng)
             for p in truth]
    dt = 1.0 / args.rate

    odom = Odometer(scale=1.025, yaw_bias_deg=0.25, seed=11)
    deltas = [odom.delta(truth[i - 1], truth[i], dt) for i in range(1, len(truth))]
    odom_poses = integrate(truth, Odometer(scale=1.025, yaw_bias_deg=0.25, seed=11), dt)

    gt_xy = np.array([p[:2] for p in truth])
    traj_len = path_length(truth)
    print(f"trajectory {traj_len:.1f} m, {len(truth)} frames, "
          f"{len(world.segments)} segments", flush=True)

    results = {"odom_only": (odom_poses, [], None)}
    for mode in ("naive", "deg_aware"):
        print(f"running {mode} ...", flush=True)
        poses, rows, mapping = run(mode, scans, deltas, truth[0],
                                   args.cond_thresh, args.prior_weight, args.prior_ratio,
                                   local_radius=args.local_map,
                                   failsafe=not args.no_failsafe)
        results[mode] = (poses, rows, mapping)

    summary, all_rows = [], []
    for mode, (poses, rows, _) in results.items():
        est_xy = np.array([p[:2] for p in poses])
        ate_al, ate_max = ate(est_xy, gt_xy, align=True)
        ate_raw, _ = ate(est_xy, gt_xy, align=False)
        rpe_t, rpe_r = rpe(poses, truth, delta=5)
        deg_frac = float(np.mean([r["cond"] > args.cond_thresh for r in rows])) if rows else 0.0
        trig = int(np.sum([r["triggered"] for r in rows])) if rows else 0
        med_dir = float(np.nanmedian([r["direction_deg"] for r in rows])) if rows else float("nan")
        summary.append(dict(mode=mode, ate_aligned=ate_al, ate_unaligned=ate_raw,
                            ate_max=ate_max, rpe_trans=rpe_t, rpe_rot_deg=rpe_r,
                            degenerate_fraction=deg_frac, triggered_frames=trig,
                            median_degenerate_dir=med_dir))
        all_rows.extend(rows)

    with open(os.path.join(out_dir, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    payload = {
        "summary": summary, "gt": np.array(truth), "odom": np.array(odom_poses),
        "naive": np.array(results["naive"][0]), "deg_aware": np.array(results["deg_aware"][0]),
        "segments": np.array(world.segments, dtype=object), "frames": all_rows,
        "scenario": args.scenario, "traj_len": traj_len,
    }
    np.save(os.path.join(out_dir, "results.npy"), payload, allow_pickle=True)

    print(f"\n=== summary ({args.scenario}) ===")
    for s in summary:
        print(f"{s['mode']:<10} ATE(aligned)={s['ate_aligned']:.3f}  "
              f"raw={s['ate_unaligned']:.3f}  max={s['ate_max']:.3f}  "
              f"RPEt={s['rpe_trans']:.4f}  RPEr={s['rpe_rot_deg']:.3f}deg  "
              f"deg%={s['degenerate_fraction']*100:.1f}  trig={s['triggered_frames']}  "
              f"dir={s['median_degenerate_dir']:.1f}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"scenario": args.scenario, "trajectory_length_m": traj_len,
                   "frames": len(truth), "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
