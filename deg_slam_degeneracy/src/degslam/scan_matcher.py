"""Scan-to-map registration in SE(2), two-stage and degeneracy-aware.

Stage 1: robust point-to-point ICP (stable with sparse, noisy scans, no normals).
Stage 2: point-to-line refinement, whose information matrix is the one used for
         degeneracy analysis (eigen-decomposition of J^T J).

Reference for the degeneracy test: Zhang, Kaess, Singh, "On Degeneracy of
Optimization-based State Estimation Problems", IROS 2016.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import se2, transform_points, wrap_angle
from .degeneracy import analyze, clamp_eigenvalues


def voxel_downsample(points, voxel=0.08):
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


class MapModel:
    """Point map with cached KD-tree and PCA normals."""

    def __init__(self, voxel=0.06, knn=10, rebuild_every=2):
        self.voxel = voxel
        self.knn = knn
        self.rebuild_every = rebuild_every
        self.points = np.empty((0, 2), dtype=float)
        self.normals = np.empty((0, 2), dtype=float)
        self.linearity = np.empty((0,), dtype=float)
        self.tree = None
        self._dirty = True
        self._age = 0
        self._tree_len = 0

    def add(self, world_points):
        if len(world_points) == 0:
            return
        merged = np.vstack([self.points, world_points]) if len(self.points) else world_points
        self.points = voxel_downsample(merged, self.voxel)
        self._dirty = True

    def trim(self, center, radius=18.0, min_points=80):
        """Keep a local window around the current pose.

        Without this, a re-visited corridor accumulates two copies of the same
        wall a few centimetres apart; the KD-tree then matches into the smear and
        the estimate slowly slides. Local windowing is what every production
        LIO front-end does (and Cartographer solves it with submaps + loop
        closure instead).
        """
        if len(self.points) == 0:
            return
        keep = np.linalg.norm(self.points - np.asarray(center, float), axis=1) < radius
        if keep.sum() < min_points:
            return
        self.points = self.points[keep]
        self._dirty = True

    def _estimate_normals(self):
        if len(self.points) < self.knn + 1:
            self.normals = np.zeros_like(self.points)
            self.linearity = np.zeros(len(self.points))
            return
        tree = cKDTree(self.points)
        _, idx = tree.query(self.points, k=self.knn)
        neigh = self.points[idx]
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / max(self.knn - 1, 1)
        w, v = np.linalg.eigh(cov)          # ascending
        self.normals = v[:, :, 0]           # smallest spread = surface normal
        # linearity: neighbourhood is line-like when lambda2 >> lambda1(local)
        lam = np.clip(w, 1e-12, None)
        self.linearity = 1.0 - lam[:, 0] / lam[:, 1]

    def build(self, force=False):
        self._age += 1
        need = (self.tree is None or force
                or (self._dirty and (self._age >= self.rebuild_every
                                     or len(self.points) != self._tree_len)))
        if need:
            self.tree = cKDTree(self.points)
            self._estimate_normals()
            self._tree_len = len(self.points)
            self._dirty = False
            self._age = 0

    def nearest(self, points):
        return self.tree.query(points, k=1)


def _huber(residuals, delta):
    a = np.abs(residuals)
    w = np.ones_like(a)
    big = a > delta
    w[big] = delta / a[big]
    return w


def _limit_step(delta, max_step, max_rot):
    n = np.hypot(delta[0], delta[1])
    if n > max_step:
        delta[:2] *= max_step / n
    delta[2] = np.clip(delta[2], -max_rot, max_rot)
    return delta


def align(scan, init_pose, map_model, max_corr_dist=0.7, iters_coarse=12,
          iters_fine=6, huber_delta=0.25, clamp_ratio=None, cond_thresh=20.0,
          max_step=0.25, max_rot=0.05):
    if len(map_model.points) < 10:
        return {"pose": np.array(init_pose, float), "info": analyze(np.zeros((3, 3))),
                "inlier_ratio": 0.0, "rmse": float("nan"), "stage": "none"}

    pose = np.array(init_pose, float)
    map_model.build()

    # ---- stage 1: point-to-point -------------------------------------------
    best = None
    H_last = np.zeros((3, 3))
    for _ in range(iters_coarse):
        T = se2(*pose)
        moved = transform_points(T, scan)
        d, idx = map_model.nearest(moved)
        valid = d < max_corr_dist
        if valid.sum() < 15:
            break
        p = moved[valid]
        q = map_model.points[idx[valid]]
        rel = p - T[:2, 2]
        r = (p - q).ravel()                              # (2M,)
        J = np.zeros((2 * len(p), 3))
        J[0::2, 0] = 1.0
        J[0::2, 2] = -rel[:, 1]
        J[1::2, 1] = 1.0
        J[1::2, 2] = rel[:, 0]
        w = np.repeat(_huber(np.linalg.norm(p - q, axis=1), huber_delta), 2)
        H = J.T @ (w[:, None] * J) + 1e-9 * np.eye(3)
        b = J.T @ (w * r)
        delta = _limit_step(-np.linalg.solve(H, b), max_step, max_rot)
        H_last = H
        pose[0] += delta[0]
        pose[1] += delta[1]
        pose[2] = wrap_angle(pose[2] + delta[2])
        best = (H, float(valid.mean()), float(np.sqrt(np.mean(np.sum((p - q) ** 2, axis=1))))
                if valid.sum() else float("nan"))
        if np.hypot(delta[0], delta[1]) < 2e-4 and abs(delta[2]) < 2e-4:
            break

    coarse_pose = pose.copy()
    coarse_inlier = best[1] if best else 0.0
    coarse_rmse = best[2] if best else float("nan")

    # ---- stage 2: point-to-line (this H drives the degeneracy analysis) ----
    info = analyze(best[0] if best else np.zeros((3, 3)))
    fine_inlier = coarse_inlier
    fine_rmse = coarse_rmse
    used_fine = False

    for _ in range(iters_fine):
        T = se2(*pose)
        moved = transform_points(T, scan)
        d, idx = map_model.nearest(moved)
        valid = (d < max_corr_dist)
        # normals are only trustworthy on line-like neighbourhoods
        valid &= np.abs(map_model.linearity[idx]) > 0.4 if len(map_model.linearity) else valid
        if valid.sum() < 15:
            break
        p = moved[valid]
        q = map_model.points[idx[valid]]
        n = map_model.normals[idx[valid]]
        diff = p - q
        r = np.einsum("ij,ij->i", diff, n)
        rel = p - T[:2, 2]
        jac = np.column_stack([n[:, 0], n[:, 1], n[:, 0] * rel[:, 1] - n[:, 1] * rel[:, 0]])
        w = _huber(r, huber_delta)
        H = jac.T @ (w[:, None] * jac) + 1e-9 * np.eye(3)
        b = jac.T @ (w * r)
        cand = analyze(H)
        delta = _limit_step(-np.linalg.solve(H, b), max_step, max_rot)
        pose[0] += delta[0]
        pose[1] += delta[1]
        pose[2] = wrap_angle(pose[2] + delta[2])
        info = cand
        H_last = H
        fine_inlier = float(valid.mean())
        fine_rmse = float(np.sqrt(np.mean(r ** 2)))
        used_fine = True
        if np.hypot(delta[0], delta[1]) < 2e-4 and abs(delta[2]) < 2e-4:
            break

    # guard: if point-to-line moved further from the map, keep the coarse result
    if used_fine:
        T = se2(*pose)
        moved = transform_points(T, scan)
        d2, _ = map_model.nearest(moved)
        if float(np.mean(d2 < max_corr_dist)) < 0.85 * coarse_inlier:
            pose = coarse_pose
            used_fine = False

    if clamp_ratio is not None and info["cond"] > cond_thresh:
        # degenerate direction: trust the odometry prior there. Re-solve from the
        # init pose with a floored information matrix.
        pose = _resolve_with_clamp(scan, init_pose, map_model, info, clamp_ratio,
                                   max_corr_dist, huber_delta, max_step, max_rot)

    return {"pose": pose, "info": info, "inlier_ratio": fine_inlier,
            "rmse": fine_rmse, "stage": "fine" if used_fine else "coarse",
            "coarse_rmse": coarse_rmse, "Hmat": H_last}


def _resolve_with_clamp(scan, init_pose, map_model, info, clamp_ratio,
                        max_corr_dist, huber_delta, max_step, max_rot, iters=4):
    """Re-run the fine stage with the information matrix eigen-clamped."""
    pose = np.array(init_pose, float)
    for _ in range(iters):
        T = se2(*pose)
        moved = transform_points(T, scan)
        d, idx = map_model.nearest(moved)
        valid = (d < max_corr_dist)
        if len(map_model.linearity):
            valid &= np.abs(map_model.linearity[idx]) > 0.4
        if valid.sum() < 15:
            break
        p = moved[valid]
        q = map_model.points[idx[valid]]
        n = map_model.normals[idx[valid]]
        r = np.einsum("ij,ij->i", p - q, n)
        rel = p - T[:2, 2]
        jac = np.column_stack([n[:, 0], n[:, 1], n[:, 0] * rel[:, 1] - n[:, 1] * rel[:, 0]])
        w = _huber(r, huber_delta)
        H = jac.T @ (w[:, None] * jac) + 1e-9 * np.eye(3)
        b = jac.T @ (w * r)
        H_c = clamp_eigenvalues(H, floor_ratio=clamp_ratio)
        delta = _limit_step(-np.linalg.solve(H_c, b), max_step, max_rot)
        pose[0] += delta[0]
        pose[1] += delta[1]
        pose[2] = wrap_angle(pose[2] + delta[2])
        if np.hypot(delta[0], delta[1]) < 2e-4 and abs(delta[2]) < 2e-4:
            break
    return pose
