"""Degeneracy-aware refinement: anchor weak directions to odometry.

Instead of shrinking the update (which simply freezes the pose), we add a soft
prior along the eigen-directions of weak constraint:

    min   ||r(pose)||^2  +  lambda * || V^T (pose_odom - pose) ||^2

with V the columns of the (near-)null-space of H.  Gauss-Newton gives

    (H + lambda V V^T) d = -J^T r + lambda V ( V^T (pose_odom - pose) )

so the weakly observable directions follow odometry while the well-constrained
directions keep trusting the lidar.  Several directions may be weak at once
(e.g. a glass wall that returns no echoes drops the lateral constraint too),
which is why V can hold more than one column.
"""
import numpy as np

from .geometry import se2, transform_points, wrap_angle
from .scan_matcher import _huber, _limit_step


def refine_with_prior(scan, pose_odom, map_model, directions, lam_prior,
                      max_corr_dist=0.7, iters=6, huber_delta=0.25,
                      max_step=0.25, max_rot=0.05):
    pose = np.array(pose_odom, float)
    V = np.asarray(directions, float)
    if V.ndim == 1:
        V = V[:, None]
    # orthonormalise the columns
    q, _ = np.linalg.qr(V)

    for _ in range(iters):
        T = se2(*pose)
        moved = transform_points(T, scan)
        d, idx = map_model.nearest(moved)
        valid = d < max_corr_dist
        if len(map_model.linearity):
            valid &= np.abs(map_model.linearity[idx]) > 0.4
        if valid.sum() < 15:
            break
        p = moved[valid]
        q_pts = map_model.points[idx[valid]]
        n = map_model.normals[idx[valid]]
        r = np.einsum("ij,ij->i", p - q_pts, n)
        rel = p - T[:2, 2]
        jac = np.column_stack([n[:, 0], n[:, 1], n[:, 0] * rel[:, 1] - n[:, 1] * rel[:, 0]])
        w = _huber(r, huber_delta)
        H = jac.T @ (w[:, None] * jac) + 1e-9 * np.eye(3)
        b = jac.T @ (w * r)

        H2 = H + lam_prior * (V @ V.T)
        rhs = -b + lam_prior * (V @ (V.T @ (np.asarray(pose_odom, float) - pose)))
        delta = _limit_step(np.linalg.solve(H2, rhs), max_step, max_rot)
        pose[0] += delta[0]
        pose[1] += delta[1]
        pose[2] = wrap_angle(pose[2] + delta[2])
        if np.hypot(delta[0], delta[1]) < 2e-4 and abs(delta[2]) < 2e-4:
            break
    return pose


def weak_translation_dirs(H, ratio=0.03, max_dirs=2):
    """Degenerate directions of the *translational* 2x2 block of H.

    Rotational observability in 2D lidar comes from far-away points and stays
    healthy even in a corridor; the failure is almost always translational
    (sliding along the aisle).  Working on the 2x2 block keeps the heading out
    of the prior, which would otherwise drag the pose towards a drifting
    yaw estimate.
    """
    H = np.asarray(H, float)
    Htt = 0.5 * (H[:2, :2] + H[:2, :2].T)
    w, V = np.linalg.eigh(Htt)          # ascending
    lam_max = max(float(w[-1]), 1e-9)
    weak = [i for i in range(2) if w[i] < ratio * lam_max]
    weak = sorted(weak, key=lambda i: w[i])[:max_dirs]
    dirs = [np.array([V[0, i], V[1, i], 0.0]) for i in weak]
    return dirs, w, lam_max
