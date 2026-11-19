"""Metrics for 2D trajectories: ATE / RPE with SE(2) alignment."""
import numpy as np

from .geometry import se2, se2_inv, to_pose


def align_umeyama2d(est_xy, gt_xy):
    """Return the rigid SE(2) transform that best maps est onto gt (translation+rotation)."""
    mu_e = est_xy.mean(axis=0)
    mu_g = gt_xy.mean(axis=0)
    A = est_xy - mu_e
    B = gt_xy - mu_g
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    theta = np.arctan2(R[1, 0], R[0, 0])
    t = mu_g - R @ mu_e
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = t
    return T, theta


def ate(est_xy, gt_xy, align=True):
    if len(est_xy) < 3:
        return float("nan"), 0.0
    if align:
        T, _ = align_umeyama2d(est_xy, gt_xy)
        est_xy = est_xy @ T[:2, :2].T + T[:2, 2]
    err = np.linalg.norm(est_xy - gt_xy, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), float(np.max(err))


def ate_without_alignment(est_xy, gt_xy):
    err = np.linalg.norm(est_xy - gt_xy, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), float(np.max(err))


def rpe(est_poses, gt_poses, delta=5):
    """Relative pose error over a fixed frame gap (translation RMSE + rotation RMSE)."""
    trans_err, rot_err = [], []
    for i in range(0, len(est_poses) - delta):
        Te = se2_inv(se2(*est_poses[i])) @ se2(*est_poses[i + delta])
        Tg = se2_inv(se2(*gt_poses[i])) @ se2(*gt_poses[i + delta])
        rel = to_pose(se2_inv(Tg) @ Te)
        trans_err.append(np.hypot(rel[0], rel[1]))
        rot_err.append(abs(rel[2]))
    if not trans_err:
        return float("nan"), float("nan")
    return (float(np.sqrt(np.mean(np.square(trans_err)))),
            float(np.sqrt(np.mean(np.square(rot_err))))
            * 180.0 / np.pi)


def path_length(poses):
    xy = np.array([p[:2] for p in poses])
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
