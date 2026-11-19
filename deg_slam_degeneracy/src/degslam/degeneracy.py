"""Degeneracy analysis of a 3-DoF (x, y, theta) scan-matching information matrix.

The method follows the classical eigenvalue test for optimization-based state
estimation (Zhang et al., "On Degeneracy of Optimization-based State Estimation
Problems"): the Hessian H = J^T J of the registration problem is decomposed, and
the smallest eigenvalue together with its eigenvector tells us (a) how weakly a
direction is constrained and (b) which direction that is.
"""
import numpy as np


def analyze(H, eig_floor=1e-9):
    """Decompose a 3x3 information matrix.

    Returns a dict with eigenvalues (descending), eigenvectors, condition number,
    degenerate direction (degrees, translation part only) and the rotation share
    of that direction.
    """
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)          # ascending
    w = np.clip(w, 0.0, None)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]

    lam_max = max(float(w[0]), eig_floor)
    lam_min = max(float(w[-1]), eig_floor)
    cond = lam_max / lam_min

    v = V[:, -1]                      # eigenvector of the weakest direction
    tx, ty = float(v[0]), float(v[1])
    transl_energy = float(np.hypot(tx, ty))
    if transl_energy > 1e-8:
        direction = float(np.degrees(np.arctan2(ty, tx)))
        if direction < 0:
            direction += 180.0        # direction is a line, not a ray
    else:
        direction = float("nan")

    return {
        "eigvals": w,
        "eigvecs": V,
        "cond": cond,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "direction_deg": direction,
        "translational_energy": transl_energy,
        "rotational_energy": float(abs(v[2]),
        ),
    }


def clamp_eigenvalues(H, floor_ratio=0.0, floor_abs=None):
    """Raise the weakest eigenvalues to a floor, in the eigenbasis.

    This is the cheapest way to express "trust the prior (odometry) along the
    degenerate direction": the update along that direction is shrunk, so the
    optimizer keeps the prediction instead of sliding along an unobservable axis.
    """
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    lam_max = max(float(w[-1]), 1e-9)
    floor = floor_abs if floor_abs is not None else floor_ratio * lam_max
    w_new = np.maximum(w, floor)
    return V @ np.diag(w_new) @ V.T


def is_degenerate(info, cond_thresh=20.0, lam_min_thresh=None):
    if lam_min_thresh is not None and info["lam_min"] < lam_min_thresh:
        return True
    return info["cond"] > cond_thresh
