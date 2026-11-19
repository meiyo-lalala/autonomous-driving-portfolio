"""Minimal SE(2) helpers used across the project."""
import numpy as np


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def se2(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def se2_inv(T):
    R = T[:2, :2]
    t = T[:2, 2]
    out = np.eye(3)
    out[:2, :2] = R.T
    out[:2, 2] = -R.T @ t
    return out


def to_pose(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def delta_to_T(delta):
    """delta = (dx, dy, dtheta) in the global frame -> SE(2) left perturbation."""
    return se2(delta[0], delta[1], delta[2])


def transform_points(T, pts):
    """pts: (N,2) -> (N,2)"""
    return pts @ T[:2, :2].T + T[:2, 2]


def invert_pose(pose):
    T = se2_inv(se2(*pose))
    return to_pose(T)


def relative_pose(a, b):
    """Pose of b expressed in frame a."""
    return to_pose(se2_inv(se2(*a)) @ se2(*b))
