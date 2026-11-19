"""Wheel-odometry model with the two errors that actually matter indoors:
scale error (wheel radius / slip) and yaw-rate bias.
"""
import numpy as np

from .geometry import relative_pose, wrap_angle, se2, to_pose


class Odometer:
    def __init__(self, scale=1.025, yaw_bias_deg=0.25, noise_xy=0.004,
                 noise_yaw_deg=0.05, seed=0):
        self.scale = scale
        self.yaw_bias = np.radians(yaw_bias_deg)
        self.noise_xy = noise_xy
        self.noise_yaw = np.radians(noise_yaw_deg)
        self.rng = np.random.default_rng(seed)

    def delta(self, pose_prev, pose_curr, dt_s):
        """Noisy odometry increment between two ground-truth poses (x, y, theta)."""
        rel = relative_pose(np.asarray(pose_prev, float), np.asarray(pose_curr, float))
        dx = rel[0] * self.scale + self.rng.normal(0, self.noise_xy)
        dy = rel[1] * self.scale + self.rng.normal(0, self.noise_xy)
        dth = rel[2] + self.yaw_bias * dt_s + self.rng.normal(0, self.noise_yaw)
        return np.array([dx, dy, dth])

    @staticmethod
    def compose(pose, delta):
        T = se2(*pose) @ se2(*delta)
        p = to_pose(T)
        p[2] = wrap_angle(p[2])
        return p


def integrate(truth_poses, odom, dt):
    """Integrate noisy increments starting from the first pose."""
    out = [np.array(truth_poses[0], float)]
    for i in range(1, len(truth_poses)):
        d = odom.delta(truth_poses[i - 1], truth_poses[i], dt)
        out.append(Odometer.compose(out[-1], d))
    return out
