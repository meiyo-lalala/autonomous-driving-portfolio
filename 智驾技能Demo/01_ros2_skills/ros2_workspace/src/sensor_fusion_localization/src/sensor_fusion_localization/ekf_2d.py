# 多传感器融合 EKF 核心（2D: x, y, theta）
# 与 local_sim/run_local_sim.py 中的 ekf_fusion 逻辑一致，供 ROS2 节点复用
import numpy as np

GRAVITY = 9.81


class Ekf2D:
    """轮速里程计预测 + 绝对观测(GNSS/RTK)更新的 2D EKF。"""

    def __init__(self, sigma_odom_xy=0.15, sigma_odom_yaw=0.0025,
                 sigma_gnss_xy=0.2, sigma_gnss_yaw=0.005):
        self.Q = np.diag([sigma_odom_xy ** 2, sigma_odom_xy ** 2, sigma_odom_yaw ** 2])
        self.R = np.diag([sigma_gnss_xy ** 2, sigma_gnss_xy ** 2, sigma_gnss_yaw ** 2])
        self.x = np.zeros(3)
        self.P = np.eye(3) * 1e-2
        self._initialized = False

    def init(self, x, y, theta):
        self.x = np.array([x, y, theta])
        self.P = np.eye(3) * 1e-4
        self._initialized = True

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def predict_odom(self, dx_world, dy_world, dyaw):
        """以里程计增量做预测（增量已换算到车体系）。"""
        th = self.x[2]
        self.x[0] += dx_world * np.cos(th) - dy_world * np.sin(th)
        self.x[1] += dx_world * np.sin(th) + dy_world * np.cos(th)
        self.x[2] = self._wrap(self.x[2] + dyaw)
        F = np.eye(3)
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss(self, zx, zy, zyaw):
        """GNSS/RTK 绝对位置+航向更新。"""
        z = np.array([zx, zy, zyaw])
        H = np.eye(3)
        y = z - self.x
        y[2] = self._wrap(y[2])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = self._wrap(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P

    @property
    def pose(self):
        return self.x.copy()

    @property
    def covariance(self):
        return self.P.copy()
