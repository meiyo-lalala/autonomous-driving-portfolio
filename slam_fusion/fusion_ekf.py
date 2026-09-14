"""
fusion_ekf.py — 松耦合多传感器融合定位 EKF
------------------------------------------
状态: X = [x, y, yaw, vx, vy, wy]  （平面位姿 + 速度 +  yaw 率）
传感器量测（均为"绝对位姿"形式，松耦合）：
  - IMU 死推算得到的绝对位姿 (x, y, yaw)
  - LiDAR-ICP 链式里程计得到的绝对位姿 (x, y, yaw)
  - GNSS/差分GPS 绝对位置 (x, y)
EKF 按各自协方差把它们加权融合，得到平滑且漂移小的定位结果。
对应候选人"LiDAR+视觉+差分GNSS 多源融合定位"的强项。
"""
import numpy as np


class PoseEKF:
    def __init__(self, dt, x0=(0.0, 0.0, 0.0)):
        self.dt = dt
        self.X = np.array([x0[0], x0[1], x0[2], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(6) * 1.0
        # 过程噪声：位置小，速度/yaw率稍大
        self.Q = np.diag([0.02, 0.02, 0.001, 0.05, 0.05, 0.002])
        self.I = np.eye(6)

    def predict(self):
        dt = self.dt
        x, y, yaw, vx, vy, wy = self.X
        self.X = np.array([
            x + vx * dt, y + vy * dt, yaw + wy * dt, vx, vy, wy
        ])
        F = np.eye(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.X
        self.X = self.X + K @ y
        self.P = (self.I - K @ H) @ self.P

    def update_gnss(self, pos, r=0.6):
        """GNSS 绝对位置量测 (x, y)。"""
        H = np.zeros((2, 6)); H[0, 0] = 1; H[1, 1] = 1
        self._update(np.array(pos, float), H, np.eye(2) * r)

    def update_pose(self, pose, r_pos=0.4, r_yaw=0.03):
        """IMU 死推算 / ICP 里程计 提供的绝对位姿 (x, y, yaw)。"""
        H = np.zeros((3, 6))
        H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1
        R = np.diag([r_pos, r_pos, r_yaw])
        self._update(np.array(pose, float), H, R)

    def pose(self):
        return self.X[:3].copy()
