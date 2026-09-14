"""
world.py — 合成世界与真值轨迹生成
----------------------------------
为在无 ROS/KITTI 环境下也能跑通"多传感器融合定位"，这里用解析方法生成：
  - 一条车辆真值 SE(3) 轨迹（含转弯）
  - 一个简单世界（地面 + 若干长方体路标）
  - 沿轨迹采样 LiDAR 点云 / IMU 量测 / GNSS 位置
真实接入：把 LiDAR 换成你的 .pcd 读取、IMU 换传感器驱动、GNSS 换 RTK 即可，
下游 ICP + EKF 融合模块完全不用改。
"""
import numpy as np


def se3_from_xyyaw(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [x, y, 0]
    return T


def generate_truth(steps=120, dt=0.1):
    """生成一条带恒定转弯率的真值轨迹。返回 list[np.eye(4)]。"""
    poses = []
    x = y = yaw = 0.0
    v = 6.0          # m/s
    omega = 0.03     # rad/s 转弯率
    for _ in range(steps):
        yaw += omega * dt
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        poses.append(se3_from_xyyaw(x, y, yaw))
    return poses


def make_world(n_landmarks=50, seed=0):
    """地面 + 沿轨迹分布的若干长方体路标（中心点 + 尺寸）。

    路标沿 x∈[-10,78]、y∈[-25,25] 的走廊铺开，保证车辆在整个前向行驶
    过程中，车体前方始终有结构化特征可供 ICP 匹配（否则驶过之后会无特征）。
    """
    rng = np.random.default_rng(seed)
    landmarks = []
    for _ in range(n_landmarks):
        cx = float(rng.uniform(-10, 78))
        cy = float(rng.uniform(-25, 25))
        l = float(rng.uniform(1.5, 4.0)); w = float(rng.uniform(1.5, 4.0)); h = float(rng.uniform(1.0, 3.0))
        landmarks.append((cx, cy, l, w, h))
    return landmarks


def sample_lidar(pose, landmarks, n_per_box=120, noise=0.03, seed=None, features_only=False):
    """在 pose 处采集一帧 LiDAR 点云（车体坐标系，已含前向视锥裁剪）。

    注意：点云在**车体坐标系内**围绕本车生成（地面为前方无限平面、
    路标按本车位姿投影进车体系），而不是在固定世界坐标框里采样——
    否则车辆驶离原点后前向视野会为空。

    features_only=True 时只返回路标（结构化特征）点——用于 ICP 里程计，
    因为均匀无限地面在车体系里与位姿无关（平移不变），无法提供自运动信息。
    """
    rng = np.random.default_rng(seed)
    R = pose[:3, :3]; t = pose[:3, 3]
    pts_body = []
    if not features_only:
        # 地面（车体前方）——仅用于 BEV 可视化，不参与 ICP
        gx = rng.uniform(0.5, 30, 600); gy = rng.uniform(-30, 30, 600)
        pts_body.append(np.stack([gx, gy, np.full_like(gx, 0.0)], 1))
    # 路标：在世界系取表面点，再投影到车体坐标系，只保留前向可见部分
    for (cx, cy, l, w, h) in landmarks:
        for _ in range(n_per_box):
            sx = rng.uniform(-l/2, l/2); sy = rng.uniform(-w/2, w/2); sz = rng.uniform(0, h)
            side = rng.integers(0, 4)
            if side == 0: sx = -l/2
            elif side == 1: sx = l/2
            elif side == 2: sy = -w/2
            else: sy = w/2
            p_world = np.array([cx + sx, cy + sy, sz])
            pb = R.T @ (p_world - t)            # 世界 -> 车体
            if pb[0] > 0.5:
                pts_body.append(pb.reshape(1, 3))
    pts_body = np.vstack(pts_body)
    pts_body += rng.normal(0, noise, pts_body.shape)
    fwd = pts_body[:, 0] > 0.5
    pts_body = pts_body[fwd]
    # 关键：ICP 是 O(n^2) 最近邻，特征点过多会爆内存。演示里把特征点下采样到 ~600。
    if features_only and pts_body.shape[0] > 600:
        idx = rng.choice(pts_body.shape[0], 600, replace=False)
        pts_body = pts_body[idx]
    return pts_body


def sample_imu(pose_prev, pose_cur, dt, accel_noise=0.05, gyro_noise=0.005,
               yaw_bias=0.020, seed=None):
    """由相邻真值位姿反推"理想"IMU 量测并加噪（演示用）。

    yaw_bias 模拟陀螺常值零偏：会让纯 IMU 死推算随里程累积航向误差而**漂移**，
    这正是真实 IMU 的痛点，也是需要 GNSS/ICP 来修正的动机。
    """
    rng = np.random.default_rng(seed)
    yaw_prev = np.arctan2(pose_prev[1, 0], pose_prev[0, 0])
    yaw_cur = np.arctan2(pose_cur[1, 0], pose_cur[0, 0])
    a = rng.normal(0.0, accel_noise, 2)            # ax, ay
    yaw_rate = (yaw_cur - yaw_prev) / dt + yaw_bias + rng.normal(0, gyro_noise)
    return np.array([a[0], a[1], yaw_rate])


def sample_gnss(pose, pos_noise=0.8, seed=None):
    """带噪声的全球位置（模拟 RTK/差分 GPS，断锁时可由融合兜底）。"""
    rng = np.random.default_rng(seed)
    return pose[:3, 3][:2] + rng.normal(0, pos_noise, 2)
