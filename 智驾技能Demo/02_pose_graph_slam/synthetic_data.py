# -*- coding: utf-8 -*-
"""
synthetic_data.py —— 生成 2D 位姿图 SLAM 仿真数据

生成一条带回环的车辆轨迹：
  - ground truth : 真值轨迹（闭合环路，含直线段 + 圆弧弯道）
  - odometry     : 里程计测量（带随机噪声 + 随时间增长的漂移，模拟陀螺仪零偏等）
  - loop_closures: 回环检测（回到相近位置时检测到的约束）

输出: 02_pose_graph_slam/data_loop.npz
"""
import numpy as np

RNG = np.random.default_rng(42)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_rounded_loop(seg=25.0, radius=5.0, n_straight=25, n_arc=12):
    """生成一个圆角矩形闭合轨迹，返回 Nx3 的 (x, y, theta)。"""
    poses = []
    theta = 0.0

    def add_straight(length, n):
        nonlocal theta
        for k in range(n):
            poses.append([poses[-1][0] + length / n * np.cos(theta),
                          poses[-1][1] + length / n * np.sin(theta), theta])

    def add_arc(radius, angle_total, n):
        nonlocal theta
        # 顺时针 90° 圆弧（向左转）
        dth = -angle_total / n
        for k in range(n):
            theta += dth / 2.0
            x = poses[-1][0] + radius * dth * np.cos(theta + dth / 2.0)
            y = poses[-1][1] + radius * dth * np.sin(theta + dth / 2.0)
            poses.append([x, y, theta])
            theta += dth / 2.0

    poses.append([0.0, 0.0, 0.0])          # 起点
    add_straight(seg, n_straight)          # → 右
    add_arc(radius, np.pi / 2, n_arc)      # 转弯（向上）
    add_straight(seg, n_straight)          # → 上
    add_arc(radius, np.pi / 2, n_arc)
    add_straight(seg, n_straight)          # → 左
    add_arc(radius, np.pi / 2, n_arc)
    add_straight(seg, n_straight)          # → 下（回到起点附近）
    add_arc(radius, np.pi / 2, n_arc)
    return np.array(poses)


def make_odometry(gt):
    """里程计：真值相对位姿 + 噪声 + 累积漂移。
    参数为典型车载轮速里程计量级：平移 0.08m/步、旋转 0.005rad/步，
    使纯里程计在长距离上产生约 1.5-2m 的随机游走误差（可被回环+GNSS校正）。"""
    n = len(gt)
    odom = np.zeros_like(gt)
    odom[0] = gt[0]
    sigma_t = 0.15        # 平移噪声 std (m)——轮速里程计量级
    sigma_r = 0.0025      # 旋转噪声 std (rad)——陀螺辅助轮速里程计量级（偏航较准）
    drift_rate = 0.0      # 偏航系统性漂移（真实系统中由GNSS/航向参考逐步校正，本演示聚焦随机游走型误差）
    for i in range(1, n):
        rel = np.zeros(3)
        # 真值相对运动（体坐标系）
        d = gt[i] - gt[i - 1]
        rel[0] = np.cos(gt[i - 1, 2]) * d[0] + np.sin(gt[i - 1, 2]) * d[1]
        rel[1] = -np.sin(gt[i - 1, 2]) * d[0] + np.cos(gt[i - 1, 2]) * d[1]
        rel[2] = wrap_angle(gt[i, 2] - gt[i - 1, 2]) + drift_rate
        # 加测量噪声
        rel[0] += RNG.normal(0, sigma_t)
        rel[1] += RNG.normal(0, sigma_t)
        rel[2] += RNG.normal(0, sigma_r)
        # 递推累积
        t = odom[i - 1, 2]
        odom[i, 0] = odom[i - 1, 0] + rel[0] * np.cos(t) - rel[1] * np.sin(t)
        odom[i, 1] = odom[i - 1, 1] + rel[0] * np.sin(t) + rel[1] * np.cos(t)
        odom[i, 2] = wrap_angle(t + rel[2])
    return odom


def make_loop_closures(gt, min_gap=20, max_dist=2.5, max_count=50, sigma=0.08):
    """在空间上相近、时间上不相邻的位姿对之间生成回环约束。
    sigma=0.08m 对应激光扫描匹配的回环精度量级。"""
    pairs, noise = [], []
    n = len(gt)
    for i in range(n):
        for j in range(i + min_gap, n):
            d = np.linalg.norm(gt[i, :2] - gt[j, :2])
            if d < max_dist:
                rel = np.zeros(3)
                dx = gt[j, 0] - gt[i, 0]
                dy = gt[j, 1] - gt[i, 1]
                rel[0] = np.cos(gt[i, 2]) * dx + np.sin(gt[i, 2]) * dy
                rel[1] = -np.sin(gt[i, 2]) * dx + np.cos(gt[i, 2]) * dy
                rel[2] = wrap_angle(gt[j, 2] - gt[i, 2])
                rel += RNG.normal(0, sigma, 3)
                pairs.append((int(i), int(j)))
                noise.append(rel)
                if len(pairs) >= max_count:
                    return np.array(pairs), np.array(noise)
    return np.array(pairs), np.array(noise)


def make_gnss(gt, stride=8, sigma_xy=0.2, sigma_yaw=0.005):
    """稀疏 GNSS 绝对观测：每隔 stride 帧给一个含噪声的全局位姿约束。
    参数取 RTK/组合导航量级（横向 0.2m、航向 0.005rad≈0.3°），
    模拟多传感器融合中高精度绝对参考对里程计漂移的周期性校正。"""
    idx = list(range(0, len(gt), stride))
    meas = gt[idx].copy()
    meas[:, 0] += RNG.normal(0, sigma_xy, len(idx))
    meas[:, 1] += RNG.normal(0, sigma_xy, len(idx))
    meas[:, 2] += RNG.normal(0, sigma_yaw, len(idx))
    return np.array(idx), meas


def main():
    gt = make_rounded_loop()
    odom = make_odometry(gt)
    pairs, closure_noise = make_loop_closures(gt)
    gnss_idx, gnss_meas = make_gnss(gt)
    np.savez_compressed(
        "data_loop.npz",
        gt=gt, odom=odom,
        loop_pairs=pairs, loop_noise=closure_noise,
        gnss_idx=gnss_idx, gnss_meas=gnss_meas,
    )
    # 同时导出 txt，供 cpp/pose_graph_slam_gtsam.cpp 读取
    with open("data_loop.txt", "w", encoding="utf-8") as f:
        f.write(f"{len(gt)}\n")
        for p in gt:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for p in odom:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        f.write(f"{len(pairs)}\n")
        for (i, j), z in zip(pairs, closure_noise):
            f.write(f"{i} {j} {z[0]:.6f} {z[1]:.6f} {z[2]:.6f}\n")
        f.write(f"{len(gnss_idx)}\n")
        for i, m in zip(gnss_idx, gnss_meas):
            f.write(f"{i} {m[0]:.6f} {m[1]:.6f} {m[2]:.6f}\n")
    print(f"轨迹位姿数: {len(gt)}")
    print(f"回环约束数: {len(pairs)}")
    print(f"GNSS 绝对约束数: {len(gnss_idx)}")
    print(f"未优化里程计终点误差: {np.linalg.norm(odom[-1, :2] - gt[-1, :2]):.3f} m")


if __name__ == "__main__":
    main()
