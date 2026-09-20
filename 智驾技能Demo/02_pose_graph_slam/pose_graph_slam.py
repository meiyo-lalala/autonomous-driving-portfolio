# -*- coding: utf-8 -*-
"""
pose_graph_slam.py —— 2D 位姿图优化（纯 Python 参考实现）

用 Gauss-Newton + Levenberg 阻尼求解位姿图，等价于 GTSAM/Ceres 内部所做的最小二乘：
    min_pose  Σ || between(pi, pj) ⊖ zij ||^2_Ω

因子（Factor）：
  - PriorFactor      : 锚定第 0 帧，消除自由度为 0 的秩亏
  - BetweenFactor    : 相邻里程计约束（Odometry）
  - LoopClosureFactor: 回环检测约束（Loop Closure）

演示目标：回环约束把累积漂移"拉回"真值附近 —— 这正是图优化（GTSAM/Ceres/g2o）在
SLAM 后端中不可替代的价值。本文件为纯 Python 实现（只依赖 numpy），可在任何环境直跑；
与 pose_graph_slam_gtsam.py / cpp/ 下的官方库实现数学完全一致。

运行:  python pose_graph_slam.py
输出:  output/pose_graph_result.png
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

from synthetic_data import main as gen_data, wrap_angle

# ----------------  SE(2) 基础运算 ----------------

def between(a, b):
    """a^{-1} ∘ b：返回从 a 坐标系到 b 坐标系的相对位姿 (dx, dy, dtheta)。"""
    dx = np.cos(a[2]) * (b[0] - a[0]) + np.sin(a[2]) * (b[1] - a[1])
    dy = -np.sin(a[2]) * (b[0] - a[0]) + np.cos(a[2]) * (b[1] - a[1])
    return np.array([dx, dy, wrap_angle(b[2] - a[2])])


def compose(p, delta):
    """p ⊕ delta：在 p 的局部坐标系下叠加增量（SE(2) retraction）。"""
    return np.array([
        p[0] + delta[0] * np.cos(p[2]) - delta[1] * np.sin(p[2]),
        p[1] + delta[0] * np.sin(p[2]) + delta[1] * np.cos(p[2]),
        wrap_angle(p[2] + delta[2]),
    ])


# ----------------  因子及其雅可比 ----------------

def rot3(t):
    """SE(2) 局部扰动的位姿雅可比: d(p⊕δ)/dδ |_{δ=0}。"""
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class BetweenFactor:
    """残差 r = between(pi, pj) - zij，信息矩阵 Ω。
    雅可比采用与 SE(2) retraction (p ⊕ δ) 一致的"局部扰动"约定，
    与 GTSAM / g2o 的 Pose2 BetweenFactor 推导完全一致。"""

    def __init__(self, i, j, z, omega):
        self.i, self.j, self.z, self.omega = i, j, z, omega

    def residual_and_jacobians(self, poses):
        i, j = self.i, self.j
        ti = poses[i, 2]
        r = between(poses[i], poses[j]) - self.z
        r[2] = wrap_angle(r[2])
        dx, dy = poses[j, 0] - poses[i, 0], poses[j, 1] - poses[i, 1]
        Ji_g = np.array([
            [-np.cos(ti), -np.sin(ti), -np.sin(ti) * dx + np.cos(ti) * dy],
            [np.sin(ti), -np.cos(ti), -np.cos(ti) * dx - np.sin(ti) * dy],
            [0.0, 0.0, -1.0],
        ])
        Jj_g = np.array([
            [np.cos(ti), np.sin(ti), 0.0],
            [-np.sin(ti), np.cos(ti), 0.0],
            [0.0, 0.0, 1.0],
        ])
        # 转成局部扰动雅可比
        return r, Ji_g @ rot3(ti), Jj_g @ rot3(poses[j, 2])


class PriorFactor:
    """将 pose 锚定到给定参考（消除轨迹整体平移/旋转自由度）。"""

    def __init__(self, i, z, omega):
        self.i, self.z, self.omega = i, z, omega
        self.j = -1  # Prior 因子只有一个端点

    def residual_and_jacobians(self, poses):
        r = poses[self.i] - self.z
        r[2] = wrap_angle(r[2])
        return r, np.eye(3) @ rot3(poses[self.i, 2]), np.zeros((3, 3))


def build_and_solve(poses_init, factors, max_iter=60, lamb=1e-3, verbose=True):
    """Levenberg-Marquardt 求解位姿图（带代价下降接受准则）。"""
    poses = poses_init.copy()

    def compute_cost(p):
        cost = 0.0
        for f in factors:
            r, _, _ = f.residual_and_jacobians(p)
            cost += float(r @ f.omega @ r)
        return cost

    cost = compute_cost(poses)
    for it in range(max_iter):
        H = np.zeros((len(poses) * 3, len(poses) * 3))
        b = np.zeros(len(poses) * 3)
        for f in factors:
            r, Ji, Jj = f.residual_and_jacobians(poses)
            r = r.reshape(3, 1)
            for J, k in ((Ji, f.i), (Jj, f.j)):
                if k < 0:
                    continue
                i3 = k * 3
                H[i3:i3 + 3, i3:i3 + 3] += J.T @ f.omega @ J
                b[i3:i3 + 3] += J.T @ f.omega @ r.ravel()
        # LM 内循环：增大阻尼直到代价下降
        accepted = False
        for _ in range(20):
            Hd = H + lamb * np.diag(np.diag(H) + 1e-9)
            try:
                dx = np.linalg.solve(Hd, -b)
            except np.linalg.LinAlgError:
                lamb *= 10
                continue
            step = dx.reshape(-1, 3)
            cand = np.array([compose(poses[k], step[k]) for k in range(len(poses))])
            cand_cost = compute_cost(cand)
            if cand_cost < cost:
                poses, cost, lamb = cand, cand_cost, max(lamb / 3.0, 1e-6)
                accepted = True
                break
            lamb *= 10
        if verbose:
            print(f"  iter {it:2d}  cost={cost:12.4f}  |dx|={np.linalg.norm(dx):.6f}")
        if not accepted:
            break
        if np.linalg.norm(dx) < 1e-7:
            break
    return poses


def rmse(a, b):
    """XY 平面 RMSE (m)——定位精度标准口径（不含角度，角度单独报告）。"""
    return float(np.sqrt(np.mean(np.sum((a[:, :2] - b[:, :2]) ** 2, axis=1))))


def rmse_yaw_deg(a, b):
    """航向 RMSE（度）。"""
    d = np.abs(a[:, 2] - b[:, 2])
    d = np.minimum(d, 2 * np.pi - d)
    return float(np.degrees(np.sqrt(np.mean(d ** 2))))


def main():
    # 1) 生成仿真数据
    if not os.path.exists("data_loop.npz"):
        gen_data()
    data = np.load("data_loop.npz")
    gt = data["gt"]
    odom = data["odom"]
    loop_pairs, loop_noise = data["loop_pairs"], data["loop_noise"]

    n = len(gt)
    sig_t, sig_r = 0.15, 0.0025
    sig_loop = 0.08
    sig_gnss_xy, sig_gnss_yaw = 0.2, 0.005
    # 里程计信息矩阵（协方差倒数）
    omega_odom = np.diag([1 / sig_t**2, 1 / sig_t**2, 1 / sig_r**2])
    omega_loop = np.diag([1 / sig_loop**2] * 3)
    omega_prior = np.diag([1 / 0.5**2] * 3)
    omega_gnss = np.diag([1 / sig_gnss_xy**2, 1 / sig_gnss_xy**2, 1 / sig_gnss_yaw**2])

    # 2) 因子库：Prior + 里程计 + 回环 + 稀疏 GNSS（后两类可开关，用于对比）
    factors_base = [PriorFactor(0, gt[0], omega_prior)]
    for i in range(1, n):
        z = between(odom[i - 1], odom[i])
        factors_base.append(BetweenFactor(i - 1, i, z, omega_odom))

    factors_closure = factors_base + [
        BetweenFactor(i, j, z, omega_loop) for (i, j), z in zip(loop_pairs, loop_noise)]
    gnss_idx, gnss_meas = data["gnss_idx"], data["gnss_meas"]
    factors_full = factors_closure + [
        PriorFactor(int(i), m, omega_gnss) for i, m in zip(gnss_idx, gnss_meas)]

    # 3) 三种配置对比优化
    print("配置A: 纯里程计（基准，不做优化）")
    print("配置B: 里程计 + 回环约束（图优化） ...")
    opt_closure = build_and_solve(odom, factors_closure, max_iter=80, verbose=False)
    print("配置C: 里程计 + 回环 + 稀疏GNSS（多传感器融合） ...")
    opt_full = build_and_solve(odom, factors_full, max_iter=80, verbose=False)

    # 4) 评估
    def end_err(p):
        return float(np.linalg.norm(p[-1, :2] - gt[-1, :2]))

    rows = [
        ("A. 纯里程计(无优化)", odom),
        ("B. +回环约束", opt_closure),
        ("C. +回环+稀疏GNSS", opt_full),
    ]
    print(f"\n=== 对比结果 ===")
    print(f"{'配置':<22}{'XY-RMSE(m)':>13}{'终点误差(m)':>13}{'航向RMSE(°)':>14}")
    for name, p in rows:
        print(f"{name:<22}{rmse(p, gt):>13.3f}{end_err(p):>13.3f}{rmse_yaw_deg(p, gt):>14.3f}")

    # 5) 画图：轨迹对比 + 误差柱状图
    fig = plt.figure(figsize=(13.5, 6.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(gt[:, 0], gt[:, 1], "-", color="green", lw=2, label="Ground Truth")
    ax1.plot(odom[:, 0], odom[:, 1], "--", color="red", lw=1.3, alpha=0.9,
             label="A. Odometry (漂移)")
    ax1.plot(opt_closure[:, 0], opt_closure[:, 1], "-", color="blue", lw=1.5,
             label="B. +Loop Closure")
    ax1.plot(opt_full[:, 0], opt_full[:, 1], "-", color="magenta", lw=1.8,
             label="C. +Loop+GNSS (融合)")
    for (i, j) in loop_pairs[:60]:
        ax1.plot([opt_full[i, 0], opt_full[j, 0]], [opt_full[i, 1], opt_full[j, 1]],
                 "-", color="gray", alpha=0.3, lw=0.6)
    ax1.scatter(gt[gnss_idx, 0], gt[gnss_idx, 1], c="purple", s=55, zorder=5,
                marker="*", label="GNSS 绝对观测")
    ax1.scatter(gt[0, 0], gt[0, 1], c="green", s=70, zorder=6, marker="o")
    ax1.set_aspect("equal")
    ax1.set_title("位姿图优化: 里程计漂移 vs 回环+GNSS 校正")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    names = [r[0].split(". ")[1] for r in rows]
    x = np.arange(3)
    rmse_vals = [rmse(p, gt) for _, p in rows]
    end_vals = [end_err(p) for _, p in rows]
    w = 0.34
    b1 = ax2.bar(x - w / 2, rmse_vals, w, label="全程 RMSE (m)", color="#1a73e8")
    b2 = ax2.bar(x + w / 2, end_vals, w, label="终点误差 (m)", color="#ff9800")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.5)
    ax2.set_ylabel("误差 (m)")
    ax2.set_title("回环 + 稀疏GNSS 对定位误差的削减")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06,
                     f"{b.get_height():.2f}", ha="center", fontsize=9)

    os.makedirs("output", exist_ok=True)
    fig.savefig("output/pose_graph_result.png", dpi=140, bbox_inches="tight")
    print("\n图片已保存: output/pose_graph_result.png")


if __name__ == "__main__":
    main()
