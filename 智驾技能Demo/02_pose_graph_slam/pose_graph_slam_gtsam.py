# -*- coding: utf-8 -*-
"""
pose_graph_slam_gtsam.py —— 使用 GTSAM 官方库的位姿图 SLAM（与纯 Python 版数学一致）

依赖:  pip install gtsam   （注意: gtsam 的 Python wheel 目前不支持 Python 3.14+，
       若安装失败，请改用 pose_graph_slam.py 纯 Python 版，
       或使用 cpp/ 下的 C++ 实现，或 WSL/Ubuntu 环境。）

运行:  python pose_graph_slam_gtsam.py
输出:  output/pose_graph_result_gtsam.png

参数与 synthetic_data.py / pose_graph_slam.py 保持完全一致：
  里程计 sigma = (0.15 m, 0.15 m, 0.0025 rad)
  回环   sigma = 0.08 rad
  GNSS   sigma = (0.2 m, 0.2 m, 0.005 rad)  （RTK/组合导航量级）
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gtsam
from gtsam.symbol_shorthand import X

from synthetic_data import main as gen_data, between, wrap_angle


def main():
    if not os.path.exists("data_loop.npz"):
        gen_data()
    data = np.load("data_loop.npz")
    gt = data["gt"]
    odom = data["odom"]
    loop_pairs, loop_noise = data["loop_pairs"], data["loop_noise"]
    gnss_idx, gnss_meas = data["gnss_idx"], data["gnss_meas"]
    n = len(gt)

    # ---------- 构建因子图 ----------
    graph = gtsam.NonlinearFactorGraph()

    # Prior: 锚定起点
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 0.5, 0.5]))
    graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(*gt[0]), prior_noise))

    # 里程计因子
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.15, 0.15, 0.0025]))
    for i in range(1, n):
        z = between(odom[i - 1], odom[i])
        graph.add(gtsam.BetweenFactorPose2(
            X(i - 1), X(i), gtsam.Pose2(z[0], z[1], z[2]), odom_noise))

    # 回环因子
    loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.08, 0.08, 0.08]))
    for (i, j), z in zip(loop_pairs, loop_noise):
        graph.add(gtsam.BetweenFactorPose2(
            X(int(i)), X(int(j)), gtsam.Pose2(z[0], z[1], z[2]), loop_noise))

    # 稀疏 GNSS 绝对因子（多传感器融合：周期性校正里程计漂移）
    gnss_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.005]))
    for i, m in zip(gnss_idx, gnss_meas):
        graph.add(gtsam.PriorFactorPose2(
            X(int(i)), gtsam.Pose2(m[0], m[1], m[2]), gnss_noise))

    # ---------- 初值（里程计轨迹） ----------
    initial = gtsam.Values()
    for i in range(n):
        initial.insert(X(i), gtsam.Pose2(odom[i, 0], odom[i, 1], odom[i, 2]))

    # ---------- LM 优化 ----------
    print("GTSAM Levenberg-Marquardt 优化中 ...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    opt = np.array([[result.atPose2(X(i)).x(),
                     result.atPose2(X(i)).y(),
                     result.atPose2(X(i)).theta()] for i in range(n)])

    def rmse_xy(a, b):
        return float(np.sqrt(np.mean(np.sum((a[:, :2] - b[:, :2]) ** 2, axis=1))))

    def end_err(a):
        return float(np.linalg.norm(a[-1, :2] - b[-1, :2]))

    err_before = rmse_xy(odom, gt)
    err_after = rmse_xy(opt, gt)
    end_before = end_err(odom)
    end_after = end_err(opt)
    print(f"XY-RMSE: 里程计 {err_before:.3f} m -> GTSAM优化后 {err_after:.3f} m")
    print(f"终点误差: 里程计 {end_before:.3f} m -> GTSAM优化后 {end_after:.3f} m")

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(gt[:, 0], gt[:, 1], "-", color="green", lw=2, label="Ground Truth")
    ax.plot(odom[:, 0], odom[:, 1], "--", color="red", lw=1.4, alpha=0.85,
            label="Odometry (漂移)")
    ax.plot(opt[:, 0], opt[:, 1], "-", color="blue", lw=1.6,
            label="GTSAM Pose-Graph Opt")
    ax.scatter(gt[gnss_idx, 0], gt[gnss_idx, 1], c="purple", s=50, marker="*",
               zorder=5, label="GNSS 绝对观测")
    ax.set_aspect("equal")
    ax.set_title(f"GTSAM Pose Graph SLAM\n"
                 f"XY-RMSE: {err_before:.3f} m -> {err_after:.3f} m")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    os.makedirs("output", exist_ok=True)
    fig.savefig("output/pose_graph_result_gtsam.png", dpi=140, bbox_inches="tight")
    print("图片已保存: output/pose_graph_result_gtsam.png")


if __name__ == "__main__":
    main()
