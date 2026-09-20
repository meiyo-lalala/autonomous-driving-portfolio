# -*- coding: utf-8 -*-
"""
01_ros2_skills/local_sim —— Windows 本机可运行的"多传感器融合"在线仿真
（无需安装 ROS2；逻辑与 ros2_workspace 中的 fusion_node.py 完全一致）

演示: 轮速里程计(高频、漂移) + 稀疏GNSS(低频、绝对) 的 EKF 融合。
输出: output/ekf_fusion_result.png（真值 / 纯里程计 / EKF融合 对比 + 误差柱状图）
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "02_pose_graph_slam"))
import synthetic_data as sd          # 复用 02 模块的轨迹/传感器仿真
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


# ---------- 1. 传感器流 ----------
def make_sensor_streams():
    """生成真值轨迹 + 里程计(高频) + GNSS(低频)。"""
    gt = sd.make_rounded_loop()                       # 真值：25x25m 闭合环路
    odom = sd.make_odometry(gt)                       # 里程计：含噪声
    gnss_idx, gnss_meas = sd.make_gnss(gt, stride=8)  # 稀疏 GNSS 绝对观测
    return gt, odom, gnss_idx, gnss_meas


# ---------- 2. EKF 融合（与 ROS2 fusion_node.py 相同逻辑） ----------
def ekf_fusion(gt, odom, gnss_idx, gnss_meas):
    """
    状态 x = [x, y, theta]
    预测: 里程计相对位移(带噪声模型 Q)
    更新: GNSS 绝对观测(带噪声模型 R)
    """
    n = len(gt)
    fused = np.zeros_like(odom)
    P = np.eye(3) * 1e-2
    # 噪声模型（与 02 模块一致的物理量级）
    Q = np.diag([0.15 ** 2, 0.15 ** 2, (0.0025) ** 2])   # 里程计
    R = np.diag([0.2 ** 2, 0.2 ** 2, (0.005) ** 2])      # GNSS/RTK
    gnss_map = {int(i): m for i, m in zip(gnss_idx, gnss_meas)}

    x = np.array([odom[0, 0], odom[0, 1], odom[0, 2]])
    fused[0] = x
    for i in range(1, n):
        # --- 预测：用里程计增量（世界系，直接叠加） ---
        dx = odom[i, :2] - odom[i - 1, :2]
        dyaw = sd.wrap_angle(odom[i, 2] - odom[i - 1, 2])
        x[0] += dx[0]
        x[1] += dx[1]
        x[2] = sd.wrap_angle(x[2] + dyaw)
        F = np.eye(3)
        P = F @ P @ F.T + Q
        # --- 更新：GNSS 观测（稀疏到达） ---
        if i in gnss_map:
            z = gnss_map[i]
            H = np.eye(3)
            y = z - x
            y[2] = sd.wrap_angle(y[2])
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            x[2] = sd.wrap_angle(x[2])
            P = (np.eye(3) - K @ H) @ P
        fused[i] = x
    return fused


# ---------- 3. 主流程 ----------
def main():
    gt, odom, gnss_idx, gnss_meas = make_sensor_streams()
    fused = ekf_fusion(gt, odom, gnss_idx, gnss_meas)

    def rmse(p):
        return float(np.sqrt(np.mean(np.sum((p[:, :2] - gt[:, :2]) ** 2, axis=1))))

    def end_err(p):
        return float(np.linalg.norm(p[-1, :2] - gt[-1, :2]))

    print("=== 在线 EKF 多传感器融合 ===")
    print("里程计   : XY-RMSE %.3f m  终点误差 %.3f m" % (rmse(odom), end_err(odom)))
    print("EKF融合  : XY-RMSE %.3f m  终点误差 %.3f m" % (rmse(fused), end_err(fused)))
    print("GNSS观测点数: %d" % len(gnss_idx))

    os.makedirs("output", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0))
    ax1.plot(gt[:, 0], gt[:, 1], "-", color="green", lw=2, label="Ground Truth")
    ax1.plot(odom[:, 0], odom[:, 1], "--", color="red", lw=1.3, alpha=0.9, label="Odometry (漂移)")
    ax1.plot(fused[:, 0], fused[:, 1], "-", color="blue", lw=1.8, label="EKF + 稀疏GNSS")
    ax1.scatter(gt[gnss_idx, 0], gt[gnss_idx, 1], c="purple", s=50, marker="*", zorder=5,
                label="GNSS 观测")
    ax1.set_aspect("equal")
    ax1.set_title("在线 EKF 融合: 里程计漂移被稀疏GNSS校正")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    x = np.arange(2)
    vals = [rmse(odom), rmse(fused)]
    ends = [end_err(odom), end_err(fused)]
    b1 = ax2.bar(x - 0.19, vals, 0.38, label="全程 XY-RMSE (m)", color="#1a73e8")
    b2 = ax2.bar(x + 0.19, ends, 0.38, label="终点误差 (m)", color="#ff9800")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["纯里程计", "EKF+GNSS"])
    ax2.set_ylabel("误差 (m)")
    ax2.set_title("EKF 融合对定位误差的削减")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for bars in (b1, b2):
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                     f"{b.get_height():.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig("output/ekf_fusion_result.png", dpi=140, bbox_inches="tight")
    print("图片已保存: output/ekf_fusion_result.png")


if __name__ == "__main__":
    main()
