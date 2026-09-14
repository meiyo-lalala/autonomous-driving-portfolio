"""
run.py — 跑通"多传感器融合定位"全链路并出图，用于作品集。
生成：
  assets/trajectory.png  估计轨迹 vs 真值 + 各基线
  assets/bev_map.png     用 EKF 轨迹累积的 BEV 地图
  assets/error.png       各方法平移误差随里程变化
用法: python run.py
"""
import os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from world import generate_truth, make_world, sample_lidar, sample_imu, sample_gnss
from icp import icp
from fusion_ekf import PoseEKF
from bev_map import build_bev

DT = 0.1
STEPS = 120


def pose_from_T(T):
    # 注意：平移在齐次矩阵的最后一列 T[0,3]/T[1,3]，不是 T[0,2]/T[1,2]（那是旋转元）
    return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])])


def main():
    os.makedirs('assets', exist_ok=True)
    truth = generate_truth(STEPS, DT)
    world = make_world()
    rng = np.random.default_rng(42)

    # 各方法轨迹累积
    ekf_traj = []          # 融合定位（EKF）
    imu_traj = []          # 仅 IMU 死推算（会漂移）
    icp_traj = []          # 仅 LiDAR-ICP 链式里程计
    gnss_pts = []          # GNSS 原始
    gt_traj = [pose_from_T(T) for T in truth]

    # 初始化
    ekf = PoseEKF(DT, x0=(0, 0, 0))
    imu_pose = np.array([0., 0., 0.])
    icp_T = np.eye(4)           # 当前 ICP 链式位姿（齐次 4x4）
    prev_feat = None
    ekf_scans, ekf_poses = [], []

    for k in range(STEPS):
        cur = truth[k]
        lidar = sample_lidar(cur, world, seed=k)                 # 全量（用于 BEV 地图）
        lidar_feat = sample_lidar(cur, world, seed=k, features_only=True)  # 结构化特征（用于 ICP）
        imu = sample_imu(truth[max(0, k-1)], cur, DT, seed=k+1000)
        gnss = sample_gnss(cur, seed=k+2000)
        gnss_pts.append(gnss)

        # ---- IMU 死推算（基线）----
        imu_pose[2] += imu[2] * DT
        v = 6.0
        imu_pose[0] += v * np.cos(imu_pose[2]) * DT
        imu_pose[1] += v * np.sin(imu_pose[2]) * DT
        imu_traj.append(imu_pose.copy())

        # ---- ICP 链式里程计（基线）----
        if lidar_feat.size == 0:
            lidar_feat = lidar          # 兜底：避免某帧前方无特征时空扫描
        if prev_feat is not None:
            T, _ = icp(lidar_feat, prev_feat, max_iter=15)
            icp_T = icp_T @ T
        prev_feat = lidar_feat
        icp_traj.append(pose_from_T(icp_T).copy())

        # ---- EKF 融合：IMU 预测 + ICP/IMU 位姿更新 + GNSS 更新 ----
        ekf.predict()
        ekf.update_pose(imu_pose, r_pos=0.8, r_yaw=0.05)    # IMU 绝对位姿
        if len(ekf_traj) > 0:
            ekf.update_pose(pose_from_T(icp_T), r_pos=0.5, r_yaw=0.04)  # ICP 绝对位姿
        ekf.update_gnss(gnss, r=0.7)
        ekf_traj.append(ekf.pose().copy())
        ekf_scans.append(lidar); ekf_poses.append(ekf.pose().copy())

    # ---- 误差分析（ATE） ----
    def ate(traj):
        e = [np.hypot(t[0]-g[0], t[1]-g[1]) for t, g in zip(traj, gt_traj)]
        return np.array(e)
    err_ekf = ate(ekf_traj); err_imu = ate(imu_traj); err_icp = ate(icp_traj)
    print(f'ATE(m)  EKF={err_ekf.mean():.2f}  IMU={err_imu.mean():.2f}  '
          f'ICP={err_icp.mean():.2f}  GNSS_std={np.std(ate(gnss_pts)):.2f}')

    # ---- 出图 ----
    gt = np.array(gt_traj); ek = np.array(ekf_traj)
    im = np.array(imu_traj); ic = np.array(icp_traj)
    gnss_arr = np.array(gnss_pts)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    ax[0].plot(gt[:, 0], gt[:, 1], 'k-', lw=2, label='Ground Truth')
    ax[0].plot(ek[:, 0], ek[:, 1], 'b-', label=f'EKF Fusion (ATE {err_ekf.mean():.2f}m)')
    ax[0].plot(gnss_arr[:, 0], gnss_arr[:, 1], 'g.', ms=3, alpha=.5, label='GNSS')
    ax[0].set_title('Trajectory: EKF Fusion'); ax[0].legend(); ax[0].axis('equal')

    ax[1].plot(gt[:, 0], gt[:, 1], 'k-', lw=2, label='Ground Truth')
    ax[1].plot(im[:, 0], im[:, 1], 'r-', label=f'IMU-only (ATE {err_imu.mean():.2f}m)')
    ax[1].plot(ic[:, 0], ic[:, 1], 'm-', label=f'ICP-only (ATE {err_icp.mean():.2f}m)')
    ax[1].set_title('Baselines (drift)'); ax[1].legend(); ax[1].axis('equal')

    t = np.arange(STEPS) * DT
    ax[2].plot(t, err_ekf, 'b-', label='EKF Fusion')
    ax[2].plot(t, err_imu, 'r-', label='IMU-only')
    ax[2].plot(t, err_icp, 'm-', label='ICP-only')
    ax[2].plot(t, ate(gnss_pts), 'g.', ms=3, label='GNSS')
    ax[2].set_title('Translation Error vs distance'); ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('error (m)'); ax[2].legend()
    plt.tight_layout(); plt.savefig('assets/trajectory.png', dpi=120); plt.close()

    # BEV 地图
    grid, _ = build_bev(ekf_poses, ekf_scans, xlim=(-30, 70), ylim=(-30, 30), res=0.5)
    plt.figure(figsize=(8, 7))
    plt.imshow(np.log1p(grid), origin='lower', cmap='magma',
               extent=[-30, 70, -30, 30])
    plt.plot(ek[:, 0], ek[:, 1], 'c-', lw=1.5, label='EKF path')
    plt.title('BEV Occupancy Map (fused localization)'); plt.legend(); plt.axis('equal')
    plt.tight_layout(); plt.savefig('assets/bev_map.png', dpi=120); plt.close()
    print('saved assets/trajectory.png, assets/bev_map.png')


if __name__ == '__main__':
    main()
