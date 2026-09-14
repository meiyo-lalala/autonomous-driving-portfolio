"""
icp.py — 手写点云配准（point-to-point ICP，SVD 求解，Horn 法）
-------------------------------------------------------------
不依赖 Open3D / PCL。给定两帧车体坐标系点云，估计相对位姿 T（当前帧 -> 上一帧）。
这是 SLAM 前端里程计的核心，也是候选人"点云特征/匹配"背景的直接体现。
"""
import numpy as np


def nearest_correspondences(src, dst):
    """src 中每点找 dst 最近点索引（暴力 O(n^2)，小点云足够）。"""
    diff = src[:, None, :] - dst[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    idx = np.argmin(dist2, axis=1)
    return idx


def icp(src, dst, max_iter=20, tol=1e-5):
    """
    src: (N,3) 当前帧；dst: (M,3) 上一帧。
    返回 T: (4,4) 使得 src @ R.T + t ≈ dst（即把 src 配到 dst）。
    """
    S = src.copy()
    T_total = np.eye(4)
    for _ in range(max_iter):
        idx = nearest_correspondences(S, dst)
        P = S                      # 源点（当前帧）
        Q = dst[idx]               # 对应目标点（上一帧，按最近邻索引）
        mu_p = P.mean(0); mu_q = Q.mean(0)
        Pc = P - mu_p; Qc = Q - mu_q
        H = Pc.T @ Qc
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = Vt.T @ U.T
        t = mu_q - R @ mu_p
        # 累计变换
        T_step = np.eye(4); T_step[:3, :3] = R; T_step[:3, 3] = t
        S = (R @ S.T + t[:, None]).T
        T_total = T_step @ T_total
        err = np.mean(np.linalg.norm(S - dst[idx], axis=1))
        if err < tol:
            break
    return T_total, err
