"""
bev_map.py — 用估计轨迹把多帧点云累积成 BEV 占据/强度地图
---------------------------------------------------------
这是"多传感器融合定位"的输出之一：定位准，地图才准。
"""
import numpy as np


def build_bev(poses_body, scans_body, xlim=(-40, 70), ylim=(-35, 35), res=0.5):
    """
    poses_body: list[(3,)] 每帧估计位姿 (x,y,yaw)
    scans_body: list[(N,3)] 对应车体坐标系点云
    """
    xr = np.arange(xlim[0], xlim[1], res)
    yr = np.arange(ylim[0], ylim[1], res)
    W, Hh = len(xr), len(yr)
    grid = np.zeros((Hh, W), dtype=np.float32)      # 命中计数
    inten = np.zeros((Hh, W), dtype=np.float32)
    for pose, pts in zip(poses_body, scans_body):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s], [s, c]])
        p = pts[:, :2] @ R.T + pose[:2]          # 车体 -> 全局
        ix = ((p[:, 0] - xlim[0]) / res).astype(int)
        iy = ((p[:, 1] - ylim[0]) / res).astype(int)
        m = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < Hh)
        ix, iy = ix[m], iy[m]
        grid[iy, ix] += 1
        inten[iy, ix] += 1.0
    inten = np.divide(inten, grid, out=np.zeros_like(inten), where=grid > 0)
    return grid, inten
