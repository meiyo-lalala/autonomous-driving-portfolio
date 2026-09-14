"""
synthetic.py — 合成 3D 单帧检测数据集生成器
------------------------------------------------
为了在无 KITTI/nuScenes 大包下载的沙箱里也能跑通，这里用几何方法
合成"地面 + 若干车辆点云"。每帧返回一个 (N,4) 点云 (x,y,z,intensity)
和一组 3D 真值框 (cx,cy,cz,l,w,h,yaw)。

真实数据接入：把 `collate_fn` 里读取 .bin/.pcd 的逻辑替换为本模块
`generate_scene` 即可，下游 BEV 投影与网络完全不用改。
"""
import numpy as np

# ---- BEV 网格参数（与 models/bev_detector.py 保持一致）----
X_MIN, X_MAX = 0.0, 50.0      # 前向距离 (m)
Y_MIN, Y_MAX = -25.0, 25.0    # 横向距离 (m)
Z_MIN, Z_MAX = -2.0, 3.0      # 高度范围 (m)
RES = 0.5                     # 网格分辨率 (m)
W = int((X_MAX - X_MIN) / RES)   # 100
H = int((Y_MAX - Y_MIN) / RES)   # 100


def _sample_box_surface(cx, cy, cz, l, w, h, yaw, n_pts=320, seed=None):
    """在一辆车（旋转的长方体）表面采样点云。"""
    rng = np.random.default_rng(seed)
    # 局部坐标：x 沿车长，y 沿车宽，z 沿车高
    pts = []
    # 四个侧面 + 顶面
    faces = [('side', l, w, h, n_pts // 2), ('top', l, w, 0.0, n_pts // 2)]
    for kind, L, Wd, zoff, cnt in faces:
        if kind == 'side':
            # 在 x∈[-L/2,L/2], z∈[0,h] 的面上取点，再扩到四个面
            for _ in range(cnt):
                sx = rng.uniform(-L/2, L/2)
                sz = rng.uniform(0, h)
                sy = rng.choice([-Wd/2, Wd/2])
                pts.append([sx, sy, sz])
            # 另一对面 y=±Wd/2 已含；再补 x=±L/2 面
            for _ in range(cnt):
                sy = rng.uniform(-Wd/2, Wd/2)
                sz = rng.uniform(0, h)
                sx = rng.choice([-L/2, L/2])
                pts.append([sx, sy, sz])
        else:
            for _ in range(cnt):
                sx = rng.uniform(-L/2, L/2)
                sy = rng.uniform(-Wd/2, Wd/2)
                pts.append([sx, sy, h])
    p = np.array(pts)
    # 绕 z 轴旋转 yaw，再平移到 (cx,cy,cz)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p = p @ R.T + np.array([cx, cy, cz])
    return p


def generate_scene(rng=None, n_cars=None, n_ground=4000):
    """生成一帧合成场景。返回 (points(N,4), boxes(list of dict))。"""
    rng = rng or np.random.default_rng()
    n_cars = n_cars or int(rng.integers(3, 9))
    boxes = []
    car_pts = []
    for i in range(n_cars):
        cx = float(rng.uniform(6, 46))
        cy = float(rng.uniform(-20, 20))
        cz = 0.75
        l = float(rng.uniform(3.6, 4.6))      # 车长（沿 x）
        w = float(rng.uniform(1.6, 2.0))      # 车宽（沿 y）
        h = 1.5
        yaw = float(rng.uniform(-0.35, 0.35))
        car_pts.append(_sample_box_surface(cx, cy, cz, l, w, h, yaw,
                                            seed=int(rng.integers(0, 1e6))))
        boxes.append(dict(cx=cx, cy=cy, cz=cz, l=l, w=w, h=h, yaw=yaw))

    cars = np.vstack(car_pts) if car_pts else np.zeros((0, 3))
    # 地面点
    gx = rng.uniform(X_MIN + 1, X_MAX - 1, n_ground)
    gy = rng.uniform(Y_MIN + 1, Y_MAX - 1, n_ground)
    gz = rng.uniform(0.0, 0.08, n_ground)
    ground = np.stack([gx, gy, gz], axis=1)

    pts = np.vstack([cars, ground])
    # 加测量噪声
    pts += rng.normal(0, 0.02, pts.shape)
    # 强度：车体偏高，地面偏低
    intensity = np.where(pts[:, 2] > 0.2,
                        rng.uniform(0.6, 1.0, pts.shape[0]),
                        rng.uniform(0.1, 0.4, pts.shape[0]))
    points = np.column_stack([pts, intensity]).astype(np.float32)
    return points, boxes


def scene_to_bev_targets(points, boxes):
    """
    点云 -> BEV 伪图像 (C,H,W)；boxes -> 训练目标。
    返回:
      bev:   (3,H,W) float32  [max_z, density, mean_intensity]
      cls_t: (H,W) 0/1 目标中心栅格
      reg_t: (6,H,W) [dx,dy,dz,log(l),log(w),yaw_cos*2+...] 仅正样本有意义
    """
    bev = np.zeros((3, H, W), dtype=np.float32)
    cls_t = np.zeros((H, W), dtype=np.float32)
    reg_t = np.zeros((6, H, W), dtype=np.float32)

    xy = points[:, :2]
    z = points[:, 2]
    inten = points[:, 3]
    ix = ((xy[:, 0] - X_MIN) / RES).astype(int)
    iy = ((xy[:, 1] - Y_MIN) / RES).astype(int)
    mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy, z, inten = ix[mask], iy[mask], z[mask], inten[mask]

    # 向量化聚合：每格 max_z / 计数 / mean_intensity（避免逐单元 Python 循环）
    cell = iy * W + ix                       # (M,)
    maxz = np.zeros(W * H, dtype=np.float32)
    np.maximum.at(maxz, cell, z)             # 每格最大高度
    sumi = np.zeros(W * H, dtype=np.float32)
    np.add.at(sumi, cell, inten)             # 每格强度之和
    cnt = np.zeros(W * H, dtype=np.float32)
    np.add.at(cnt, cell, 1.0)               # 每格点数
    meani = np.zeros(W * H, dtype=np.float32)
    np.divide(sumi, cnt, out=meani, where=cnt > 0)
    dens = cnt / 50.0                        # 归一化

    bev[0] = (maxz.reshape(H, W) - Z_MIN) / (Z_MAX - Z_MIN)
    bev[1] = np.clip(dens.reshape(H, W), 0, 1)
    bev[2] = np.clip(meani.reshape(H, W), 0, 1)

    # 检测目标：CenterPoint 风格的高斯热图（峰值在车中心，向周围衰减），
    # 避免"单格=1"的极稀疏目标导致检测头学不动；
    # 关键：回归量（每格到车中心的 offset + 长宽 + yaw）写进**整个高斯 blob**，
    # 而非仅中心格——否则解码在任意峰格读到的是未训练的 0，框尺寸塌成 1x1m。
    sigma = 1.2   # 高斯半径 (m)
    R = 4         # 热图扩散半径 (格)
    for b in boxes:
        cxm, cym = b['cx'], b['cy']
        gx = int((cxm - X_MIN) / RES)
        gy = int((cym - Y_MIN) / RES)
        if 0 <= gx < W and 0 <= gy < H:
            for dy in range(-R, R + 1):
                for dx in range(-R, R + 1):
                    yy, xx = gy + dy, gx + dx
                    if 0 <= xx < W and 0 <= yy < H:
                        d2 = ((xx - gx) * RES) ** 2 + ((yy - gy) * RES) ** 2
                        v = np.exp(-d2 / (2 * sigma * sigma))
                        if v > cls_t[yy, xx]:
                            cls_t[yy, xx] = v
                        # 该格到车中心的回归目标（整 blob 都填，offset 随格变化）
                        reg_t[0, yy, xx] = cxm - (X_MIN + (xx + 0.5) * RES)
                        reg_t[1, yy, xx] = cym - (Y_MIN + (yy + 0.5) * RES)
                        reg_t[2, yy, xx] = b['cz']
                        reg_t[3, yy, xx] = np.log(b['l'])
                        reg_t[4, yy, xx] = np.log(b['w'])
                        reg_t[5, yy, xx] = b['yaw']

    # BEV 语义分割目标：每格 0=背景 1=路面 2=车辆（对应简历"分割"缺口）
    seg = np.zeros((H, W), dtype=np.int64)
    veh_flat = np.zeros(W * H, dtype=bool)
    road_flat = np.zeros(W * H, dtype=bool)
    np.maximum.at(veh_flat, cell, (z > 0.2).astype(np.float32))
    np.maximum.at(road_flat, cell, (z <= 0.2).astype(np.float32))
    seg_flat = np.zeros(W * H, dtype=np.int64)
    seg_flat[road_flat] = 1
    seg_flat[veh_flat] = 2            # 车辆格优先于路面格
    seg = seg_flat.reshape(H, W)
    return bev, cls_t, reg_t, seg


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    pts, boxes = generate_scene(rng)
    bev, cls, reg, seg = scene_to_bev_targets(pts, boxes)
    print('points:', pts.shape, 'boxes:', len(boxes))
    print('bev:', bev.shape, 'pos cells:', int(cls.sum()), 'seg classes:', np.unique(seg, return_counts=True))
