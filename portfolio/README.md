# 智能驾驶算法作品集（Portfolio）

> 为补齐简历中「现代 DL 框架 / 检测 / 分割 / BEV」能力缺口、并把量产融合定位经验落到可运行代码，
> 手搓了两个 **零外部感知/SLAM 框架依赖** 的项目，可直接托管到 GitHub 作为算法能力佐证。

## 目录

| 项目 | 内容 | 关键技术 | 依赖 |
|------|------|----------|------|
| [`slam_fusion/`](slam_fusion) | 多传感器融合定位（SLAM / 里程计） | 手写 ICP（SVD）+ 松耦合 EKF + BEV 建图 | 纯 NumPy |
| [`pytorch_3d_bev/`](pytorch_3d_bev) | BEV 多任务感知（检测 + 语义分割） | BEV 投影 + Center 检测头 + 分割头 + Focal/Smooth-L1/加权CE | PyTorch |
| `index.html` | 本展示页（此 README 的可视化版） | — | 浏览器直接打开 |

## 与简历的呼应

- 简历强项「LiDAR+视觉+差分 GNSS 多源融合定位（量产 >10 万台）」→ `slam_fusion` 即其最小可运行原型；
- 简历强项「国家科技进步二等奖（核心成员）」→ 融合定位正是获奖方向的技术底座；
- 简历缺口「PyTorch / 检测 / 分割 / BEV」→ `pytorch_3d_bev` 全部手写补齐，证明能自己搭检测头而非只会调包。

## 实测指标（固定种子，可复现）

**① 融合定位**（`python slam_fusion/run.py`）

```
ATE(m)  EKF=1.06  IMU=2.94  ICP=1.27  GNSS_std=0.52
```

**② BEV 3D 检测**（`python pytorch_3d_bev/train.py && evaluate.py && demo.py`）

```
samples=200 验证集:  recall=0.965  precision=0.992  平均定位 L1=0.090 m  (IoU≥0.5)
BEV 分割 mIoU=0.974   per-class bg/road/veh=[0.997 / 0.999 / 0.927]
```

## 如何运行

```bash
# ① 融合定位（无需 torch）
pip install numpy matplotlib
python slam_fusion/run.py          # 产出 assets/trajectory.png, assets/bev_map.png

# ② BEV 3D 检测
pip install torch numpy matplotlib tqdm --index-url https://download.pytorch.org/whl/cpu
cd pytorch_3d_bev
# 限制线程数可避免小内存机器被 OOM 强杀（实测 20 epoch / 1500 样本约 1 小时 CPU）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python train.py --epochs 20 --samples 1500 --save checkpoints/seg.pt
python evaluate.py --ckpt checkpoints/seg.pt --n 200
python demo.py --ckpt checkpoints/seg.pt --out demo_bev.png
```

## 面试可讲亮点

1. **数据 / 算法解耦**：两个项目都用合成数据跑通，但把「数据采样」与「核心算法」彻底解耦——
   换成真实 KITTI/nuScenes `.bin/.pcd`、真实 IMU/RTK 驱动后，算法模块一行不用改。
2. **底层实现**：手写 ICP、EKF、BEV 投影、检测头与损失函数，比"调 mmdet3d / Open3D"更能体现底层理解。
3. **结果可信**：所有指标来自真实运行输出，固定随机种子可复现，不是纸面宣传。
