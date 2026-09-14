# PyTorch 3D 检测 / BEV 感知 — 手搓版

> 对应简历**能力缺口**：现代 DL 框架（PyTorch）、目标检测、语义分割、BEV 感知。
> 本项目是**从零手写**的 BEV 单阶段 3D 检测器（CenterPoint / BirdNet 简化版），
> **不依赖 mmdet3d / OpenPCDet / Open3D**，证明"会用 PyTorch 自己搭检测头"，
> 而不是只会调现成框架。

## 它解决了什么问题

自动驾驶感知里，**BEV（鸟瞰图）感知**是当下主流范式（Tesla、蔚来、理想都在用）。
本项目用最小可运行的方式，完整跑通一个**多任务 BEV 感知头**：

> 点云 → BEV 伪图像 → CNN 主干 → ① 分类头(有无车) + 回归头(3D 框) ② **语义分割头(背景/路面/车辆)**

其中 ② 正是简历里「**语义分割**」缺口的对应补齐——BEV 分割（把 BEV 图像按像素分成背景/路面/车辆）是量产 BEV 感知栈的标准组成（占据网络、车道/可行驶区域分割都属此类）。

## 模块拆解（全部手写）

| 文件 | 内容 | 对应能力点 |
|------|------|-----------|
| `data/synthetic.py` | 合成 3D 场景（地面 + 车辆点云）+ 点云→BEV 投影 + 训练目标生成 | 数据工程、BEV 投影 |
| `models/bev_detector.py` | `BEVBackbone`（stem+5 ConvBlock+上采样融合）、`BEVDetector`、检测头、`seg_head` 语义分割头、**软目标 Focal Loss / Smooth-L1 / 加权 CE 分割损失 / 3×3 峰值抑制解码**、`decode_boxes` | 网络设计、损失函数 |
| `train.py` | `SynthDataset` + `DataLoader` + 训练循环（多任务损失：检测 + 分割） | 训练工程 |
| `evaluate.py` | BEV 检测指标：recall / precision / 定位 L1 误差、2D IoU 匹配 | 评估方法 |
| `demo.py` | 推理一帧并画 **BEV 检测对比图**（GT vs Pred） | 可视化、作品集素材 |

**BEV 投影**：点云按 (x,y) 栅格化，每格取 max-height / density / mean-intensity 三通道，
与 PointPillars / CenterPoint 的 BEV 编码一致。

**检测头**：anchor-free、center-based——每格预测 objectness 与
`(dx, dy, dz, log l, log w, yaw)`，直接回归 3D 框，省去 anchor 设计。
训练目标用 **CenterPoint 风格的高斯热图**（车中心峰值=1，向周围按高斯衰减），
而非单格 0/1，避免极稀疏正样本导致检测头学不动；推理时先做 **3×3 局部峰值抑制**
再按阈值出框，保证每车只出一个框。

**分割头（对应简历「分割」缺口）**：同一个 BEV 主干上再接一个 1×1 卷积，
对每格做 **语义分割**（0=背景 / 1=路面 / 2=车辆）。损失用按类加权的
交叉熵（背景格占多数，降权避免被淹没）。这样检测 + 分割共享主干，
正是量产 BEV 感知「一套 backbone 出多任务」的做法。

## 运行

```bash
cd projects/pytorch_3d_bev
pip install torch numpy matplotlib tqdm --index-url https://download.pytorch.org/whl/cpu
python train.py --epochs 20 --samples 1500 --save checkpoints/seg.pt
python evaluate.py --ckpt checkpoints/seg.pt --n 200
python demo.py --ckpt checkpoints/seg.pt --out demo_bev.png
```

> 说明：CPU 训练 1500 帧约几分钟。若想更快，去掉 `--index-url`（用 CUDA 版 torch）即可。

## 实测效果（固定种子，可复现）

- samples=200 验证集：**recall=0.965  precision=0.992  平均定位 L1=0.090 m**（2D IoU≥0.5 匹配）；
- **BEV 语义分割 mIoU=0.974**，per-class bg/road/veh = **0.997 / 0.999 / 0.927**，分割近乎完美；
- 多任务损失稳定下降（检测 Focal+Smooth-L1 与分割加权 CE 联合收敛），检测头从"塌缩全背景"修到高召回；
- `demo_bev.png` 三连图：BEV 强度图 / 3D 检测框(红预测套绿真值) / **BEV 分割(GT实色+预测叠加)**，单帧 GT=8 预测=8 完全重合。

## 怎么接真实数据（面试时能讲的点）

`data/synthetic.py` 只是**数据生成器**。接 KITTI / nuScenes 时：

1. 把 `SynthDataset.__getitem__` 换成读取 `.bin`/`.pcd` 点云；
2. 复用 `scene_to_bev_targets` 的 BEV 投影逻辑（真实数据同样栅格化）；
3. 真值框按 KITTI 标注格式填入 `cls_t` / `reg_t`；
4. `models/bev_detector.py` 与 `train.py` **完全不用改**。

即：**网络与训练是通用的，换数据即可上真实数据集**——这也正是量产落地的常态。

## 与简历的呼应

- 本项目补齐了简历里的 DL/检测/BEV 缺口，证明具备现代感知算法开发能力；
- 手写 BEV 投影 + 检测头，比"调 mmdet3d"更能体现**底层理解**，适合智能驾驶感知岗面试深挖。
