# Autonomous Driving Algorithm Portfolio — Hand-built, Zero-framework

Two core autonomous-driving modules implemented **from scratch** — no deep-learning
framework black boxes, no off-the-shelf SLAM libraries — to demonstrate foundational
understanding beyond tool usage. Built to close the gap between traditional
CV/SIFT engineering and the modern AD perception + fusion stack.

## Highlights

### 1. BEV 3D Object Detection + Semantic Segmentation (PyTorch, from scratch)
- Point-cloud → BEV pseudo-image projection (hand-written voxelization)
- Anchor-free, CenterPoint-style detection head + BEV semantic segmentation head
  (background / road / vehicle)
- Gaussian heatmap training targets, soft-target Focal loss, Smooth-L1 regression,
  weighted cross-entropy segmentation loss, 3×3 local-peak NMS decode
- **Metrics (synthetic eval set, 200 samples):**
  - Recall **96.5%** / Precision **99.2%** / Localization L1 **0.090 m** (IoU ≥ 0.5)
  - BEV segmentation **mIoU 97.4%** (bg / road / vehicle = 0.997 / 0.999 / 0.927)

### 2. Multi-sensor Fusion Localization (SLAM, NumPy, from scratch)
- Hand-written point-to-point ICP (SVD / Horn method)
- Loosely-coupled EKF fusing LiDAR pose + IMU (with constant gyro bias) + GNSS
- Cumulative BEV occupancy mapping
- **Result:** EKF fusion ATE **1.06 m** vs IMU-only 2.94 m / ICP-only 1.27 m
  — sensor fusion clearly beats any single sensor.

### 3. Degeneracy Detection in 2D LiDAR SLAM (from scratch, ROS 2 optional)
- Synthetic parking-garage / plain-corridor / glass-wall worlds, 720-beam lidar,
  wheel odometry with a real 2.5% scale error
- Two-stage scan-to-map front-end: point-to-point coarse (Huber) + point-to-line fine
  (only high-linearity PCA normals), with fail-safe fallback to odometry
- Degeneracy detection on the Hessian: condition number + eigen-direction, then a
  soft prior `min ‖r‖² + λ‖Vᵀ(x_odom−x)‖²` along the weak translation direction only
- **Results (aligned ATE, 104 m round trip):** garage 0.908 m vs odometry 2.716 m;
  corridor 99.8% of frames flagged degenerate with direction error < 10°;
  glass wall + 8 m range **4.754 → 1.292 m (−73%)**, but the same trick makes
  corridor + 8 m **worse** (4.787 → 5.560 m) because the odometry it anchors to is biased
- **Cartographer (ROS 2 Humble) cross-check:** without odometry it cannot advance in a
  parallel corridor at all — the direct evidence that a garage robot needs wheel/IMU odometry

## Repository structure

| Path | What |
|------|------|
| `pytorch_3d_bev/` | BEV detection + segmentation (data / model / train / eval / demo) |
| `slam_fusion/` | ICP + EKF + BEV mapping (NumPy) |
| `deg_slam_degeneracy/` | Degeneracy-aware 2D lidar SLAM: synthetic worlds, scan matcher, experiments, Cartographer repro |
| `nav2_navigation_demo/` | Nav2 navigation demo + benchmark |
| `portfolio/` | Showcase page (`index.html`) with live metrics |
| `上传GitHub.md` | Chinese step-by-step GitHub upload guide |

## Run (CPU-only)

```bash
# BEV detection + segmentation
pip install torch numpy matplotlib tqdm --index-url https://download.pytorch.org/whl/cpu
cd pytorch_3d_bev
python train.py --epochs 20 --samples 1500 --batch 8 --save checkpoints/seg.pt
python evaluate.py --ckpt checkpoints/seg.pt --n 200
python demo.py --ckpt checkpoints/seg.pt --out demo_bev.png

# SLAM fusion localization
cd slam_fusion
python run.py

# Degeneracy experiments (no ROS needed)
cd deg_slam_degeneracy
pip install -r requirements.txt
python src/run_sim_experiment.py --scenario garage --max-range 20 --laps 1 --out results
python src/plot_results.py
```

## Why this repo exists

Demonstrates hands-on deep learning (detection / segmentation / BEV), modern AD
perception, and multi-sensor fusion — extending a C++/CUDA/OpenCV/SIFT vision
background into the current autonomous-driving stack. Every component is readable
and reproducible on CPU, with no hidden pretrained weights.
