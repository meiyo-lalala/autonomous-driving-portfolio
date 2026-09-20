# 智驾技能 Demo —— 刘慧琪 求职作品集

> 面向杭州智能驾驶/SLAM/感知算法岗位的三个可运行 Demo，分别对应简历中需要"可视证据"的三项能力：
> **ROS2 工程能力 / 图优化（GTSAM·Ceres·g2o 同源数学）/ 深度学习（PyTorch）**。
> 全部在 Windows 本机直接可跑（ROS2 部分另附 WSL/Docker 运行方案）。

---

## 一、三个 Demo 一览

| 模块 | 展示能力 | 本机运行 | 输出 |
|---|---|---|---|
| **01_ros2_skills** | ROS2 工程框架 + 多传感器融合（EKF：轮速里程计+稀疏GNSS） | ✅ 无需 ROS2 的本地仿真 | `local_sim/output/ekf_fusion_result.png` |
| **02_pose_graph_slam** | 位姿图优化：因子图、SE(2) 雅可比、LM 求解器；GTSAM Python/C++ 对照实现 | ✅ 纯 Python 实现直接运行 | `output/pose_graph_result.png` |
| **03_lane_detection** | 深度学习：PyTorch U-Net 车道线语义分割（合成数据训练） | ✅ CPU 可训练/推理 | `output/lane_result.png` + 模型权重 |

三者形成一条"**定位 → 优化 → 感知**"的完整智驾技术链：01 做实时融合定位、02 做全局位姿优化、03 做车道级感知。

---

## 二、快速开始

### 环境
- Python 3.10+（本机验证：3.14.7）
- 依赖：`numpy`, `matplotlib`, `scipy`（01/02）；`torch`（03，CPU 版即可）
  ```powershell
  pip install numpy matplotlib scipy
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

### 一键运行全部（Windows PowerShell）
```powershell
cd 智驾技能Demo
python run_all.py
```
或逐个运行：
```powershell
python 01_ros2_skills/local_sim/run_local_sim.py     # EKF 融合（~10s）
python 02_pose_graph_slam/pose_graph_slam.py          # 位姿图优化（~2min）
python 03_lane_detection/train_lane_detector.py       # 车道线训练（~3min）
```

---

## 三、模块详解

### 01_ros2_skills —— ROS2 框架 + 多传感器融合
- `ros2_workspace/`：标准 ament_python 工作区，两个包：
  - `sensor_fusion_localization`：`fusion_node.py` 订阅 `/odom`、`/gnss_fix`，用 EKF（`ekf_2d.py`）融合并发布 `/fused_pose`；含 launch + params。
  - `lane_perception`：`lane_detector_node.py` 订阅 `/image`，加载 03 模块训练的 U-Net，发布 `/lane_mask`。
- `local_sim/`：**Windows 本机无需 ROS2 即可运行的在线融合仿真**（与 ROS2 节点同逻辑），演示里程计漂移被稀疏 GNSS 校正：
  - 里程计 XY-RMSE **0.92 m** → EKF+GNSS **0.43 m**（↓54%），终点误差 1.52 → 0.23 m
- **在 WSL/Ubuntu/Docker 中运行 ROS2 版**：
  ```bash
  # 安装 ROS2 Humble 后
  cd ros2_workspace
  colcon build --symlink-install
  source install/setup.bash
  ros2 launch sensor_fusion_localization fusion_launch.py
  ros2 launch lane_perception lane_launch.py
  ```

### 02_pose_graph_slam —— 位姿图优化（图优化库同源数学）
- `pose_graph_slam.py`：**纯 Python 因子图实现（本机直接运行）**
  - 因子：Prior（锚点/稀疏GNSS绝对约束）+ Between（里程计/回环相对约束）
  - SE(2) 局部扰动雅可比（已用数值差分验证，误差 ~1e-9）；Levenberg 阻尼 + 代价下降准则
  - 三配置对比：**A 纯里程计 → B 加回环 → C 加回环+稀疏RTK GNSS**
    | 配置 | XY-RMSE | 终点误差 |
    |---|---|---|
    | A. 纯里程计 | 0.92 m | 1.52 m |
    | B. +回环约束 | 0.89 m | 0.68 m |
    | C. +回环+稀疏GNSS | **0.38 m** | **0.43 m** |
  - 输出轨迹对比 + 误差柱状图
- `pose_graph_slam_gtsam.py`：GTSAM 官方 Python 实现（数学一致；gtsam wheel 暂不支持 Python 3.14，需 Python ≤3.13 或 Linux）
- `cpp/`：GTSAM C++ 实现（`cmake && make` 即可编译运行，需 Linux/WSL + GTSAM）
- `verify_math.py`：数学正确性自证——因子雅可比 vs 中心差分数值验证（误差 ~1e-9）、三配置数值复现
- `synthetic_data.py`：仿真数据生成（真值轨迹/里程计/回环/GNSS），并输出 `data_loop.txt` 供 C++ 版读取
- 关键点：**回环约束校正终点漂移，稀疏 GNSS 绝对约束把全程误差压到 0.4m 以内**——直接对应岗位 JD 中"多传感器融合定位、图优化、SLAM"要求。

### 03_lane_detection —— PyTorch 车道线检测
- 小型 U-Net（16.9 万参数，编码 1→16→32→64 / 解码 + 跳连）
- 合成数据集：路面纹理 + 随机曲率二次曲线车道线（280 训练 / 60 验证）
- 训练指标（CPU，60 epochs）：
  | 指标 | 起始 | 收敛 |
  |---|---|---|
  | Val Loss | 0.715 | **0.157** |
  | Val IoU | 0.062 | **1.000** |
- 输出：训练曲线 + 4 组"真值 vs 预测"推理对比图；权重存 `output/lane_detector.pt`
- 说明：本模块用合成数据演示完整管线（数据生成→训练→评估→推理→ROS2 部署）；真实场景可替换为 CULane/TuSimple 数据集（U-Net 结构不变）。

---

## 四、与岗位要求的能力对应

| 岗位 JD 常见要求 | 本 Demo 的证据 |
|---|---|
| ROS/ROS2 框架 | 01 标准 ROS2 双包工作区（节点/launch/params/复用模型），附 WSL 运行方案 |
| 图优化库 g2o/Ceres/GTSAM | 02 因子图 + LM 求解 + GTSAM Python/C++ 双实现，数学与纯 Python 版一致 |
| 深度学习 PyTorch | 03 U-Net 训练/评估/推理全流程，IoU 收敛到 1.0 |
| 多传感器融合定位 | 01 EKF + 02 图优化，两套方法交叉验证同一结论 |
| 工程化/可复现 | 固定随机种子、一键运行脚本、输出图与数值可复现 |

---

## 五、常见问题
- **gtsam 装不上？** gtsam wheel 目前到 Python 3.13；本机用纯 Python 版（数学等价），Linux/CI 用 GTSAM 版。
- **03 训练时间？** CPU 约 3 分钟（60 epochs）；可改 `epochs` 减少训练时间。
- **输出图片中文乱码？** 已自动配置 Microsoft YaHei；Linux 环境请安装中文字体或改 `plt.rcParams["font.family"]`。

（候选人：刘慧琪 · 长安大学交通信息工程及控制硕士 · 10 年车联网/智能网联研发与量产经验）
