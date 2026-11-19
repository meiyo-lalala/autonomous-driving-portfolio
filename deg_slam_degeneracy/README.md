# deg-slam-degeneracy

**停车场 / 地库场景下 2D 激光 SLAM 的退化（degeneracy）检测实验。**

一个可复现的最小工程：在合成的地库、纯走廊、玻璃幕墙三种场景上生成 2D 激光 + 带误差的轮速里程计，
用两阶段 scan-to-map 配准跑通，逐帧计算 Hessian 的条件数与退化方向，比较"无处理 / 退化感知"
两条管线的 ATE，并用 **Cartographer（ROS 2 Humble）** 在同一份 rosbag2 数据上做对照。

> 结论先行：长直通道里沿行进方向不可观测，**这不是算法能"算"回来的**——
> 退化检测能准确告诉你"哪一维丢了"（实测方向误差 < 10°），
> 但只靠沿该方向锚定里程计无法根治，因为里程计本身有尺度误差。
> 真正能压住漂移的是**绝对锚点**（反光柱 / 二维码 / UWB）或带回环的完整 SLAM。

---

## 这个仓库回答的三个问题

1. **退化能不能被可靠检测出来？**
   能。通道轴向的 cond 中位数 > 40、退化方向稳定指向通道轴（实测 174.1° / 2.9° / 7.2°，
   真值 180° / 0° / 0°，误差 < 10°），且不随帧抖动。

2. **检测到之后锚定里程计有没有用？**
   分场景：玻璃幕墙 + 8 m 量程时 ATE 从 4.754 m 降到 1.292 m（**−73%**）；
   纯走廊 + 8 m 量程反而从 4.787 m 升到 5.560 m（**+16%**，更差）。
   原因是锚定的参考是里程计，而这里的里程计带 **2.5% 尺度误差**——锚得越狠，偏置越实。

3. **Cartographer 在同一份数据上什么水平？**
   见 `results/cartographer/`：offline 建图正常产出 `map.pgm` + `trajectory.pbstream`，
   有回环与 submap 优化，漂移显著小于本仓库的无回环前端。这也解释了为什么生产环境不裸用 scan-to-map。

---

## 快速开始（不需要 ROS）

```bash
pip install -r requirements.txt

# 单个场景（约 500 帧 / 104 m 往返）
python src/run_sim_experiment.py --scenario garage --max-range 20 --laps 1 --out results

# 六组全跑（3 场景 × 2 量程）
bash scripts/run_sim_all.sh

# 出图 + 汇总表
python src/plot_results.py
```

产物：

```
results/<scenario>_r<range>/frames.csv     # 逐帧 cond / 退化方向 / inlier / rmse / 触发帧
results/<scenario>_r<range>/results.npy    # 轨迹（真值 / 里程计 / naive / deg_aware）
results/<scenario>_r<range>/summary.json   # ATE、RPE、退化帧占比
results/figures/fig1_trajectories.png
results/figures/fig2_condition_number.png
results/figures/fig3_direction_and_error.png
results/summary_table.csv
```

## 在 ROS 2 上复现 Cartographer

```bash
sudo apt install ros-humble-cartographer ros-humble-cartographer-ros \
                 ros-humble-rosbag2 ros-humble-rosbag2-storage-default-plugins
source /opt/ros/humble/setup.bash

python3 src/make_ros2_bag.py --scenario garage --out data/bags/garage_r20
bash scripts/run_cartographer.sh data/bags/garage_r20 results/cartographer/garage_r20
```

`run_cartographer.sh` 用 gflags 调用（**不是** `-p`，offline node 不吃 ROS 参数）：
`-configuration_directory`、`-configuration_basenames`（复数）、`-bag_filenames`、`-save_state_filename`。

---

## 实验设计

| 项 | 设置 |
| --- | --- |
| 场景 | `garage`（平行通道 + 车位 + 立柱）/ `corridor`（纯平行墙）/ `glass`（单侧墙缺失，模拟玻璃幕墙无回波） |
| 场地 | 60 × 30 m，通道 y ∈ [12, 18]，车沿 x 从 6 到 54 m |
| 轨迹 | 直线 → 半径 1 m 半圆 U 型转弯 → 偏移 2 m 的回程直线 → 半圆，按弧长重采样；**不做瞬时掉头**（360° 激光会把倒车误匹配成正确朝向，这是真实的坑） |
| 激光 | 720 束 / 360°，噪声 σ = 0.02 m，量程 20 m 与 8 m 两组 |
| 里程计 | 尺度 1.025（+2.5%）、偏航零偏 0.25 °/s、位置噪声 4 mm、航向噪声 0.05° |
| 运动 | 1.0 m/s，5 Hz，约 500 帧 / 104 m |
| 指标 | ATE（Umeyama 对齐）、RPE（Δ=5 帧）、逐帧 cond / 退化方向 / inlier / rmse |

三条管线跑同一份扫描：

- `odom_only`：只用轮速里程计累加。
- `naive`：scan-to-map 两阶段配准（point-to-point 粗 + 仅用高线性度法向的 point-to-line 精），
  无任何退化处理，附带故障保护（inlier < 0.35 或 rmse > 0.25 m 时回退里程计先验）。
- `deg_aware`：同上，另在 Hessian **平移 2×2 子块**上做特征分解，若 λmin/λmax < 3%，
  沿该方向加软先验 `min ‖r‖² + λ‖Vᵀ(x_odom − x)‖²`，用 Gauss-Newton 消元求解。

> 为什么只对平移子块做：3×3 整体的最小特征向量会把 θ 也锚到漂移的偏航上
> （本实验里程计 yaw 零偏 0.25 °/s，102 s 就是 25°），实测会把轨迹整体拉歪。

## 结果（ATE / m，Umeyama 对齐）

| 场景 | odom | naive | deg_aware | 退化帧占比 | 触发帧 |
| --- | --- | --- | --- | --- | --- |
| garage, 20 m | 2.716 | **0.908** | 0.908 | 74.1% | 0 |
| garage, 8 m | 2.716 | **0.982** | 0.982 | 33.5% | 0 |
| corridor, 20 m | 2.716 | **1.334** | 1.334 | 99.8% | 0 |
| corridor, 8 m | 2.716 | **4.787** | 5.560 | 87.8% | 6 |
| glass, 20 m | 2.716 | 16.810 | 16.810 | 100% | 0 |
| glass, 8 m | 2.716 | 4.754 | **1.292** | 73.3% | 13 |

完整数字见 `results/summary_table.csv`，逐帧曲线见 `results/figures/`。

### 怎么读这张表

- **地库（garage）**：立柱和车位开口提供了横向和部分纵向约束，scan matching 稳定赢过里程计（2.72 → 0.91 m）。
  退化判据一次都没触发（λmin/λmax 未跌破 3%），说明"条件数高 ≠ 一定退化"。
- **纯走廊（corridor）**：99.8% 帧被判退化，条件数中位数 ~50，退化方向 2.9°，与通道轴一致。
  20 m 量程时靠 20 m 外的墙端仍有微弱约束，结果 1.33 m；量程缩到 8 m 后连这点信息都没了，涨到 4.8 m。
- **玻璃墙（glass）**：20 m 时 16.8 m——几乎等于"闭眼走"，因为只剩一侧墙，横向完全不可观测；
  8 m 时反而更好，因为远距噪声点被砍掉，剩下的近处结构让软先验真正起作用，**−73%** 是本仓库最好的正面结果。
- **deg_aware 不是万灵药**：corridor_r8 触发 6 帧却更差。锚定方向来自里程计，
  而里程计有 2.5% 尺度误差——方向对了，参考值是错的。

## 方法要点

- **配准**：`src/degslam/scan_matcher.py`。粗配准 12 iter（Huber 0.25、单步限幅 0.25 m / 0.05 rad），
  精配准 6 iter 仅用线性度 > 0.4 的 PCA 法向；精配准 inlier 低于粗配准 85% 则回退粗结果。
- **退化分析**：`src/degslam/degeneracy.py`，`H = JᵀJ` 的特征分解（Zhang et al., IROS 2016 的思路）。
- **软先验**：`src/degslam/fusion.py`。注意符号——写成 `−b − λV(Vᵀ(x_odom−x))` 会把位姿推向里程计的反方向。
- **地图**：voxel 0.06 m 累积地图；**不用滑窗**。实测滑窗（16 m）会让地图跟着错误位姿一起移动，
  形成正反馈，玻璃场景漂移从 2 m 放大到 35 m。

## 局限

- 前端**没有回环检测**，长距离往返必然漂移；这不是 bug，而是本仓库用来对照 Cartographer 的基线设定。
- 数据是合成的（射线投射 + 高斯噪声），没有真实地库的动态障碍、反光、地面点与 IMU 重力对齐。
- 未覆盖 3D 激光（FAST-LIO2 需要 3D 点云与 IMU），扩展思路见 `results/report.md`。

## 为什么不用 TUM RGB-D

原计划用 `fr3_long_office_household`，实测本机下载速率约 50 KB/s（1.48 GB 需要数小时），
放弃并改为自建场景。合成数据的好处是**真值精确**、场景可控（玻璃墙这种退化 case 现实里很难找到），
代价是缺少真实噪声特性——这点在报告里明确标注，不冒充真实数据。

## 许可

MIT。自用与二次实验均可。
