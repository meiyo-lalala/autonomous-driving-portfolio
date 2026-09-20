# -*- coding: utf-8 -*-
"""
run_all.py —— 一键运行三个技能 Demo（Windows PowerShell / Linux 均可）

  python run_all.py
按顺序运行:
  01 EKF 多传感器融合仿真 (~10s)
  02 位姿图优化 (~2min)
  03 PyTorch 车道线训练 (~3min)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("01 | EKF 多传感器融合 (ROS2 同逻辑本机仿真)",
     [sys.executable, "01_ros2_skills/local_sim/run_local_sim.py"]),
    ("02 | 位姿图优化 (因子图 + LM, 纯Python/GTSAM同源)",
     [sys.executable, "02_pose_graph_slam/pose_graph_slam.py"]),
    ("03 | PyTorch 车道线 U-Net 训练",
     [sys.executable, "03_lane_detection/train_lane_detector.py"]),
]


def main():
    for title, cmd in STEPS:
        print("\n" + "=" * 60)
        print(">>", title)
        print("=" * 60)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print("[失败] %s (exit %d)" % (title, r.returncode))
            sys.exit(r.returncode)
    print("\n全部完成！产物见各模块 output/ 目录。")


if __name__ == "__main__":
    main()
