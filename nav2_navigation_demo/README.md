# Nav2 Navigation Demo — ROS2 Humble + Nav2 + Gazebo

A from-scratch differential-drive robot (URDF + Gazebo) navigating an 8×8 m
cluttered world with Nav2: **NavFn (A\*) global planner + DWB local planner**,
slam-free (known map), 6-goal autonomous navigation benchmark.

## Hardware / Software

- Ubuntu 22.04 (WSL2), ROS2 Humble, Navigation2, Gazebo (headless)
- Robot: custom URDF differential drive — 360° LiDAR (PointCloud2 → LaserScan),
  IMU, diff_drive controller
- World: 8×8 m, 4 walls + 4 static obstacles (boxes / cylinder)

## Pipeline

```
LiDAR → LaserScan → Nav2 costmap (obstacle layer) → NavFn global planner (A*)
→ DWB local planner (sim_time 2.0 s) → diff_drive cmd_vel → Gazebo odometry
```

- Global path: NavFn (A\*) with inflation radius 0.5 m
- Local control: DWB (BaseObstacle / PathAlign / PathDist / GoalAlign / GoalDist)
- 6 goals traverse the whole map: (2.5,2.5)→(2.5,-2.5)→(-2.5,-2.5)→(-2.5,2.5)→(0.5,1.0)→(1.0,-0.5)

## Benchmark results (6-goal run, real simulation)

| Metric | Value |
|--------|-------|
| Goal success rate | **6/6 = 100%** |
| Collision rate | **0/6 = 0%** |
| Average total time per goal | **19.8 s** |
| Average plan latency | **12 ms** |
| Average path length | **3.75 m** |
| Goal tolerance | 0.25 m (reach within 0.3 m) |

## Demo video

`Nav2导航Demo_女声解说.mp4` — 6-goal trajectory animation with Chinese
voice-over commentary (84.9 s, 1280×720).

## Reproduce

```bash
# inside WSL2 Ubuntu 22.04 with ROS2 Humble + Nav2 + Gazebo
cd nav2_navigation_demo
colcon build
# terminal 1: launch simulation (headless)
ros2 launch robot_sim sim.launch.py gui:=false
# terminal 2: run 6-goal benchmark
ros2 run nav_demo nav_benchmark --ros-args -p use_sim_time:=true
```

Key sources: `nav_benchmark.py` (benchmark runner), `nav2_params.yaml`
(Nav2 tuning), `traj_rec.py` (trajectory recorder), `nav_anim.py`
(Matplotlib animation generator).
