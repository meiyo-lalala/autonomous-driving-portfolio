#!/usr/bin/env bash
# Build the ROS 2 bags and run Cartographer offline on them.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /opt/ros/humble/setup.bash

python3 src/make_ros2_bag.py --scenario garage --out data/bags/garage_r20
python3 src/make_ros2_bag.py --scenario glass --max-range 8 --out data/bags/glass_r8

bash scripts/run_cartographer.sh data/bags/garage_r20 results/cartographer/garage_r20
bash scripts/run_cartographer.sh data/bags/glass_r8   results/cartographer/glass_r8

echo "ALL CARTOGRAPHER RUNS DONE"
