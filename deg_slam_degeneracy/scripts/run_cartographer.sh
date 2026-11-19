#!/usr/bin/env bash
# Run Cartographer ROS 2 offline on a bag produced by src/make_ros2_bag.py
#
#   ./scripts/run_cartographer.sh data/bags/garage_r20 results/cartographer/garage_r20
#
set -eo pipefail

BAG="${1:?usage: run_cartographer.sh <bag_uri> <out_dir>}"
OUT="${2:?usage: run_cartographer.sh <bag_uri> <out_dir>}"
CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../config" && pwd)"

source /opt/ros/humble/setup.bash
mkdir -p "$OUT"

echo "[1/3] offline SLAM -> pbstream"
ros2 run cartographer_ros cartographer_offline_node \
  -configuration_directory "$CFG" \
  -configuration_basenames degslam_2d.lua \
  -bag_filenames "$BAG" \
  -save_state_filename "$OUT/trajectory.pbstream" 2>&1 | tee "$OUT/offline.log" | tail -12

echo "[2/3] pbstream -> ros map"
ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
  -pbstream_filename "$OUT/trajectory.pbstream" \
  -map_filestem "$OUT/map" \
  -resolution 0.05 2>&1 | tail -5

echo "[3/3] done: $OUT"
ls -la "$OUT"
