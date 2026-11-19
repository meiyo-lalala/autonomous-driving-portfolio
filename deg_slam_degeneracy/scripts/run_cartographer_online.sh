#!/usr/bin/env bash
# Online Cartographer run: node + bag playback + /tf recording.
#   ./scripts/run_cartographer_online.sh data/bags/garage_r20 results/cartographer/garage_r20
set -eo pipefail
BAG="${1:?usage: run_online.sh <bag> <out_dir>}"
OUT="${2:?usage: run_online.sh <bag> <out_dir>}"
CFG="$(cd "$(dirname "${BASH_SOURCE[0]}")/../config" && pwd)"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
BAG="$(cd "$(dirname "$BAG")" && pwd)/$(basename "$BAG")"
mkdir -p "$OUT"

source /opt/ros/humble/setup.bash

# cartographer_node takes gflags. Note the singular -configuration_basename
# (offline_node uses the plural -configuration_basenames). Calling the
# executable directly keeps ros2 run from rejecting the flags.
/opt/ros/humble/lib/cartographer_ros/cartographer_node \
  -configuration_directory "$CFG" -configuration_basename degslam_2d.lua \
  --ros-args -p use_sim_time:=true \
  > "$OUT/cartographer_node.log" 2>&1 &
NODE_PID=$!
sleep 3

sleep 2
python3 "$SRC/record_tf.py" --out "$OUT/trajectory_online.csv" --duration 120 > "$OUT/recorder.log" 2>&1 &
REC_PID=$!

sleep 2
ros2 bag play "$BAG" --clock -r 2.0 > "$OUT/bag_play.log" 2>&1 || true

sleep 5
kill $REC_PID 2>/dev/null || true
wait $REC_PID 2>/dev/null || true
kill $NODE_PID 2>/dev/null || true
wait $NODE_PID 2>/dev/null || true

echo "online run done -> $OUT/trajectory_online.csv"
head -3 "$OUT/trajectory_online.csv" 2>/dev/null
wc -l "$OUT/trajectory_online.csv" 2>/dev/null
