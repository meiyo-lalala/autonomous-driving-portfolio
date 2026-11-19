"""Generate a ROS 2 bag from the synthetic parking-garage world.

The bag carries exactly what a 2D lidar SLAM front-end needs:
  /scan        sensor_msgs/LaserScan  (frame_id: laser)
  /clock       rosgraph_msgs/Clock    (sim time)
  /tf_static   base_link -> laser
"""
import argparse
import os
import sys

import numpy as np
import rclpy.serialization as serialization
import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degslam.scan_sim import GarageWorld, make_truth_trajectory
from degslam.odom import Odometer, integrate


class Writer:
    def __init__(self, uri):
        self.w = rosbag2_py.SequentialWriter()
        self.w.open(
            rosbag2_py.StorageOptions(uri=uri, storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        for name, typ in [("/scan", "sensor_msgs/msg/LaserScan"),
                          ("/clock", "rosgraph_msgs/msg/Clock"),
                          ("/tf_static", "tf2_msgs/msg/TFMessage"),
                          ("/odom", "nav_msgs/msg/Odometry"),
                          ("/tf", "tf2_msgs/msg/TFMessage")]:
            self.w.create_topic(rosbag2_py.TopicMetadata(
                name=name, type=typ, serialization_format="cdr"))

    def write(self, topic, msg, stamp_ns):
        self.w.write(topic, serialization.serialize_message(msg), stamp_ns)


def stamp(ns):
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["garage", "corridor", "glass"], default="garage")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--max-range", type=float, default=20.0)
    ap.add_argument("--n-beams", type=int, default=720)
    ap.add_argument("--out", default="data/bags/garage_r20")
    args = ap.parse_args()

    if args.scenario == "garage":
        world = GarageWorld(with_pillars=True, with_stalls=True)
    elif args.scenario == "corridor":
        world = GarageWorld(with_pillars=False, with_stalls=False)
    else:
        world = GarageWorld(with_pillars=False, with_stalls=False, glass_side=True)

    truth = make_truth_trajectory(rate=args.rate)
    if args.frames:
        truth = truth[: args.frames]

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        import shutil
        shutil.rmtree(out)
    # rosbag2 creates the directory itself; pre-creating it is an error
    os.makedirs(os.path.dirname(out), exist_ok=True)

    db = Writer(out)
    dt_ns = int(1e9 / args.rate)

    # static transform base_link -> laser (identity, sensor at the robot origin)
    tf_static = TFMessage()
    ts = TransformStamped()
    ts.header.frame_id = "base_link"
    ts.child_frame_id = "laser"
    ts.transform.translation.x = 0.35
    ts.transform.translation.y = 0.0
    ts.transform.translation.z = 0.30
    ts.transform.rotation.w = 1.0
    tf_static.transforms.append(ts)
    db.write("/tf_static", tf_static, 0)

    rng = np.random.default_rng(3)
    n_beams = args.n_beams
    angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)

    # wheel odometry as a real chassis would provide it: ~2% scale error,
    # small yaw drift. (The self-written front-end experiments deliberately
    # use a worse odometry, 2.5% + 0.25 deg/s, to stress degeneracy.)
    odom_poses = integrate(truth, Odometer(scale=1.02, yaw_bias_deg=0.03, seed=11),
                           dt_ns * 1e-9)

    for i, pose in enumerate(truth):
        now = i * dt_ns
        ranges = world.raycast(pose[:2], angles, args.max_range, 0.02, rng)
        # cartographer expects a monotonic, finite range array; use +inf for no return
        msg = LaserScan()
        msg.header.stamp = stamp(now)
        msg.header.frame_id = "laser"
        msg.angle_min = float(-np.pi)
        # angle_max must satisfy: angle_min + increment * (n_beams - 1).
        # Getting this off by one increment makes Cartographer reject every scan.
        msg.angle_max = float(-np.pi + (2 * np.pi / n_beams) * (n_beams - 1))
        msg.angle_increment = float(2 * np.pi / n_beams)
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / args.rate
        msg.range_min = 0.1
        msg.range_max = float(args.max_range)
        msg.ranges = [float(r) if np.isfinite(r) else float("inf") for r in ranges]
        db.write("/scan", msg, now)

        clk = Clock()
        clk.clock = stamp(now)
        db.write("/clock", clk, now)

        # odometry + the matching odom -> base_link transform
        op = odom_poses[i]
        od = Odometry()
        od.header.stamp = stamp(now)
        od.header.frame_id = "odom"
        od.child_frame_id = "base_link"
        od.pose.pose.position.x = float(op[0])
        od.pose.pose.position.y = float(op[1])
        od.pose.pose.orientation.z = float(np.sin(op[2] / 2))
        od.pose.pose.orientation.w = float(np.cos(op[2] / 2))
        db.write("/odom", od, now)

        tfm = TFMessage()
        t = TransformStamped()
        t.header.stamp = stamp(now)
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(op[0])
        t.transform.translation.y = float(op[1])
        t.transform.rotation.z = float(np.sin(op[2] / 2))
        t.transform.rotation.w = float(np.cos(op[2] / 2))
        tfm.transforms.append(t)
        db.write("/tf", tfm, now)

        if i % 100 == 0:
            print(f"  frame {i}/{len(truth)}", flush=True)

    print(f"bag written to {out}")


if __name__ == "__main__":
    main()
