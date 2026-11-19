"""Record map->base_link from /tf into a CSV while Cartographer runs online."""
import argparse
import csv
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class Recorder(Node):
    def __init__(self, out):
        super().__init__("tf_recorder")
        self.rows = []
        self.out = out
        qos = QoSProfile(depth=100)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE
        self.create_subscription(TFMessage, "/tf", self.cb, qos)

    def cb(self, msg):
        for t in msg.transforms:
            if t.header.frame_id.lstrip("/") == "map" and t.child_frame_id.lstrip("/") == "base_link":
                q = t.transform.rotation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                                 1 - 2 * (q.y * q.y + q.z * q.z))
                stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
                self.rows.append([stamp, t.transform.translation.x,
                                  t.transform.translation.y, yaw])

    def dump(self):
        with open(self.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stamp", "x", "y", "yaw"])
            w.writerows(self.rows)
        print(f"wrote {len(self.rows)} poses to {self.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=120.0)
    args = ap.parse_args()
    rclpy.init()
    node = Recorder(args.out)
    import time
    t0 = time.time()
    while time.time() - t0 < args.duration:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.dump()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
