#!/usr/bin/env python3
"""记录 /odom 轨迹到 CSV（与 nav_benchmark 并行运行）"""
import sys, math, time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class TrajRec(Node):
    def __init__(self, out):
        super().__init__('traj_rec')
        self.f = open(out, 'w', newline='')
        self.w = csv_writer = None
        import csv
        self.w = csv.writer(self.f)
        self.w.writerow(['t_sim', 't_wall', 'x', 'y', 'yaw'])
        self.sub = self.create_subscription(Odometry, '/odom', self.cb, 30)
    def cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        t = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        self.w.writerow([round(t,3), round(time.time(),3),
                         round(msg.pose.pose.position.x,4), round(msg.pose.pose.position.y,4), round(yaw,4)])
        self.f.flush()

def main():
    rclpy.init()
    out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/traj.csv'
    node = TrajRec(out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.f.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
