# -*- coding: utf-8 -*-
"""
sensor_fusion_localization/fusion_node.py —— ROS2 多传感器融合定位节点

订阅:
  /odom            (nav_msgs/Odometry)    轮速里程计, 高频
  /gnss_fix        (sensor_msgs/NavSatFix) 稀疏GNSS, 低频 (仅用经纬度/航向, 演示用简化消息)
发布:
  /fused_pose      (geometry_msgs/PoseStamped)  EKF 融合后的定位结果

运行(需 ROS2 环境, 见 README):
  ros2 launch sensor_fusion_localization fusion_launch.py
"""
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix

from sensor_fusion_localization.ekf_2d import Ekf2D


class FusionNode(Node):
    def __init__(self):
        super().__init__("fusion_node")
        self.declare_parameter("sigma_odom_xy", 0.15)
        self.declare_parameter("sigma_odom_yaw", 0.0025)
        self.declare_parameter("sigma_gnss_xy", 0.2)
        self.declare_parameter("sigma_gnss_yaw", 0.005)

        self.ekf = Ekf2D(
            self.get_parameter("sigma_odom_xy").value,
            self.get_parameter("sigma_odom_yaw").value,
            self.get_parameter("sigma_gnss_xy").value,
            self.get_parameter("sigma_gnss_yaw").value,
        )
        self._last_odom_xy = None
        self._last_odom_yaw = None
        self._first_odom = True

        self.sub_odom = self.create_subscription(
            Odometry, "/odom", self._on_odom, 10)
        self.sub_gnss = self.create_subscription(
            NavSatFix, "/gnss_fix", self._on_gnss, 10)
        self.pub_fused = self.create_publisher(PoseStamped, "/fused_pose", 10)
        self.get_logger().info("fusion_node 已启动 (EKF: 轮速里程计 + 稀疏GNSS)")

    def _on_odom(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        x, y = p.x, p.y
        # 四元数 -> 偏航角 (简化: 仅用 z/w)
        yaw = 2.0 * np.arctan2(q.z, q.w)
        if self._first_odom:
            self.ekf.init(x, y, yaw)
            self._last_odom_xy, self._last_odom_yaw = (x, y), yaw
            self._first_odom = False
            return
        dx = x - self._last_odom_xy[0]
        dy = y - self._last_odom_xy[1]
        dyaw = self._wrap(yaw - self._last_odom_yaw)
        self.ekf.predict_odom(dx, dy, dyaw)     # 里程计高频预测
        self._last_odom_xy, self._last_odom_yaw = (x, y), yaw
        self._publish()

    def _on_gnss(self, msg: NavSatFix):
        # 演示: 假设节点上游已将经纬度转成局部平面坐标并写入 position_covariance
        c = np.array(msg.position_covariance).reshape(3, 3)
        self.ekf.R = np.diag([c[0, 0], c[1, 1], 0.005 ** 2])
        self.ekf.update_gnss(msg.latitude, msg.longitude, 0.0)  # 占位, 见 README 说明
        self._publish()

    def _publish(self):
        pose = self.ekf.pose
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = float(np.sin(pose[2] / 2.0))
        msg.pose.orientation.w = float(np.cos(pose[2] / 2.0))
        self.pub_fused.publish(msg)

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
