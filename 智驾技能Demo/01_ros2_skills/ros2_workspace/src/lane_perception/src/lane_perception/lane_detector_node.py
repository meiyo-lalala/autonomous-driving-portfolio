# -*- coding: utf-8 -*-
"""
lane_perception/lane_detector_node.py —— ROS2 车道线感知节点（演示）

订阅:
  /image            (sensor_msgs/Image)  前视灰度图
发布:
  /lane_mask        (sensor_msgs/Image)  车道线二值掩码(与输入同尺寸)
  /lane_points      (geometry_msgs/PointStamped) 每条线的底部锚点(占位, 演示)

模型: 03_lane_detection 训练的小型 U-Net (output/lane_detector.pt)。
运行(需 ROS2 + torch 环境, 见 README):
  ros2 launch lane_perception lane_launch.py
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

# 复用 03 模块的模型定义
import sys
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, "..", "..", "..", "..", "..", "..", "..",
                                "..", "03_lane_detection"))
# 注: 实际部署时请将 03 的训练脚本作为依赖包安装; 这里为演示路径
from train_lane_detector import TinyUNet  # noqa: E402


class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__("lane_detector_node")
        self.declare_parameter("model_path", "output/lane_detector.pt")
        self.declare_parameter("conf_thresh", 0.5)

        self.model = TinyUNet()
        model_path = self.get_parameter("model_path").value
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.get_logger().info("已加载模型: %s" % model_path)
        else:
            self.get_logger().warn("模型不存在(%s)，使用随机权重演示" % model_path)
        self.model.eval()
        self.thresh = self.get_parameter("conf_thresh").value

        self.sub = self.create_subscription(Image, "/image", self._on_image, 10)
        self.pub_mask = self.create_publisher(Image, "/lane_mask", 10)
        self.pub_pts = self.create_publisher(PointStamped, "/lane_points", 10)

    def _on_image(self, msg: Image):
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
            msg.height, msg.width, -1)[:, :, 0].astype(np.float32) / 255.0
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logit = self.model(t)
            prob = torch.sigmoid(logit)[0, 0].numpy()
        mask = (prob > self.thresh).astype(np.uint8) * 255

        out = Image()
        out.header = msg.header
        out.height, out.width, out.encoding = msg.height, msg.width, "mono8"
        out.step = msg.width
        out.data = mask.reshape(-1).tolist()
        self.pub_mask.publish(out)
        self.get_logger().info("推理一帧: 车道线像素 %d (%.1f%%)" %
                               (int((mask > 0).sum()), 100.0 * (mask > 0).mean()))


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
