from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="sensor_fusion_localization",
            executable="fusion_node",
            name="fusion_node",
            output="screen",
            parameters=["config/fusion_params.yaml"],
        ),
    ])
