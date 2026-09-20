from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="lane_perception",
            executable="lane_detector_node",
            name="lane_detector_node",
            output="screen",
            parameters=["config/lane_params.yaml"],
        ),
    ])
