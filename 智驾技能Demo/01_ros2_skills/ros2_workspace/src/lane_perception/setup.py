from setuptools import find_packages, setup

package_name = "lane_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/lane_launch.py"]),
        ("share/" + package_name + "/config", ["config/lane_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Liu Huiqi",
    maintainer_email="liuhuiqi@example.com",
    description="车道线感知：PyTorch U-Net 语义分割（演示）",
    license="MIT",
    entry_points={
        "console_scripts": [
            "lane_detector_node = lane_perception.lane_detector_node:main",
        ],
    },
)
