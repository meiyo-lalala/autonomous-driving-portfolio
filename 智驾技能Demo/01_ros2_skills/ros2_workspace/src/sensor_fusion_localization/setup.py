from setuptools import find_packages, setup

package_name = "sensor_fusion_localization"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/fusion_launch.py"]),
        ("share/" + package_name + "/config", ["config/fusion_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Liu Huiqi",
    maintainer_email="liuhuiqi@example.com",
    description="多传感器融合定位：轮速里程计 + 稀疏GNSS 的 EKF 融合",
    license="MIT",
    entry_points={
        "console_scripts": [
            "fusion_node = sensor_fusion_localization.fusion_node:main",
        ],
    },
)
