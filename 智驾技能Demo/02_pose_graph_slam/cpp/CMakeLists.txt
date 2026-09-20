cmake_minimum_required(VERSION 3.10)
project(pose_graph_slam_gtsam CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(pose_graph_slam_gtsam pose_graph_slam_gtsam.cpp)
target_link_libraries(pose_graph_slam_gtsam gtsam)
target_include_directories(pose_graph_slam_gtsam PRIVATE ${EIGEN3_INCLUDE_DIR})
