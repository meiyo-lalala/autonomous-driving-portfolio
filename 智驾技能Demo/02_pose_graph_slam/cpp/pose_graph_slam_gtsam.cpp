// pose_graph_slam_gtsam.cpp —— GTSAM C++ 实现 2D 位姿图 SLAM
// 与 pose_graph_slam.py / pose_graph_slam_gtsam.py 数学一致。
//
// 编译运行（Linux / WSL / Docker，需已安装 GTSAM 与 Eigen）:
//   mkdir build && cd build
//   cmake .. && make -j
//   ./pose_graph_slam_gtsam
//
// 本文件演示: 因子图构建（Prior / Between）、GTSAM 优化 API、
//             Pose2 位姿流形、噪声模型配置 —— 覆盖岗位 JD 中的
//             "熟悉 g2o / Ceres / GTSAM 等图优化库" 要求。

#include <fstream>
#include <iostream>
#include <vector>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

using namespace gtsam;
using symbol_shorthand::X;

int main() {
  // ---- 1. 载入仿真数据（由 Python 端 synthetic_data.py 生成）----
  std::ifstream fin("data_loop.txt");
  if (!fin) {
    std::cerr << "未找到 data_loop.txt，请先运行 python synthetic_data.py 生成\n";
    return 1;
  }
  int n;
  fin >> n;
  std::vector<Pose2> gt(n), odom(n);
  double x, y, t;
  for (int i = 0; i < n; ++i) { fin >> x >> y >> t; gt[i] = Pose2(x, y, t); }
  for (int i = 0; i < n; ++i) { fin >> x >> y >> t; odom[i] = Pose2(x, y, t); }
  int m;
  fin >> m;
  std::vector<std::pair<int, int>> pairs(m);
  std::vector<Pose2> loop_noise(m);
  for (int k = 0; k < m; ++k) {
    int i, j;
    fin >> i >> j >> x >> y >> t;
    pairs[k] = {i, j};
    loop_noise[k] = Pose2(x, y, t);
  }
  int g;
  fin >> g;
  std::vector<int> gnss_idx(g);
  std::vector<Pose2> gnss_meas(g);
  for (int k = 0; k < g; ++k) {
    fin >> gnss_idx[k] >> x >> y >> t;
    gnss_meas[k] = Pose2(x, y, t);
  }

  // ---- 2. 构建因子图 ----
  NonlinearFactorGraph graph;

  // Prior: 锚定第 0 帧
  auto prior_noise = noiseModel::Diagonal::Sigmas(Vector3(0.5, 0.5, 0.5));
  graph.addPrior(X(0), gt[0], prior_noise);

  // 里程计因子 (轮速里程计: 平移 0.15m, 偏航 0.0025rad)
  auto odom_noise = noiseModel::Diagonal::Sigmas(Vector3(0.15, 0.15, 0.0025));
  for (int i = 1; i < n; ++i) {
    Pose2 rel = odom[i - 1].between(odom[i]);
    graph.emplace_shared<BetweenFactor<Pose2>>(X(i - 1), X(i), rel, odom_noise);
  }

  // 回环因子 (激光扫描匹配量级)
  auto loop_ns = noiseModel::Diagonal::Sigmas(Vector3(0.08, 0.08, 0.08));
  for (int k = 0; k < m; ++k) {
    graph.emplace_shared<BetweenFactor<Pose2>>(X(pairs[k].first),
                                               X(pairs[k].second),
                                               loop_noise[k], loop_ns);
  }

  // 稀疏 GNSS 绝对因子 (RTK/组合导航: 横向 0.2m, 航向 0.005rad)
  auto gnss_ns = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.005));
  for (int k = 0; k < g; ++k) {
    graph.addPrior(X(gnss_idx[k]), gnss_meas[k], gnss_ns);
  }

  // ---- 3. 初值：里程计轨迹 ----
  Values initial;
  for (int i = 0; i < n; ++i) initial.insert(X(i), odom[i]);

  // ---- 4. 优化 ----
  LevenbergMarquardtOptimizer optimizer(graph, initial);
  Values result = optimizer.optimize();

  double err_before = 0.0, err_after = 0.0;
  for (int i = 0; i < n; ++i) {
    Pose2 r_before = odom[i].between(gt[i]);
    Pose2 r_after = result.at<Pose2>(X(i)).between(gt[i]);
    err_before += r_before.x() * r_before.x() + r_before.y() * r_before.y();
    err_after += r_after.x() * r_after.x() + r_after.y() * r_after.y();
  }
  std::cout << "RMSE before: " << std::sqrt(err_before / n) << " m\n";
  std::cout << "RMSE after : " << std::sqrt(err_after / n) << " m\n";
  std::cout << "优化完成，共 " << n << " 个位姿, " << graph.size() << " 个因子\n";
  return 0;
}
