#!/usr/bin/env python3
"""Nav2 导航基准评测节点（同步串行 + 单目标 45s 超时）"""
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

STATIC_OBS = [(1.5, 1.5, 0.52), (-1.8, 1.2, 0.55), (-1.5, -1.8, 0.40), (2.2, -1.5, 0.55)]
WALL_LIMIT = 3.75
OBS_ROBOT_R = 0.40
GOAL_TIMEOUT = 45.0

GOALS = [
    ('起点->东北', 2.5, 2.5),
    ('东北->东南', 2.5, -2.5),
    ('东南->西南', -2.5, -2.5),
    ('西南->西北', -2.5, 2.5),
    ('西北->中上(动态障碍区)', 0.5, 1.0),
    ('中上->中下(动态障碍区)', 1.0, -0.5),
]


class NavBenchmark(Node):
    def __init__(self):
        super().__init__('nav_benchmark', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.obs_sub = self.create_subscription(Odometry, '/odom_obs', self.obs_cb, 10)
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)
        self.plan_sub = self.create_subscription(Path, '/plan', self.plan_cb, 10)
        self.odom_pos = None
        self.map_pos = None
        self.obs_pos = None
        self.results = []
        self.current_goal = None
        self.plan_ts = None
        self.goal_ts = None
        self.dist_accum = 0.0
        self.last_odom = None
        self.collided = False
        self._plan_received = False
        self.get_logger().info('等待 navigate_to_pose action 服务器...')
        while not self.ac.wait_for_server(timeout_sec=30.0):
            self.get_logger().info('仍在等待 action 服务器...')
        self.get_logger().info('action 服务器就绪，发布初始位姿给 amcl')
        self.initial_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        for i in range(3):
            ip = PoseWithCovarianceStamped()
            ip.header.frame_id = 'map'
            ip.header.stamp = self.get_clock().now().to_msg()
            ip.pose.pose.position.x = 0.0
            ip.pose.pose.position.y = 0.0
            ip.pose.pose.orientation.w = 1.0
            ip.pose.covariance[0] = 0.1
            ip.pose.covariance[7] = 0.1
            ip.pose.covariance[35] = 0.1
            self.initial_pub.publish(ip)
            time.sleep(1.0)
        self.get_logger().info('初始位姿已发布，等待定位与 costmap 稳定 12s')
        time.sleep(12.0)

    def odom_cb(self, msg):
        self.odom_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.last_odom is not None and self.current_goal is not None:
            dx = msg.pose.pose.position.x - self.last_odom[0]
            dy = msg.pose.pose.position.y - self.last_odom[1]
            self.dist_accum += math.hypot(dx, dy)
        self.last_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def plan_cb(self, msg):
        if self.current_goal is not None and not self._plan_received:
            self._plan_received = True
            self.plan_ts = time.time()

    def obs_cb(self, msg):
        self.obs_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def amcl_cb(self, msg):
        self.map_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def check_collision(self):
        pos = self.map_pos if self.map_pos is not None else self.odom_pos
        if pos is None:
            return False
        px, py = pos
        if abs(px) > WALL_LIMIT or abs(py) > WALL_LIMIT:
            return True
        for cx, cy, safe in STATIC_OBS:
            if math.hypot(cx - px, cy - py) < safe:
                return True
        if self.obs_pos is not None:
            ox = self.obs_pos[0]
            oy = self.obs_pos[1] - 0.5
            if math.hypot(ox - px, oy - py) < OBS_ROBOT_R:
                return True
        return False

    def record(self, label, success, status, elapsed, plan, dist, collision=False):
        self.results.append({
            'goal': label, 'success': success, 'status': status,
            'total_time_s': round(elapsed, 2),
            'plan_latency_s': round(plan, 3) if plan else None,
            'path_length_m': round(dist, 2) if dist else None,
            'collision': collision,
        })

    def run_goal(self, label, x, y):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0
        self.current_goal = label
        self.goal_ts = time.time()
        self.plan_ts = None
        self._plan_received = False
        self.dist_accum = 0.0
        self.collided = False
        self.last_odom = None
        self.get_logger().info(f'>>> 发送目标: {label} ({x},{y})')
        send_future = self.ac.send_goal_async(goal)
        t0 = time.time()
        while rclpy.ok() and not send_future.done() and time.time() - t0 < 10.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not send_future.done():
            self.get_logger().warn(f'发送超时: {label}')
            self.record(label, False, 'SEND_TIMEOUT', 10.0, None, self.dist_accum)
            self.current_goal = None
            return
        gh = send_future.result()
        if not gh.accepted:
            self.get_logger().warn(f'目标被拒: {label}')
            self.record(label, False, 'REJECTED', 0.0, None, None)
            self.current_goal = None
            return
        result_future = gh.get_result_async()
        t0 = time.time()
        while rclpy.ok() and not result_future.done() and time.time() - t0 < GOAL_TIMEOUT:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not result_future.done():
            self.get_logger().warn(f'目标超时 {GOAL_TIMEOUT}s，取消: {label}')
            cancel_future = gh.cancel_goal_async()
            c0 = time.time()
            while rclpy.ok() and not cancel_future.done() and time.time() - c0 < 5.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.5)
            elapsed = time.time() - self.goal_ts
            plan_latency = (self.plan_ts - self.goal_ts) if self.plan_ts else None
            collided = self.check_collision()
            self.record(label, False, 'TIMEOUT', elapsed, plan_latency, self.dist_accum, collided)
            self.current_goal = None
            return
        result = result_future.result()
        elapsed = time.time() - self.goal_ts
        plan_latency = (self.plan_ts - self.goal_ts) if self.plan_ts else None
        success = result.status == 4
        collided = self.check_collision()
        if collided:
            success = False
        self.record(label, success, f'STATUS_{result.status}', elapsed, plan_latency, self.dist_accum, collided)
        plan_str = f'{plan_latency:.2f}s' if plan_latency is not None else 'N/A'
        self.get_logger().info(f'<<< {label}: {"成功" if success else "失败"} 耗时{elapsed:.1f}s 规划{plan_str} 路径{self.dist_accum:.2f}m 碰撞={collided}')
        self.current_goal = None

    def run(self):
        for label, x, y in GOALS:
            self.run_goal(label, x, y)
        n = len(self.results)
        ok = sum(1 for r in self.results if r['success'])
        coll = sum(1 for r in self.results if r['collision'])
        summary = {
            'total_goals': n,
            'success_rate': round(ok / n, 3),
            'collision_rate': round(coll / n, 3),
            'avg_total_time_s': round(sum(r['total_time_s'] for r in self.results) / n, 2),
            'avg_plan_latency_s': round(
                sum(r['plan_latency_s'] for r in self.results if r['plan_latency_s']) /
                max(1, sum(1 for r in self.results if r['plan_latency_s'])), 3),
            'avg_path_length_m': round(
                sum(r['path_length_m'] for r in self.results if r['path_length_m']) /
                max(1, sum(1 for r in self.results if r['path_length_m'])), 2),
        }
        out = {'summary': summary, 'results': self.results}
        with open('/root/nav2_demo_ws/benchmark_result.json', 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print('===== BENCHMARK SUMMARY =====')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        self.get_logger().info('评测完成，结果已写入 /root/nav2_demo_ws/benchmark_result.json')


def main():
    import sys
    rclpy.init(args=sys.argv)
    node = NavBenchmark()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
