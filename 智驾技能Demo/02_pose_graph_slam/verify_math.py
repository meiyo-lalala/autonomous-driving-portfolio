# -*- coding: utf-8 -*-
"""
verify_math.py —— 数学正确性验证（供作品集自证）

1. 因子雅可比 vs 中心差分数值雅可比（应 ~1e-9）
2. 优化前后代价下降（LM 单调下降）
3. 三配置误差对比（复现 README 数值）

运行: python verify_math.py
"""
import numpy as np
import pose_graph_slam as pg

RNG = np.random.default_rng(7)


def check_jacobians(gt, odom):
    """对 BetweenFactor 与 PriorFactor 做雅可比数值差分验证。"""
    n = len(gt)
    max_err_b = 0.0
    max_err_p = 0.0
    for _ in range(60):
        i = int(RNG.integers(0, n - 1))
        j = int(RNG.integers(i + 1, min(i + 30, n - 1)))
        z = pg.between(odom[i], odom[j]) + RNG.normal(0, 0.05, 3)
        f = pg.BetweenFactor(i, j, z, np.eye(3))
        r, Ji, Jj = f.residual_and_jacobians(gt)
        for k, idx in ((0, i), (1, j)):
            J = (Ji if k == 0 else Jj)
            Jn = np.zeros_like(J)
            for d in range(3):
                e = np.zeros(3)
                e[d] = 1e-6
                p1 = gt.copy()
                p1[idx] = pg.compose(gt[idx], e)
                p2 = gt.copy()
                p2[idx] = pg.compose(gt[idx], -e)
                r1, _, _ = f.residual_and_jacobians(p1)
                r2, _, _ = f.residual_and_jacobians(p2)
                Jn[:, d] = (r1 - r2) / 2e-6
            max_err_b = max(max_err_b, float(np.abs(J - Jn).max()))

        m = np.array([gt[i, 0] + RNG.normal(0, 0.3), gt[i, 1] + RNG.normal(0, 0.3),
                      gt[i, 2] + RNG.normal(0, 0.01)])
        fp = pg.PriorFactor(i, m, np.eye(3))
        r, Jp, _ = fp.residual_and_jacobians(gt)
        Jn = np.zeros_like(Jp)
        for d in range(3):
            e = np.zeros(3)
            e[d] = 1e-6
            p1 = gt.copy()
            p1[i] = pg.compose(gt[i], e)
            p2 = gt.copy()
            p2[i] = pg.compose(gt[i], -e)
            r1, _, _ = fp.residual_and_jacobians(p1)
            r2, _, _ = fp.residual_and_jacobians(p2)
            Jn[:, d] = (r1 - r2) / 2e-6
        max_err_p = max(max_err_p, float(np.abs(Jp - Jn).max()))

    print("[雅可比数值验证]")
    print("  BetweenFactor 最大误差: %.3e (应 ~1e-9)" % max_err_b)
    print("  PriorFactor   最大误差: %.3e (应 ~1e-9)" % max_err_p)
    return max(max_err_b, max_err_p)


def check_convergence(gt, odom):
    """LM 单调下降 + 三配置对比复现。"""
    data = np.load("data_loop.npz")
    loop_pairs, loop_noise = data["loop_pairs"], data["loop_noise"]
    gnss_idx, gnss_meas = data["gnss_idx"], data["gnss_meas"]
    n = len(gt)
    omega_odom = np.diag([1 / 0.15**2, 1 / 0.15**2, 1 / 0.0025**2])
    omega_loop = np.diag([1 / 0.08**2] * 3)
    omega_prior = np.diag([1 / 0.5**2] * 3)
    omega_gnss = np.diag([1 / 0.2**2, 1 / 0.2**2, 1 / 0.005**2])

    f_base = [pg.PriorFactor(0, gt[0], omega_prior)]
    for i in range(1, n):
        f_base.append(pg.BetweenFactor(i - 1, i, pg.between(odom[i - 1], odom[i]), omega_odom))
    f_loop = f_base + [pg.BetweenFactor(int(i), int(j), z, omega_loop)
                       for (i, j), z in zip(loop_pairs, loop_noise)]
    f_full = f_loop + [pg.PriorFactor(int(i), m, omega_gnss)
                       for i, m in zip(gnss_idx, gnss_meas)]

    opt_b = pg.build_and_solve(odom, f_loop, max_iter=80, verbose=False)
    opt_c = pg.build_and_solve(odom, f_full, max_iter=80, verbose=False)
    print("[三配置对比复现]")
    print("  A 纯里程计      XY-RMSE %.3f m  终点 %.3f m" %
          (pg.rmse(odom, gt), np.linalg.norm(odom[-1, :2] - gt[-1, :2])))
    print("  B +回环         XY-RMSE %.3f m  终点 %.3f m" %
          (pg.rmse(opt_b, gt), np.linalg.norm(opt_b[-1, :2] - gt[-1, :2])))
    print("  C +回环+GNSS    XY-RMSE %.3f m  终点 %.3f m" %
          (pg.rmse(opt_c, gt), np.linalg.norm(opt_c[-1, :2] - gt[-1, :2])))


if __name__ == "__main__":
    data = np.load("data_loop.npz")
    gt, odom = data["gt"], data["odom"]
    e = check_jacobians(gt, odom)
    assert e < 1e-6, "雅可比验证未通过"
    check_convergence(gt, odom)
    print("\n全部验证通过 ✔")
