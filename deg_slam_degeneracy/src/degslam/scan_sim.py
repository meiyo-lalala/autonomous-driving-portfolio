"""Synthetic underground-parking-garage world + 2D lidar ray casting.

Geometry mirrors the failure mode that matters for a parking-garage charging
robot: one long straight aisle bounded by two parallel walls, a row of
near-identical parking stalls on one side and only a handful of pillars
providing longitudinally informative features.
"""
import numpy as np


def _rect(x0, y0, x1, y1):
    """Four segments of an axis-aligned rectangle."""
    p = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(np.array(p[i], float), np.array(p[(i + 1) % 4], float)) for i in range(4)]


def _seg(a, b):
    return [(np.array(a, float), np.array(b, float))]


class GarageWorld:
    def __init__(self, length=60.0, width=30.0, aisle=(12.0, 18.0),
                 stall_pitch=3.0, stall_depth=5.0, n_pillars=3,
                 with_pillars=True, with_stalls=True,
                 glass_side=False, seed=0):
        self.length = length
        self.width = width
        self.y_lo, self.y_hi = aisle
        self.with_pillars = with_pillars
        self.with_stalls = with_stalls
        self.glass_side = glass_side
        self.segments = []
        self.rng = np.random.default_rng(seed)
        self._build(stall_pitch, stall_depth, n_pillars)

    def _build(self, pitch, depth, n_pillars):
        L, W = self.length, self.width
        lo, hi = self.y_lo, self.y_hi

        # outer shell
        self.segments += _rect(0.0, 0.0, L, W)

        # low side of the aisle: a glass / dark-paint surface returns no echo,
        # which is exactly what happens next to curtain walls and black cars
        if not self.glass_side:
            self.segments += _seg((0.0, lo), (L, lo))

        if self.with_stalls:
            # high side: stall openings every `pitch` metres
            n_stalls = int(np.floor(L / pitch))
            for k in range(n_stalls):
                x0 = k * pitch
                self.segments += _seg((x0, hi), (x0 + 0.3, hi))
                self.segments += _seg((x0 + pitch - 0.3, hi), (x0 + pitch, hi))
                self.segments += _seg((x0, hi), (x0, hi + depth))
                self.segments += _rect(x0 + 0.55, hi + 0.55,
                                       x0 + 2.05, hi + depth - 0.55)
            self.segments += _seg((0.0, hi + depth), (L, hi + depth))
        else:
            # plain parallel corridor: no openings, no parked cars
            self.segments += _seg((0.0, hi), (L, hi))

        # pillars: the only longitudinally informative features
        if self.with_pillars:
            xs = np.linspace(L * 0.34, L * 0.83, n_pillars)
            for x in xs:
                self.segments += _rect(x - 0.3, lo + 0.9, x + 0.3, lo + 1.5)
            self.pillars_x = xs
        else:
            self.pillars_x = np.array([])

    def raycast(self, origin, angles, max_range=20.0, noise_std=0.02, rng=None):
        """Vectorised ray/segment intersection. Returns ranges (inf where no hit)."""
        rng = rng or self.rng
        origin = np.asarray(origin, float)
        dirs = np.column_stack([np.cos(angles), np.sin(angles)])
        best = np.full(len(angles), np.inf)

        for a, b in self.segments:
            e = b - a
            # [dx, -ex; dy, -ey] [t; u] = [rx; ry]
            det = -dirs[:, 0] * e[1] + e[0] * dirs[:, 1]
            ok = np.abs(det) > 1e-12
            if not np.any(ok):
                continue
            rx = a[0] - origin[0]
            ry = a[1] - origin[1]
            d = dirs[ok]
            t = (e[0] * ry - e[1] * rx) / det[ok] * (-1.0)  # derived below
            # recompute cleanly: t = (-rx*ey + ex*ry)/det
            t = (-rx * e[1] + e[0] * ry) / det[ok]
            u = (d[:, 0] * ry - d[:, 1] * rx) / det[ok]
            hit = (t > 0.05) & (t <= max_range) & (u >= 0.0) & (u <= 1.0)
            if not np.any(hit):
                continue
            idx = np.flatnonzero(ok)[hit]
            np.minimum.at(best, idx, t[hit])

        ranges = best
        if noise_std > 0.0:
            finite = np.isfinite(ranges)
            ranges[finite] += rng.normal(0.0, noise_std, finite.sum())
        return ranges

    def to_scan(self, pose, n_beams=720, max_range=20.0, noise_std=0.02, rng=None):
        """Scan points expressed in the ROBOT (sensor) frame, origin at pose[:2]."""
        angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        ranges = self.raycast(pose[:2], angles, max_range, noise_std, rng)
        finite = np.isfinite(ranges)
        a = angles[finite]
        pts = np.column_stack([ranges[finite] * np.cos(a), ranges[finite] * np.sin(a)])
        # rotate by the robot heading so the scan lives in the body frame
        c, s = np.cos(-pose[2]), np.sin(-pose[2])
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T


def make_truth_trajectory(length=60.0, aisle_y=15.0, speed=1.0, rate=5.0,
                          x_start=6.0, x_end=54.0, laps=1.0, turn_radius=1.0):
    """Back-and-forth aisle traversal with U-turns.

    A real chassis cannot reverse its heading instantaneously; an abrupt pi
    flip also makes a 360-degree lidar happily match the "backwards" scan and
    flip the pose.  So the path is: straight leg -> half circle -> straight leg
    on the offset line -> half circle back, repeat for `laps` round trips.
    """
    r = turn_radius
    y0 = aisle_y
    dense = []

    def line(p0, p1, th, n=400):
        for s in np.linspace(0.0, 1.0, n):
            dense.append((p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1]), th))

    def half_turn(cx, cy, th_from, th_to, n=200):
        # sweep from th_from to th_to through the point (cx, cy - r) etc.
        for s in np.linspace(0.0, 1.0, n):
            phi = np.pi / 2 - s * np.pi
            x = cx - r * np.cos(phi)
            y = cy - r * np.sin(phi)
            dense.append((x, y, phi - np.pi / 2))

    for lap in range(int(np.ceil(laps))):
        line((x_start, y0), (x_end, y0), 0.0)
        half_turn(x_end, y0 + r, 0.0, -np.pi)
        line((x_end, y0 + 2 * r), (x_start, y0 + 2 * r), np.pi)
        half_turn(x_start, y0 + r, np.pi, 0.0)

    dense = np.array(dense)
    seg = np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    n = int((laps * 2 * (x_end - x_start) + laps * 4 * np.pi * r / np.pi) / (speed / rate))
    n = max(n, 10)
    s_samples = np.linspace(0.0, total, n)
    poses = []
    for s in s_samples:
        idx = np.clip(np.searchsorted(arc, s) - 1, 0, len(dense) - 2)
        a, b = dense[idx], dense[idx + 1]
        d = arc[idx + 1] - arc[idx]
        t = 0.0 if d < 1e-9 else (s - arc[idx]) / d
        xy = a[:2] + t * (b[:2] - a[:2])
        th = np.arctan2(np.sin(b[2]), np.cos(b[2])) if t > 0.5 else a[2]
        poses.append(np.array([xy[0], xy[1], th]))
    return poses
