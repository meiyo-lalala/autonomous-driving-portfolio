#!/usr/bin/env python3
"""Matplotlib 后处理动画：地图+障碍+轨迹+目标+指标 -> MP4"""
import csv, math, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname='/usr/share/fonts/truetype/msyh/msyh.ttc')
RES = 0.05
N = 160
ORIGIN = -4.0
GOALS = [('起点', (0.0, 0.0)), ('目标1', (2.5, 2.5)), ('目标2', (2.5, -2.5)),
         ('目标3', (-2.5, -2.5)), ('目标4', (-2.5, 2.5)), ('目标5', (0.5, 1.0)),
         ('目标6', (1.0, -0.5))]
GOAL_PTS = [(2.5, 2.5), (2.5, -2.5), (-2.5, -2.5), (-2.5, 2.5), (0.5, 1.0), (1.0, -0.5)]
BOXES = [(3.2, 2.8, 0.3, 0.8, 0.6), (-3.2, 3.0, 0.8, 0.7, 0.9), (3.0, -3.2, -0.4, 1.0, 0.5)]
CYL = [(-3.4, -3.4, 0.35)]

# 读轨迹
traj = []
with open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/traj.csv') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        try:
            traj.append((float(row[0]), float(row[2]), float(row[3])))
        except ValueError:
            pass
print('TRAJ_POINTS', len(traj))
if not traj:
    sys.exit('EMPTY_TRAJ')

# 读地图
img = np.array(Image.open('/root/nav2_demo_ws/src/nav_demo/params/nav_world_map.pgm'))
occ = (img < 128).astype(float)

# 到达判定（距离<0.3m 且时间递增）
arrived = []
for i, (gx, gy) in enumerate(GOAL_PTS):
    t0 = None
    for t, x, y in traj:
        if math.hypot(x - gx, y - gy) < 0.3:
            t0 = t
            break
    arrived.append(t0)

tmax = traj[-1][0]
FPS = 25
SPEED = 1.6  # 时间加速系数（仿真秒->播放秒）
DUR = max(8.0, tmax / SPEED)
n_frames = int(DUR * FPS)
print('DUR', DUR, 'FRAMES', n_frames, 'TMAX', tmax)

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
ax = fig.add_subplot(111)

def draw_static():
    ax.clear()
    # 地图背景（含墙与障碍像素）
    extent = [ORIGIN, ORIGIN + N*RES, ORIGIN, ORIGIN + N*RES]
    ax.imshow(occ, cmap='Greys', extent=extent, origin='lower', alpha=0.9, zorder=0)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

def draw_frame(i):
    draw_static()
    # 播放时间 -> 仿真时间
    play_t = i / FPS
    sim_t = play_t * SPEED
    # 轨迹
    pts = [(x, y) for t, x, y in traj if t <= sim_t]
    if pts:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, '-', color='#1565c0', lw=2.2, zorder=3)
        ax.plot(xs[-1], ys[-1], 'o', color='#c62828', ms=9, zorder=4)
    # 目标点
    for k, (gx, gy) in enumerate(GOAL_PTS):
        ok = arrived[k] is not None and arrived[k] <= sim_t
        ax.plot(gx, gy, 'o', color=('#2e7d32' if ok else '#f9a825'),
                ms=11, mec='black', mew=0.8, zorder=4)
        ax.annotate('G%d' % (k+1), (gx, gy), textcoords='offset points', xytext=(0, -16),
                    ha='center', fontsize=11, color='black', fontproperties=FONT, zorder=5)
    # 标题/指标
    n_ok = sum(1 for a in arrived if a is not None and a <= sim_t)
    ax.set_title('Nav2 自主导航仿真 Demo ｜ 已到达 %d/6 目标' % n_ok,
                 fontproperties=FONT, fontsize=15, color='#1a237e', pad=10)
    ax.text(4.15, 3.9, '仿真时间 %.0fs' % sim_t, fontproperties=FONT, fontsize=11,
            ha='right', color='#333333')
    ax.text(-4.15, 3.9, '成功率 %d%%' % (n_ok/6*100), fontproperties=FONT, fontsize=11,
            ha='left', color='#2e7d32')

import matplotlib.animation as anim
ani = anim.FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000/FPS)
w = anim.PillowWriter(fps=FPS)
out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/nav_anim.mp4'
ani.save(out, writer='ffmpeg', fps=FPS, dpi=100,
         extra_args=['-vcodec', 'libx264', '-preset', 'fast', '-crf', '20'])
print('ANIM_SAVED', out)
