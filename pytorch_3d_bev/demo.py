"""
demo.py — 推理一帧并画 BEV 检测 + BEV 语义分割可视化，保存 demo_bev.png（用于作品集）。
用法: python demo.py --ckpt checkpoints/final.pt --out demo_bev.png
"""
import argparse, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.synthetic import generate_scene, scene_to_bev_targets, X_MIN, X_MAX, Y_MIN, Y_MAX, RES, W, H
from models.bev_detector import BEVDetector, decode_boxes

# 分割配色：0=背景(深灰) 1=路面(蓝) 2=车辆(红)
SEG_CMAP = {0: (0.15, 0.15, 0.15), 1: (0.20, 0.50, 0.90), 2: (0.90, 0.20, 0.20)}


def seg_rgb(seg):
    h, w = seg.shape
    rgb = np.zeros((h, w, 3))
    for k, col in SEG_CMAP.items():
        rgb[seg == k] = col
    return rgb


def draw(bev, gts, gseg, preds, spred, out):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    # 左：BEV 伪图像
    ax[0].imshow(bev[0], origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], cmap='viridis')
    ax[0].set_title('BEV pseudo-image (max height)'); ax[0].set_xlabel('x (m)'); ax[0].set_ylabel('y (m)')
    # 中：3D 检测结果
    ax[1].imshow(bev[0], origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], cmap='gray', alpha=.5)
    for g in gts:
        rect = plt.Rectangle((g['cx']-g['l']/2, g['cy']-g['w']/2), g['l'], g['w'],
                             fill=False, edgecolor='g', lw=1.5, label='GT' if g is gts[0] else "")
        ax[1].add_patch(rect)
    for p in preds:
        rect = plt.Rectangle((p['cx']-p['l']/2, p['cy']-p['w']/2), p['l'], p['w'],
                             fill=False, edgecolor='r', lw=1.2, label='Pred' if p is preds[0] else "")
        ax[1].add_patch(rect)
    ax[1].set_title(f'3D Detection  (GT={len(gts)}, Pred={len(preds)})')
    ax[1].legend(loc='upper right'); ax[1].set_xlabel('x (m)'); ax[1].set_ylabel('y (m)')
    # 右：BEV 语义分割（GT 实色 + 预测叠加）
    ax[2].imshow(seg_rgb(gseg), origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax[2].imshow(seg_rgb(spred), origin='lower', extent=[X_MIN, X_MAX, Y_MIN, Y_MAX], alpha=0.55)
    ax[2].set_title('BEV Segmentation  (GT solid / Pred overlay)')
    ax[2].set_xlabel('x (m)'); ax[2].set_ylabel('y (m)')
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print('saved', out, '| GT', len(gts), 'Pred', len(preds))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='demo_bev.png')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = BEVDetector().to(device); m.load_state_dict(torch.load(args.ckpt, map_location=device)); m.eval()
    rng = np.random.default_rng(args.seed)
    pts, gts = generate_scene(rng)
    bev, _, _, gseg = scene_to_bev_targets(pts, gts)
    with torch.no_grad():
        c, r, s = m(torch.from_numpy(bev).unsqueeze(0).to(device))
    preds = decode_boxes(c, r, score_thr=0.4)
    spred = s.squeeze(0).argmax(0).detach().cpu().numpy()
    draw(bev, gts, gseg, preds, spred, args.out)
