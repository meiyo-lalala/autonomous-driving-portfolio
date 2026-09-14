"""
evaluate.py — 在合成验证集上算 BEV 检测指标（recall / 定位误差）。
用法: python evaluate.py --ckpt checkpoints/best.pt --n 200
"""
import argparse, numpy as np, torch
from data.synthetic import generate_scene, scene_to_bev_targets
from models.bev_detector import BEVDetector, decode_boxes


def iou_2d(b1, b2):
    # 俯视 2D IoU（按中心+长宽+yaw≈0 近似）
    x1, y1, l1, w1 = b1['cx'], b1['cy'], b1['l'], b1['w']
    x2, y2, l2, w2 = b2['cx'], b2['cy'], b2['l'], b2['w']
    xa, xb = max(x1-l1/2, x2-l2/2), min(x1+l1/2, x2+l2/2)
    ya, yb = max(y1-w1/2, y2-w2/2), min(y1+w1/2, y2+w2/2)
    inter = max(0, xb-xa) * max(0, yb-ya)
    union = l1*w1 + l2*w2 - inter
    return inter / union if union > 0 else 0


def evaluate(ckpt, n=200, iou_thr=0.5, score_thr=0.4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BEVDetector().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    tp = fp = fn = 0
    loc_err = []
    conf = np.zeros((3, 3), dtype=np.int64)   # 分割混淆矩阵 0背景/1路面/2车辆
    rng = np.random.default_rng(999)
    for _ in range(n):
        pts, gts = generate_scene(rng)
        bev, _, _, gseg = scene_to_bev_targets(pts, gts)
        with torch.no_grad():
            c, r, s = model(torch.from_numpy(bev).unsqueeze(0).to(device))
        preds = decode_boxes(c, r, score_thr=score_thr)
        used = [False]*len(gts)
        for p in preds:
            hit = -1
            for i, g in enumerate(gts):
                if not used[i] and iou_2d(p, g) >= iou_thr:
                    hit = i; break
            if hit >= 0:
                used[hit] = True; tp += 1
                loc_err.append(abs(p['cx']-gts[hit]['cx']) + abs(p['cy']-gts[hit]['cy']))
            else:
                fp += 1
        fn += sum(1 for u in used if not u)
        # 分割：逐格 argmax
        sp = s.squeeze(0).argmax(0).detach().cpu().numpy()
        for a in range(3):
            for b in range(3):
                conf[a, b] += int(((sp == a) & (gseg == b)).sum())
    recall = tp / (tp + fn) if (tp+fn) else 0
    prec = tp / (tp + fp) if (tp+fp) else 0
    ious = []
    for cc in range(3):
        inter = conf[cc, cc]
        union = conf[cc, :].sum() + conf[:, cc].sum() - inter
        ious.append(inter / union if union > 0 else 0)
    miou = float(np.mean(ious))
    print(f'samples={n}  recall={recall:.3f}  precision={prec:.3f}  '
          f'avg_loc_L1={(np.mean(loc_err) if loc_err else 0):.3f} m  (IoU>={iou_thr})')
    print(f'seg mIoU={miou:.3f}  per-class IoU bg/road/veh={[round(x,3) for x in ious]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=200)
    evaluate(**vars(ap.parse_args()))
