"""
train.py — 训练 BEV 3D 检测器（合成数据，无需下载 KITTI）。
用法: python train.py --epochs 12 --batch 8 --save checkpoints/best.pt
"""
import os
# 限制线程数，避免 CPU 训练时内存峰值触发沙箱 OOM 被强杀
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import argparse, numpy as np, torch
torch.set_num_threads(2)
from torch.utils.data import Dataset, DataLoader
from data.synthetic import generate_scene, scene_to_bev_targets, H, W
from models.bev_detector import BEVDetector, focal_loss, reg_loss, seg_loss


class SynthDataset(Dataset):
    def __init__(self, n=2000, seed=123):
        self.n = n
        self.rng = np.random.default_rng(seed)
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        pts, boxes = generate_scene(self.rng)
        bev, cls_t, reg_t, seg_t = scene_to_bev_targets(pts, boxes)
        return (torch.from_numpy(bev),
                torch.from_numpy(cls_t).unsqueeze(0),
                torch.from_numpy(reg_t),
                torch.from_numpy(seg_t).long())


def collate_fn(batch):
    bev, cls, reg, seg = zip(*batch)
    return torch.stack(bev), torch.stack(cls), torch.stack(reg), torch.stack(seg)


def train(args):
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = SynthDataset(n=args.samples)
    dl = DataLoader(ds, batch_size=args.batch, collate_fn=collate_fn, num_workers=0)
    model = BEVDetector().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(dl))
    # 背景格占多数：降低其权重，让道路/车辆分割可学习
    seg_w = [0.2, 0.8, 2.0]

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        tot = 0
        for bev, clst, regt, segt in dl:
            bev = bev.to(device)
            clst, regt, segt = clst.to(device), regt.to(device), segt.to(device)
            c, r, s = model(bev)
            loss = (focal_loss(c, clst)
                    + 3.0 * reg_loss(r, regt, clst)
                    + 0.3 * seg_loss(s, segt, weight=seg_w))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        avg = tot / len(dl)
        print(f'[epoch {ep+1}/{args.epochs}] loss={avg:.4f}')
        if avg < best:
            best = avg
            torch.save(model.state_dict(), args.save)
            print('  -> saved best')
    print('done. best loss', round(best, 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--samples', type=int, default=1500)
    ap.add_argument('--save', type=str, default='checkpoints/best.pt')
    train(ap.parse_args())
