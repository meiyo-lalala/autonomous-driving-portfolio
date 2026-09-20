# -*- coding: utf-8 -*-
"""
03_lane_detection —— 基于 PyTorch 的车道线检测 Demo

小型 U-Net（约 6 万参数）在合成车道数据集上训练，
输出: output/lane_result.png（损失曲线 + 4 组推理对比）、output/lane_detector.pt（模型权重）

说明: 本演示使用程序生成的合成车道图（路面纹理 + 二次曲线车道线），
用于展示"图像 → 语义分割 → 车道线掩码"的完整深度学习管线；
真实车载场景可无缝替换为 CULane/TuSimple 等公开数据集（见 README）。
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 合成车道数据集 ----------
RNG = np.random.default_rng(2026)


def make_lane_image(h=64, w=96, seed=None):
    """生成一张合成车道图 + 二值车道线掩码。"""
    r = np.random.default_rng(seed)
    # 路面：低频噪声纹理（模拟沥青）+ 中间偏亮的行车带
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    road = r.normal(0.5, 0.06, (h, w)).astype(np.float32)
    road += 0.08 * np.sin(xx / 9.0 + r.uniform(0, 6)) * np.cos(yy / 7.0 + r.uniform(0, 6))
    road = np.clip(road, 0, 1)
    # 本车道左右两条车道线（二次曲线），随机曲率与横向偏移
    base = w / 2.0 + r.uniform(-w * 0.18, w * 0.18)
    curv = r.uniform(-0.0022, 0.0022)          # 曲率
    lane_off = r.uniform(w * 0.10, w * 0.16)   # 车道半宽
    img = road.copy()
    mask = np.zeros((h, w), dtype=np.float32)
    for side in (-1.0, 1.0):
        for i in range(h):                     # 逐行画车道线（二次曲线）
            xpos = base + side * lane_off + curv * (i - h) ** 2
            xi = int(round(xpos))
            for dx in (-1, 0, 1):              # 线宽 3px
                if 0 <= xi + dx < w:
                    img[i, xi + dx] = 0.95
                    mask[i, xi + dx] = 1.0
    # 随机加减速摄像头抖动（整体平移）
    shift = int(r.uniform(-3, 4))
    if shift != 0:
        img = np.roll(img, shift, axis=1)
        mask = np.roll(mask, shift, axis=1)
    return img, mask


class LaneDataset(Dataset):
    def __init__(self, n=300, h=64, w=96, base_seed=1000):
        self.items = [(make_lane_image(h, w, seed=base_seed + i)) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, mask = self.items[idx]
        t = torch.from_numpy(img).unsqueeze(0)      # 1xHxW
        m = torch.from_numpy(mask).unsqueeze(0)
        return t, m


# ---------- 2. 小型 U-Net ----------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, cin, cout, skip):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.block = nn.Sequential(
            nn.Conv2d(cout + skip, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], dim=1)
        return self.block(x)


class TinyUNet(nn.Module):
    """1->16->32->64 编码 / 64->32->16 解码的紧凑 U-Net（~6 万参数）。
    跳连使用池化前的同分辨率特征。"""

    def __init__(self):
        super().__init__()
        self.d1 = ConvBlock(1, 16)
        self.d2 = ConvBlock(16, 32)
        self.d3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.u1 = Up(64, 32, 64)
        self.u2 = Up(32, 16, 32)
        self.u3 = Up(16, 8, 16)
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        s1 = self.d1(x)          # 64x96
        s2 = self.d2(self.pool(s1))   # 32x48
        s3 = self.d3(self.pool(s2))   # 16x24
        m = self.mid(self.pool(s3))   # 8x12
        m = self.u1(m, s3)       # ->16x24
        m = self.u2(m, s2)       # ->32x48
        m = self.u3(m, s1)       # ->64x96
        return self.head(m)


def dice_loss(pred, target, eps=1e-6):
    p = torch.sigmoid(pred)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


# ---------- 3. 训练 ----------
def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    H, W = 64, 96
    train = LaneDataset(n=280, h=H, w=W, base_seed=1000)
    val = LaneDataset(n=60, h=H, w=W, base_seed=5000)
    tl = DataLoader(train, batch_size=32, shuffle=True)
    vl = DataLoader(val, batch_size=32)

    model = TinyUNet().to(device)
    print("参数量: %d" % sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
    epochs = 60
    hist = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        tl_ = 0.0
        for x, y in tl:
            opt.zero_grad()
            out = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(out, y) + 0.5 * dice_loss(out, y)
            loss.backward()
            opt.step()
            tl_ += loss.item() * len(x)
        sched.step()
        model.eval()
        vl_, iou_ = 0.0, 0.0
        with torch.no_grad():
            for x, y in vl:
                out = model(x)
                loss = nn.functional.binary_cross_entropy_with_logits(out, y)
                vl_ += loss.item() * len(x)
                p = (torch.sigmoid(out) > 0.5).float()
                inter = (p * y).sum(dim=(1, 2, 3))
                union = (p + y).clamp(0, 1).sum(dim=(1, 2, 3))
                iou_ += (inter / (union + 1e-6)).sum().item()
        hist["train"].append(tl_ / len(train))
        hist["val"].append(vl_ / len(val))
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:3d}  train {hist['train'][-1]:.4f}  val {hist['val'][-1]:.4f}  valIoU {iou_/len(val):.3f}")

    os.makedirs("output", exist_ok=True)
    torch.save(model.state_dict(), "output/lane_detector.pt")
    print("模型已保存: output/lane_detector.pt")

    # ---------- 4. 可视化：损失曲线 + 4 组推理对比 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    ax.plot(hist["train"], label="Train loss", color="#1a73e8")
    ax.plot(hist["val"], label="Val loss", color="#ff9800")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("U-Net 训练曲线 (PyTorch, CPU)")
    ax.legend()
    ax.grid(alpha=0.3)

    model.eval()
    samples = [val[i] for i in range(4)]
    with torch.no_grad():
        preds = [torch.sigmoid(model(s[0].unsqueeze(0)))[0, 0].numpy() for s in samples]
    ax = axes[1]
    ax.axis("off")
    for i in range(4):
        img, mask = samples[i]
        row, col = i // 2, i % 2
        bx = col * 2.0
        by = row * 1.5
        # 左半格: 原图 + 真值(绿)
        ax.imshow(img[0].numpy(), cmap="gray", extent=[bx, bx + 1.0, by + 1.0, by])
        ax.imshow(np.ma.masked_where(mask[0].numpy() < 0.5, mask[0].numpy()), cmap="Greens",
                  extent=[bx + 0.04, bx + 0.96, by + 0.96, by + 0.04], alpha=0.9)
        # 右半格: 原图 + 预测(红)
        ax.imshow(img[0].numpy(), cmap="gray", extent=[bx + 1.0, bx + 2.0, by + 1.0, by])
        ax.imshow(np.ma.masked_where(preds[i] < 0.5, preds[i]), cmap="Reds",
                  extent=[bx + 1.04, bx + 1.96, by + 0.96, by + 0.04], alpha=0.9)
        if i < 2:
            ax.text(bx + 0.5, by + 1.12, "真值(GT)", ha="center", fontsize=10)
            ax.text(bx + 1.5, by + 1.12, "预测", ha="center", fontsize=10)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(3.1, -0.1)
    ax.set_title("推理对比: 左=原图+真值(绿)  右=原图+预测(红)")

    fig.tight_layout()
    fig.savefig("output/lane_result.png", dpi=140, bbox_inches="tight")
    print("图片已保存: output/lane_result.png")


if __name__ == "__main__":
    main()
