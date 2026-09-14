"""
bev_detector.py — 手搓的 BEV 单阶段 3D 检测器（CenterPoint/BirdNet 简化版）
-----------------------------------------------------------------------
流程：点云 -> BEV 伪图像 (C,H,W) -> CNN 主干 -> 两个检测头
  - 分类头：每格 objectness（是否有物体中心）
  - 回归头：每格预测 (dx,dy,dz, log l, log w, yaw)
纯 PyTorch 实现，无 mmdet3d / OpenPCDet 依赖。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.synthetic import X_MIN, Y_MIN, RES, W, H


class ConvBnReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class BEVBackbone(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.stem = ConvBnReLU(in_ch, base)
        self.block1 = ConvBnReLU(base, base)
        self.block2 = ConvBnReLU(base, base * 2, s=2)   # /2
        self.block3 = ConvBnReLU(base * 2, base * 2)
        self.block4 = ConvBnReLU(base * 2, base * 4, s=2)  # /4
        self.block5 = ConvBnReLU(base * 4, base * 4)
        # 上采样回原分辨率，做多尺度融合
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.fuse = ConvBnReLU(base * 2, base)

    def forward(self, x):
        s0 = self.block1(self.stem(x))           # (B,32,H,W)
        s1 = self.block3(self.block2(s0))         # (B,64,H/2,W/2)
        s2 = self.block5(self.block4(s1))         # (B,128,H/4,W/4)
        u1 = self.up1(s2)                         # (B,64,H/2,W/2)
        u1 = u1 + s1
        u2 = self.up2(u1)                         # (B,32,H,W)
        out = self.fuse(torch.cat([u2, s0], dim=1))
        return out


class BEVDetector(nn.Module):
    def __init__(self, in_ch=3, base=32, num_seg=3):
        super().__init__()
        self.backbone = BEVBackbone(in_ch, base)
        self.cls_head = nn.Conv2d(base, 1, 1)           # objectness logits
        self.reg_head = nn.Conv2d(base, 6, 1)           # dx,dy,dz,log l,log w,yaw
        self.seg_head = nn.Conv2d(base, num_seg, 1)     # BEV 语义分割（背景/路面/车辆）

    def forward(self, bev):
        f = self.backbone(bev)
        cls = self.cls_head(f)                  # (B,1,H,W)
        reg = self.reg_head(f)                  # (B,6,H,W)
        seg = self.seg_head(f)                 # (B,num_seg,H,W)
        return cls, reg, seg


# ----------------------------- 损失函数 -----------------------------
def focal_loss(pred_logits, target, alpha=0.5, gamma=2.0):
    """对极度不平衡的 BEV 中心栅格做 focal loss（兼容高斯软目标）。
    正样本权重正比于 target，负样本正比于 (1-target)，平滑处理热图衰减区。"""
    pred = pred_logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    pt = torch.where(target > 0.5, pred, 1 - pred)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** gamma
    return (w * ce).mean()


def reg_loss(pred_reg, target_reg, target_cls, smooth=1.0):
    """在整张高斯热图 blob 上算 smooth-L1（target_cls>0.1 即视为有车区域），
    让回归头拿到足量梯度、学到真实车长宽，而非只在中心单格监督。"""
    mask = (target_cls > 0.1).unsqueeze(1).float()  # (B,1,H,W)
    diff = torch.abs(pred_reg - target_reg) * mask
    loss = torch.where(diff < smooth, 0.5 * diff ** 2 / smooth, diff - 0.5 * smooth)
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def seg_loss(pred_seg, target_seg, weight=None):
    """BEV 语义分割交叉熵（背景格占多数，按类加权避免被背景淹没）。"""
    if weight is not None:
        weight = torch.as_tensor(weight, dtype=torch.float32, device=pred_seg.device)
    return F.cross_entropy(pred_seg, target_seg.long(), weight=weight)


def decode_boxes(cls_logits, reg, score_thr=0.4):
    """
    把网络输出解码成 3D 框列表。返回 list[ dict(cx,cy,cz,l,w,h,yaw,score) ]。
    先做 3x3 局部峰值抑制（高斯热图下每车只取一个峰），再按阈值出框，
    最后做基于中心距离的非极大抑制（同车多个峰只留最高分），避免一圈重复框。
    """
    scores = cls_logits.sigmoid().squeeze(0).squeeze(0).detach().cpu().numpy()
    reg = reg.squeeze(0).detach().cpu().numpy()
    # 局部峰值抑制：与 3x3 最大池化结果"接近"即视为峰（吸收浮点并列峰）
    pooled = torch.nn.functional.max_pool2d(cls_logits.detach(), 3, 1, 1)
    is_peak = (cls_logits.detach() >= pooled - 1e-4).squeeze(0).squeeze(0).cpu().numpy()
    ys, xs = np.where(is_peak & (scores > score_thr))
    # 收集候选
    cands = []
    for gy, gx in zip(ys, xs):
        dx, dy, dz, logl, logw, yaw = reg[:, gy, gx]
        cx = X_MIN + (gx + 0.5) * RES + dx
        cy = Y_MIN + (gy + 0.5) * RES + dy
        cands.append(dict(cx=float(cx), cy=float(cy), cz=float(dz),
                          l=float(np.exp(logl)), w=float(np.exp(logw)), h=1.5,
                          yaw=float(yaw), score=float(scores[gy, gx])))
    # 中心距离 NMS：按分数降序，抑制 2.0m 内的低分框（BEV 车宽~2m）
    cands.sort(key=lambda b: b['score'], reverse=True)
    keep = []
    for b in cands:
        if all(((b['cx'] - k['cx']) ** 2 + (b['cy'] - k['cy']) ** 2) > 2.0 ** 2 for k in keep):
            keep.append(b)
    return keep


if __name__ == '__main__':
    m = BEVDetector()
    bev = torch.randn(2, 3, H, W)
    c, r, s = m(bev)
    print('cls:', c.shape, 'reg:', r.shape, 'seg:', s.shape)
    print('params:', sum(p.numel() for p in m.parameters()) / 1e3, 'k')
