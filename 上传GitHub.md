# 一键上传 GitHub 说明（智能驾驶算法作品集）

本目录就是仓库根（含 `slam_fusion/`、`pytorch_3d_bev/`、`portfolio/`）。
`portfolio/index.html` 用相对路径引用两个项目的产出图，因此**仓库根必须是本 `projects/` 目录**，
不要到子目录里单独 init。

> 全部命令在 **Git Bash**（Windows）或终端里粘贴执行即可。
> 把下面 `你的用户名` 和 `仓库名` 换成你自己的。

---

## 0. 前置（只做一次）

1. 安装 Git：https://git-scm.com/downloads
2. 在 GitHub 网页上 **New repository**，仓库名例如 `ad-perception-portfolio`，
   **不要**勾选自动生成 README/.gitignore（我们已经有了）。
3. 配好本地身份（首次）：
   ```bash
   git config --global user.name  "你的名字"
   git config --global user.email "你的邮箱"
   ```

---

## 1. 初始化并提交（在本 projects/ 目录下执行）

```bash
# 进入仓库根
cd /c/Users/ZQQ/WorkBuddy/求职/projects

# 初始化
git init
git add .
git commit -m "feat: 手写 BEV 多任务感知(检测+分割) + SLAM 融合定位 作品集"
```

> `.gitignore` 已配好：日志(`*.log`)、`__pycache__`、虚拟环境、旧权重(`best.pt/best_v2.pt/final.pt`)
> 都会被忽略，只保留最终多任务权重 `checkpoints/seg.pt` 与可视化图。

---

## 2. 关联远程并推送

```bash
git branch -M main
git remote add origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

推送后，仓库地址就是：`https://github.com/你的用户名/仓库名`

---

## 3. （可选）让作品集页面在线可看

GitHub 不能直接渲染本地相对路径的 HTML 图片，但有两种零成本方式：

**方式 A — 直接看源码里的图**：在仓库里打开
`pytorch_3d_bev/demo_bev.png`、`slam_fusion/assets/trajectory.png`、`slam_fusion/assets/bev_map.png`
即可，简历里贴这几个图片链接。

**方式 B — GitHub Pages（推荐，最像作品集）**：
1. 仓库 → Settings → Pages → Source 选 `main` 分支 `/root`；
2. 等 1 分钟，访问 `https://你的用户名.github.io/仓库名/portfolio/index.html`
   （`portfolio/index.html` 的相对路径引用在 Pages 下同样有效）。

---

## 4. 别人 clone 后怎么跑（写进简历/README 很加分）

```bash
# ① 融合定位（无需 torch）
pip install numpy matplotlib
python slam_fusion/run.py          # 产出 assets/trajectory.png, assets/bev_map.png

# ② BEV 多任务感知（检测 + 语义分割）
pip install torch numpy matplotlib --index-url https://download.pytorch.org/whl/cpu
cd pytorch_3d_bev
python train.py --epochs 15 --samples 1500 --save checkpoints/seg.pt
python evaluate.py --ckpt checkpoints/seg.pt --n 200
python demo.py --ckpt checkpoints/seg.pt --out demo_bev.png
```

---

## 5. 更新已发布内容

改完代码/图后：
```bash
git add .
git commit -m "update: ..."
git push
```

---

### 备注（面试可讲）
- 两个项目**零外部感知/SLAM 框架依赖**（不依赖 mmdet3d / OpenPCDet / Open3D / ROS），
  全部手写，体现底层实现能力；
- 数据用合成数据跑通，但「数据采样」与「核心算法」解耦，换 KITTI/nuScenes `.bin/.pcd`
  后网络与训练一行不改；
- 所有指标来自真实运行输出、固定随机种子可复现。
