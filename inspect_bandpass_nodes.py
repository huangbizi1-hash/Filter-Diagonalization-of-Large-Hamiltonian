"""
inspect_bandpass_nodes.py
可视化 Ashkenazy 贪心算法选取插值节点的顺序与位置。

对 nc = 100, 200, 500, 1000, 3000, 5000 分别展示：
  - 上图：节点 x 值 vs 选取顺序（贪心序列轨迹）
  - 下图：节点位置的分布直方图（密度 vs x）

运行：
    python inspect_bandpass_nodes.py
输出：
    figs_nodes/bandpass_node_order.png
    figs_nodes/bandpass_node_density.png
    figs_nodes/bandpass_node_combined.png  ← 所有 nc 合并图
"""

import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 直接从 fft_code 导入内部函数
sys.path.insert(0, ".")
from fft_code.filter_coeff import _samp_points_ashkenazy

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
NC_LIST  = [100, 200, 500, 1000, 3000, 5000]
SMIN, SMAX = -2.0, 2.0
OUT_DIR  = Path("figs_nodes"); OUT_DIR.mkdir(exist_ok=True)

COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, len(NC_LIST)))

# ──────────────────────────────────────────────
# 计算各 nc 的节点序列
# ──────────────────────────────────────────────
print("计算插值节点...")
nodes_dict = {}
for nc in NC_LIST:
    print(f"  nc={nc:5d} ...", end=" ", flush=True)
    pts = _samp_points_ashkenazy(SMIN, SMAX, nc)
    nodes_dict[nc] = pts
    print(f"  范围 [{pts.min():.4f}, {pts.max():.4f}]，首节点={pts[0]:.4f}")

# ──────────────────────────────────────────────
# 图1：合并图（每个 nc 一行，左 = 序列轨迹，右 = 密度直方图）
# ──────────────────────────────────────────────
n_rows = len(NC_LIST)
fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.8 * n_rows))
fig.suptitle("Ashkenazy 插值节点：选取顺序与分布（bandpass，区间 [-2, 2]）",
             fontsize=13, y=1.002)

for row, (nc, color) in enumerate(zip(NC_LIST, COLORS)):
    pts = nodes_dict[nc]
    order = np.arange(nc)

    # ── 左图：x 值 vs 选取顺序 ──────────────────
    ax_left = axes[row, 0]
    ax_left.scatter(order, pts, s=max(0.5, 6 - nc // 1000),
                    c=[color], alpha=0.7, linewidths=0)
    ax_left.set_xlim(-5, nc + 5)
    ax_left.set_ylim(SMIN - 0.1, SMAX + 0.1)
    ax_left.set_ylabel("节点 x 值", fontsize=9)
    ax_left.set_title(f"nc={nc}  —  节点 x 值 vs 选取顺序", fontsize=10)
    ax_left.axhline(SMIN, color="gray", lw=0.8, ls="--")
    ax_left.axhline(SMAX, color="gray", lw=0.8, ls="--")
    ax_left.grid(True, alpha=0.25)
    if row == n_rows - 1:
        ax_left.set_xlabel("选取顺序（贪心步编号）", fontsize=9)

    # ── 右图：密度直方图 ─────────────────────────
    ax_right = axes[row, 1]
    n_bins = min(200, nc // 2)
    ax_right.hist(pts, bins=n_bins, range=(SMIN, SMAX),
                  color=color, alpha=0.8, edgecolor="none")
    ax_right.set_xlim(SMIN - 0.1, SMAX + 0.1)
    ax_right.set_ylabel("节点计数", fontsize=9)
    ax_right.set_title(f"nc={nc}  —  节点位置分布直方图", fontsize=10)
    ax_right.grid(True, alpha=0.25, axis="y")
    if row == n_rows - 1:
        ax_right.set_xlabel("节点 x 值（缩放坐标）", fontsize=9)

fig.tight_layout()
combined_path = OUT_DIR / "bandpass_node_combined.png"
fig.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n合并图已保存：{combined_path}")

# ──────────────────────────────────────────────
# 图2：前 200 步的放大轨迹（揭示贪心细节）
# ──────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8))
fig2.suptitle("前 200 步贪心选取轨迹（x 值 vs 选取顺序）", fontsize=13)

for ax, nc, color in zip(axes2.flat, NC_LIST, COLORS):
    pts = nodes_dict[nc]
    n_show = min(200, nc)
    order = np.arange(n_show)
    ax.plot(order, pts[:n_show], "o-", color=color,
            markersize=4, lw=0.8, alpha=0.85)
    ax.set_xlim(-2, n_show + 2)
    ax.set_ylim(SMIN - 0.15, SMAX + 0.15)
    ax.axhline(SMIN, color="gray", lw=0.8, ls="--")
    ax.axhline(SMAX, color="gray", lw=0.8, ls="--")
    ax.set_title(f"nc={nc}", fontsize=10)
    ax.set_xlabel("选取顺序", fontsize=9)
    ax.set_ylabel("节点 x 值", fontsize=9)
    ax.grid(True, alpha=0.25)

    # 标注前 10 步编号
    for i in range(min(10, n_show)):
        ax.annotate(str(i), (i, pts[i]),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=7, color="black")

fig2.tight_layout()
order_path = OUT_DIR / "bandpass_node_order_zoom.png"
fig2.savefig(order_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"前200步放大图已保存：{order_path}")

# ──────────────────────────────────────────────
# 图3：所有 nc 覆盖图（同一坐标轴，看 nc 增大时的变化）
# ──────────────────────────────────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("不同 nc 下 Ashkenazy 节点序列对比", fontsize=12)

for nc, color in zip(NC_LIST, COLORS):
    pts = nodes_dict[nc]
    order_norm = np.arange(nc) / nc   # 归一化到 [0,1]

    # 左：归一化顺序 vs x 值
    ax3a.scatter(order_norm, pts, s=0.5, c=[color], alpha=0.4, label=f"nc={nc}")

    # 右：累积分布（节点排序后的经验 CDF）
    pts_sorted = np.sort(pts)
    cdf = (np.arange(nc) + 0.5) / nc
    ax3b.plot(pts_sorted, cdf, color=color, lw=1.2, alpha=0.8, label=f"nc={nc}")

# 对比 Chebyshev 理论 CDF（均匀弧密度：CDF = (1/π)·arccos(-x)）
x_ref = np.linspace(SMIN + 0.01, SMAX - 0.01, 500)
cdf_cheb = np.arccos(-x_ref) / np.pi
ax3b.plot(x_ref, cdf_cheb, "k--", lw=1.5, label="Chebyshev 理论 CDF")

ax3a.axhline(SMIN, color="gray", lw=0.8, ls="--")
ax3a.axhline(SMAX, color="gray", lw=0.8, ls="--")
ax3a.set_xlabel("选取顺序（归一化 0→1）", fontsize=10)
ax3a.set_ylabel("节点 x 值", fontsize=10)
ax3a.set_title("贪心序列轨迹（归一化顺序）", fontsize=10)
ax3a.legend(fontsize=8, markerscale=6)
ax3a.grid(True, alpha=0.25)

ax3b.set_xlabel("节点 x 值", fontsize=10)
ax3b.set_ylabel("经验 CDF", fontsize=10)
ax3b.set_title("节点分布 vs Chebyshev 理论 CDF", fontsize=10)
ax3b.legend(fontsize=8)
ax3b.grid(True, alpha=0.25)

fig3.tight_layout()
compare_path = OUT_DIR / "bandpass_node_compare.png"
fig3.savefig(compare_path, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"对比图已保存：{compare_path}")

# ──────────────────────────────────────────────
# 终端打印前 30 步选取序列（便于直接观察）
# ──────────────────────────────────────────────
print("\n===== 前 30 步选取序列（x 值）=====")
print(f"{'步骤':>4}  " + "  ".join(f"nc={nc:5d}" for nc in NC_LIST))
print("-" * (6 + 12 * len(NC_LIST)))
for i in range(30):
    row_vals = "  ".join(f"{nodes_dict[nc][i]:+.5f}" for nc in NC_LIST)
    print(f"{i:4d}  {row_vals}")
