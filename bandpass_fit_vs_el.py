"""
bandpass_fit_vs_el.py
固定 nc=5000，扫描带通中心 El，观察拟合质量随 El 变化的情况。

重点区域：靠近 x=0（贪心第 2 步选取的"最中间"节点）附近，
预期：当 El 落在晚选节点（高序号）附近时拟合误差更大。

使用向量化 Newton 除差（替代纯 Python O(n²) 实现），速度快 ~100x。
对 beta=10 / 50 / 100 三种情况展示。

输出（figs_nodes/）
    bandpass_fit_vs_el_quality.png   — 拟合误差 vs El，横轴标注节点顺序
    bandpass_fit_curves.png          — 选取若干 El 的实际拟合曲线（beta=50）
"""

import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from fft_code.filter_coeff import _samp_points_ashkenazy, _eval_newton_grid

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
NC     = 5000
SMIN, SMAX = -2.0, 2.0
E1     = 0.05          # 带通半宽（缩放坐标）
BETAS  = [10, 50, 100] # 三种陡峭程度

EL_SCAN_LO, EL_SCAN_HI = -0.25, 0.25
N_EL_SCAN = 300
N_EVAL    = 4000
OUT_DIR   = Path("figs_nodes"); OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 向量化 Newton 除差（替代纯 Python _newton_coefficients）
# O(n²) 但全程 numpy，比纯 Python 快 ~100x
# ──────────────────────────────────────────────
def newton_coeffs_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = np.array(y, dtype=float)
    n = len(x)
    for j in range(1, n):
        c[j:] = (c[j:] - c[j-1:n-1]) / (x[j:] - x[:n-j])
    return c

# ──────────────────────────────────────────────
# 1. 生成 nc=5000 节点（记录贪心顺序）
# ──────────────────────────────────────────────
print(f"生成 nc={NC} Ashkenazy 节点...")
nodes = _samp_points_ashkenazy(SMIN, SMAX, NC)   # nodes[i] = 第i步选取的节点

# 按 x 坐标排序（标注用）
sort_idx         = np.argsort(nodes)
nodes_sorted_x   = nodes[sort_idx]
nodes_sorted_step = sort_idx

print(f"  步骤 0: x={nodes[0]:+.5f}")
print(f"  步骤 1: x={nodes[1]:+.5f}")
print(f"  步骤 2: x={nodes[2]:+.5f}")

# ──────────────────────────────────────────────
# 2. 带通函数与拟合
# ──────────────────────────────────────────────
def bandpass_val(x, El, beta):
    EL = El - E1;  ER = El + E1
    return 0.5 * (np.tanh(beta * (x - EL)) - np.tanh(beta * (x - ER)))

s_eval = np.linspace(SMIN, SMAX, N_EVAL)

def fit_error(El, beta):
    f_nodes = bandpass_val(nodes, El, beta)
    an      = newton_coeffs_vec(nodes, f_nodes)
    f_fit   = _eval_newton_grid(nodes, an, s_eval)
    f_true  = bandpass_val(s_eval, El, beta)
    err     = np.abs(f_fit - f_true)
    return err.max(), err.mean(), f_true, f_fit

# ──────────────────────────────────────────────
# 3. 扫描 El 对三种 beta
# ──────────────────────────────────────────────
el_list = np.linspace(EL_SCAN_LO, EL_SCAN_HI, N_EL_SCAN)
results = {}   # beta -> (max_errors, mae_errors)

for beta in BETAS:
    print(f"\n扫描 beta={beta}，{N_EL_SCAN} 个 El 值...")
    max_errs = np.zeros(N_EL_SCAN)
    mae_errs = np.zeros(N_EL_SCAN)
    for i, El in enumerate(el_list):
        max_errs[i], mae_errs[i], _, _ = fit_error(El, beta)
        if i % 60 == 0:
            print(f"  El={El:+.4f}  max_err={max_errs[i]:.2e}")
    results[beta] = (max_errs, mae_errs)

# ──────────────────────────────────────────────
# 4. 扫描区间内的节点（用于横轴标注）
# ──────────────────────────────────────────────
margin    = 0.06
in_range  = (nodes_sorted_x >= EL_SCAN_LO - margin) & \
            (nodes_sorted_x <= EL_SCAN_HI + margin)
annot_x    = nodes_sorted_x[in_range]
annot_step = nodes_sorted_step[in_range]

print(f"\n扫描区间内节点（{in_range.sum()} 个）：")
print(f"  {'步骤':>6}  {'x':>9}")
for s, x in sorted(zip(annot_step, annot_x)):
    print(f"  {s:6d}  {x:+.5f}")

# ──────────────────────────────────────────────
# 图1：三种 beta 的拟合误差 vs El
# ──────────────────────────────────────────────
fig, axes = plt.subplots(len(BETAS), 1, figsize=(14, 4 * len(BETAS)), sharex=True)
cmap_step = plt.cm.RdYlGn_r
step_max  = annot_step.max() if len(annot_step) > 0 else 1

for ax, beta in zip(axes, BETAS):
    max_errs, mae_errs = results[beta]
    ax.semilogy(el_list, max_errs, color="steelblue", lw=1.5, label="max error")
    ax.semilogy(el_list, mae_errs, color="tomato",    lw=1.2, ls="--", alpha=0.8, label="MAE")

    # 节点竖线（颜色 = 步骤序号）
    for x_n, s_n in zip(annot_x, annot_step):
        ax.axvline(x_n, color=cmap_step(s_n / step_max), lw=0.9, alpha=0.55)

    ax.set_ylabel("拟合误差", fontsize=10)
    ax.set_title(f"beta={beta}，E1={E1}，nc={NC}", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

# 上方双坐标轴标注节点步骤序号
ax_top = axes[0].twiny()
ax_top.set_xlim(axes[0].get_xlim())
ax_top.set_xticks(annot_x)
ax_top.set_xticklabels([f"#{s}" for s in annot_step],
                        rotation=90, fontsize=7, ha="center")
ax_top.set_xlabel("节点步骤序号（贪心选取顺序）", fontsize=9)

# 颜色条
sm = plt.cm.ScalarMappable(cmap=cmap_step,
                            norm=plt.Normalize(vmin=0, vmax=step_max))
sm.set_array([])
cb = fig.colorbar(sm, ax=axes.tolist(), pad=0.01, fraction=0.015)
cb.set_label("节点步骤序号", fontsize=9)

axes[-1].set_xlabel("带通中心 El（缩放坐标）", fontsize=11)
fig.suptitle(
    f"Bandpass 拟合误差 vs El（nc={NC}，E1={E1}）\n"
    "竖线 = 插值节点；颜色深红 = 晚选（高步骤序号），绿 = 早选",
    fontsize=11, y=1.005,
)
fig.tight_layout()
path1 = OUT_DIR / "bandpass_fit_vs_el_quality.png"
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n误差图已保存：{path1}")

# ──────────────────────────────────────────────
# 图2：beta=50，选几个典型 El 展示拟合曲线
# ──────────────────────────────────────────────
BETA_CURVES = 50
max_errs_50, _ = results[BETA_CURVES]

# 挑选：误差最小、最大，以及若干峰值
idx_min = int(np.argmin(max_errs_50))
top_idx = list(np.argsort(max_errs_50)[::-1])
showcase = [idx_min]
for ii in top_idx:
    if all(abs(ii - s) > N_EL_SCAN // 15 for s in showcase):
        showcase.append(ii)
    if len(showcase) >= 5:
        break
showcase = sorted(showcase, key=lambda i: el_list[i])

fig2, axes2 = plt.subplots(len(showcase), 1, figsize=(13, 3.5 * len(showcase)))
if len(showcase) == 1:
    axes2 = [axes2]
colors_c = plt.cm.tab10(np.linspace(0, 0.9, len(showcase)))

for ax_c, idx_c, col in zip(axes2, showcase, colors_c):
    El = float(el_list[idx_c])
    me, _, f_true, f_fit = fit_error(El, BETA_CURVES)

    ax_c.plot(s_eval, f_true, "k-",  lw=2.0, label="真实 bandpass", zorder=3)
    ax_c.plot(s_eval, f_fit,  "--",  color=col, lw=1.5, alpha=0.9,
              label=f"Newton 拟合 (nc={NC})", zorder=2)

    # 残差（右轴）
    ax_r = ax_c.twinx()
    ax_r.fill_between(s_eval, 0, f_fit - f_true, alpha=0.2, color="tomato")
    ax_r.set_ylabel("残差", fontsize=8, color="tomato")
    ax_r.tick_params(axis="y", labelcolor="tomato", labelsize=7)

    # 标注节点位置和步骤序号（只标注 El 附近 ±0.15 范围内）
    near = (annot_x >= El - 0.15) & (annot_x <= El + 0.15)
    for x_n, s_n in zip(annot_x[near], annot_step[near]):
        ax_c.axvline(x_n, color=cmap_step(s_n / step_max), lw=0.8, alpha=0.6)
        ylim = ax_c.get_ylim()
        ypos = ylim[0] + 0.88 * (ylim[1] - ylim[0])
        ax_c.text(x_n, ypos, f"#{s_n}", fontsize=7, ha="center",
                  va="top", color="dimgray", rotation=90)

    # 带通边沿标注
    ax_c.axvline(El - E1, color="navy", lw=1.0, ls=":", alpha=0.7, label="EL/ER")
    ax_c.axvline(El + E1, color="navy", lw=1.0, ls=":", alpha=0.7)

    ax_c.set_xlim(El - 0.18, El + 0.18)
    ax_c.set_xlabel("x（缩放坐标）", fontsize=9)
    ax_c.set_title(
        f"El={El:+.5f}  |  max_err={me:.2e}  |  beta={BETA_CURVES}",
        fontsize=10,
    )
    ax_c.legend(fontsize=8, loc="upper right")
    ax_c.grid(True, alpha=0.3)

fig2.suptitle(
    f"Bandpass 拟合曲线（nc={NC}, beta={BETA_CURVES}, E1={E1}）\n"
    "灰竖线 = 插值节点（颜色深 = 步骤序号大），蓝点线 = 带通边沿",
    fontsize=11, y=1.005,
)
fig2.tight_layout()
path2 = OUT_DIR / "bandpass_fit_curves.png"
fig2.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"拟合曲线图已保存：{path2}")

# ──────────────────────────────────────────────
# 终端统计
# ──────────────────────────────────────────────
print("\n===== 各 beta 最大/最小误差 =====")
for beta in BETAS:
    me, _ = results[beta]
    print(f"  beta={beta:3d}  min={me.min():.2e}  max={me.max():.2e}"
          f"  ratio={me.max()/me.min():.1f}")
