"""
bandpass_node_sensitivity.py
固定 nc=500，研究 bandpass 拟合对最近插值节点位置的敏感性。

参数：El=-0.12，E1=0.05，beta=50
操作：找到最靠近 El 的 Ashkenazy 节点，
      将其 x 坐标平移 delta（正方向和负方向），
      观察拟合曲线和误差如何变化。

输出（figs_nodes/）
    node_sensitivity_curves.png   — 不同 delta 的拟合曲线
    node_sensitivity_error.png    — max_error vs delta（半对数）
"""

import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

sys.path.insert(0, ".")
from fft_code.filter_coeff import _samp_points_ashkenazy, _eval_newton_grid

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
NC    = 500
SMIN, SMAX = -2.0, 2.0
EL    = -0.12
E1    = 0.05
BETA  = 50

# 平移量（正方向）：从几乎为0到接近节点间距的一倍以上
DELTAS_POS = np.array([0.0, 0.0005, 0.001, 0.002, 0.004, 0.007,
                        0.01, 0.015, 0.02, 0.03, 0.05])
# 负方向（对称）
DELTAS_NEG = -DELTAS_POS[1:]   # 跳过 delta=0

N_EVAL = 6000
OUT_DIR = Path("figs_nodes"); OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 向量化 Newton 除差
# ──────────────────────────────────────────────
def newton_coeffs_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    c = np.array(y, dtype=float)
    n = len(x)
    for j in range(1, n):
        c[j:] = (c[j:] - c[j-1:n-1]) / (x[j:] - x[:n-j])
    return c

# ──────────────────────────────────────────────
# 1. 生成节点（原始顺序 = 贪心顺序）
# ──────────────────────────────────────────────
print(f"生成 nc={NC} Ashkenazy 节点...")
nodes_orig = _samp_points_ashkenazy(SMIN, SMAX, NC)
# nodes_orig[i] = 贪心第 i 步选取的节点

# 找最靠近 El 的节点（在原始贪心顺序中的索引）
k_near = int(np.argmin(np.abs(nodes_orig - EL)))
print(f"最靠近 El={EL} 的节点：")
print(f"  贪心步骤序号 = #{k_near}")
print(f"  x 值        = {nodes_orig[k_near]:+.6f}")
print(f"  距离        = {abs(nodes_orig[k_near] - EL):.6f}")

# 排序索引（用于标注横轴）
sort_idx          = np.argsort(nodes_orig)
nodes_sorted_x    = nodes_orig[sort_idx]
nodes_sorted_step = sort_idx   # step[i] = 第 i 个排序节点的贪心序号

# 找扫描区域 [EL-0.15, EL+0.15] 内的节点
annot_margin = 0.15
in_range  = (nodes_sorted_x >= EL - annot_margin) & \
            (nodes_sorted_x <= EL + annot_margin)
annot_x    = nodes_sorted_x[in_range]
annot_step = nodes_sorted_step[in_range]

print(f"\nEl 附近节点（±{annot_margin}）：")
print(f"  {'步骤':>5}  {'x':>9}")
for s, x in sorted(zip(annot_step, annot_x)):
    marker = " <-- 被扰动" if s == k_near else ""
    print(f"  #{s:4d}  {x:+.6f}{marker}")

# ──────────────────────────────────────────────
# 2. 带通函数
# ──────────────────────────────────────────────
def bandpass_val(x, El=EL):
    EL_edge = El - E1;  ER_edge = El + E1
    return 0.5 * (np.tanh(BETA * (x - EL_edge)) - np.tanh(BETA * (x - ER_edge)))

s_eval  = np.linspace(SMIN, SMAX, N_EVAL)
f_true  = bandpass_val(s_eval)

# ──────────────────────────────────────────────
# 3. 拟合函数（扰动节点 k_near 的 x 值）
# ──────────────────────────────────────────────
def fit_with_delta(delta: float):
    """平移节点 k_near，重新拟合，返回 (max_err, mae, f_fit)。"""
    nodes = nodes_orig.copy()
    nodes[k_near] += delta
    f_nodes = bandpass_val(nodes)
    an      = newton_coeffs_vec(nodes, f_nodes)
    f_fit   = _eval_newton_grid(nodes, an, s_eval)
    err     = np.abs(f_fit - f_true)
    return err.max(), err.mean(), f_fit

# ──────────────────────────────────────────────
# 4. 计算所有 delta
# ──────────────────────────────────────────────
all_deltas = np.concatenate([np.sort(DELTAS_NEG), DELTAS_POS])

delta_vals  = []
max_errors  = []
mae_errors  = []
fit_curves  = {}   # delta -> f_fit （只存部分）

# 选择要画曲线的 delta（正方向，含 0）
deltas_to_plot = DELTAS_POS   # 正方向全部画

print(f"\n计算 {len(all_deltas)} 个 delta 值...")
for delta in all_deltas:
    me, mae, f_fit = fit_with_delta(delta)
    delta_vals.append(delta)
    max_errors.append(me)
    mae_errors.append(mae)
    if delta in deltas_to_plot:
        fit_curves[delta] = f_fit.copy()
    status = f"max_err={me:.2e}"
    if me > 1.0:
        status += "  *** 爆炸 ***"
    print(f"  delta={delta:+.5f}  {status}")

delta_vals = np.array(delta_vals)
max_errors = np.array(max_errors)
mae_errors = np.array(mae_errors)

# ──────────────────────────────────────────────
# 颜色映射（delta=0 蓝，delta 大 → 红）
# ──────────────────────────────────────────────
n_plot   = len(deltas_to_plot)
colors_c = cm.plasma(np.linspace(0.05, 0.92, n_plot))

# ──────────────────────────────────────────────
# 图1：拟合曲线（正方向 delta）
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                          gridspec_kw={"height_ratios": [2, 1]})

ax_main  = axes[0]
ax_err   = axes[1]

# ── 拟合曲线 ──
ax_main.plot(s_eval, f_true, "k-", lw=2.5, zorder=5, label="真实 bandpass")

for delta, col in zip(deltas_to_plot, colors_c):
    f_fit = fit_curves[delta]
    lw    = 2.0 if delta == 0.0 else 1.3
    ls    = "-" if delta == 0.0 else "--"
    label = f"δ={delta:.4f}"
    ax_main.plot(s_eval, f_fit, ls, color=col, lw=lw, alpha=0.85, label=label)

# 带通边沿
ax_main.axvline(EL - E1, color="navy", lw=1.0, ls=":", alpha=0.6)
ax_main.axvline(EL + E1, color="navy", lw=1.0, ls=":", alpha=0.6)
ax_main.axvline(EL,      color="navy", lw=0.8, ls="-", alpha=0.3)

# 标注 El 附近节点竖线 + 步骤序号
for x_n, s_n in zip(annot_x, annot_step):
    lw_n    = 1.8 if s_n == k_near else 0.8
    col_n   = "red" if s_n == k_near else "gray"
    alpha_n = 0.8 if s_n == k_near else 0.4
    ax_main.axvline(x_n, color=col_n, lw=lw_n, alpha=alpha_n, zorder=2)
    ax_main.text(x_n, 1.08, f"#{s_n}", fontsize=7, ha="center", va="bottom",
                 color="red" if s_n == k_near else "dimgray",
                 fontweight="bold" if s_n == k_near else "normal",
                 transform=ax_main.get_xaxis_transform())

ax_main.set_xlim(EL - 0.18, EL + 0.18)
ax_main.set_ylim(-0.15, 1.25)
ax_main.set_ylabel("f(x)", fontsize=11)
ax_main.set_title(
    f"Bandpass 拟合曲线  (nc={NC}, El={EL}, E1={E1}, beta={BETA})\n"
    f"红竖线/红标 = 被扰动的节点（步骤 #{k_near}，x={nodes_orig[k_near]:+.6f}）",
    fontsize=11,
)
ax_main.legend(fontsize=8, ncol=4, loc="upper right",
               bbox_to_anchor=(1.0, 1.0))
ax_main.grid(True, alpha=0.25)

# ── 残差（|f_fit - f_true|）──
for delta, col in zip(deltas_to_plot, colors_c):
    f_fit = fit_curves[delta]
    ax_err.semilogy(s_eval, np.abs(f_fit - f_true) + 1e-17,
                    color=col, lw=1.1 if delta > 0 else 2.0,
                    alpha=0.85, label=f"δ={delta:.4f}")

for x_n, s_n in zip(annot_x, annot_step):
    ax_err.axvline(x_n, color="red" if s_n == k_near else "gray",
                   lw=1.8 if s_n == k_near else 0.7,
                   alpha=0.7 if s_n == k_near else 0.3)

ax_err.axvline(EL - E1, color="navy", lw=1.0, ls=":", alpha=0.5)
ax_err.axvline(EL + E1, color="navy", lw=1.0, ls=":", alpha=0.5)
ax_err.set_xlim(EL - 0.18, EL + 0.18)
ax_err.set_ylabel("|残差|", fontsize=10)
ax_err.set_xlabel("x（缩放坐标）", fontsize=11)
ax_err.grid(True, which="both", alpha=0.25)
ax_err.set_title("点态残差 |f_fit - f_true|（半对数）", fontsize=10)
ax_err.legend(fontsize=7, ncol=4, loc="upper right")

fig.tight_layout()
path1 = OUT_DIR / "node_sensitivity_curves.png"
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n曲线图已保存：{path1}")

# ──────────────────────────────────────────────
# 图2：max_error vs delta（正负方向）
# ──────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

# 正方向
pos_mask = delta_vals >= 0
neg_mask = delta_vals < 0
ax2.semilogy(delta_vals[pos_mask], max_errors[pos_mask],
             "o-", color="steelblue", lw=1.8, ms=6, label="delta > 0（正方向）")
ax2.semilogy(-delta_vals[neg_mask], max_errors[neg_mask],
             "s--", color="tomato", lw=1.5, ms=5, alpha=0.8, label="|delta|（负方向）")

# 标出相邻节点间距（近似）
sort_x = np.sort(nodes_orig)
k_sorted = np.searchsorted(sort_x, nodes_orig[k_near])
if 0 < k_sorted < NC - 1:
    spacing_l = sort_x[k_sorted] - sort_x[k_sorted - 1]
    spacing_r = sort_x[k_sorted + 1] - sort_x[k_sorted]
    ax2.axvline(spacing_l, color="green", lw=1.2, ls="--", alpha=0.7,
                label=f"左邻间距 {spacing_l:.4f}")
    ax2.axvline(spacing_r, color="purple", lw=1.2, ls="--", alpha=0.7,
                label=f"右邻间距 {spacing_r:.4f}")

ax2.set_xlabel("|delta|（节点位移量）", fontsize=11)
ax2.set_ylabel("max |f_fit - f_true|", fontsize=11)
ax2.set_title(
    f"最大拟合误差 vs 节点位移  (nc={NC}, 被扰动节点步骤 #{k_near})\n"
    f"原始 x={nodes_orig[k_near]:+.6f}，El={EL}",
    fontsize=11,
)
ax2.legend(fontsize=9)
ax2.grid(True, which="both", alpha=0.3)

fig2.tight_layout()
path2 = OUT_DIR / "node_sensitivity_error.png"
fig2.savefig(path2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"误差图已保存：{path2}")

# ──────────────────────────────────────────────
# 终端摘要
# ──────────────────────────────────────────────
print("\n===== 正方向误差摘要 =====")
print(f"  {'delta':>8}  {'max_err':>10}  {'MAE':>10}")
print("  " + "-" * 34)
for d, me, mae in zip(delta_vals[pos_mask], max_errors[pos_mask], mae_errors[pos_mask]):
    flag = "  *** 爆炸 ***" if me > 0.1 else ""
    print(f"  {d:+.5f}  {me:10.3e}  {mae:10.3e}{flag}")
