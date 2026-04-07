"""
bandpass_node_sensitivity.py
固定 nc=500，研究 bandpass 拟合对最近插值节点位置的敏感性。

参数（与 main.py CONFIG 完全一致）
    dE=50, Vmin=-5  →  物理域 [-5, 45] Hartree，缩放域 [-2, 2]
    El=-0.12（物理，Hartree），E1=0.05，beta=50，nc=500

关键说明
--------
Ashkenazy 节点在缩放坐标 [-2,2] 生成；E1、beta 是物理参数，
等效缩放值：
  E1_s   = 0.05 * 4/50 = 0.004  （极窄！平均节点间距 ~0.008 >> E1_s）
  beta_s = 50  * 50/4  = 625    （极陡！）
→ 即使 delta=0 误差也较大，是因为节点在过渡区太稀疏。

操作：找到最靠近 El 的缩放节点，平移 delta（缩放坐标单位），
      观察拟合曲线和误差如何变化。

输出（figs_nodes/）
    node_sensitivity_curves.png   — 不同 delta 的拟合曲线（物理坐标轴）
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
# 配置（与 main.py CONFIG 保持一致）
# ──────────────────────────────────────────────
NC      = 500
SMIN, SMAX = -2.0, 2.0   # 缩放域

# 物理参数
EL_PHYS = -0.12    # 带通中心（Hartree）
E1_PHYS = 0.05     # 带通半宽（Hartree）
BETA    = 50.0     # 过渡陡峭系数（物理坐标）
DE      = 50.0     # dE（Hartree）
VMIN    = -5.0     # Vmin（Hartree）

# 物理 ↔ 缩放换算
def phys_to_scaled(x_phys):
    return 4.0 * (x_phys - VMIN) / DE - 2.0

def scaled_to_phys(s):
    return (s + 2.0) * DE / 4.0 + VMIN

EL_SCALED  = phys_to_scaled(EL_PHYS)
E1_SCALED  = E1_PHYS * 4.0 / DE      # 0.004
BETA_SCALED = BETA * DE / 4.0        # 625.0

print(f"物理参数：El={EL_PHYS}, E1={E1_PHYS}, beta={BETA}, dE={DE}, Vmin={VMIN}")
print(f"等效缩放：El_s={EL_SCALED:.5f}, E1_s={E1_SCALED:.5f}, beta_s={BETA_SCALED:.1f}")
print(f"缩放空间节点平均间距 ~{(SMAX-SMIN)/NC:.5f}  vs  E1_s={E1_SCALED:.5f}")

# 平移量（缩放坐标单位）；节点间距 ~0.008，E1_s=0.004
DELTAS_POS = np.array([0.0, 0.0002, 0.0005, 0.001, 0.002, 0.003,
                        0.005, 0.008, 0.012, 0.02])
DELTAS_NEG = -DELTAS_POS[1:]

N_EVAL  = 6000
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
# 1. 生成节点（缩放坐标，贪心顺序）
# ──────────────────────────────────────────────
print(f"\n生成 nc={NC} Ashkenazy 节点（缩放坐标）...")
nodes_orig = _samp_points_ashkenazy(SMIN, SMAX, NC)

# 找最靠近 EL_SCALED 的节点（在贪心序列中的索引）
k_near = int(np.argmin(np.abs(nodes_orig - EL_SCALED)))
x_near_s = nodes_orig[k_near]
x_near_p = scaled_to_phys(x_near_s)
print(f"最靠近 El={EL_PHYS} 的节点：")
print(f"  贪心步骤序号 = #{k_near}")
print(f"  缩放坐标    = {x_near_s:+.6f}")
print(f"  物理坐标    = {x_near_p:+.6f} Hartree")
print(f"  距离（物理）= {abs(x_near_p - EL_PHYS):.6f} Hartree")

# 按 x 排序（标注用）
sort_idx           = np.argsort(nodes_orig)
nodes_sorted_s     = nodes_orig[sort_idx]          # 缩放坐标，升序
nodes_sorted_step  = sort_idx                      # 对应贪心步骤序号

# El 附近 ±0.03 Hartree（物理）的节点
annot_margin_phys = 0.03
annot_lo_s = phys_to_scaled(EL_PHYS - annot_margin_phys)
annot_hi_s = phys_to_scaled(EL_PHYS + annot_margin_phys)
in_range   = (nodes_sorted_s >= annot_lo_s) & (nodes_sorted_s <= annot_hi_s)
annot_s    = nodes_sorted_s[in_range]
annot_p    = scaled_to_phys(annot_s)              # 物理坐标
annot_step = nodes_sorted_step[in_range]

print(f"\nEl 附近节点（物理 ±{annot_margin_phys} Hartree）：")
print(f"  {'步骤':>5}  {'物理坐标':>10}  {'缩放坐标':>10}")
for s_n, x_p, x_s in sorted(zip(annot_step, annot_p, annot_s)):
    marker = "  <-- 被扰动" if s_n == k_near else ""
    print(f"  #{s_n:4d}  {x_p:+10.6f}  {x_s:+10.6f}{marker}")

# ──────────────────────────────────────────────
# 2. 带通函数（物理坐标）
# ──────────────────────────────────────────────
def bandpass_phys(x_phys):
    EL_edge = EL_PHYS - E1_PHYS
    ER_edge = EL_PHYS + E1_PHYS
    return 0.5 * (np.tanh(BETA * (x_phys - EL_edge))
                - np.tanh(BETA * (x_phys - ER_edge)))

# 细密评估网格（缩放坐标 → 物理坐标）
s_eval  = np.linspace(SMIN, SMAX, N_EVAL)
x_eval  = scaled_to_phys(s_eval)          # 物理坐标，用于绘图和误差计算
f_true  = bandpass_phys(x_eval)

# ──────────────────────────────────────────────
# 3. 拟合函数（扰动节点 k_near 的缩放坐标）
# ──────────────────────────────────────────────
def fit_with_delta(delta_scaled: float):
    """平移节点 k_near（缩放坐标 +delta），重新拟合。"""
    nodes_s  = nodes_orig.copy()
    nodes_s[k_near] += delta_scaled
    # 把节点缩放坐标转换为物理坐标，再求带通函数值
    nodes_p  = scaled_to_phys(nodes_s)
    f_nodes  = bandpass_phys(nodes_p)
    an       = newton_coeffs_vec(nodes_s, f_nodes)
    f_fit    = _eval_newton_grid(nodes_s, an, s_eval)   # 在缩放网格求值
    err      = np.abs(f_fit - f_true)
    return err.max(), err.mean(), f_fit

# ──────────────────────────────────────────────
# 4. 计算所有 delta
# ──────────────────────────────────────────────
all_deltas     = np.concatenate([np.sort(DELTAS_NEG), DELTAS_POS])
deltas_to_plot = DELTAS_POS

delta_vals = []
max_errors = []
mae_errors = []
fit_curves = {}

print(f"\n计算 {len(all_deltas)} 个 delta 值...")
for delta in all_deltas:
    me, mae, f_fit = fit_with_delta(delta)
    delta_vals.append(delta)
    max_errors.append(me)
    mae_errors.append(mae)
    if any(delta == d for d in deltas_to_plot):
        fit_curves[delta] = f_fit.copy()
    flag = "  *** 爆炸 ***" if me > 0.5 else ""
    delta_p = delta * DE / 4.0   # 缩放 delta 对应的物理位移（Hartree）
    print(f"  delta_s={delta:+.5f}  delta_p={delta_p:+.5f} Ha"
          f"  max_err={me:.2e}  MAE={mae:.2e}{flag}")

delta_vals = np.array(delta_vals)
max_errors = np.array(max_errors)
mae_errors = np.array(mae_errors)

# ──────────────────────────────────────────────
# 颜色映射
# ──────────────────────────────────────────────
n_plot   = len(deltas_to_plot)
colors_c = cm.plasma(np.linspace(0.05, 0.92, n_plot))

# ──────────────────────────────────────────────
# 图1：拟合曲线（物理坐标轴）
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                          gridspec_kw={"height_ratios": [2, 1]})
ax_main = axes[0]
ax_err  = axes[1]

# 显示范围：物理坐标 El ± 0.12
xlo_p = EL_PHYS - 0.12
xhi_p = EL_PHYS + 0.12
xlo_s = phys_to_scaled(xlo_p)
xhi_s = phys_to_scaled(xhi_p)

# ── 拟合曲线 ──
ax_main.plot(x_eval, f_true, "k-", lw=2.5, zorder=5, label="真实 bandpass")

for delta, col in zip(deltas_to_plot, colors_c):
    f_fit  = fit_curves[delta]
    lw_    = 2.0 if delta == 0.0 else 1.3
    ls_    = "-" if delta == 0.0 else "--"
    dp_str = f"{delta * DE / 4.0:+.4f} Ha"
    ax_main.plot(x_eval, f_fit, ls_, color=col, lw=lw_, alpha=0.85,
                 label=f"δ_s={delta:.4f} ({dp_str})")

# 带通边沿（物理坐标）
ax_main.axvline(EL_PHYS - E1_PHYS, color="navy", lw=1.0, ls=":", alpha=0.6)
ax_main.axvline(EL_PHYS + E1_PHYS, color="navy", lw=1.0, ls=":", alpha=0.6)
ax_main.axvline(EL_PHYS,           color="navy", lw=0.8, ls="-", alpha=0.3)

# 节点标注（物理坐标）
for x_p, x_s, s_n in zip(annot_p, annot_s, annot_step):
    is_near = (s_n == k_near)
    ax_main.axvline(x_p, color="red" if is_near else "gray",
                    lw=1.8 if is_near else 0.7,
                    alpha=0.8 if is_near else 0.35, zorder=2)
    ax_main.text(x_p, 1.08, f"#{s_n}", fontsize=7,
                 ha="center", va="bottom",
                 color="red" if is_near else "dimgray",
                 fontweight="bold" if is_near else "normal",
                 transform=ax_main.get_xaxis_transform())

ax_main.set_xlim(xlo_p, xhi_p)
ax_main.set_ylim(-0.15, 1.25)
ax_main.set_ylabel("f(E) [Hartree]", fontsize=11)
ax_main.set_title(
    f"Bandpass 拟合曲线  (nc={NC}, El={EL_PHYS} Ha, E1={E1_PHYS} Ha, "
    f"beta={BETA}, dE={DE}, Vmin={VMIN})\n"
    f"红竖线/红标 = 被扰动节点（步骤 #{k_near}, "
    f"x_phys={x_near_p:+.5f} Ha）",
    fontsize=10,
)
ax_main.legend(fontsize=7, ncol=3, loc="upper right")
ax_main.grid(True, alpha=0.25)
ax_main.set_xlabel("E (Hartree)", fontsize=11)

# ── 残差 ──
for delta, col in zip(deltas_to_plot, colors_c):
    f_fit = fit_curves[delta]
    ax_err.semilogy(x_eval, np.abs(f_fit - f_true) + 1e-17,
                    color=col, lw=1.1 if delta > 0 else 2.0, alpha=0.85,
                    label=f"δ_s={delta:.4f}")

for x_p, s_n in zip(annot_p, annot_step):
    ax_err.axvline(x_p, color="red" if s_n == k_near else "gray",
                   lw=1.5 if s_n == k_near else 0.6,
                   alpha=0.7 if s_n == k_near else 0.3)

ax_err.axvline(EL_PHYS - E1_PHYS, color="navy", lw=1.0, ls=":", alpha=0.5)
ax_err.axvline(EL_PHYS + E1_PHYS, color="navy", lw=1.0, ls=":", alpha=0.5)
ax_err.set_xlim(xlo_p, xhi_p)
ax_err.set_ylabel("|残差|", fontsize=10)
ax_err.set_xlabel("E (Hartree)", fontsize=11)
ax_err.grid(True, which="both", alpha=0.25)
ax_err.set_title("点态残差 |f_fit - f_true|（半对数）", fontsize=10)
ax_err.legend(fontsize=7, ncol=4, loc="upper right")

fig.tight_layout()
path1 = OUT_DIR / "node_sensitivity_curves.png"
fig.savefig(path1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n曲线图已保存：{path1}")

# ──────────────────────────────────────────────
# 图2：max_error vs delta（正负方向，x轴用物理单位）
# ──────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))

pos_mask = delta_vals >= 0
neg_mask = delta_vals < 0

# x轴换算为物理位移（Hartree）
delta_p_pos = delta_vals[pos_mask] * DE / 4.0
delta_p_neg = (-delta_vals[neg_mask]) * DE / 4.0

ax2.semilogy(delta_p_pos, max_errors[pos_mask],
             "o-", color="steelblue", lw=1.8, ms=6, label="delta > 0（正方向）")
ax2.semilogy(delta_p_neg, max_errors[neg_mask],
             "s--", color="tomato", lw=1.5, ms=5, alpha=0.8, label="|delta|（负方向）")

# 相邻节点间距（物理单位）
sort_s   = np.sort(nodes_orig)
k_sorted = np.searchsorted(sort_s, nodes_orig[k_near])
if 0 < k_sorted < NC - 1:
    sp_l_p = (sort_s[k_sorted] - sort_s[k_sorted - 1]) * DE / 4.0
    sp_r_p = (sort_s[k_sorted + 1] - sort_s[k_sorted]) * DE / 4.0
    ax2.axvline(sp_l_p, color="green",  lw=1.2, ls="--", alpha=0.7,
                label=f"左邻间距 {sp_l_p:.4f} Ha")
    ax2.axvline(sp_r_p, color="purple", lw=1.2, ls="--", alpha=0.7,
                label=f"右邻间距 {sp_r_p:.4f} Ha")

ax2.set_xlabel("|delta| (Hartree，物理坐标位移)", fontsize=11)
ax2.set_ylabel("max |f_fit - f_true|", fontsize=11)
ax2.set_title(
    f"最大拟合误差 vs 节点位移  (nc={NC}, 被扰动节点步骤 #{k_near})\n"
    f"物理坐标 x={x_near_p:+.6f} Ha，El={EL_PHYS} Ha",
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
print(f"  {'delta_s':>9}  {'delta_p(Ha)':>12}  {'max_err':>10}  {'MAE':>10}")
print("  " + "-" * 50)
for d, dp, me, mae in zip(delta_vals[pos_mask], delta_p_pos,
                           max_errors[pos_mask], mae_errors[pos_mask]):
    flag = "  *** 爆炸 ***" if me > 0.5 else ""
    print(f"  {d:+.6f}  {dp:+12.5f}  {me:10.3e}  {mae:10.3e}{flag}")
