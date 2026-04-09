"""
plot_filter_windows.py
画切比雪夫爆炸滤波器窗函数 |T_m(a·E + b)|，比较两种计算方法：
  方法 1（递推）：用 Chebyshev 递推关系 T_{k+1}(x)=2x*T_k(x)-T_{k-1}(x) 计算
  方法 2（多项式）：将 T_m(x) 展开为单项式多项式 T_m(x)=sum c_k x^k，
                   再把 x=aE+b 代入，得到关于 E 的多项式，直接求值

两种方法在 x∈[-1,1] 内都精确，但方法 2 在 |x|>>1 时会因为高次项系数交错消去而数值爆炸。

默认：m=20, E_upper=33.0, E_lower=[-0.5,-0.4,-0.3,-0.2,-0.1]

用法：
  python plot_filter_windows.py
  python plot_filter_windows.py --m 40 --E_upper 50.0 --E_lower -0.6 -0.4 -0.2
  python plot_filter_windows.py --out_dir my_figs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev as _Cheb, cheb2poly
from numpy.polynomial.polynomial import Polynomial

parser = argparse.ArgumentParser()
parser.add_argument("--m",       type=int,   default=20)
parser.add_argument("--E_upper", type=float, default=33.0)
parser.add_argument("--E_lower", type=float, nargs="+",
                    default=[-0.5, -0.4, -0.3, -0.2, -0.1])
parser.add_argument("--E_min",   type=float, default=None,
                    help="Left edge of x-axis (default: min(E_lower)-1)")
parser.add_argument("--n_pts",   type=int,   default=5000,
                    help="Number of energy sample points")
parser.add_argument("--clip",    type=float, default=100.0,
                    help="Clip |T_m| at this value for display")
parser.add_argument("--out_dir", type=str,   default="figs_filter_windows")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)

E_min_plot = args.E_min if args.E_min is not None else min(args.E_lower) - 1.0
E_arr = np.linspace(E_min_plot, args.E_upper, args.n_pts)

colors = plt.cm.tab10(np.linspace(0, 0.9, len(args.E_lower)))

# ── 预计算 T_m 的单项式系数（方法 2 用）──
# cheb2poly: 返回 c[k] = T_m(x) 中 x^k 的系数
Tm_cheb_basis = np.zeros(args.m + 1)
Tm_cheb_basis[args.m] = 1.0
std_coef = cheb2poly(Tm_cheb_basis)   # T_m(x) = sum_k std_coef[k] * x^k

# ── 计算每条曲线（两种方法）──
curves_rec  = {}   # E_lower_val -> |T_m| via Chebyshev recursion
curves_poly = {}   # E_lower_val -> |T_m| via direct polynomial
poly_coefs  = {}   # E_lower_val -> poly coef array (coef of E^k)

for E_lo in args.E_lower:
    a = 2.0 / (args.E_upper - E_lo)
    b = -(args.E_upper + E_lo) / (args.E_upper - E_lo)
    coef = np.zeros(args.m + 1)
    coef[args.m] = 1.0

    # 方法 1：Chebyshev 递推（numerically stable）
    Tm_rec = _Cheb(coef)(a * E_arr + b)
    curves_rec[E_lo] = np.abs(Tm_rec)

    # 方法 2：展开成关于 E 的单项式多项式
    # T_m(aE+b) = sum_k std_coef[k] * (aE+b)^k
    x_poly = Polynomial([b, a])       # b + a*E  (ascending order)
    result = Polynomial([0.0])
    for deg, c in enumerate(std_coef):
        result = result + c * x_poly**deg
    poly_coefs[E_lo] = result.coef    # coef[k] = coefficient of E^k
    Tm_poly = result(E_arr)           # 直接求值（Horner 法，但系数本身已大）
    curves_poly[E_lo] = np.abs(Tm_poly)

# ── 画图：3 行 ──
# 行 0：完整范围，两种方法叠加
# 行 1：zoom 近 E_lower，两种方法叠加
# 行 2：绝对差值 |poly - recursion|（log 纵轴）
fig, axes = plt.subplots(3, 1, figsize=(13, 14))

xlims = [
    (E_min_plot, args.E_upper),
    (min(args.E_lower) - 0.3, max(args.E_lower) + 0.3),
    (E_min_plot, args.E_upper),
]
row_titles = [
    f"Full range  (m={args.m}, E_upper={args.E_upper})",
    "Zoom near E_lower thresholds",
    "Absolute difference  |poly − recursion|  (log scale)",
]

for row, (ax, xlim, title) in enumerate(zip(axes, xlims, row_titles)):
    for E_lo, col in zip(args.E_lower, colors):
        if row < 2:
            # 两种方法叠加：实线=递推，虚线=多项式
            rec_clipped  = np.clip(curves_rec[E_lo],  0, args.clip)
            poly_clipped = np.clip(curves_poly[E_lo], 0, args.clip)
            ax.plot(E_arr, rec_clipped,  color=col, lw=1.8, ls='-',
                    label=f"recursion  E_lo={E_lo}")
            ax.plot(E_arr, poly_clipped, color=col, lw=1.2, ls='--', alpha=0.8,
                    label=f"poly       E_lo={E_lo}")
            ax.axvline(E_lo, color=col, ls=':', lw=0.8)
            ax.axhline(1.0, color='gray', ls='--', lw=0.7)
            ax.set_ylim(0, args.clip)
            ax.set_ylabel(f"|T_{args.m}(E)| (clipped at {args.clip})")
        else:
            # 差值（log scale）
            diff = np.abs(curves_poly[E_lo] - curves_rec[E_lo])
            diff = np.where(diff > 0, diff, 1e-300)   # avoid log(0)
            ax.semilogy(E_arr, diff, color=col, lw=1.4,
                        label=f"E_lo={E_lo}  max={diff.max():.2e}")
            ax.axvline(E_lo, color=col, ls=':', lw=0.8)
            ax.set_ylabel("|poly − recursion|")

    ax.set_xlim(*xlim)
    ax.set_xlabel("Energy")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls='--', alpha=0.4)

# 在行 0 加图例说明线型
axes[0].plot([], [], color='k', lw=1.8, ls='-',  label="── recursion (stable)")
axes[0].plot([], [], color='k', lw=1.2, ls='--', label="-- polynomial (direct)")
axes[0].legend(fontsize=8, ncol=2)

fig.tight_layout()
png_path = out_dir / "filter_windows.png"
fig.savefig(png_path, dpi=150)
plt.close(fig)
print(f"Plot saved → {png_path}")

# ── 控制台：打印最大差值 ──
print(f"\n{'─'*60}")
print(f"  Max |poly − recursion| over E_arr  (m={args.m})")
print(f"{'─'*60}")
for E_lo in args.E_lower:
    diff     = np.abs(curves_poly[E_lo] - curves_rec[E_lo])
    rel_diff = diff / np.maximum(curves_rec[E_lo], 1e-30)
    print(f"  E_lo={E_lo:5.2f}:  abs_max={diff.max():.3e}  "
          f"rel_max={rel_diff.max():.3e}  "
          f"(at E={E_arr[diff.argmax()]:.3f})")
print(f"{'─'*60}")

# ── 存储数据点 ──
data = {
    "params": {
        "m":       args.m,
        "E_upper": args.E_upper,
        "E_lower": args.E_lower,
        "E_min":   float(E_min_plot),
        "n_pts":   args.n_pts,
    },
    "E_arr":        E_arr.tolist(),
    "curves_rec":   {str(E_lo): curves_rec[E_lo].tolist()  for E_lo in args.E_lower},
    "curves_poly":  {str(E_lo): curves_poly[E_lo].tolist() for E_lo in args.E_lower},
}
json_path = out_dir / "filter_windows_data.json"
with open(json_path, "w") as f:
    json.dump(data, f)
print(f"Data saved  → {json_path}")
for E_lo in args.E_lower:
    peak_rec  = curves_rec[E_lo].max()
    peak_poly = curves_poly[E_lo].max()
    print(f"  E_lower={E_lo:5.2f}: peak |T_m| recursion={peak_rec:.3e}  "
          f"poly={peak_poly:.3e}")
