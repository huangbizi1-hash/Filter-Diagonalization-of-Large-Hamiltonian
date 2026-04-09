"""
plot_filter_windows.py
画切比雪夫爆炸滤波器窗函数 |T_m(a·E + b)|，不需要加载势能。

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
from numpy.polynomial.chebyshev import Chebyshev as _Cheb

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

# ── 计算每条曲线 ──
curves = {}   # E_lower_val -> Tm_vals array
for E_lo in args.E_lower:
    a    = 2.0 / (args.E_upper - E_lo)
    b    = -(args.E_upper + E_lo) / (args.E_upper - E_lo)
    coef = np.zeros(args.m + 1)
    coef[args.m] = 1.0
    Tm   = _Cheb(coef)(a * E_arr + b)
    curves[E_lo] = np.abs(Tm)

# ── 画图 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (ax, xlim, title_suffix) in enumerate(zip(
    axes,
    [(E_min_plot, args.E_upper),
     (min(args.E_lower) - 0.3, max(args.E_lower) + 0.3)],
    ["full range", "zoom near E_lower"],
)):
    for E_lo, col in zip(args.E_lower, colors):
        vals = np.clip(curves[E_lo], 0, args.clip)
        ax.plot(E_arr, vals, color=col, lw=1.6,
                label=f"E_lower = {E_lo}")
        ax.axvline(E_lo, color=col, ls=':', lw=0.9)

    ax.axhline(1.0, color='gray', ls='--', lw=0.8, label="|T_m|=1")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, args.clip)
    ax.set_xlabel("Energy")
    ax.set_ylabel(f"|T_{args.m}(E)| (clipped at {args.clip})")
    ax.set_title(f"Chebyshev explosion filter  m={args.m},  E_upper={args.E_upper}"
                 f"\n({title_suffix})")
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', alpha=0.5)

fig.tight_layout()
png_path = out_dir / "filter_windows.png"
fig.savefig(png_path, dpi=150)
plt.close(fig)
print(f"Plot saved → {png_path}")

# ── 存储数据点 ──
data = {
    "params": {
        "m":       args.m,
        "E_upper": args.E_upper,
        "E_lower": args.E_lower,
        "E_min":   float(E_min_plot),
        "n_pts":   args.n_pts,
    },
    "E_arr":  E_arr.tolist(),
    "curves": {str(E_lo): curves[E_lo].tolist() for E_lo in args.E_lower},
}
json_path = out_dir / "filter_windows_data.json"
with open(json_path, "w") as f:
    json.dump(data, f)
print(f"Data saved  → {json_path}")
print(f"  E_arr shape: {len(E_arr)} points in [{E_arr[0]:.3f}, {E_arr[-1]:.3f}]")
for E_lo in args.E_lower:
    peak = curves[E_lo].max()
    print(f"  E_lower={E_lo:5.2f}: peak |T_m| = {peak:.3e}")
