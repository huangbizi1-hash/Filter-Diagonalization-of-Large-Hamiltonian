"""
print_filter_expressions.py
输出切比雪夫爆炸滤波器的数学表达式：
  f_k(E) = T_m( a_k * E + b_k )
  其中  a_k = 2 / (E_upper - E_lower_k)
        b_k = -(E_upper + E_lower_k) / (E_upper - E_lower_k)

对每条线输出：
  1. 紧凑符号形式（人类可读）
  2. LaTeX 表达式
  3. 作为 E 的多项式（degree-m）的系数
存储到 out_dir/filter_expressions.json 和 filter_expressions.txt。

用法：
  python print_filter_expressions.py
  python print_filter_expressions.py --m 40 --E_upper 70.0 --E_lower -0.6 -0.4
"""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import cheb2poly
from numpy.polynomial.polynomial import Polynomial

parser = argparse.ArgumentParser()
parser.add_argument("--m",       type=int,   default=20)
parser.add_argument("--E_upper", type=float, default=33.0)
parser.add_argument("--E_lower", type=float, nargs="+",
                    default=[-0.5, -0.4, -0.3, -0.2, -0.1])
parser.add_argument("--out_dir", type=str,   default="figs_filter_windows")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)

# ── T_m 的标准多项式系数（以 x 为变量）──
# numpy cheb2poly: coef[k] 是 x^k 的系数
Tm_cheb = np.zeros(args.m + 1)
Tm_cheb[args.m] = 1.0          # 只有第 m 阶 Chebyshev 分量
std_coef = cheb2poly(Tm_cheb)  # T_m(x) = sum_k std_coef[k] * x^k

records = []
lines_txt = [
    f"Chebyshev Explosion Filter:  T_{args.m}(a*E + b)",
    f"  m = {args.m}",
    f"  E_upper = {args.E_upper}",
    "=" * 70,
]

for k, E_lo in enumerate(args.E_lower):
    dE = args.E_upper - E_lo
    a  = 2.0 / dE
    b  = -(args.E_upper + E_lo) / dE

    # ── x = a*E + b 代入 T_m(x) → T_m(a*E + b) 的 E 多项式系数 ──
    x_poly = Polynomial([b, a])      # b + a*E  (coef in ascending order)
    result = Polynomial([0.0])
    for deg, c in enumerate(std_coef):
        result = result + c * x_poly**deg
    poly_in_E = result.coef          # [c0, c1, ..., cm], coef of E^k = poly_in_E[k]

    # ── 符号表达式 ──
    # T_m((2E - (E_upper + E_lower)) / (E_upper - E_lower))
    numerator_const  = args.E_upper + E_lo    # E_upper + E_lower
    denominator      = dE                     # E_upper - E_lower
    compact  = (f"T_{args.m}( (2*E - {numerator_const:.6g}) / {denominator:.6g} )")
    latex    = (f"T_{{{args.m}}}\\!\\left(\\frac{{2E - {numerator_const:.6g}}}"
                f"{{{denominator:.6g}}}\\right)")

    # ── 多项式字符串（只打印非零项）──
    poly_terms = []
    for deg in range(len(poly_in_E)):
        c = poly_in_E[deg]
        if abs(c) < 1e-10 * abs(poly_in_E).max():
            continue
        if deg == 0:
            poly_terms.append(f"{c:.6e}")
        elif deg == 1:
            poly_terms.append(f"({c:.6e})*E")
        else:
            poly_terms.append(f"({c:.6e})*E^{deg}")
    poly_str = " + ".join(poly_terms) if poly_terms else "0"

    record = {
        "k":           k,
        "E_lower":     E_lo,
        "E_upper":     args.E_upper,
        "m":           args.m,
        "a":           float(a),
        "b":           float(b),
        "compact":     compact,
        "latex":       latex,
        "poly_coef_E": [float(c) for c in poly_in_E],  # poly_coef_E[k] = coef of E^k
    }
    records.append(record)

    block = [
        f"\nStage {k}  (E_lower = {E_lo})",
        f"  a         = {a:.10f}",
        f"  b         = {b:.10f}",
        f"  compact   : {compact}",
        f"  LaTeX     : {latex}",
        f"  poly(E)   : {poly_str}",
        f"  poly_coef : [c0, c1, ..., cm] where f(E) = sum_k coef[k] * E^k",
    ]
    for deg, c in enumerate(poly_in_E):
        block.append(f"    coef[{deg:2d}] = {c:+.10e}")
    lines_txt.extend(block)

    # ── 控制台输出 ──
    print(f"\nStage {k}  E_lower={E_lo}")
    print(f"  a = {a:.10f}   b = {b:.10f}")
    print(f"  compact : {compact}")
    print(f"  LaTeX   : {latex}")
    print(f"  poly(E) : {poly_str[:120]}{'...' if len(poly_str)>120 else ''}")

# ── 保存 JSON ──
out_json = {
    "params": {"m": args.m, "E_upper": args.E_upper, "E_lower": args.E_lower},
    "note": ("poly_coef_E[k] is the coefficient of E^k in the degree-m polynomial "
             "T_m(a*E + b) expanded in standard monomial basis"),
    "stages": records,
}
json_path = out_dir / "filter_expressions.json"
with open(json_path, "w") as f:
    json.dump(out_json, f, indent=2)
print(f"\nJSON saved → {json_path}")

# ── 保存 txt ──
txt_path = out_dir / "filter_expressions.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(lines_txt) + "\n")
print(f"TXT  saved → {txt_path}")
