"""
scan_jdqmr_fft.py
在高斯重构势能上扫描多个 E_target，使用 FFT-DVR 离散化 + PRIMME JDQMR 求解器。

JDQMR（Jacobi-Davidson QMR）由 PRIMME 库提供，每个目标能量独立调用
primme.eigsh(which=E_target)，可一次返回目标附近的多个本征值。
FFT 算子复杂度 O(N³ log N)，远优于 Sinc-DVR 的 O(N⁴)。

用法
----
python scan_jdqmr_fft.py [选项]

选项
----
--e_targets FLOAT ...    显式指定 E_target 列表（指定后忽略 --e_min/--e_max/--e_step）
--e_min     FLOAT        扫描起始能量   （默认 -0.25）
--e_max     FLOAT        扫描终止能量（含）（默认  0.20）
--e_step    FLOAT        扫描间隔        （默认  0.002）
--n_levels  INT          每个目标附近求解的本征值数（默认 1）
--n         INT          每轴网格点数    （默认 64）
--r_cut     FLOAT        Gaussian 截断半径 Angstrom（默认 7.0）
--cube_file STR          势能 cube 文件路径（默认 localPot.cube）
--params_file STR        高斯拟合参数文件  （默认 gaussian_fit_params.json）
--tol       FLOAT        PRIMME 迭代收敛精度（默认 1e-6）
--tolerance FLOAT        候选本征值筛选阈值 |E-E_target|（默认 0.003）
--output    STR          结果保存文件（JSON，默认不保存）
--verbose                打印每个目标能量的 PRIMME 迭代细节

示例
----
# 默认扫描 -0.25 到 0.20，间隔 0.002
python scan_jdqmr_fft.py

# 指定 E_target 列表
python scan_jdqmr_fft.py --e_targets -0.10 -0.05 0.00 0.05

# 每个目标求 3 个本征值，保存结果
python scan_jdqmr_fft.py --n_levels 3 --output results_jdqmr.json
"""

import argparse
import json
import sys
import time
import numpy as np

import ho3d_solvers_v2 as solver
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid

try:
    import primme
    HAS_PRIMME = True
except ImportError:
    HAS_PRIMME = False


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="扫描高斯重构势能的多个目标能量（FFT-DVR + PRIMME JDQMR）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--e_targets",   type=float, nargs="+", default=None,
                   help="显式指定 E_target 列表（指定后忽略 --e_min/--e_max/--e_step）")
    p.add_argument("--e_min",       type=float, default=-0.25,
                   help="扫描起始能量（默认 -0.25）")
    p.add_argument("--e_max",       type=float, default=0.20,
                   help="扫描终止能量（含，默认 0.20）")
    p.add_argument("--e_step",      type=float, default=0.002,
                   help="扫描间隔（默认 0.002）")
    p.add_argument("--n_levels",    type=int,   default=1,
                   help="每个目标附近求解的本征值数（默认 1）")
    p.add_argument("--n",           type=int,   default=64,
                   help="每轴网格点数（默认 64）")
    p.add_argument("--r_cut",       type=float, default=7.0,
                   help="Gaussian 截断半径（Angstrom，默认 7.0）")
    p.add_argument("--cube_file",   type=str,   default="localPot.cube",
                   help="势能 cube 文件路径（默认 localPot.cube）")
    p.add_argument("--params_file", type=str,   default="gaussian_fit_params.json",
                   help="高斯拟合参数文件（默认 gaussian_fit_params.json）")
    p.add_argument("--tol",         type=float, default=1e-6,
                   help="PRIMME 迭代收敛精度（默认 1e-6）")
    p.add_argument("--tolerance",   type=float, default=0.003,
                   help="候选本征值筛选阈值 |E-E_target|（默认 0.003）")
    p.add_argument("--output",      type=str,   default=None,
                   help="结果保存路径（JSON，默认不保存）")
    p.add_argument("--verbose",     action="store_true",
                   help="打印 PRIMME 迭代细节（verbosity=3）")
    return p.parse_args()


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────

def main():
    if not HAS_PRIMME:
        print("错误：未安装 primme 库，请执行 pip install primme", file=sys.stderr)
        sys.exit(1)

    args = parse_args()

    # ── 构建 E_target 列表 ──────────────────────
    if args.e_targets is not None:
        targets = np.array(sorted(args.e_targets))
    else:
        n_pts = int(round((args.e_max - args.e_min) / args.e_step)) + 1
        targets = np.round(np.linspace(args.e_min, args.e_max, n_pts), 8)

    print("扫描参数：")
    print(f"  离散化   ：FFT-DVR")
    print(f"  求解器   ：PRIMME JDQMR")
    print(f"  目标能量 ：{len(targets)} 个，[{targets[0]:.4f}, {targets[-1]:.4f}]，间隔 {args.e_step}")
    print(f"  每目标本征值数：{args.n_levels}")
    print(f"  PRIMME tol：{args.tol}")
    print(f"  网格分辨 ：N={args.n}，总点数 {args.n**3}")

    # ── 加载势能 ────────────────────────────────
    print("\n加载高斯重构势能...")
    try:
        builder = GaussianPotentialBuilder(
            cube_file=args.cube_file,
            params_file=args.params_file,
            r_cut=args.r_cut,
        )
        x, y, z, V = builder.build_potential(args.n)
        pot_grid = PotentialGrid(x, y, z, V, source="gaussian_files")
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e}", file=sys.stderr)
        print("请确保 cube 文件和参数文件在当前目录或通过 --cube_file/--params_file 指定。",
              file=sys.stderr)
        sys.exit(1)

    print(f"  空间范围：[{x[0]:.3f}, {x[-1]:.3f}] Bohr")
    print(f"  V 范围  ：[{V.min():.4f}, {V.max():.4f}] Hartree")

    # ── 预构建 FFT 算符（所有目标共享同一算符）──
    print("\n构建 FFT-DVR 哈密顿量算符...")
    t_build_start = time.perf_counter()
    H_op, n_un, _ = solver.build_3d_fft_operator(args.n, pot_grid)
    t_build = time.perf_counter() - t_build_start
    print(f"  算符维度：{n_un}  构建耗时：{t_build:.2f}s")

    # ── 扫描 ────────────────────────────────────
    print(f"\n开始扫描（共 {len(targets)} 个目标）...")
    header = (f"  {'E_target':>10}  {'E_ritz(s)':>40}  "
              f"{'min|ΔE|':>10}  {'耗时':>7}")
    print(header)
    print(f"  {'-'*80}")

    verbosity = 3 if args.verbose else 0   # maps to primme printLevel
    results = []
    t_scan_start = time.perf_counter()

    for i, E_t in enumerate(targets):
        t0 = time.perf_counter()
        try:
            evals, _ = primme.eigsh(
                H_op,
                k=args.n_levels,
                which=float(E_t),
                method="PRIMME_JDQMR",
                tol=args.tol,
                printLevel=verbosity,
            )
            evals = np.sort(evals)
            min_diff = float(np.min(np.abs(evals - E_t)))
            closest = float(evals[np.argmin(np.abs(evals - E_t))])
            success = True
            err_msg = ""
        except Exception as exc:
            evals = np.array([float("nan")] * args.n_levels)
            min_diff = float("nan")
            closest = float("nan")
            success = False
            err_msg = str(exc)

        t_it = time.perf_counter() - t0

        results.append({
            "E_target":  float(E_t),
            "evals":     evals.tolist(),
            "closest":   closest,
            "min_diff":  min_diff,
            "success":   success,
            "err_msg":   err_msg,
            "time_s":    t_it,
        })

        evals_str = "  ".join(f"{e:10.6f}" for e in evals)
        diff_str = f"{min_diff:.3e}" if success else "  ERROR  "
        print(f"  {E_t:>10.4f}  {evals_str:>40}  {diff_str:>10}  {t_it:>6.1f}s")

        if not success:
            print(f"    !! {err_msg}")

        # 进度提示（每25个）
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t_scan_start
            remaining = elapsed / (i + 1) * (len(targets) - i - 1)
            print(f"  -- {i+1}/{len(targets)} 完成，已用 {elapsed:.0f}s，"
                  f"预计剩余 {remaining:.0f}s --")

    t_total = time.perf_counter() - t_scan_start

    # ── 汇总 ────────────────────────────────────
    n_ok = sum(r["success"] for r in results)
    print(f"\n扫描完成：{len(targets)} 个目标，{n_ok} 个成功，总耗时 {t_total:.1f}s")
    print(f"  平均每个目标：{t_total/len(targets):.1f}s")

    candidates = [r for r in results if r["success"] and r["min_diff"] < args.tolerance]
    if candidates:
        print(f"\n候选本征值（|E_closest - E_target| < {args.tolerance:.3e}）：")
        for r in candidates:
            print(f"  E_target={r['E_target']:>8.4f}  E_closest={r['closest']:>10.6f}  "
                  f"diff={r['min_diff']:.3e}")
    else:
        print(f"\n未找到与目标相差 < {args.tolerance:.3e} 的候选本征值。")

    # ── 保存 ────────────────────────────────────
    if args.output:
        output_data = {
            "params": {
                "method":      "fft_dvr:jdqmr",
                "n_levels":    args.n_levels,
                "n":           args.n,
                "r_cut":       args.r_cut,
                "cube_file":   args.cube_file,
                "params_file": args.params_file,
                "primme_tol":  args.tol,
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至：{args.output}")


if __name__ == "__main__":
    main()
