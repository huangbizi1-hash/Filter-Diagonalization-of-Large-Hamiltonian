"""
scan_gaussian_energy.py
在高斯重构势能上扫描多个 E_target，利用 fft_dvr:cgmin 计算对应 Ritz 能量。

用法
----
python scan_gaussian_energy.py [选项]

选项
----
--delta_e   FLOAT        能量收敛阈值 |ΔE| （默认 5e-4）
--maxiter   INT          最大迭代次数   （默认 2000）
--e_targets FLOAT ...    指定 E_target 列表（指定后忽略 --e_min/--e_max/--e_step）
--e_min     FLOAT        扫描起始能量   （默认 -0.25）
--e_max     FLOAT        扫描终止能量（含）（默认  0.20）
--e_step    FLOAT        扫描间隔        （默认  0.002）
--n         INT          每轴网格点数    （默认 20）
--r_cut     FLOAT        Gaussian 截断半径 Angstrom（默认 7.0）
--cube_file STR          势能 cube 文件路径（默认 localPot.cube）
--params_file STR        高斯拟合参数文件  （默认 gaussian_fit_params.json）
--output    STR          结果保存文件（JSON，默认不保存）
--verbose                打印每个目标能量的 CG 迭代细节

示例
----
# 默认扫描 -0.25 到 0.20，间隔 0.002，ΔE<5e-4
python scan_gaussian_energy.py

# 自定义 ΔE 和迭代次数
python scan_gaussian_energy.py --delta_e 1e-4 --maxiter 3000

# 指定 E_target 列表
python scan_gaussian_energy.py --e_targets -0.10 -0.05 0.00 0.05

# 保存结果
python scan_gaussian_energy.py --output results.json
"""

import argparse
import json
import sys
import time
import numpy as np

import ho3d_solvers_v2 as solver
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="扫描高斯重构势能的多个目标能量（fft_dvr:cgmin）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--delta_e",    type=float, default=5e-4,
                   help="能量收敛阈值 |ΔE|（默认 5e-4）")
    p.add_argument("--tolerance",  type=float, default=0.003,
                   help="候选本征值筛选阈值 |E_ritz-E_target|（默认 0.003）")
    p.add_argument("--maxiter",    type=int,   default=2000,
                   help="最大迭代次数（默认 2000）")
    p.add_argument("--e_targets",  type=float, nargs="+", default=None,
                   help="显式指定 E_target 列表（指定后忽略 --e_min/--e_max/--e_step）")
    p.add_argument("--e_min",      type=float, default=-0.25,
                   help="扫描起始能量（默认 -0.25）")
    p.add_argument("--e_max",      type=float, default=0.20,
                   help="扫描终止能量（含，默认 0.20）")
    p.add_argument("--e_step",     type=float, default=0.002,
                   help="扫描间隔（默认 0.002）")
    p.add_argument("--n",          type=int,   default=64,
                   help="每轴网格点数（默认 64）")
    p.add_argument("--r_cut",      type=float, default=7.0,
                   help="Gaussian 截断半径（Angstrom，默认 7.0）")
    p.add_argument("--cube_file",  type=str,   default="localPot.cube",
                   help="势能 cube 文件路径（默认 localPot.cube）")
    p.add_argument("--params_file",type=str,   default="gaussian_fit_params.json",
                   help="高斯拟合参数文件（默认 gaussian_fit_params.json）")
    p.add_argument("--output",     type=str,   default=None,
                   help="结果保存路径（JSON，默认不保存）")
    p.add_argument("--verbose",    action="store_true",
                   help="打印每个目标能量的 CG 迭代细节")
    return p.parse_args()


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 构建 E_target 列表 ──────────────────────
    if args.e_targets is not None:
        targets = np.array(sorted(args.e_targets))
    else:
        # np.arange 会因浮点误差略微偏移，用 round 消除
        n_pts = int(round((args.e_max - args.e_min) / args.e_step)) + 1
        targets = np.round(np.linspace(args.e_min, args.e_max, n_pts), 8)

    print(f"扫描参数：")
    print(f"  目标能量：{len(targets)} 个，[{targets[0]:.4f}, {targets[-1]:.4f}]，间隔 {args.e_step}")
    print(f"  ΔE 阈值 ：{args.delta_e}")
    print(f"  最大迭代：{args.maxiter}")
    print(f"  网格分辨：N={args.n}，总点数 {args.n**3}")

    # ── 加载势能 ────────────────────────────────
    print(f"\n加载高斯重构势能...")
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
    print(f"\n构建 FFT-DVR 哈密顿量算符...")
    t_build_start = time.perf_counter()
    H_op, n_un, _ = solver.build_3d_fft_operator(args.n, pot_grid)
    t_build = time.perf_counter() - t_build_start
    print(f"  算符维度：{n_un}  构建耗时：{t_build:.2f}s")

    # ── 扫描 ────────────────────────────────────
    print(f"\n开始扫描（共 {len(targets)} 个目标）...")
    print(f"\n  {'E_target':>10}  {'E_ritz':>12}  {'|E_ritz-E_t|':>13}  "
          f"{'eig_res':>10}  {'fold_res':>10}  "
          f"{'n_iter':>7}  {'收敛':>5}  {'原因':<20}  {'耗时':>7}")
    print(f"  {'-'*115}")

    results = []
    rng = np.random.default_rng(42)
    t_scan_start = time.perf_counter()

    for i, E_t in enumerate(targets):
        t0 = time.perf_counter()

        x0 = rng.standard_normal(n_un)
        cg_res = solver.cg_minimize_folded(
            H_op, float(E_t), x0,
            maxiter=args.maxiter,
            energy_tol=args.delta_e,
            verbose=args.verbose,
        )

        t_it = time.perf_counter() - t0
        E_r = cg_res["E_ritz"]
        diff = abs(E_r - float(E_t))
        eig_res  = cg_res["eig_res"]   # ||Hx - E_ritz x||
        fold_res = cg_res["fold_res"]  # ||(H - target I)x||

        results.append({
            "E_target":    float(E_t),
            "E_ritz":      E_r,
            "diff":        diff,
            "eig_res":     eig_res,
            "fold_res":    fold_res,
            "n_iter":      cg_res["n_iter"],
            "converged":   cg_res["converged"],
            "conv_reason": cg_res["conv_reason"],
            "time_s":      t_it,
        })

        conv_mark = "✓" if cg_res["converged"] else "✗"
        print(f"  {E_t:>10.4f}  {E_r:>12.6f}  {diff:>13.4e}  "
              f"{eig_res:>10.3e}  {fold_res:>10.3e}  "
              f"{cg_res['n_iter']:>7d}  {conv_mark:>5}  "
              f"{cg_res['conv_reason']:<20}  {t_it:>6.1f}s")

        # 进度提示（每25个）
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t_scan_start
            remaining = elapsed / (i + 1) * (len(targets) - i - 1)
            print(f"  -- {i+1}/{len(targets)} 完成，已用 {elapsed:.0f}s，"
                  f"预计剩余 {remaining:.0f}s --")

    t_total = time.perf_counter() - t_scan_start

    # ── 汇总 ────────────────────────────────────
    n_conv = sum(r["converged"] for r in results)
    print(f"\n扫描完成：{len(targets)} 个目标，{n_conv} 个收敛，总耗时 {t_total:.1f}s")
    print(f"  平均每个目标：{t_total/len(targets):.1f}s")

    # 找出接近自身 E_target 的结果（|E_ritz - E_target| < 5*delta_e，候选本征值）
    candidates = [r for r in results if r["diff"] < args.tolerance]
    if candidates:
        print(f"\n候选本征值（|E_ritz - E_target| < {args.tolerance:.3e}）：")
        for r in candidates:
            print(f"  E_target={r['E_target']:>8.4f}  E_ritz={r['E_ritz']:>10.6f}  "
                  f"diff={r['diff']:.3e}")
    else:
        print(f"\n未找到与目标相差 < {args.tolerance:.3e} 的候选本征值。")

    # ── 保存 ────────────────────────────────────
    if args.output:
        output_data = {
            "params": {
                "delta_e":    args.delta_e,
                "maxiter":    args.maxiter,
                "n":          args.n,
                "r_cut":      args.r_cut,
                "cube_file":  args.cube_file,
                "params_file":args.params_file,
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至：{args.output}")


if __name__ == "__main__":
    main()
