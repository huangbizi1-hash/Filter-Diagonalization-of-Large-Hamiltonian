"""
scan_jdqmr_fft.py
在高斯重构势能上用 FFT-DVR + PRIMME JDQMR 求解本征值。

支持两种模式：
1) 扫描模式（兼容旧流程）：对多个 E_target 分别求解；
2) 直接模式（新增，推荐）：一次性求出最低 n_levels 个本征值，并筛选 E <= energy_cut。

示例
----
# 直接模式：一次求 40 个能级，筛选 E <= -0.18
python scan_jdqmr_fft.py --direct_below --energy_cut -0.18 --n_levels 40

# 扫描模式：兼容旧做法
python scan_jdqmr_fft.py --e_min -0.25 --e_max 0.2 --e_step 0.002 --n_levels 3
"""

import argparse
import json
import sys
import time
from datetime import datetime
import numpy as np

import ho3d_solvers_v2 as solver
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid


def parse_args():
    p = argparse.ArgumentParser(
        description="FFT-DVR + JDQMR 求解高斯势能本征值（支持扫描 / 直接阈值筛选）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 模式控制
    p.add_argument("--direct_below", action="store_true",
                   help="直接模式：一次求解最低 n_levels 个本征值，再筛选 E<=energy_cut")
    p.add_argument("--energy_cut", type=float, default=-0.18,
                   help="直接模式筛选阈值，保留 E<=energy_cut（默认 -0.18）")

    # 扫描模式参数（不启用 direct_below 时使用）
    p.add_argument("--e_targets", type=float, nargs="+", default=None,
                   help="显式指定 E_target 列表（指定后忽略 --e_min/--e_max/--e_step）")
    p.add_argument("--e_min", type=float, default=-0.25,
                   help="扫描起始能量（默认 -0.25）")
    p.add_argument("--e_max", type=float, default=0.20,
                   help="扫描终止能量（含，默认 0.20）")
    p.add_argument("--e_step", type=float, default=0.002,
                   help="扫描间隔（默认 0.002）")
    p.add_argument("--tolerance", type=float, default=0.003,
                   help="扫描模式候选阈值 |E-E_target|（默认 0.003）")

    # solve_ho3d 关键参数（用户要求可调）
    p.add_argument("--n_levels", type=int, default=1,
                   help="每次 solve_ho3d 求解的本征值数（默认 1）")
    p.add_argument("--ncv", type=int, default=None,
                   help="PRIMME 子空间维度 ncv（默认由 solver 自动设置）")
    p.add_argument("--maxBlockSize", type=int, default=1,
                   help="PRIMME maxBlockSize（默认 1）")

    # 网格与势能参数
    p.add_argument("--n", type=int, default=64,
                   help="每轴网格点数（默认 64）")
    p.add_argument("--r_cut", type=float, default=7.0,
                   help="Gaussian 截断半径（Angstrom，默认 7.0）")
    p.add_argument("--cube_file", type=str, default="localPot.cube",
                   help="势能 cube 文件路径（默认 localPot.cube）")
    p.add_argument("--params_file", type=str, default="gaussian_fit_params.json",
                   help="高斯拟合参数文件（默认 gaussian_fit_params.json）")
    p.add_argument("--tol", type=float, default=1e-6,
                   help="PRIMME 收敛精度 tol（默认 1e-6）")

    p.add_argument("--output", type=str, default=None,
                   help="JSON 输出路径（默认不保存）")
    p.add_argument("--job_title", type=str, default="",
                   help="任务标题，写入 JSON 便于后续识别（默认空）")
    return p.parse_args()


def build_potential(args):
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
        print("请确保 cube 文件和参数文件在当前目录或通过 --cube_file/--params_file 指定。", file=sys.stderr)
        sys.exit(1)

    print(f"  空间范围：[{x[0]:.3f}, {x[-1]:.3f}] Bohr")
    print(f"  V 范围  ：[{V.min():.4f}, {V.max():.4f}] Hartree")
    return pot_grid


def solve_once(pot_grid, args, target=None):
    kwargs = {"tol": args.tol}
    if args.ncv is not None:
        kwargs["ncv"] = args.ncv

    return solver.solve_ho3d(
        method_spec="fft_dvr:jdqmr",
        N=args.n,
        n_levels=args.n_levels,
        potential_grid=pot_grid,
        target=target,
        maxBlockSize=args.maxBlockSize,
        **kwargs,
    )


def run_direct_mode(args, pot_grid):
    print("\n模式：直接模式（一次性求多个能级）")
    print(f"  求解 n_levels={args.n_levels}，筛选 E <= {args.energy_cut:.6f}")
    t0 = time.perf_counter()
    res = solve_once(pot_grid, args, target=None)
    dt = time.perf_counter() - t0

    evals = np.array(res.evals, dtype=float)
    selected = evals[evals <= args.energy_cut]

    print(f"  实际求得本征值数：{len(evals)}")
    print(f"  满足 E <= {args.energy_cut:.6f} 的本征值数：{len(selected)}")
    for i, e in enumerate(selected):
        print(f"    [{i:02d}] E = {e:.8f}")
    print(f"  求解耗时：{dt:.2f}s")

    return {
        "mode": "direct_below",
        "energy_cut": args.energy_cut,
        "all_evals": evals.tolist(),
        "selected_evals": selected.tolist(),
        "solver_meta": res.meta,
        "time_s": dt,
    }


def run_scan_mode(args, pot_grid):
    print("\n模式：扫描模式（兼容旧流程）")
    if args.e_targets is not None:
        targets = np.array(sorted(args.e_targets))
    else:
        n_pts = int(round((args.e_max - args.e_min) / args.e_step)) + 1
        targets = np.round(np.linspace(args.e_min, args.e_max, n_pts), 8)

    print(f"  扫描目标：{len(targets)} 个，[{targets[0]:.4f}, {targets[-1]:.4f}]")
    print(f"  每个目标求 n_levels={args.n_levels}")
    print(f"  tolerance={args.tolerance:.3e}")
    print(f"\n  {'E_target':>10}  {'E_ritz(s)':>40}  {'min|ΔE|':>10}  {'耗时':>7}")
    print(f"  {'-'*80}")

    results = []
    t_scan = time.perf_counter()
    for E_t in targets:
        t0 = time.perf_counter()
        try:
            res = solve_once(pot_grid, args, target=float(E_t))
            evals = np.array(res.evals, dtype=float)
            min_diff = float(np.min(np.abs(evals - E_t)))
            closest = float(evals[np.argmin(np.abs(evals - E_t))])
            success, err_msg = True, ""
            meta = res.meta
        except Exception as exc:
            evals = np.array([float("nan")] * args.n_levels)
            min_diff = float("nan")
            closest = float("nan")
            success, err_msg = False, str(exc)
            meta = {}

        dt = time.perf_counter() - t0
        evals_str = "  ".join(f"{e:10.6f}" for e in evals)
        diff_str = f"{min_diff:.3e}" if success else "  ERROR  "
        print(f"  {E_t:>10.4f}  {evals_str:>40}  {diff_str:>10}  {dt:>6.1f}s")
        if not success:
            print(f"    !! {err_msg}")

        results.append({
            "E_target": float(E_t),
            "evals": evals.tolist(),
            "closest": closest,
            "min_diff": min_diff,
            "success": success,
            "err_msg": err_msg,
            "time_s": dt,
            "solver_meta": meta,
        })

    total = time.perf_counter() - t_scan
    candidates = [r for r in results if r["success"] and r["min_diff"] < args.tolerance]
    print(f"\n扫描完成：{len(targets)} 个目标，总耗时 {total:.1f}s")
    print(f"候选数量（|E-E_target|<{args.tolerance:.3e}）：{len(candidates)}")

    return {
        "mode": "scan_targets",
        "targets": targets.tolist(),
        "tolerance": args.tolerance,
        "results": results,
        "candidates": candidates,
        "time_s": total,
    }


def main():
    args = parse_args()
    start_dt = datetime.now()
    print("运行参数：")
    print(f"  method=fft_dvr:jdqmr, N={args.n}, n_levels={args.n_levels}, tol={args.tol}")
    print(f"  ncv={args.ncv}, maxBlockSize={args.maxBlockSize}")
    if args.job_title:
        print(f"  job_title={args.job_title}")

    pot_grid = build_potential(args)
    output = run_direct_mode(args, pot_grid) if args.direct_below else run_scan_mode(args, pot_grid)

    if args.output:
        output_data = {
            "job_title":      args.job_title,
            "start_datetime": start_dt.isoformat(),
            "end_datetime":   datetime.now().isoformat(),
            "params":         vars(args),
            "result":         output,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至：{args.output}")


if __name__ == "__main__":
    main()

