"""
scan_jdqmr_fft.py
在高斯重构势能上扫描多个 E_target，使用 FFT-DVR 离散化 + PRIMME JDQMR 求解器。

JDQMR（Jacobi-Davidson QMR）由 PRIMME 库提供，每个目标能量独立调用
primme.eigsh(which=E_target)，可一次返回目标附近的多个本征值。
FFT 算子复杂度 O(N³ log N)，远优于 Sinc-DVR 的 O(N⁴)。

残差输出方式（三种，可独立组合）
----------------------------------
方式①  return_stats=True（始终启用）
        每次求解完成后，从 stats['rnorms'] 读取每个收敛本征对的最终残差范数，
        打印在结果行末，并写入 JSON。

方式②  --monitor
        通过 monitor 回调在每轮外迭代时实时打印当前 Ritz 值与各本征对残差，
        适合观察迭代收敛行为。

方式③  --history
        启用 return_history=True，将完整迭代历史（每轮本征值估计 + 残差序列）
        保存至 JSON；与 --output 配合使用。

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
--verbose                打印 PRIMME 内置迭代细节（printLevel=3）
--monitor                方式②：每轮外迭代实时打印 Ritz 值 + 残差（monitor 回调）
--history                方式③：保存完整迭代历史到 JSON（需配合 --output）

示例
----
# 默认扫描 -0.25 到 0.20，间隔 0.002
python scan_jdqmr_fft.py

# 指定 E_target 列表
python scan_jdqmr_fft.py --e_targets -0.10 -0.05 0.00 0.05

# 每个目标求 3 个本征值，实时查看残差，保存含历史的结果
python scan_jdqmr_fft.py --n_levels 3 --monitor --history --output results_jdqmr.json
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
                   help="打印 PRIMME 内置迭代细节（printLevel=3）")
    p.add_argument("--monitor",     action="store_true",
                   help="方式②：每轮外迭代实时打印 Ritz 值 + 残差（monitor 回调）")
    p.add_argument("--history",     action="store_true",
                   help="方式③：启用 return_history=True，将完整迭代历史保存至 JSON")
    return p.parse_args()


# ──────────────────────────────────────────────
# 残差方式②：monitor 回调工厂
# ──────────────────────────────────────────────

def make_monitor(E_t):
    """
    返回一个 PRIMME monitor 回调。
    PRIMME 在每次外迭代结束时调用：
        monitor(evals, evecs, rnorms, stats, flag)
    其中 rnorms[i] 是第 i 个 Ritz 对的当前残差范数，flag[i]=0 表示已收敛。
    """
    def _monitor(evals, evecs, rnorms, stats, flag):
        if evals is None or rnorms is None:
            return
        it = stats.get("numOuterIterations", "?") if stats else "?"
        ev_str = "  ".join(
            f"{e:10.6f}" for e in evals if e is not None
        )
        rn_str = "  ".join(
            f"{r:.2e}" for r in rnorms if r is not None
        )
        conv_str = "  ".join(
            ("收敛" if (f is not None and f == 0) else "    ")
            for f in (flag if flag is not None else [None] * len(evals))
        )
        print(f"      [it={it:>4}] E=[{ev_str}]  |r|=[{rn_str}]  {conv_str}")
    return _monitor


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
    print(f"  残差输出 ：方式① stats(始终) | 方式② monitor={'开' if args.monitor else '关'}"
          f" | 方式③ history={'开' if args.history else '关'}")

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
              f"{'min|ΔE|':>10}  {'最终|r|(s)':>20}  {'耗时':>7}")
    print(header)
    print(f"  {'-'*100}")

    print_level = 3 if args.verbose else 0
    results = []
    t_scan_start = time.perf_counter()

    for i, E_t in enumerate(targets):
        t0 = time.perf_counter()

        # ── 方式②：monitor 回调 ──────────────────
        monitor_cb = make_monitor(E_t) if args.monitor else None
        if args.monitor:
            print(f"\n  [E_target={E_t:.4f}] 开始迭代（monitor 模式）：")

        try:
            # ── 方式①③ return_stats / return_history ──
            ret = primme.eigsh(
                H_op,
                k=args.n_levels,
                which=float(E_t),
                method="PRIMME_JDQMR",
                tol=args.tol,
                printLevel=print_level,
                return_stats=True,          # 方式①：始终开启，获取 stats['rnorms']
                return_history=args.history, # 方式③：按需开启
                monitor=monitor_cb,          # 方式②：按需开启
            )

            # 解包返回值（return_stats/return_history 会多返回 dict）
            if args.history:
                evals, evecs, stats, history = ret
            else:
                evals, evecs, stats = ret

            evals = np.sort(evals)
            min_diff = float(np.min(np.abs(evals - E_t)))
            closest  = float(evals[np.argmin(np.abs(evals - E_t))])

            # 方式①：最终残差范数（每个收敛本征对一个值）
            final_rnorms = stats.get("rnorms", [])
            if hasattr(final_rnorms, "tolist"):
                final_rnorms = final_rnorms.tolist()

            # 打印 PRIMME stats 中所有收敛相关信息
            _fmt_stat = lambda v: (
                f"{v:.4e}" if isinstance(v, float) else
                (", ".join(f"{x:.4e}" for x in v)
                 if hasattr(v, "__iter__") else str(v))
            )
            stat_fields = [
                ("numOuterIterations", "外迭代次数"),
                ("numRestarts",        "重启次数"),
                ("numMatvecs",         "矩阵向量乘次数"),
                ("numPreconds",        "预条件器次数"),
                ("elapsedTime",        "求解耗时(s)"),
                ("rnorms",             "各本征对最终|r|"),
                ("estimateMinEVal",    "估计 E_min"),
                ("estimateMaxEVal",    "估计 E_max"),
                ("estimateLargestSVal","估计最大奇异值"),
            ]
            print(f"    收敛统计 (E_target={E_t:.4f}):")
            for key, label in stat_fields:
                if key in stats and stats[key] is not None:
                    print(f"      {label:<20}: {_fmt_stat(stats[key])}")

            history_data = None
            if args.history:
                # 将 numpy 数组转为列表以便 JSON 序列化
                history_data = {
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in history.items()
                }

            success  = True
            err_msg  = ""

        except Exception as exc:
            evals        = np.array([float("nan")] * args.n_levels)
            min_diff     = float("nan")
            closest      = float("nan")
            final_rnorms = []
            history_data = None
            success      = False
            err_msg      = str(exc)

        t_it = time.perf_counter() - t0

        results.append({
            "E_target":     float(E_t),
            "evals":        evals.tolist(),
            "closest":      closest,
            "min_diff":     min_diff,
            "final_rnorms": final_rnorms,   # 方式① 数据
            "history":      history_data,    # 方式③ 数据（None 若未启用）
            "success":      success,
            "err_msg":      err_msg,
            "time_s":       t_it,
        })

        evals_str  = "  ".join(f"{e:10.6f}" for e in evals)
        diff_str   = f"{min_diff:.3e}" if success else "  ERROR  "
        rnorm_str  = ("  ".join(f"{r:.2e}" for r in final_rnorms)
                      if final_rnorms else ("N/A" if success else ""))
        print(f"  {E_t:>10.4f}  {evals_str:>40}  {diff_str:>10}  {rnorm_str:>20}  {t_it:>6.1f}s")

        if not success:
            print(f"    !! {err_msg}")

        # 进度提示（每25个）
        if (i + 1) % 25 == 0:
            elapsed   = time.perf_counter() - t_scan_start
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
            rn = ("  ".join(f"{v:.2e}" for v in r["final_rnorms"])
                  if r["final_rnorms"] else "N/A")
            print(f"  E_target={r['E_target']:>8.4f}  E_closest={r['closest']:>10.6f}  "
                  f"diff={r['min_diff']:.3e}  |r|=[{rn}]")
    else:
        print(f"\n未找到与目标相差 < {args.tolerance:.3e} 的候选本征值。")

    # ── 保存 ────────────────────────────────────
    if args.output:
        output_data = {
            "params": {
                "method":         "fft_dvr:jdqmr",
                "n_levels":       args.n_levels,
                "n":              args.n,
                "r_cut":          args.r_cut,
                "cube_file":      args.cube_file,
                "params_file":    args.params_file,
                "primme_tol":     args.tol,
                "residual_modes": {
                    "stats":   True,
                    "monitor": args.monitor,
                    "history": args.history,
                },
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至：{args.output}")


if __name__ == "__main__":
    main()
