"""
diagnose_jdqmr.py
详细诊断 JDQMR 迭代行为：H·ψ 乘法次数、重启次数、每轮迭代快照等。

用法
----
python diagnose_jdqmr.py [选项]

常用示例
--------
# 基本诊断（默认参数，target=-0.17，n=64，n_levels=50）
python diagnose_jdqmr.py

# 指定 printLevel（3=收敛过程，5=最详细）
python diagnose_jdqmr.py --print_level 3

# 双网格对比：先跑 N=32，再跑 N=64 并用插值结果作初始猜测
python diagnose_jdqmr.py --n_coarse 32

# 不同 maxBlockSize 对比
python diagnose_jdqmr.py --maxBlockSize 3
"""

import argparse
import sys
import time
import numpy as np
import scipy.sparse.linalg as spla

import ho3d_solvers_v2 as solver
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid


# ──────────────────────────────────────────────
# Matvec 计数包装器（独立计数，与 PRIMME 内部计数交叉验证）
# ──────────────────────────────────────────────

class _CountingOp:
    """包装 LinearOperator，统计实际 matvec 调用次数。"""
    def __init__(self, op):
        self._op = op
        self.count = 0
        self.shape = op.shape
        self.dtype = op.dtype

    def matvec(self, v):
        self.count += 1
        return self._op.matvec(v)

    def as_linear_operator(self):
        return spla.LinearOperator(self.shape, matvec=self.matvec, dtype=self.dtype)


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="JDQMR 迭代诊断：H·ψ 次数、重启、每轮快照",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--target",      type=float, default=-0.17,
                   help="目标能量（默认 -0.17）")
    p.add_argument("--n_levels",    type=int,   default=50,
                   help="求解本征值数（默认 50）")
    p.add_argument("--maxBlockSize",type=int,   default=1,
                   help="PRIMME maxBlockSize（默认 1）")
    p.add_argument("--ncv",         type=int,   default=None,
                   help="PRIMME ncv 子空间维度（默认自动：max(80, 2*n_levels)）")
    p.add_argument("--tol",         type=float, default=1e-6,
                   help="PRIMME 收敛精度（默认 1e-6）")
    p.add_argument("--print_level", type=int,   default=3,
                   help="PRIMME printLevel（0~5，3=收敛过程，5=最详细；默认 3）")
    p.add_argument("--n",           type=int,   default=64,
                   help="细网格每轴点数（默认 64）")
    p.add_argument("--n_coarse",    type=int,   default=None,
                   help="双网格粗网格点数；指定后先在 n_coarse 求解，插值作初始猜测")
    p.add_argument("--r_cut",       type=float, default=7.0,
                   help="Gaussian 截断半径（默认 7.0 Angstrom）")
    p.add_argument("--cube_file",   type=str,   default="localPot.cube")
    p.add_argument("--params_file", type=str,   default="gaussian_fit_params.json")
    p.add_argument("--no_count",    action="store_true",
                   help="不使用 matvec 计数包装（printLevel 输出仍然生效）")
    return p.parse_args()


# ──────────────────────────────────────────────
# 辅助：格式化 PRIMME stats
# ──────────────────────────────────────────────

def _print_stats(stats: dict, t_wall: float):
    print("\n" + "="*60)
    print("  PRIMME 统计信息")
    print("="*60)
    keys_order = [
        ("numMatvecs",          "H·ψ 乘法总次数"),
        ("numRestarts",         "重启次数"),
        ("numOuterIterations",  "外迭代次数"),
        ("numPreconds",         "预条件乘法次数"),
        ("numConverged",        "已收敛本征对数"),
        ("elapsedTime",         "PRIMME 内部计时 (s)"),
        ("estimateMinEVal",     "最小 Ritz 值估计"),
        ("estimateMaxEVal",     "最大 Ritz 值估计"),
        ("estimateLargestSVal", "最大奇异值估计"),
    ]
    for key, label in keys_order:
        if key in stats:
            val = stats[key]
            if isinstance(val, float):
                print(f"  {label:<28}: {val:.6g}")
            else:
                print(f"  {label:<28}: {val}")
    print(f"  {'挂钟总耗时 (s)':<28}: {t_wall:.3f}")

    # rnorms（每个本征对的残差）
    if "rnorms" in stats:
        rnorms = np.asarray(stats["rnorms"])
        print(f"\n  残差 ||Hx-λx|| 统计：")
        print(f"    max = {rnorms.max():.3e}  min = {rnorms.min():.3e}  "
              f"mean = {rnorms.mean():.3e}")

    # 历史快照
    if "hist" in stats and stats["hist"]:
        hist = stats["hist"]
        mv   = np.asarray(hist.get("numMatvecs",   []))
        t    = np.asarray(hist.get("elapsedTime",  []))
        nc   = np.asarray(hist.get("nconv",        []))
        ev   = np.asarray(hist.get("eval",         []))
        rn   = np.asarray(hist.get("resNorm",      []))

        print(f"\n  迭代历史（共 {len(mv)} 条快照）：")
        print(f"  {'#':>4}  {'累计Hψ':>8}  {'time(s)':>8}  "
              f"{'已收敛':>6}  {'首未收敛Eig':>14}  {'resNorm':>10}")
        print(f"  {'-'*60}")
        for i in range(len(mv)):
            ev_str = f"{ev[i]:14.8f}" if i < len(ev) else "              "
            rn_str = f"{rn[i]:10.3e}" if i < len(rn) else "          "
            print(f"  {i:>4}  {int(mv[i]):>8d}  {t[i]:>8.3f}  "
                  f"{int(nc[i]):>6}  {ev_str}  {rn_str}")


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  JDQMR 诊断脚本")
    print("=" * 60)
    print(f"  target      = {args.target}")
    print(f"  n_levels    = {args.n_levels}")
    print(f"  N (fine)    = {args.n}")
    print(f"  n_coarse    = {args.n_coarse}  （{'启用' if args.n_coarse else '禁用'}双网格）")
    print(f"  maxBlockSize= {args.maxBlockSize}")
    print(f"  ncv         = {args.ncv if args.ncv else '自动'}")
    print(f"  tol         = {args.tol:.1e}")
    print(f"  print_level = {args.print_level}")

    # ── 加载势能 ────────────────────────────────
    print(f"\n加载高斯重构势能（N={args.n}）...")
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
        sys.exit(1)
    print(f"  空间范围：[{x[0]:.3f}, {x[-1]:.3f}] Bohr，V ∈ [{V.min():.4f}, {V.max():.4f}]")

    # ── 构建细网格 H 算符 ──────────────────────
    print(f"\n构建 FFT-DVR 算符（N={args.n})...")
    t0 = time.perf_counter()
    H_op_raw, n_un, _ = solver.build_3d_fft_operator(args.n, pot_grid)
    t_build = time.perf_counter() - t0
    print(f"  维度：{n_un}  构建耗时：{t_build:.2f}s")

    # ── 可选：matvec 计数包装 ──────────────────
    if not args.no_count:
        counter = _CountingOp(H_op_raw)
        H_op = counter.as_linear_operator()
        print("  已启用 matvec 计数包装器")
    else:
        counter = None
        H_op = H_op_raw

    # ── 可选：双网格初始猜测 ───────────────────
    v0 = None
    if args.n_coarse is not None:
        print(f"\n双网格：粗网格求解（N={args.n_coarse}）...")
        t0 = time.perf_counter()
        try:
            x_c, y_c, z_c, V_c = builder.build_potential(args.n_coarse)
            pot_coarse = PotentialGrid(x_c, y_c, z_c, V_c, source="gaussian_files")
            ncv_c = args.ncv if args.ncv is not None else max(80, 2 * args.n_levels)
            res_c = solver.solve_ho3d(
                method_spec="fft_dvr:jdqmr",
                N=args.n_coarse,
                n_levels=args.n_levels,
                potential_grid=pot_coarse,
                target=args.target,
                maxBlockSize=args.maxBlockSize,
                primme_print_level=0,
                return_primme_stats=False,
                tol=args.tol,
                ncv=ncv_c,
            )
            t_coarse = time.perf_counter() - t0
            print(f"  粗网格完成，耗时 {t_coarse:.1f}s")
            print(f"  粗网格本征值：{np.round(res_c.evals, 6).tolist()}")
            v0 = solver.interpolate_eigenvector(
                res_c.evecs, N_coarse=args.n_coarse, N_fine=args.n,
                potential_grid=pot_coarse,
            )
            print(f"  插值后 v0 形状：{v0.shape}")
        except Exception as exc:
            print(f"  粗网格求解失败：{exc}，将使用随机初始猜测。")

    # ── 主求解（细网格，带完整诊断）──────────
    print(f"\n{'='*60}")
    print(f"  开始 JDQMR 细网格求解（N={args.n}，printLevel={args.print_level}）")
    print(f"{'='*60}")

    import primme
    ncv_val = args.ncv if args.ncv is not None else max(80, 2 * args.n_levels)

    t_start = time.perf_counter()
    evals, evecs, stats = primme.eigsh(
        H_op,
        k=args.n_levels,
        which=args.target,
        tol=args.tol,
        maxBlockSize=args.maxBlockSize,
        ncv=ncv_val,
        v0=v0,
        method="PRIMME_JDQMR",
        printLevel=args.print_level,
        return_stats=True,
        return_history=True,
    )
    t_wall = time.perf_counter() - t_start

    evals = np.sort(np.asarray(evals, dtype=float))

    # ── 打印本征值 ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"  求得本征值（共 {len(evals)} 个）")
    print(f"{'='*60}")
    for i, e in enumerate(evals):
        print(f"  [{i:02d}]  E = {e:16.10f}")

    # ── 打印统计 ────────────────────────────────
    stats["numConverged"] = len(evals)
    _print_stats(stats, t_wall)

    # ── 独立计数器对比 ──────────────────────────
    if counter is not None:
        print(f"\n  [计数器] 实际 matvec 调用次数：{counter.count}")
        primme_mv = stats.get("numMatvecs", "N/A")
        print(f"  [PRIMME] numMatvecs：{primme_mv}")
        if isinstance(primme_mv, int) and counter.count != primme_mv:
            diff = counter.count - primme_mv
            print(f"  差值 {diff:+d}（PRIMME 可能在初始化阶段有未计入的 matvec）")

    # ── 效率指标 ───────────────────────────────
    n_mv = stats.get("numMatvecs", counter.count if counter else 0)
    if n_mv and t_wall > 0:
        print(f"\n  每次 Hψ 平均耗时：{t_wall / n_mv * 1e3:.3f} ms")
        print(f"  每收敛一个本征对平均 Hψ 次数：{n_mv / max(len(evals), 1):.0f}")


if __name__ == "__main__":
    main()
