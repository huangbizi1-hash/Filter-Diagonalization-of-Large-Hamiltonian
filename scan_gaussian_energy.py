"""
scan_gaussian_energy.py
在高斯重构势能上扫描多个 E_target，利用 fft_dvr:cgmin 计算对应 Ritz 能量，
并通过迭代 Rayleigh-Ritz 对角化精化本征能量。

算法流程
--------
1. 初始扫描：对每个目标能量 E_t 用随机初始猜测运行 CG，保存所有输出向量。
2. Rayleigh-Ritz：把所有向量堆成子空间，QR+SVD 截断后计算 H̃ = Uᵣᵀ H Uᵣ，
   对角化得到 Ritz 值（更准确的本征能量估计）和 Ritz 向量。
3. 迭代精化：以 Ritz 值为新目标、Ritz 向量为初始猜测再次运行 CG，
   将新向量加入子空间，重复 Rayleigh-Ritz，直到 Ritz 值变化 < ritz_tol 或达到最大迭代次数。

参数
----
--delta_e   FLOAT        能量收敛阈值 |ΔE| （默认 5e-4）
--maxiter   INT          最大 CG 迭代次数   （默认 2000）
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
--max_ritz_iters INT     Rayleigh-Ritz 最大精化轮次（默认 3，0 = 仅初始扫描）
--ritz_tol  FLOAT        Ritz 值收敛阈值（默认 1e-5）
--svd_tol   FLOAT        子空间 SVD 相对截断阈值（默认 1e-6）
--ritz_margin FLOAT      Ritz 精化阶段的能量扩展边距（默认 3×e_step）

示例
----
# 默认扫描 -0.25 到 0.20，间隔 0.002，ΔE<5e-4，最多 3 轮 Ritz 精化
python scan_gaussian_energy.py

# 自定义 ΔE 和迭代次数
python scan_gaussian_energy.py --delta_e 1e-4 --maxiter 3000

# 指定 E_target 列表，不做 Ritz 精化
python scan_gaussian_energy.py --e_targets -0.10 -0.05 0.00 0.05 --max_ritz_iters 0

# 保存结果
python scan_gaussian_energy.py --output results.json
"""

import argparse
import json
import sys
import time
import numpy as np
from scipy.linalg import eigh

import ho3d_solvers_v2 as solver
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid


# ──────────────────────────────────────────────
# Rayleigh-Ritz 子空间对角化
# ──────────────────────────────────────────────

def _rayleigh_ritz(Hv, vectors, svd_tol_rel=1e-6):
    """
    在向量列表张成的子空间中做 Rayleigh-Ritz 对角化（参考 fft_code/rayleigh_ritz.py）。

    Parameters
    ----------
    Hv          : callable, v -> H@v（matvec）
    vectors     : list of 1D arrays，每个向量长度相同
    svd_tol_rel : 相对 SVD 截断阈值（相对于最大奇异值）

    Returns
    -------
    ritz_vals : 1D array, 升序排列
    ritz_vecs : (n, r) array，列为 Ritz 向量（单位化）
    rank      : 保留子空间维数 r
    """
    U = np.column_stack(vectors)           # (n, k)
    norms = np.linalg.norm(U, axis=0)
    mask = norms > 0.0
    U = U[:, mask] / norms[mask]

    # QR 正交化
    Q, R = np.linalg.qr(U, mode='reduced')

    # SVD 截断：保留线性无关方向
    _, sigma, _ = np.linalg.svd(R, full_matrices=False)
    r = int(np.sum(sigma > svd_tol_rel * sigma[0]))
    r = max(r, 1)
    Ur = Q[:, :r]

    # H̃ = Uᵣᵀ H Uᵣ
    HUr = np.zeros_like(Ur)
    for i in range(r):
        HUr[:, i] = Hv(Ur[:, i])

    H_tilde = Ur.T @ HUr
    H_tilde = 0.5 * (H_tilde + H_tilde.T)   # 数值对称化

    ritz_vals, W = eigh(H_tilde)             # 升序
    ritz_vecs = Ur @ W                        # 列 = Ritz 向量
    return ritz_vals, ritz_vecs, r


def _eig_residuals(Hv, ritz_vals, ritz_vecs):
    """计算每个 Ritz 对的本征残差 ||H v - E v||。"""
    res = []
    for i, E in enumerate(ritz_vals):
        v = ritz_vecs[:, i]
        r = Hv(v) - E * v
        res.append(float(np.linalg.norm(r)))
    return res


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="扫描高斯重构势能的多个目标能量（fft_dvr:cgmin + 迭代 Rayleigh-Ritz）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--delta_e",       type=float, default=5e-4,
                   help="CG 能量收敛阈值 |ΔE|（默认 5e-4）")
    p.add_argument("--tolerance",     type=float, default=0.003,
                   help="候选本征值筛选阈值 |E_ritz-E_target|（默认 0.003）")
    p.add_argument("--maxiter",       type=int,   default=2000,
                   help="CG 最大迭代次数（默认 2000）")
    p.add_argument("--e_targets",     type=float, nargs="+", default=None,
                   help="显式指定 E_target 列表（指定后忽略 --e_min/--e_max/--e_step）")
    p.add_argument("--e_min",         type=float, default=-0.25,
                   help="扫描起始能量（默认 -0.25）")
    p.add_argument("--e_max",         type=float, default=0.20,
                   help="扫描终止能量（含，默认 0.20）")
    p.add_argument("--e_step",        type=float, default=0.002,
                   help="扫描间隔（默认 0.002）")
    p.add_argument("--n",             type=int,   default=64,
                   help="每轴网格点数（默认 64）")
    p.add_argument("--r_cut",         type=float, default=7.0,
                   help="Gaussian 截断半径（Angstrom，默认 7.0）")
    p.add_argument("--cube_file",     type=str,   default="localPot.cube",
                   help="势能 cube 文件路径（默认 localPot.cube）")
    p.add_argument("--params_file",   type=str,   default="gaussian_fit_params.json",
                   help="高斯拟合参数文件（默认 gaussian_fit_params.json）")
    p.add_argument("--output",        type=str,   default=None,
                   help="结果保存路径（JSON，默认不保存）")
    p.add_argument("--verbose",       action="store_true",
                   help="打印每个目标能量的 CG 迭代细节")
    # Ritz 精化参数
    p.add_argument("--max_ritz_iters", type=int,  default=3,
                   help="Rayleigh-Ritz 最大精化轮次（默认 3，0 = 仅初始扫描）")
    p.add_argument("--ritz_tol",      type=float, default=1e-5,
                   help="Ritz 值收敛阈值：相邻轮 max|ΔE_ritz|（默认 1e-5）")
    p.add_argument("--svd_tol",       type=float, default=1e-6,
                   help="子空间 SVD 相对截断阈值（默认 1e-6）")
    p.add_argument("--ritz_margin",   type=float, default=None,
                   help="Ritz 精化阶段的能量扩展边距（默认 3×e_step）")
    return p.parse_args()


# ──────────────────────────────────────────────
# 单次 CG 扫描（给定目标列表与初始猜测列表）
# ──────────────────────────────────────────────

def _run_cg_pass(H_op, targets, x0_list, args, rng, pass_label):
    """
    对每个目标能量运行 CG，返回 (results_list, vectors_list)。

    x0_list : None（随机初始化）或与 targets 等长的初始猜测列表。
    """
    n_un = H_op.shape[0]
    print(f"\n  {'E_target':>10}  {'E_ritz':>12}  {'|E_ritz-E_t|':>13}  "
          f"{'eig_res':>10}  {'fold_res':>10}  "
          f"{'n_iter':>7}  {'收敛':>5}  {'原因':<20}  {'耗时':>7}")
    print(f"  {'-'*115}")

    results = []
    vectors = []
    t_pass_start = time.perf_counter()

    for i, E_t in enumerate(targets):
        t0 = time.perf_counter()

        if x0_list is None:
            x0 = rng.standard_normal(n_un)
        else:
            x0 = x0_list[i]

        cg_res = solver.cg_minimize_folded(
            H_op, float(E_t), x0,
            maxiter=args.maxiter,
            energy_tol=args.delta_e,
            verbose=args.verbose,
        )

        t_it = time.perf_counter() - t0
        E_r   = cg_res["E_ritz"]
        diff  = abs(E_r - float(E_t))
        eig_res  = cg_res["eig_res"]
        fold_res = cg_res["fold_res"]

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
        vectors.append(cg_res["x"])

        conv_mark = "✓" if cg_res["converged"] else "✗"
        print(f"  {E_t:>10.4f}  {E_r:>12.6f}  {diff:>13.4e}  "
              f"{eig_res:>10.3e}  {fold_res:>10.3e}  "
              f"{cg_res['n_iter']:>7d}  {conv_mark:>5}  "
              f"{cg_res['conv_reason']:<20}  {t_it:>6.1f}s")

        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t_pass_start
            remaining = elapsed / (i + 1) * (len(targets) - i - 1)
            print(f"  -- {i+1}/{len(targets)} 完成，已用 {elapsed:.0f}s，"
                  f"预计剩余 {remaining:.0f}s --")

    t_pass = time.perf_counter() - t_pass_start
    n_conv = sum(r["converged"] for r in results)
    print(f"  {pass_label} 完成：{len(targets)} 目标，{n_conv} 收敛，耗时 {t_pass:.1f}s")
    return results, vectors


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 构建 E_target 列表 ──────────────────────
    if args.e_targets is not None:
        targets = np.array(sorted(args.e_targets))
    else:
        n_pts = int(round((args.e_max - args.e_min) / args.e_step)) + 1
        targets = np.round(np.linspace(args.e_min, args.e_max, n_pts), 8)

    ritz_margin = args.ritz_margin if args.ritz_margin is not None else 3 * args.e_step

    print(f"扫描参数：")
    print(f"  目标能量：{len(targets)} 个，[{targets[0]:.4f}, {targets[-1]:.4f}]，间隔 {args.e_step}")
    print(f"  ΔE 阈值 ：{args.delta_e}，最大 CG 迭代：{args.maxiter}")
    print(f"  Ritz 精化：最多 {args.max_ritz_iters} 轮，收敛阈值 {args.ritz_tol:.1e}")
    print(f"  SVD 截断 ：{args.svd_tol:.1e}（相对），Ritz 边距 {ritz_margin:.4f}")
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
        sys.exit(1)

    print(f"  空间范围：[{x[0]:.3f}, {x[-1]:.3f}] Bohr")
    print(f"  V 范围  ：[{V.min():.4f}, {V.max():.4f}] Hartree")

    # ── 预构建 FFT 算符 ──────────────────────────
    print(f"\n构建 FFT-DVR 哈密顿量算符...")
    t_build_start = time.perf_counter()
    H_op, n_un, _ = solver.build_3d_fft_operator(args.n, pot_grid)
    t_build = time.perf_counter() - t_build_start
    print(f"  算符维度：{n_un}  构建耗时：{t_build:.2f}s")

    # matvec 函数（供 Rayleigh-Ritz 使用）
    Hv = H_op.matvec

    rng = np.random.default_rng(42)
    t_total_start = time.perf_counter()

    # ── 初始扫描（第 0 轮）──────────────────────
    print(f"\n=== 初始扫描（第 0 轮，{len(targets)} 个目标） ===")
    all_results_by_pass = []
    all_vectors = []

    pass_results, pass_vectors = _run_cg_pass(
        H_op, targets, None, args, rng, pass_label="初始扫描"
    )
    all_results_by_pass.append({
        "pass": 0,
        "type": "initial_scan",
        "n_targets": len(targets),
        "cg_results": pass_results,
    })
    all_vectors.extend(pass_vectors)

    # ── Rayleigh-Ritz 精化循环 ──────────────────
    ritz_vals_prev = None
    ritz_vals_final = None
    ritz_vecs_final = None
    ritz_residuals_final = None
    ritz_summary = []

    for ritz_iter in range(1, args.max_ritz_iters + 1):
        if len(all_vectors) < 2:
            print("\n子空间不足 2 个向量，跳过 Rayleigh-Ritz。")
            break

        print(f"\n--- Rayleigh-Ritz（子空间维数 {len(all_vectors)}） ---")
        t_rr = time.perf_counter()
        ritz_vals, ritz_vecs, rank = _rayleigh_ritz(Hv, all_vectors, args.svd_tol)
        t_rr = time.perf_counter() - t_rr
        ritz_res = _eig_residuals(Hv, ritz_vals, ritz_vecs)

        print(f"  保留秩 r = {rank}，耗时 {t_rr:.1f}s")
        print(f"  Ritz 值（前 20）：{np.round(ritz_vals[:20], 6).tolist()}")

        # 收敛判断
        converged_ritz = False
        max_diff = None
        if ritz_vals_prev is not None:
            n_cmp = min(len(ritz_vals), len(ritz_vals_prev))
            max_diff = float(np.max(np.abs(ritz_vals[:n_cmp] - ritz_vals_prev[:n_cmp])))
            print(f"  Ritz 变化：max|ΔE_ritz| = {max_diff:.2e}  （阈值 {args.ritz_tol:.1e}）")
            if max_diff < args.ritz_tol:
                converged_ritz = True

        ritz_summary.append({
            "ritz_iter": ritz_iter,
            "subspace_size": len(all_vectors),
            "rank": rank,
            "ritz_vals": ritz_vals.tolist(),
            "ritz_residuals": ritz_res,
            "max_diff_prev": max_diff,
            "time_s": t_rr,
        })
        ritz_vals_final = ritz_vals
        ritz_vecs_final = ritz_vecs
        ritz_residuals_final = ritz_res

        if converged_ritz:
            print(f"  Ritz 值收敛（max|ΔE|={max_diff:.2e} < {args.ritz_tol:.1e}），停止精化。")
            break

        ritz_vals_prev = ritz_vals.copy()

        # 选取范围内的 Ritz 值作为新目标
        e_lo = targets[0]  - ritz_margin
        e_hi = targets[-1] + ritz_margin
        in_range = (ritz_vals >= e_lo) & (ritz_vals <= e_hi)
        new_targets = ritz_vals[in_range]
        new_x0_list = [ritz_vecs[:, j] for j in np.where(in_range)[0]]

        if len(new_targets) == 0:
            print(f"  扫描范围 [{e_lo:.4f}, {e_hi:.4f}] 内无 Ritz 值，停止精化。")
            break

        print(f"\n=== Ritz 精化第 {ritz_iter} 轮（{len(new_targets)} 个目标） ===")
        pass_results, pass_vectors = _run_cg_pass(
            H_op, new_targets, new_x0_list, args, rng,
            pass_label=f"Ritz 精化第 {ritz_iter} 轮"
        )
        all_results_by_pass.append({
            "pass": ritz_iter,
            "type": "ritz_refinement",
            "n_targets": len(new_targets),
            "cg_results": pass_results,
        })
        all_vectors.extend(pass_vectors)

    # 若从未做过 Ritz（max_ritz_iters=0），也做一次最终 Ritz 汇总
    if ritz_vals_final is None and len(all_vectors) >= 2:
        print(f"\n--- 最终 Rayleigh-Ritz（子空间维数 {len(all_vectors)}） ---")
        ritz_vals_final, ritz_vecs_final, rank = _rayleigh_ritz(Hv, all_vectors, args.svd_tol)
        ritz_residuals_final = _eig_residuals(Hv, ritz_vals_final, ritz_vecs_final)
        print(f"  保留秩 r = {rank}")
        print(f"  Ritz 值（前 20）：{np.round(ritz_vals_final[:20], 6).tolist()}")
        ritz_summary.append({
            "ritz_iter": 0,
            "subspace_size": len(all_vectors),
            "rank": rank,
            "ritz_vals": ritz_vals_final.tolist(),
            "ritz_residuals": ritz_residuals_final,
            "max_diff_prev": None,
            "time_s": 0.0,
        })

    t_total = time.perf_counter() - t_total_start

    # ── 汇总 ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"全部完成，总耗时 {t_total:.1f}s，子空间大小 {len(all_vectors)}")

    if ritz_vals_final is not None:
        # 筛选扫描范围内的 Ritz 值
        e_lo = targets[0]
        e_hi = targets[-1]
        in_range = (ritz_vals_final >= e_lo) & (ritz_vals_final <= e_hi)
        candidates = ritz_vals_final[in_range]
        cand_res   = [ritz_residuals_final[j] for j in np.where(in_range)[0]]
        if len(candidates) > 0:
            print(f"\n扫描范围 [{e_lo:.4f}, {e_hi:.4f}] 内的 Ritz 本征值（{len(candidates)} 个）：")
            print(f"  {'E_ritz':>12}  {'||Hv-Ev||':>12}")
            for E, r in zip(candidates, cand_res):
                print(f"  {E:>12.6f}  {r:>12.3e}")
        else:
            print("扫描范围内未找到 Ritz 本征值。")

    # 旧版候选本征值筛选（与初始扫描结果对比）
    flat_results = [r for p in all_results_by_pass for r in p["cg_results"]]
    candidates_old = [r for r in flat_results if r["diff"] < args.tolerance]
    if candidates_old:
        print(f"\n（参考）CG 候选本征值（|E_ritz - E_target| < {args.tolerance:.3e}，共 {len(candidates_old)} 个）：")
        seen = set()
        for r in candidates_old:
            key = round(r["E_ritz"], 5)
            if key not in seen:
                seen.add(key)
                print(f"  E_target={r['E_target']:>8.4f}  E_ritz={r['E_ritz']:>10.6f}  diff={r['diff']:.3e}")

    # ── 保存 ────────────────────────────────────
    if args.output:
        output_data = {
            "params": {
                "delta_e":        args.delta_e,
                "maxiter":        args.maxiter,
                "e_min":          float(targets[0]),
                "e_max":          float(targets[-1]),
                "e_step":         args.e_step,
                "n":              args.n,
                "r_cut":          args.r_cut,
                "cube_file":      args.cube_file,
                "params_file":    args.params_file,
                "max_ritz_iters": args.max_ritz_iters,
                "ritz_tol":       args.ritz_tol,
                "svd_tol":        args.svd_tol,
                "ritz_margin":    ritz_margin,
                "total_time_s":   t_total,
            },
            "passes":       all_results_by_pass,
            "ritz_summary": ritz_summary,
            "final_ritz_vals": ritz_vals_final.tolist() if ritz_vals_final is not None else [],
            "final_ritz_residuals": ritz_residuals_final if ritz_residuals_final is not None else [],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至：{args.output}")


if __name__ == "__main__":
    main()
