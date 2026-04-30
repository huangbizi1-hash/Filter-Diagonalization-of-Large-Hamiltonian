"""
compare_fd_filter.py
比较不同阶数有限差分动能算符对滤波对角化（Filter Diagonalization）性能的影响。

默认使用 localPot.cube（N=64）；通过 --qd-radius R 切换到
QD_Outputs/QD_R{R}.cube 对应的 QD 网格（N 由 cube 文件决定）。

有限差分阶数：2, 4, 6, 8, 10, 12, 14, 16, 18, 20
参考基准：FFT 算符（精确动能）

采用与 main.py apply_filter_H_all 相同的策略：
  所有 El 共享 Newton 基底向量，H-apply 次数为 nc（而非 ms×nc）。
通用 H_op.matvec 接口，不依赖 pyfftw/T_k_diagonal，
可直接接受有限差分或 FFT 算符。

用法：
    python compare_fd_filter.py                         # localPot.cube，N=64
    python compare_fd_filter.py --qd-radius 17          # QD_Outputs/QD_R17.cube
    python compare_fd_filter.py --El -0.15 --nc 3000    # 改滤波中心和阶数
    python compare_fd_filter.py --fd-orders 4 8 12 16   # 只测这几阶
    python compare_fd_filter.py --fft-only              # 仅跑 FFT 基准
    python compare_fd_filter.py --n-random 32 --svd-tol 1e-4

输出（fd_results/）
    compare_fd_filter_{TAG}_TIMESTAMP.json
    compare_fd_filter_{TAG}_TIMESTAMP.md
    compare_fd_filter_{TAG}_TIMESTAMP.png
"""

import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.interpolate import RegularGridInterpolator

from ho3d_solvers_v2 import (
    build_3d_fft_operator,
    build_3d_fd_operator,
    FD_STENCILS,
)
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid
from fft_code.params       import PhysParams
from fft_code.filter_coeff import build_filter_coefficients
from fft_code.filter_coeff import _filt_func_gaussian

_ALL_FD_ORDERS = sorted(FD_STENCILS.keys())   # [2, 4, 6, …, 20]

# ──────────────────────────────────────────────
# 读取 cube 文件头（N, d, origin）
# ──────────────────────────────────────────────

def _read_cube_header(cube_path: str):
    with open(cube_path) as f:
        f.readline(); f.readline()
        parts  = f.readline().split()
        origin = float(parts[1])
        line4  = f.readline().split()
        N_c    = int(line4[0])
        d_c    = float(line4[1])
    return N_c, d_c, origin


# ──────────────────────────────────────────────
# 通用 apply_filter_H_all
# ──────────────────────────────────────────────

def _apply_filter_H_all(H_op, psi_flat, nodes, an_, par_):
    """
    f_i(H)|ψ⟩，对所有 ms 个 El_i 共享 Newton 基底向量。
    H-apply 次数 = nc（而非 ms×nc）。

    返回 (ms, n_grid)。
    """
    ms_, nc_ = an_.shape
    results  = an_[:, 0:1] * psi_flat[None, :]
    psi_prev = psi_flat.copy()

    for j in range(1, nc_):
        H_psi    = H_op.matvec(psi_prev)
        psi_curr = ((4.0 / par_.dE) * (H_psi - par_.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        results += an_[:, j:j+1] * psi_curr[None, :]
        psi_prev = psi_curr

    return results


# ──────────────────────────────────────────────
# Rayleigh-Ritz（通用 H_op）
# ──────────────────────────────────────────────

def _rayleigh_ritz(basis_mat, H_op, svd_tol, max_energies):
    """SVD + Rayleigh-Ritz，返回排序特征值（最多 max_energies 个）及有效秩 r。"""
    norms = np.linalg.norm(basis_mat, axis=0)
    mask  = norms > 0
    B     = basis_mat[:, mask] / norms[None, mask]

    Q, R         = np.linalg.qr(B, mode="reduced")
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    r = max(1, int(np.sum(sigma > svd_tol)))
    print(f"    SVD rank r={r}  (sigma_max={sigma[0]:.3e}, "
          f"sigma_r={sigma[r-1]:.3e})")
    Ur = (Q @ U1)[:, :r]

    H_tilde = np.zeros((r, r), dtype=float)
    for j in range(r):
        HUj           = H_op.matvec(Ur[:, j])
        H_tilde[:, j] = Ur.T @ HUj

    evals, _ = eigh(H_tilde)
    return np.sort(evals.real)[:max_energies], r


# ──────────────────────────────────────────────
# 单次滤波对角化
# ──────────────────────────────────────────────

def _run_filter(H_op, label, N, nc_true, ms, n_random, samp, an, par,
                svd_tol, max_energies, rng):
    n_grid = N ** 3
    print(f"\n  [{label}]  nc={nc_true}, n_random={n_random}")
    t0 = time.perf_counter()
    try:
        filtered_psi_matrix = np.zeros((ms * n_random, n_grid), dtype=float)
        for i in range(n_random):
            v        = rng.standard_normal(n_grid)
            psi_rand = v / np.linalg.norm(v)
            psi_filt_all = _apply_filter_H_all(H_op, psi_rand, samp, an, par)
            for ie in range(ms):
                psi_filt = psi_filt_all[ie]
                norm     = np.linalg.norm(psi_filt)
                if norm > 0:
                    filtered_psi_matrix[ie * n_random + i] = psi_filt / norm
            if (i + 1) % 10 == 0:
                print(f"    filtered {i+1}/{n_random} ...", flush=True)

        n_H_filter = nc_true * n_random

        energies, r = _rayleigh_ritz(
            filtered_psi_matrix.T, H_op, svd_tol, max_energies)

        n_H_total = n_H_filter + r
        t_wall    = time.perf_counter() - t0
        eval0     = float(energies[0]) if len(energies) > 0 else float("nan")
        success   = True
        err_msg   = ""
        print(f"    E[0]={eval0:.8f}  T={t_wall:.2f}s  "
              f"N_H={n_H_total} (filter={n_H_filter}, RR={r})")
    except Exception as exc:
        t_wall    = time.perf_counter() - t0
        eval0     = float("nan")
        n_H_total = -1
        n_H_filter = -1
        success   = False
        err_msg   = str(exc)
        energies  = []
        r         = 0
        print(f"    FAILED: {err_msg}")

    return dict(label=label, eval=eval0, t_wall=t_wall,
                n_H=n_H_total, n_H_filter=n_H_filter,
                rr_rank=r,
                energies=(energies.tolist() if hasattr(energies, "tolist")
                          else list(energies)),
                success=success, err_msg=err_msg)


# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────

def _make_plot(results, eval_ref, N, TAG, EL, nc_true, n_random, plot_path):
    succ = [r for r in results if r["success"] and r["fd_order"] is not None]
    if not succ:
        fft_row = results[0] if results and results[0]["success"] else None
        fig, ax = plt.subplots(figsize=(6, 4))
        if fft_row:
            ax.bar(["FFT"], [fft_row["t_wall"]], color="steelblue")
            ax.set_ylabel("Wall time (s)", fontsize=11)
            ax.set_title(
                f"Filter Diag (FFT only)\n"
                f"{TAG}, N={N}, El={EL}, nc={nc_true}, n_random={n_random}",
                fontsize=10)
            ax.text(0, fft_row["t_wall"] * 0.5,
                    f"E[0]={fft_row['eval']:.6f}\nRR rank={fft_row['rr_rank']}",
                    ha="center", va="center", fontsize=10, color="white")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    orders  = [r["fd_order"] for r in succ]
    evals   = [r["eval"]     for r in succ]
    t_walls = [r["t_wall"]   for r in succ]
    d_evals = [abs(e - eval_ref) for e in evals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.semilogy(orders, d_evals, "o-", color="steelblue")
    ax.set_xlabel("FD order", fontsize=11)
    ax.set_ylabel("|E_FD - E_FFT| (Hartree)", fontsize=11)
    ax.set_title("Eigenvalue error vs FD order\n(Filter Diagonalization)", fontsize=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.set_xticks(orders)

    ax = axes[1]
    ax.plot(orders, t_walls, "o-", color="darkorange")
    if results[0]["success"]:
        ax.axhline(results[0]["t_wall"], color="gray", lw=1.2,
                   ls="--", label="FFT ref")
        ax.legend(fontsize=9)
    ax.set_xlabel("FD order", fontsize=11)
    ax.set_ylabel("Wall time (s)", fontsize=11)
    ax.set_title("Wall time vs FD order\n(Filter Diagonalization)", fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_xticks(orders)

    fig.suptitle(
        f"Filter Diag: FD vs FFT  "
        f"({TAG}, N={N}, El={EL}, nc={nc_true}, Gaussian, n_random={n_random})",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FD vs FFT filter diagonalization — compare filter accuracy across FD orders on QD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 默认：localPot.cube，N=64，所有 FD 阶数
  python compare_fd_filter.py

  # 指定 QD 半径（读取 QD_Outputs/QD_R17.cube）
  python compare_fd_filter.py --qd-radius 17

  # 改滤波中心和 Newton 阶数
  python compare_fd_filter.py --El -0.15 --nc 3000

  # 只测部分 FD 阶数
  python compare_fd_filter.py --fd-orders 4 8 12 16 20

  # 只跑 FFT 参考，跳过所有 FD
  python compare_fd_filter.py --fft-only

  # 调随机初态数量和 SVD 截断
  python compare_fd_filter.py --n-random 32 --svd-tol 1e-4
""",
    )

    # ---- 势能 ----
    parser.add_argument("--qd-radius", type=int, default=None,
                        metavar="R",
                        help="QD 半径（Bohr），读取 QD_Outputs/QD_R{R}.cube；"
                             "不指定则使用 localPot.cube（N=64）")
    parser.add_argument("--cube-file", type=str, default="localPot.cube",
                        metavar="PATH",
                        help="默认势能 cube 文件（default: localPot.cube）")
    parser.add_argument("--params-file", type=str, default="gaussian_fit_params.json",
                        metavar="PATH",
                        help="高斯拟合参数 JSON（default: gaussian_fit_params.json）")
    parser.add_argument("--r-cut", type=float, default=7.0,
                        metavar="Bohr",
                        help="高斯截断半径（default: 7.0 Bohr）")

    # ---- 滤波窗口 ----
    parser.add_argument("--El", type=float, default=-0.17,
                        metavar="Ha",
                        help="滤波中心能量（Hartree，default: -0.17）")
    parser.add_argument("--nc", type=int, default=5000,
                        metavar="INT",
                        help="Newton 节点数初始估计（default: 5000）")
    parser.add_argument("--dE", type=float, default=50.0,
                        metavar="Ha",
                        help="滤波窗口宽度（Hartree，default: 50.0）")
    parser.add_argument("--Vmin", type=float, default=-5.0,
                        metavar="Ha",
                        help="滤波窗口下界（Hartree，default: -5.0）")

    # ---- 随机态 / RR ----
    parser.add_argument("--n-random", type=int, default=64,
                        metavar="INT",
                        help="每个 El 的随机初态数量（default: 64）")
    parser.add_argument("--svd-tol", type=float, default=1e-3,
                        metavar="FLOAT",
                        help="Rayleigh-Ritz SVD 秩截断阈值（default: 1e-3）")
    parser.add_argument("--max-energies", type=int, default=20,
                        metavar="INT",
                        help="输出能级数上限（default: 20）")
    parser.add_argument("--seed", type=int, default=42,
                        metavar="INT",
                        help="随机种子（default: 42）")

    # ---- FD 阶数选择 ----
    parser.add_argument("--fd-orders", type=int, nargs="+",
                        default=None,
                        metavar="ORDER",
                        help=f"要测试的 FD 阶数（可多个，default: 所有 {_ALL_FD_ORDERS}）")
    parser.add_argument("--fft-only", action="store_true",
                        help="只运行 FFT 算符，跳过有限差分阶数比较")

    # ---- 输出 ----
    parser.add_argument("--out-dir", type=str, default="fd_results",
                        metavar="DIR",
                        help="结果输出目录（default: fd_results）")

    args = parser.parse_args()

    # ── 参数整理 ──
    EL           = args.El
    NC           = args.nc
    DE           = args.dE
    VMIN         = args.Vmin
    N_RANDOM     = args.n_random
    SVD_TOL      = args.svd_tol
    MAX_ENERGIES = args.max_energies
    QD_RADIUS    = args.qd_radius
    FFT_ONLY     = args.fft_only
    CUBE_FILE    = args.cube_file
    PARAMS_FILE  = args.params_file
    R_CUT        = args.r_cut

    # 验证 FD 阶数
    if args.fd_orders is not None:
        bad = [o for o in args.fd_orders if o not in _ALL_FD_ORDERS]
        if bad:
            parser.error(f"不支持的 FD 阶数 {bad}；可选：{_ALL_FD_ORDERS}")
        FD_ORDERS = sorted(args.fd_orders)
    else:
        FD_ORDERS = _ALL_FD_ORDERS

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True)
    TS  = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = np.random.default_rng(args.seed)

    # ──────────────────────────────────────────────
    # 构建势能
    # ──────────────────────────────────────────────
    if QD_RADIUS is None:
        TAG = "localPot"
        N   = 64
        print(f"势能来源：{CUBE_FILE}（N={N}）")
        builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
        x0, y0, z0, V0 = builder.build_potential(N)
        pot = PotentialGrid(x0, y0, z0, V0, source="localPot")
    else:
        TAG     = f"QD_R{QD_RADIUS}"
        qd_cube = Path(f"QD_Outputs/QD_R{QD_RADIUS}.cube")
        if not qd_cube.exists():
            parser.error(f"未找到 {qd_cube}，请先运行 generate_QD_cubes.py")
        N, d_qd, origin_qd = _read_cube_header(str(qd_cube))
        print(f"势能来源：{qd_cube}  N={N}  d={d_qd:.4f}  origin={origin_qd:.4f}")

        builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
        x0, y0, z0, V0 = builder.build_potential(64)
        _interp_V = RegularGridInterpolator(
            (x0, y0, z0), V0.astype(float),
            method="linear", bounds_error=False, fill_value=0.0)

        x_new = origin_qd + np.arange(N) * d_qd
        X, Y, Z = np.meshgrid(x_new, x_new, x_new, indexing="ij")
        pts   = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        V_new = _interp_V(pts).reshape(N, N, N)
        pot   = PotentialGrid(x_new, x_new, x_new, V_new, source=TAG)

    print(f"TAG={TAG}  N={N}  N_grid={N**3:,}")

    dt  = (NC / (DE * 2.5)) ** 2
    par = PhysParams(dE=DE, Vmin=VMIN, dt=dt)
    print(f"参数：El={EL}, nc={NC}, dE={DE}, Vmin={VMIN}")
    print(f"  dt={dt:.4f},  sigma={1/np.sqrt(2*dt):.6f} Hartree")

    # ──────────────────────────────────────────────
    # 构建 Newton 插值节点和系数
    # ──────────────────────────────────────────────
    El_list     = np.array([EL])
    filter_func = lambda x, el: _filt_func_gaussian(x, el, dt)

    print("\n构建 Newton 滤波系数...")
    t_coef0 = time.perf_counter()
    an, samp = build_filter_coefficients(
        El_list, par, NC,
        filter_func=filter_func,
        samp_method="ashkenazy",
        interpolation_tolerance=1e-6,
        enhance_step=10,
        max_enhance_iters=30,
    )
    t_coef   = time.perf_counter() - t_coef0
    nc_true  = len(samp)
    ms       = len(El_list)
    print(f"  nc_true={nc_true}  ms={ms}  (构建耗时 {t_coef:.2f}s)")

    # ──────────────────────────────────────────────
    # 主循环
    # ──────────────────────────────────────────────
    results = []

    print("\n=== FFT 参考 ===")
    H_fft, _, _ = build_3d_fft_operator(N, pot)
    row = _run_filter(H_fft, "FFT", N, nc_true, ms, N_RANDOM,
                      samp, an, par, SVD_TOL, MAX_ENERGIES, rng)
    row["fd_order"] = None
    results.append(row)
    eval_ref = row["eval"]

    if FFT_ONLY:
        print("\n=== 跳过有限差分（--fft-only 已指定）===")
    else:
        print(f"\n=== 有限差分各阶 {FD_ORDERS} ===")
        for order in FD_ORDERS:
            H_fd, _, _ = build_3d_fd_operator(N, pot, fd_order=order)
            row = _run_filter(H_fd, f"FD-{order:2d}", N, nc_true, ms, N_RANDOM,
                              samp, an, par, SVD_TOL, MAX_ENERGIES, rng)
            row["fd_order"] = order
            results.append(row)

    # ──────────────────────────────────────────────
    # 保存 JSON
    # ──────────────────────────────────────────────
    output = {
        "script"  : "compare_fd_filter.py",
        "datetime": TS,
        "config"  : dict(TAG=TAG, N=N, El=EL, nc=NC, dE=DE, Vmin=VMIN,
                         dt=dt, n_random=N_RANDOM, svd_tol=SVD_TOL,
                         nc_true=nc_true, ms=ms, fd_orders=FD_ORDERS,
                         seed=args.seed),
        "results" : results,
    }
    json_path = OUT_DIR / f"compare_fd_filter_{TAG}_{TS}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved: {json_path}")

    # ──────────────────────────────────────────────
    # Markdown 表格
    # ──────────────────────────────────────────────
    lines = [
        f"# FD vs FFT — Filter Diagonalization  "
        f"({TAG}, N={N}, El={EL}, nc={nc_true}, Gaussian, n_random={N_RANDOM})\n",
        f"Generated: {TS}\n",
        "| Method | E[0] (Hartree) | ΔE vs FFT | T_wall (s) | N_H total | RR rank |",
        "|--------|--------------|-----------|------------|-----------|---------|",
    ]
    for row in results:
        method = row["label"]
        if row["success"]:
            de = row["eval"] - eval_ref if not np.isnan(eval_ref) else float("nan")
            lines.append(
                f"| {method:8s} | {row['eval']:14.8f} | {de:+.2e} "
                f"| {row['t_wall']:10.3f} | {row['n_H']:9d} | {row['rr_rank']:7d} |"
            )
        else:
            lines.append(
                f"| {method:8s} | FAILED | — | {row['t_wall']:.3f} | — | — |")

    md_text = "\n".join(lines) + "\n"
    print("\n" + md_text)
    md_path = OUT_DIR / f"compare_fd_filter_{TAG}_{TS}.md"
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"Markdown saved: {md_path}")

    # ──────────────────────────────────────────────
    # 绘图
    # ──────────────────────────────────────────────
    plot_path = OUT_DIR / f"compare_fd_filter_{TAG}_{TS}.png"
    _make_plot(results, eval_ref, N, TAG, EL, nc_true, N_RANDOM, plot_path)
    print(f"Plot saved:  {plot_path}")


if __name__ == "__main__":
    main()
