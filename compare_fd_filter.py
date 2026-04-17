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
    python compare_fd_filter.py                  # 用 localPot.cube，N=64
    python compare_fd_filter.py --qd-radius 17   # 用 QD_Outputs/QD_R17.cube

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

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FD vs FFT filter diagonalization")
parser.add_argument("--qd-radius", type=int, default=None,
                    help="QD 半径（Bohr），读取 QD_Outputs/QD_R{R}.cube；"
                         "不指定则使用 localPot.cube（N=64）")
parser.add_argument("--fft-only", action="store_true",
                    help="只运行 FFT 算符，跳过有限差分阶数比较")
parser.add_argument("--Vmin", type=float, default=-5.0,
                    help="滤波窗口下界 Vmin（Ha），默认 -5.0")
parser.add_argument("--dE", type=float, default=50.0,
                    help="滤波窗口宽度 dE（Ha），默认 50.0")
args = parser.parse_args()
QD_RADIUS = args.qd_radius
FFT_ONLY  = args.fft_only

# ──────────────────────────────────────────────
# 固定超参数
# ──────────────────────────────────────────────
EL           = -0.17      # 滤波中心（物理单位，Hartree）
NC           = 5000       # Newton 节点数（初始估计）
DE           = args.dE    # 滤波窗口宽度
VMIN         = args.Vmin  # 窗口下界
N_RANDOM     = 64         # 每个 El 的随机初态数量
SVD_TOL      = 1e-3       # Rayleigh-Ritz SVD 秩截断
MAX_ENERGIES = 20         # 输出能级数上限
R_CUT        = 7.0
CUBE_FILE    = "localPot.cube"
PARAMS_FILE  = "gaussian_fit_params.json"

OUT_DIR = Path("fd_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

FD_ORDERS = sorted(FD_STENCILS.keys())   # [2, 4, 6, ..., 20]


# ──────────────────────────────────────────────
# 读取 cube 文件头（N, d, origin）
# ──────────────────────────────────────────────
def read_cube_header(cube_path: str):
    with open(cube_path) as f:
        f.readline(); f.readline()
        parts  = f.readline().split()
        origin = float(parts[1])
        line4  = f.readline().split()
        N_c    = int(line4[0])
        d_c    = float(line4[1])
    return N_c, d_c, origin


# ──────────────────────────────────────────────
# 构建势能
# ──────────────────────────────────────────────
if QD_RADIUS is None:
    # 默认：localPot.cube，N=64
    TAG = "localPot"
    N   = 64
    print(f"势能来源：{CUBE_FILE}（N={N}）")
    builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
    x0, y0, z0, V0 = builder.build_potential(N)
    pot = PotentialGrid(x0, y0, z0, V0, source="localPot")
else:
    # QD cube：从文件头读取 N 和 d，再插值势能
    TAG       = f"QD_R{QD_RADIUS}"
    qd_cube   = Path(f"QD_Outputs/QD_R{QD_RADIUS}.cube")
    if not qd_cube.exists():
        raise FileNotFoundError(f"未找到 {qd_cube}，请先运行 generate_QD_cubes.py")
    N, d_qd, origin_qd = read_cube_header(str(qd_cube))
    print(f"势能来源：{qd_cube}  N={N}  d={d_qd:.4f}  origin={origin_qd:.4f}")

    # 在 N=64 的参考网格上建立高斯势插值器
    builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
    x0, y0, z0, V0 = builder.build_potential(64)
    _interp_V = RegularGridInterpolator(
        (x0, y0, z0), V0.astype(float),
        method="linear", bounds_error=False, fill_value=0.0)

    x_new = origin_qd + np.arange(N) * d_qd
    X, Y, Z = np.meshgrid(x_new, x_new, x_new, indexing='ij')
    pts   = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    V_new = _interp_V(pts).reshape(N, N, N)
    pot   = PotentialGrid(x_new, x_new, x_new, V_new, source=TAG)

print(f"TAG={TAG}  N={N}  N_grid={N**3:,}")

# dt 自动推导：sigma ≈ dE/(2.5*nc)，dt = 1/(2*sigma²)
dt  = (NC / (DE * 2.5)) ** 2
par = PhysParams(dE=DE, Vmin=VMIN, dt=dt)
print(f"参数：El={EL}, nc={NC}, dE={DE}, Vmin={VMIN}")
print(f"  dt={dt:.4f},  sigma={1/np.sqrt(2*dt):.6f} Hartree")

# ──────────────────────────────────────────────
# 构建 Newton 插值节点和系数（节点与 H 无关，共用一套）
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
t_coef  = time.perf_counter() - t_coef0
nc_true = len(samp)
ms      = len(El_list)
print(f"  nc_true={nc_true}  ms={ms}  (构建耗时 {t_coef:.2f}s)")
# an: (ms, nc_true)

# ──────────────────────────────────────────────
# 通用 apply_filter_H_all（对应 hamiltonian.apply_filter_H_all）
# ──────────────────────────────────────────────
def apply_filter_H_all_generic(H_op, psi_flat, nodes, an_, par_):
    """
    f_i(H)|ψ⟩，对所有 ms 个 El_i 共享 Newton 基底向量。

    H-apply 次数 = nc（而非 ms×nc），与 apply_filter_H_all 逻辑完全一致。

    参数
    ----
    H_op     : LinearOperator (n_grid, n_grid)
    psi_flat : (n_grid,)  输入向量
    nodes    : (nc,)      Newton 节点（缩放坐标）
    an_      : (ms, nc)   Newton 系数矩阵
    par_     : PhysParams

    返回
    ----
    results : (ms, n_grid)   results[ie] = f_ie(H)|ψ⟩
    """
    ms_, nc_ = an_.shape
    # 基底 j=0：basis_0 = psi_flat
    results  = an_[:, 0:1] * psi_flat[None, :]   # (ms, n_grid)
    psi_prev = psi_flat.copy()

    for j in range(1, nc_):
        H_psi    = H_op.matvec(psi_prev)
        # 缩放递推（与 hamiltonian.py 完全相同）：
        #   basis_j = (4/dE*(H-Vmin) - 2 - nodes[j-1]) * basis_{j-1}
        psi_curr = ((4.0 / par_.dE) * (H_psi - par_.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        results += an_[:, j:j+1] * psi_curr[None, :]   # (ms,1)*(1,n_grid)
        psi_prev = psi_curr

    return results   # (ms, n_grid)


# ──────────────────────────────────────────────
# Rayleigh-Ritz 对角化（通用 H_op）
# ──────────────────────────────────────────────
def rayleigh_ritz_generic(basis_mat, H_op, svd_tol, max_energies):
    """
    SVD + Rayleigh-Ritz：在 basis_mat 列向量张成的子空间中对角化 H_op。

    basis_mat : (n_grid, n_basis)
    返回排序特征值（最多 max_energies 个）及有效秩 r。
    """
    # 列归一化，去除零列
    norms = np.linalg.norm(basis_mat, axis=0)
    mask  = norms > 0
    B     = basis_mat[:, mask] / norms[None, mask]

    # QR + SVD（与 svd_rayleigh_ritz 完全相同的流程）
    Q, R         = np.linalg.qr(B, mode="reduced")
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    r = max(1, int(np.sum(sigma > svd_tol)))
    print(f"    SVD rank r={r}  (sigma_max={sigma[0]:.3e}, "
          f"sigma_r={sigma[r-1]:.3e})")
    Ur = (Q @ U1)[:, :r]   # (n_grid, r)  正交归一基

    # H 的子空间矩阵 H̃[i,j] = ⟨ur_i|H|ur_j⟩
    H_tilde = np.zeros((r, r), dtype=float)
    for j in range(r):
        HUj             = H_op.matvec(Ur[:, j])
        H_tilde[:, j]   = Ur.T @ HUj

    evals, _ = eigh(H_tilde)
    return np.sort(evals.real)[:max_energies], r


# ──────────────────────────────────────────────
# 随机初态（与 main.py random_sine_psi 等价的简单版本）
# ──────────────────────────────────────────────
rng = np.random.default_rng(42)

def random_psi(n_grid):
    v = rng.standard_normal(n_grid)
    return v / np.linalg.norm(v)


# ──────────────────────────────────────────────
# 运行单次滤波对角化
# ──────────────────────────────────────────────
def run_filter(H_op, label):
    n_grid = N ** 3
    print(f"\n  [{label}]  El={EL}, nc={nc_true}, n_random={N_RANDOM}")
    t0 = time.perf_counter()
    try:
        # ── 滤波随机态，所有 El 共享基底（nc H-apply / 随机态） ──
        filtered_psi_matrix = np.zeros((ms * N_RANDOM, n_grid), dtype=float)
        for i in range(N_RANDOM):
            psi_rand    = random_psi(n_grid)
            psi_filt_all = apply_filter_H_all_generic(
                H_op, psi_rand, samp, an, par)   # (ms, n_grid)
            for ie in range(ms):
                psi_filt = psi_filt_all[ie]
                norm     = np.linalg.norm(psi_filt)
                if norm > 0:
                    filtered_psi_matrix[ie * N_RANDOM + i] = psi_filt / norm
            if (i + 1) % 10 == 0:
                print(f"    filtered {i+1}/{N_RANDOM} ...", flush=True)

        n_H_filter = nc_true * N_RANDOM   # 滤波阶段 H-apply 次数

        # ── Rayleigh-Ritz ──
        energies, r = rayleigh_ritz_generic(
            filtered_psi_matrix.T, H_op, SVD_TOL, MAX_ENERGIES)

        n_H_total = n_H_filter + r   # RR 额外 r 次 H-apply
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
        success   = False
        err_msg   = str(exc)
        energies  = []
        r         = 0
        print(f"    FAILED: {err_msg}")

    return dict(label=label, eval=eval0, t_wall=t_wall,
                n_H=n_H_total,
                n_H_filter=nc_true * N_RANDOM if success else -1,
                rr_rank=r,
                energies=(energies.tolist() if hasattr(energies, 'tolist')
                           else list(energies)),
                success=success, err_msg=err_msg)


# ──────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────
results = []

# 参考：FFT
print("\n=== FFT 参考 ===")
H_fft, _, _ = build_3d_fft_operator(N, pot)
row = run_filter(H_fft, "FFT")
row["fd_order"] = None
results.append(row)
eval_ref = row["eval"]

# 各阶有限差分（仅在未指定 --fft-only 时运行）
if FFT_ONLY:
    print("\n=== 跳过有限差分（--fft-only 已指定）===")
else:
    print("\n=== 有限差分各阶 ===")
    for order in FD_ORDERS:
        H_fd, _, _ = build_3d_fd_operator(N, pot, fd_order=order)
        row = run_filter(H_fd, f"FD-{order:2d}")
        row["fd_order"] = order
        results.append(row)

# ──────────────────────────────────────────────
# 保存 JSON
# ──────────────────────────────────────────────
output = {
    "script"  : "compare_fd_filter.py",
    "datetime": TS,
    "config"  : dict(TAG=TAG, N=N, EL=EL, NC=NC, DE=DE, VMIN=VMIN,
                     dt=dt, N_RANDOM=N_RANDOM, SVD_TOL=SVD_TOL,
                     nc_true=nc_true, ms=ms),
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
succ    = [r for r in results if r["success"] and r["fd_order"] is not None]
plot_path = OUT_DIR / f"compare_fd_filter_{TAG}_{TS}.png"

if not succ:
    # --fft-only 模式：只有一个 FFT 结果，不画 FD 比较图
    fft_row = results[0] if results and results[0]["success"] else None
    fig, ax = plt.subplots(figsize=(6, 4))
    if fft_row:
        ax.bar(["FFT"], [fft_row["t_wall"]], color="steelblue")
        ax.set_ylabel("Wall time (s)", fontsize=11)
        ax.set_title(
            f"Filter Diag (FFT only)\n"
            f"{TAG}, N={N}, El={EL}, nc={nc_true}, n_random={N_RANDOM}",
            fontsize=10)
        ax.text(0, fft_row["t_wall"] * 0.5,
                f"E[0]={fft_row['eval']:.6f}\nRR rank={fft_row['rr_rank']}",
                ha="center", va="center", fontsize=10, color="white")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
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
        f"({TAG}, N={N}, El={EL}, nc={nc_true}, Gaussian, n_random={N_RANDOM})",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"Plot saved:  {plot_path}")
