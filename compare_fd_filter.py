"""
compare_fd_filter.py
比较不同阶数有限差分动能算符对滤波对角化（Filter Diagonalization）性能的影响。

配置：N=64，高斯窗，nc=5000，El=-0.17，使用 localPot.cube 势能。

有限差分阶数：2, 4, 6, 8, 10, 12, 14, 16, 18, 20
参考基准：FFT 算符（精确动能）

Newton 多项式滤波器作用于任意 LinearOperator（H_op.matvec），
不依赖 pyfftw 或 T_k_diagonal，可直接接受有限差分 H_op。

输出（fd_results/）
    compare_fd_filter_TIMESTAMP.json
    compare_fd_filter_TIMESTAMP.md
    compare_fd_filter_TIMESTAMP.png
"""

import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.linalg import eigh

from ho3d_solvers_v2 import (
    build_3d_fft_operator,
    build_3d_fd_operator,
    FD_STENCILS,
)
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid
from fft_code.params       import PhysParams
from fft_code.filter_coeff import build_filter_coefficients
from fft_code.filter_coeff import _filt_func_gaussian  # noqa: F401 (用于构造窗函数)

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
N           = 64
EL          = -0.17       # 滤波中心（物理单位，Hartree）
NC          = 5000        # Newton 节点数
DE          = 50.0        # 滤波窗口宽度
VMIN        = -5.0        # 窗口下界
N_RANDOM    = 3           # 每个 El 的随机初态数量
SVD_TOL     = 1e-3        # Rayleigh-Ritz SVD 秩截断
MAX_ENERGIES= 20          # 输出能级数上限
R_CUT       = 7.0
CUBE_FILE   = "localPot.cube"
PARAMS_FILE = "gaussian_fit_params.json"

OUT_DIR = Path("fd_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

FD_ORDERS = sorted(FD_STENCILS.keys())   # [2, 4, 6, ..., 20]

# dt 自动推导：sigma ≈ dE/(2.5*nc)，dt = 1/(2*sigma²)
dt = (NC / (DE * 2.5)) ** 2
par = PhysParams(dE=DE, Vmin=VMIN, dt=dt)
print(f"参数：N={N}, El={EL}, nc={NC}, dE={DE}, Vmin={VMIN}")
print(f"  dt={dt:.4f},  sigma={1/np.sqrt(2*dt):.6f} Hartree")

# ──────────────────────────────────────────────
# 构建势能
# ──────────────────────────────────────────────
print(f"\n加载势能（N={N}）...")
builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(N)
pot = PotentialGrid(x0, y0, z0, V0, source=f"N{N}")

# ──────────────────────────────────────────────
# 构建 Newton 插值节点和系数（共用一套，与 H 无关）
# ──────────────────────────────────────────────
El_list = np.array([EL])
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
t_coef = time.perf_counter() - t_coef0
nc_true = len(samp)
print(f"  nc_true={nc_true}  (构建耗时 {t_coef:.2f}s)")
# an: (1, nc_true)，取第 0 个 El 的系数
coeffs = an[0]   # (nc_true,)

# ──────────────────────────────────────────────
# Newton 滤波器作用（与 H_op 无关的通用实现）
# ──────────────────────────────────────────────
def apply_newton_filter(H_op, psi_flat, nodes, coeffs, par_):
    """
    f(H)|ψ⟩ — Newton 多项式求值，使用通用 H_op.matvec。

    缩放变量：x̃ = 4(H - Vmin)/dE - 2
    Newton 递推：
        basis_0 = ψ
        basis_j = (H̃ - nodes[j-1]) * basis_{j-1}，其中 H̃ = 4/dE*(H-Vmin) - 2
        f(H)ψ  = Σ_j coeffs[j] * basis_j
    """
    result   = coeffs[0] * psi_flat.copy()
    psi_prev = psi_flat.copy()

    for j in range(1, len(nodes)):
        H_psi    = H_op.matvec(psi_prev)
        # 缩放：H̃ ψ = 4/dE * (H ψ - Vmin ψ) - 2 ψ
        Hs_psi   = (4.0 / par_.dE) * (H_psi - par_.Vmin * psi_prev) - 2.0 * psi_prev
        psi_curr = Hs_psi - nodes[j - 1] * psi_prev
        result  += coeffs[j] * psi_curr
        psi_prev = psi_curr

    return result


# ──────────────────────────────────────────────
# Rayleigh-Ritz 对角化（使用通用 H_op）
# ──────────────────────────────────────────────
def rayleigh_ritz_generic(basis_mat, H_op, svd_tol, max_energies):
    """
    SVD + Rayleigh-Ritz：在由 basis_mat（列向量）张成的子空间中对角化 H_op。

    basis_mat : (n_grid, n_basis)
    返回排序好的特征值（最多 max_energies 个）。
    """
    # 列归一化，去除全零列
    norms = np.linalg.norm(basis_mat, axis=0)
    mask  = norms > 0
    B     = basis_mat[:, mask] / norms[None, mask]

    # QR 分解降维
    Q, R = np.linalg.qr(B, mode="reduced")
    # SVD on R 选取有效秩
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    r = max(1, int(np.sum(sigma > svd_tol)))
    print(f"    Rayleigh-Ritz rank r={r}")
    Ur = (Q @ U1)[:, :r]   # (n_grid, r)

    # 构建子空间矩阵 H_tilde[i,j] = <ur_i|H|ur_j>
    H_tilde = np.zeros((r, r), dtype=float)
    for j in range(r):
        HUj = H_op.matvec(Ur[:, j])
        H_tilde[:, j] = Ur.T @ HUj

    evals, _ = eigh(H_tilde)
    return np.sort(evals.real)[:max_energies]


# ──────────────────────────────────────────────
# 随机初态（正弦加权，确保在盒内衰减）
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
        # 滤波随机态 → 子空间基
        basis_list = []
        n_H = 0
        for _ in range(N_RANDOM):
            psi0 = random_psi(n_grid)
            filt = apply_newton_filter(H_op, psi0, samp, coeffs, par)
            norm = np.linalg.norm(filt)
            if norm > 0:
                basis_list.append(filt / norm)
            n_H += nc_true

        if not basis_list:
            raise RuntimeError("所有随机态滤波后范数为零")

        basis_mat = np.column_stack(basis_list)   # (n_grid, n_basis)

        # Rayleigh-Ritz
        energies = rayleigh_ritz_generic(basis_mat, H_op, SVD_TOL, MAX_ENERGIES)

        t_wall = time.perf_counter() - t0
        eval0  = float(energies[0]) if len(energies) > 0 else float("nan")
        success = True
        err_msg = ""
        print(f"    E[0]={eval0:.8f}  T={t_wall:.2f}s  N_H(filter)={n_H}")
    except Exception as exc:
        t_wall  = time.perf_counter() - t0
        eval0   = float("nan")
        n_H     = -1
        success = False
        err_msg = str(exc)
        energies = []
        print(f"    FAILED: {err_msg}")

    return dict(label=label, eval=eval0, t_wall=t_wall, n_H=n_H,
                energies=list(energies) if hasattr(energies, '__iter__') else [],
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

# 各阶有限差分
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
    "config"  : dict(N=N, EL=EL, NC=NC, DE=DE, VMIN=VMIN,
                     dt=dt, N_RANDOM=N_RANDOM, SVD_TOL=SVD_TOL,
                     nc_true=nc_true),
    "results" : results,
}
json_path = OUT_DIR / f"compare_fd_filter_{TS}.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

# ──────────────────────────────────────────────
# Markdown 表格
# ──────────────────────────────────────────────
lines = [
    f"# FD vs FFT — Filter Diagonalization  "
    f"(N={N}, El={EL}, nc={nc_true}, Gaussian)\n",
    f"Generated: {TS}\n",
    "| Method | E[0] (Hartree) | ΔE vs FFT | T_wall (s) | N_H (filter) |",
    "|--------|--------------|-----------|------------|--------------|",
]
for row in results:
    method = row["label"]
    if row["success"]:
        de = row["eval"] - eval_ref if not np.isnan(eval_ref) else float("nan")
        lines.append(
            f"| {method:8s} | {row['eval']:14.8f} | {de:+.2e} "
            f"| {row['t_wall']:10.3f} | {row['n_H']:12d} |"
        )
    else:
        lines.append(f"| {method:8s} | FAILED | — | {row['t_wall']:.3f} | — |")

md_text = "\n".join(lines) + "\n"
print("\n" + md_text)
md_path = OUT_DIR / f"compare_fd_filter_{TS}.md"
with open(md_path, "w") as f:
    f.write(md_text)
print(f"Markdown saved: {md_path}")

# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────
succ = [r for r in results if r["success"] and r["fd_order"] is not None]
orders  = [r["fd_order"]  for r in succ]
evals   = [r["eval"]      for r in succ]
t_walls = [r["t_wall"]    for r in succ]
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
    ax.axhline(results[0]["t_wall"], color="gray", lw=1.2, ls="--", label="FFT ref")
    ax.legend(fontsize=9)
ax.set_xlabel("FD order", fontsize=11)
ax.set_ylabel("Wall time (s)", fontsize=11)
ax.set_title("Wall time vs FD order\n(Filter Diagonalization)", fontsize=10)
ax.grid(True, alpha=0.4)
ax.set_xticks(orders)

fig.suptitle(
    f"Filter Diag: FD vs FFT  (N={N}, El={EL}, nc={nc_true}, Gaussian)",
    fontsize=11)
fig.tight_layout()
plot_path = OUT_DIR / f"compare_fd_filter_{TS}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved:  {plot_path}")
