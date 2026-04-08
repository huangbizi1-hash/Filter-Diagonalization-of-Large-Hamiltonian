"""
scaling_ngrid_QD_filter.py
测试滤波对角化（Filter Diagonalization, FFT 动能）在真实 QD 网格上的 scaling。

与 scaling_ngrid_QD.py（JDQMR）并列，采用相同的 QD cube 几何，
用以比较两种方法在 N_grid 增大时的 wall time / N_H scaling 规律。

关键参数
--------
  d        = 0.625  Bohr（固定步长，N 由 QD 半径决定）
  nc       = 5000   Newton 节点数（初始估计）
  dE       = 50.0   Hartree
  Vmin     = -5.0   Hartree
  El_list  = 20 个滤波中心（-0.24 ~ -0.11 均匀分布）
  n_random = 64     每个 El 的随机初态数量
  filter   = Gaussian 窗（dt 自动推导）

算法（参考 main.py apply_filter_H_all）
--------------------------------------
  1. 构建 Newton 滤波系数（节点与 H 无关，所有 QD 共用）
  2. 对每个随机初态 ψ，调用 apply_filter_H_all_generic：
       所有 ms 个 El 共享 Newton 基底向量，H-apply 次数 = nc
     得到 ms × n_random 个滤波态
  3. SVD + Rayleigh-Ritz 对角化子空间，输出近 El 的本征能级

用法
----
  python scaling_ngrid_QD_filter.py --qd-radius 17
  python scaling_ngrid_QD_filter.py --qd-radius 21

输出（scaling_results/）
  scaling_ngrid_QD_filter_R{R}_TIMESTAMP.json
  scaling_ngrid_QD_filter_R{R}_TIMESTAMP.png
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

from ho3d_solvers_v2 import build_3d_fft_operator
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid
from fft_code.params       import PhysParams
from fft_code.filter_coeff import build_filter_coefficients
from fft_code.filter_coeff import _filt_func_gaussian

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Filter diagonalization N_grid scaling on QD cubes")
parser.add_argument("--qd-radius", type=int, required=True,
                    help="QD 半径（Bohr），读取 QD_Outputs/QD_R{R}.cube")
args      = parser.parse_args()
QD_RADIUS = args.qd_radius

# ──────────────────────────────────────────────
# 固定参数
# ──────────────────────────────────────────────
D_FIXED     = 0.625          # Bohr，步长固定
NC          = 5000           # Newton 节点数（初始估计）
DE          = 50.0           # Hartree
VMIN        = -5.0           # Hartree
N_RANDOM    = 64             # 每个 El 的随机初态数量
SVD_TOL     = 1e-3           # Rayleigh-Ritz SVD 秩截断
MAX_ENERGIES = 30            # 输出能级数上限
R_CUT       = 7.0
CUBE_FILE   = "localPot.cube"
PARAMS_FILE = "gaussian_fit_params.json"

EL_LIST = [
    -0.24, -0.2331578947368421, -0.2263157894736842,
    -0.21947368421052632, -0.21263157894736842, -0.20578947368421052,
    -0.19894736842105262, -0.19210526315789472, -0.18526315789473682,
    -0.17842105263157892, -0.17157894736842105, -0.16473684210526313,
    -0.15789473684210525, -0.15105263157894736, -0.14421052631578946,
    -0.13736842105263158, -0.13052631578947366, -0.12368421052631577,
    -0.11684210526315789, -0.11,
]
El_list = np.array(EL_LIST)

OUT_DIR = Path("scaling_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")


# ──────────────────────────────────────────────
# 读取 cube 文件头
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
# 读取 QD cube，构建势能
# ──────────────────────────────────────────────
qd_cube = Path(f"QD_Outputs/QD_R{QD_RADIUS}.cube")
if not qd_cube.exists():
    raise FileNotFoundError(f"未找到 {qd_cube}，请先运行 generate_QD_cubes.py")

N, d_cube, origin_qd = read_cube_header(str(qd_cube))
print(f"QD_R{QD_RADIUS}: N={N}  d={d_cube:.4f}  origin={origin_qd:.4f}")
print(f"N_grid = {N**3:,}")

# 验证步长一致
if abs(d_cube - D_FIXED) > 1e-6:
    import warnings
    warnings.warn(f"cube 步长 d={d_cube:.6f} ≠ D_FIXED={D_FIXED}，使用 cube 实际步长")

# 在 N=64 参考网格上建立高斯势插值器
builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(64)
_interp_V = RegularGridInterpolator(
    (x0, y0, z0), V0.astype(float),
    method="linear", bounds_error=False, fill_value=0.0)

x_new = origin_qd + np.arange(N) * d_cube
X, Y, Z = np.meshgrid(x_new, x_new, x_new, indexing='ij')
pts   = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
V_new = _interp_V(pts).reshape(N, N, N)
pot   = PotentialGrid(x_new, x_new, x_new, V_new, source=f"QD_R{QD_RADIUS}")

print(f"V_min={V_new.min():.4f}  V_max={V_new.max():.4f}  V_mean={V_new.mean():.4f}")

# ──────────────────────────────────────────────
# 构建 FFT 哈密顿量
# ──────────────────────────────────────────────
H_op, n_un, V_flat = build_3d_fft_operator(N, pot)

# ──────────────────────────────────────────────
# 滤波参数
# ──────────────────────────────────────────────
dt  = (NC / (DE * 2.5)) ** 2
par = PhysParams(dE=DE, Vmin=VMIN, dt=dt)
ms  = len(El_list)
print(f"\n滤波参数: nc={NC}, dE={DE}, Vmin={VMIN}, ms={ms}, n_random={N_RANDOM}")
print(f"  dt={dt:.4f},  sigma={1/np.sqrt(2*dt):.6f} Hartree")

# ──────────────────────────────────────────────
# 构建 Newton 系数（共用，与 H 无关）
# ──────────────────────────────────────────────
filter_func = lambda x, el: _filt_func_gaussian(x, el, dt)

print("\n构建 Newton 滤波系数...")
t0 = time.perf_counter()
an, samp = build_filter_coefficients(
    El_list, par, NC,
    filter_func=filter_func,
    samp_method="ashkenazy",
    interpolation_tolerance=1e-6,
    enhance_step=10,
    max_enhance_iters=30,
)
t_coef  = time.perf_counter() - t0
nc_true = len(samp)
print(f"  nc_true={nc_true}  (构建耗时 {t_coef:.2f}s)")
# an: (ms, nc_true)

# ──────────────────────────────────────────────
# apply_filter_H_all（通用 H_op.matvec，对应 main.py）
# ──────────────────────────────────────────────
def apply_filter_H_all_generic(H_op_, psi_flat, nodes, an_, par_):
    """
    f_i(H)|ψ⟩，对所有 ms 个 El_i 共享 Newton 基底向量。
    H-apply 次数 = nc（而非 ms×nc），与 main.py apply_filter_H_all 完全一致。
    返回: (ms, n_grid)
    """
    ms_, nc_ = an_.shape
    results  = an_[:, 0:1] * psi_flat[None, :]   # (ms, n_grid)
    psi_prev = psi_flat.copy()
    for j in range(1, nc_):
        H_psi    = H_op_.matvec(psi_prev)
        psi_curr = ((4.0 / par_.dE) * (H_psi - par_.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        results += an_[:, j:j+1] * psi_curr[None, :]
        psi_prev = psi_curr
    return results   # (ms, n_grid)


# ──────────────────────────────────────────────
# Rayleigh-Ritz（通用 H_op）
# ──────────────────────────────────────────────
def rayleigh_ritz_generic(basis_mat, H_op_, svd_tol, max_energies):
    """
    SVD + Rayleigh-Ritz。basis_mat: (n_grid, n_basis)
    返回 (排序特征值, rank r)。
    """
    norms = np.linalg.norm(basis_mat, axis=0)
    mask  = norms > 0
    B     = basis_mat[:, mask] / norms[None, mask]

    Q, R         = np.linalg.qr(B, mode="reduced")
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    r = max(1, int(np.sum(sigma > svd_tol)))
    print(f"  SVD rank r={r}  (sigma_max={sigma[0]:.3e}, sigma_r={sigma[r-1]:.3e})")
    Ur = (Q @ U1)[:, :r]   # (n_grid, r)

    H_tilde = np.zeros((r, r), dtype=float)
    for j in range(r):
        HUj           = H_op_.matvec(Ur[:, j])
        H_tilde[:, j] = Ur.T @ HUj

    evals, _ = eigh(H_tilde)
    return np.sort(evals.real)[:max_energies], r


# ──────────────────────────────────────────────
# 随机初态
# ──────────────────────────────────────────────
rng = np.random.default_rng(42)

def random_psi(n_grid):
    v = rng.standard_normal(n_grid)
    return v / np.linalg.norm(v)


# ──────────────────────────────────────────────
# 主运行：滤波 + Rayleigh-Ritz
# ──────────────────────────────────────────────
n_grid = N ** 3
print(f"\n开始滤波（n_random={N_RANDOM}, ms={ms}, nc={nc_true}）...")
filtered_psi_matrix = np.zeros((ms * N_RANDOM, n_grid), dtype=float)

t_filter_start = time.perf_counter()
for i in range(N_RANDOM):
    psi_rand     = random_psi(n_grid)
    psi_filt_all = apply_filter_H_all_generic(
        H_op, psi_rand, samp, an, par)   # (ms, n_grid)
    for ie in range(ms):
        psi_filt = psi_filt_all[ie]
        norm     = np.linalg.norm(psi_filt)
        if norm > 0:
            filtered_psi_matrix[ie * N_RANDOM + i] = psi_filt / norm
    if (i + 1) % 8 == 0:
        print(f"  filtered {i+1}/{N_RANDOM} ...", flush=True)

t_filter = time.perf_counter() - t_filter_start
n_H_filter = nc_true * N_RANDOM   # 每个随机态 nc 次 H-apply（共享 ms 个 El）
print(f"滤波耗时: {t_filter:.2f}s  N_H(filter)={n_H_filter}")

print("\nRayleigh-Ritz 对角化...")
t_rr_start = time.perf_counter()
energies, rr_rank = rayleigh_ritz_generic(
    filtered_psi_matrix.T, H_op, SVD_TOL, MAX_ENERGIES)
t_rr   = time.perf_counter() - t_rr_start
t_total = t_filter + t_rr + t_coef
n_H_total = n_H_filter + rr_rank

print(f"Rayleigh-Ritz 耗时: {t_rr:.2f}s  rank={rr_rank}")
print(f"总耗时: {t_total:.2f}s  N_H_total={n_H_total}")
print(f"前 {min(10, len(energies))} 个本征能级: "
      f"{np.round(energies[:10], 6).tolist()}")

# ──────────────────────────────────────────────
# 保存 JSON
# ──────────────────────────────────────────────
result = dict(
    script    = "scaling_ngrid_QD_filter.py",
    datetime  = TS,
    config    = dict(
        QD_RADIUS=QD_RADIUS, N=N, N_grid=n_grid,
        d=float(d_cube), origin=float(origin_qd),
        NC=NC, nc_true=nc_true, DE=DE, VMIN=VMIN,
        N_RANDOM=N_RANDOM, SVD_TOL=SVD_TOL,
        ms=ms, El_list=El_list.tolist(),
    ),
    timing    = dict(
        t_coef=t_coef, t_filter=t_filter, t_rr=t_rr, t_total=t_total,
    ),
    n_H       = dict(
        n_H_filter=n_H_filter, n_H_rr=rr_rank, n_H_total=n_H_total,
    ),
    rr_rank   = rr_rank,
    energies  = energies.tolist(),
    V_stats   = dict(V_min=float(V_new.min()),
                     V_max=float(V_new.max()),
                     V_mean=float(V_new.mean())),
)
json_path = OUT_DIR / f"scaling_ngrid_QD_filter_R{QD_RADIUS}_{TS}.json"
with open(json_path, "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

# ──────────────────────────────────────────────
# 绘图：本征能级分布
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.plot(El_list, El_list, "k--", lw=0.8, label="y=x (ideal)")
# 对每个 El，找最近的本征能级
nearest = [energies[np.argmin(np.abs(energies - el))] for el in El_list
           if len(energies) > 0]
ax.scatter(El_list[:len(nearest)], nearest,
           s=30, color="steelblue", zorder=3, label="Nearest energy level")
ax.set_xlabel("El (Hartree)", fontsize=11)
ax.set_ylabel("Nearest eigenvalue (Hartree)", fontsize=11)
ax.set_title(f"QD_R{QD_RADIUS} (N={N}, N_grid={n_grid:,})\n"
             f"本征能级 vs 滤波中心", fontsize=10)
ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

ax = axes[1]
ax.hist(energies, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
for el in El_list:
    ax.axvline(el, color="gray", lw=0.5, alpha=0.5)
ax.set_xlabel("Eigenvalue (Hartree)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(f"Eigenvalue histogram\n"
             f"T={t_total:.1f}s  N_H={n_H_total}", fontsize=10)
ax.grid(True, alpha=0.4)

fig.suptitle(
    f"Filter Diag Scaling — QD_R{QD_RADIUS}  "
    f"(N={N}, nc={nc_true}, n_random={N_RANDOM})",
    fontsize=11)
fig.tight_layout()
plot_path = OUT_DIR / f"scaling_ngrid_QD_filter_R{QD_RADIUS}_{TS}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved:  {plot_path}")
