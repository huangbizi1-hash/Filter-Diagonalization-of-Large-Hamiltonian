"""
scaling_ngrid_QD_primme_gd_jacobi.py
测试 PRIMME_GD（Jacobi 预条件）在真实 QD 网格上的 scaling。

输出（scaling_results/）
    scaling_ngrid_QD_PRIMME_GD_Jacobi_TIMESTAMP.json
    scaling_ngrid_QD_PRIMME_GD_Jacobi_TIMESTAMP.png
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, time
from datetime import datetime
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

import primme
import scipy.sparse.linalg as spla
from ho3d_solvers_v2 import build_3d_fft_operator
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
QD_DIR      = Path("QD_Outputs")
TARGETS     = [-0.17, -0.20, -0.22, -0.25]
N_LEVELS    = 1
BLOCKSIZE   = 1
TOL         = 1e-6
MAX_MATVECS = 30000
R_CUT       = 7.0
CUBE_FILE   = "localPot.cube"
PARAMS_FILE = "gaussian_fit_params.json"
EPS_PRECOND = 1e-3

OUT_DIR = Path("scaling_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

TARGET_COLORS = ["steelblue", "darkorange", "forestgreen", "crimson"]
METHOD_NAME = "PRIMME_GD"
PRECOND_NAME = "Jacobi"

print(f"方法：{METHOD_NAME} + {PRECOND_NAME}")


def read_cube_header(cube_path: str):
    with open(cube_path) as f:
        f.readline(); f.readline()
        parts = f.readline().split()
        origin = float(parts[1])
        line4  = f.readline().split()
        N      = int(line4[0])
        d      = float(line4[1])
    return N, d, origin


cube_files = sorted(QD_DIR.glob("QD_R*.cube"),
                    key=lambda p: int(p.stem.split("R")[1]))
if not cube_files:
    raise FileNotFoundError(
        f"未在 {QD_DIR} 找到 QD_R*.cube，请先运行 generate_QD_cubes.py")

qd_list = []
for cp in cube_files:
    r      = int(cp.stem.split("R")[1])
    N_cube, d_cube, origin = read_cube_header(str(cp))
    qd_list.append((r, cp, N_cube, d_cube, origin))

print(f"发现 {len(qd_list)} 个 QD Cube 文件。")

print(f"\n加载基础高斯势能（{CUBE_FILE}）...")
builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(64)
_interp_V = RegularGridInterpolator(
    (x0, y0, z0), V0.astype(float),
    method="linear", bounds_error=False, fill_value=0.0)


def build_qd_pot(N: int, d_qd: float, origin_qd: float) -> PotentialGrid:
    x_new = origin_qd + np.arange(N) * d_qd
    X, Y, Z = np.meshgrid(x_new, x_new, x_new, indexing='ij')
    pts   = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    V_new = _interp_V(pts).reshape(N, N, N)
    return PotentialGrid(x_new, x_new, x_new, V_new, source=f"QD_N{N}")


def make_jacobi_precond(N: int, d_qd: float, E_target: float):
    k1d    = 2.0 * np.pi * np.fft.fftfreq(N, d=d_qd)
    K1d_sq = k1d ** 2
    T_k    = (K1d_sq[:, None, None] +
              K1d_sq[None, :, None] +
              K1d_sq[None, None, :]) / 2.0
    T_k = np.minimum(T_k, 30.0)
    denom = np.abs(T_k - E_target) + EPS_PRECOND
    n_un = N ** 3

    def mv(v):
        psi_k = np.fft.fftn(v.reshape(N, N, N))
        return np.fft.ifftn(psi_k / denom).real.ravel()

    return spla.LinearOperator((n_un, n_un), matvec=mv, dtype=float)


def power_fit(x_arr, y_arr):
    x, y  = np.asarray(x_arr, float), np.asarray(y_arr, float)
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None, None, None
    lx, ly = np.log(x[valid]), np.log(y[valid])
    b, loga = np.polyfit(lx, ly, 1)
    ss_res  = np.sum((ly - (b * lx + loga)) ** 2)
    ss_tot  = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return np.exp(loga), b, r2


print(f"\n开始 scaling（{len(qd_list)} 个 QD × {len(TARGETS)} 个 target）...")
results = []

for r, cp, N, d_qd, origin_qd in qd_list:
    N_grid = N ** 3
    print(f"\n  QD R={r}  N={N}  N_grid={N_grid:,}")
    pot = build_qd_pot(N, d_qd, origin_qd)
    H_op, n_un, _ = build_3d_fft_operator(N, pot)
    ncv = max(80, 2 * N_LEVELS)

    for target in TARGETS:
        P_op = make_jacobi_precond(N, d_qd, target)
        print(f"    target={target} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            evals, _, stats = primme.eigsh(
                H_op,
                k            = N_LEVELS,
                which        = target,
                method       = METHOD_NAME,
                OPinv        = P_op,
                maxBlockSize = BLOCKSIZE,
                ncv          = ncv,
                tol          = TOL,
                maxMatvecs   = MAX_MATVECS,
                return_stats = True,
                return_history = False,
            )
            T_wall  = time.perf_counter() - t0
            N_H     = int(stats["numMatvecs"])
            success = True
            evals_l = sorted(float(e) for e in evals)
            err_msg = ""
        except Exception as exc:
            T_wall  = time.perf_counter() - t0
            N_H     = None; success = False
            evals_l = []; err_msg = str(exc)

        row = dict(radius=r, N=N, N_grid=N_grid, d=d_qd,
                   target=target, method=METHOD_NAME, precond=PRECOND_NAME,
                   T_wall=T_wall, N_H=N_H,
                   evals=evals_l, success=success, err_msg=err_msg)
        results.append(row)

        if success:
            print(f"T={T_wall:.1f}s  N_H={N_H}  E[0]={evals_l[0]:.6f}")
        else:
            print(f"FAILED: {err_msg}")

output = {
    "script"  : "scaling_ngrid_QD_primme_gd_jacobi.py",
    "datetime": TS,
    "config"  : dict(TARGETS=TARGETS, N_LEVELS=N_LEVELS,
                     BLOCKSIZE=BLOCKSIZE, TOL=TOL,
                     METHOD=METHOD_NAME, PRECOND=PRECOND_NAME,
                     QD_radii=[r for r, *_ in qd_list]),
    "results" : results,
}
json_path = OUT_DIR / f"scaling_ngrid_QD_PRIMME_GD_Jacobi_{TS}.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, ykey, ylabel in zip(
    axes,
    ["T_wall",          "N_H"],
    ["Wall time T (s)", "Matvec count N_H"],
):
    for target, color in zip(TARGETS, TARGET_COLORS):
        rows = [row for row in results
                if row["target"] == target and row["success"]
                and row[ykey] is not None]
        if not rows:
            continue
        x_vals = np.array([row["N_grid"] for row in rows])
        y_vals = np.array([row[ykey]     for row in rows])
        ax.loglog(x_vals, y_vals, "o", color=color, markersize=7, zorder=3)
        a_fit, b_fit, r2 = power_fit(x_vals, y_vals)
        if b_fit is not None:
            x_fit = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 200)
            ax.loglog(x_fit, a_fit * x_fit ** b_fit, "--",
                      color=color, lw=1.6,
                      label=f"target={target}  ∝ N³^{b_fit:.2f}  R²={r2:.3f}")

    ax.set_xlabel("N_grid = N³", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} vs N_grid\n（{METHOD_NAME}+{PRECOND_NAME}）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.35)

fig.tight_layout()
plot_path = OUT_DIR / f"scaling_ngrid_QD_PRIMME_GD_Jacobi_{TS}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved:  {plot_path}")
