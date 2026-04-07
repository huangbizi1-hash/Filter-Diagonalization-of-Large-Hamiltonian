"""
scaling_ngrid_QD_cg_folded.py
测试 CG + Folded Spectrum 在真实 QD 网格上的 scaling。

说明：
- 每个 target 用 cg_minimize_folded 求 1 个态
- 记录 wall time、CG 迭代次数 n_iter
- 额外给出等效 Hψ 次数 N_H_equiv = 3*n_iter + 2（近似）

输出（scaling_results/）
    scaling_ngrid_QD_CG_FOLDED_TIMESTAMP.json
    scaling_ngrid_QD_CG_FOLDED_TIMESTAMP.png
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, time
from datetime import datetime
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

from ho3d_solvers_v2 import build_3d_fft_operator, cg_minimize_folded
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid

QD_DIR      = Path("QD_Outputs")
TARGETS     = [-0.17, -0.20, -0.22, -0.25]
R_CUT       = 7.0
CUBE_FILE   = "localPot.cube"
PARAMS_FILE = "gaussian_fit_params.json"

CG_MAXITER     = 2000
CG_GTOL        = 1e-8
CG_ENERGY_TOL  = 5e-4
CG_THETA_MAX   = 0.8
CG_VERBOSE     = False

OUT_DIR = Path("scaling_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")
TARGET_COLORS = ["steelblue", "darkorange", "forestgreen", "crimson"]


def read_cube_header(cube_path: str):
    with open(cube_path) as f:
        f.readline(); f.readline()
        parts = f.readline().split()
        origin = float(parts[1])
        line4  = f.readline().split()
        N      = int(line4[0])
        d      = float(line4[1])
    return N, d, origin


cube_files = sorted(QD_DIR.glob("QD_R*.cube"), key=lambda p: int(p.stem.split("R")[1]))
if not cube_files:
    raise FileNotFoundError(f"未在 {QD_DIR} 找到 QD_R*.cube，请先运行 generate_QD_cubes.py")

qd_list = []
for cp in cube_files:
    r = int(cp.stem.split("R")[1])
    N_cube, d_cube, origin = read_cube_header(str(cp))
    qd_list.append((r, cp, N_cube, d_cube, origin))

print(f"方法：CG + Folded Spectrum，共 {len(qd_list)} 个 QD")

builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(64)
_interp_V = RegularGridInterpolator(
    (x0, y0, z0), V0.astype(float),
    method="linear", bounds_error=False, fill_value=0.0)


def build_qd_pot(N: int, d_qd: float, origin_qd: float) -> PotentialGrid:
    x_new = origin_qd + np.arange(N) * d_qd
    X, Y, Z = np.meshgrid(x_new, x_new, x_new, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    V_new = _interp_V(pts).reshape(N, N, N)
    return PotentialGrid(x_new, x_new, x_new, V_new, source=f"QD_N{N}")


def power_fit(x_arr, y_arr):
    x, y = np.asarray(x_arr, float), np.asarray(y_arr, float)
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None, None, None
    lx, ly = np.log(x[valid]), np.log(y[valid])
    b, loga = np.polyfit(lx, ly, 1)
    ss_res = np.sum((ly - (b * lx + loga)) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return np.exp(loga), b, r2


results = []
for r, cp, N, d_qd, origin_qd in qd_list:
    N_grid = N ** 3
    print(f"\nQD R={r}, N={N}, N_grid={N_grid:,}")
    pot = build_qd_pot(N, d_qd, origin_qd)
    H_op, n_un, _ = build_3d_fft_operator(N, pot)

    for target in TARGETS:
        print(f"  target={target} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            rng = np.random.default_rng(42)
            x0_cg = rng.standard_normal(n_un)
            cg_res = cg_minimize_folded(
                H_op,
                target,
                x0_cg,
                maxiter=CG_MAXITER,
                gtol=CG_GTOL,
                energy_tol=CG_ENERGY_TOL,
                theta_max=CG_THETA_MAX,
                verbose=CG_VERBOSE,
            )
            T_wall = time.perf_counter() - t0
            n_iter = int(cg_res["n_iter"])
            N_H_equiv = int(3 * n_iter + 2)
            eval0 = float(cg_res["E_ritz"])
            success = bool(cg_res["converged"])
            err_msg = "" if success else f"not converged: {cg_res['conv_reason']}"
            conv_reason = str(cg_res["conv_reason"])
        except Exception as exc:
            T_wall = time.perf_counter() - t0
            n_iter = None
            N_H_equiv = None
            eval0 = None
            success = False
            err_msg = str(exc)
            conv_reason = "exception"

        row = dict(
            radius=r, N=N, N_grid=N_grid, d=d_qd,
            target=target, method="CG_FOLDED", precond="None",
            T_wall=T_wall, n_iter=n_iter, N_H_equiv=N_H_equiv,
            evals=[] if eval0 is None else [eval0],
            success=success, conv_reason=conv_reason, err_msg=err_msg,
        )
        results.append(row)

        if success:
            print(f"T={T_wall:.1f}s  n_iter={n_iter}  N_H≈{N_H_equiv}  E={eval0:.6f}")
        else:
            print(f"FAILED: {err_msg}")

output = {
    "script": "scaling_ngrid_QD_cg_folded.py",
    "datetime": TS,
    "config": dict(
        TARGETS=TARGETS,
        method="CG_FOLDED",
        CG_MAXITER=CG_MAXITER,
        CG_GTOL=CG_GTOL,
        CG_ENERGY_TOL=CG_ENERGY_TOL,
        CG_THETA_MAX=CG_THETA_MAX,
        QD_radii=[r for r, *_ in qd_list],
    ),
    "results": results,
}
json_path = OUT_DIR / f"scaling_ngrid_QD_CG_FOLDED_{TS}.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, ykey, ylabel in zip(
    axes,
    ["T_wall", "N_H_equiv"],
    ["Wall time T (s)", "Equivalent matvec N_H≈3*n_iter+2"],
):
    for target, color in zip(TARGETS, TARGET_COLORS):
        rows = [row for row in results if row["target"] == target and row["success"] and row[ykey] is not None]
        if not rows:
            continue
        x_vals = np.array([row["N_grid"] for row in rows])
        y_vals = np.array([row[ykey] for row in rows])
        ax.loglog(x_vals, y_vals, "o", color=color, markersize=7)
        a_fit, b_fit, r2 = power_fit(x_vals, y_vals)
        if b_fit is not None:
            x_fit = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 200)
            ax.loglog(x_fit, a_fit * x_fit**b_fit, "--", color=color, lw=1.6,
                      label=f"target={target}  ∝ N³^{b_fit:.2f}  R²={r2:.3f}")

    ax.set_xlabel("N_grid = N³", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} vs N_grid\n（CG + Folded Spectrum）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.35)

fig.tight_layout()
plot_path = OUT_DIR / f"scaling_ngrid_QD_CG_FOLDED_{TS}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved: {plot_path}")
