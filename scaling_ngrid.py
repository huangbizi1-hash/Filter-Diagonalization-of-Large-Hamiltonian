"""
scaling_ngrid.py
T (wall time) 和 N_H (Hψ 次数) 关于总格点数 N_grid = N³ 的 scaling。

策略：固定 d = base_d（通过加大 L 保持间距不变，避免数值不稳定）。
对 4 个不同 target 各画一条曲线，便于比较不同目标能量处的差异。

输出（scaling_results/）
    scaling_ngrid_TIMESTAMP.json
    scaling_ngrid_TIMESTAMP.png
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, time, sys
from datetime import datetime
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

import primme
from ho3d_solvers_v2 import build_3d_fft_operator
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
BASE_N      = 64
N_LIST      = np.arange(64, 80, 1, dtype=int)   # [64, 65, 66, ..., 79]
TARGETS     = [-0.17, -0.20, -0.22, -0.25]
N_LEVELS    = 1
BLOCKSIZE   = 1
TOL         = 1e-6
R_CUT       = 7.0
CUBE_FILE   = "localPot.cube"
PARAMS_FILE = "gaussian_fit_params.json"

OUT_DIR = Path("scaling_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

TARGET_COLORS = ["steelblue", "darkorange", "forestgreen", "crimson"]

# ──────────────────────────────────────────────
# 基础势能（N=64）
# ──────────────────────────────────────────────
print("加载基础势能（N=64）...")
builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(BASE_N)
d_base = float(x0[1] - x0[0])
cx = 0.5 * (float(x0[0]) + float(x0[-1]))
cy = 0.5 * (float(y0[0]) + float(y0[-1]))
cz = 0.5 * (float(z0[0]) + float(z0[-1]))
print(f"  d_base = {d_base:.6f} Bohr,  center = ({cx:.3f}, {cy:.3f}, {cz:.3f})")
print(f"  N_list = {N_LIST.tolist()},  N_grid = {[int(N)**3 for N in N_LIST]}")

# ──────────────────────────────────────────────
# 辅助：构建固定 d 的扩展势能
# ──────────────────────────────────────────────
_interp_cache = None

def _get_interp():
    global _interp_cache
    if _interp_cache is None:
        _interp_cache = RegularGridInterpolator(
            (x0, y0, z0), V0.astype(float),
            method="linear", bounds_error=False, fill_value=0.0,
        )
    return _interp_cache


def build_extended_pot(N: int) -> PotentialGrid:
    """固定 d=d_base，扩大 box 构建势能。高斯势在 r>r_cut 处精确为零，zero-padding 正确。"""
    if N == BASE_N:
        return PotentialGrid(x0, y0, z0, V0, source="base_N64")
    half_x = (N - 1) * d_base / 2
    half_y = (N - 1) * (float(y0[1]) - float(y0[0])) / 2   # keep original y/z d
    half_z = (N - 1) * (float(z0[1]) - float(z0[0])) / 2
    x_new = np.linspace(cx - half_x, cx + half_x, N)
    y_new = np.linspace(cy - half_y, cy + half_y, N)
    z_new = np.linspace(cz - half_z, cz + half_z, N)
    interp = _get_interp()
    Xn, Yn, Zn = np.meshgrid(x_new, y_new, z_new, indexing="ij")
    pts = np.stack([Xn.ravel(), Yn.ravel(), Zn.ravel()], axis=-1)
    V_new = interp(pts).reshape(N, N, N)
    return PotentialGrid(x_new, y_new, z_new, V_new, source=f"ext_N{N}")

# ──────────────────────────────────────────────
# 辅助：幂律拟合
# ──────────────────────────────────────────────
def power_fit(x_arr, y_arr):
    """y = a·x^b，返回 (a, b, r2)。数据少于 2 点或含零/负值返回 (None, None, None)。"""
    x, y = np.asarray(x_arr, float), np.asarray(y_arr, float)
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return None, None, None
    lx, ly = np.log(x[valid]), np.log(y[valid])
    b, loga = np.polyfit(lx, ly, 1)
    y_pred = b * lx + loga
    ss_res = np.sum((ly - y_pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return np.exp(loga), b, r2

# ──────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────
print(f"\n开始 scaling（{len(N_LIST)} 个 N × {len(TARGETS)} 个 target）...")
results_by_target = {str(t): [] for t in TARGETS}

for N in N_LIST:
    N_grid = int(N) ** 3
    print(f"\nN={N} (N_grid={N_grid:,})")
    pot = build_extended_pot(int(N))
    d_actual = float(pot.x[1] - pot.x[0])
    print(f"  d={d_actual:.6f}  box=[{pot.x[0]:.3f},{pot.x[-1]:.3f}]")
    H_op, n_un, _ = build_3d_fft_operator(int(N), pot)
    ncv = max(80, 2 * N_LEVELS)

    for target in TARGETS:
        print(f"  target={target:6.2f} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            evals, _, stats = primme.eigsh(
                H_op, k=N_LEVELS, which=target,
                method="PRIMME_JDQMR",
                maxBlockSize=BLOCKSIZE, ncv=ncv, tol=TOL,
                return_stats=True, return_history=False,
            )
            T_wall   = time.perf_counter() - t0
            N_H      = int(stats["numMatvecs"])
            restarts = int(stats.get("numRestarts", -1))
            outer    = int(stats.get("numOuterIterations", -1))
            success  = True
            evals_l  = sorted(float(e) for e in evals)
            err_msg  = ""
        except Exception as exc:
            T_wall   = time.perf_counter() - t0
            N_H      = None; restarts = None; outer = None
            success  = False; evals_l = []; err_msg = str(exc)

        row = dict(N=int(N), N_grid=N_grid, d=d_actual, target=target,
                   T_wall=T_wall, N_H=N_H, numRestarts=restarts,
                   numOuterIterations=outer, evals=evals_l,
                   success=success, err_msg=err_msg)
        results_by_target[str(target)].append(row)

        if success:
            print(f"T={T_wall:.1f}s  N_H={N_H}  E={evals_l}  restarts={restarts}")
        else:
            print(f"FAILED: {err_msg}")

# ──────────────────────────────────────────────
# 保存 JSON
# ──────────────────────────────────────────────
output = {
    "script": "scaling_ngrid.py",
    "datetime": TS,
    "config": dict(BASE_N=BASE_N, d_base=d_base, N_LIST=N_LIST.tolist(),
                   TARGETS=TARGETS, N_LEVELS=N_LEVELS, BLOCKSIZE=BLOCKSIZE, TOL=TOL),
    "results_by_target": results_by_target,
}
json_path = OUT_DIR / f"scaling_ngrid_{TS}.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, ykey, ylabel in zip(
    axes,
    ["T_wall",         "N_H"],
    ["Wall time T (s)", "Matvec count N_H"],
):
    for color, target in zip(TARGET_COLORS, TARGETS):
        rows = [r for r in results_by_target[str(target)]
                if r["success"] and r[ykey] is not None]
        if not rows:
            continue
        x_vals = np.array([r["N_grid"] for r in rows])
        y_vals = np.array([r[ykey]    for r in rows])
        ax.loglog(x_vals, y_vals, "o", color=color, markersize=7, zorder=3)

        a, b, r2 = power_fit(x_vals, y_vals)
        if b is not None:
            x_fit = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 200)
            ax.loglog(x_fit, a * x_fit ** b, "--", color=color, lw=1.8,
                      label=f"target={target}: ∝ N³^{b:.2f}  (R²={r2:.3f})")

    ax.set_xlabel("N_grid = N³", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} vs N_grid\n(n_levels={N_LEVELS}, blocksize={BLOCKSIZE}, d={d_base:.4f})")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.35)

fig.tight_layout()
plot_path = OUT_DIR / f"scaling_ngrid_{TS}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved:  {plot_path}")
