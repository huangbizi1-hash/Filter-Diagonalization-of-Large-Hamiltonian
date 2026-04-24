"""
scaling_blocksize.py
T (wall time) 和 N_H (Hψ 次数) 关于 blocksize 的 scaling。

策略：固定 N=64，target=-0.22，n_levels=10。
blocksize_list = np.arange(1, 21, 2)  → [1, 3, 5, ..., 19]

输出（scaling_results/）
    scaling_blocksize_TIMESTAMP.json
    scaling_blocksize_TIMESTAMP.png
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, time
from datetime import datetime
from pathlib import Path

import primme
from ho3d_solvers_v2 import build_3d_fft_operator
from gaussian_potential_builder import GaussianPotentialBuilder

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
N           = 64
TARGET      = -0.22
N_LEVELS    = 10
TOL         = 1e-6
R_CUT       = 7.0
CUBE_FILE   = "localPot.cube"
PARAMS_FILE = "gaussian_fit_params.json"

BLOCKSIZE_LIST = np.arange(1, 21, 2, dtype=int)   # [1, 3, 5, ..., 19]

OUT_DIR = Path("scaling_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

# ──────────────────────────────────────────────
# 构建势能与算符（固定 N=64）
# ──────────────────────────────────────────────
print(f"构建势能（N={N}）...")
builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(N)
from gaussian_potential_builder import PotentialGrid
pot = PotentialGrid(x0, y0, z0, V0, source=f"N{N}")

print(f"构建 H 算符...")
H_op, n_un, _ = build_3d_fft_operator(N, pot)
print(f"  H_op shape={H_op.shape},  N_grid={N**3:,}")

# ──────────────────────────────────────────────
# 辅助：幂律拟合
# ──────────────────────────────────────────────
def power_fit(x_arr, y_arr):
    """y = a·x^b，返回 (a, b, r2)。"""
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
ncv = max(80, 2 * N_LEVELS)
print(f"\n开始 scaling（{len(BLOCKSIZE_LIST)} 个 blocksize，target={TARGET}，n_levels={N_LEVELS}）...")
results = []

for blocksize in BLOCKSIZE_LIST:
    print(f"  blocksize={blocksize:2d}  ncv={ncv} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        evals, _, stats = primme.eigsh(
            H_op, k=N_LEVELS, which=TARGET,
            method="PRIMME_JDQMR",
            maxBlockSize=int(blocksize), ncv=ncv, tol=TOL,
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

    row = dict(blocksize=int(blocksize), N=N, target=TARGET, n_levels=N_LEVELS,
               T_wall=T_wall, N_H=N_H, numRestarts=restarts,
               numOuterIterations=outer, evals=evals_l,
               success=success, err_msg=err_msg)
    results.append(row)

    if success:
        print(f"T={T_wall:.1f}s  N_H={N_H}  E[0]={evals_l[0]:.6f}  restarts={restarts}")
    else:
        print(f"FAILED: {err_msg}")

# ──────────────────────────────────────────────
# 保存 JSON
# ──────────────────────────────────────────────
output = {
    "script": "scaling_blocksize.py",
    "datetime": TS,
    "config": dict(N=N, TARGET=TARGET, N_LEVELS=N_LEVELS, TOL=TOL,
                   BLOCKSIZE_LIST=BLOCKSIZE_LIST.tolist()),
    "results": results,
}
json_path = OUT_DIR / f"scaling_blocksize_{TS}.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

# ──────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, ykey, ylabel in zip(
    axes,
    ["T_wall",          "N_H"],
    ["Wall time T (s)", "Matvec count N_H"],
):
    rows = [r for r in results if r["success"] and r[ykey] is not None]
    if not rows:
        ax.set_title(f"{ylabel}: no data")
        continue

    x_vals = np.array([r["blocksize"] for r in rows])
    y_vals = np.array([r[ykey]        for r in rows])
    ax.loglog(x_vals, y_vals, "o", color="darkorange", markersize=8, zorder=3,
              label="measured")

    a, b, r2 = power_fit(x_vals, y_vals)
    if b is not None:
        x_fit = np.logspace(np.log10(x_vals.min()), np.log10(x_vals.max()), 200)
        ax.loglog(x_fit, a * x_fit ** b, "--", color="darkorange", lw=1.8,
                  label=f"∝ blocksize^{b:.2f}  (R²={r2:.3f})")

    ax.set_xlabel("blocksize", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} vs blocksize\n(N={N}, target={TARGET}, n_levels={N_LEVELS})")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.35)

fig.tight_layout()
plot_path = OUT_DIR / f"scaling_blocksize_{TS}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved:  {plot_path}")
