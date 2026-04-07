"""
compare_fd_jdqmr.py
比较不同阶数有限差分动能算符对 JDQMR 求解器性能的影响。

配置：N=64，n_levels=1，blocksize=1，target=-0.17，使用 localPot.cube 势能。

有限差分阶数：2, 4, 6, 8, 10, 12, 14, 16, 18, 20
参考基准：FFT 算符（精确动能）

输出（fd_results/）
    compare_fd_jdqmr_TIMESTAMP.json
    compare_fd_jdqmr_TIMESTAMP.md
    compare_fd_jdqmr_TIMESTAMP.png
"""

import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import primme

from ho3d_solvers_v2 import (
    build_3d_fft_operator,
    build_3d_fd_operator,
    FD_STENCILS,
)
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
N           = 64
TARGET      = -0.17
N_LEVELS    = 1
BLOCKSIZE   = 1
TOL         = 1e-6
MAX_MATVECS = 30000
R_CUT       = 7.0
CUBE_FILE   = "localPot.cube"
PARAMS_FILE = "gaussian_fit_params.json"

OUT_DIR = Path("fd_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

FD_ORDERS = sorted(FD_STENCILS.keys())   # [2, 4, 6, ..., 20]

# ──────────────────────────────────────────────
# 构建势能
# ──────────────────────────────────────────────
print(f"加载势能（N={N}）...")
builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(N)
pot = PotentialGrid(x0, y0, z0, V0, source=f"N{N}")

# ──────────────────────────────────────────────
# 运行单次 JDQMR
# ──────────────────────────────────────────────
def run_jdqmr(H_op, label):
    ncv = max(80, 2 * N_LEVELS)
    print(f"  [{label}]  target={TARGET} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        evals, _, stats = primme.eigsh(
            H_op,
            k            = N_LEVELS,
            which        = TARGET,
            method       = "PRIMME_JDQMR",
            maxBlockSize = BLOCKSIZE,
            ncv          = ncv,
            tol          = TOL,
            maxMatvecs   = MAX_MATVECS,
            return_stats = True,
            return_history = False,
        )
        t_wall  = time.perf_counter() - t0
        n_mv    = int(stats["numMatvecs"])
        eval0   = float(evals[0])
        success = True
        err_msg = ""
    except Exception as exc:
        t_wall  = time.perf_counter() - t0
        n_mv    = -1
        eval0   = float("nan")
        success = False
        err_msg = str(exc)

    if success:
        print(f"E={eval0:.8f}  T={t_wall:.2f}s  N_H={n_mv}")
    else:
        print(f"FAILED: {err_msg}")
    return dict(label=label, eval=eval0, t_wall=t_wall, n_mv=n_mv,
                success=success, err_msg=err_msg)


# ──────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────
results = []

# 参考：FFT
print("\n=== FFT 参考 ===")
H_fft, n_un, _ = build_3d_fft_operator(N, pot)
row = run_jdqmr(H_fft, "FFT")
row["fd_order"] = None
results.append(row)
eval_ref = row["eval"]

# 各阶有限差分
print("\n=== 有限差分各阶 ===")
for order in FD_ORDERS:
    H_fd, _, _ = build_3d_fd_operator(N, pot, fd_order=order)
    row = run_jdqmr(H_fd, f"FD-{order:2d}")
    row["fd_order"] = order
    results.append(row)

# ──────────────────────────────────────────────
# 保存 JSON
# ──────────────────────────────────────────────
output = {
    "script"  : "compare_fd_jdqmr.py",
    "datetime": TS,
    "config"  : dict(N=N, TARGET=TARGET, N_LEVELS=N_LEVELS,
                     BLOCKSIZE=BLOCKSIZE, TOL=TOL, MAX_MATVECS=MAX_MATVECS),
    "results" : results,
}
json_path = OUT_DIR / f"compare_fd_jdqmr_{TS}.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

# ──────────────────────────────────────────────
# Markdown 表格
# ──────────────────────────────────────────────
lines = [
    f"# FD vs FFT — JDQMR  (N={N}, target={TARGET}, n_levels={N_LEVELS})\n",
    f"Generated: {TS}\n",
    "| Method | Eigenvalue | ΔE vs FFT | T_wall (s) | N_H matvec |",
    "|--------|-----------|-----------|------------|------------|",
]
for row in results:
    method = row["label"]
    if row["success"]:
        de = row["eval"] - eval_ref if eval_ref is not None else float("nan")
        lines.append(
            f"| {method:8s} | {row['eval']:12.8f} | {de:+.2e} "
            f"| {row['t_wall']:10.3f} | {row['n_mv']:10d} |"
        )
    else:
        lines.append(f"| {method:8s} | FAILED | — | {row['t_wall']:.3f} | — |")

md_text = "\n".join(lines) + "\n"
print("\n" + md_text)
md_path = OUT_DIR / f"compare_fd_jdqmr_{TS}.md"
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
n_mvs   = [r["n_mv"]      for r in succ]
d_evals = [abs(e - eval_ref) for e in evals]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.semilogy(orders, d_evals, "o-", color="steelblue")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("FD order", fontsize=11)
ax.set_ylabel("|E_FD - E_FFT| (Hartree)", fontsize=11)
ax.set_title("Eigenvalue error vs FD order", fontsize=10)
ax.grid(True, which="both", alpha=0.4)
ax.set_xticks(orders)

ax = axes[1]
ax.plot(orders, t_walls, "o-", color="darkorange")
if results[0]["success"]:
    ax.axhline(results[0]["t_wall"], color="gray", lw=1.2, ls="--", label="FFT ref")
    ax.legend(fontsize=9)
ax.set_xlabel("FD order", fontsize=11)
ax.set_ylabel("Wall time (s)", fontsize=11)
ax.set_title("Wall time vs FD order", fontsize=10)
ax.grid(True, alpha=0.4)
ax.set_xticks(orders)

ax = axes[2]
ax.plot(orders, n_mvs, "o-", color="forestgreen")
if results[0]["success"]:
    ax.axhline(results[0]["n_mv"], color="gray", lw=1.2, ls="--", label="FFT ref")
    ax.legend(fontsize=9)
ax.set_xlabel("FD order", fontsize=11)
ax.set_ylabel("Matvec count N_H", fontsize=11)
ax.set_title("Matvec count vs FD order", fontsize=10)
ax.grid(True, alpha=0.4)
ax.set_xticks(orders)

fig.suptitle(f"JDQMR: FD vs FFT  (N={N}, target={TARGET})", fontsize=12)
fig.tight_layout()
plot_path = OUT_DIR / f"compare_fd_jdqmr_{TS}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved:  {plot_path}")
