"""
scaling_ngrid_QD.py
T (wall time) 和 N_H (Hψ 次数) 关于真实 QD 格点数 N_grid = N³ 的 scaling。

与 scaling_ngrid.py 的区别：
  - 格点来自真实 InAs QD 几何（generate_QD_cubes.py 生成）
  - d=0.625 Bohr 固定，N 因 QD 尺寸不同而自然不同
  - 势能用高斯拟合参数（localPot.cube / gaussian_fit_params.json）
    在扩展网格上 zero-padding 插值（r > r_cut 时 V≈0 精确成立）

用法：
    python scaling_ngrid_QD.py [--precond {None,Jacobi,TPA,ShiftedKinetic}]

输出（scaling_results/）
    scaling_ngrid_QD_{PRECOND}_TIMESTAMP.json
    scaling_ngrid_QD_{PRECOND}_TIMESTAMP.png
"""

import argparse
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
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--precond", default="None",
                    choices=["None", "Jacobi", "TPA", "ShiftedKinetic"],
                    help="预条件子类型（默认 None）")
args = parser.parse_args()
PRECOND_NAME = args.precond

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

print(f"预条件子：{PRECOND_NAME}")

# ──────────────────────────────────────────────
# 读取 Cube 文件头
# ──────────────────────────────────────────────
def read_cube_header(cube_path: str):
    with open(cube_path) as f:
        f.readline(); f.readline()
        parts = f.readline().split()
        origin = float(parts[1])
        line4  = f.readline().split()
        N      = int(line4[0])
        d      = float(line4[1])
    return N, d, origin


# ──────────────────────────────────────────────
# 扫描 QD 目录
# ──────────────────────────────────────────────
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

print(f"发现 {len(qd_list)} 个 QD Cube 文件：")
for r, cp, N, dc, orig in qd_list:
    print(f"  R={r:3d} Bohr  N={N:4d}  N³={N**3:>12,}  d={dc:.4f}  origin={orig:.4f}")

# ──────────────────────────────────────────────
# 基础势能（插值用）
# ──────────────────────────────────────────────
print(f"\n加载基础高斯势能（{CUBE_FILE}）...")
builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(64)
d_base = float(x0[1] - x0[0])
cx = 0.5 * (float(x0[0]) + float(x0[-1]))
cy = 0.5 * (float(y0[0]) + float(y0[-1]))
cz = 0.5 * (float(z0[0]) + float(z0[-1]))
print(f"  d_base={d_base:.6f},  center=({cx:.3f},{cy:.3f},{cz:.3f})")

_interp_V = RegularGridInterpolator(
    (x0, y0, z0), V0.astype(float),
    method="linear", bounds_error=False, fill_value=0.0)


def build_qd_pot(N: int, d_qd: float, origin_qd: float) -> PotentialGrid:
    x_new = origin_qd + np.arange(N) * d_qd
    X, Y, Z = np.meshgrid(x_new, x_new, x_new, indexing='ij')
    pts   = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    V_new = _interp_V(pts).reshape(N, N, N)
    return PotentialGrid(x_new, x_new, x_new, V_new, source=f"QD_N{N}")


# ──────────────────────────────────────────────
# 预条件子工厂（与 compare_preconditioners.py 保持一致）
# ──────────────────────────────────────────────
def make_precond(name: str, N: int, d_qd: float,
                 V_flat: np.ndarray, E_target: float):
    """为给定 (N, d, V, target) 构建预条件 LinearOperator，None 返回 None。"""
    if name == "None":
        return None

    # 计算该 QD 网格的 T_k
    k1d    = 2.0 * np.pi * np.fft.fftfreq(N, d=d_qd)
    K1d_sq = k1d ** 2
    T_k    = (K1d_sq[:, None, None] +
              K1d_sq[None, :, None] +
              K1d_sq[None, None, :]) / 2.0
    T_k    = np.minimum(T_k, 30.0)
    V_mean = float(V_flat.mean())
    n_un   = N ** 3

    if name == "Jacobi":
        # k 空间对角：P⁻¹(k) = 1 / |T(k) - E|
        denom = np.abs(T_k - E_target) + EPS_PRECOND
        def mv(v):
            psi_k = np.fft.fftn(v.reshape(N, N, N))
            return np.fft.ifftn(psi_k / denom).real.ravel()

    elif name == "TPA":
        # Teter-Payne-Allan：f(t)/|T(k)+V_mean-E|，f=27/(27+18t+12t²+8t³)
        T_ref = max(float(T_k.mean()), 0.5)
        t     = T_k / T_ref
        f     = 27.0 / (27.0 + 18.0*t + 12.0*t**2 + 8.0*t**3)
        denom = np.abs(T_k + V_mean - E_target) + EPS_PRECOND
        K     = f / denom
        def mv(v):
            psi_k = np.fft.fftn(v.reshape(N, N, N))
            return np.fft.ifftn(K * psi_k).real.ravel()

    elif name == "ShiftedKinetic":
        # P⁻¹x = IFFT(FFT(x) / |T(k)+V_mean-E|)
        denom = np.abs(T_k + V_mean - E_target) + EPS_PRECOND
        def mv(v):
            psi_k = np.fft.fftn(v.reshape(N, N, N))
            return np.fft.ifftn(psi_k / denom).real.ravel()

    else:
        raise ValueError(f"未知预条件子：{name}")

    return spla.LinearOperator((n_un, n_un), matvec=mv, dtype=float)


# ──────────────────────────────────────────────
# 幂律拟合
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────
print(f"\n开始 scaling（{len(qd_list)} 个 QD × {len(TARGETS)} 个 target，"
      f"precond={PRECOND_NAME}）...")
results = []

for r, cp, N, d_qd, origin_qd in qd_list:
    N_grid = N ** 3
    print(f"\n  QD R={r}  N={N}  N_grid={N_grid:,}")
    pot            = build_qd_pot(N, d_qd, origin_qd)
    H_op, n_un, V_flat = build_3d_fft_operator(N, pot)
    ncv = max(80, 2 * N_LEVELS)

    for target in TARGETS:
        P_op = make_precond(PRECOND_NAME, N, d_qd, V_flat, target)
        print(f"    target={target} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            kwargs = dict(
                k            = N_LEVELS,
                which        = target,
                method       = "PRIMME_JDQMR",
                maxBlockSize = BLOCKSIZE,
                ncv          = ncv,
                tol          = TOL,
                maxMatvecs   = MAX_MATVECS,
                return_stats = True,
                return_history = False,
            )
            if P_op is not None:
                kwargs["OPinv"] = P_op
            evals, _, stats = primme.eigsh(H_op, **kwargs)
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
                   target=target, precond=PRECOND_NAME,
                   T_wall=T_wall, N_H=N_H,
                   evals=evals_l, success=success, err_msg=err_msg)
        results.append(row)

        if success:
            print(f"T={T_wall:.1f}s  N_H={N_H}  E[0]={evals_l[0]:.6f}")
        else:
            print(f"FAILED: {err_msg}")

# ──────────────────────────────────────────────
# 保存 JSON
# ──────────────────────────────────────────────
output = {
    "script"  : "scaling_ngrid_QD.py",
    "datetime": TS,
    "config"  : dict(TARGETS=TARGETS, N_LEVELS=N_LEVELS,
                     BLOCKSIZE=BLOCKSIZE, TOL=TOL,
                     PRECOND=PRECOND_NAME,
                     QD_radii=[r for r, *_ in qd_list]),
    "results" : results,
}
json_path = OUT_DIR / f"scaling_ngrid_QD_{PRECOND_NAME}_{TS}.json"
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
    for target, color in zip(TARGETS, TARGET_COLORS):
        rows = [row for row in results
                if row["target"] == target and row["success"]
                and row[ykey] is not None]
        if not rows:
            continue
        x_vals = np.array([row["N_grid"] for row in rows])
        y_vals = np.array([row[ykey]     for row in rows])
        ax.loglog(x_vals, y_vals, "o", color=color, markersize=7, zorder=3)
        for xv, yv, row in zip(x_vals, y_vals, rows):
            ax.annotate(f"N={row['N']}", (xv, yv),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=6, color=color)
        a_fit, b_fit, r2 = power_fit(x_vals, y_vals)
        if b_fit is not None:
            x_fit = np.logspace(np.log10(x_vals.min()),
                                np.log10(x_vals.max()), 200)
            ax.loglog(x_fit, a_fit * x_fit ** b_fit, "--",
                      color=color, lw=1.6,
                      label=f"target={target}  ∝ N³^{b_fit:.2f}  R²={r2:.3f}")

    ax.set_xlabel("N_grid = N³", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel} vs N_grid\n"
                 f"（真实 InAs QD，d=0.625 Bohr，precond={PRECOND_NAME}）",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.35)

fig.tight_layout()
plot_path = OUT_DIR / f"scaling_ngrid_QD_{PRECOND_NAME}_{TS}.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved:  {plot_path}")
