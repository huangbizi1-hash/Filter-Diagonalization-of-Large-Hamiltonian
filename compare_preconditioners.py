"""
compare_preconditioners.py
比较不同预条件子对 JDQMR 求解器性能的影响。

配置：N=64，n_levels=1，blocksize=1，4 个 target（-0.17/-0.20/-0.22/-0.25）

预条件子：
  1. None             — 无预条件（PRIMME 默认）
  2. Jacobi           — 对角预条件：P⁻¹x = x / |diag(H) - E_target|
  3. TPA              — Teter-Payne-Allan k 空间多项式滤波器
  4. ShiftedKinetic   — P⁻¹x = IFFT(FFT(x) / |T(k) + V_mean - E_target|)

TimingLinearOperator 对 H 和 P⁻¹ 分别计时，输出 Markdown 表格。

输出（precond_results/）
    compare_precond_TIMESTAMP.json
    compare_precond_TIMESTAMP.md
"""

import sys, time, json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import primme

from ho3d_solvers_v2 import build_3d_fft_operator
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
N         = 64
TARGETS   = [-0.17, -0.20, -0.22, -0.25]
N_LEVELS  = 1
BLOCKSIZE = 1
TOL       = 1e-6
R_CUT     = 7.0
CUBE_FILE   = "localPot.cube"
PARAMS_FILE = "gaussian_fit_params.json"
EPS_PRECOND = 1e-3      # 预条件分母保护

OUT_DIR = Path("precond_results"); OUT_DIR.mkdir(exist_ok=True)
TS      = datetime.now().strftime("%Y%m%d_%H%M%S")

# ──────────────────────────────────────────────
# TimingLinearOperator
# ──────────────────────────────────────────────
class TimingLinearOperator(spla.LinearOperator):
    """封装任意 LinearOperator，统计 _matvec 内耗时。"""
    def __init__(self, op: spla.LinearOperator):
        super().__init__(dtype=op.dtype, shape=op.shape)
        self._op        = op
        self.time_spent = 0.0
        self.call_count = 0

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        result = self._op.matvec(v)
        self.time_spent += time.perf_counter() - t0
        self.call_count += 1
        return result

    def reset(self):
        self.time_spent = 0.0
        self.call_count = 0

# ──────────────────────────────────────────────
# 构建势能与 H 算符
# ──────────────────────────────────────────────
print(f"加载势能（N={N}）...")
builder = GaussianPotentialBuilder(CUBE_FILE, PARAMS_FILE, R_CUT)
x0, y0, z0, V0 = builder.build_potential(N)
pot  = PotentialGrid(x0, y0, z0, V0, source=f"N{N}")

print("构建 H 算符（FFT-DVR）...")
H_op, n_un, V_flat = build_3d_fft_operator(N, pot)
print(f"  H_op shape={H_op.shape},  N_grid={N**3:,}")

# 独立计算 T_k（与 build_3d_fft_operator 内部一致）
d_use = float(x0[1] - x0[0])
k1d   = 2.0 * np.pi * np.fft.fftfreq(N, d=d_use)
K1d_sq = k1d ** 2
T_k   = (K1d_sq[:, None, None] +
         K1d_sq[None, :, None] +
         K1d_sq[None, None, :]) / 2.0
T_k   = np.minimum(T_k, 30.0)   # 与 build_3d_fft_operator 的 kinetic_cut 一致
T_k_flat = T_k.ravel()
V_mean   = float(V_flat.mean())

print(f"  d={d_use:.4f},  V_mean={V_mean:.4f},  T_k_mean={T_k_flat.mean():.4f}")

# ──────────────────────────────────────────────
# 预条件子工厂
# ──────────────────────────────────────────────

def make_jacobi(E_target: float) -> spla.LinearOperator:
    """对角预条件：P⁻¹x = x / |diag(H) - E_target|
    diag(T) 在实空间 = mean(T_k)（周期边界条件下对角元均等）。
    """
    T_diag = float(T_k_flat.mean())
    denom  = np.abs(V_flat + T_diag - E_target) + EPS_PRECOND
    def mv(v): return v / denom
    return spla.LinearOperator((n_un, n_un), matvec=mv, dtype=float)


def make_tpa(E_target: float) -> spla.LinearOperator:
    """Teter-Payne-Allan k 空间预条件子（Teter, Payne & Allan, PRB 40, 1989）。
    在 k 空间应用：
        f(t)   = 27 / (27 + 18t + 12t² + 8t³)   （t = T(k)/T_ref）
        P⁻¹(k) = f(t) / (|T(k) + V_mean - E| + eps)
    对低 k（物理模式）f≈1，对高 k（噪声）f→0，有效抑制高频振荡。
    """
    T_ref = max(float(T_k_flat.mean()), 0.5)
    t  = T_k / T_ref
    f  = 27.0 / (27.0 + 18.0*t + 12.0*t**2 + 8.0*t**3)
    denom = np.abs(T_k + V_mean - E_target) + EPS_PRECOND
    K  = (f / denom)                          # (N, N, N)

    def mv(v: np.ndarray) -> np.ndarray:
        psi   = v.reshape(N, N, N)
        psi_k = np.fft.fftn(psi)
        return np.fft.ifftn(K * psi_k).real.ravel()

    return spla.LinearOperator((n_un, n_un), matvec=mv, dtype=float)


def make_shifted_kinetic(E_target: float) -> spla.LinearOperator:
    """Shifted Kinetic 预条件子（纯 k 空间对角）：
        P⁻¹x = IFFT( FFT(x) / (|T(k) + V_mean - E_target| + eps) )
    直接用 T(k)+V_mean-E 近似 H-E 的 k 空间对角元。
    """
    denom = np.abs(T_k + V_mean - E_target) + EPS_PRECOND

    def mv(v: np.ndarray) -> np.ndarray:
        psi   = v.reshape(N, N, N)
        psi_k = np.fft.fftn(psi)
        return np.fft.ifftn(psi_k / denom).real.ravel()

    return spla.LinearOperator((n_un, n_un), matvec=mv, dtype=float)


PRECOND_NAMES = ["None", "Jacobi", "TPA", "ShiftedKinetic"]

def build_precond(name: str, E_target: float):
    if name == "None":           return None
    if name == "Jacobi":         return make_jacobi(E_target)
    if name == "TPA":            return make_tpa(E_target)
    if name == "ShiftedKinetic": return make_shifted_kinetic(E_target)
    raise ValueError(name)

# ──────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────
results = []

for target in TARGETS:
    print(f"\n{'='*60}")
    print(f"  target = {target}")
    print(f"{'='*60}")

    for pname in PRECOND_NAMES:
        # ── 构建带计时包装的 H ──
        H_timed = TimingLinearOperator(H_op)
        H_timed.reset()

        # ── 构建带计时包装的预条件子 ──
        P_raw   = build_precond(pname, target)
        P_timed = TimingLinearOperator(P_raw) if P_raw is not None else None

        ncv = max(80, 2 * N_LEVELS)

        # ── 运行 PRIMME ──
        print(f"  [{pname:16s}]  target={target} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            kwargs = dict(
                k          = N_LEVELS,
                which      = target,
                method     = "PRIMME_JDQMR",
                maxBlockSize = BLOCKSIZE,
                ncv        = ncv,
                tol        = TOL,
                return_stats = True,
                return_history = False,
            )
            if P_timed is not None:
                kwargs["OPinv"] = P_timed

            evals, evecs, stats = primme.eigsh(H_timed, **kwargs)
            t_wall   = time.perf_counter() - t0
            success  = True
            eval_val = float(evals[0])
            rnorm    = float(np.linalg.norm(
                H_op.matvec(evecs[:, 0]) - eval_val * evecs[:, 0]))
            n_mv     = int(stats["numMatvecs"])
            n_outer  = int(stats.get("numOuterIterations", -1))
            err_msg  = ""
        except Exception as exc:
            t_wall   = time.perf_counter() - t0
            success  = False
            eval_val = float("nan")
            rnorm    = float("nan")
            n_mv     = -1; n_outer = -1
            err_msg  = str(exc)

        t_H = H_timed.time_spent
        t_P = P_timed.time_spent if P_timed is not None else 0.0
        t_overhead = max(t_wall - t_H - t_P, 0.0)

        row = dict(
            target     = target,
            precond    = pname,
            eval       = eval_val,
            rnorm      = rnorm,
            t_wall     = t_wall,
            t_H        = t_H,
            t_P        = t_P,
            t_overhead = t_overhead,
            n_mv       = n_mv,
            n_outer    = n_outer,
            H_calls    = H_timed.call_count,
            P_calls    = P_timed.call_count if P_timed else 0,
            success    = success,
            err_msg    = err_msg,
        )
        results.append(row)

        if success:
            print(f"E={eval_val:.6f}  T={t_wall:.2f}s  "
                  f"T_H={t_H:.2f}s  T_P={t_P:.2f}s  "
                  f"T_ovhd={t_overhead:.2f}s  N_H={n_mv}")
        else:
            print(f"FAILED: {err_msg}")

# ──────────────────────────────────────────────
# 保存 JSON
# ──────────────────────────────────────────────
output = {
    "script"  : "compare_preconditioners.py",
    "datetime": TS,
    "config"  : dict(N=N, TARGETS=TARGETS, N_LEVELS=N_LEVELS,
                     BLOCKSIZE=BLOCKSIZE, TOL=TOL, EPS_PRECOND=EPS_PRECOND),
    "results" : results,
}
json_path = OUT_DIR / f"compare_precond_{TS}.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

# ──────────────────────────────────────────────
# 生成 Markdown 表格
# ──────────────────────────────────────────────
lines = []
lines.append(f"# Preconditioner Comparison  (N={N}, n_levels={N_LEVELS}, blocksize={BLOCKSIZE})\n")
lines.append(f"Generated: {TS}\n")

header = ("| Target | Preconditioner | Eigenvalue | T_wall (s) "
          "| T_H (s) | T_P (s) | T_overhead (s) "
          "| N_H matvec | N_outer | Residual |")
sep    = ("|--------|---------------|------------|------------|"
          "---------|---------|----------------|"
          "------------|---------|----------|")

for target in TARGETS:
    lines.append(f"\n## target = {target}\n")
    lines.append(header)
    lines.append(sep)
    for row in results:
        if row["target"] != target:
            continue
        if row["success"]:
            lines.append(
                f"| {row['target']:6} | {row['precond']:13} "
                f"| {row['eval']:10.6f} | {row['t_wall']:10.3f} "
                f"| {row['t_H']:7.3f} | {row['t_P']:7.3f} "
                f"| {row['t_overhead']:14.3f} "
                f"| {row['n_mv']:10d} | {row['n_outer']:7d} "
                f"| {row['rnorm']:8.2e} |"
            )
        else:
            lines.append(
                f"| {row['target']:6} | {row['precond']:13} "
                f"| FAILED | — | — | — | — | — | — | — |"
            )

md_text = "\n".join(lines) + "\n"

# 终端打印
print("\n" + md_text)

# 保存文件
md_path = OUT_DIR / f"compare_precond_{TS}.md"
with open(md_path, "w") as f:
    f.write(md_text)
print(f"Markdown saved: {md_path}")
