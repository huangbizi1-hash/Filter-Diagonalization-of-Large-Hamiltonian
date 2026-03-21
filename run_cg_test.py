"""
run_cg_test.py
==============
依次在两个体系上测试 CG-Minimization（折叠谱）求解器：
  1. 3D 谐振子   (E_target = 2.5)
  2. 高斯重构势能 (E_target = -0.7)

用法：
  python run_cg_test.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import ho3d_solvers_v2 as solver
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid

# ===========================================================
# 公共参数
# ===========================================================
N_HO     = 15      # 3D-HO 网格分辨率（每维格点数）
N_GAUSS  = 15      # 高斯势网格分辨率

N_LEVELS  = 1      # 每次求几个态（cgmin 支持多态，通过偏移投影）
CG_MAXITER  = 2000
CG_GTOL     = 1e-7
CG_THETA_MAX = 0.8
CG_VERBOSE  = True  # 打印 CG 内部迭代信息

OUT_DIR = "results_cgmin"
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================
# 辅助：打印分隔线
# ===========================================================
def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ===========================================================
# 1.  3D 谐振子  (E_target = 2.5)
# ===========================================================
section("TEST 1: 3D Harmonic Oscillator  |  E_target = 2.5")

# 3D HO 精确能级 E = (nx+ny+nz+3/2)*omega，omega=1
# E=2.5 对应 (nx+ny+nz)=1，简并度3；下一组 E=3.5 简并度6
print(f"N={N_HO}, L=5.0, n_levels={N_LEVELS}")
print(f"CG params: maxiter={CG_MAXITER}, gtol={CG_GTOL}, theta_max={CG_THETA_MAX}")
print()

t0 = time.perf_counter()
res_ho = solver.solve_ho3d(
    "sinc_dvr:cgmin",
    N=N_HO,
    L=5.0,
    n_levels=N_LEVELS,
    potential_grid=None,      # 谐振子模式
    target=2.5,
    cg_maxiter=CG_MAXITER,
    cg_gtol=CG_GTOL,
    cg_theta_max=CG_THETA_MAX,
    cg_verbose=CG_VERBOSE,
)
t_ho = time.perf_counter() - t0

print(f"\n---- 3D-HO 结果 ----")
print(f"求解耗时: {t_ho:.2f} s  (build={res_ho.meta['t_build']:.3f}s, eig={res_ho.meta['t_eig']:.3f}s)")
print(f"E_target = 2.5000000000")
for i, E in enumerate(res_ho.evals):
    # 最接近的精确 HO 能级
    n_near = int(round(E - 1.5))
    E_exact = n_near + 1.5
    print(f"  state {i}: E_cgmin = {E:.10f}  |  "
          f"nearest exact = {E_exact:.10f}  |  "
          f"error = {abs(E - E_exact):.3e}")

# 保存 CG 收敛曲线（只做第一个态）
hist_ho = res_ho.meta.get("cg_history_0")   # 暂未存入 meta，用 history 字段

# 为了方便分析，重新单独跑一次获取 history
print("\n  [重跑单态，获取收敛历史用于绘图]")
from ho3d_solvers_v2 import build_3d_sinc_dvr_operator, cg_minimize_folded
H_op_ho, n_un_ho, _ = build_3d_sinc_dvr_operator(N_HO, potential_grid=None, L=5.0)
rng = np.random.default_rng(0)
x0_ho = rng.standard_normal(n_un_ho)
res_cg_ho = cg_minimize_folded(
    H_op_ho, E_target=2.5, x0=x0_ho,
    maxiter=CG_MAXITER, gtol=CG_GTOL,
    theta_max=CG_THETA_MAX, verbose=False,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
it = np.arange(1, len(res_cg_ho["history"]["f"]) + 1)
axes[0].semilogy(it, res_cg_ho["history"]["f"])
axes[0].set_xlabel("iteration"); axes[0].set_ylabel("f(x) = ||(H-E)x||²")
axes[0].set_title("3D-HO  CG objective (E_target=2.5)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(it, res_cg_ho["history"]["E_ritz"])
axes[1].axhline(2.5, linestyle="--", color="red", label="E_target=2.5")
axes[1].set_xlabel("iteration"); axes[1].set_ylabel("Ritz energy ⟨x|H|x⟩")
axes[1].set_title("3D-HO  Ritz energy convergence")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "cgmin_ho3d.png"), dpi=200)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/cgmin_ho3d.png")


# ===========================================================
# 2.  高斯重构势能  (E_target = -0.7)
# ===========================================================
section("TEST 2: Gaussian Reconstructed Potential  |  E_target = -0.7")

print("Loading GaussianPotentialBuilder...")
builder = GaussianPotentialBuilder(
    cube_file="localPot.cube",
    params_file="gaussian_fit_params.json",
    r_cut=7.0,
)
print(f"  Spatial range: [{builder.x_min:.3f}, {builder.x_max:.3f}] Bohr")

print(f"\nBuilding potential on N={N_GAUSS} grid...")
t_pot = time.perf_counter()
x_g, y_g, z_g, V_g = builder.build_potential(N_GAUSS)
t_pot = time.perf_counter() - t_pot
potential_grid = PotentialGrid(x_g, y_g, z_g, V_g,
                               source=f"Gaussian r_cut={builder.r_cut}")
print(f"  Done in {t_pot:.3f}s  |  V range: [{V_g.min():.4f}, {V_g.max():.4f}]")

print(f"\nCG params: maxiter={CG_MAXITER}, gtol={CG_GTOL}, theta_max={CG_THETA_MAX}")
print()

t0 = time.perf_counter()
res_gauss = solver.solve_ho3d(
    "sinc_dvr:cgmin",
    N=N_GAUSS,
    L=5.0,            # 对 custom potential 无效，使用 cube 范围
    n_levels=N_LEVELS,
    potential_grid=potential_grid,
    target=-0.7,
    cg_maxiter=CG_MAXITER,
    cg_gtol=CG_GTOL,
    cg_theta_max=CG_THETA_MAX,
    cg_verbose=CG_VERBOSE,
)
t_gauss = time.perf_counter() - t0

print(f"\n---- 高斯势 结果 ----")
print(f"求解耗时: {t_gauss:.2f} s  (build={res_gauss.meta['t_build']:.3f}s, eig={res_gauss.meta['t_eig']:.3f}s)")
print(f"Spatial range: {res_gauss.meta['spatial_range']}")
print(f"E_target = -0.7000000000")
for i, E in enumerate(res_gauss.evals):
    print(f"  state {i}: E_cgmin = {E:.10f}  |  "
          f"|E - E_target| = {abs(E - (-0.7)):.3e}")

# 绘图：高斯势收敛历史
print("\n  [重跑单态，获取收敛历史用于绘图]")
from ho3d_solvers_v2 import build_3d_sinc_dvr_operator
H_op_g, n_un_g, _ = build_3d_sinc_dvr_operator(N_GAUSS, potential_grid=potential_grid)
rng2 = np.random.default_rng(1)
x0_g = rng2.standard_normal(n_un_g)
res_cg_g = cg_minimize_folded(
    H_op_g, E_target=-0.7, x0=x0_g,
    maxiter=CG_MAXITER, gtol=CG_GTOL,
    theta_max=CG_THETA_MAX, verbose=False,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
it_g = np.arange(1, len(res_cg_g["history"]["f"]) + 1)
axes[0].semilogy(it_g, res_cg_g["history"]["f"])
axes[0].set_xlabel("iteration"); axes[0].set_ylabel("f(x) = ||(H-E)x||²")
axes[0].set_title("Gaussian Potential  CG objective (E_target=-0.7)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(it_g, res_cg_g["history"]["E_ritz"])
axes[1].axhline(-0.7, linestyle="--", color="red", label="E_target=-0.7")
axes[1].set_xlabel("iteration"); axes[1].set_ylabel("Ritz energy ⟨x|H|x⟩")
axes[1].set_title("Gaussian Potential  Ritz energy convergence")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "cgmin_gaussian.png"), dpi=200)
plt.close(fig)
print(f"  Saved: {OUT_DIR}/cgmin_gaussian.png")


# ===========================================================
# 汇总
# ===========================================================
section("SUMMARY")
print(f"{'体系':<25}  {'E_target':>12}  {'E_cgmin':>15}  {'|ΔE|':>12}  {'时间(s)':>10}")
print("-" * 80)
for i, E in enumerate(res_ho.evals):
    n_near = int(round(E - 1.5)); E_ex = n_near + 1.5
    print(f"{'3D-HO (state '+str(i)+')':<25}  {2.5:>12.4f}  {E:>15.10f}  {abs(E-E_ex):>12.3e}  {t_ho:>10.2f}")
for i, E in enumerate(res_gauss.evals):
    print(f"{'Gaussian (state '+str(i)+')':<25}  {-0.7:>12.4f}  {E:>15.10f}  {abs(E-(-0.7)):>12.3e}  {t_gauss:>10.2f}")

print(f"\nPlots saved to: {OUT_DIR}/")
