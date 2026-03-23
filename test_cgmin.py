"""
test_cgmin.py
验证 ho3d_solvers_v2 中新增的 cgmin 求解器。

依次测试：
  1. 3D 谐振子 (omega=1)，E_target=2.5
     精确值：E = (n_x+n_y+n_z+3/2)，第一激发简并族 E=2.5
  2. 高斯重构势能，E_target=-0.7
     参考值约 -0.67
"""

import time
import numpy as np
import ho3d_solvers_v2 as solver

# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def run_and_report(label, method_spec, N, E_target, E_ref, potential_grid=None, L=5.0):
    print(f"\n  方法  : {method_spec}  (N={N})")
    print(f"  E_target = {E_target},  参考值 ≈ {E_ref}")
    t0 = time.perf_counter()
    res = solver.solve_ho3d(
        method_spec,
        N=N,
        n_levels=1,
        potential_grid=potential_grid,
        L=L,
        target=E_target,
        cgmin_maxiter=2000,
        cgmin_gtol=1e-7,
        cgmin_verbose=True,
    )
    dt = time.perf_counter() - t0
    E_got = res.evals[0]
    print(f"\n  ── 结果 ──")
    print(f"  E_ritz   = {E_got:.10f}")
    print(f"  E_ref    = {E_ref}")
    print(f"  |差值|   = {abs(E_got - E_ref):.4e}")
    print(f"  耗时     = {dt:.2f} s")
    ok = abs(E_got - E_ref) < 0.05
    print(f"  验证     : {'PASS ✓' if ok else 'FAIL ✗  (差值超过 0.05)'}")
    return E_got, dt


# ──────────────────────────────────────────────
# 测试 1：3D 谐振子
# ──────────────────────────────────────────────

print_section("测试 1 / 3D 谐振子  (E_target=2.5, 精确值=2.5)")

# N=15 → 3375 个网格点，精度已足够
E1, t1 = run_and_report(
    label="HO3D",
    method_spec="sinc_dvr:cgmin",
    N=15,
    E_target=2.5,
    E_ref=2.5,
    potential_grid=None,
    L=6.0,
)

# ──────────────────────────────────────────────
# 测试 2：高斯重构势能
# ──────────────────────────────────────────────

print_section("测试 2 / 高斯重构势能  (E_target=-0.7, 参考值≈-0.67)")

try:
    from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid

    N_gauss = 20
    builder = GaussianPotentialBuilder(
        cube_file="localPot.cube",
        params_file="gaussian_fit_params.json",
        r_cut=7.0,
    )
    x, y, z, V = builder.build_potential(N_gauss)
    pot_grid = PotentialGrid(x, y, z, V, source="gaussian_files")

    print(f"\n  势能网格空间范围: [{x[0]:.3f}, {x[-1]:.3f}] Bohr")
    print(f"  网格大小: {N_gauss}³ = {N_gauss**3} 点")

    E2, t2 = run_and_report(
        label="Gaussian",
        method_spec="sinc_dvr:cgmin",
        N=N_gauss,
        E_target=-0.7,
        E_ref=-0.67,
        potential_grid=pot_grid,
    )

except FileNotFoundError as e:
    print(f"\n  ⚠ 跳过测试 2：找不到文件 ({e})")
    print("    请确保 localPot.cube 和 gaussian_fit_params.json 在当前目录。")
    E2, t2 = None, None

# ──────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────

print_section("汇总")
print(f"  测试 1 (HO3D)    : E={E1:.6f}  (ref=2.5)      耗时={t1:.2f}s")
if E2 is not None:
    print(f"  测试 2 (Gaussian): E={E2:.6f}  (ref≈-0.67)   耗时={t2:.2f}s")
else:
    print("  测试 2 (Gaussian): 已跳过")
