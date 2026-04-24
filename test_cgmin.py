"""
test_cgmin.py
验证 ho3d_solvers_v2 中 cgmin 求解器，并比较 Sinc-DVR 与 FFT-DVR 的耗时。

依次测试：
  1. 3D 谐振子 (omega=1)，E_target=2.5
     精确值：E = (nx+ny+nz+3/2)，第一激发简并族 E=2.5
  2. 高斯重构势能，E_target=-0.7
     参考值约 -0.67

对每个势能，分别用 sinc_dvr:cgmin 和 fft_dvr:cgmin 计算，比较结果与耗时。
"""

import time
import numpy as np
import ho3d_solvers_v2 as solver

# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run_single(method_spec, N, E_target, potential_grid=None, L=5.0):
    """运行一次 cgmin，返回 (E_ritz, t_build, t_eig, n_iter)。"""
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
        cgmin_verbose=False,   # 关掉每步打印，方便看对比表格
    )
    t_total = time.perf_counter() - t0
    return res.evals[0], res.meta["t_build"], res.meta["t_eig"], t_total


def compare(label, N, E_target, E_ref, potential_grid=None, L=5.0):
    """对同一问题比较 sinc_dvr 和 fft_dvr。"""
    print(f"\n  势能/参数  : {label}  (N={N}, E_target={E_target}, ref≈{E_ref})")
    print(f"  {'方法':<20}  {'E_ritz':>14}  {'|E-ref|':>10}  "
          f"{'t_build':>9}  {'t_matvec':>9}  {'t_total':>9}  {'Pass?':>6}")
    print(f"  {'-'*85}")

    results = {}
    for method in ("sinc_dvr:cgmin", "fft_dvr:cgmin"):
        try:
            E, tb, te, tt = run_single(method, N, E_target, potential_grid, L)
            diff = abs(E - E_ref)
            ok = "PASS" if diff < 0.05 else "FAIL"
            print(f"  {method:<20}  {E:>14.8f}  {diff:>10.4e}  "
                  f"{tb:>9.2f}s  {te:>9.2f}s  {tt:>9.2f}s  {ok:>6}")
            results[method] = dict(E=E, t_build=tb, t_eig=te, t_total=tt)
        except Exception as exc:
            print(f"  {method:<20}  ERROR: {exc}")
            results[method] = None

    # 速度比较
    sinc = results.get("sinc_dvr:cgmin")
    fft  = results.get("fft_dvr:cgmin")
    if sinc and fft and sinc["t_eig"] > 0:
        ratio = sinc["t_eig"] / fft["t_eig"]
        print(f"\n  ⚡ matvec 耗时比 sinc_dvr / fft_dvr = {ratio:.2f}x  "
              f"({'FFT 更快' if ratio > 1 else 'Sinc-DVR 更快'})")
    return results


# ──────────────────────────────────────────────
# 测试 1：3D 谐振子
# ──────────────────────────────────────────────

print_section("测试 1 / 3D 谐振子  (E_target=2.5, 精确值=2.5)")
r1 = compare(
    label="Harmonic Oscillator",
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

    print(f"\n  势能空间范围: [{x[0]:.3f}, {x[-1]:.3f}] Bohr  |  N={N_gauss}³={N_gauss**3} 点")

    r2 = compare(
        label="Gaussian Potential",
        N=N_gauss,
        E_target=-0.7,
        E_ref=-0.67,
        potential_grid=pot_grid,
    )

except FileNotFoundError as e:
    print(f"\n  ⚠ 跳过测试 2：找不到文件 ({e})")
    print("    请确保 localPot.cube 和 gaussian_fit_params.json 在当前目录。")
    r2 = None

# ──────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────

print_section("汇总")

def _fmt(d, method):
    if d is None:
        return "skipped"
    r = d.get(method)
    if r is None:
        return "ERROR"
    return f"E={r['E']:.6f}  t_matvec={r['t_eig']:.2f}s"

print(f"  测试 1 (HO3D):")
print(f"    sinc_dvr:cgmin  →  {_fmt(r1, 'sinc_dvr:cgmin')}")
print(f"    fft_dvr:cgmin   →  {_fmt(r1, 'fft_dvr:cgmin')}")
if r2 is not None:
    print(f"  测试 2 (Gaussian):")
    print(f"    sinc_dvr:cgmin  →  {_fmt(r2, 'sinc_dvr:cgmin')}")
    print(f"    fft_dvr:cgmin   →  {_fmt(r2, 'fft_dvr:cgmin')}")
else:
    print(f"  测试 2 (Gaussian): 已跳过")
