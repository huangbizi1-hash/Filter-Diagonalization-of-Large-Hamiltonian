"""
explosion_demo.py
切比雪夫爆炸滤波器演示 —— 3D 谐振子。

算法
----
1. 构建 3D 调和势（V = ½(x²+y²+z²)），L=5，N=20。
2. 对 n_states 个随机波函数逐一应用 T_m(H_scaled)：
       H_scaled = a·H + b，将 [E_lower, E_upper] 映射到 [-1, 1]
   低于 E_lower 的本征态 → x < -1 → T_m 迅速放大（幂次增长）。
3. 把所有滤波后向量堆成矩阵，SVD + Rayleigh-Ritz 提取 Ritz 能量。
4. 筛选 E < E_lower 的 Ritz 值，与精确谐振子能量比较。

精确能级（ω=ħ=m=1）：
    E_{nxnynz} = nx + ny + nz + 3/2
    < 4.0：1.5(×1), 2.5(×3), 3.5(×6) → 共 10 个能级

输出文件（figs_explosion/）
--------------------------
explosion_window.png    —— T_m(aE+b) 形状（全域 + 抑制区放大）
filter_interpolation.png—— 窗函数图（无插值节点）
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fft_code.grid       import build_grid_box, build_k_diagonal
from fft_code.hamiltonian import apply_H, apply_chebyshev_explosion
from fft_code.wavefunction import normalize_psi, random_sine_psi
from fft_code.rayleigh_ritz import svd_rayleigh_ritz
from fft_code.plotting    import plot_explosion_window, plot_filter_interpolation
from fft_code.params      import PhysParams

# ──────────────────────────────────────────────
# 配置参数
# ──────────────────────────────────────────────
L        = 5       # 盒子半长（box 长 = 2L = 10 Bohr）
N        = 20      # 每轴网格点数
E_lower  = 4.0    # 放大/抑制分界（我们想要 E < E_lower 的态）
E_upper  = 30.0   # 抑制上界（覆盖全谱高端）
m        = 40      # 切比雪夫多项式阶数（越大滤波越锐利，每个态 m 次 H-apply）
n_states = 60      # 随机初态数量（越多子空间越完整）
svd_tol  = 1e-3    # Rayleigh-Ritz SVD 截断阈值
out_dir  = Path("figs_explosion")
out_dir.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ──────────────────────────────────────────────
# 精确能级（参考）
# ──────────────────────────────────────────────
def ho3d_exact(n_max=10):
    """3D 谐振子精确能级 E = nx+ny+nz+1.5，按升序。"""
    levels = []
    for nx in range(n_max):
        for ny in range(n_max):
            for nz in range(n_max):
                E = nx + ny + nz + 1.5
                if E < E_lower + 5:   # 多取几个用于对比
                    levels.append(E)
    return np.unique(levels)

exact_all = ho3d_exact()
exact_below = exact_all[exact_all < E_lower]
print("精确能级（E < E_lower）:", exact_below)

# ──────────────────────────────────────────────
# 构建网格与势能
# ──────────────────────────────────────────────
print(f"\n构建网格：N={N}, L={L}  （盒长 2L={2*L}，步长 d={2*L/N:.3f}）")
x, y, z, X, Y, Z, x_grid = build_grid_box(N, L=2*L)
V = 0.5 * (X**2 + Y**2 + Z**2)

T_k = build_k_diagonal(x_grid)

print(f"  势能范围：[{V.min():.3f}, {V.max():.3f}] Hartree")
print(f"  动能最大值：{T_k.max():.3f} Hartree")

# ──────────────────────────────────────────────
# 绘制滤波窗形状
# ──────────────────────────────────────────────
print(f"\n绘制切比雪夫爆炸窗（m={m}）...")
plot_explosion_window(m, E_lower, E_upper, out_dir,
                      E_min_plot=0.0, E_max_plot=E_upper,
                      clip_val=50.0)

# 用 plot_filter_interpolation（samp=None 模式）再画一张简单窗函数图
from numpy.polynomial.chebyshev import Chebyshev as _Cheb
_a = 2.0 / (E_upper - E_lower)
_b = -(E_upper + E_lower) / (E_upper - E_lower)
_coef = np.zeros(m + 1); _coef[m] = 1.0
_Tm   = _Cheb(_coef)
_par  = PhysParams(dE=E_upper - E_lower, Vmin=0.0, dt=1.0)  # 仅供 plot 内部用
plot_filter_interpolation(
    El_list=[E_lower],   # El 不影响 Chebyshev explosion（函数无 El 依赖）
    an=None,
    samp=None,           # ← 无插值节点
    par=_par,
    interval=[0.0, E_upper],
    out_dir=out_dir,
    filter_func=lambda x_arr, El: np.clip(_Tm(_a * x_arr + _b), -50, 50),
    filter_label=f"Chebyshev Explosion T_{m}",
)

# ──────────────────────────────────────────────
# 应用滤波器
# ──────────────────────────────────────────────
print(f"\n应用滤波器 T_{m}(H_scaled) 到 {n_states} 个随机态...")
print(f"  E_lower={E_lower}, E_upper={E_upper}")
print(f"  每个态需要 {m} 次 H-apply，共 {n_states * m} 次")

t0 = time.perf_counter()
filtered = []
for i in range(n_states):
    psi0 = random_sine_psi(X, Y, Z, rng=rng).real
    psi_f = apply_chebyshev_explosion(psi0, V, T_k, m, E_lower, E_upper)
    filtered.append(psi_f)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{n_states}  {time.perf_counter()-t0:.1f}s")

t_filter = time.perf_counter() - t0
print(f"滤波完成，耗时 {t_filter:.2f}s")

# ──────────────────────────────────────────────
# SVD + Rayleigh-Ritz
# ──────────────────────────────────────────────
print(f"\nSVD + Rayleigh-Ritz（svd_tol={svd_tol}）...")
# 堆成 (n_states, Nx, Ny, Nz) 矩阵
filtered_matrix = np.stack(filtered, axis=0)  # (n_states, N, N, N)

t1 = time.perf_counter()
energies, Ur, rank = svd_rayleigh_ritz(
    filtered_matrix, x_grid, V, N, N, N, T_k,
    svd_tol=svd_tol, max_energies=n_states,
)
t_ritz = time.perf_counter() - t1
print(f"  Rayleigh-Ritz 完成，秩 r={rank}，耗时 {t_ritz:.2f}s")

# ──────────────────────────────────────────────
# 结果对比
# ──────────────────────────────────────────────
ritz_below = energies[energies < E_lower]

print(f"\n{'='*55}")
print(f"  Ritz 能量（E < {E_lower}，共 {len(ritz_below)} 个）")
print(f"{'='*55}")
print(f"  {'#':>3}  {'Ritz E':>12}  {'精确 E':>12}  {'误差':>10}")
print(f"  {'-'*50}")
for i, E_r in enumerate(ritz_below):
    if i < len(exact_below):
        E_ex = exact_below[i]
        print(f"  {i:>3}  {E_r:>12.6f}  {E_ex:>12.6f}  {abs(E_r-E_ex):>10.3e}")
    else:
        print(f"  {i:>3}  {E_r:>12.6f}  {'—':>12}  {'—':>10}")

# 统计
n_exact = len(exact_below)
n_found = len(ritz_below)
print(f"\n  应有 {n_exact} 个，找到 {n_found} 个")
if n_found > 0:
    errors = np.abs(ritz_below[:min(n_found, n_exact)] - exact_below[:min(n_found, n_exact)])
    print(f"  最大误差：{errors.max():.3e}")
    print(f"  平均误差：{errors.mean():.3e}")

# ──────────────────────────────────────────────
# 绘制 Ritz 能量 vs 精确能级
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(ritz_below, np.zeros_like(ritz_below),
           marker='|', s=300, linewidths=2, color='steelblue', label='Ritz values')
ax.scatter(exact_below, np.ones_like(exact_below) * 0.3,
           marker='|', s=300, linewidths=2, color='red', label='Exact (HO)')
ax.set_yticks([])
ax.set_xlabel('Energy (Hartree)')
ax.set_title(f'Chebyshev Explosion m={m}: Ritz vs Exact  (E < {E_lower})')
ax.axvline(E_lower, color='gray', ls='--', lw=1, label=f'E_lower={E_lower}')
ax.legend()
ax.grid(True, axis='x')
fig.tight_layout()
path = out_dir / "ritz_vs_exact.png"
path.parent.mkdir(exist_ok=True)
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  图像已保存至 {path}")
print(f"\n全部完成，总耗时 {t_filter+t_ritz:.2f}s")
