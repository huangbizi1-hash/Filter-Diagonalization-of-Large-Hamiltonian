"""
explosion_demo2.py
两级切比雪夫爆炸滤波器演示 —— 3D 谐振子。

两级滤波策略
------------
第一级 [E_lower1=4.0, E_upper=30.0]：
    T_{m1}(H_s1) 将 E < 4.0 的态放大，[4.0, 30.0] 被抑制。
    → set1：主要富含 E < 4.0 的低能分量。

第二级 [E_lower2=6.0, E_upper=30.0]，作用在 set1 上：
    T_{m2}(H_s2) · T_{m1}(H_s1)|ψ⟩
    对各能量区间的联合放大因子：
        E < 4.0        → 两个 T_m 均放大 → 极强
        E ∈ [4.0, 6.0) → T_{m1} ≤ 1，T_{m2} 放大 → 净正放大
        E ≥ 6.0        → 两个 T_m 均抑制
    → set2：E ∈ [4.0, 6.0) 分量相对 set1 被二次凸显。

将 set1 + set2 合并后做 SVD + Rayleigh-Ritz，期望同时找到：
    E < 4.0  的态：1.5(×1), 2.5(×3), 3.5(×6)           共 10 个
    E ∈ [4.0,6.0) 的态：4.5(×10), 5.5(×15)              共 25 个

输出文件（figs_explosion2/）
---------------------------
combined_filter.png  — 两级联合放大因子 |f1(E)·f2(E)| 与各自分量
ritz_vs_exact.png    — Ritz 能量 vs 精确能级散点图
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev as _Cheb

from fft_code.grid        import build_grid_box, build_k_diagonal
from fft_code.hamiltonian  import apply_H, apply_chebyshev_explosion
from fft_code.wavefunction import random_sine_psi
from fft_code.rayleigh_ritz import svd_rayleigh_ritz

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
L        = 5
N        = 20
E_upper  = 70.0

E_lower1 = 4.0    # 第一级滤波下界
E_lower2 = 6.0    # 第二级滤波下界

m1 = 40           # 第一级多项式阶数
m2 = 40           # 第二级多项式阶数

n_states = 120    # 随机初态数（多留些，保证低能子空间覆盖充分）
svd_tol  = 1e-3
out_dir  = Path("figs_explosion2")
out_dir.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ──────────────────────────────────────────────
# 精确能级参考
# ──────────────────────────────────────────────
def ho3d_exact_levels(E_cut):
    levels = {}
    for nx in range(15):
        for ny in range(15):
            for nz in range(15):
                E = nx + ny + nz + 1.5
                if E <= E_cut:
                    levels.setdefault(round(E, 6), 0)
                    levels[round(E, 6)] += 1
    return sorted(levels.items())   # [(E, degeneracy), ...]

exact_table = ho3d_exact_levels(E_lower2 + 0.5)
print("精确能级（E ≤ {:.1f}）：".format(E_lower2))
print(f"  {'E':>8}  {'简并度':>6}  {'< E_lower1':>10}  {'< E_lower2':>10}")
for E, deg in exact_table:
    t1 = "✓" if E < E_lower1 else ""
    t2 = "✓" if E < E_lower2 else ""
    print(f"  {E:>8.4f}  {deg:>6d}  {t1:>10}  {t2:>10}")

exact_flat1 = np.array([E for E, deg in exact_table for _ in range(deg) if E < E_lower1])
exact_flat2 = np.array([E for E, deg in exact_table for _ in range(deg) if E < E_lower2])

# ──────────────────────────────────────────────
# 构建网格与势能
# ──────────────────────────────────────────────
print(f"\n构建网格：N={N}, L={L}...")
x, y, z, X, Y, Z, x_grid = build_grid_box(N, L=2*L)
V = 0.5 * (X**2 + Y**2 + Z**2)
T_k = build_k_diagonal(x_grid)
d = (2 * L) / N

def ritz_energy(psi):
    H_psi  = apply_H(psi, V, T_k)
    norm2  = np.sum(psi**2)    * d**3
    expval = np.sum(psi * H_psi) * d**3
    return expval / norm2 if norm2 > 0 else float('nan')

# ──────────────────────────────────────────────
# 绘制联合放大因子
# ──────────────────────────────────────────────
def _cheb_val(m, E_lo, E_up, E_arr):
    a = 2.0 / (E_up - E_lo)
    b = -(E_up + E_lo) / (E_up - E_lo)
    coef = np.zeros(m + 1); coef[m] = 1.0
    return _Cheb(coef)(a * E_arr + b)

E_arr = np.linspace(0.0, E_upper, 3000)
f1 = _cheb_val(m1, E_lower1, E_upper, E_arr)
f2 = _cheb_val(m2, E_lower2, E_upper, E_arr)
combined = f1 * f2
CLIP = 100.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(E_arr, np.clip(np.abs(f1),       0, CLIP), color='steelblue',
        lw=1.2, label=f'|T_{m1}(H_s1)|  E_lower={E_lower1}')
ax.plot(E_arr, np.clip(np.abs(f2),       0, CLIP), color='darkorange',
        lw=1.2, label=f'|T_{m2}(H_s2)|  E_lower={E_lower2}', ls='--')
ax.plot(E_arr, np.clip(np.abs(combined), 0, CLIP), color='crimson',
        lw=1.8, label='|f1·f2|  (joint)')
ax.axvline(E_lower1, color='steelblue', ls=':', lw=1)
ax.axvline(E_lower2, color='darkorange', ls=':', lw=1)
ax.axhline(1, color='gray', ls=':', lw=0.8)
ax.fill_betweenx([0, CLIP], E_lower1, E_lower2, alpha=0.07,
                 color='darkorange', label=f'[{E_lower1},{E_lower2}) Target interval')
ax.set_ylim(0, CLIP)
ax.set_xlabel('Energy (Hartree)')
ax.set_ylabel(f'|filter(E)| (clipped at {CLIP})')
ax.set_title('Two-stage joint amplification')
ax.legend(fontsize=8)
ax.grid(True)

# 右图：仅 [0, E_lower2] 区间，线性尺度更清楚
ax2 = axes[1]
mask = E_arr <= E_lower2 + 0.5
ax2.plot(E_arr[mask], np.clip(np.abs(f1[mask]),       0, CLIP),
         color='steelblue', lw=1.2, label=f'|T_{m1}|')
ax2.plot(E_arr[mask], np.clip(np.abs(f2[mask]),       0, CLIP),
         color='darkorange', lw=1.2, ls='--', label=f'|T_{m2}|')
ax2.plot(E_arr[mask], np.clip(np.abs(combined[mask]), 0, CLIP),
         color='crimson', lw=1.8, label='|f1·f2|')
ax2.axvline(E_lower1, color='steelblue', ls=':', lw=1, label=f'E_lower1={E_lower1}')
ax2.axvline(E_lower2, color='darkorange', ls=':', lw=1, label=f'E_lower2={E_lower2}')
ax2.axhline(1, color='gray', ls=':', lw=0.8)
# 标记精确能级
for E, deg in exact_table:
    ax2.axvline(E, color='green', ls='-', lw=0.4, alpha=0.6)
ax2.set_ylim(0, CLIP)
ax2.set_xlabel('Energy (Hartree)')
ax2.set_title(f'Zoom (E ≤ {E_lower2+0.5}, green lines = Exact eigenlevels)')
ax2.legend(fontsize=8)
ax2.grid(True)

fig.tight_layout()
path_cf = out_dir / "combined_filter.png"
fig.savefig(path_cf, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n联合放大因子图已保存：{path_cf}")

# ──────────────────────────────────────────────
# 第一级滤波
# ──────────────────────────────────────────────
print(f"\n第一级滤波 T_{m1}(H_s1)，E_lower={E_lower1}，共 {n_states} 个态...")
t0 = time.perf_counter()
set1        = []
ritz1_after = []
norms1      = []
for i in range(n_states):
    psi0  = random_sine_psi(X, Y, Z, rng=rng).real
    psi_f = apply_chebyshev_explosion(psi0, V, T_k, m1, E_lower1, E_upper)
    # 归一化，保证第二级滤波数值稳定
    nrm = np.sqrt(np.sum(psi_f**2) * d**3)
    norms1.append(nrm)
    psi_f_norm = psi_f / nrm if nrm > 1e-30 else psi_f
    set1.append(psi_f_norm)
    ritz1_after.append(ritz_energy(psi_f_norm))
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{n_states}  {time.perf_counter()-t0:.1f}s")
t1_done = time.perf_counter() - t0
print(f"  第一级完成，耗时 {t1_done:.2f}s")

print(f"\n{'─'*65}")
print(f"  第一级滤波后 Ritz 能量（归一化后）")
print(f"  {'#':>3}  {'E_after1':>10}  {'norm_before_norm':>18}")
print(f"{'─'*65}")
for i in range(n_states):
    marker = " ← target" if ritz1_after[i] < E_lower1 else ""
    print(f"  {i:>3}  {ritz1_after[i]:>10.4f}  {norms1[i]:>18.3e}{marker}")
print(f"{'─'*65}")

# ──────────────────────────────────────────────
# 第二级滤波（作用在 set1 上）
# ──────────────────────────────────────────────
print(f"\n第二级滤波 T_{m2}(H_s2)，E_lower={E_lower2}，作用在归一化 set1 上...")
t0 = time.perf_counter()
set2        = []
ritz2_after = []
norms2      = []
for i, psi_s1 in enumerate(set1):
    psi_s2 = apply_chebyshev_explosion(psi_s1, V, T_k, m2, E_lower2, E_upper)
    nrm = np.sqrt(np.sum(psi_s2**2) * d**3)
    norms2.append(nrm)
    set2.append(psi_s2)   # 不归一化，保留幅度信息供 SVD 使用
    ritz2_after.append(ritz_energy(psi_s2 / nrm) if nrm > 1e-30 else float('nan'))
    if (i + 1) % 30 == 0:
        print(f"  {i+1}/{n_states}  {time.perf_counter()-t0:.1f}s")
t2_done = time.perf_counter() - t0
print(f"  第二级完成，耗时 {t2_done:.2f}s")

print(f"\n{'─'*65}")
print(f"  第二级滤波后 Ritz 能量（T_m2 作用在归一化 set1 上）")
print(f"  {'#':>3}  {'E_after1':>10}  {'E_after2':>10}  {'norm2':>12}")
print(f"{'─'*65}")
for i in range(n_states):
    m2_marker = " ← in [E1,E2)" if E_lower1 <= ritz2_after[i] < E_lower2 else (
                " ← <E1"        if ritz2_after[i] < E_lower1 else "")
    print(f"  {i:>3}  {ritz1_after[i]:>10.4f}  {ritz2_after[i]:>10.4f}  {norms2[i]:>12.3e}{m2_marker}")
print(f"{'─'*65}")

# ──────────────────────────────────────────────
# 合并子空间：set1 + set2
# ──────────────────────────────────────────────
all_states = set1 + set2          # 2 * n_states 个向量
print(f"\n合并子空间：set1({len(set1)}) + set2({len(set2)}) = {len(all_states)} 个向量")

# ──────────────────────────────────────────────
# SVD + Rayleigh-Ritz
# ──────────────────────────────────────────────
print(f"\nSVD + Rayleigh-Ritz（svd_tol={svd_tol}）...")
filtered_matrix = np.stack(all_states, axis=0)   # (2*n_states, N, N, N)
t0 = time.perf_counter()
energies, Ur, rank = svd_rayleigh_ritz(
    filtered_matrix, x_grid, V, N, N, N, T_k,
    svd_tol=svd_tol, max_energies=len(all_states),
)
t_ritz = time.perf_counter() - t0
print(f"  秩 r={rank}，耗时 {t_ritz:.2f}s")

# ── 全部 Ritz 值 ──
print(f"\n全部 Ritz 能量（共 {len(energies)} 个）：")
for i, E in enumerate(energies):
    marker = (" ← <E1"         if E < E_lower1 else
              " ← [E1,E2)"     if E < E_lower2 else "")
    print(f"  {i:>3}  {E:>12.6f}{marker}")

# ──────────────────────────────────────────────
# 结果对比
# ──────────────────────────────────────────────
ritz1 = energies[energies < E_lower1]
ritz2 = energies[(energies >= E_lower1) & (energies < E_lower2)]

print(f"\n{'='*60}")
print(f"  一级目标：E < {E_lower1}（应有 {len(exact_flat1)} 个）")
print(f"{'='*60}")
print(f"  {'#':>3}  {'Ritz E':>12}  {'精确 E':>12}  {'误差':>10}")
for i, E_r in enumerate(ritz1):
    E_ex = exact_flat1[i] if i < len(exact_flat1) else float('nan')
    err  = abs(E_r - E_ex) if not np.isnan(E_ex) else float('nan')
    print(f"  {i:>3}  {E_r:>12.6f}  {E_ex:>12.6f}  {err:>10.3e}")
if len(ritz1) != len(exact_flat1):
    print(f"  !! 找到 {len(ritz1)} 个，期望 {len(exact_flat1)} 个")

print(f"\n{'='*60}")
print(f"  二级目标：E ∈ [{E_lower1}, {E_lower2})（应有 {len(exact_flat2)-len(exact_flat1)} 个）")
print(f"{'='*60}")
exact_mid = exact_flat2[len(exact_flat1):]
print(f"  {'#':>3}  {'Ritz E':>12}  {'精确 E':>12}  {'误差':>10}")
for i, E_r in enumerate(ritz2):
    E_ex = exact_mid[i] if i < len(exact_mid) else float('nan')
    err  = abs(E_r - E_ex) if not np.isnan(E_ex) else float('nan')
    print(f"  {i:>3}  {E_r:>12.6f}  {E_ex:>12.6f}  {err:>10.3e}")
if len(ritz2) != len(exact_mid):
    print(f"  !! 找到 {len(ritz2)} 个，期望 {len(exact_mid)} 个")

# ──────────────────────────────────────────────
# 绘制 Ritz vs 精确
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

ax.scatter(ritz1, np.full_like(ritz1, 0.0),
           marker='|', s=400, linewidths=2, color='steelblue',
           label=f'Ritz (E<{E_lower1}): {len(ritz1)}')
ax.scatter(ritz2, np.full_like(ritz2, 0.0),
           marker='|', s=400, linewidths=2, color='darkorange',
           label=f'Ritz ({E_lower1}≤E<{E_lower2}): {len(ritz2)}')
ax.scatter(exact_flat1, np.full_like(exact_flat1, 0.4),
           marker='|', s=400, linewidths=2, color='navy',
           label=f'Exact (E<{E_lower1}): {len(exact_flat1)}')
ax.scatter(exact_mid,   np.full_like(exact_mid, 0.4),
           marker='|', s=400, linewidths=2, color='chocolate',
           label=f'Exact ({E_lower1}≤E<{E_lower2}): {len(exact_mid)}')

ax.axvline(E_lower1, color='steelblue',  ls='--', lw=1.2, label=f'E_lower1={E_lower1}')
ax.axvline(E_lower2, color='darkorange', ls='--', lw=1.2, label=f'E_lower2={E_lower2}')
ax.set_yticks([])
ax.set_xlabel('Energy (Hartree)')
ax.set_title(f'Two-stage Chebyshev explosion filter m1={m1}, m2={m2}: Ritz vs Exact')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, axis='x')
ax.set_xlim(0, E_lower2 + 0.5)
fig.tight_layout()
path_rv = out_dir / "ritz_vs_exact.png"
fig.savefig(path_rv, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n图像保存：{path_rv}")
print(f"\n全部完成，总耗时 {t1_done+t2_done+t_ritz:.1f}s")
