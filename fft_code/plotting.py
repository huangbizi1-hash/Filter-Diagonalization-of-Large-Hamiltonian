"""
绘图函数集合。所有函数均将图像保存为文件，不调用 plt.show()。
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .params import PhysParams


def _savefig(fig, path: Path, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved: {path}")
    if close:
        plt.close(fig)


# ──────────────────────────────────────────────
# 滤波函数 vs Newton 插值
# ──────────────────────────────────────────────

def plot_filter_interpolation(El_list, an, samp, par: PhysParams,
                               interval, out_dir: Path,
                               filter_func: Optional[Callable] = None,
                               filter_label: str = "Filter",
                               samp_ref: Optional[np.ndarray] = None,
                               samp_ref_label: str = "ref nodes",
                               extra_components: Optional[list] = None) -> None:
    """
    真实窗函数 vs Newton 多项式插值。

    参数
    ----
    filter_func      : callable(x_arr, El) -> y_arr
        若为 None，默认使用高斯公式 sqrt(dt/π)·exp(-(x-El)²·dt)。
    filter_label     : 图例中真实函数的名称。
    samp             : Newton 节点数组（缩放坐标 [-2, 2]）。
        传 None 时跳过插值曲线与 rug plot，仅绘制真实窗函数。
        适用于 chebyshev_explosion 等无插值节点的滤波方式。
    an               : Newton 系数矩阵。samp=None 时可同时为 None。
    samp_ref         : 可选，第二组参考节点（缩放坐标 [-2, 2]）。
        若给定，则在 rug plot 中用不同颜色和 y 偏移同时显示两组节点，
        方便直观对比节点分布（如普通 Chebyshev vs density-mapped）。
    samp_ref_label   : samp_ref 的图例名称。
    extra_components : 可选，额外分量列表，每项为 (an_comp, func_comp, label_comp)。
        用于 split_bandpass 等需要同时展示多个分量插值质量的场景。
        每个分量用独立颜色对（实线=真实，虚线=Newton 插值）绘制。
        extra_components 中的分量不参与 rug plot（共用主 samp）。
    """
    has_nodes = samp is not None      # False → chebyshev_explosion 等无节点模式

    if filter_func is None:
        filter_func = lambda x, El: (
            np.sqrt(par.dt / np.pi) * np.exp(-(x - El) ** 2 * par.dt)
        )

    def _scaled_eval(x_eval, nodes, coeffs):
        x = 4.0 * (x_eval - par.Vmin) / par.dE - 2.0
        result = coeffs[0]
        basis  = 1.0
        for j in range(1, len(nodes)):
            basis  *= (x - nodes[j - 1])
            result += coeffs[j] * basis
        return result

    if has_nodes:
        # 将节点从缩放坐标 [-2, 2] 转换为物理坐标（Hartree）
        # 转换关系：x_phys = (samp + 2) * dE/4 + Vmin
        x_nodes = (samp + 2.0) * par.dE / 4.0 + par.Vmin

    x_plot = np.linspace(interval[0], interval[1], 1000)
    fig, ax = plt.subplots(figsize=(10, 6))

    # 主滤波分量（蓝色=真实，红色虚线=Newton 插值）
    for ie, El in enumerate(El_list):
        y_true = filter_func(x_plot, El)
        ax.plot(x_plot, y_true, color='blue',
                label=filter_label if ie == 0 else "")
        if has_nodes and an is not None:
            y_interp = np.array([_scaled_eval(x, samp, an[ie]) for x in x_plot])
            mae = np.mean(np.abs(y_true - y_interp))
            print(f"  El={El:.4f}  MAE={mae:.6e}")
            ax.plot(x_plot, y_interp, '--', color='red',
                    label=f'Newton interp (nc={len(samp)})' if ie == 0 else "")

    # 额外分量（split_bandpass 的高通/低通等），每个分量用独立颜色对
    _comp_colors = [('forestgreen', 'darkorange'),
                    ('purple',      'gold'),
                    ('teal',        'coral')]
    if extra_components:
        for k, (an_c, func_c, label_c) in enumerate(extra_components):
            c_true, c_interp = _comp_colors[k % len(_comp_colors)]
            for ie, El in enumerate(El_list):
                y_true = func_c(x_plot, El)
                ax.plot(x_plot, y_true, color=c_true,
                        label=label_c if ie == 0 else "")
                if has_nodes and an_c is not None:
                    y_interp = np.array([_scaled_eval(x, samp, an_c[ie]) for x in x_plot])
                    mae = np.mean(np.abs(y_true - y_interp))
                    print(f"  [{label_c}] El={El:.4f}  MAE={mae:.6e}")
                    ax.plot(x_plot, y_interp, '--', color=c_interp,
                            label=f'{label_c} Newton interp' if ie == 0 else "")

    # ── Rug plot：节点分布可视化（仅 has_nodes 时）──────────────────
    if has_nodes:
        y_lo, y_hi = ax.get_ylim()
        rug_span = 0.06 * (y_hi - y_lo)

        if samp_ref is not None:
            x_ref = (samp_ref + 2.0) * par.dE / 4.0 + par.Vmin
            mask_ref = (x_ref >= interval[0]) & (x_ref <= interval[1])
            rug_y_ref = y_lo - rug_span
            ax.plot(x_ref[mask_ref], np.full(mask_ref.sum(), rug_y_ref),
                    '|', color='steelblue', markersize=10, markeredgewidth=1.5,
                    label=f'{samp_ref_label} ({mask_ref.sum()}/{len(samp_ref)} in view)',
                    clip_on=False)
            mask = (x_nodes >= interval[0]) & (x_nodes <= interval[1])
            rug_y = y_lo - 2.0 * rug_span
            ax.plot(x_nodes[mask], np.full(mask.sum(), rug_y),
                    '|', color='darkorange', markersize=10, markeredgewidth=1.5,
                    label=f'density-mapped nodes ({mask.sum()}/{len(samp)} in view)',
                    clip_on=False)
        else:
            mask = (x_nodes >= interval[0]) & (x_nodes <= interval[1])
            rug_y = y_lo - rug_span
            ax.plot(x_nodes[mask], np.full(mask.sum(), rug_y),
                    '|', color='darkorange', markersize=10, markeredgewidth=1.5,
                    label=f'Newton nodes ({mask.sum()}/{len(samp)} in view)',
                    clip_on=False)
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel('Energy (Hartree)')
    ax.set_ylabel('f(E)')
    if has_nodes:
        ax.set_title(f'{filter_label} vs Newton Interpolation (nc={len(samp)})')
    else:
        ax.set_title(f'{filter_label}')
    ax.set_xlim(interval[0], interval[1])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _savefig(fig, out_dir / "filter_interpolation.png")


def plot_explosion_window(m: int, E_lower: float, E_upper: float,
                          out_dir: Path,
                          E_min_plot: Optional[float] = None,
                          E_max_plot: Optional[float] = None,
                          clip_val: float = 20.0) -> None:
    """
    绘制切比雪夫爆炸滤波器 T_m(aE+b) 的形状。

    参数
    ----
    m            : 切比雪夫多项式阶数
    E_lower      : 放大/抑制分界下沿
    E_upper      : 放大/抑制分界上沿
    out_dir      : 输出目录
    E_min_plot   : 绘图左边界（默认 E_lower - 0.3*(E_upper-E_lower)）
    E_max_plot   : 绘图右边界（默认 E_upper）
    clip_val     : 纵轴截断值（T_m 在域外增长迅速，截断方便显示）
    """
    dE = E_upper - E_lower
    if E_min_plot is None:
        E_min_plot = E_lower - 0.3 * dE
    if E_max_plot is None:
        E_max_plot = E_upper

    a = 2.0 / dE
    b = -(E_upper + E_lower) / dE

    E_arr = np.linspace(E_min_plot, E_max_plot, 2000)
    x_arr = a * E_arr + b         # 映射到缩放坐标

    # 用 numpy Chebyshev 精确求值
    from numpy.polynomial.chebyshev import Chebyshev
    coef = np.zeros(m + 1); coef[m] = 1.0
    Tm_vals = Chebyshev(coef)(x_arr)
    Tm_clipped = np.clip(Tm_vals, -clip_val, clip_val)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：完整域，截断纵轴
    ax = axes[0]
    ax.plot(E_arr, Tm_clipped, color='steelblue', lw=1.5)
    ax.axvline(E_lower, color='red',   ls='--', lw=1.2, label=f'E_lower={E_lower}')
    ax.axvline(E_upper, color='green', ls='--', lw=1.2, label=f'E_upper={E_upper}')
    ax.axhline(0,       color='k',     lw=0.5)
    ax.axhline( 1,      color='gray',  ls=':',  lw=0.8)
    ax.axhline(-1,      color='gray',  ls=':',  lw=0.8)
    ax.fill_betweenx([-clip_val, clip_val], E_lower, E_upper,
                     alpha=0.08, color='green', label='suppressed region')
    ax.set_xlabel('Energy (Hartree)')
    ax.set_ylabel(f'T_{m}(aE+b)  [clipped at ±{clip_val}]')
    ax.set_title(f'Chebyshev Explosion Filter  m={m}')
    ax.set_ylim(-clip_val, clip_val)
    ax.legend(fontsize=9)
    ax.grid(True)

    # 右图：仅 [-1, 1] 范围内（抑制区）放大显示
    mask_in = (E_arr >= E_lower) & (E_arr <= E_upper)
    ax2 = axes[1]
    ax2.plot(E_arr[mask_in], Tm_vals[mask_in], color='steelblue', lw=1.5)
    ax2.axhline( 1, color='gray', ls=':', lw=0.8)
    ax2.axhline(-1, color='gray', ls=':', lw=0.8)
    ax2.axhline( 0, color='k',   lw=0.5)
    ax2.set_xlabel('Energy (Hartree)')
    ax2.set_ylabel(f'T_{m}(aE+b)  [suppressed zone]')
    ax2.set_title(f'Suppressed region: E ∈ [{E_lower}, {E_upper}]')
    ax2.grid(True)

    fig.tight_layout()
    _savefig(fig, out_dir / "explosion_window.png")


# ──────────────────────────────────────────────
# 多种窗函数对比图
# ──────────────────────────────────────────────

def plot_window_comparison(
        El_list,
        par: PhysParams,
        interval: Tuple[float, float],
        out_dir: Path,
        window_funcs: Optional[Dict[str, Callable]] = None,
        highlight_band: Optional[Tuple[float, float]] = None,
        gap_band: Optional[Tuple[float, float]] = None,
) -> None:
    """
    在同一图上对比多种窗函数的形状。

    参数
    ----
    window_funcs : {label: callable(x_arr, El) -> y_arr}
        若为 None，自动用 par.dt 绘制默认高斯窗和示例 Gabor 窗。
    highlight_band : (lo, hi) — 感兴趣能量区间（绿色阴影）。
    gap_band       : (lo, hi) — 无本征值的 gap 区间（红色阴影）。
    """
    if window_funcs is None:
        from .filter_coeff import _filt_func_gaussian, _filt_func_gabor
        window_funcs = {
            f"Gaussian (sigma={1/np.sqrt(2*par.dt):.3f})": (
                lambda x, El: _filt_func_gaussian(x, El, par.dt)
            ),
            "Gabor (alpha_f=0.5, k_f=1.0)": (
                lambda x, El: _filt_func_gabor(x, El, 0.5, 1.0)
            ),
        }

    x_plot = np.linspace(interval[0], interval[1], 2000)
    colors = plt.cm.tab10(np.linspace(0, 1, len(window_funcs)))

    fig, axes = plt.subplots(1, len(El_list), figsize=(6 * len(El_list), 5),
                             squeeze=False)

    for col, El in enumerate(El_list):
        ax = axes[0][col]

        # 感兴趣区间 / gap 阴影
        if highlight_band is not None:
            ax.axvspan(highlight_band[0], highlight_band[1],
                       alpha=0.15, color='green', label='Target band')
        if gap_band is not None:
            ax.axvspan(gap_band[0], gap_band[1],
                       alpha=0.20, color='red', label='Gap (no eigenvalues)')

        for (label, func), color in zip(window_funcs.items(), colors):
            y = func(x_plot, El)
            ax.plot(x_plot, y, label=label, color=color)

        ax.axvline(El, color='black', linestyle=':', linewidth=0.8, label=f'El={El:.3f}')
        ax.set_xlabel('Energy (Hartree)')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Window functions  El={El:.4f}')
        ax.legend(fontsize=7)
        ax.grid(True)

    fig.tight_layout()
    _savefig(fig, out_dir / "window_comparison.png")


# ──────────────────────────────────────────────
# 单项式验证
# ──────────────────────────────────────────────

def plot_filter_monomial(El_list, cn, par: PhysParams,
                         interval, out_dir: Path,
                         filter_func: Optional[Callable] = None) -> None:
    """用单项式系数（Horner 法）验证 Newton→单项式转换的精度。"""
    if filter_func is None:
        filter_func = lambda x, El: (
            np.sqrt(par.dt / np.pi) * np.exp(-(x - El) ** 2 * par.dt)
        )

    def _mono_eval(x_eval, mono):
        x = 4.0 * (x_eval - par.Vmin) / par.dE - 2.0
        result = mono[-1]
        for k in range(len(mono) - 2, -1, -1):
            result = result * x + mono[k]
        return result

    x_plot = np.linspace(interval[0], interval[1], 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    for ie, El in enumerate(El_list):
        y_true = filter_func(x_plot, El)
        y_mono = np.array([_mono_eval(x, cn[ie]) for x in x_plot])
        mae = np.mean(np.abs(y_true - y_mono))
        print(f"  [monomial] El={El:.4f}  MAE={mae:.6e}")
        ax.plot(x_plot, y_true,  color='blue',
                label='True filter' if ie == 0 else "")
        ax.plot(x_plot, y_mono, '--', color='orange',
                label=f'Monomial eval (nc={len(cn[0])})' if ie == 0 else "")
    ax.set_xlabel('Energy (Hartree)')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Filter Function vs Monomial Evaluation (nc={len(cn[0])})')
    ax.set_xlim(interval[0], interval[1])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _savefig(fig, out_dir / "filter_monomial.png")


# ──────────────────────────────────────────────
# 其余绘图函数（不变）
# ──────────────────────────────────────────────

def plot_filtered_energies(El_list, E_mean_all, E_std_all, N_values,
                            error_mean_all, n_random: int,
                            out_dir: Path) -> None:
    """误差棒图：滤波能量期望值 vs 滤波中心 El。"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, N in enumerate(N_values):
        ax.errorbar(El_list, E_mean_all[i], yerr=E_std_all[i],
                    fmt='o-', capsize=5,
                    label=f'N={N}, MeanErr={error_mean_all[i]:.4f}')
    ax.set_xlabel('El')
    ax.set_ylabel('E_filtered')
    ax.set_title(f'Filtered Energy by FFT (Averaged over {n_random} Random States)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _savefig(fig, out_dir / "filtered_energies.png")


def plot_energy_levels(energies_filtered: np.ndarray,
                       exact_energies: np.ndarray,
                       out_dir: Path) -> None:
    """左右并排水平能级图。"""
    energies_filtered = np.sort(energies_filtered)
    fig, ax = plt.subplots(figsize=(5, 6))
    for i, e in enumerate(energies_filtered):
        ax.hlines(e, 0, 0.45, color='blue', linestyle='-',
                  label='FD Energies' if i == 0 else "")
    for i, e in enumerate(exact_energies):
        ax.hlines(e, 0.55, 1.0, color='red', linestyle='-',
                  label='Exact Energies' if i == 0 else "")
    ax.set_xlabel('Category')
    ax.set_ylabel('Energy')
    ax.set_xticks([0.225, 0.775])
    ax.set_xticklabels(['FD Energies', 'Exact Energies'])
    ax.grid(True, axis='y')
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir / "energy_levels.png")


def plot_energy_errors(energies_dict, exact_energies: np.ndarray,
                       out_dir: Path) -> None:
    """对数散点图：|E_FD - E_exact| vs 最近精确能级。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(energies_dict)))
    for (label, energies), color in zip(energies_dict.items(), colors):
        energies = np.sort(energies)
        nearest  = exact_energies[
            np.argmin(np.abs(energies[:, None] - exact_energies[None, :]), axis=1)]
        errors   = np.abs(energies - nearest)
        ax.scatter(nearest, errors, color=color, alpha=0.5, s=30, label=label)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yscale('log')
    ax.set_xlabel('Nearest Exact Energy Level')
    ax.set_ylabel('|FD - Exact|')
    ax.set_title('Energy Error per FD Level')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir / "energy_errors.png")


def plot_potential_slice(V: np.ndarray, x: np.ndarray, z: np.ndarray,
                          out_dir: Path) -> None:
    """势能在 y=0 截面的二维色图。"""
    mid = V.shape[1] // 2
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(x, z, V[:, mid, :].T, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='V (a.u.)')
    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('z (a.u.)')
    ax.set_title('Potential slice at y = 0')
    fig.tight_layout()
    _savefig(fig, out_dir / "potential_slice.png")
