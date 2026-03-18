"""
绘图函数集合。所有函数均将图像保存为文件，不调用 plt.show()。
"""
from pathlib import Path

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


def plot_filter_interpolation(El_list, an, samp, par: PhysParams,
                               interval, out_dir: Path) -> None:
    """真实滤波函数 vs Newton 插值。"""
    def _scaled_eval(x_eval, nodes, coeffs):
        x = 4.0 * (x_eval - par.Vmin) / par.dE - 2.0
        result = coeffs[0]
        basis  = 1.0
        for j in range(1, len(nodes)):
            basis  *= (x - nodes[j - 1])
            result += coeffs[j] * basis
        return result

    x_plot = np.linspace(interval[0], interval[1], 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    for ie, El in enumerate(El_list):
        y_true   = np.sqrt(par.dt / np.pi) * np.exp(-(x_plot - El)**2 * par.dt)
        y_interp = np.array([_scaled_eval(x, samp, an[ie]) for x in x_plot])
        mae = np.mean(np.abs(y_true - y_interp))
        print(f"  El={El:.2f}  MAE={mae:.6e}")
        ax.plot(x_plot, y_true,   color='blue',
                label='True filter' if ie == 0 else "")
        ax.plot(x_plot, y_interp, '--', color='red',
                label=f'Newton interp (nc={len(samp)})' if ie == 0 else "")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Filter Function vs Newton Interpolation (nc={len(samp)})')
    ax.set_xlim(interval[0], interval[1])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _savefig(fig, out_dir / "filter_interpolation.png")


def plot_filter_monomial(El_list, cn, par: PhysParams,
                         interval, out_dir: Path) -> None:
    """真实滤波函数 vs 单项式多项式（用于验证 Newton→单项式转换）。"""
    def _monomial_eval(x_eval, coeffs):
        x = 4.0 * (x_eval - par.Vmin) / par.dE - 2.0
        # Horner 法求值
        result = coeffs[-1]
        for k in range(len(coeffs) - 2, -1, -1):
            result = result * x + coeffs[k]
        return result

    x_plot = np.linspace(interval[0], interval[1], 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    for ie, El in enumerate(El_list):
        y_true   = np.sqrt(par.dt / np.pi) * np.exp(-(x_plot - El)**2 * par.dt)
        y_mono   = np.array([_monomial_eval(x, cn[ie]) for x in x_plot])
        mae = np.mean(np.abs(y_true - y_mono))
        print(f"  El={El:.4f}  MAE={mae:.6e}")
        ax.plot(x_plot, y_true, color='blue',
                label='True filter' if ie == 0 else "")
        ax.plot(x_plot, y_mono, '--', color='orange',
                label=f'Monomial (nc={len(cn[0])})' if ie == 0 else "")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Filter Function vs Monomial Polynomial (nc={len(cn[0])})')
    ax.set_xlim(interval[0], interval[1])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _savefig(fig, out_dir / "filter_monomial.png")


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
