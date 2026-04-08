"""
data.py — 波函数生成

提供两种波函数类型（高斯波包叠加 / 正弦波叠加）及统一接口。
所有函数返回 (psi_sparse, H_psi_target)：
  psi_sparse   : 稀疏网格上的波函数值  (N_sparse, N_sparse, N_sparse)
  H_psi_target : 精确 H|ψ⟩ 在稀疏网格上的值（用 FFT 精细网格计算后下采样）
"""

import numpy as np
from .physics import (
    X_f, Y_f, Z_f, L, fft_hamiltonian,
)


def gen_gaussian_wavefunction():
    """
    3 个随机高斯波包叠加（fine grid），
    返回 (psi_sparse, H_psi_target)。
    """
    psi_fine = np.zeros_like(X_f, dtype=np.float64)
    for _ in range(3):
        cx, cy, cz = (np.random.rand(3) - 0.5) * L * 0.5
        width      = 0.5 + np.random.rand() * 0.5
        psi_fine  += np.exp(
            -((X_f - cx)**2 + (Y_f - cy)**2 + (Z_f - cz)**2)
            / (2 * width**2))

    H_psi_fine   = fft_hamiltonian(psi_fine)
    psi_sparse   = psi_fine[::2, ::2, ::2]
    H_psi_target = H_psi_fine[::2, ::2, ::2]
    return psi_sparse, H_psi_target


def gen_sine_wavefunction(k_max: int = 2):
    """
    3–6 个正弦模式叠加：sin(k·r + b)，
    k 分量为 [-k_max, k_max] 内的整数倍 2π/L。
    返回 (psi_sparse, H_psi_target)。
    """
    psi_fine = np.zeros_like(X_f, dtype=np.float64)
    n_modes  = np.random.randint(3, 7)
    dk       = 2 * np.pi / L

    for _ in range(n_modes):
        kx = np.random.randint(-k_max, k_max + 1) * dk
        ky = np.random.randint(-k_max, k_max + 1) * dk
        kz = np.random.randint(-k_max, k_max + 1) * dk
        b  = np.random.rand() * 2 * np.pi
        psi_fine += np.sin(kx * X_f + ky * Y_f + kz * Z_f + b)

    H_psi_fine   = fft_hamiltonian(psi_fine)
    psi_sparse   = psi_fine[::2, ::2, ::2]
    H_psi_target = H_psi_fine[::2, ::2, ::2]
    return psi_sparse, H_psi_target


def generate_wavefunction_and_target(wf_type: str = 'gaussian', k_max: int = 2):
    """
    统一接口：wf_type ∈ {'gaussian', 'sine'}
    返回 (psi_sparse, H_psi_target)。
    """
    if wf_type == 'gaussian':
        return gen_gaussian_wavefunction()
    elif wf_type == 'sine':
        return gen_sine_wavefunction(k_max=k_max)
    else:
        raise ValueError(f"Unknown wf_type: {wf_type!r}. Choose 'gaussian' or 'sine'.")
