"""
data.py — 波函数生成

提供两种波函数类型（高斯波包叠加 / 正弦波叠加）及统一接口。
所有函数返回 (psi_sparse, H_psi_target)：
  psi_sparse   : 稀疏网格上的波函数值  (N_sparse, N_sparse, N_sparse)
  H_psi_target : 精确 H|ψ⟩ 在稀疏网格上的值（用 FFT 精细网格计算后下采样）
"""

import numpy as np
from .physics import (
    X_f, Y_f, Z_f, L, fft_hamiltonian, d_fine,
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


def gen_fine_wavefunction(wf_type: str = 'gaussian', k_max: int = 2) -> np.ndarray:
    """
    Generate a wavefunction on the fine grid (N_fine³) without downsampling.
    Returns psi_fine of shape (N_fine, N_fine, N_fine).
    Used when the FFT reference energy series (fine grid) is needed separately.
    """
    if wf_type == 'gaussian':
        psi_fine = np.zeros_like(X_f, dtype=np.float64)
        for _ in range(3):
            cx, cy, cz = (np.random.rand(3) - 0.5) * L * 0.5
            width = 0.5 + np.random.rand() * 0.5
            psi_fine += np.exp(
                -((X_f - cx)**2 + (Y_f - cy)**2 + (Z_f - cz)**2)
                / (2 * width**2))
    elif wf_type == 'sine':
        psi_fine = np.zeros_like(X_f, dtype=np.float64)
        n_modes = np.random.randint(3, 7)
        dk = 2 * np.pi / L
        for _ in range(n_modes):
            kx = np.random.randint(-k_max, k_max + 1) * dk
            ky = np.random.randint(-k_max, k_max + 1) * dk
            kz = np.random.randint(-k_max, k_max + 1) * dk
            b = np.random.rand() * 2 * np.pi
            psi_fine += np.sin(kx * X_f + ky * Y_f + kz * Z_f + b)
    else:
        raise ValueError(f"Unknown wf_type: {wf_type!r}. Choose 'gaussian' or 'sine'.")
    return psi_fine


def generate_chain(psi_0_fine: np.ndarray,
                   chain_len: int = 1,
                   kinetic_cutoff: float = 30.0) -> list:
    """
    Generate a chain of (psi_k_sparse, target_k_sparse) training pairs.

    Starting from psi_0_fine, repeatedly applies H_FFT and normalises to
    produce the next input, matching the GNN inference loop exactly.

    Step k:
      input  = psi_k_fine[::2,::2,::2]          (normalised, on sparse grid)
      target = H_FFT(psi_k_fine)[::2,::2,::2]   (unnormalised H·psi_k)
      psi_{k+1}_fine = H_FFT(psi_k_fine) / ‖H_FFT(psi_k_fine)‖

    Parameters
    ----------
    psi_0_fine     : initial wavefunction on fine grid (will be L2-normalised)
    chain_len      : number of steps; returns list of length ≤ chain_len
    kinetic_cutoff : forwarded to fft_hamiltonian (Ha, default 30.0)

    Returns
    -------
    list of (psi_sparse, target_sparse) numpy arrays, length = chain_len
    (may be shorter if norm collapses to zero)
    """
    pairs = []
    psi = psi_0_fine.copy()
    # L2-normalise on fine grid
    nrm = np.sqrt(np.sum(psi**2) * d_fine**3)
    if nrm > 1e-30:
        psi /= nrm

    for _ in range(chain_len):
        H_psi = fft_hamiltonian(psi, kinetic_cutoff=kinetic_cutoff)
        pairs.append((psi[::2, ::2, ::2].copy(),
                      H_psi[::2, ::2, ::2].copy()))
        nrm = np.sqrt(np.sum(H_psi**2) * d_fine**3)
        if nrm < 1e-30 or not np.isfinite(nrm):
            break
        psi = H_psi / nrm      # normalise for next step

    return pairs


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
