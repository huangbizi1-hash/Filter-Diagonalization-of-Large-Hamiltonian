"""
data.py — 波函数生成

提供三种波函数类型（高斯波包叠加 / 正弦波叠加 / 格点随机 ±1）及统一接口。
所有函数返回 (psi, H_psi_target)，均在单一网格（d=0.5）上，无下采样。
GNN 输入已通过 generate_chain 归一化，保证训练稳定。
"""

import numpy as np
from .physics import (
    X, Y, Z, L, fft_hamiltonian, d, N,
)


def gen_gaussian_wavefunction():
    """
    3 个随机高斯波包叠加，返回 (psi, H_psi_target)。
    """
    psi = np.zeros_like(X, dtype=np.float64)
    for _ in range(3):
        cx, cy, cz = (np.random.rand(3) - 0.5) * L * 0.5
        width      = 0.5 + np.random.rand() * 0.5
        psi       += np.exp(
            -((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
            / (2 * width**2))

    H_psi = fft_hamiltonian(psi)
    return psi, H_psi


def gen_sine_wavefunction(k_max: int = 2):
    """
    3–6 个正弦模式叠加：sin(k·r + b)，
    k 分量为 [-k_max, k_max] 内的整数倍 2π/L。
    返回 (psi, H_psi_target)。
    """
    psi    = np.zeros_like(X, dtype=np.float64)
    n_modes = np.random.randint(3, 7)
    dk      = 2 * np.pi / L

    for _ in range(n_modes):
        kx = np.random.randint(-k_max, k_max + 1) * dk
        ky = np.random.randint(-k_max, k_max + 1) * dk
        kz = np.random.randint(-k_max, k_max + 1) * dk
        b  = np.random.rand() * 2 * np.pi
        psi += np.sin(kx * X + ky * Y + kz * Z + b)

    H_psi = fft_hamiltonian(psi)
    return psi, H_psi


def gen_pm1_wavefunction(rng: np.random.Generator = None):
    """
    格点独立随机 ±1 波函数（flat 随机态），返回 (psi, H_psi_target)。
    """
    if rng is not None:
        signs = rng.choice(np.array([-1.0, 1.0]), size=X.shape)
    else:
        signs = np.random.choice([-1.0, 1.0], size=X.shape)
    psi   = signs.astype(np.float64)
    H_psi = fft_hamiltonian(psi)
    return psi, H_psi


def gen_fine_wavefunction(wf_type: str = 'gaussian', k_max: int = 2,
                          rng: np.random.Generator = None) -> np.ndarray:
    """
    生成初始波函数（形状 (N, N, N)），不归一化，用于 generate_chain 的起点。
    """
    if wf_type == 'gaussian':
        psi = np.zeros_like(X, dtype=np.float64)
        for _ in range(3):
            cx, cy, cz = (np.random.rand(3) - 0.5) * L * 0.5
            width = 0.5 + np.random.rand() * 0.5
            psi += np.exp(
                -((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
                / (2 * width**2))
    elif wf_type == 'sine':
        psi = np.zeros_like(X, dtype=np.float64)
        n_modes = np.random.randint(3, 7)
        dk = 2 * np.pi / L
        for _ in range(n_modes):
            kx = np.random.randint(-k_max, k_max + 1) * dk
            ky = np.random.randint(-k_max, k_max + 1) * dk
            kz = np.random.randint(-k_max, k_max + 1) * dk
            b = np.random.rand() * 2 * np.pi
            psi += np.sin(kx * X + ky * Y + kz * Z + b)
    elif wf_type == 'pm1':
        gen = rng if rng is not None else np.random.default_rng()
        psi = gen.choice(np.array([-1.0, 1.0]),
                         size=X.shape).astype(np.float64)
    else:
        raise ValueError(f"Unknown wf_type: {wf_type!r}. Choose 'gaussian', 'sine', or 'pm1'.")
    return psi


def generate_chain(psi_0: np.ndarray,
                   chain_len: int = 10,
                   kinetic_cutoff: float = 30.0) -> list:
    """
    生成 chain_len 步的 (psi_k, target_k) 训练对。

    输入波函数先 L2 归一化（保证 GNN 输入始终为归一化波函数），
    每步：
      input  = psi_k                    （归一化，直接用于 GNN 输入）
      target = H_FFT(psi_k)             （未归一化的 H·psi_k）
      psi_{k+1} = H_FFT(psi_k) / ‖H_FFT(psi_k)‖

    Parameters
    ----------
    psi_0          : 初始波函数，形状 (N, N, N)，将被 L2 归一化
    chain_len      : 链长（步数），默认 10
    kinetic_cutoff : 传递给 fft_hamiltonian（Ha，默认 30.0）

    Returns
    -------
    list of (psi, target) numpy arrays，长度 ≤ chain_len
    """
    pairs = []
    psi = psi_0.copy()
    nrm = np.sqrt(np.sum(psi**2) * d**3)
    if nrm > 1e-30:
        psi /= nrm

    for _ in range(chain_len):
        H_psi = fft_hamiltonian(psi, kinetic_cutoff=kinetic_cutoff)
        pairs.append((psi.copy(), H_psi.copy()))
        nrm = np.sqrt(np.sum(H_psi**2) * d**3)
        if nrm < 1e-30 or not np.isfinite(nrm):
            break
        psi = H_psi / nrm

    return pairs


def generate_wavefunction_and_target(wf_type: str = 'gaussian', k_max: int = 2,
                                     rng: np.random.Generator = None):
    """
    统一接口：wf_type ∈ {'gaussian', 'sine', 'pm1'}
    返回 (psi, H_psi_target)。
    """
    if wf_type == 'gaussian':
        return gen_gaussian_wavefunction()
    elif wf_type == 'sine':
        return gen_sine_wavefunction(k_max=k_max)
    elif wf_type == 'pm1':
        return gen_pm1_wavefunction(rng=rng)
    else:
        raise ValueError(f"Unknown wf_type: {wf_type!r}. Choose 'gaussian', 'sine', or 'pm1'.")
