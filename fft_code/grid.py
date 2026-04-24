"""
实空间网格构建与 k 空间动能对角元。
"""
from typing import Tuple
import numpy as np


def grid_to_vec(y_grid: np.ndarray) -> np.ndarray:
    """(n, Nx, Ny, Nz) → (Nx*Ny*Nz, n)"""
    return y_grid.reshape(y_grid.shape[0], -1).T


def vec_to_grid(y_vec: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """(Nx*Ny*Nz, n) → (n, Nx, Ny, Nz)"""
    return y_vec.T.reshape(y_vec.shape[-1], Nx, Ny, Nz)


def build_grid(N: int, d: float) -> Tuple:
    """
    均匀网格：步长 *d*，每轴 *N* 个格点，以原点为中心。
    返回 x, y, z, X, Y, Z, x_grid=[x,y,z]。
    """
    L = (N - 1) * d / 2
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    z = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return x, y, z, X, Y, Z, [x, y, z]


def build_grid_box(N: int, L: float) -> Tuple:
    """
    均匀周期盒网格：每轴 *N* 个格点，x ∈ [-L/2, L/2)。
    步长 d = L/N（周期边界约定）。
    返回 x, y, z, X, Y, Z, x_grid=[x,y,z]。
    """
    d = L / N
    x = (np.arange(N) - N / 2) * d
    y = x.copy()
    z = x.copy()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return x, y, z, X, Y, Z, [x, y, z]


def build_k_diagonal(x_grid, kinetic_cut: float = 30.0) -> np.ndarray:
    """T(k) = |k|²/2，截断于 *kinetic_cut*。"""
    x, y, z = x_grid
    d = x[1] - x[0]
    kx = 2 * np.pi * np.fft.fftfreq(len(x), d=d)
    ky = 2 * np.pi * np.fft.fftfreq(len(y), d=d)
    kz = 2 * np.pi * np.fft.fftfreq(len(z), d=d)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    T_k = (Kx**2 + Ky**2 + Kz**2) / 2
    return np.minimum(T_k, kinetic_cut)
