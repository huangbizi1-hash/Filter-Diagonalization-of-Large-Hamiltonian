"""
SVD + Rayleigh-Ritz 对角化。
"""
from typing import Tuple

import numpy as np
from scipy.linalg import eigh

from .grid import grid_to_vec, vec_to_grid
from .hamiltonian import apply_H


def svd_rayleigh_ritz(filtered_psi_matrix: np.ndarray,
                      x_grid, V: np.ndarray,
                      Nx: int, Ny: int, Nz: int,
                      T_k_diagonal: np.ndarray,
                      svd_tol: float = 1e-3,
                      max_energies: int = 200) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    在滤波子空间中做 Rayleigh-Ritz 对角化。

    返回 energies, Ur, rank r。
    """
    C_f = grid_to_vec(filtered_psi_matrix)
    C_f = C_f / np.linalg.norm(C_f, axis=0)
    C_f = C_f[:, ~np.any(np.isinf(C_f), axis=0)]

    print("  QR decomposition ...")
    Q, R = np.linalg.qr(C_f, mode='reduced')

    print("  SVD on R ...")
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    U = Q @ U1

    r = int(np.sum(sigma > svd_tol))
    print(f"  Selected rank r = {r}  (tol = {svd_tol})")
    Ur = U[:, :r]

    Ur_grid  = vec_to_grid(Ur, Nx, Ny, Nz)
    HUr_grid = np.zeros_like(Ur_grid, dtype='complex128')
    for i in range(r):
        HUr_grid[i] = apply_H(Ur_grid[i], V, T_k_diagonal)

    H_tilde     = Ur.T.conj() @ grid_to_vec(HUr_grid)
    energies, _ = eigh(H_tilde)
    energies    = np.sort(energies.real)[:max_energies]
    return energies, Ur, r
