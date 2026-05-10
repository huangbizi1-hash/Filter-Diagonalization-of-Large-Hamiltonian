"""
SVD + Rayleigh-Ritz 对角化——向后兼容层。

原始 QR/SVD/eigh 实现已迁移到 `filter_core.svd_rayleigh_ritz_op`，
此文件仅保留同名 thin wrapper。
"""
from typing import Tuple

import numpy as np

from .hamiltonian import apply_H


def svd_rayleigh_ritz(filtered_psi_matrix: np.ndarray,
                      x_grid,
                      V: np.ndarray,
                      Nx: int, Ny: int, Nz: int,
                      T_k_diagonal: np.ndarray,
                      svd_tol: float = 1e-3,
                      max_energies: int = 200) -> Tuple[np.ndarray, np.ndarray, int]:
    """Thin wrapper → filter_core.svd_rayleigh_ritz_op。

    将 filtered_psi_matrix (n_basis, Nx, Ny, Nz) reshape 为
    basis_mat (n_grid, n_basis) 后委托到算符无关版本。
    返回 (energies, Ur, rank)。
    """
    from filter_core import svd_rayleigh_ritz_op
    n_grid    = Nx * Ny * Nz
    basis_mat = filtered_psi_matrix.reshape(-1, n_grid).T  # (n_grid, n_basis)
    H_apply   = lambda psi_flat: apply_H(
        psi_flat.reshape(Nx, Ny, Nz), V, T_k_diagonal).ravel()
    return svd_rayleigh_ritz_op(basis_mat, H_apply, svd_tol, max_energies,
                                hermitian=True)
