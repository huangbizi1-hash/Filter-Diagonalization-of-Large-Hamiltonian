"""
filter_core.py
===============
算符无关（operator-agnostic）的滤波对角化核心模块。

设计理念
--------
RBF-FD、FFT、以及有限差分（FD）三种方法的区别只在
「哈密顿量如何作用到波函数上」（H|ψ⟩ 的实现），而：

  1. Newton 多项式滤波系数的构建（fft_code/filter_coeff.py）
  2. 滤波递推（只需 H|ψ⟩ 一次调用）
  3. Rayleigh-Ritz 对角化（只需再调用 H|ψ⟩ r 次）

都与 H 的具体实现无关。本模块把 (2) 和 (3) 从 `fft_code` 中剥离出来，
接受任意满足 `H_apply(psi) -> Hψ` 签名的可调用对象。

公开接口
--------
PhysParams                 ─ 数据类（从 fft_code.params 转出）
build_filter_coefficients  ─ 构建 Newton 系数（从 fft_code.filter_coeff 转出）
make_filter_func           ─ 窗函数工厂
compute_newton_an          ─ 固定节点下的 an 计算
apply_filter_H_all_op      ─ 对任意 H_apply 做多中心滤波
apply_filter_H_op          ─ 对任意 H_apply 做单中心滤波
svd_rayleigh_ritz_op       ─ 对任意 H_apply 做 QR+SVD+Rayleigh-Ritz
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
from scipy.linalg import eigh

# ─── 直接转出现有窗/系数构建（与算符无关） ────────────────────────────────
from fft_code.params import PhysParams, IstParams
from fft_code.filter_coeff import (
    build_filter_coefficients,
    make_filter_func,
    compute_newton_an,
)


__all__ = [
    "PhysParams",
    "IstParams",
    "build_filter_coefficients",
    "make_filter_func",
    "compute_newton_an",
    "apply_filter_H_op",
    "apply_filter_H_all_op",
    "svd_rayleigh_ritz_op",
]


# ──────────────────────────────────────────────────────────────────────
# 通用滤波递推：只依赖 H_apply(psi) -> Hψ
# ──────────────────────────────────────────────────────────────────────

def apply_filter_H_op(
    H_apply: Callable[[np.ndarray], np.ndarray],
    psi: np.ndarray,
    nodes: np.ndarray,
    coeffs: np.ndarray,
    par: PhysParams,
) -> np.ndarray:
    """
    f(H)|ψ⟩ — 与 fft_code.hamiltonian.apply_filter_H 算法完全一致，
    但 H 通过任意可调用对象 H_apply 作用。

    递推（缩放坐标 x̃ = 4(H-Vmin)/dE - 2）：
        basis_0 = ψ
        basis_j = (4/dE)·(H - Vmin)·basis_{j-1} - 2·basis_{j-1} - nodes[j-1]·basis_{j-1}
        result  = Σ_j coeffs[j] · basis_j
    """
    n = len(nodes)
    psi_prev = psi.copy()
    result = coeffs[0] * psi

    for j in range(1, n):
        H_psi = H_apply(psi_prev)
        psi_curr = ((4.0 / par.dE) * (H_psi - par.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        result += coeffs[j] * psi_curr
        psi_prev = psi_curr

    return result


def apply_filter_H_all_op(
    H_apply: Callable[[np.ndarray], np.ndarray],
    psi: np.ndarray,
    nodes: np.ndarray,
    an: np.ndarray,
    par: PhysParams,
) -> np.ndarray:
    """
    同时对所有滤波中心计算 f_i(H)|ψ⟩，共享 Newton 基底向量。

    H-apply 次数从 ms·nc 降至 nc（ms = an.shape[0]，nc = an.shape[1]）。
    与 fft_code.hamiltonian.apply_filter_H_all 算法完全一致，只是 ψ 可以
    是任何形状的 numpy 数组（1-D 向量或 3-D 网格都行），H_apply 负责处理。

    参数
    ----
    H_apply : callable(psi) -> Hψ
    psi     : ndarray，任意形状
    nodes   : (nc,) Newton 节点
    an      : (ms, nc) Newton 系数矩阵
    par     : PhysParams

    返回
    ----
    results : (ms, *psi.shape)   results[ie] = f_ie(H)|ψ⟩
    """
    ms, nc = an.shape
    shape_extra = psi.shape
    # 广播 an[:, 0] (ms,) × psi (*shape) → results (ms, *shape)
    idx = (slice(None),) + (None,) * psi.ndim
    results = an[:, 0][idx] * psi[None, ...]

    psi_prev = psi.copy()
    for j in range(1, nc):
        H_psi = H_apply(psi_prev)
        psi_curr = ((4.0 / par.dE) * (H_psi - par.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        results += an[:, j][idx] * psi_curr[None, ...]
        psi_prev = psi_curr

    return results


# ──────────────────────────────────────────────────────────────────────
# 通用 Rayleigh-Ritz（只依赖 H_apply(psi) -> Hψ）
# ──────────────────────────────────────────────────────────────────────

def svd_rayleigh_ritz_op(
    basis_mat: np.ndarray,
    H_apply: Callable[[np.ndarray], np.ndarray],
    svd_tol: float = 1e-3,
    max_energies: int = 200,
    hermitian: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    SVD + Rayleigh-Ritz。与 fft_code.rayleigh_ritz.svd_rayleigh_ritz 流程一致，
    但 H 通过 H_apply 作用，basis_mat 为平坦矩阵（n_grid, n_basis）。

    参数
    ----
    basis_mat : (n_grid, n_basis) 列向量为滤波态（未正交化，可含零列）
    H_apply   : callable(psi_flat) -> Hψ_flat
    svd_tol   : SVD 秩截断阈值
    max_energies: 输出最多多少个能级（从低到高）
    hermitian : True 则用 scipy.linalg.eigh（对 H̃ 额外强制对称化避免数值噪声）；
                False 则用 numpy.linalg.eig 并取实部排序。

    返回
    ----
    energies : (r_eff,) 排序后的本征能级（最多 max_energies 个）
    Ur       : (n_grid, r) 正交基
    r        : 有效秩
    """
    # 列归一化，去除零列 / inf 列
    if basis_mat.ndim != 2:
        raise ValueError(f"basis_mat must be 2-D, got shape={basis_mat.shape}")

    norms = np.linalg.norm(basis_mat, axis=0)
    mask = np.isfinite(norms) & (norms > 0)
    n_grid = basis_mat.shape[0]
    if not np.any(mask):
        # 所有列都无效（零向量/NaN/Inf），返回空结果而不是在后续索引崩溃
        empty_Ur = np.zeros((n_grid, 0), dtype=np.result_type(basis_mat.dtype, np.float64))
        empty_evals = np.array([], dtype=np.float64)
        return empty_evals, empty_Ur, 0

    B = basis_mat[:, mask] / norms[None, mask]

    # QR + SVD → 正交基
    Q, R = np.linalg.qr(B, mode="reduced")
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    if sigma.size == 0:
        empty_Ur = np.zeros((n_grid, 0), dtype=np.result_type(B.dtype, np.float64))
        empty_evals = np.array([], dtype=np.float64)
        return empty_evals, empty_Ur, 0

    r_eff = int(np.sum(sigma > svd_tol))
    r = max(1, r_eff)
    Ur = (Q @ U1)[:, :r]
    if Ur.shape[1] == 0:
        empty_evals = np.array([], dtype=np.float64)
        return empty_evals, Ur, 0

    # H̃[i,j] = ⟨ur_i|H|ur_j⟩
    H_tilde = np.zeros((r, r), dtype=np.result_type(Ur.dtype, np.float64))
    for j in range(r):
        HUj = H_apply(Ur[:, j])
        H_tilde[:, j] = Ur.conj().T @ HUj

    if hermitian:
        # 对非厄米矩阵强制 (H̃ + H̃ᵀ)/2，吸收数值不对称
        H_sym = 0.5 * (H_tilde + H_tilde.conj().T)
        evals, _ = eigh(H_sym)
        evals = np.sort(evals.real)
    else:
        evals_c, _ = np.linalg.eig(H_tilde)
        evals = np.sort(evals_c.real)

    return evals[:max_energies], Ur, r
