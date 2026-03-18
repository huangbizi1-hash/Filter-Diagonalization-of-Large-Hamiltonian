"""
FFT 动能算符、哈密顿量作用，以及 Newton 多项式滤波器作用。
"""
import numpy as np
import pyfftw

from .params import PhysParams


def eval_kinetic(psi: np.ndarray, T_k_diagonal: np.ndarray) -> np.ndarray:
    """T̂|ψ⟩ — 通过 pyfftw 三维 FFT 计算。"""
    shape    = psi.shape
    fft_in   = pyfftw.empty_aligned(shape, dtype='complex128')
    fft_out  = pyfftw.empty_aligned(shape, dtype='complex128')
    ifft_out = pyfftw.empty_aligned(shape, dtype='complex128')
    fft1  = pyfftw.FFTW(fft_in,  fft_out,  axes=(0, 1, 2),
                         direction='FFTW_FORWARD',  flags=('FFTW_MEASURE',))
    ifft1 = pyfftw.FFTW(fft_out, ifft_out, axes=(0, 1, 2),
                         direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',))
    fft_in[:] = psi
    fft1()
    fft_out[:] = T_k_diagonal * fft_out
    ifft1()
    return ifft_out.copy()


def apply_H(psi: np.ndarray, V: np.ndarray,
            T_k_diagonal: np.ndarray) -> np.ndarray:
    """H|ψ⟩ = T̂|ψ⟩ + V|ψ⟩。"""
    return eval_kinetic(psi, T_k_diagonal) + V * psi


def apply_filter_H(psi: np.ndarray, V: np.ndarray,
                   nodes: np.ndarray, coeffs: np.ndarray,
                   par: PhysParams,
                   T_k_diagonal: np.ndarray) -> np.ndarray:
    """
    f(H)|ψ⟩ — 通过缩放哈密顿量的 Newton 多项式求值。
    缩放变量：x̃ = 4(H - Vmin)/dE - 2
    """
    n        = len(nodes)
    psi_prev = psi.copy()
    result   = coeffs[0] * psi

    for j in range(1, n):
        T_psi    = eval_kinetic(psi_prev, T_k_diagonal)
        H_psi    = T_psi + V * psi_prev
        psi_curr = ((4.0 / par.dE) * (H_psi - par.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        result  += coeffs[j] * psi_curr
        psi_prev = psi_curr

    return result


def apply_filter_H_all(psi: np.ndarray, V: np.ndarray,
                       nodes: np.ndarray, an: np.ndarray,
                       par: PhysParams,
                       T_k_diagonal: np.ndarray) -> np.ndarray:
    """
    同时对所有滤波中心计算 f_i(H)|ψ⟩，共享 Newton 基底向量。

    与对每个 El_i 单独调用 apply_filter_H 相比，H 的作用次数从
    ms * nc 减少到 nc（ms = len(El_list)，nc = len(nodes)），
    加速比为 ms 倍。

    参数
    ----
    psi          : (Nx, Ny, Nz)    输入波函数
    V            : (Nx, Ny, Nz)    势能
    nodes        : (nc,)           Ashkenazy 节点（来自 build_filter_coefficients）
    an           : (ms, nc)        Newton 系数矩阵（来自 build_filter_coefficients）
    par          : PhysParams
    T_k_diagonal : (Nx, Ny, Nz)    k 空间动能对角元

    返回
    ----
    results : (ms, Nx, Ny, Nz)  results[ie] = f_ie(H)|ψ⟩
    """
    ms, nc = an.shape
    # an[:, 0] shape (ms,) -> (ms,1,1,1)；psi shape (Nx,Ny,Nz) -> (1,Nx,Ny,Nz)
    results = an[:, 0, None, None, None] * psi[None, :, :, :]  # (ms,Nx,Ny,Nz)

    psi_prev = psi.copy()
    for j in range(1, nc):
        T_psi    = eval_kinetic(psi_prev, T_k_diagonal)
        H_psi    = T_psi + V * psi_prev
        psi_curr = ((4.0 / par.dE) * (H_psi - par.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        # 对所有 ms 个滤波中心同时累加：results[ie] += an[ie, j] * psi_curr
        results += an[:, j, None, None, None] * psi_curr[None, :, :, :]
        psi_prev = psi_curr

    return results
