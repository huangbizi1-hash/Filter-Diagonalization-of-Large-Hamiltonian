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
