"""
FFT 动能算符、哈密顿量作用。

滤波递推（apply_filter_H / apply_filter_H_all）已迁移到
`filter_core`（算符无关实现），本模块仅保留同名 thin wrapper。
"""
import numpy as np
import pyfftw

from .params import PhysParams


def eval_kinetic(psi: np.ndarray, T_k_diagonal: np.ndarray) -> np.ndarray:
    """朔̂|ψ⟩ — 通过 pyfftw 三维 FFT 计算。"""
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
    """H|ψ⟩ = 朔̂|ψ⟩ + V|ψ⟩。"""
    return eval_kinetic(psi, T_k_diagonal) + V * psi


def apply_filter_H(psi: np.ndarray, V: np.ndarray,
                   nodes: np.ndarray, coeffs: np.ndarray,
                   par: PhysParams,
                   T_k_diagonal: np.ndarray) -> np.ndarray:
    """Thin wrapper → filter_core.apply_filter_H_op。

    原始 Newton 递推实现已迁移到 filter_core，此处仅保留向后兼容层。
    """
    from filter_core import apply_filter_H_op
    return apply_filter_H_op(
        lambda p: apply_H(p, V, T_k_diagonal),
        psi, nodes, coeffs, par,
    )


def apply_chebyshev_explosion(psi: np.ndarray, V: np.ndarray,
                              T_k_diagonal: np.ndarray,
                              m: int,
                              E_lower: float, E_upper: float) -> np.ndarray:
    """
    T_m(H_scaled)|ψ⟩ — 用三项递推计算切比雪夫多项式滤波。

    缩放变换：H_scaled = a·H + b，将 [E_lower, E_upper] 映射到 [-1, 1]。
        a = 2 / (E_upper - E_lower)
        b = -(E_upper + E_lower) / (E_upper - E_lower)

    三项递推（数值稳定，每步一次 H-apply，共 m 次）：
        y_0 = ψ
        y_1 = H_s ψ
        y_{k+1} = 2 H_s y_k - y_{k-1}
    """
    a = 2.0 / (E_upper - E_lower)
    b = -(E_upper + E_lower) / (E_upper - E_lower)

    def _apply_Hs(phi: np.ndarray) -> np.ndarray:
        return a * apply_H(phi, V, T_k_diagonal) + b * phi

    y_prev = psi.copy()
    if m == 0:
        return y_prev
    y_curr = _apply_Hs(psi)
    for _ in range(2, m + 1):
        y_next = 2.0 * _apply_Hs(y_curr) - y_prev
        y_prev = y_curr
        y_curr = y_next
    return y_curr


def apply_filter_H_all(psi: np.ndarray, V: np.ndarray,
                       nodes: np.ndarray, an: np.ndarray,
                       par: PhysParams,
                       T_k_diagonal: np.ndarray) -> np.ndarray:
    """Thin wrapper → filter_core.apply_filter_H_all_op。

    原始 Newton 递推实现已迁移到 filter_core，此处仅保留向后兼容层。
    返回 shape = (ms, *psi.shape)。
    """
    from filter_core import apply_filter_H_all_op
    return apply_filter_H_all_op(
        lambda p: apply_H(p, V, T_k_diagonal),
        psi, nodes, an, par,
    )
