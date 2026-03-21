"""
Newton 插值滤波系数构建。

公开接口：build_filter_coefficients
"""
from typing import Tuple

import numpy as np

from .params import PhysParams


# ---- 内部辅助 ----

def _filt_func_values(x: np.ndarray, dt: float) -> np.ndarray:
    return np.sqrt(dt / np.pi) * np.exp(-x**2 * dt)


def _samp_points_ashkenazy(min_val: float, max_val: float, nc: int) -> np.ndarray:
    """Ashkenazy 最优采样节点（贪心最大化行列式）。"""
    nc3  = 32 * nc
    samp = (min_val + np.arange(nc3) * (max_val - min_val) / (nc3 - 1)).astype(complex)
    point = np.zeros(nc, dtype=complex)
    point[0] = samp[0]
    dv = (samp.real - point[0].real)**2 + (samp.imag - point[0].imag)**2
    veca = np.where(dv < 1e-10, -1e30, np.log(dv))
    for j in range(1, nc):
        kmax     = np.argmax(veca)
        point[j] = samp[kmax]
        dv   = (samp.real - point[j].real)**2 + (samp.imag - point[j].imag)**2
        veca = np.where(dv < 1e-10, -1e30, veca + np.log(dv))
    return point.real


def _newton_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    f = np.zeros((n, n))
    f[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            f[i, j] = (f[i, j - 1] - f[i - 1, j - 1]) / (x[i] - x[i - j])
    return f[np.arange(n), np.arange(n)]


# ---- 公开接口 ----

def _newton_to_monomial(coeffs: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    将 Newton 除差系数展开为单项式系数（升幂）。

    Newton 形式：p(x̃) = c[0] + c[1](x̃-x0) + c[2](x̃-x0)(x̃-x1) + …
    输出：mono[k] 满足  p(x̃) = Σ_k mono[k] * x̃^k

    实现方式：从最内层系数出发，用 Horner 展开逐步展开基底多项式。
    """
    n = len(coeffs)
    # 从最高阶开始，poly 表示升幂系数数组
    poly = np.array([coeffs[-1]], dtype=complex)
    for j in range(n - 2, -1, -1):
        # poly ← poly * (x̃ - nodes[j]) + coeffs[j]
        new_poly = np.zeros(len(poly) + 1, dtype=complex)
        new_poly[:-1] -= nodes[j] * poly   # 乘以 -nodes[j]
        new_poly[1:]  += poly               # 乘以 x̃
        new_poly[0]   += coeffs[j]          # 加常数项
        poly = new_poly
    return poly.real


def build_monomial_coefficients(an: np.ndarray,
                                 samp: np.ndarray) -> np.ndarray:
    """
    将 (ms, nc) Newton 系数矩阵批量转换为单项式系数矩阵。

    参数
    ----
    an   : (ms, nc) — build_filter_coefficients 返回的 Newton 系数
    samp : (nc,)    — 对应的 Ashkenazy 节点（[-2,2] 内的缩放坐标）

    返回
    ----
    cn : (ms, nc)  — 单项式系数（升幂），cn[ie, k] 对应 x̃^k 的系数

    注意
    ----
    高阶单项式基底在数值上不稳定：nc > ~50 时展开误差会因灾难性消去
    急剧增大，仅建议在 nc ≤ 50 时用于验证。
    """
    ms, nc = an.shape
    if nc > 50:
        import warnings
        warnings.warn(
            f"build_monomial_coefficients: nc={nc} > 50, "
            "Newton→monomial conversion may be numerically unstable "
            "(catastrophic cancellation). Use for small-nc verification only.",
            RuntimeWarning, stacklevel=2,
        )
    cn = np.zeros((ms, nc))
    for ie in range(ms):
        cn[ie] = _newton_to_monomial(an[ie], samp)
    return cn


def build_filter_coefficients(El_list: np.ndarray, par: PhysParams,
                               nc: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    为 El_list 中每个滤波中心构建 Newton 插值系数。

    返回
    ----
    an   : (ms, nc) 系数矩阵
    samp : (nc,)    [-2, 2] 区间的 Ashkenazy 节点
    """
    Smin, Smax = -2.0, 2.0
    scale  = (Smax - Smin) / par.dE
    samp   = _samp_points_ashkenazy(Smin, Smax, nc)
    x_phys = (samp + 2.0) / scale + par.Vmin

    an = np.zeros((len(El_list), nc))
    for ie, El in enumerate(El_list):
        y = _filt_func_values(x_phys - El, par.dt)
        an[ie] = _newton_coefficients(samp, y)
    return an, samp
