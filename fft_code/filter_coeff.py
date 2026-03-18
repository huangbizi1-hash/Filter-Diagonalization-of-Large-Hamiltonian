"""
Newton 插值滤波系数构建。

公开接口：build_filter_coefficients, build_monomial_coefficients
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


def _newton_to_monomial(coeffs: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    将 Newton 插值多项式系数转换为单项式（幂次）基底系数。

    Newton 形式：p(x̃) = a[0] + a[1]*(x̃-s[0]) + a[2]*(x̃-s[0])*(x̃-s[1]) + ...
    转换为：p(x̃) = c[0] + c[1]*x̃ + c[2]*x̃² + ...

    参数
    ----
    coeffs : (nc,)  Newton 除差系数
    nodes  : (nc,)  Ashkenazy 节点（缩放后 [-2, 2] 区间）

    返回
    ----
    c : (nc,)  单项式基底系数，c[k] 为 x̃^k 的系数
    """
    nc = len(coeffs)
    result = np.zeros(nc)
    # w 表示当前基底多项式 prod_{i<j}(x̃ - nodes[i])，按升幂存储
    w = np.array([1.0])
    for j in range(nc):
        result[:len(w)] += coeffs[j] * w
        if j < nc - 1:
            # w *= (x̃ - nodes[j])
            new_w = np.zeros(len(w) + 1)
            new_w[1:] += w
            new_w[:len(w)] -= nodes[j] * w
            w = new_w
    return result


def build_monomial_coefficients(an: np.ndarray,
                                 samp: np.ndarray) -> np.ndarray:
    """
    将所有滤波中心的 Newton 系数矩阵转换为单项式系数矩阵。

    参数
    ----
    an   : (ms, nc)  Newton 系数矩阵（来自 build_filter_coefficients）
    samp : (nc,)     Ashkenazy 节点

    返回
    ----
    cn : (ms, nc)  单项式系数矩阵，cn[ie, k] 为第 ie 个滤波函数 x̃^k 的系数
    """
    ms, nc = an.shape
    cn = np.zeros((ms, nc))
    for ie in range(ms):
        cn[ie] = _newton_to_monomial(an[ie], samp)
    return cn


# ---- 公开接口 ----

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
