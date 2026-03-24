"""
Newton 插值滤波系数构建。

公开接口
--------
make_filter_func          — 窗函数工厂（返回 callable）
build_filter_coefficients — 构建 Newton 插值系数矩阵

支持的窗函数类型
----------------
"gaussian"
    经典高斯滤波（默认），参数由 PhysParams.dt 控制：
        f(x) = sqrt(dt/π) · exp(-(x-El)² · dt)
    展宽 sigma = 1/sqrt(2·dt)（物理单位，Hartree）。
    高精度需要高阶多项式（dt 越大 → sigma 越窄 → nc 越大）。

"gabor"
    Gabor 型滤波（高斯包络 × 余弦调制，关于 El 对称）：
        f(x) = exp(-alpha_f·(x-El)²) · cos(k_f·(x-El))
    参数：
        alpha_f (float, 默认 0.5)：高斯包络宽度，sigma = 1/sqrt(2·alpha_f)。
        k_f     (float, 默认 1.0)：余弦调制频率（Hartree⁻¹）。
    特点：alpha_f 远小于 dt（如 0.5 vs 1600），包络宽 ~1 Hartree，
    所需 Newton 节点数 nc 大幅低于高斯情形（约 150-300 vs 5000），
    适合在不需要极窄窗的场合降低计算量。
    余弦形式使滤波窗关于 El 严格偶对称，避免了 sin(k_f·x) 因 x 平移
    引入的相位偏差。

    filter_func 签名：filter_func(x_phys: np.ndarray, El: float) -> np.ndarray
"""
from typing import Callable, Optional, Tuple

import numpy as np

from .params import PhysParams


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────

def _filt_func_gaussian(x_phys: np.ndarray, El: float, dt: float) -> np.ndarray:
    """高斯窗：sqrt(dt/π)·exp(-(x-El)²·dt)"""
    return np.sqrt(dt / np.pi) * np.exp(-(x_phys - El) ** 2 * dt)


def _filt_func_gabor(x_phys: np.ndarray, El: float,
                     alpha_f: float, k_f: float) -> np.ndarray:
    """Gabor 窗：exp(-alpha_f·(x-El)²)·cos(k_f·(x-El))，关于 El 对称"""
    return np.exp(-alpha_f * (x_phys - El) ** 2) * np.cos(k_f * (x_phys - El))


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


# ──────────────────────────────────────────────
# 单项式转换（仅供小 nc 验证）
# ──────────────────────────────────────────────

def _newton_to_monomial(coeffs: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """
    将 Newton 除差系数展开为单项式系数（升幂）。

    Newton 形式：p(x̃) = c[0] + c[1](x̃-x0) + c[2](x̃-x0)(x̃-x1) + …
    输出：mono[k] 满足  p(x̃) = Σ_k mono[k] * x̃^k

    实现方式：从最内层系数出发，用 Horner 展开逐步展开基底多项式。
    """
    n = len(coeffs)
    poly = np.array([coeffs[-1]], dtype=complex)
    for j in range(n - 2, -1, -1):
        new_poly = np.zeros(len(poly) + 1, dtype=complex)
        new_poly[:-1] -= nodes[j] * poly
        new_poly[1:]  += poly
        new_poly[0]   += coeffs[j]
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


# ──────────────────────────────────────────────
# 公开接口：窗函数工厂
# ──────────────────────────────────────────────

def make_filter_func(filter_type: str,
                     dt: float = 1.0,
                     alpha_f: float = 0.5,
                     k_f: float = 1.0) -> Callable:
    """
    返回 filter_func(x_phys, El) -> np.ndarray。

    参数
    ----
    filter_type : "gaussian" 或 "gabor"
    dt          : 高斯参数（仅 filter_type="gaussian" 使用）
    alpha_f     : Gabor 高斯包络宽度（仅 filter_type="gabor" 使用）
    k_f         : Gabor 余弦调制频率（仅 filter_type="gabor" 使用）
    """
    if filter_type == "gaussian":
        return lambda x, El: _filt_func_gaussian(x, El, dt)
    elif filter_type == "gabor":
        return lambda x, El: _filt_func_gabor(x, El, alpha_f, k_f)
    else:
        raise ValueError(
            f"Unknown filter_type={filter_type!r}. "
            "Supported: 'gaussian', 'gabor'."
        )


# ──────────────────────────────────────────────
# 公开接口：构建 Newton 系数
# ──────────────────────────────────────────────

def build_filter_coefficients(
        El_list: np.ndarray,
        par: PhysParams,
        nc: int,
        filter_func: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为 El_list 中每个滤波中心构建 Newton 插值系数。

    参数
    ----
    El_list     : 滤波中心能量列表（物理单位，Hartree）
    par         : PhysParams（提供 dE、Vmin、dt）
    nc          : Newton 节点数
    filter_func : callable(x_phys, El) -> np.ndarray，可选。
                  若为 None，默认使用高斯滤波（由 par.dt 控制宽度）。
                  可由 make_filter_func() 构造。

    返回
    ----
    an   : (ms, nc) 系数矩阵
    samp : (nc,)    [-2, 2] 区间的 Ashkenazy 节点
    """
    if filter_func is None:
        filter_func = make_filter_func("gaussian", dt=par.dt)

    Smin, Smax = -2.0, 2.0
    scale  = (Smax - Smin) / par.dE
    samp   = _samp_points_ashkenazy(Smin, Smax, nc)
    x_phys = (samp + 2.0) / scale + par.Vmin

    an = np.zeros((len(El_list), nc))
    for ie, El in enumerate(El_list):
        y = filter_func(x_phys, El)
        an[ie] = _newton_coefficients(samp, y)
    return an, samp
