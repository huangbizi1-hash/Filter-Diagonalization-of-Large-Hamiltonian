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
    Gabor 型滤波（超高斯包络 × 余弦调制，关于 El 对称）：
        f(x) = exp(-alpha_f·|x-El|^n0) · cos(k_f·(x-El))
    参数：
        alpha_f (float, 默认 0.5)：包络衰减系数。
        k_f     (float, 默认 1.0)：余弦调制频率（Hartree⁻¹）。
        n0      (int,   默认 4)  ：包络指数（n0=2 → 普通高斯；n0>2 → 超高斯，
                                  平顶更宽、边沿衰减更陡）。
    特点：余弦形式使滤波窗关于 El 严格偶对称；超高斯包络（n0=4）在目标
    区间内响应更平坦，边沿截止更锐利。

"bandpass"
    平滑带通窗（双 tanh 差分），关于 El 对称：
        w(x) = 0.5·[tanh(β·(x-EL)) - tanh(β·(x-ER))]
    其中 EL = El - E1，ER = El + E1。
    参数：
        beta (float, 默认 10.0)：边沿陡峭程度（越大截止越锐利）。
        E1   (float, 默认 0.05)：带通半宽（Hartree），即窗口覆盖 [El-E1, El+E1]。
    特点：在 [El-E1, El+E1] 内响应近似为 1，在 El±E1 处以 tanh 平滑过渡到 0；
    无余弦震荡，适合直接截取能量区间。

    filter_func 签名：filter_func(x_phys: np.ndarray, El: float) -> np.ndarray
"""
from typing import Callable, List, Optional, Tuple

import numpy as np

from .params import PhysParams


# ──────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────

def _filt_func_gaussian(x_phys: np.ndarray, El: float, dt: float) -> np.ndarray:
    """高斯窗：sqrt(dt/π)·exp(-(x-El)²·dt)"""
    return np.sqrt(dt / np.pi) * np.exp(-(x_phys - El) ** 2 * dt)


def _filt_func_gabor(x_phys: np.ndarray, El: float,
                     alpha_f: float, k_f: float, n0: int = 4) -> np.ndarray:
    """超高斯 Gabor 窗：exp(-alpha_f·|x-El|^n0)·cos(k_f·(x-El))，关于 El 对称"""
    return np.exp(-alpha_f * np.abs(x_phys - El) ** n0) * np.cos(k_f * (x_phys - El))


def _filt_func_bandpass(x_phys: np.ndarray, El: float,
                        beta: float, E1: float) -> np.ndarray:
    """平滑带通窗：0.5·[tanh(β(x-EL)) - tanh(β(x-ER))]，EL=El-E1，ER=El+E1"""
    EL = El - E1
    ER = El + E1
    return 0.5 * (np.tanh(beta * (x_phys - EL)) - np.tanh(beta * (x_phys - ER)))


def _samp_points_ashkenazy(min_val: float, max_val: float, nc: int) -> np.ndarray:
    """Ashkenazy 最优采样节点（贪心最大化 Vandermonde 行列式）。

    算法：在 32*nc 个均匀候选点中贪心地选 nc 个，每步选使
    log|Vandermonde 行列式| 增量最大的候选点。结果近似于
    Chebyshev 节点（两端密、中间稀），Lebesgue 常数为 O(log nc)。
    """
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


def _leja_extend(
        existing_nodes: np.ndarray,
        n_extra: int,
        min_val: float = -2.0,
        max_val: float =  2.0,
        enhance_lo: Optional[float] = None,
        enhance_hi: Optional[float] = None,
        enhance_density_factor: int = 16,
) -> np.ndarray:
    """在已有 Ashkenazy 节点基础上，用 Leja 延伸法追加 n_extra 个节点。

    原理（为什么不重跑 Ashkenazy）
    --------------------------------
    重新以 nc+k 运行 Ashkenazy 会把**所有** nc+k 个节点重新选一遍——
    已有的 nc 个节点会整体偏移（实测：100 个节点中 26 个偏移 >0.001），
    这会完全改变多项式的结构，导致插值远处剧烈振荡。

    Leja 延伸只向现有节点集追加新节点，不移动任何已有节点：
        x_{nc+j} = argmax_{x ∈ candidates} Σ_i log|x − x_i|
    新节点填入使 Vandermonde 行列式增量最大的位置，
    现有多项式结构保持不变，不引入振荡。

    参数
    ----
    existing_nodes        : 已有的 Ashkenazy/Leja 节点（一维，已排序）
    n_extra               : 需要追加的节点数
    enhance_lo/enhance_hi : 若给定，则在该区间内放加密候选点（偏置 Leja）
    enhance_density_factor: 加密区间候选点数 = factor * n_extra（默认 16）
    """
    nc_base = len(existing_nodes)
    # 全域均匀候选点（排除已有节点附近）
    nc_cands = 32 * (nc_base + n_extra)
    cands = np.linspace(min_val, max_val, nc_cands)
    if enhance_lo is not None and enhance_hi is not None:
        nc_extra_cands = enhance_density_factor * max(n_extra, 1)
        cands_enh = np.linspace(enhance_lo, enhance_hi, nc_extra_cands + 2)[1:-1]
        cands = np.sort(np.concatenate([cands, cands_enh]))

    # 去重
    cands = cands[np.concatenate([[True], np.diff(cands) > 1e-12])]

    # 初始化 veca = Σ_i log|c − x_i| （对所有已有节点累加）
    veca = np.zeros(len(cands))
    for x in existing_nodes:
        dv   = (cands - x) ** 2
        veca += np.where(dv < 1e-10, -1e30, np.log(dv))
    # 排除候选点与已有节点重合
    for x in existing_nodes:
        veca[np.abs(cands - x) < 1e-5] = -1e30

    # Leja 贪心：逐个追加 n_extra 个节点
    new_nodes = np.empty(n_extra)
    for j in range(n_extra):
        kmax          = int(np.argmax(veca))
        new_nodes[j]  = cands[kmax]
        dv            = (cands - cands[kmax]) ** 2
        veca         += np.where(dv < 1e-10, -1e30, np.log(dv))

    return np.sort(np.concatenate([existing_nodes, new_nodes]))


def _samp_points_chebyshev(min_val: float, max_val: float, nc: int) -> np.ndarray:
    """第一类 Chebyshev 节点，映射到 [min_val, max_val]。

    x_k = cos((2k+1)π/(2n))，k=0..n-1，两端密、中间稀。
    最小化插值误差上界的节点分布，Lebesgue 常数 ~ (2/π)·log(n)。
    """
    k = np.arange(nc)
    # cos((2k+1)π/(2n)) 在 [-1,1] 上升序排列（k 从大到小对应升序）
    nodes_unit = np.cos((2 * (nc - 1 - k) + 1) * np.pi / (2 * nc))
    return 0.5 * (min_val + max_val) + 0.5 * (max_val - min_val) * nodes_unit


def _samp_points_derivative_adapted(
        min_val: float, max_val: float, nc: int,
        El_list_phys: np.ndarray,
        E1: float,
        beta: float,
        Vmin: float,
        dE: float,
        bg_frac: float = 0.2,
) -> np.ndarray:
    """基于带通窗导数的反 CDF 自适应采样节点。

    原理
    ----
    带通窗 w(x) = 0.5·[tanh(β(x-EL)) - tanh(β(x-ER))] 的导数为：
        |dw/dx| ∝ sech²(β(x-EL)) + sech²(β(x-ER))
    两个 sech² 峰分别集中在过渡点 EL=El-E1 和 ER=El+E1 附近，
    宽度约 4/β（β 越大过渡越陡，峰越窄）。

    算法：以 |dw/dx| + 均匀背景 构造概率密度，对 CDF 反函数均匀采样，
    使过渡区密集放点，通带顶部（函数≈1）和远端衰减区（函数≈0）放较少点。

    参数
    ----
    El_list_phys : 所有滤波中心的物理坐标（所有窗共享同一套节点，
                   密度取各 El 对应 sech² 之和）
    E1           : 带通半宽（Hartree）
    beta         : tanh 过渡陡峭系数
    Vmin         : 物理坐标最小值（scaled→physical 转换用）
    dE           : 能量范围（physical 坐标跨度）
    bg_frac      : 均匀背景占比（0 = 纯自适应；0.2 = 20% 均匀背景）。
                   适当背景保证远端衰减区（函数已为 0）也有几个节点，
                   避免插值在域边界外推失控。
    """
    n_fine = 8000
    s_fine = np.linspace(min_val, max_val, n_fine)  # scaled [-2, 2]
    x_fine = (s_fine + 2.0) * dE / 4.0 + Vmin       # physical

    # 密度 = 各 El 对应两个过渡点的 sech² 之和
    density = np.zeros(n_fine)
    for El in El_list_phys:
        density += 1.0 / np.cosh(beta * (x_fine - (El - E1))) ** 2
        density += 1.0 / np.cosh(beta * (x_fine - (El + E1))) ** 2

    # 归一化后叠加均匀背景
    peak = density.max()
    if peak > 0:
        density /= peak
    density = (1.0 - bg_frac) * density + bg_frac

    # 梯形积分得 CDF，再均匀采样反 CDF
    ds = s_fine[1] - s_fine[0]
    cdf = np.concatenate([[0.0],
                           np.cumsum(0.5 * (density[:-1] + density[1:]) * ds)])
    cdf /= cdf[-1]
    quantiles = (np.arange(nc) + 0.5) / nc
    return np.interp(quantiles, cdf, s_fine)


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
                     k_f: float = 1.0,
                     n0: int = 4,
                     beta: float = 10.0,
                     E1: float = 0.05) -> Callable:
    """
    返回 filter_func(x_phys, El) -> np.ndarray。

    参数
    ----
    filter_type : "gaussian"、"gabor" 或 "bandpass"
    dt          : 高斯参数（仅 filter_type="gaussian" 使用）
    alpha_f     : Gabor 包络衰减系数（仅 filter_type="gabor" 使用）
    k_f         : Gabor 余弦调制频率（仅 filter_type="gabor" 使用）
    n0          : Gabor 包络指数（仅 filter_type="gabor" 使用，默认 4）
    beta        : 带通窗边沿陡峭系数（仅 filter_type="bandpass" 使用，默认 10.0）
    E1          : 带通窗半宽（仅 filter_type="bandpass" 使用，默认 0.05 Hartree）
    """
    if filter_type == "gaussian":
        return lambda x, El: _filt_func_gaussian(x, El, dt)
    elif filter_type == "gabor":
        return lambda x, El: _filt_func_gabor(x, El, alpha_f, k_f, n0)
    elif filter_type == "bandpass":
        return lambda x, El: _filt_func_bandpass(x, El, beta, E1)
    else:
        raise ValueError(
            f"Unknown filter_type={filter_type!r}. "
            "Supported: 'gaussian', 'gabor', 'bandpass'."
        )


# ──────────────────────────────────────────────
# 内部辅助：在稠密网格上批量求值 Newton 多项式
# ──────────────────────────────────────────────

def _eval_newton_grid(samp: np.ndarray,
                      an_row: np.ndarray,
                      s_eval: np.ndarray) -> np.ndarray:
    """在 s_eval 上用前向 Newton Horner 公式对单行系数求值。

    参数
    ----
    samp   : (nc,) 插值节点（缩放坐标）
    an_row : (nc,) Newton 除差系数
    s_eval : (m,)  求值点（缩放坐标）

    返回
    ----
    (m,) 多项式在各 s_eval 处的值
    """
    result = np.full(len(s_eval), an_row[0])
    basis  = np.ones(len(s_eval))
    for j in range(1, len(samp)):
        basis  *= (s_eval - samp[j - 1])
        result += an_row[j] * basis
    return result


# ──────────────────────────────────────────────
# 公开接口：构建 Newton 系数
# ──────────────────────────────────────────────

def build_filter_coefficients(
        El_list: np.ndarray,
        par: PhysParams,
        nc: int,
        filter_func: Optional[Callable] = None,
        samp_method: str = "ashkenazy",
        interval_samp_enhance: Optional[Tuple[float, float]] = None,
        interpolation_tolerance: float = 1e-3,
        enhance_step: int = 10,
        max_enhance_iters: int = 30,
        **samp_kw,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为 El_list 中每个滤波中心构建 Newton 插值系数。

    参数
    ----
    El_list                 : 滤波中心能量列表（物理单位，Hartree）
    par                     : PhysParams（提供 dE、Vmin、dt）
    nc                      : Newton 节点数（初始估计；启用自适应增强后
                              实际节点数 nc_true = len(返回的 samp) 可能更大）
    filter_func             : callable(x_phys, El) -> np.ndarray，可选。
                              若为 None，默认使用高斯滤波（由 par.dt 控制宽度）。
                              可由 make_filter_func() 构造。
    samp_method             : 插值节点选取方式，可选：
                              "ashkenazy"          — 贪心最大化 Vandermonde 行列式（默认）；
                              "chebyshev"          — 第一类 Chebyshev 节点，两端密、中间稀；
                              "derivative_adapted" — 基于带通窗导数 |dw/dx| 的反 CDF 自适应，
                                                     在 EL=El-E1 和 ER=El+E1 过渡区密集放点。
    interval_samp_enhance   : (lo, hi) 物理坐标（Hartree），指定需要加密插值节点的能量区间。
                              若为 None（默认），不做自适应增强。
                              启用后，若所有 El 窗函数的插值最大 MAE > interpolation_tolerance，
                              则每次在该区间内追加 enhance_step 个均匀节点并重新拟合，
                              直到误差达标或迭代次数超过 max_enhance_iters。
    interpolation_tolerance : 插值 MAE 阈值（默认 1e-3）；在全域 [-2, 2] 稠密网格上
                              对所有 El 窗函数取最大 MAE，低于此值停止增强。
    enhance_step            : 每轮在 interval_samp_enhance 内新增的节点数（默认 10）。
    max_enhance_iters       : 最大增强轮数（默认 30），防止不收敛时无限循环。
    **samp_kw               : 额外关键字参数：
                              E1                    (float, 默认 0.05) — derivative_adapted 带通半宽
                              beta                  (float, 默认 10.0) — derivative_adapted tanh 系数
                              bg_frac               (float, 默认 0.2)  — derivative_adapted 均匀背景占比
                              enhance_density_factor (int,  默认 16)   — Leja 延伸时在增强区间内放
                                                     factor × enhance_step 个额外候选点，
                                                     值越大 Leja 新节点越偏向该区间

    返回
    ----
    an   : (ms, nc_true) 系数矩阵，nc_true = len(samp)
    samp : (nc_true,)    [-2, 2] 区间的插值节点
    """
    if filter_func is None:
        filter_func = make_filter_func("gaussian", dt=par.dt)

    Smin, Smax = -2.0, 2.0

    if samp_method == "ashkenazy":
        samp = _samp_points_ashkenazy(Smin, Smax, nc)
    elif samp_method == "chebyshev":
        samp = _samp_points_chebyshev(Smin, Smax, nc)
    elif samp_method == "derivative_adapted":
        samp = _samp_points_derivative_adapted(
            Smin, Smax, nc,
            El_list_phys=np.asarray(El_list),
            E1=samp_kw.get("E1", 0.05),
            beta=samp_kw.get("beta", 10.0),
            Vmin=par.Vmin,
            dE=par.dE,
            bg_frac=samp_kw.get("bg_frac", 0.2),
        )
    else:
        raise ValueError(
            f"Unknown samp_method={samp_method!r}. "
            "Supported: 'ashkenazy', 'chebyshev', 'derivative_adapted'."
        )

    scale = (Smax - Smin) / par.dE

    def _build_an(s: np.ndarray) -> np.ndarray:
        xp = (s + 2.0) / scale + par.Vmin
        a  = np.zeros((len(El_list), len(s)))
        for ie, El in enumerate(El_list):
            a[ie] = _newton_coefficients(s, filter_func(xp, El))
        return a

    an = _build_an(samp)

    # ── 自适应节点增强（Leja 延伸） ────────────────────────────────────
    # 核心设计原则：
    #   绝不重新运行 Ashkenazy(nc+k)——重跑会整体改变所有节点位置，
    #   导致多项式远处剧烈振荡（实测：100 个节点中 26 个偏移 >0.001）。
    #
    #   正确做法：Leja 延伸（_leja_extend）——保持已有 nc 个节点不动，
    #   通过贪心 Vandermonde 最大化依次追加 enhance_step 个新节点。
    #   已有多项式结构完整保留，新节点只"填补"最需要的空隙。
    if interval_samp_enhance is not None:
        lo_phys, hi_phys = interval_samp_enhance
        # 物理坐标 → 缩放坐标 [-2, 2]
        s_lo = max(4.0 * (lo_phys - par.Vmin) / par.dE - 2.0, Smin)
        s_hi = min(4.0 * (hi_phys - par.Vmin) / par.dE - 2.0, Smax)
        enhance_density_factor = samp_kw.get("enhance_density_factor", 16)

        # 误差评估：仅在增强区间内取最大绝对误差（不用全域均值）
        s_eval_enh = np.linspace(s_lo, s_hi, max(200, 20 * enhance_step))
        x_eval_enh = (s_eval_enh + 2.0) / scale + par.Vmin

        def _max_mae(s: np.ndarray, a: np.ndarray) -> float:
            worst = 0.0
            for ie, El in enumerate(El_list):
                y_true   = filter_func(x_eval_enh, El)
                y_interp = _eval_newton_grid(s, a[ie], s_eval_enh)
                worst    = max(worst, float(np.max(np.abs(y_true - y_interp))))
            return worst

        mae     = _max_mae(samp, an)
        n_added = 0

        for _ in range(max_enhance_iters):
            if mae <= interpolation_tolerance:
                break
            # Leja 延伸：保持现有节点不动，仅追加 enhance_step 个新节点
            samp    = _leja_extend(samp, enhance_step,
                                   Smin, Smax, s_lo, s_hi,
                                   enhance_density_factor)
            n_added += enhance_step
            an      = _build_an(samp)
            mae     = _max_mae(samp, an)

        if n_added > 0:
            print(f"   [samp_enhance] Leja-extend  +{n_added} nodes "
                  f"(enhance region [{lo_phys}, {hi_phys}] Hartree) "
                  f"→ nc_true={len(samp)},  max_MAE_in_region={mae:.2e}"
                  + ("  ✓" if mae <= interpolation_tolerance
                     else f"  (tolerance {interpolation_tolerance:.1e} not reached)"))

    return an, samp
