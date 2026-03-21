"""
ho3d_solvers_v2.py
增强版：支持两种模式和多网格初始猜测
1. 标准模式：求解最低的n_levels个特征值
2. 目标模式：求解最接近target能量的n_levels个特征值
3. 多网格模式：使用粗网格解作为细网格初始猜测
4. 正确处理cube文件的空间范围
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
import os
import matplotlib.pyplot as plt

try:
    import primme
    HAS_PRIMME = True
except ImportError:
    HAS_PRIMME = False

try:
    from grid_reader import PotentialGrid
    HAS_GRID_READER = True
except ImportError:
    HAS_GRID_READER = False
    PotentialGrid = None

@dataclass
class SolveResult:
    evals: np.ndarray
    evecs: Optional[np.ndarray]
    meta: Dict[str, Union[int, float, str]]

# ---------------------------
# 内部辅助函数
# ---------------------------

def _parse_method_spec(method_str: str) -> Tuple[str, str]:
    """解析 '离散化:求解器' 字符串"""
    parts = method_str.split(":")
    if len(parts) == 1:
        m = parts[0].lower()
        if m in ["gd", "lobpcg", "davidson", "jd"]: return "sinc_dvr", m
        return m, "lanczos"
    return parts[0].lower(), parts[1]

def _label(disc: str, solver: str) -> str:
    """生成图表显示的专业标签"""
    disc_map = {"numerov3d": "Numerov-3D", "chebyshev": "Chebyshev", "sinc_dvr": "Sinc-DVR"}
    
    s_clean = solver
    if s_clean.upper().startswith("PRIMME_"):
        s_clean = s_clean[7:]
    
    if s_clean.lower() == "lanczos": s_clean = "Lanczos"
    elif s_clean.lower() == "lobpcg": s_clean = "LOBPCG"
    elif s_clean.lower() == "davidson": s_clean = "Davidson"
    elif s_clean.lower() == "cgmin": s_clean = "CG-Min"
    
    d = disc_map.get(disc, disc.capitalize())
    return f"{d} ({s_clean})"

def _kronsum_3(D: sp.spmatrix, I: sp.spmatrix) -> sp.spmatrix:
    return (sp.kron(sp.kron(D, I), I, format="csc") +
            sp.kron(sp.kron(I, D), I, format="csc") +
            sp.kron(sp.kron(I, I), D, format="csc"))

def _sort_eigs(evals: np.ndarray, evecs: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    idx = np.argsort(evals)
    evals = np.asarray(evals)[idx]
    if evecs is None: return evals, None
    return evals, evecs[:, idx]

def _get_spatial_range(potential_grid: Optional[PotentialGrid], L: float) -> Tuple[float, float]:
    """
    获取空间范围
    
    Returns:
    --------
    x_min, x_max : 空间坐标的最小值和最大值
    """
    if potential_grid is None:
        # 谐振子：使用L参数
        return -L, L
    else:
        # Custom potential：从grid读取
        x_min = potential_grid.x[0]
        x_max = potential_grid.x[-1]
        return x_min, x_max

# ---------------------------
# 多网格插值函数
# ---------------------------

def interpolate_eigenvector(evec_coarse: np.ndarray, N_coarse: int, N_fine: int, 
                           potential_grid: Optional[PotentialGrid] = None,
                           L: float = 5.0) -> np.ndarray:
    """
    将粗网格特征向量插值到细网格
    
    Parameters:
    -----------
    evec_coarse : np.ndarray
        粗网格特征向量 (N_coarse^3,) 或 (N_coarse^3, k)
    N_coarse : int
        粗网格分辨率
    N_fine : int
        细网格分辨率
    potential_grid : PotentialGrid, optional
        势能网格（用于获取坐标范围）
    L : float
        空间范围参数（仅用于谐振子）
        
    Returns:
    --------
    evec_fine : np.ndarray
        细网格特征向量 (N_fine^3,) 或 (N_fine^3, k)
    """
    # 确定坐标范围
    x_min, x_max = _get_spatial_range(potential_grid, L)
    
    # 创建粗网格坐标
    x_coarse = np.linspace(x_min, x_max, N_coarse)
    
    # 处理单个向量或多个向量
    if evec_coarse.ndim == 1:
        evec_coarse = evec_coarse.reshape(-1, 1)
        single_vector = True
    else:
        single_vector = False
    
    n_vecs = evec_coarse.shape[1]
    evec_fine_list = []
    
    for i in range(n_vecs):
        # 将1D向量重塑为3D
        psi_coarse = evec_coarse[:, i].reshape((N_coarse, N_coarse, N_coarse))
        
        # 创建插值器
        interpolator = RegularGridInterpolator(
            (x_coarse, x_coarse, x_coarse), 
            psi_coarse,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 创建细网格坐标
        x_fine = np.linspace(x_min, x_max, N_fine)
        
        X_fine, Y_fine, Z_fine = np.meshgrid(x_fine, x_fine, x_fine, indexing='ij')
        points_fine = np.stack([X_fine.ravel(), Y_fine.ravel(), Z_fine.ravel()], axis=1)
        
        # 插值
        psi_fine = interpolator(points_fine)
        
        # 归一化
        psi_fine = psi_fine / np.linalg.norm(psi_fine)
        
        evec_fine_list.append(psi_fine)
    
    evec_fine = np.column_stack(evec_fine_list)
    
    if single_vector:
        return evec_fine.ravel()
    else:
        return evec_fine

# ---------------------------
# 离散化算子构造
# ---------------------------

def build_numerov3d_matrices(N: int, potential_grid: Optional[PotentialGrid] = None, L: float = 5.0):
    """
    构建Numerov 3D矩阵
    
    Parameters:
    -----------
    N : int
        网格点数
    potential_grid : PotentialGrid, optional
        自定义势能网格
    L : float
        空间范围（仅用于谐振子）
    """
    if potential_grid is None:
        # 谐振子：使用L参数
        x = np.linspace(-L, L, N)
        h = x[1] - x[0]
        Nx = N - 2
        X, Y, Z = np.meshgrid(x[1:-1], x[1:-1], x[1:-1], indexing="ij")
        V = (0.5 * (X**2 + Y**2 + Z**2)).reshape(Nx, Nx, Nx)
    else:
        # Custom potential：从grid读取
        if potential_grid.Nx != N:
            from grid_reader import resample_potential
            potential_grid = resample_potential(potential_grid, N)
        V_full = potential_grid.potential
        V = V_full[1:-1, 1:-1, 1:-1]
        Nx = N - 2
        x = potential_grid.x
        h = x[1] - x[0]
    
    S_off = np.array([[3, -4, 3], [-4, 16, -4], [3, -4, 3]])
    S_mid = np.array([[-4, 16, -4], [16, -72, 16], [-4, 16, -4]])
    kin_pref = -1.0 / (2.0 * h * h)
    rowsA, colsA, valsA = [], [], []
    rowsB, colsB, valsB = [], [], []
    for K in range(Nx):
        for J in range(Nx):
            for I in range(Nx):
                p = (K * Nx + J) * Nx + I
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        w = S_mid[dy+1, dx+1]
                        if 0 <= I+dx < Nx and 0 <= J+dy < Nx:
                            q = (K * Nx + (J+dy)) * Nx + (I+dx)
                            rowsA.append(p); colsA.append(q); valsA.append(kin_pref * w)
                for dz in (-1, 1):
                    if 0 <= K+dz < Nx:
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                w = S_off[dy+1, dx+1]
                                if 0 <= I+dx < Nx and 0 <= J+dy < Nx:
                                    q = ((K+dz) * Nx + (J+dy)) * Nx + (I+dx)
                                    rowsA.append(p); colsA.append(q); valsA.append(kin_pref * w)
                rowsA.append(p); colsA.append(p); valsA.append(6.0 * V[I, J, K])
                rowsB.append(p); colsB.append(p); valsB.append(6.0)
                for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                    if 0 <= I+di < Nx and 0 <= J+dj < Nx and 0 <= K+dk < Nx:
                        q = ((K+dk) * Nx + (J+dj)) * Nx + (I+di)
                        valsA.append(1.0 * V[I+di, J+dj, K+dk]); rowsA.append(p); colsA.append(q)
                        valsB.append(1.0); rowsB.append(p); colsB.append(q)
    return sp.csc_matrix((valsA, (rowsA, colsA))), sp.csc_matrix((valsB, (rowsB, colsB)))

def build_3d_sinc_dvr_operator(N: int, potential_grid: Optional[PotentialGrid] = None, L: float = 5.0):
    """
    构建3D Sinc-DVR算子
    
    Parameters:
    -----------
    N : int
        网格点数
    potential_grid : PotentialGrid, optional
        自定义势能网格
    L : float
        空间范围（仅用于谐振子）
    """
    if potential_grid is None:
        # 谐振子：使用L参数定义范围 [-L, L]
        Lbox = 2.0 * L
        h = Lbox / (N - 1)
        x = np.linspace(-L, L, N)
        V_flat = 0.5 * (np.add.outer(np.add.outer(x**2, x**2), x**2)).reshape(-1)
    else:
        # Custom potential：从grid读取真实范围
        if potential_grid.Nx != N:
            from grid_reader import resample_potential
            potential_grid = resample_potential(potential_grid, N)
        x = potential_grid.x
        V_flat = potential_grid.potential.ravel()
        h = x[1] - x[0]
    
    i, j = np.indices((N, N))
    diff = i - j
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = diff.astype(float)**2
        T1d = np.where(diff == 0, np.pi**2/3.0, 2.0*((-1.0)**diff)/denom)
    T1d = (1.0 / (2.0 * h * h)) * T1d
    
    def matvec(v: np.ndarray) -> np.ndarray:
        k = 1 if v.ndim == 1 else v.shape[1]
        psi = v.reshape((N, N, N, k))
        res = (np.einsum("ij,jklm->iklm", T1d, psi) +
               np.einsum("ij,kjlm->kilm", T1d, psi) +
               np.einsum("ij,kljm->klim", T1d, psi)).reshape(-1, k)
        return res + V_flat[:, np.newaxis] * v.reshape(-1, k)
        
    return spla.LinearOperator((N**3, N**3), matvec=matvec, dtype=float), N**3, V_flat

# ---------------------------
# CG-Minimization (Folded Spectrum)
# ---------------------------

def _sphere_geodesic(x: np.ndarray, p: np.ndarray):
    """
    单位球面上的测地线：x(θ) = x cosθ + p̂ sinθ，其中 x·p=0。
    返回 (x_of_theta 函数, ||p||)。
    """
    pn = np.linalg.norm(p)
    if pn == 0.0:
        return lambda th: x.copy(), 0.0
    p_hat = p / pn
    return lambda th: x * np.cos(th) + p_hat * np.sin(th), pn


def _golden_section_search(func, a: float, b: float,
                           tol: float = 1e-4, maxit: int = 60):
    """
    黄金分割法最小化 func(theta) 在 [a, b] 上。
    返回 (theta_min, func_min)。
    """
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = func(c), func(d)
    for _ in range(maxit):
        if (b - a) <= tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = func(d)
    th = 0.5 * (a + b)
    return th, func(th)


def cg_minimize_folded(H_op, E_target: float, x0: np.ndarray,
                       maxiter: int = 500, gtol: float = 1e-8,
                       theta_max: float = 0.8,
                       verbose: bool = False) -> dict:
    """
    在单位球面上用非线性CG最小化折叠谱目标函数 f(x) = ||(H-E_target)x||²。

    适用于稀疏矩阵、scipy.sparse.linalg.LinearOperator 或支持 @ 运算符的任何算子。
    找到满足 (H-E_target)x≈0 的特征向量，即最接近 E_target 的本征态。

    Parameters
    ----------
    H_op : matrix or LinearOperator
        Hamiltonian 算子，支持 H_op @ v。
    E_target : float
        目标能量（折叠谱中心）。
    x0 : np.ndarray, shape (n,)
        初始猜测向量（无需归一化）。
    maxiter : int
        最大迭代次数。
    gtol : float
        梯度范数收敛阈值。
    theta_max : float
        线搜索区间上界（弧度），通常 0.5–1.0。
    verbose : bool
        是否打印迭代信息。

    Returns
    -------
    dict with keys:
        'x'      : 收敛的单位特征向量
        'E_ritz' : Ritz 能量 <x|H|x>
        'f_val'  : 最终目标函数值
        'n_iter' : 实际迭代次数
        'history': {'f', 'gnorm', 'E_ritz'} 列表
    """
    def A(v):
        return H_op @ v - E_target * v

    x = np.array(x0, dtype=float)
    x /= np.linalg.norm(x)

    def f_of(xv):
        y = A(xv)
        return float(np.dot(y, y))

    def grad_tangent(xv):
        y = A(xv)          # (H-E)x
        z = A(y)           # (H-E)²x
        g = 2.0 * z
        g -= np.dot(xv, g) * xv   # 投影到切空间
        return g, y

    hist = {"f": [], "gnorm": [], "E_ritz": []}
    f0 = f_of(x)
    g, _ = grad_tangent(x)
    p = -g - np.dot(x, -g) * x    # 初始搜索方向（切空间内）
    g_prev, p_prev = g.copy(), p.copy()

    E_ritz = float(np.dot(x, H_op @ x))
    n_iter = 0

    for it in range(1, maxiter + 1):
        n_iter = it
        gnorm = np.linalg.norm(g)
        E_ritz = float(np.dot(x, H_op @ x))
        hist["f"].append(f0)
        hist["gnorm"].append(gnorm)
        hist["E_ritz"].append(E_ritz)

        if gnorm < gtol:
            if verbose:
                print(f"  CG 收敛 it={it}: ||g||={gnorm:.3e}, E≈{E_ritz:.10f}")
            break

        # 重新投影方向到切空间
        p = p - np.dot(x, p) * x
        if np.linalg.norm(p) == 0.0:
            if verbose:
                print("  方向消失，停止。")
            break

        x_of_th, _ = _sphere_geodesic(x, p)

        def phi(th):
            return f_of(x_of_th(th))

        theta, f_new = _golden_section_search(phi, 0.0, theta_max, tol=1e-4)
        x = x_of_th(theta)
        x /= np.linalg.norm(x)

        # Polak-Ribière beta（带重启）
        g, _ = grad_tangent(x)
        dg = g - g_prev
        denom = float(np.dot(g_prev, g_prev)) + 1e-30
        beta = max(0.0, float(np.dot(g, dg)) / denom)
        p = -g + beta * p_prev
        p = p - np.dot(x, p) * x

        g_prev, p_prev = g.copy(), p.copy()
        f0 = f_new

        if verbose and (it % 50 == 0 or it == 1):
            print(f"  it={it:5d}: f={f0:.4e}, ||g||={gnorm:.3e}, "
                  f"theta={theta:.3e}, E≈{E_ritz:.10f}")

    return {"x": x, "E_ritz": E_ritz, "f_val": f0,
            "n_iter": n_iter, "history": hist}


# ---------------------------
# 核心求解器（增强版）
# ---------------------------

def solve_ho3d(method_spec: str, N: int, n_levels: int, 
               potential_grid: Optional[PotentialGrid] = None,
               L: float = 5.0,
               target: Optional[float] = None,
               v0: Optional[np.ndarray] = None,
               maxBlockSize: int = 1, **kwargs):
    """
    求解3D薛定谔方程（增强版，支持多网格）
    
    Parameters:
    -----------
    method_spec : str
        方法规格 "离散化:求解器"
    N : int
        网格点数
    n_levels : int
        求解的能级数
    potential_grid : PotentialGrid, optional
        自定义势能网格。如果为None，使用谐振子势能
    L : float
        空间范围（仅用于谐振子，[-L, L]）
        对于custom potential，此参数被忽略，使用cube文件的真实范围
    target : float, optional
        目标能量。如果指定，则求解最接近target的n_levels个特征值
        如果为None，则求解最低的n_levels个特征值
    v0 : np.ndarray, optional
        初始猜测向量 (N^3, k)，用于多网格方法
        如果提供，将作为迭代的初始猜测
    maxBlockSize : int
        PRIMME块大小
    **kwargs : 
        传递给求解器的额外参数
        
    Returns:
    --------
    SolveResult
        求解结果
    """
    disc, solver_method = _parse_method_spec(method_spec)
    
    t_start = time.perf_counter()
    
    if disc == "numerov3d":
        H_op, M_op = build_numerov3d_matrices(N, potential_grid, L)
        n_un = (N-2)**3
    elif disc == "chebyshev":
        # 简化的Chebyshev离散化
        T = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j: T[i, j] = (N**2 - 1.0) / 3.0 if i > 0 else (N**2 - 1.0) / 6.0
                elif (i + j) % 2 == 1: T[i, j] = 2.0 * ((-1.0)**(i - j)) / ((i - j)**2)
        D = T @ np.diag(np.sqrt(1.0 - np.cos(np.pi * np.arange(N) / (N - 1))**2))
        
        if potential_grid is None:
            # 谐振子
            x = L * np.cos(np.pi * np.arange(N) / (N - 1))[::-1]
            Xg, Yg, Zg = np.meshgrid(x[1:-1], x[1:-1], x[1:-1], indexing="ij")
            sl = slice(1, -1); n1 = N-2
            H_op = (-0.5) * _kronsum_3(sp.csc_matrix(D[sl, sl] @ D[sl, sl]), sp.eye(n1))
            H_op += sp.diags(0.5 * (Xg**2 + Yg**2 + Zg**2).ravel(), 0, format="csc")
        else:
            # Custom potential
            if potential_grid.Nx != N:
                from grid_reader import resample_potential
                potential_grid = resample_potential(potential_grid, N)
            x_range = potential_grid.x[-1] - potential_grid.x[0]
            scale = 2.0 / x_range
            D2 = (scale**2) * (D @ D)
            sl = slice(1, -1); n1 = N-2
            H_op = (-0.5) * _kronsum_3(sp.csc_matrix(D2[sl, sl]), sp.eye(n1))
            V_interior = potential_grid.potential[1:-1, 1:-1, 1:-1]
            H_op += sp.diags(V_interior.ravel(), 0, format="csc")
        n_un, M_op = n1**3, None
    else:  # sinc_dvr
        H_op, n_un, V_diag = build_3d_sinc_dvr_operator(N, potential_grid, L)
        
        # 获取正确的步长
        if potential_grid is None:
            h = (2.0*L)/(N-1)
        else:
            h = potential_grid.x[1] - potential_grid.x[0]
        
        T_diag_val = (np.pi**2)/(2.0*h**2)
        shift = target if target is not None else 0.0
        M_vals = 1.0 / (T_diag_val + V_diag - shift + 1e-3)
        M_op = sp.diags(M_vals, format='csr')
    
    t_build = time.perf_counter() - t_start
    t_eig_start = time.perf_counter()
    
    # 求解器分发（根据是否有target选择不同模式）
    if solver_method.lower() == "lanczos":
        if target is not None:
            # Shift-invert模式：求解最接近target的特征值
            evals, evecs = spla.eigsh(H_op, k=n_levels, 
                                     M=M_op if disc=="numerov3d" else None,
                                     sigma=target,  # shift值
                                     which='LM',  # Largest Magnitude after shift
                                     v0=v0,  # 初始猜测
                                     **kwargs)
        else:
            # 标准模式：求解最小的特征值
            evals, evecs = spla.eigsh(H_op, k=n_levels, 
                                     M=M_op if disc=="numerov3d" else None,
                                     which="SA",  # Smallest Algebraic
                                     v0=v0,  # 初始猜测
                                     **kwargs)
    
    elif solver_method.lower() == "lobpcg":
        if v0 is not None and v0.shape[1] >= n_levels:
            X0 = v0[:, :n_levels]
        else:
            X0 = np.random.standard_normal((n_un, n_levels))
        evals, evecs = spla.lobpcg(H_op, X0,
                                  M=M_op if disc=="sinc_dvr" else None,
                                  largest=False, **kwargs)
        # LOBPCG不支持target模式，如果指定target会给出警告
        if target is not None:
            print("Warning: LOBPCG does not support target mode, computing smallest eigenvalues")

    elif solver_method.lower() == "cgmin":
        # ---- 折叠谱 CG 最小化 ----
        if target is None:
            raise ValueError("cgmin solver 需要指定 target 能量")

        cg_maxiter  = kwargs.get("cg_maxiter",  500)
        cg_gtol     = kwargs.get("cg_gtol",     1e-8)
        cg_theta_max= kwargs.get("cg_theta_max",0.8)
        cg_verbose  = kwargs.get("cg_verbose",  False)

        rng = np.random.default_rng(42)
        evals_list, evecs_list = [], []
        found_vecs = []   # 已找到的本征向量，用于投影偏移

        for k in range(n_levels):
            # 初始向量：若提供 v0 则取第 k 列，否则随机
            if v0 is not None and v0.ndim == 1 and k == 0:
                x0_k = v0.copy()
            elif v0 is not None and v0.ndim == 2 and k < v0.shape[1]:
                x0_k = v0[:, k].copy()
            else:
                x0_k = rng.standard_normal(n_un)
            # 对已找到的态正交化
            for fv in found_vecs:
                x0_k -= np.dot(fv, x0_k) * fv
            norm_k = np.linalg.norm(x0_k)
            if norm_k < 1e-14:
                x0_k = rng.standard_normal(n_un)
            x0_k /= np.linalg.norm(x0_k)

            # 构建偏移算子：已找到的态加上大的惩罚，使其远离 E_target
            if found_vecs:
                shift_val = 100.0
                _fv_arr = found_vecs  # closure
                def _deflated_matvec(v, _fvs=_fv_arr, _s=shift_val):
                    Hv = H_op @ v
                    for fv in _fvs:
                        Hv = Hv + _s * np.dot(fv, v) * fv
                    return Hv
                H_eff = spla.LinearOperator((n_un, n_un),
                                            matvec=_deflated_matvec, dtype=float)
            else:
                H_eff = H_op

            res_cg = cg_minimize_folded(
                H_eff, E_target=target, x0=x0_k,
                maxiter=cg_maxiter, gtol=cg_gtol,
                theta_max=cg_theta_max, verbose=cg_verbose,
            )

            xk = res_cg["x"]
            # Ritz 能量用原始（未偏移）H 计算
            Ek = float(np.dot(xk, H_op @ xk))
            evals_list.append(Ek)
            evecs_list.append(xk)
            found_vecs.append(xk)

            if cg_verbose or True:  # 总是打印每个态的结果
                print(f"  [cgmin] state {k}: E={Ek:.10f}, "
                      f"n_iter={res_cg['n_iter']}, "
                      f"f={res_cg['f_val']:.3e}, "
                      f"||g||≈{res_cg['history']['gnorm'][-1]:.3e}")

        evals = np.array(evals_list)
        evecs = np.column_stack(evecs_list)

    else:
        # PRIMME系列求解器
        if not HAS_PRIMME:
            raise ImportError("PRIMME required")
        
        p_map = {
            "gd": "PRIMME_GD",
            "davidson": "PRIMME_DEFAULT_METHOD",
            "jd": "PRIMME_JDQMR"
        }
        
        actual_method = solver_method if solver_method.upper().startswith("PRIMME_") else \
                       p_map.get(solver_method.lower(), "PRIMME_DEFAULT_METHOD")
        
        # 设置默认ncv，但允许kwargs覆盖
        default_ncv = max(80, 2*n_levels)
        ncv_value = kwargs.pop('ncv', default_ncv)  # 从kwargs中提取ncv，如果没有则使用默认值
        
        if target is not None:
            # 目标模式：求解最接近target的特征值
            evals, evecs = primme.eigsh(
                H_op,
                k=n_levels,
                which=target,   # 关键修改
                OPinv=M_op if disc=="sinc_dvr" else None,
                method=actual_method,
                maxBlockSize=maxBlockSize,
                ncv=ncv_value,
                v0=v0,  # 初始猜测
                **kwargs
            )

        else:
            # 标准模式：求解最小的特征值
            evals, evecs = primme.eigsh(H_op, k=n_levels, 
                                       which='SA',  # Smallest Algebraic
                                       OPinv=M_op if disc=="sinc_dvr" else None,
                                       method=actual_method,
                                       maxBlockSize=maxBlockSize,
                                       ncv=ncv_value,
                                       v0=v0,  # 初始猜测
                                       **kwargs)
    
    t_eig = time.perf_counter() - t_eig_start
    evals, evecs = _sort_eigs(evals, evecs)
    
    pot_source = "Custom Grid" if potential_grid is not None else "Harmonic Oscillator"
    mode = f"target≈{target}" if target is not None else "smallest"
    used_v0 = "Yes" if v0 is not None else "No"
    
    # 记录使用的空间范围
    x_min, x_max = _get_spatial_range(potential_grid, L)
    
    return SolveResult(evals, evecs, {
        "method": method_spec, 
        "method_label": _label(disc, solver_method), 
        "N": N, 
        "total_nodes": N**3, 
        "t_build": t_build, 
        "t_eig": t_eig,
        "potential": pot_source,
        "mode": mode,
        "used_v0": used_v0,
        "spatial_range": (x_min, x_max),  # 记录实际使用的空间范围
    })

# ---------------------------
# Benchmark & Plot
# ---------------------------

def benchmark_methods(methods, N_list, n_levels=9, L=5.0, 
                     potential_grid: Optional[PotentialGrid] = None,
                     target: Optional[float] = None,
                     outdir="figs_methods", maxBlockSize=0, **kwargs):
    """基准测试多个方法"""
    os.makedirs(outdir, exist_ok=True)
    records = []
    
    pot_info = "Custom Potential" if potential_grid is not None else "Harmonic Oscillator"
    mode_info = f"Target≈{target}" if target is not None else "Smallest eigenvalues"
    print(f"\n{'='*60}")
    print(f"Benchmarking with: {pot_info}")
    print(f"Mode: {mode_info}")
    print(f"{'='*60}")
    
    for m in methods:
        print(f"\n--- Running: {m} ---")
        for N in N_list:
            try:
                res = solve_ho3d(m, N=int(N), L=float(L), n_levels=int(n_levels), 
                               potential_grid=potential_grid,
                               target=target,
                               maxBlockSize=maxBlockSize, **kwargs)
                records.append(res.meta)
                print(f"  N={N:2d} (Nodes={N**3:6d}) evals[:3]: {res.evals[:3]}")
            except Exception as e:
                print(f"  N={N} failed: {repr(e)}")
    
    return records

def plot_method_comparison(records, outdir="figs_methods", title_prefix="HO3D"):
    """生成性能对比图"""
    if not records:
        return
    
    os.makedirs(outdir, exist_ok=True)
    methods = sorted(list(set(r["method"] for r in records)))
    
    pot_type = records[0].get("potential", "Unknown")
    mode_type = records[0].get("mode", "standard")
    full_title = f"{title_prefix} - {pot_type} ({mode_type})"
    
    # 1) Solver Time vs N^3
    plt.figure(figsize=(10, 6))
    for m in methods:
        sub = sorted([r for r in records if r["method"] == m], key=lambda x: x["N"])
        if not sub:
            continue
        plt.plot([r["total_nodes"] for r in sub], [r["t_eig"] for r in sub], 
                "o-", label=sub[0]["method_label"], linewidth=2, markersize=6)
    
    plt.xlabel("Total Grid Points (N³)", fontsize=12)
    plt.ylabel("Solver Time (s)", fontsize=12)
    plt.title(f"{full_title} - Solver Time", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_vs_N3.png"), dpi=200)
    print(f"Saved: {os.path.join(outdir, 'time_vs_N3.png')}")
    
    # 2) Log-Log Scaling
    plt.figure(figsize=(10, 6))
    for m in methods:
        sub = sorted([r for r in records if r["method"] == m], key=lambda x: x["N"])
        if not sub:
            continue
        
        x_vals = np.array([r["total_nodes"] for r in sub], dtype=float)
        y_vals = np.array([r["t_eig"] for r in sub], dtype=float)
        mask = (x_vals > 0) & (y_vals > 0)
        
        if np.sum(mask) >= 2:
            slope, intercept = np.polyfit(np.log(x_vals[mask]), np.log(y_vals[mask]), 1)
            label = f"{sub[0]['method_label']} (slope={slope:.2f})"
        else:
            label = sub[0]['method_label']
        
        plt.loglog(x_vals, y_vals, "o-", label=label, linewidth=2, markersize=6)
    
    plt.xlabel("Total Grid Points (N³)", fontsize=12)
    plt.ylabel("Time (s)", fontsize=12)
    plt.title(f"{full_title} - Scaling Analysis", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loglog_fit.png"), dpi=200)
    print(f"Saved: {os.path.join(outdir, 'loglog_fit.png')}")
    
    plt.show()


if __name__ == "__main__":
    print("=== Testing ho3d_solvers_v2.py ===\n")
    
    # 测试1：标准模式
    print("Test 1: Standard mode (smallest eigenvalues)")
    res = solve_ho3d("sinc_dvr:lanczos", N=11, L=5.0, n_levels=5)
    print(f"Eigenvalues: {res.evals}")
    print(f"Spatial range: {res.meta['spatial_range']}")
    
    # 测试2：目标模式
    print("\nTest 2: Target mode (eigenvalues near -0.25)")
    res = solve_ho3d("sinc_dvr:lanczos", N=11, L=5.0, n_levels=5, target=-0.25)
    print(f"Eigenvalues: {res.evals}")
    print(f"Spatial range: {res.meta['spatial_range']}")
    
    # 测试3：多网格模式
    print("\nTest 3: Multigrid mode")
    # 先求解粗网格
    res_coarse = solve_ho3d("sinc_dvr:PRIMME_JDQMR", N=10, L=5.0, n_levels=3, target=-0.25)
    print(f"Coarse grid (N=10) eigenvalues: {res_coarse.evals}")
    
    # 插值到细网格
    v0_fine = interpolate_eigenvector(res_coarse.evecs, N_coarse=10, N_fine=13, L=5.0)
    print(f"Interpolated v0 shape: {v0_fine.shape}")
    
    # 使用插值结果作为初始猜测
    res_fine = solve_ho3d("sinc_dvr:PRIMME_JDQMR", N=13, L=5.0, n_levels=3, 
                         target=-0.25, v0=v0_fine)
    print(f"Fine grid (N=13) eigenvalues: {res_fine.evals}")
    print(f"Used v0: {res_fine.meta['used_v0']}")
