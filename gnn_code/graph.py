"""
graph.py — 稀疏网格图结构构建

提供两种图：

build_graph(N, d, grid_L)
    3×3×3 Mehrstellen 各向同性模板（26 近邻）。
    N, d, grid_L 可选，默认使用 physics.py 中的训练网格参数。
    edge_attr: [dx, dy, dz, dist, w]

build_star_graph(fd_order, n_co, N, d, grid_L)
    高阶十字星模板 + correction 立方体，返回 **两套边**：
      FD 边   — 沿 3 轴高阶差分，每轴 fd_order 个点（不含自身）
               edge_attr: [dx, dy, dz, dist, w]   (w = FD 动能权重)
      Co 边   — n_co³-1 立方体邻域（含轴向近邻），用于 NN correction
               edge_attr: [dx, dy, dz, dist]       (无 w)

两个函数均支持任意 N（均匀周期性网格），可将训练好的 GNN 迁移到更大网格。
图构建已向量化（numpy），N=80 时仍在数秒内完成。
"""

import numpy as np
import torch
from .physics import d_sparse, N_sparse


# ─────────────────────────────────────────────────────────────────────────────
# FD 系数（复用）
# ─────────────────────────────────────────────────────────────────────────────

def _fd2_coefficients(fd_order: int) -> np.ndarray:
    """
    1D 二阶导数 FD 系数，精度 fd_order（偶数）。
    返回长度 2m+1 的数组（m = fd_order//2），偏移量 k = -m,...,+m。
    """
    m = fd_order // 2
    offsets = np.arange(-m, m + 1, dtype=float)
    n_pts = 2 * m + 1
    M = np.array([[offsets[ki]**n for ki in range(n_pts)] for n in range(n_pts)])
    rhs = np.zeros(n_pts)
    rhs[2] = 2.0
    return np.linalg.solve(M, rhs)


# ─────────────────────────────────────────────────────────────────────────────
# 3×3×3 Mehrstellen 图（cube）
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(N: int = None, d: float = None, grid_L: float = None):
    """
    构建均匀周期性网格图（PBC，3×3×3 Mehrstellen 模板，26 近邻）。

    Parameters
    ----------
    N      : 每轴网格点数（默认 physics.py 的 N_sparse）
    d      : 网格步长（Bohr，默认 physics.py 的 d_sparse）
    grid_L : 周期性盒子边长（Bohr，默认 N * d）

    对于均匀周期性网格，相同偏移量 (di,dj,dk) 的所有边属性相同，
    采用向量化实现，支持任意 N。
    """
    if N is None:      N      = N_sparse
    if d is None:      d      = d_sparse
    if grid_L is None: grid_L = N * d   # default: exact periodic box

    S_off = np.array([[3, -4, 3], [-4, 16, -4], [3, -4, 3]])
    S_mid = np.array([[-4, 16, -4], [16, -72, 16], [-4, 16, -4]])
    S     = np.stack([S_off, S_mid, S_off], axis=0)   # (3, 3, 3)
    kin_pref = -1.0 / (24.0 * d**2)

    n_nodes = N ** 3
    flat    = np.arange(n_nodes, dtype=np.int64)
    i_arr, j_arr, k_arr = np.unravel_index(flat, (N, N, N))

    all_src   = []
    all_dst   = []
    all_attrs = []

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue

                # Displacement after PBC unwrapping (uniform grid → constant per offset)
                dr_raw = np.array([di * d, dj * d, dk * d], dtype=np.float64)
                dr     = dr_raw - grid_L * np.round(dr_raw / grid_L)
                dist   = float(np.linalg.norm(dr))
                w      = float(S[di + 1, dj + 1, dk + 1]) * kin_pref

                ni = (i_arr + di) % N
                nj = (j_arr + dj) % N
                nk = (k_arr + dk) % N
                dst = (ni * N + nj) * N + nk

                all_src.append(flat)
                all_dst.append(dst)

                attr = np.empty((n_nodes, 5), dtype=np.float32)
                attr[:] = [float(dr[0]), float(dr[1]), float(dr[2]), dist, w]
                all_attrs.append(attr)

    src_all  = np.concatenate(all_src)
    dst_all  = np.concatenate(all_dst)
    attr_all = np.concatenate(all_attrs, axis=0)

    edge_idx = np.stack([src_all, dst_all], axis=0).astype(np.int64)
    return (
        torch.tensor(edge_idx,  dtype=torch.long),
        torch.tensor(attr_all,  dtype=torch.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 高阶十字星 + correction 立方体（cross / SO3）
# ─────────────────────────────────────────────────────────────────────────────

def build_star_graph(fd_order: int = 4, n_co: int = 3,
                     N: int = None, d: float = None, grid_L: float = None):
    """
    构建两套边：

    1. FD 边（高阶十字星，3 轴）
       edge_attr: [dx, dy, dz, dist, w]   (w = FD 动能权重)

    2. Co 边（correction 立方体，n_co³-1 邻居）
       edge_attr: [dx, dy, dz, dist]

    Parameters
    ----------
    fd_order : FD 精度阶数（偶数）
    n_co     : correction 立方体边长（正整数）。精确生成 n_co³-1 个邻居：
               每轴取 n_co 个偏移量，范围 [-(n_co//2), (n_co-1)//2]，
               跳过中心点。奇数 n_co 对称，偶数 n_co 负方向多一格。
    N        : 每轴网格点数（默认 N_sparse）
    d        : 网格步长（Bohr，默认 d_sparse）
    grid_L   : 周期性盒子边长（Bohr，默认 N * d）
    """
    if N is None:      N      = N_sparse
    if d is None:      d      = d_sparse
    if grid_L is None: grid_L = N * d

    # ── FD 系数 ──
    m          = fd_order // 2
    c          = _fd2_coefficients(fd_order)
    offsets_1d = np.arange(-m, m + 1, dtype=int)
    fd_w       = {int(k): float(-c[ki] / (2.0 * d**2))
                  for ki, k in enumerate(offsets_1d) if k != 0}

    # n_co 直接决定每轴取 n_co 个偏移，确保 n_co³-1 个邻居：
    #   偶数 n_co：[-n_co//2, (n_co-1)//2]（负方向多一格）
    #   奇数 n_co：[-(n_co-1)//2, (n_co-1)//2]（对称）
    co_neg = -(n_co // 2)
    co_pos = (n_co - 1) // 2
    AXES  = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    n_nodes = N ** 3
    flat    = np.arange(n_nodes, dtype=np.int64)
    i_arr, j_arr, k_arr = np.unravel_index(flat, (N, N, N))

    fd_ei_list, fd_ea_list = [], []
    co_ei_list, co_ea_list = [], []

    # ── 1. FD 边 ──
    for ex, ey, ez in AXES:
        for step, w in fd_w.items():
            dr_raw = np.array([step * ex * d, step * ey * d, step * ez * d], dtype=np.float64)
            dr     = dr_raw - grid_L * np.round(dr_raw / grid_L)
            dist   = float(np.linalg.norm(dr))

            ni = (i_arr + step * ex) % N
            nj = (j_arr + step * ey) % N
            nk = (k_arr + step * ez) % N
            dst = (ni * N + nj) * N + nk

            fd_ei_list.append(np.stack([flat, dst], axis=0))
            attr = np.empty((n_nodes, 5), dtype=np.float32)
            attr[:] = [float(dr[0]), float(dr[1]), float(dr[2]), dist, w]
            fd_ea_list.append(attr)

    # ── 2. Co 边 ──
    for di in range(co_neg, co_pos + 1):
        for dj in range(co_neg, co_pos + 1):
            for dk in range(co_neg, co_pos + 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue

                dr_raw = np.array([di * d, dj * d, dk * d], dtype=np.float64)
                dr     = dr_raw - grid_L * np.round(dr_raw / grid_L)
                dist   = float(np.linalg.norm(dr))

                ni = (i_arr + di) % N
                nj = (j_arr + dj) % N
                nk = (k_arr + dk) % N
                dst = (ni * N + nj) * N + nk

                co_ei_list.append(np.stack([flat, dst], axis=0))
                attr = np.empty((n_nodes, 4), dtype=np.float32)
                attr[:] = [float(dr[0]), float(dr[1]), float(dr[2]), dist]
                co_ea_list.append(attr)

    fd_ei = torch.tensor(np.concatenate(fd_ei_list, axis=1), dtype=torch.long)
    fd_ea = torch.tensor(np.concatenate(fd_ea_list, axis=0), dtype=torch.float32)
    if co_ei_list:
        co_ei = torch.tensor(np.concatenate(co_ei_list, axis=1), dtype=torch.long)
        co_ea = torch.tensor(np.concatenate(co_ea_list, axis=0), dtype=torch.float32)
    else:
        co_ei = torch.zeros((2, 0), dtype=torch.long)
        co_ea = torch.zeros((0, 4), dtype=torch.float32)

    return fd_ei, fd_ea, co_ei, co_ea
