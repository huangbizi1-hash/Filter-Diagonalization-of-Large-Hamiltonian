"""
graph.py — 稀疏网格图结构构建

提供两种图：

build_graph()
    3×3×3 Mehrstellen 各向同性模板（26 近邻）。
    edge_attr: [dx, dy, dz, dist, w]

build_star_graph(fd_order, n_co)
    高阶十字星模板 + correction 立方体，返回 **两套边**：
      FD 边   — 沿 3 轴高阶差分，每轴 fd_order 个点（不含自身）
               edge_attr: [dx, dy, dz, dist, w]   (w = FD 动能权重)
      Co 边   — n_co³-1 立方体邻域（含轴向近邻），用于 NN correction
               edge_attr: [dx, dy, dz, dist]       (无 w)
"""

import numpy as np
import torch
from .physics import X_s, Y_s, Z_s, d_sparse, N_sparse, L


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────

def _node_idx(i: int, j: int, k: int) -> int:
    return (i % N_sparse) * N_sparse**2 + (j % N_sparse) * N_sparse + (k % N_sparse)


def _pbc_delta(pos_v: np.ndarray, pos_u: np.ndarray) -> np.ndarray:
    """PBC 最短距离向量。"""
    dr = pos_v - pos_u
    dr -= L * np.round(dr / L)
    return dr


def _fd2_coefficients(fd_order: int) -> np.ndarray:
    """
    1D 二阶导数 FD 系数，精度 fd_order（偶数）。

    返回长度 2m+1 的数组 c（m = fd_order//2），
    对应偏移量 k = -m,...,0,...,+m，满足：
        (1/h²) Σ_k c[k+m] ψ_{i+k}  ≈  ψ''_i
    等价条件：Σ c[k] k^n = 2·δ_{n,2}  for n = 0,...,2m。
    """
    m = fd_order // 2
    offsets = np.arange(-m, m + 1, dtype=float)
    n_pts = 2 * m + 1
    # M[n, k_idx] = offsets[k_idx]^n
    M = np.array([[offsets[ki]**n for ki in range(n_pts)] for n in range(n_pts)])
    rhs = np.zeros(n_pts)
    rhs[2] = 2.0   # only the 2nd power condition
    return np.linalg.solve(M, rhs)   # shape (n_pts,)


# ─────────────────────────────────────────────────────────────────────────────
# 3×3×3 Mehrstellen 图（原有）
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    构建稀疏网格图（PBC，3×3×3 模板）。

    差分权重系数 S[di,dj,dk]（各向同性 4 阶拉普拉斯）：
        角点  : +3  面中心: +16  棱中心: -4
        中心  : -72（不含自身，纳入消息传递时处理）
    prefactor = -1 / (24 d²)，使最终结果对应 -1/2 ∇² 算子。
    """
    S_off = np.array([[3, -4, 3], [-4, 16, -4], [3, -4, 3]])
    S_mid = np.array([[-4, 16, -4], [16, -72, 16], [-4, 16, -4]])
    S     = np.stack([S_off, S_mid, S_off], axis=0)   # (3, 3, 3)

    # The stencil satisfies: ∑_{j≠i} S[offset] * (ψ_j - ψ_i) ≈ 12*d²*∇²ψ
    # (from Taylor: ∑ S*di² = 24 per axis, giving h²/2*24 = 12h² coefficient)
    # T = -½∇² requires multiplying by -1/(2 * 12 * d²) = -1/(24d²)
    kin_pref = -1.0 / (24.0 * d_sparse**2)

    edge_index: list = []
    edge_attr:  list = []

    for i in range(N_sparse):
        for j in range(N_sparse):
            for k in range(N_sparse):
                u     = _node_idx(i, j, k)
                pos_u = np.array([X_s[i, j, k], Y_s[i, j, k], Z_s[i, j, k]])

                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue

                            ni, nj, nk = i + di, j + dj, k + dk
                            v = _node_idx(ni, nj, nk)
                            pos_v = np.array([
                                X_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                                Y_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                                Z_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                            ])

                            dr   = _pbc_delta(pos_v, pos_u)
                            dist = float(np.linalg.norm(dr))
                            w    = float(S[di + 1, dj + 1, dk + 1]) * kin_pref

                            edge_index.append([u, v])
                            edge_attr.append([float(dr[0]), float(dr[1]), float(dr[2]), dist, w])

    return (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_attr,  dtype=torch.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 高阶十字星 + correction 立方体（两套边）
# ─────────────────────────────────────────────────────────────────────────────

def build_star_graph(fd_order: int = 4, n_co: int = 3):
    """
    构建两套边：

    1. FD 边（高阶十字星，3 轴）
       每轴偏移量 ±1,...,±(fd_order//2)，共 fd_order 个邻居，3 轴合计
       fd_order×3 条有向边。
       edge_attr: [dx, dy, dz, dist, w]
       w = -c_k / (2 h²)，使  Σ_j w*(ψ_j - ψ_i) ≈ T_FD|ψ⟩_i。

    2. Co 边（correction 立方体，n_co³-1 邻居）
       偏移量 (di,dj,dk) ∈ {−r,...,r}³ \ {(0,0,0)}，r = n_co//2。
       edge_attr: [dx, dy, dz, dist]   （无 w；供 NN correction 使用）

    两套边的轴向近邻（如 (±1,0,0)）会各自独立出现，分别承担
    FD 动能基准 和 NN correction 两种角色，互不干扰。

    Returns
    -------
    fd_edge_index : LongTensor  [2, E_fd]
    fd_edge_attr  : FloatTensor [E_fd, 5]
    co_edge_index : LongTensor  [2, E_co]
    co_edge_attr  : FloatTensor [E_co, 4]
    """
    # ── FD 系数 ──
    m    = fd_order // 2
    c    = _fd2_coefficients(fd_order)          # shape (2m+1,)
    offsets_1d = np.arange(-m, m + 1, dtype=int)

    # off-diagonal weights: w_k = -c_k / (2 h²)
    fd_w = {}
    for ki, k in enumerate(offsets_1d):
        if k != 0:
            fd_w[int(k)] = float(-c[ki] / (2.0 * d_sparse**2))

    # ── Co cube radius ──
    r_co = n_co // 2                            # default 1 for n_co=3

    fd_ei: list = []
    fd_ea: list = []
    co_ei: list = []
    co_ea: list = []

    AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]   # x, y, z unit steps

    for i in range(N_sparse):
        for j in range(N_sparse):
            for k in range(N_sparse):
                u     = _node_idx(i, j, k)
                pos_u = np.array([X_s[i, j, k], Y_s[i, j, k], Z_s[i, j, k]])

                # ── 1. FD 边：沿 3 轴 ──
                for (ex, ey, ez) in AXES:
                    for step, w in fd_w.items():
                        ni, nj, nk = i + step*ex, j + step*ey, k + step*ez
                        v = _node_idx(ni, nj, nk)
                        pos_v = np.array([
                            X_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                            Y_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                            Z_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                        ])
                        dr   = _pbc_delta(pos_v, pos_u)
                        dist = float(np.linalg.norm(dr))
                        fd_ei.append([u, v])
                        fd_ea.append([float(dr[0]), float(dr[1]), float(dr[2]), dist, w])

                # ── 2. Co 边：n_co³-1 立方体 ──
                for di in range(-r_co, r_co + 1):
                    for dj in range(-r_co, r_co + 1):
                        for dk in range(-r_co, r_co + 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            ni, nj, nk = i + di, j + dj, k + dk
                            v = _node_idx(ni, nj, nk)
                            pos_v = np.array([
                                X_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                                Y_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                                Z_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                            ])
                            dr   = _pbc_delta(pos_v, pos_u)
                            dist = float(np.linalg.norm(dr))
                            co_ei.append([u, v])
                            co_ea.append([float(dr[0]), float(dr[1]), float(dr[2]), dist])

    return (
        torch.tensor(fd_ei, dtype=torch.long).t().contiguous(),
        torch.tensor(fd_ea, dtype=torch.float32),
        torch.tensor(co_ei, dtype=torch.long).t().contiguous(),
        torch.tensor(co_ea, dtype=torch.float32),
    )

