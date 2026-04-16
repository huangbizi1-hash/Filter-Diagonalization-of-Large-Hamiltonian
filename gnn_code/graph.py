"""
graph.py — 稀疏网格图结构构建

在周期边界条件（PBC）下，用 3×3×3 模板构建图：
  - 节点：稀疏网格上的每个格点
  - 边：26 个近邻（排除自身）
  - edge_attr: [dx, dy, dz, dist, w_ij]
    其中 w_ij 来自 4 阶各向同性差分系数（Mehrstellen 格式），
    可用于初始化 "差分先验"。

返回值
------
edge_index : torch.LongTensor  [2, E]
edge_attr  : torch.FloatTensor [E, 5]
"""

import numpy as np
import torch
from .physics import X_s, Y_s, Z_s, d_sparse, N_sparse, L


def build_graph():
    """
    构建稀疏网格图（PBC，3×3×3 模板）。

    差分权重系数 S[di,dj,dk]（各向同性 4 阶拉普拉斯）：
        角点  : +3  面中心: +16  棱中心: -4
        中心  : -72（不含自身，纳入消息传递时处理）
    prefactor = -1 / (2 d²)，使最终结果对应 -1/2 ∇² 算子。
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

    def idx(i: int, j: int, k: int) -> int:
        return (i % N_sparse) * N_sparse**2 + (j % N_sparse) * N_sparse + (k % N_sparse)

    for i in range(N_sparse):
        for j in range(N_sparse):
            for k in range(N_sparse):
                u     = idx(i, j, k)
                pos_u = np.array([X_s[i, j, k], Y_s[i, j, k], Z_s[i, j, k]])

                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue   # 自身系数由 FiniteDiffHamiltonian 单独处理

                            ni, nj, nk = i + di, j + dj, k + dk
                            v = idx(ni, nj, nk)

                            pos_v = np.array([
                                X_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                                Y_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                                Z_s[ni % N_sparse, nj % N_sparse, nk % N_sparse],
                            ])

                            # PBC 最短距离向量
                            delta_r  = pos_v - pos_u
                            delta_r -= L * np.round(delta_r / L)
                            dist     = float(np.linalg.norm(delta_r))

                            w = float(S[di + 1, dj + 1, dk + 1]) * kin_pref

                            edge_index.append([u, v])
                            edge_attr.append([
                                float(delta_r[0]), float(delta_r[1]), float(delta_r[2]),
                                dist, w,
                            ])

    return (
        torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        torch.tensor(edge_attr,  dtype=torch.float32),
    )
