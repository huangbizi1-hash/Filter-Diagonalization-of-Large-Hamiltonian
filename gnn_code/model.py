"""
model.py — 神经网络模型定义

HamiltonianGNN（配合 build_graph，3×3×3 Mehrstellen 模板）
    一次消息传递 ≈ 作用一次哈密顿量 H = T_GNN + V。
    每条边的消息 = 基础差分项（差分先验）+ MLP 修正项。
    MLP 输入: [ψ_i, ψ_j, dx, dy, dz, r]  →  修正量（标量）。

HamiltonianGNN_Cross（配合 build_star_graph，高阶十字星 + correction 立方体）
    两套边，干净分离：
      FD 边 → 固定线性动能 T_FD(ψ)（无 NN 参数）
      Co 边 → 纯 MLP correction（无固定权重）
    H|ψ⟩ ≈ T_FD(ψ) + MLP_correction(ψ) + V·ψ

SO3HamiltonianNet（配合 build_star_graph，SO(3) 启发的 l=0 + l=1 修正）
    在 HamiltonianGNN_Cross 的 FD 基准上加入两个 SO(3) 修正通道：
      T^(0)_i = Σ_j f0(r_ij) · (u_j − u_i)                     [l=0 各向同性]
      h_j^(1) = Σ_k f1(r_jk) · (u_k − u_j) · r̂_jk             [l=1 矢量中间量]
      T^(1)_i = Σ_j g1(r_ij) · (h_j^(1) · r̂_ij)               [l=1 标量投影]
    线性于 u，常数场零响应，无需 e3nn。

FiniteDiffHamiltonian / FiniteDiffHamiltonian_Cross
    对应两种图的无参数纯差分基准。

Edge convention (both graph types)
-----------------------------------
  edge_index[0] = source node  (j, "sender")
  edge_index[1] = target node  (i, "receiver", accumulation point)
  edge_attr[:, 0:3] = dr = r_i − r_j  (displacement source → target)
  edge_attr[:, 3]   = |dr| = distance
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class HamiltonianGNN(MessagePassing):
    """
    图神经网络哈密顿量算符（含 MLP 差分修正）。

    Parameters
    ----------
    hidden_dim : int
        MLP 隐藏层维度（默认 64）。
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__(aggr='add')
        # 输入维度: ψ_i(1) + ψ_j(1) + dx,dy,dz(3) + r(1) = 6
        self.phi_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                u:          torch.Tensor,   # [N, 1]  波函数值
                edge_index: torch.Tensor,   # [2, E]
                edge_attr:  torch.Tensor,   # [E, 5]
                V:          torch.Tensor,   # [N, 1]  势能值
                ) -> torch.Tensor:          # [N, 1]  H|ψ⟩
        laplacian_action = self.propagate(edge_index, x=u, edge_attr=edge_attr)
        return laplacian_action + V * u

    def message(self,
                x_i:       torch.Tensor,   # [E, 1]
                x_j:       torch.Tensor,   # [E, 1]
                edge_attr: torch.Tensor,   # [E, 5]
                ) -> torch.Tensor:         # [E, 1]
        w_ij = edge_attr[:, 4:5]           # 差分系数权重

        base_laplacian = w_ij * (x_j - x_i)

        mlp_input  = torch.cat([
            x_i, x_j,
            edge_attr[:, 0:1],   # dx
            edge_attr[:, 1:2],   # dy
            edge_attr[:, 2:3],   # dz
            edge_attr[:, 3:4],   # r
        ], dim=-1)
        correction = self.phi_mlp(mlp_input)

        return base_laplacian + correction


class FiniteDiffHamiltonian:
    """
    纯差分哈密顿量（无神经网络，用于基准测试）。

    Parameters
    ----------
    edge_index : torch.LongTensor  [2, E]
    edge_attr  : torch.FloatTensor [E, 5]  — attr[:,4] 为 w
    V          : torch.FloatTensor [N, 1]
    device     : torch.device
    """

    def __init__(self,
                 edge_index: torch.Tensor,
                 edge_attr:  torch.Tensor,
                 V:          torch.Tensor,
                 device:     torch.device):
        self.edge_index = edge_index
        self.edge_attr  = edge_attr
        self.V          = V
        self.device     = device

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """u: [N, 1] → H_FD|ψ⟩: [N, 1]"""
        src  = self.edge_index[0]
        dst  = self.edge_index[1]
        w    = self.edge_attr[:, 4:5].to(self.device)
        msg  = w * (u[src] - u[dst])
        lap  = torch.zeros_like(u)
        lap.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return lap + self.V * u


# ─────────────────────────────────────────────────────────────────────────────
# 高阶十字星 + correction 立方体（两套边）
# ─────────────────────────────────────────────────────────────────────────────

class HamiltonianGNN_Cross(MessagePassing):
    """
    GNN 哈密顿量算符，配合 build_star_graph() 使用。

    两套边，干净分离：
      FD 边  → 固定线性动能 T_FD(ψ) = Σ_j w*(ψ_j−ψ_i)，无 NN
      Co 边  → 纯 MLP correction，无固定权重

    H|ψ⟩ ≈ T_FD(ψ) + Σ_{co 边} MLP(ψ_i, ψ_j, dx, dy, dz, r) + V·ψ

    Parameters
    ----------
    hidden_dim : MLP 隐藏层维度（默认 64）
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__(aggr='add')
        # correction MLP: [ψ_i, ψ_j, dx, dy, dz, r] → scalar
        self.phi_mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self,
                u:              torch.Tensor,   # [N, 1]
                fd_edge_index:  torch.Tensor,   # [2, E_fd]
                fd_edge_attr:   torch.Tensor,   # [E_fd, 5]
                co_edge_index:  torch.Tensor,   # [2, E_co]
                co_edge_attr:   torch.Tensor,   # [E_co, 4]
                V:              torch.Tensor,   # [N, 1]
                ) -> torch.Tensor:
        # ── FD kinetic baseline (no NN) ──
        src, dst = fd_edge_index
        w   = fd_edge_attr[:, 4:5]
        fd_msg = w * (u[src] - u[dst])
        fd_lap = torch.zeros_like(u)
        fd_lap.scatter_add_(0, dst.unsqueeze(-1).expand_as(fd_msg), fd_msg)

        # ── NN correction (co edges only) ──
        co_corr = self.propagate(co_edge_index, x=u, edge_attr=co_edge_attr)

        return fd_lap + co_corr + V * u

    def message(self,
                x_i:       torch.Tensor,   # [E, 1]
                x_j:       torch.Tensor,   # [E, 1]
                edge_attr: torch.Tensor,   # [E, 4]  — co edges only
                ) -> torch.Tensor:
        return self.phi_mlp(torch.cat([
            x_i, x_j,
            edge_attr[:, 0:1],   # dx
            edge_attr[:, 1:2],   # dy
            edge_attr[:, 2:3],   # dz
            edge_attr[:, 3:4],   # r
        ], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# SO(3)-motivated Hamiltonian (l=0 + l=1 correction channels)
# ─────────────────────────────────────────────────────────────────────────────

class RadialMLP(nn.Module):
    """
    Small MLP mapping scalar edge distance r → radial basis values.

    Input:  r  [E, 1]      — internode distances (Bohr)
    Output:    [E, out_dim] — radial function values

    Used by SO3HamiltonianNet to produce f0(r), f1(r), g1(r) from a single
    shared trunk to avoid redundant computation.
    """

    def __init__(self, out_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class SO3HamiltonianNet(nn.Module):
    """
    SO(3)-motivated sparse Hamiltonian with l=0 and l=1 correction channels.

    Works with build_star_graph() output (FD edges + correction edges).
    Does NOT require e3nn — the SO(3) structure is implemented directly via
    radial functions and unit-vector dot products.

    H|ψ⟩ = T_FD|ψ⟩ + T^(0) + T^(1) + V·ψ

    Corrections (both linear in ψ, zero for constant ψ):

      l=0 scalar:
          T^(0)_i = Σ_j f0(r_ij) · (u_j − u_i)

      l=1 vector intermediate at each node j:
          h_j^(1) = Σ_k f1(r_jk) · (u_k − u_j) · r̂_jk

      l=1 scalar contraction at node i:
          T^(1)_i = Σ_j g1(r_ij) · ( h_j^(1) · r̂_ij )

    where r̂_ij = (r_j − r_i) / |r_j − r_i| (unit vector from i to j).

    Properties
    ----------
    Linearity in ψ:   f0, f1, g1 depend only on distance r (geometry).
    Constant-field:   u_i = const  →  all (u_j − u_i) = 0  →  T^(0) = T^(1) = 0.
    Rotation structure:
      · l=0 part uses only |r_ij| (isotropic).
      · l=1 part uses r̂ unit vectors through dot products only.

    Parameters
    ----------
    radial_hidden_dim : hidden width of the shared radial MLP
    """

    def __init__(self, radial_hidden_dim: int = 32):
        super().__init__()
        # One shared trunk r → (f0, f1, g1); outputs split below.
        self.radial = RadialMLP(out_dim=3, hidden_dim=radial_hidden_dim)

    def forward(self,
                u:             torch.Tensor,   # [N, 1]  wavefunction values
                fd_edge_index: torch.Tensor,   # [2, E_fd]
                fd_edge_attr:  torch.Tensor,   # [E_fd, 5]  col 4 = FD weight w
                co_edge_index: torch.Tensor,   # [2, E_co]
                co_edge_attr:  torch.Tensor,   # [E_co, 4]  [dx, dy, dz, dist]
                V:             torch.Tensor,   # [N, 1]  potential
                ) -> torch.Tensor:             # [N, 1]  H|ψ⟩
        N = u.shape[0]

        # ── FD kinetic baseline (no NN, same as HamiltonianGNN_Cross) ────────
        src, dst = fd_edge_index
        fd_msg   = fd_edge_attr[:, 4:5] * (u[src] - u[dst])
        fd_lap   = torch.zeros_like(u)
        fd_lap.scatter_add_(0, dst.unsqueeze(-1).expand_as(fd_msg), fd_msg)

        # ── Co-edge geometry ─────────────────────────────────────────────────
        # row = source node  (j in spec notation when col = i)
        # col = target node  (i — accumulation point)
        # dr  = r_col − r_row  (displacement source → target)
        #
        # Unit-vector convention:
        #   rhat = −dr / r  =  (r_row − r_col) / r
        #        → points FROM target TO source
        #        → equals r̂_ij (i→j) when col=i, row=j  (spec notation)
        #        → equals r̂_jk (j→k) when col=j, row=k  (for h^(1) pass)
        row = co_edge_index[0]           # source  j (or k in h^(1) pass)
        col = co_edge_index[1]           # target  i (or j in h^(1) pass)

        dr   = co_edge_attr[:, 0:3]      # r_col − r_row  [E, 3]
        r    = co_edge_attr[:, 3:4]      # |dr|           [E, 1]
        rhat = -dr / (r + 1e-8)          # r̂_ij = (r_row−r_col)/r  [E, 3]

        # Shared radial functions: r → (f0, f1, g1)
        rad   = self.radial(r)           # [E, 3]
        f0    = rad[:, 0:1]              # [E, 1]
        f1    = rad[:, 1:2]              # [E, 1]
        g1    = rad[:, 2:3]              # [E, 1]

        # Signed wavefunction difference along each edge
        delta_u = u[row] - u[col]        # u_j − u_i  [E, 1]

        # ── T^(0): l=0 scalar correction ─────────────────────────────────────
        # T^(0)_i = Σ_j f0(r_ij) · (u_j − u_i)
        # Aggregate f0·Δu at target col = i:
        T0_msg = f0 * delta_u            # [E, 1]
        T0     = torch.zeros_like(u)
        T0.scatter_add_(0, col.unsqueeze(-1).expand_as(T0_msg), T0_msg)

        # ── h^(1): l=1 vector intermediate (two-pass) ────────────────────────
        # h_j^(1) = Σ_k f1(r_jk) · (u_k − u_j) · r̂_jk
        # Pass uses the same edges with col = j (target = j):
        #   rhat here acts as r̂_jk  (points j→k, consistent with rhat formula)
        #   delta_u = u_k − u_j = u[row] − u[col]  ✓
        h1_msg = f1 * delta_u * rhat     # [E, 3]
        h1     = torch.zeros(N, 3, device=u.device, dtype=u.dtype)
        h1.scatter_add_(0, col.unsqueeze(-1).expand(-1, 3), h1_msg)

        # ── T^(1): l=1 scalar contraction ────────────────────────────────────
        # T^(1)_i = Σ_j g1(r_ij) · ( h_j^(1) · r̂_ij )
        # h^(1) at source j = h1[row];  r̂_ij = rhat (same formula)
        # Aggregate at target col = i:
        dot_ij = (h1[row] * rhat).sum(dim=-1, keepdim=True)   # [E, 1]
        T1_msg = g1 * dot_ij             # [E, 1]
        T1     = torch.zeros_like(u)
        T1.scatter_add_(0, col.unsqueeze(-1).expand_as(T1_msg), T1_msg)

        return fd_lap + T0 + T1 + V * u


class FiniteDiffHamiltonian_Cross:
    """
    纯差分基准（高阶十字星，无 NN），配合 build_star_graph() 使用。
    只用 FD 边（co 边忽略）。

    Parameters
    ----------
    fd_edge_index : LongTensor  [2, E_fd]
    fd_edge_attr  : FloatTensor [E_fd, 5]  — attr[:,4] 为 w
    V             : FloatTensor [N, 1]
    device        : torch.device
    """

    def __init__(self,
                 fd_edge_index: torch.Tensor,
                 fd_edge_attr:  torch.Tensor,
                 V:             torch.Tensor,
                 device:        torch.device):
        self.fd_edge_index = fd_edge_index
        self.fd_edge_attr  = fd_edge_attr
        self.V             = V
        self.device        = device

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """u: [N, 1] → H_FD|ψ⟩: [N, 1]"""
        src, dst = self.fd_edge_index
        w   = self.fd_edge_attr[:, 4:5].to(self.device)
        msg = w * (u[src] - u[dst])
        lap = torch.zeros_like(u)
        lap.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        return lap + self.V * u

