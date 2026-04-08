"""
model.py — 神经网络模型定义

HamiltonianGNN
    一次消息传递 ≈ 作用一次哈密顿量 H = T_GNN + V。
    每条边的消息 = 基础差分项（差分先验）+ MLP 修正项。
    MLP 输入: [ψ_i, ψ_j, dx, dy, dz, r]  →  修正量（标量）。

FiniteDiffHamiltonian
    无参数的纯差分基准。
    消息 = w_ij * (ψ_j - ψ_i)，聚合后加 V·ψ。
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
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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
    edge_attr  : torch.FloatTensor [E, 5]
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
        w_ij = self.edge_attr[:, 4:5].to(self.device)

        messages = w_ij * (u[src] - u[dst])
        lap      = torch.zeros_like(u)
        lap.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        return lap + self.V * u
