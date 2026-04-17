"""
gnn_operator.py — Wrap trained GNN (or pure FD) as a scipy LinearOperator.

Filter diagonalization passes un-normalized vectors to H.  The GNN requires
a normalized input, so we implement the scale-preserving identity:

    H_GNN|ψ⟩  =  ‖ψ‖ · GNN( ψ / ‖ψ‖ )

When use_fd=True the operator falls back to the pure FiniteDiffHamiltonian
(no NN, same graph) so the caller can compare filter results directly.
"""

import json
import os

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator

from .physics import N_sparse, V_sparse
from .graph   import build_graph, build_star_graph
from .model   import (HamiltonianGNN,       FiniteDiffHamiltonian,
                      HamiltonianGNN_Cross, FiniteDiffHamiltonian_Cross)


def build_gnn_operator(run_dir: str,
                       use_fd:  bool = False,
                       device:  str  = 'cpu') -> LinearOperator:
    """
    Return a scipy LinearOperator that applies H on the sparse grid.

    Parameters
    ----------
    run_dir : GNN run directory (must contain config.json + epoch_*.pt)
    use_fd  : True  → pure FD Hamiltonian, no NN (for baseline comparison)
              False → GNN Hamiltonian with normalisation trick (default)
    device  : torch device string ('cpu' or 'cuda')

    Returns
    -------
    LinearOperator of shape (N_sparse³, N_sparse³), dtype float64.
    """
    dev    = torch.device(device)
    n_grid = N_sparse ** 3

    # ── load config ──────────────────────────────────────────────────────────
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    hidden_dim = config.get('hidden_dim', 64)
    graph_type = config.get('graph_type', 'cube')
    fd_order   = config.get('fd_order',   4)
    n_co       = config.get('n_co',       3)

    # ── build graph ──────────────────────────────────────────────────────────
    V_t = torch.tensor(V_sparse.flatten(),
                       dtype=torch.float32).unsqueeze(-1).to(dev)

    if graph_type == 'cross':
        fd_ei, fd_ea, co_ei, co_ea = build_star_graph(fd_order, n_co)
        fd_ei = fd_ei.to(dev);  fd_ea = fd_ea.to(dev)
        co_ei = co_ei.to(dev);  co_ea = co_ea.to(dev)
        edge_index = edge_attr = None
    else:
        edge_index, edge_attr = build_graph()
        edge_index = edge_index.to(dev);  edge_attr = edge_attr.to(dev)
        fd_ei = fd_ea = co_ei = co_ea = None

    # ── choose apply function ─────────────────────────────────────────────────
    if use_fd:
        if graph_type == 'cross':
            _fd = FiniteDiffHamiltonian_Cross(fd_ei, fd_ea, V_t, dev)
        else:
            _fd = FiniteDiffHamiltonian(edge_index, edge_attr, V_t, dev)

        def _apply(u_t: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return _fd(u_t)

        label = f"FD-{graph_type}"

    else:
        # load latest checkpoint
        ckpts = sorted(
            [fn for fn in os.listdir(run_dir)
             if fn.startswith('epoch_') and fn.endswith('.pt')],
            key=lambda fn: int(fn[len('epoch_'):-len('.pt')]),
        )
        if not ckpts:
            raise RuntimeError(f"No checkpoints found in {run_dir}")
        ckpt_path = os.path.join(run_dir, ckpts[-1])
        ckpt = torch.load(ckpt_path, map_location=dev)

        if graph_type == 'cross':
            model = HamiltonianGNN_Cross(hidden_dim=hidden_dim).to(dev)
        else:
            model = HamiltonianGNN(hidden_dim=hidden_dim).to(dev)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"  GNN operator: loaded {ckpts[-1]}")

        if graph_type == 'cross':
            def _gnn(u_t):
                return model(u_t, fd_ei, fd_ea, co_ei, co_ea, V_t)
        else:
            def _gnn(u_t):
                return model(u_t, edge_index, edge_attr, V_t)

        def _apply(u_t: torch.Tensor) -> torch.Tensor:
            nrm = torch.norm(u_t)
            if nrm < 1e-30:
                return torch.zeros_like(u_t)
            with torch.no_grad():
                return _gnn(u_t / nrm) * nrm   # preserve scale

        label = f"GNN-{graph_type}"

    # ── wrap as scipy LinearOperator ─────────────────────────────────────────
    def _matvec(psi_flat: np.ndarray) -> np.ndarray:
        u_t = torch.tensor(np.asarray(psi_flat, dtype=np.float32),
                           dtype=torch.float32).unsqueeze(-1).to(dev)
        out = _apply(u_t)
        return out.cpu().numpy().flatten().astype(np.float64)

    op = LinearOperator(shape=(n_grid, n_grid), matvec=_matvec, dtype=np.float64)
    op.label = label   # attach label for printing
    return op
