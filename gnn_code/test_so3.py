"""
test_so3.py — Sanity checks for SO3HamiltonianNet.

Checks:
  1. Shape correctness   — output has same shape as input u: [N, 1]
  2. Constant-field zero — if u_i = const, then T^(0) = 0 and T^(1) = 0
  3. Backward-compat     — existing models (HamiltonianGNN_Cross) still work

Run directly:
    python -c "from gnn_code.test_so3 import test_so3_sanity; test_so3_sanity()"
"""

import torch


def test_so3_sanity(device: str = 'cpu', radial_hidden_dim: int = 16):
    """
    Run three sanity checks on SO3HamiltonianNet.

    Parameters
    ----------
    device           : torch device ('cpu' or 'cuda')
    radial_hidden_dim: width of the radial MLP to test
    """
    from .graph import build_star_graph
    from .model import SO3HamiltonianNet, HamiltonianGNN_Cross
    from .physics import N_sparse, V_sparse

    dev = torch.device(device)
    print(f"\n{'='*50}")
    print(f"  SO3HamiltonianNet sanity checks  (device={device})")
    print(f"{'='*50}")

    # Build graph once
    fd_ei, fd_ea, co_ei, co_ea = build_star_graph(fd_order=4, n_co=3)
    fd_ei = fd_ei.to(dev); fd_ea = fd_ea.to(dev)
    co_ei = co_ei.to(dev); co_ea = co_ea.to(dev)

    N    = N_sparse ** 3
    V    = torch.tensor(V_sparse.flatten(), dtype=torch.float32, device=dev).unsqueeze(-1)
    model = SO3HamiltonianNet(radial_hidden_dim=radial_hidden_dim).to(dev)
    model.eval()

    # ── Check 1: shape ────────────────────────────────────────────────────────
    with torch.no_grad():
        u_rand = torch.randn(N, 1, device=dev)
        out    = model(u_rand, fd_ei, fd_ea, co_ei, co_ea, V)
    assert out.shape == (N, 1), f"Shape mismatch: expected ({N}, 1), got {out.shape}"
    print(f"  [PASS] Shape: input {u_rand.shape} → output {out.shape}")

    # ── Check 2: constant-field zero response ─────────────────────────────────
    # If u_i = c for all i, then all (u_j - u_i) = 0,
    # so T^(0) = 0 and T^(1) = 0 exactly.  Only FD + V*c survive.
    c     = 3.7   # arbitrary nonzero constant
    u_const = torch.full((N, 1), c, dtype=torch.float32, device=dev)

    with torch.no_grad():
        out_const = model(u_const, fd_ei, fd_ea, co_ei, co_ea, V)

    # FD baseline for constant field: Σ_j w*(c-c) = 0, so fd_lap = 0.
    # Therefore out_const = 0 + 0 + V * c.
    expected = V * c
    max_err  = (out_const - expected).abs().max().item()
    tol      = 1e-5
    assert max_err < tol, (
        f"Constant-field residual too large: max|T_corr| = {max_err:.2e} > {tol:.0e}")
    print(f"  [PASS] Constant-field zero: max|T_corr| = {max_err:.2e} < {tol:.0e}")

    # ── Check 3: backward compatibility (HamiltonianGNN_Cross unchanged) ──────
    model_cross = HamiltonianGNN_Cross(hidden_dim=32).to(dev)
    model_cross.eval()
    with torch.no_grad():
        out_cross = model_cross(u_rand, fd_ei, fd_ea, co_ei, co_ea, V)
    assert out_cross.shape == (N, 1), \
        f"HamiltonianGNN_Cross shape wrong: {out_cross.shape}"
    print(f"  [PASS] HamiltonianGNN_Cross still works: output shape {out_cross.shape}")

    print(f"\n  All checks passed.\n")
    return True
