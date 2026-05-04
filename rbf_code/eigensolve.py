"""rbf_code.eigensolve — eigensolvers, parameter sweeps, and problem assembler.

Exposes:
    build_problem            — simple HO box problem (Poisson-disc nodes)
    solve_lowest_eigenvalues — n lowest eigenvalues of H (CPU or CUDA)
    sweep_rbf_kernels        — kernel sweep with spectral accuracy metrics
    sweep_stencil_eps        — (stencil_size, eps) 2-D sweep
    iterate_hamiltonian      — repeated H application diagnostic
    _ho_exact_levels         — first n exact 3-D HO eigenvalues
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla
from rbf.pde.fd import weight_matrix

from rbf_code.config import (
    Array, RBFConfig, RBFProblem, IterationRecord, KERNELS_ALL,
)
from rbf_code.nodes import generate_nodes, make_grid_points
from rbf_code.laplacian import build_hamiltonian_matrix


def build_problem(
    config: Optional[RBFConfig] = None,
    build_interpolation: bool = True,
) -> RBFProblem:
    """Assemble a simple harmonic-oscillator box problem (Poisson-disc nodes)."""
    config = RBFConfig() if config is None else config
    nodes, groups = generate_nodes(spacing=config.spacing, L=config.L)
    interior_idx = groups["interior"]

    laplacian_matrix = weight_matrix(
        x=nodes[interior_idx],
        p=nodes,
        n=config.stencil_size,
        diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        phi=config.phi,
        eps=config.eps,
        order=config.order,
    )

    psi_interp_matrix = None
    lap_interp_matrix = None
    grid_points       = None
    grid_shape        = None

    if build_interpolation:
        grid_points, grid_shape = make_grid_points(config.L, config.grid_N)
        psi_interp_matrix = weight_matrix(
            x=grid_points,
            p=nodes[interior_idx],
            n=config.stencil_size,
            diffs=[0, 0, 0],
            phi=config.phi,
            eps=config.eps,
            order=config.order,
        )
        lap_interp_matrix = psi_interp_matrix

    return RBFProblem(
        config=config,
        nodes=nodes,
        groups=groups,
        interior_idx=interior_idx,
        laplacian_matrix=laplacian_matrix,
        psi_interp_matrix=psi_interp_matrix,
        lap_interp_matrix=lap_interp_matrix,
        grid_points=grid_points,
        grid_shape=grid_shape,
    )


def solve_lowest_eigenvalues(
    problem: RBFProblem,
    n_eigs: int = 6,
    device: str = "cpu",
    symmetrize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (eigenvalues, eigenvectors) for the n_eigs lowest eigenvalues of H.

    symmetrize=False (default) — non-symmetric solver, complex output
        CPU  : scipy ARPACK eigs at sigma=0
        CUDA : torch.linalg.eig on dense tensor

    symmetrize=True — symmetric solver, real output
        CPU  : scipy ARPACK eigsh
        CUDA : torch.linalg.eigh on dense tensor
    """
    H = build_hamiltonian_matrix(problem, symmetrize=symmetrize)

    if device == "cpu":
        if symmetrize:
            vals, vecs = spla.eigsh(H, k=n_eigs, which="SM")
            order = np.argsort(vals)
            return vals[order].astype(np.float64), vecs[:, order]
        else:
            vals, vecs = spla.eigs(H, k=n_eigs, sigma=0.0, which="LM")
            order = np.argsort(vals.real)
            return vals[order].astype(np.complex128), vecs[:, order].astype(np.complex128)

    elif device == "cuda":
        import torch
        H_dense = torch.tensor(H.toarray(), dtype=torch.float64, device="cuda")
        if symmetrize:
            vals_t, vecs_t = torch.linalg.eigh(H_dense)
            vals = vals_t[:n_eigs].cpu().numpy().astype(np.float64)
            vecs = vecs_t[:, :n_eigs].cpu().numpy()
        else:
            vals_t, vecs_t = torch.linalg.eig(H_dense)
            order = torch.argsort(vals_t.real)[:n_eigs]
            vals = vals_t[order].cpu().numpy().astype(np.complex128)
            vecs = vecs_t[:, order].cpu().numpy().astype(np.complex128)
        return vals, vecs

    else:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")


def _ho_exact_levels(n: int) -> np.ndarray:
    """First n exact 3D HO eigenvalues in ascending order (including degeneracy)."""
    levels: list[float] = []
    for s in range(100):
        e = s + 1.5
        deg = (s + 1) * (s + 2) // 2
        levels.extend([e] * deg)
        if len(levels) >= n:
            break
    return np.array(levels[:n], dtype=float)


def sweep_rbf_kernels(
    nodes: Array,
    groups: Dict[str, Array],
    stencil_size: int = 16,
    eps: float = 0.7,
    order: int = 0,
    n_eigs: int = 10,
    device: str = "cpu",
    symmetrize: bool = False,
    kernels: Optional[list] = None,
) -> list:
    """
    For each RBF kernel in `kernels`, rebuild the Laplacian (nodes fixed),
    assemble H, solve n_eigs lowest eigenvalues, and compute mean relative
    spectral error against exact 3D HO levels.

    Returns list of result dicts sorted ascending by mean_rel_err.
    """
    if kernels is None:
        kernels = KERNELS_ALL

    interior_idx = groups["interior"]
    exact   = _ho_exact_levels(n_eigs)
    results = []

    for phi in kernels:
        print(f"  {phi:<10}", end="  ", flush=True)
        try:
            lap = weight_matrix(
                x=nodes[interior_idx], p=nodes, n=stencil_size,
                diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                phi=phi, eps=eps, order=order,
            )
            prob = RBFProblem(
                config=RBFConfig(stencil_size=stencil_size, phi=phi,
                                 eps=eps, order=order),
                nodes=nodes, groups=groups, interior_idx=interior_idx,
                laplacian_matrix=lap,
                psi_interp_matrix=None, lap_interp_matrix=None,
                grid_points=None, grid_shape=None,
            )
            vals, _ = solve_lowest_eigenvalues(
                prob, n_eigs=n_eigs, device=device, symmetrize=symmetrize)
            rel_errs     = np.abs(np.abs(vals) - exact) / np.abs(exact)
            mean_rel_err = float(np.mean(rel_errs))
            max_imag     = (float(np.max(np.abs(np.imag(vals))))
                            if np.iscomplexobj(vals) else 0.0)
            print(f"mean_rel_err={mean_rel_err:.4e}  max|Im|={max_imag:.2e}")
            results.append({
                "phi":          phi,
                "status":       "ok",
                "eigenvalues_re": [float(v.real) for v in vals],
                "eigenvalues_im": ([float(v.imag) for v in vals]
                                   if np.iscomplexobj(vals) else [0.0] * len(vals)),
                "exact":        [float(v) for v in exact],
                "rel_errs":     [float(v) for v in rel_errs],
                "mean_rel_err": mean_rel_err,
                "max_imag":     max_imag,
            })
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({"phi": phi, "status": "failed", "error": str(exc)})

    return sorted(results, key=lambda r: r.get("mean_rel_err", float("inf")))


def sweep_stencil_eps(
    nodes: Array,
    groups: Dict[str, Array],
    phi: str = "ga",
    stencil_sizes: Optional[list] = None,
    eps_values: Optional[list] = None,
    order: int = 0,
    n_eigs: int = 10,
    device: str = "cpu",
    symmetrize: bool = False,
) -> list:
    """
    2D sweep over (stencil_size, eps) for a fixed RBF kernel.
    Returns flat list of result dicts sorted ascending by mean_rel_err.
    """
    if stencil_sizes is None:
        stencil_sizes = [8, 16, 32, 64, 128, 256]
    if eps_values is None:
        eps_values = [round(0.1 * i, 10) for i in range(1, 11)]

    interior_idx = groups["interior"]
    exact   = _ho_exact_levels(n_eigs)
    results = []
    n_total = len(stencil_sizes) * len(eps_values)
    done    = 0

    for n_st in stencil_sizes:
        for eps in eps_values:
            done += 1
            print(f"  [{done:3d}/{n_total}]  stencil={n_st:4d}  eps={eps:.2f}",
                  end="  ", flush=True)
            try:
                lap = weight_matrix(
                    x=nodes[interior_idx], p=nodes, n=n_st,
                    diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                    phi=phi, eps=eps, order=order,
                )
                prob = RBFProblem(
                    config=RBFConfig(stencil_size=n_st, phi=phi,
                                     eps=eps, order=order),
                    nodes=nodes, groups=groups, interior_idx=interior_idx,
                    laplacian_matrix=lap,
                    psi_interp_matrix=None, lap_interp_matrix=None,
                    grid_points=None, grid_shape=None,
                )
                vals, _ = solve_lowest_eigenvalues(
                    prob, n_eigs=n_eigs, device=device, symmetrize=symmetrize)
                rel_errs     = np.abs(np.abs(vals) - exact) / np.abs(exact)
                mean_rel_err = float(np.mean(rel_errs))
                max_imag     = (float(np.max(np.abs(np.imag(vals))))
                                if np.iscomplexobj(vals) else 0.0)
                print(f"mean_rel_err={mean_rel_err:.4e}  max|Im|={max_imag:.2e}")
                results.append({
                    "stencil_size":   n_st,
                    "eps":            float(eps),
                    "status":         "ok",
                    "mean_rel_err":   mean_rel_err,
                    "max_imag":       max_imag,
                    "eigenvalues_re": [float(v.real) for v in vals],
                    "eigenvalues_im": ([float(v.imag) for v in vals]
                                       if np.iscomplexobj(vals)
                                       else [0.0] * len(vals)),
                })
            except Exception as exc:
                print(f"FAILED: {exc}")
                results.append({
                    "stencil_size": n_st, "eps": float(eps),
                    "status": "failed", "error": str(exc),
                    "mean_rel_err": float("inf"),
                })

    return sorted(results, key=lambda r: r.get("mean_rel_err", float("inf")))


def iterate_hamiltonian(
    problem: RBFProblem,
    n_max: int = 20,
    normalize_each_step: bool = True,
):
    psi_nodes = problem.ground_state()
    records   = []
    cumulative_scale = 1.0

    for n in range(1, n_max + 1):
        Hpsi, Tpsi, Vpsi = problem.apply_hamiltonian(psi_nodes)

        psi_interior = psi_nodes[problem.interior_idx]
        H_interior   = Hpsi[problem.interior_idx]
        T_interior   = Tpsi[problem.interior_idx]
        V_interior   = Vpsi[problem.interior_idx]

        den  = float(np.dot(psi_interior, psi_interior))
        E_H  = float(np.dot(psi_interior, H_interior) * cumulative_scale / den)
        E_T  = float(np.dot(psi_interior, T_interior) / den)
        E_V  = float(np.dot(psi_interior, V_interior) / den)
        exact    = float(1.5**n)
        rel_err  = float(abs(E_H - exact) / abs(exact))

        records.append(
            IterationRecord(
                n=n, E_H=E_H, E_T=E_T, E_V=E_V,
                exact=exact, rel_err=rel_err,
                scale_factor=cumulative_scale,
            )
        )

        psi_nodes = Hpsi.copy()
        if normalize_each_step:
            max_val = float(np.max(np.abs(psi_nodes)))
            if max_val > 0.0:
                cumulative_scale *= max_val
                psi_nodes /= max_val

    return records
