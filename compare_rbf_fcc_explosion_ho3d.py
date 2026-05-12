#!/usr/bin/env python3
"""compare_rbf_fcc_explosion_ho3d.py

Chebyshev explosion filter on the 3-D harmonic oscillator using
FCC-lattice nodes and an RBF-FD Hamiltonian.

Node placement
--------------
FCC (face-centred cubic) lattice with conventional cell parameter `a`.
All lattice points inside the closed cube [-box_L, box_L]^3 are kept.
Boundary nodes: those where max(|x|,|y|,|z|) > box_L - margin*a.
Interior nodes: the rest (Dirichlet ψ=0 on boundary).

RBF-FD Laplacian
----------------
Polyharmonic-spline RBF φ(r) = r^phi_order (default phs3 = r³)
augmented with full monomials up to poly_degree (default 2).
For each interior node, the k nearest neighbours (interior + boundary)
form the stencil.  The sparse Laplacian is assembled from the resulting
per-node weight vectors.

Hamiltonian: H = -½ L_rbf + diag(V_ho3d)

Filter
------
Chebyshev explosion T_m(aH+b) via the 3-term recurrence.  Both the
filter step and the subsequent Rayleigh-Ritz step use the same H_rbf.

Sweep
-----
--a_sweep a1,a2,...   list of FCC lattice parameters (controls density)
--a_sweep START:STOP:STEP  range syntax (float step via np.arange)

Usage examples
--------------
  python compare_rbf_fcc_explosion_ho3d.py \\
      --box_L 5.0 \\
      --a_sweep 1.4,1.2,1.0 \\
      --cheb_m 5 --E_lo 1.5 --E_hi 80.0 \\
      --n_random 300 --n_print 30

  python compare_rbf_fcc_explosion_ho3d.py \\
      --box_L 4.0 --a_sweep 1.6:0.9:-0.2 \\
      --cheb_m 8 --E_lo 1.5 --E_hi 60.0 \\
      --n_random 400 --k_neighbors 32 --phi_order 3 --poly_degree 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from itertools import product as iproduct

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from filter_core import svd_rayleigh_ritz_op

# ── exact HO3D energy levels ────────────────────────────────────────────────

def exact_ho3d_levels(n_levels: int, omega: float = 1.0) -> np.ndarray:
    """Return the first n_levels distinct energy levels of 3-D isotropic HO."""
    levels = []
    for shell in range(200):
        E = omega * (shell + 1.5)
        # degeneracy = (shell+1)*(shell+2)//2
        deg = (shell + 1) * (shell + 2) // 2
        levels.extend([E] * deg)
        if len(levels) >= n_levels:
            break
    return np.array(sorted(levels)[:n_levels])


# ── FCC lattice ──────────────────────────────────────────────────────────────

def fcc_nodes_in_box(box_L: float, a: float) -> np.ndarray:
    """Return FCC lattice points inside [-box_L, box_L]^3 (closed)."""
    # FCC basis (fractional coordinates of conventional cubic cell)
    basis_frac = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ])  # shape (4, 3)
    basis = basis_frac * a  # Cartesian

    n_rep = int(np.ceil(box_L / a)) + 1
    rng_idx = np.arange(-n_rep, n_rep + 1)
    ix, iy, iz = np.meshgrid(rng_idx, rng_idx, rng_idx, indexing='ij')
    cell_origins = np.stack(
        [ix.ravel(), iy.ravel(), iz.ravel()], axis=1
    ).astype(float) * a  # (M, 3)

    # All candidate points: broadcast cell_origins over basis
    pts = cell_origins[:, None, :] + basis[None, :, :]  # (M, 4, 3)
    pts = pts.reshape(-1, 3)

    mask = np.all(np.abs(pts) <= box_L + 1e-10, axis=1)
    pts = pts[mask]
    # Remove duplicates (to within numerical tolerance)
    if len(pts) > 0:
        pts = np.unique(np.round(pts, 10), axis=0)
    return pts


def split_boundary_interior(
    nodes: np.ndarray,
    box_L: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split node indices into (interior_idx, boundary_idx).

    Boundary: max(|x|,|y|,|z|) > box_L - margin.
    """
    d_max = np.max(np.abs(nodes), axis=1)
    boundary_mask = d_max > box_L - margin
    all_idx = np.arange(len(nodes))
    interior_idx = all_idx[~boundary_mask]
    boundary_idx = all_idx[boundary_mask]
    return interior_idx, boundary_idx


# ── RBF-FD Laplacian ─────────────────────────────────────────────────────────

def _monomials_3d(degree: int) -> list[tuple[int, int, int]]:
    """All (i, j, k) with i+j+k <= degree."""
    monos = []
    for total in range(degree + 1):
        for i in range(total + 1):
            for j in range(total - i + 1):
                k = total - i - j
                monos.append((i, j, k))
    return monos


def _eval_poly(pts: np.ndarray, exps: list[tuple]) -> np.ndarray:
    """Evaluate monomials at pts.  Returns (n_pts, n_monos)."""
    n_pts = pts.shape[0]
    n_m = len(exps)
    P = np.empty((n_pts, n_m), dtype=np.float64)
    for col, (ei, ej, ek) in enumerate(exps):
        P[:, col] = (
            pts[:, 0] ** ei
            * pts[:, 1] ** ej
            * pts[:, 2] ** ek
        )
    return P


def _lap_poly_at(xi: np.ndarray, exps: list[tuple]) -> np.ndarray:
    """Laplacian of each monomial evaluated at xi (shape (n_monos,))."""
    n_m = len(exps)
    b = np.zeros(n_m, dtype=np.float64)
    for col, (ei, ej, ek) in enumerate(exps):
        val = 0.0
        # ∂²/∂x²
        if ei >= 2:
            val += ei * (ei - 1) * xi[0] ** (ei - 2) * xi[1] ** ej * xi[2] ** ek
        # ∂²/∂y²
        if ej >= 2:
            val += ej * (ej - 1) * xi[0] ** ei * xi[1] ** (ej - 2) * xi[2] ** ek
        # ∂²/∂z²
        if ek >= 2:
            val += ek * (ek - 1) * xi[0] ** ei * xi[1] ** ej * xi[2] ** (ek - 2)
        b[col] = val
    return b


def _phs_lap_at_center(r_vec: np.ndarray, phi_order: int) -> np.ndarray:
    """
    ∇²_x [phi(|x - x_α|)] evaluated at x = x_i,
    where r_vec[α] = |x_i - x_α|.

    For φ(r) = r^p (PHS): ∇²φ = p*(p+1)*r^(p-2)  [3-D formula]
    Special cases:
      p=1: ∇²r = 2/r  (diverges at 0, avoid or use phs ≥ 3)
      p=3: 12r  (well-defined everywhere including r=0)
    """
    r = r_vec
    p = phi_order
    if p == 1:
        b = np.where(r > 1e-14, 2.0 / r, 0.0)
    elif p == 2:
        # ∇²(r²) = 6 in 3D
        b = np.full_like(r, 6.0, dtype=np.float64)
        # At center r=0 this is still 6 (constant derivative of 2)
    else:
        # General: p*(p+1)*r^(p-2)
        # For p=3: 12*r (no singularity at r=0)
        b = p * (p + 1) * np.where(r > 1e-14, r ** (p - 2), 0.0)
    return b


def rbf_fd_laplacian(
    nodes: np.ndarray,
    interior_idx: np.ndarray,
    k_neighbors: int = 40,
    phi_order: int = 3,
    poly_degree: int = 2,
) -> sp.csr_matrix:
    """
    Build sparse Laplacian matrix (interior × all_nodes) using PHS-augmented RBF-FD.

    The matrix L has shape (n_interior, n_nodes).
    For Dirichlet BCs: multiply by a vector that is zero on boundary nodes.
    Equivalently, L_int = L[:, interior_idx] gives the interior–interior part.

    Parameters
    ----------
    nodes : (n_nodes, 3)
    interior_idx : (n_interior,)
    k_neighbors  : stencil size (includes center node itself)
    phi_order    : PHS exponent (3 → r³)
    poly_degree  : polynomial augmentation degree (2 → up to quadratics)

    Returns
    -------
    L_int : sparse (n_interior, n_interior)  — Dirichlet BCs applied
    """
    n_total    = nodes.shape[0]
    n_interior = len(interior_idx)
    monomials  = _monomials_3d(poly_degree)
    M = len(monomials)

    # Map global → local interior index (-1 if boundary)
    global_to_local = np.full(n_total, -1, dtype=np.int64)
    global_to_local[interior_idx] = np.arange(n_interior, dtype=np.int64)

    tree = cKDTree(nodes)

    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    data_list: list[np.ndarray] = []

    for local_i, global_i in enumerate(interior_idx):
        xi = nodes[global_i]  # (3,)

        # Find k nearest neighbors (includes xi itself at distance 0)
        actual_k = min(k_neighbors, n_total)
        _, nbr_global = tree.query(xi, k=actual_k)
        stencil_pts = nodes[nbr_global]  # (k, 3)
        k = len(nbr_global)

        # Relative coordinates for stability
        dx = stencil_pts - xi  # (k, 3)

        # RBF matrix Φ[α,β] = φ(|x_α - x_β|)
        diff = stencil_pts[:, None, :] - stencil_pts[None, :, :]  # (k,k,3)
        r_mat = np.sqrt(np.sum(diff ** 2, axis=2))  # (k,k)
        Phi = r_mat ** phi_order  # (k,k)

        # Polynomial matrix P[α,m] = p_m(x_α) using relative coords for stability
        P = _eval_poly(dx, monomials)  # (k, M)

        # System matrix (k+M, k+M)
        A = np.zeros((k + M, k + M), dtype=np.float64)
        A[:k, :k] = Phi
        A[:k, k:] = P
        A[k:, :k] = P.T
        # A[k:, k:] = 0 (zero block for polynomial constraints)

        # RHS: evaluate operator at xi
        r_xi = np.sqrt(np.sum(dx ** 2, axis=1))  # (k,)
        b_rbf  = _phs_lap_at_center(r_xi, phi_order)
        # Polynomial part: Laplacian of p_m at xi (using relative coords → xi=0)
        b_poly = _lap_poly_at(np.zeros(3, dtype=np.float64), monomials)

        b = np.concatenate([b_rbf, b_poly])

        # Solve for weights
        try:
            w_full = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w_full, *_ = np.linalg.lstsq(A, b, rcond=None)

        weights = w_full[:k]  # (k,) — discard polynomial Lagrange multipliers

        # Accumulate sparse entries; Dirichlet BC: skip boundary contributions
        for j, gj in enumerate(nbr_global):
            loc_j = global_to_local[gj]
            if loc_j >= 0:  # interior node only
                rows_list.append(local_i)
                cols_list.append(loc_j)
                data_list.append(weights[j])

    rows = np.array(rows_list, dtype=np.int64)
    cols = np.array(cols_list, dtype=np.int64)
    data = np.array(data_list, dtype=np.float64)
    L_int = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_interior, n_interior),
        dtype=np.float64,
    )
    return L_int


# ── Hamiltonian ───────────────────────────────────────────────────────────────

def build_H_rbf(
    nodes: np.ndarray,
    interior_idx: np.ndarray,
    k_neighbors: int,
    phi_order: int,
    poly_degree: int,
    omega: float = 1.0,
) -> sp.csr_matrix:
    """H_rbf = -0.5 * L_int + diag(V_ho3d) on interior nodes."""
    L_int = rbf_fd_laplacian(nodes, interior_idx,
                              k_neighbors=k_neighbors,
                              phi_order=phi_order,
                              poly_degree=poly_degree)
    T_int = -0.5 * L_int
    pts_int = nodes[interior_idx]  # (n_interior, 3)
    V_int   = 0.5 * omega ** 2 * np.sum(pts_int ** 2, axis=1)
    H       = T_int + sp.diags(V_int, format='csr')
    return H


# ── Chebyshev explosion filter ─────────────────────────────────────────────────

def chebyshev_explosion(
    H_apply,
    psi: np.ndarray,
    m: int,
    E_lo: float,
    E_hi: float,
) -> np.ndarray:
    """
    Apply T_m(a*H + b) to psi via 3-term recurrence.
    a = 2/(E_hi - E_lo), b = -(E_hi + E_lo)/(E_hi - E_lo)
    """
    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)

    def Hs(phi):
        return a * H_apply(phi) + b * phi

    if m == 0:
        return psi.copy()
    y_prev = psi.copy()
    y_curr = Hs(psi)
    for _ in range(2, m + 1):
        y_next = 2.0 * Hs(y_curr) - y_prev
        y_prev, y_curr = y_curr, y_next
    return y_curr


# ── single run ────────────────────────────────────────────────────────────────

def run_single(
    box_L: float,
    lattice_a: float,
    cheb_m: int,
    E_lo: float,
    E_hi: float,
    n_random: int,
    k_neighbors: int,
    phi_order: int,
    poly_degree: int,
    omega: float,
    boundary_margin_frac: float,
    svd_tol: float,
    seed: int,
    n_print: int,
) -> dict:
    rng = np.random.default_rng(seed)

    # 1. FCC nodes
    t0 = time.perf_counter()
    nodes = fcc_nodes_in_box(box_L, lattice_a)
    margin = boundary_margin_frac * lattice_a
    interior_idx, boundary_idx = split_boundary_interior(nodes, box_L, margin)
    n_total    = len(nodes)
    n_interior = len(interior_idx)
    n_boundary = len(boundary_idx)
    t_nodes = time.perf_counter() - t0
    print(f"  a={lattice_a:.3f}  total={n_total}  interior={n_interior}  "
          f"boundary={n_boundary}  (nodes {t_nodes:.2f}s)")

    if n_interior < 10:
        print("  Too few interior nodes — skip.")
        return {"lattice_a": lattice_a, "n_interior": n_interior, "skipped": True}

    # 2. RBF-FD Hamiltonian
    t0 = time.perf_counter()
    H_sparse = build_H_rbf(
        nodes, interior_idx,
        k_neighbors=k_neighbors,
        phi_order=phi_order,
        poly_degree=poly_degree,
        omega=omega,
    )
    t_build = time.perf_counter() - t0
    print(f"  H_rbf built: nnz={H_sparse.nnz}  ({t_build:.2f}s)")

    def H_apply(psi):
        return H_sparse.dot(psi)

    # 3. Chebyshev explosion on n_random random states
    t0 = time.perf_counter()
    basis_cols = []
    for _ in range(n_random):
        psi = rng.standard_normal(n_interior)
        psi /= np.linalg.norm(psi)
        psi_f = chebyshev_explosion(H_apply, psi, cheb_m, E_lo, E_hi)
        norm_f = np.linalg.norm(psi_f)
        if norm_f > 1e-14:
            basis_cols.append(psi_f / norm_f)
    t_filter = time.perf_counter() - t0

    if not basis_cols:
        return {"lattice_a": lattice_a, "n_interior": n_interior, "skipped": True,
                "reason": "all filtered states vanished"}

    basis_mat = np.column_stack(basis_cols)  # (n_interior, n_random)
    print(f"  Filter done: {len(basis_cols)} states  ({t_filter:.2f}s)")

    # 4. Rayleigh-Ritz
    t0 = time.perf_counter()
    energies, _Ur, rank = svd_rayleigh_ritz_op(
        basis_mat, H_apply,
        svd_tol=svd_tol,
        max_energies=min(n_random, 200),
        hermitian=True,
    )
    t_ritz = time.perf_counter() - t0
    print(f"  Ritz: rank={rank}  n_eigs={len(energies)}  ({t_ritz:.2f}s)")

    # 5. Compare with exact
    exact = exact_ho3d_levels(max(len(energies), n_print), omega=omega)
    n_cmp = min(n_print, len(energies), len(exact))
    errs = np.abs(energies[:n_cmp] - exact[:n_cmp])
    print(f"  First {n_cmp} energies (RBF / exact / |err|):")
    for k in range(n_cmp):
        print(f"    [{k:3d}]  {energies[k]:10.6f}  {exact[k]:10.6f}  {errs[k]:.3e}")

    return {
        "lattice_a":        lattice_a,
        "n_total":          int(n_total),
        "n_interior":       int(n_interior),
        "n_boundary":       int(n_boundary),
        "H_nnz":            int(H_sparse.nnz),
        "energies":         energies.tolist(),
        "exact":            exact[:len(energies)].tolist(),
        "abs_errors":       errs.tolist(),
        "timings": {
            "nodes_s":  t_nodes,
            "build_s":  t_build,
            "filter_s": t_filter,
            "ritz_s":   t_ritz,
            "total_s":  t_nodes + t_build + t_filter + t_ritz,
        },
        "skipped": False,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_a_sweep(s: str) -> list[float]:
    """Parse 'a1,a2,...' or 'start:stop:step' (np.arange-style) float list."""
    if ':' in s:
        parts = s.split(':')
        if len(parts) == 2:
            vals = np.arange(float(parts[0]), float(parts[1]))
        elif len(parts) == 3:
            vals = np.arange(float(parts[0]), float(parts[1]), float(parts[2]))
        else:
            raise ValueError(f"Bad a_sweep range: {s!r}")
        return [round(float(v), 8) for v in vals]
    return [float(x) for x in s.split(',')]


def main():
    p = argparse.ArgumentParser(
        description="FCC+RBF-FD Chebyshev explosion on HO3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--box_L',     type=float, default=5.0)
    p.add_argument('--a_sweep',   type=str,   default='1.4,1.2,1.0',
                   help='FCC lattice parameter(s): comma list or start:stop:step')
    p.add_argument('--cheb_m',    type=int,   default=5)
    p.add_argument('--E_lo',      type=float, default=1.5)
    p.add_argument('--E_hi',      type=float, default=80.0)
    p.add_argument('--n_random',  type=int,   default=300)
    p.add_argument('--k_neighbors', type=int, default=40)
    p.add_argument('--phi_order', type=int,   default=3,
                   help='PHS exponent (1,3,5,...)')
    p.add_argument('--poly_degree', type=int, default=2,
                   help='Polynomial augmentation degree')
    p.add_argument('--omega',     type=float, default=1.0)
    p.add_argument('--boundary_margin_frac', type=float, default=1.0,
                   help='Boundary layer thickness in units of lattice_a')
    p.add_argument('--svd_tol',   type=float, default=1e-3)
    p.add_argument('--seed',      type=int,   default=42)
    p.add_argument('--n_print',   type=int,   default=20)
    p.add_argument('--out_json',  type=str,   default='rbf_fcc_explosion_ho3d.json')
    args = p.parse_args()

    a_list = parse_a_sweep(args.a_sweep)
    print(f"FCC + RBF-FD Chebyshev Explosion — HO3D")
    print(f"  box_L={args.box_L}  cheb_m={args.cheb_m}  "
          f"E_lo={args.E_lo}  E_hi={args.E_hi}")
    print(f"  n_random={args.n_random}  k_neighbors={args.k_neighbors}  "
          f"phi=phs{args.phi_order}  poly_deg={args.poly_degree}")
    print(f"  a sweep: {a_list}\n")

    sweep_results = []
    for a in a_list:
        print(f"{'='*60}")
        print(f"lattice_a = {a}")
        res = run_single(
            box_L=args.box_L,
            lattice_a=a,
            cheb_m=args.cheb_m,
            E_lo=args.E_lo,
            E_hi=args.E_hi,
            n_random=args.n_random,
            k_neighbors=args.k_neighbors,
            phi_order=args.phi_order,
            poly_degree=args.poly_degree,
            omega=args.omega,
            boundary_margin_frac=args.boundary_margin_frac,
            svd_tol=args.svd_tol,
            seed=args.seed,
            n_print=args.n_print,
        )
        sweep_results.append(res)

    out = {
        "params": {
            "box_L":               args.box_L,
            "a_sweep":             a_list,
            "cheb_m":              args.cheb_m,
            "E_lo":                args.E_lo,
            "E_hi":                args.E_hi,
            "n_random":            args.n_random,
            "k_neighbors":         args.k_neighbors,
            "phi_order":           args.phi_order,
            "poly_degree":         args.poly_degree,
            "omega":               args.omega,
            "boundary_margin_frac": args.boundary_margin_frac,
            "svd_tol":             args.svd_tol,
            "seed":                args.seed,
        },
        "sweep": sweep_results,
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out_json}")

    # Accuracy summary table
    print("\n── Accuracy summary (MAE on first n_print levels) ──")
    print(f"{'a':>8}  {'n_int':>8}  {'n_eigs':>7}  {'MAE':>12}  {'total_s':>9}")
    valid = [r for r in sweep_results if not r.get('skipped')]
    for r in sweep_results:
        if r.get('skipped'):
            print(f"{r['lattice_a']:8.3f}  SKIPPED")
            continue
        errs = np.array(r['abs_errors'][:args.n_print])
        mae  = float(np.mean(errs)) if len(errs) > 0 else float('nan')
        print(f"{r['lattice_a']:8.3f}  {r['n_interior']:8d}  "
              f"{len(r['energies']):7d}  {mae:12.4e}  "
              f"{r['timings']['total_s']:9.2f}s")

    # ── Figures ──────────────────────────────────────────────────────────────────
    if not valid:
        return

    stem = args.out_json.replace('.json', '')
    a_vals    = [r['lattice_a']        for r in valid]
    n_int     = [r['n_interior']       for r in valid]
    total_s   = [r['timings']['total_s'] for r in valid]
    mae_vals  = [float(np.mean(r['abs_errors'][:args.n_print]))
                 if r['abs_errors'] else float('nan') for r in valid]

    # Figure 1: Accuracy vs node count
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(n_int, mae_vals, 'o-', ms=7, color='steelblue')
    for ni, mae, a in zip(n_int, mae_vals, a_vals):
        ax.annotate(f'a={a}', (ni, mae), textcoords='offset points',
                    xytext=(4, 4), fontsize=8)
    ax.set_xlabel('N interior nodes')
    ax.set_ylabel(f'MAE on first {args.n_print} levels (Hartree)')
    ax.set_title('RBF-FD accuracy vs node count')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    # Plot RBF Ritz vs exact for all valid runs
    for r in valid:
        eigs = np.array(r['energies'][:args.n_print])
        exact = np.array(r['exact'][:len(eigs)])
        ax2.plot(exact, eigs, 'o', ms=4, alpha=0.7,
                 label=f'a={r["lattice_a"]}  n={r["n_interior"]}')
    if valid:
        e_range = [min(r['exact'][0] for r in valid if r['exact']),
                   max(r['exact'][min(args.n_print-1, len(r['exact'])-1)]
                       for r in valid if r['exact'])]
        ax2.plot(e_range, e_range, 'k--', lw=1, label='exact')
    ax2.set_xlabel('Exact energy (Hartree)')
    ax2.set_ylabel('RBF-FD Ritz energy (Hartree)')
    ax2.set_title('Ritz vs exact eigenvalues')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'FCC+RBF-FD Explosion — box_L={args.box_L}  '
                 f'm={args.cheb_m}  E_lo={args.E_lo}  phs{args.phi_order}+p{args.poly_degree}',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{stem}_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {stem}_accuracy.png")

    # Figure 2: Timing breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    build_s  = [r['timings']['build_s']  for r in valid]
    filter_s = [r['timings']['filter_s'] for r in valid]
    ritz_s   = [r['timings']['ritz_s']   for r in valid]
    x_pos = np.arange(len(valid))
    ax.bar(x_pos, build_s,  label='H build',  color='steelblue')
    ax.bar(x_pos, filter_s, bottom=build_s, label='filter', color='salmon')
    ax.bar(x_pos, ritz_s,
           bottom=[b+f for b,f in zip(build_s, filter_s)],
           label='Ritz', color='seagreen')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'a={r["lattice_a"]}\nn={r["n_interior"]}' for r in valid],
                       fontsize=8)
    ax.set_ylabel('Wall time (s)')
    ax.set_title('Timing breakdown per lattice parameter')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    ax2.loglog(n_int, total_s, 'o-', ms=7, color='darkorange', label='total')
    ax2.loglog(n_int, build_s,  's--', ms=5, color='steelblue', label='H build')
    ax2.loglog(n_int, filter_s, '^--', ms=5, color='salmon',    label='filter')
    ax2.loglog(n_int, ritz_s,   'd--', ms=5, color='seagreen',  label='Ritz')
    ax2.set_xlabel('N interior nodes')
    ax2.set_ylabel('Wall time (s)')
    ax2.set_title('Timing scaling')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'FCC+RBF-FD Timing — box_L={args.box_L}  n_random={args.n_random}',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f'{stem}_timing.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {stem}_timing.png")


if __name__ == '__main__':
    main()
