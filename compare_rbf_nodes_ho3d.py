#!/usr/bin/env python3
"""compare_rbf_nodes_ho3d.py

Compare different RBF-FD node generation strategies on the 3-D harmonic
oscillator H = -½∇² + ½ω²r².

Node methods
------------
regular     : Cartesian grid nodes (spacing = h, uniform).
sphere      : Poisson-disc nodes inside a sphere of radius box_L.
fcc         : Face-centred cubic lattice nodes inside [-box_L, box_L]^3.
poisson_box : Poisson-disc nodes inside [-box_L, box_L]^3.

All methods apply Dirichlet ψ=0 on boundary nodes.
No unit-cell tiling is needed or supported (HO3D is a single-domain problem).

Solver options (--solver)
--------------------------
arnoldi   : scipy.sparse.linalg.eigs (ARPACK, non-Hermitian Arnoldi).
            If --target is set, uses shift-invert mode (sigma=target);
            otherwise computes the n_levels smallest-real-part eigenvalues.
explosion : Chebyshev explosion filter T_m(aH+b) + non-Hermitian Rayleigh-Ritz.

Note: The RBF-FD discretisation is non-symmetric, so we use non-Hermitian
solvers (Arnoldi instead of Lanczos, eig() instead of eigh() in Ritz).
Reported energies are sorted by real part; max |Im(λ)| is logged.

Spacing sweep
-------------
--spacing_sweep 1.4,1.2,1.0,0.8   comma-separated list
--spacing_sweep 1.4:0.8:-0.2       start:stop:step (np.arange-style)

For 'fcc' the spacing value is used as the lattice parameter a.

Usage examples
--------------
  # Lowest 20 eigenvalues, non-Hermitian Arnoldi
  python compare_rbf_nodes_ho3d.py \\
      --node_methods regular,sphere,fcc,poisson_box \\
      --box_L 5.0 --spacing_sweep 1.4,1.2,1.0,0.8 \\
      --solver arnoldi --n_levels 20 \\
      --k_neighbors 40 --phi_order 3 --poly_degree 2 \\
      --out_json rbf_nodes_ho3d_arnoldi.json

  # 20 eigenvalues near target (shift-invert)
  python compare_rbf_nodes_ho3d.py \\
      --node_methods regular,sphere,fcc,poisson_box \\
      --box_L 5.0 --spacing_sweep 1.4,1.2,1.0,0.8 \\
      --solver arnoldi --target 10.0 --n_levels 20 \\
      --k_neighbors 40 --phi_order 3 --poly_degree 2 \\
      --out_json rbf_nodes_ho3d_target.json

  # Chebyshev explosion filter
  python compare_rbf_nodes_ho3d.py \\
      --node_methods regular,sphere,fcc,poisson_box \\
      --box_L 5.0 --spacing_sweep 1.4,1.2,1.0,0.8 \\
      --solver explosion \\
      --cheb_m 12 --E_lo 7.0 --E_hi 50.0 \\
      --n_random 1000 --n_print 20 \\
      --k_neighbors 40 --phi_order 3 --poly_degree 2 \\
      --out_json rbf_nodes_ho3d_explosion.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from filter_core import svd_rayleigh_ritz_op

try:
    from rbf.pde.nodes import poisson_disc_nodes
    HAS_RBF = True
except ImportError:
    HAS_RBF = False


# ── exact HO3D energy levels ──────────────────────────────────────────────────

def exact_ho3d_levels(n_levels: int, omega: float = 1.0) -> np.ndarray:
    levels = []
    for shell in range(500):
        E = omega * (shell + 1.5)
        deg = (shell + 1) * (shell + 2) // 2
        levels.extend([E] * deg)
        if len(levels) >= n_levels:
            break
    return np.array(sorted(levels)[:n_levels])


# ── node generation ───────────────────────────────────────────────────────────

def _regular_nodes(box_L: float, spacing: float):
    """Cartesian grid nodes in [-box_L, box_L]^3."""
    N = max(3, int(round(2.0 * box_L / spacing)) + 1)
    x1d = np.linspace(-box_L, box_L, N)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    # boundary: nodes on the outer face(s)
    on_face = np.any(np.abs(nodes) >= box_L - 1e-10, axis=1)
    all_idx = np.arange(len(nodes))
    interior_idx = all_idx[~on_face]
    boundary_idx = all_idx[on_face]
    return nodes, interior_idx, boundary_idx


def _fcc_nodes(box_L: float, spacing: float, boundary_margin_frac: float = 1.0):
    """FCC lattice nodes in [-box_L, box_L]^3."""
    a = spacing
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.5, 0.5, 0.0],
                      [0.5, 0.0, 0.5],
                      [0.0, 0.5, 0.5]]) * a
    n_rep = int(np.ceil(box_L / a)) + 1
    idx = np.arange(-n_rep, n_rep + 1)
    ix, iy, iz = np.meshgrid(idx, idx, idx, indexing='ij')
    origins = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(float) * a
    pts = (origins[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    mask = np.all(np.abs(pts) <= box_L + 1e-10, axis=1)
    pts = pts[mask]
    if len(pts) > 0:
        pts = np.unique(np.round(pts, 10), axis=0)
    margin = boundary_margin_frac * a
    d_max = np.max(np.abs(pts), axis=1)
    on_boundary = d_max > box_L - margin
    all_idx = np.arange(len(pts))
    interior_idx = all_idx[~on_boundary]
    boundary_idx = all_idx[on_boundary]
    return pts, interior_idx, boundary_idx


def _sphere_nodes(box_L: float, spacing: float, n_subdivide: int = 3):
    """Poisson-disc nodes inside sphere of radius box_L."""
    if not HAS_RBF:
        raise ImportError("rbf package not installed; cannot use 'sphere' node method.")
    from rbf_core import generate_sphere_nodes
    nodes, groups = generate_sphere_nodes(spacing, R=box_L, n_subdivide=n_subdivide)
    interior_idx = np.asarray(groups.get('interior', []), dtype=np.int64)
    boundary_idx = np.asarray(groups.get('boundary', []), dtype=np.int64)
    if len(interior_idx) == 0:
        # fallback: use distance criterion
        r = np.linalg.norm(nodes, axis=1)
        all_idx = np.arange(len(nodes))
        boundary_idx = all_idx[r >= box_L - spacing * 0.5]
        interior_idx = all_idx[r < box_L - spacing * 0.5]
    return nodes, interior_idx, boundary_idx


def _poisson_box_nodes(box_L: float, spacing: float):
    """Poisson-disc nodes inside [-box_L, box_L]^3."""
    if not HAS_RBF:
        raise ImportError("rbf package not installed; cannot use 'poisson_box' node method.")
    from rbf_core import generate_nodes
    nodes, groups = generate_nodes(spacing=spacing, L=box_L)
    interior_idx = np.asarray(groups.get('interior', []), dtype=np.int64)
    boundary_idx = np.asarray(groups.get('boundary', []), dtype=np.int64)
    return nodes, interior_idx, boundary_idx


def generate_nodes_for_method(
    method: str,
    box_L: float,
    spacing: float,
    boundary_margin_frac: float = 1.0,
    sphere_subdivide: int = 3,
):
    """Dispatch to the appropriate node generator.

    Returns (nodes, interior_idx, boundary_idx).
    """
    if method == 'regular':
        return _regular_nodes(box_L, spacing)
    elif method == 'fcc':
        return _fcc_nodes(box_L, spacing, boundary_margin_frac)
    elif method == 'sphere':
        return _sphere_nodes(box_L, spacing, n_subdivide=sphere_subdivide)
    elif method == 'poisson_box':
        return _poisson_box_nodes(box_L, spacing)
    else:
        raise ValueError(f"Unknown node method: {method!r}. "
                         "Choose from: regular, fcc, sphere, poisson_box")


# ── RBF-FD Laplacian ──────────────────────────────────────────────────────────
# (self-contained; mirrors compare_rbf_fcc_explosion_ho3d.py)

def _monomials_3d(degree: int) -> list[tuple[int, int, int]]:
    monos = []
    for total in range(degree + 1):
        for i in range(total + 1):
            for j in range(total - i + 1):
                monos.append((i, j, total - i - j))
    return monos


def _eval_poly(pts: np.ndarray, exps: list) -> np.ndarray:
    P = np.empty((len(pts), len(exps)), dtype=np.float64)
    for col, (ei, ej, ek) in enumerate(exps):
        P[:, col] = pts[:, 0]**ei * pts[:, 1]**ej * pts[:, 2]**ek
    return P


def _lap_poly_at_origin(exps: list) -> np.ndarray:
    b = np.zeros(len(exps), dtype=np.float64)
    for col, (ei, ej, ek) in enumerate(exps):
        val = 0.0
        if ei >= 2:
            val += ei * (ei - 1) * int(ej == 0) * int(ek == 0)
        if ej >= 2:
            val += ej * (ej - 1) * int(ei == 0) * int(ek == 0)
        if ek >= 2:
            val += ek * (ek - 1) * int(ei == 0) * int(ej == 0)
        b[col] = val
    return b


def _phs_lap_at_center(r: np.ndarray, p: int) -> np.ndarray:
    if p == 2:
        return np.full_like(r, 6.0)
    return p * (p + 1) * np.where(r > 1e-14, r ** (p - 2), 0.0)


def rbf_fd_laplacian(
    nodes: np.ndarray,
    interior_idx: np.ndarray,
    k_neighbors: int = 40,
    phi_order: int = 3,
    poly_degree: int = 2,
) -> sp.csr_matrix:
    """PHS-augmented RBF-FD Laplacian, Dirichlet BCs on boundary."""
    n_total    = len(nodes)
    n_interior = len(interior_idx)
    monomials  = _monomials_3d(poly_degree)
    M = len(monomials)

    g2l = np.full(n_total, -1, dtype=np.int64)
    g2l[interior_idx] = np.arange(n_interior, dtype=np.int64)

    tree = cKDTree(nodes)
    rows_l, cols_l, data_l = [], [], []

    b_poly = _lap_poly_at_origin(monomials)

    for local_i, global_i in enumerate(interior_idx):
        xi = nodes[global_i]
        k = min(k_neighbors, n_total)
        _, nbr = tree.query(xi, k=k)
        dx = nodes[nbr] - xi

        diff = nodes[nbr][:, None, :] - nodes[nbr][None, :, :]
        r_mat = np.sqrt(np.sum(diff**2, axis=2))
        Phi = r_mat ** phi_order

        P = _eval_poly(dx, monomials)
        A = np.zeros((k + M, k + M))
        A[:k, :k] = Phi
        A[:k, k:] = P
        A[k:, :k] = P.T

        r_xi = np.sqrt(np.sum(dx**2, axis=1))
        b = np.concatenate([_phs_lap_at_center(r_xi, phi_order), b_poly])

        try:
            w = np.linalg.solve(A, b)[:k]
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0][:k]

        for j, gj in enumerate(nbr):
            lj = g2l[gj]
            if lj >= 0:
                rows_l.append(local_i)
                cols_l.append(lj)
                data_l.append(w[j])

    return sp.csr_matrix(
        (np.array(data_l), (np.array(rows_l, dtype=np.int64),
                            np.array(cols_l, dtype=np.int64))),
        shape=(n_interior, n_interior),
    )


def build_H_rbf(nodes, interior_idx, k_neighbors, phi_order, poly_degree, omega=1.0):
    L = rbf_fd_laplacian(nodes, interior_idx,
                         k_neighbors=k_neighbors,
                         phi_order=phi_order,
                         poly_degree=poly_degree)
    pts = nodes[interior_idx]
    V   = 0.5 * omega**2 * np.sum(pts**2, axis=1)
    return -0.5 * L + sp.diags(V, format='csr')


# ── solvers ───────────────────────────────────────────────────────────────────

def solve_arnoldi(
    H_sparse,
    n_levels: int,
    tol: float,
    max_matvecs: int,
    target: float | None = None,
):
    """Non-Hermitian Arnoldi (scipy ARPACK) eigensolver.

    target=None  → smallest real-part eigenvalues (which='SR').
    target=float → shift-invert mode (sigma=target, which='LM').
    """
    n = H_sparse.shape[0]
    ncv = min(max(4 * n_levels, 40), n - 1)
    n_levels_eff = min(n_levels, n - 2)
    if n_levels_eff < 1:
        return np.array([]), 0.0, 0.0, -1, False, "matrix too small"

    t0 = time.perf_counter()
    try:
        if target is None:
            evals_c = spla.eigs(
                H_sparse,
                k=n_levels_eff,
                which='SR',
                ncv=ncv,
                tol=tol,
                maxiter=max_matvecs,
                return_eigenvectors=False,
            )
        else:
            evals_c = spla.eigs(
                H_sparse,
                k=n_levels_eff,
                sigma=float(target),
                which='LM',
                ncv=ncv,
                tol=tol,
                maxiter=max_matvecs,
                return_eigenvectors=False,
            )
        t_wall = time.perf_counter() - t0
        order = np.argsort(evals_c.real)
        evals_c = evals_c[order]
        max_im = float(np.max(np.abs(evals_c.imag))) if len(evals_c) else 0.0
        return evals_c.real, t_wall, max_im, -1, True, ""
    except Exception as exc:
        t_wall = time.perf_counter() - t0
        return (np.full(n_levels, np.nan), t_wall, float('nan'), -1,
                False, str(exc))


def _chebyshev_explosion(H_apply, psi, m, E_lo, E_hi):
    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)
    def Hs(v): return a * H_apply(v) + b * v
    if m == 0:
        return psi.copy()
    yp, yc = psi.copy(), Hs(psi)
    for _ in range(2, m + 1):
        yp, yc = yc, 2.0 * Hs(yc) - yp
    return yc


def solve_explosion(H_sparse, n_interior, cheb_m, E_lo, E_hi, n_random, svd_tol, seed):
    """Chebyshev explosion + non-Hermitian Rayleigh-Ritz."""
    rng = np.random.default_rng(seed)
    H_apply = H_sparse.dot
    t0 = time.perf_counter()
    cols = []
    for _ in range(n_random):
        psi = rng.standard_normal(n_interior)
        psi /= np.linalg.norm(psi)
        pf = _chebyshev_explosion(H_apply, psi, cheb_m, E_lo, E_hi)
        nf = np.linalg.norm(pf)
        if nf > 1e-14 and np.all(np.isfinite(pf)):
            cols.append(pf / nf)
    t_filter = time.perf_counter() - t0
    if not cols:
        return (np.array([]), t_filter, float('nan'), 0,
                False, "all filtered states vanished or non-finite")
    basis = np.column_stack(cols)
    energies, _, rank = svd_rayleigh_ritz_op(
        basis, H_apply, svd_tol=svd_tol, max_energies=min(n_random, 200),
        hermitian=False,
    )
    t_wall = time.perf_counter() - t0
    # svd_rayleigh_ritz_op already returns sorted real parts; max_im is not
    # surfaced by that helper, so we set NaN as a placeholder.
    return energies, t_wall, float('nan'), int(rank), True, ""


# ── single (method, spacing) run ─────────────────────────────────────────────

def run_single(
    method: str,
    box_L: float,
    spacing: float,
    solver: str,
    # arnoldi params
    n_levels: int,
    arnoldi_tol: float,
    max_matvecs: int,
    target: float | None,
    # explosion params
    cheb_m: int,
    E_lo: float,
    E_hi: float,
    n_random: int,
    svd_tol: float,
    seed: int,
    # rbf-fd params
    k_neighbors: int,
    phi_order: int,
    poly_degree: int,
    omega: float,
    boundary_margin_frac: float,
    sphere_subdivide: int,
    n_print: int,
) -> dict:
    print(f"  method={method}  spacing={spacing:.3f}")

    # 1. Generate nodes
    t0 = time.perf_counter()
    try:
        nodes, interior_idx, boundary_idx = generate_nodes_for_method(
            method, box_L, spacing,
            boundary_margin_frac=boundary_margin_frac,
            sphere_subdivide=sphere_subdivide,
        )
    except Exception as exc:
        print(f"  [SKIP] node generation failed: {exc}")
        return {"method": method, "spacing": spacing, "skipped": True,
                "reason": f"node gen: {exc}"}
    t_nodes = time.perf_counter() - t0

    n_total    = len(nodes)
    n_interior = len(interior_idx)
    n_boundary = len(boundary_idx)
    print(f"  total={n_total}  interior={n_interior}  boundary={n_boundary}  "
          f"({t_nodes:.2f}s)")

    if n_interior < 10:
        print("  [SKIP] too few interior nodes")
        return {"method": method, "spacing": spacing, "n_interior": n_interior,
                "skipped": True, "reason": "too few interior nodes"}

    # 2. Build H
    t0 = time.perf_counter()
    try:
        H = build_H_rbf(nodes, interior_idx, k_neighbors, phi_order, poly_degree, omega)
    except Exception as exc:
        print(f"  [SKIP] H build failed: {exc}")
        return {"method": method, "spacing": spacing, "n_interior": n_interior,
                "skipped": True, "reason": f"H build: {exc}"}
    t_build = time.perf_counter() - t0
    print(f"  H nnz={H.nnz}  ({t_build:.2f}s)")

    # 2b. Benchmark a single H matvec (100 warm-up + 100 timed applications)
    _mv_warmup = 10
    _mv_reps   = 100
    _psi_bm    = np.ones(n_interior, dtype=np.float64)
    _psi_bm   /= np.linalg.norm(_psi_bm)
    for _ in range(_mv_warmup):
        H.dot(_psi_bm)
    _t_mv0 = time.perf_counter()
    for _ in range(_mv_reps):
        H.dot(_psi_bm)
    t_matvec_avg = (time.perf_counter() - _t_mv0) / _mv_reps
    print(f"  H matvec avg ({_mv_reps} reps): {t_matvec_avg*1e3:.4f} ms")

    # 3. Solve
    if solver == 'arnoldi':
        evals, t_solve, max_im, rank, ok, msg = solve_arnoldi(
            H, n_levels, arnoldi_tol, max_matvecs, target=target)
    else:  # explosion
        evals, t_solve, max_im, rank, ok, msg = solve_explosion(
            H, n_interior, cheb_m, E_lo, E_hi, n_random, svd_tol, seed)

    if not ok:
        print(f"  [SKIP] solver failed: {msg}")
        return {"method": method, "spacing": spacing, "n_interior": n_interior,
                "skipped": True, "reason": f"solver: {msg}"}

    if np.isfinite(max_im):
        print(f"  solver={solver}  n_eigs={len(evals)}  t_solve={t_solve:.2f}s  "
              f"max|Im(λ)|={max_im:.2e}")
    else:
        print(f"  solver={solver}  n_eigs={len(evals)}  t_solve={t_solve:.2f}s")

    # 4. Compare with exact
    # In target mode the user is interested in levels nearest target, so we
    # match each computed eigenvalue to the closest exact level individually.
    n_ref = max(len(evals), n_print, 200)
    exact_pool = exact_ho3d_levels(n_ref, omega=omega)
    if target is None:
        n_cmp = min(n_print, len(evals), len(exact_pool))
        ref = exact_pool[:n_cmp]
        errs = np.abs(evals[:n_cmp] - ref)
    else:
        n_cmp = min(n_print, len(evals))
        # nearest exact level for each computed value
        ref = np.array([exact_pool[np.argmin(np.abs(exact_pool - e))]
                        for e in evals[:n_cmp]])
        errs = np.abs(evals[:n_cmp] - ref)

    print(f"  First {n_cmp} eigenvalues  (computed / exact / |err|):")
    for k in range(n_cmp):
        print(f"    [{k:3d}]  {evals[k]:10.6f}  {ref[k]:10.6f}  {errs[k]:.3e}")

    return {
        "method":     method,
        "spacing":    spacing,
        "n_total":    int(n_total),
        "n_interior": int(n_interior),
        "n_boundary": int(n_boundary),
        "H_nnz":      int(H.nnz),
        "energies":   evals.tolist(),
        "exact":      ref.tolist(),
        "abs_errors": errs.tolist(),
        "solver":     solver,
        "target":     target,
        "max_imag":   None if not np.isfinite(max_im) else float(max_im),
        "rank":       int(rank) if rank >= 0 else None,
        "timings": {
            "nodes_s":       t_nodes,
            "build_s":       t_build,
            "matvec_avg_s":  t_matvec_avg,
            "solve_s":       t_solve,
            "total_s":       t_nodes + t_build + t_solve,
        },
        "skipped": False,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _make_plots(results_by_method: dict, args, stem: str, n_print: int):
    methods = list(results_by_method.keys())
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Figure 1: MAE vs n_interior, one line per method
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for i, m in enumerate(methods):
        valid = [r for r in results_by_method[m] if not r.get('skipped') and r.get('abs_errors')]
        if not valid:
            continue
        ni  = [r['n_interior']  for r in valid]
        mae = [float(np.mean(r['abs_errors'][:n_print])) for r in valid]
        sp_ = [r['spacing']     for r in valid]
        ax.semilogy(ni, mae, 'o-', color=colors[i % len(colors)], label=m, ms=6)
        for x, y, s in zip(ni, mae, sp_):
            ax.annotate(f'{s:.2f}', (x, y), textcoords='offset points',
                        xytext=(4, 4), fontsize=7)
    ax.set_xlabel('N interior nodes')
    ax.set_ylabel(f'MAE on first {n_print} levels')
    ax.set_title(f'RBF-FD accuracy vs node count\nsolver={args.solver}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Figure panel 2: Ritz vs exact scatter for all methods/spacings
    ax2 = axes[1]
    for i, m in enumerate(methods):
        valid = [r for r in results_by_method[m] if not r.get('skipped')]
        for r in valid:
            eigs  = np.array(r['energies'][:n_print])
            exact = np.array(r['exact'][:len(eigs)])
            ax2.plot(exact, eigs, 'o', ms=3, alpha=0.5, color=colors[i % len(colors)])
    # ideal line
    all_exact = []
    for m in methods:
        for r in results_by_method[m]:
            if not r.get('skipped') and r.get('exact'):
                all_exact.extend(r['exact'][:n_print])
    if all_exact:
        lo, hi = min(all_exact), max(all_exact)
        ax2.plot([lo, hi], [lo, hi], 'k--', lw=1, label='exact')
    # legend entries (one per method)
    for i, m in enumerate(methods):
        ax2.plot([], [], 'o', color=colors[i % len(colors)], label=m, ms=5)
    ax2.set_xlabel('Exact energy')
    ax2.set_ylabel('Ritz energy')
    ax2.set_title('Ritz vs exact eigenvalues')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'RBF-FD node methods — HO3D  box_L={args.box_L}  '
                 f'phi=phs{args.phi_order}+p{args.poly_degree}  solver={args.solver}',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{stem}_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {stem}_accuracy.png")

    # Figure 2: timing
    fig2, ax3 = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(methods):
        valid = [r for r in results_by_method[m] if not r.get('skipped')]
        if not valid:
            continue
        ni      = [r['n_interior'] for r in valid]
        total_s = [r['timings']['total_s'] for r in valid]
        ax3.loglog(ni, total_s, 'o-', color=colors[i % len(colors)], label=m, ms=6)
    ax3.set_xlabel('N interior nodes')
    ax3.set_ylabel('Total wall time (s)')
    ax3.set_title(f'Timing scaling — solver={args.solver}')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f'{stem}_timing.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved → {stem}_timing.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_float_sweep(s: str) -> list[float]:
    """Parse 'v1,v2,...' or 'start:stop:step' into a list of floats."""
    if ':' in s:
        parts = s.split(':')
        if len(parts) == 2:
            vals = np.arange(float(parts[0]), float(parts[1]))
        elif len(parts) == 3:
            vals = np.arange(float(parts[0]), float(parts[1]), float(parts[2]))
        else:
            raise ValueError(f"Bad sweep spec: {s!r}")
        return [round(float(v), 8) for v in vals]
    return [float(x) for x in s.split(',')]


def main():
    ap = argparse.ArgumentParser(
        description="Compare RBF-FD node methods on HO3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Node / domain
    ap.add_argument('--node_methods', type=str,
                    default='regular,sphere,fcc,poisson_box',
                    help='Comma-separated list: regular,sphere,fcc,poisson_box')
    ap.add_argument('--box_L',        type=float, default=5.0,
                    help='Domain half-size (box [-L,L]^3 or sphere radius L)')
    ap.add_argument('--spacing_sweep', type=str, default='1.4,1.2,1.0',
                    help='Node spacing values: comma list or start:stop:step')
    ap.add_argument('--boundary_margin_frac', type=float, default=1.0,
                    help='(fcc) boundary layer thickness in units of spacing')
    ap.add_argument('--sphere_subdivide', type=int, default=3,
                    help='(sphere) icosphere subdivision level')
    ap.add_argument('--omega',         type=float, default=1.0)

    # RBF-FD
    ap.add_argument('--k_neighbors', type=int,   default=40)
    ap.add_argument('--phi_order',   type=int,   default=3,
                    help='PHS exponent (3 → r³)')
    ap.add_argument('--poly_degree', type=int,   default=2,
                    help='Polynomial augmentation degree')

    # Solver
    ap.add_argument('--solver', type=str, default='arnoldi',
                    choices=['arnoldi', 'explosion'],
                    help='Eigenvalue solver (non-Hermitian): arnoldi or explosion')

    # Arnoldi params (non-Hermitian ARPACK)
    ap.add_argument('--n_levels',    type=int,   default=20,
                    help='(arnoldi) number of eigenvalues to compute')
    ap.add_argument('--target',      type=float, default=None,
                    help='(arnoldi) target energy: if set, shift-invert near '
                         'this value; else compute smallest-real-part eigs')
    ap.add_argument('--arnoldi_tol', type=float, default=1e-6)
    ap.add_argument('--max_matvecs', type=int,   default=10000)

    # Explosion params
    ap.add_argument('--cheb_m',   type=int,   default=5,
                    help='(explosion) Chebyshev degree m')
    ap.add_argument('--E_lo',     type=float, default=1.5,
                    help='(explosion) filter window lower bound')
    ap.add_argument('--E_hi',     type=float, default=80.0,
                    help='(explosion) filter window upper bound')
    ap.add_argument('--n_random', type=int,   default=300,
                    help='(explosion) number of random initial states')
    ap.add_argument('--svd_tol',  type=float, default=1e-3,
                    help='(explosion) SVD truncation tolerance for Ritz')
    ap.add_argument('--seed',     type=int,   default=42)

    # Output
    ap.add_argument('--n_print',  type=int,   default=20,
                    help='Number of levels to print/compare in summary')
    ap.add_argument('--out_json', type=str,   default='rbf_nodes_ho3d.json')

    args = ap.parse_args()

    methods  = [m.strip() for m in args.node_methods.split(',')]
    spacings = _parse_float_sweep(args.spacing_sweep)

    print("RBF-FD node method comparison — HO3D (non-Hermitian)")
    print(f"  box_L={args.box_L}  omega={args.omega}")
    print(f"  methods: {methods}")
    print(f"  spacing sweep: {spacings}")
    print(f"  solver: {args.solver}")
    if args.solver == 'arnoldi':
        tgt = "lowest" if args.target is None else f"near target={args.target}"
        print(f"  n_levels={args.n_levels}  mode={tgt}  tol={args.arnoldi_tol}")
    else:
        print(f"  cheb_m={args.cheb_m}  E_lo={args.E_lo}  E_hi={args.E_hi}  "
              f"n_random={args.n_random}")
    print(f"  k_neighbors={args.k_neighbors}  phi=phs{args.phi_order}  "
          f"poly_deg={args.poly_degree}\n")

    results_by_method: dict[str, list] = {m: [] for m in methods}
    all_results = []

    for method in methods:
        print(f"{'='*60}")
        print(f"NODE METHOD: {method}")
        for sp in spacings:
            print(f"  --- spacing={sp} ---")
            res = run_single(
                method=method,
                box_L=args.box_L,
                spacing=sp,
                solver=args.solver,
                n_levels=args.n_levels,
                arnoldi_tol=args.arnoldi_tol,
                max_matvecs=args.max_matvecs,
                target=args.target,
                cheb_m=args.cheb_m,
                E_lo=args.E_lo,
                E_hi=args.E_hi,
                n_random=args.n_random,
                svd_tol=args.svd_tol,
                seed=args.seed,
                k_neighbors=args.k_neighbors,
                phi_order=args.phi_order,
                poly_degree=args.poly_degree,
                omega=args.omega,
                boundary_margin_frac=args.boundary_margin_frac,
                sphere_subdivide=args.sphere_subdivide,
                n_print=args.n_print,
            )
            results_by_method[method].append(res)
            all_results.append(res)

    # Summary table
    print(f"\n{'='*80}")
    print("── Summary (MAE on first n_print levels) ──")
    print(f"{'method':>12}  {'spacing':>8}  {'n_int':>8}  {'n_eigs':>7}  "
          f"{'MAE':>12}  {'mv_avg_ms':>11}  {'total_s':>9}")
    for r in all_results:
        if r.get('skipped'):
            print(f"{r['method']:>12}  {r['spacing']:8.3f}  SKIPPED  ({r.get('reason','')})")
            continue
        errs = np.array(r['abs_errors'][:args.n_print])
        mae  = float(np.mean(errs)) if len(errs) else float('nan')
        mv_ms = r['timings']['matvec_avg_s'] * 1e3
        print(f"{r['method']:>12}  {r['spacing']:8.3f}  {r['n_interior']:8d}  "
              f"{len(r['energies']):7d}  {mae:12.4e}  {mv_ms:11.4f}  "
              f"{r['timings']['total_s']:9.2f}s")

    # Save JSON
    out = {
        "params": {
            "node_methods":    methods,
            "box_L":           args.box_L,
            "spacing_sweep":   spacings,
            "solver":          args.solver,
            "n_levels":        args.n_levels,
            "target":          args.target,
            "arnoldi_tol":     args.arnoldi_tol,
            "max_matvecs":     args.max_matvecs,
            "cheb_m":          args.cheb_m,
            "E_lo":            args.E_lo,
            "E_hi":            args.E_hi,
            "n_random":        args.n_random,
            "svd_tol":         args.svd_tol,
            "k_neighbors":     args.k_neighbors,
            "phi_order":       args.phi_order,
            "poly_degree":     args.poly_degree,
            "omega":           args.omega,
            "boundary_margin_frac": args.boundary_margin_frac,
        },
        "results": {m: results_by_method[m] for m in methods},
    }
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out_json}")

    # Plots
    stem = args.out_json.replace('.json', '')
    _make_plots(results_by_method, args, stem, args.n_print)


if __name__ == '__main__':
    main()
