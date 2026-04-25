"""
Validate that `weight_matrix_conv_cell_reuse` reproduces `rbf.weight_matrix`
on a conv_cell tiling and is faster.

Run:
    python test_conv_cell_weight_reuse.py
"""

from __future__ import annotations

import time

import numpy as np

from rbf.pde.fd import weight_matrix
from rbf_core import (
    generate_conv_cell_nodes,
    read_cube_file,
    weight_matrix_conv_cell_reuse,
)


def _build_conv_cell_nodes(cube_file: str = "localPot.cube") -> tuple[np.ndarray, np.ndarray]:
    x_grid, y_grid, z_grid, _pot = read_cube_file(cube_file)
    bbox_min = np.array([x_grid[0], y_grid[0], z_grid[0]], dtype=np.float64)
    bbox_max = np.array(
        [
            x_grid[-1] + (x_grid[1] - x_grid[0]),
            y_grid[-1] + (y_grid[1] - y_grid[0]),
            z_grid[-1] + (z_grid[1] - z_grid[0]),
        ],
        dtype=np.float64,
    )

    nodes, groups, _stats = generate_conv_cell_nodes(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        a=11.4523,
        d_min_frac=0.06,
        n_random_target=120,
        seed=42,
        include_parity=True,
        template_mode="hybrid",
        boundary_margin_frac=0.06,
        use_rbf_poisson=True,
        domain_shape="cube",
        include_interior=True,
        include_boundary=True,
        min_dist_cart=0.0,
        verbose=False,
    )
    return nodes, groups["interior"]


def _time_once(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main() -> None:
    stencil_size = 80
    phi = "phs3"
    eps = 0.5
    order = 2
    diffs = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    nodes, interior_idx = _build_conv_cell_nodes()
    x = nodes[interior_idx]
    p = nodes
    print(f"[setup] N_nodes={p.shape[0]}  N_interior={x.shape[0]}  "
          f"stencil={stencil_size}")

    # ── reference (old path) ─────────────────────────────────────────────────
    t_old, L_old = _time_once(
        lambda: weight_matrix(
            x=x, p=p, n=stencil_size, diffs=diffs,
            phi=phi, eps=eps, order=order,
        ).tocsr()
    )
    L_old.sum_duplicates()
    L_old.sort_indices()

    # ── optimised (new path) ─────────────────────────────────────────────────
    t_new, L_new = _time_once(
        lambda: weight_matrix_conv_cell_reuse(
            x=x, p=p, n=stencil_size, diffs=diffs,
            phi=phi, eps=eps, order=order,
        )
    )
    L_new.sum_duplicates()
    L_new.sort_indices()

    # ── shape / sparsity pattern ─────────────────────────────────────────────
    assert L_new.shape == L_old.shape, (L_new.shape, L_old.shape)
    assert L_new.nnz == L_old.nnz, (L_new.nnz, L_old.nnz)
    assert np.array_equal(L_new.indptr, L_old.indptr), "CSR indptr mismatch"
    assert np.array_equal(L_new.indices, L_old.indices), (
        "CSR indices mismatch (after sort_indices) — KNN stencils differ"
    )

    # ── numerical equality ──────────────────────────────────────────────────
    diff = (L_new - L_old).tocsr()
    diff.sum_duplicates()
    max_abs = 0.0 if diff.nnz == 0 else float(np.max(np.abs(diff.data)))
    assert max_abs < 1e-10, f"max |L_new-L_old| = {max_abs:.3e}"

    rng = np.random.default_rng(0)
    vec = rng.standard_normal(p.shape[0])
    y_old = L_old.dot(vec)
    y_new = L_new.dot(vec)
    matvec_err = float(np.max(np.abs(y_new - y_old)))
    assert matvec_err < 1e-10, f"max |L_new@v - L_old@v| = {matvec_err:.3e}"

    # ── timing ──────────────────────────────────────────────────────────────
    speedup = t_old / max(t_new, 1e-12)
    print(
        f"[conv_cell_weight_reuse] old={t_old:.3f}s  new={t_new:.3f}s  "
        f"speedup={speedup:.2f}x  max|ΔL|={max_abs:.2e}  "
        f"max|Δ(Lv)|={matvec_err:.2e}"
    )
    assert t_new < t_old, (
        f"expected new path to be faster, got old={t_old:.3f}s "
        f"new={t_new:.3f}s"
    )

    print("[OK] weight_matrix_conv_cell_reuse matches rbf.weight_matrix and is faster.")


if __name__ == "__main__":
    main()
