"""rbf_code.periodic — periodic-geometry utilities (fractional-coordinate helpers).

All functions operate in fractional coordinates [0, 1)^3 unless noted.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from rbf_code.config import Array, _periodic_delta_frac


def _wrap_frac(x: Array) -> Array:
    return np.asarray(x, dtype=np.float64) % 1.0


def _unique_rows_mod1(arr: Array, tol: float = 1e-10) -> Array:
    arr = _wrap_frac(arr)
    arr_q = np.round(arr / tol).astype(np.int64)
    _, idx = np.unique(arr_q, axis=0, return_index=True)
    return arr[np.sort(idx)]


def _unique_cart_rows(arr: Array, tol: float = 1e-8) -> Array:
    arr_q = np.round(arr / tol).astype(np.int64)
    _, idx = np.unique(arr_q, axis=0, return_index=True)
    return arr[np.sort(idx)]


def _periodic_diff(x: Array, y: Array) -> Array:
    return np.abs(_periodic_delta_frac(x, y))


def _periodic_dist(x: Array, y: Array) -> float:
    return float(np.linalg.norm(_periodic_diff(x, y)))


def _build_uniform_frac_grid(n_per_axis: int) -> Array:
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis must be >= 2, got {n_per_axis}")
    t = np.linspace(0.0, 1.0, int(n_per_axis), endpoint=False, dtype=np.float64)
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _estimate_grad_laplacian_uniform(
    V: Array,
    n_per_axis: int,
    a: float,
) -> tuple[Array, Array]:
    """Estimate |∇V| and |ΔV| on a uniform [0,1)^3 grid of one conventional cell."""
    V3 = np.asarray(V, dtype=np.float64).reshape(n_per_axis, n_per_axis, n_per_axis)
    h = float(a) / float(n_per_axis)
    dVdx = (np.roll(V3, -1, axis=0) - np.roll(V3, 1, axis=0)) / (2.0 * h)
    dVdy = (np.roll(V3, -1, axis=1) - np.roll(V3, 1, axis=1)) / (2.0 * h)
    dVdz = (np.roll(V3, -1, axis=2) - np.roll(V3, 1, axis=2)) / (2.0 * h)
    grad_norm = np.sqrt(dVdx * dVdx + dVdy * dVdy + dVdz * dVdz)
    lap = (
        np.roll(V3, -1, axis=0) + np.roll(V3, 1, axis=0) +
        np.roll(V3, -1, axis=1) + np.roll(V3, 1, axis=1) +
        np.roll(V3, -1, axis=2) + np.roll(V3, 1, axis=2) -
        6.0 * V3
    ) / (h * h)
    return grad_norm.ravel(), np.abs(lap.ravel())


def _adaptive_accept_and_filter_periodic(
    candidates_frac: Array,
    weights_h: Array,
    d_min_frac: float,
    n_target: int,
    rng: np.random.Generator,
    pinned_frac: Optional[Array] = None,
) -> Array:
    """Weighted accept-reject + periodic KDTree minimum-distance filter."""
    cand = _wrap_frac(np.asarray(candidates_frac, dtype=np.float64).reshape(-1, 3))
    if cand.size == 0 or n_target <= 0:
        return np.empty((0, 3), dtype=np.float64)

    w = np.asarray(weights_h, dtype=np.float64).reshape(-1)
    if w.shape[0] != cand.shape[0]:
        raise ValueError("weights_h size mismatch with candidates_frac")
    h_max = float(np.max(w)) if w.size else 0.0
    if h_max <= 0.0:
        return np.empty((0, 3), dtype=np.float64)

    p_accept = np.clip(w / h_max, 0.0, 1.0)
    accepted_mask = rng.random(cand.shape[0]) < p_accept
    accepted = cand[accepted_mask]
    if accepted.shape[0] == 0:
        accepted = cand[np.argsort(-w)[: min(8, cand.shape[0])]]

    # process high-weight points first, improves quality for fixed budget
    acc_w = w[accepted_mask] if np.any(accepted_mask) else np.full(len(accepted), h_max)
    accepted = accepted[np.argsort(-acc_w)]

    shifts = np.array(
        [[i, j, k] for i in (-1.0, 0.0, 1.0)
         for j in (-1.0, 0.0, 1.0)
         for k in (-1.0, 0.0, 1.0)],
        dtype=np.float64,
    )

    kept: list[Array] = []
    pinned = (_wrap_frac(np.asarray(pinned_frac, dtype=np.float64).reshape(-1, 3))
              if pinned_frac is not None and len(pinned_frac) else
              np.empty((0, 3), dtype=np.float64))

    tree_points = pinned.copy()
    tree_aug = np.vstack([tree_points + s for s in shifts]) if len(tree_points) else np.empty((0, 3))
    tree = cKDTree(tree_aug) if len(tree_aug) else None
    r = float(d_min_frac)

    for x in accepted:
        if len(kept) >= n_target:
            break
        if tree is not None and tree.query_ball_point(x, r):
            continue
        kept.append(x)
        px = x[None, :]
        if tree_points.size:
            tree_points = np.vstack([tree_points, px])
        else:
            tree_points = px.copy()
        tree_aug = np.vstack([tree_points + s for s in shifts])
        tree = cKDTree(tree_aug)

    if not kept:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(kept, axis=0)


def _greedy_filter_by_dmin_periodic(points_frac: Array, d_min: float) -> Array:
    """Greedy d_min filter using periodic (minimum-image) distance in fractional
    coordinates.  Preserves the input order of `points_frac`."""
    kept: list[Array] = []
    for x in points_frac:
        if kept:
            dmin = min(_periodic_dist(x, y) for y in kept)
            if dmin < d_min:
                continue
        kept.append(x)
    if not kept:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(kept, axis=0)


def _make_unit_cube_surface(a: float) -> Tuple[Array, Array]:
    """8-vertex / 12-triangle surface of the cube [0,a]^3 for rbf's poisson_disc_nodes."""
    vert = np.array([
        [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
        [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
    ], dtype=np.float64)
    smp = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ], dtype=int)
    return vert, smp
