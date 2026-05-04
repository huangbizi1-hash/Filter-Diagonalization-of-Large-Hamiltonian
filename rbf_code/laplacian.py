"""rbf_code.laplacian — Laplacian/Hamiltonian assembly and weight-matrix utilities.

Exposes:
    relative_laplacian_error   — diagnostic accuracy check
    build_hamiltonian_matrix   — sparse H on interior nodes
    compute_node_quality       — q/h/ρ mesh-quality metrics
    weight_matrix_conv_cell_reuse   — translation-reuse weight matrix
    weight_matrix_ball_fingerprint  — ball-radius stencil with fingerprint caching
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from rbf.pde.fd import weight_matrix

from rbf_code.config import Array, RBFProblem


def relative_laplacian_error(
    problem: RBFProblem,
    psi_nodes: Optional[Array] = None,
) -> Dict[str, float]:
    if psi_nodes is None:
        psi_nodes = problem.ground_state()
    exact = problem.exact_laplacian()[problem.interior_idx]
    approx = problem.laplacian_matrix.dot(psi_nodes)
    abs_err = np.abs(approx - exact)
    denom = np.maximum(np.abs(exact), 1e-14)
    rel_err = abs_err / denom
    return {
        "mean_abs_err": float(np.mean(abs_err)),
        "max_abs_err":  float(np.max(abs_err)),
        "mean_rel_err": float(np.mean(rel_err)),
        "max_rel_err":  float(np.max(rel_err)),
    }


def build_hamiltonian_matrix(
    problem: RBFProblem,
    symmetrize: bool = False,
) -> sp.csr_matrix:
    """
    Assemble sparse H = -0.5 * L_int + diag(V) on interior nodes.

    symmetrize=False (default)
        Return H as-is.  RBF-FD Laplacians are generally non-symmetric;
        use a non-symmetric eigensolver.
    symmetrize=True
        Apply (H + Hᵀ)/2 before returning, enabling eigsh (real, symmetric).
    """
    L_int = sp.csr_matrix(problem.laplacian_matrix)[:, problem.interior_idx]
    V_diag = sp.diags(problem.potential(), format="csr")
    H = -0.5 * sp.csr_matrix(L_int) + V_diag
    if symmetrize:
        H = 0.5 * (H + H.T)
    return H.tocsr()


def compute_node_quality(
    nodes: Array,
    bbox_min: Optional[Array] = None,
    bbox_max: Optional[Array] = None,
    probe_method: str = "uniform",
    n_probe: Optional[int] = None,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    Compute geometric quality metrics for a scattered node set X ⊂ Ω.

        q = ½ · min_{i≠j} ‖xᵢ − xⱼ‖        (node separation)
        h = sup_{x∈Ω} min_i ‖x − xᵢ‖       (fill radius / mesh norm)
        ρ = h / q                         (mesh ratio; → 1 is uniform)
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[0] < 2:
        return {
            "q": float("nan"), "h": float("nan"), "rho": float("nan"),
            "n_nodes": int(nodes.shape[0] if nodes.size else 0),
            "n_probe": 0, "probe_method": probe_method,
            "min_pair_dist": float("nan"),
        }

    if bbox_min is None:
        bbox_min = nodes.min(axis=0)
    if bbox_max is None:
        bbox_max = nodes.max(axis=0)
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)

    tree = cKDTree(nodes)
    dists2, _ = tree.query(nodes, k=2)
    min_pair = float(np.min(dists2[:, 1]))
    q = 0.5 * min_pair

    if n_probe is None:
        n_probe_target = max(4096, 32 * nodes.shape[0])
    else:
        n_probe_target = int(n_probe)

    if probe_method == "uniform":
        m = max(4, int(round(n_probe_target ** (1.0 / 3.0))))
        xs = np.linspace(bbox_min[0], bbox_max[0], m)
        ys = np.linspace(bbox_min[1], bbox_max[1], m)
        zs = np.linspace(bbox_min[2], bbox_max[2], m)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        probes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    elif probe_method == "random":
        rng = np.random.default_rng(random_seed)
        probes = rng.uniform(low=bbox_min, high=bbox_max, size=(n_probe_target, 3))
    else:
        raise ValueError(f"probe_method must be 'uniform' or 'random', got {probe_method!r}")

    probe_d, _ = tree.query(probes, k=1)
    h = float(np.max(probe_d))

    return {
        "q": q, "h": h, "rho": (h / q if q > 0 else float("inf")),
        "n_nodes": int(nodes.shape[0]),
        "n_probe": int(probes.shape[0]),
        "probe_method": probe_method,
        "min_pair_dist": min_pair,
        "domain_bbox_min": bbox_min.tolist(),
        "domain_bbox_max": bbox_max.tolist(),
    }


def weight_matrix_conv_cell_reuse(
    x: Array,
    p: Array,
    n: int,
    diffs: Any,
    phi: str,
    eps: float,
    order: int,
    round_decimals: int = 8,
) -> sp.csr_matrix:
    """
    RBF-FD weight matrix that reuses repeated local stencil solves.

    For tiled domains (e.g. conv_cell), stencils are exact translations;
    this groups rows by their canonicalised offset pattern and solves once
    per equivalence class, then broadcasts weights to all matching rows.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    p = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x must have shape (N, 3), got {x.shape}")
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"p must have shape (M, 3), got {p.shape}")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    N, M = x.shape[0], p.shape[0]
    k = int(min(n, M))
    if k == 0 or N == 0:
        return sp.csr_matrix((N, M), dtype=np.float64)

    tree = cKDTree(p)
    _dist, neigh = tree.query(x, k=k)
    if neigh.ndim == 1:
        neigh = neigh[:, None]
    neigh = neigh.astype(np.int64, copy=False)

    offsets = p[neigh] - x[:, None, :]
    rounded = np.round(offsets, round_decimals)
    sort_idx = np.lexsort(
        (rounded[:, :, 2], rounded[:, :, 1], rounded[:, :, 0]),
        axis=1,
    )
    sorted_offsets = np.take_along_axis(offsets, sort_idx[:, :, None], axis=1)
    sorted_neigh   = np.take_along_axis(neigh, sort_idx, axis=1)
    key_arr = np.round(sorted_offsets, round_decimals)

    keys = [key_arr[i].tobytes() for i in range(N)]
    rep_for_key: Dict[bytes, int] = {}
    for i, key in enumerate(keys):
        if key not in rep_for_key:
            rep_for_key[key] = i

    cached_w: Dict[bytes, Array] = {}
    for key, i_rep in rep_for_key.items():
        center   = x[i_rep:i_rep + 1]
        local_p  = p[sorted_neigh[i_rep]]
        local_w_sparse = weight_matrix(
            x=center, p=local_p, n=k,
            diffs=diffs, phi=phi, eps=eps, order=order,
        ).tocsr()
        local_w = np.zeros(k, dtype=np.float64)
        local_w[local_w_sparse.indices] = local_w_sparse.data
        cached_w[key] = local_w

    data_per_row = np.empty((N, k), dtype=np.float64)
    for i, key in enumerate(keys):
        data_per_row[i] = cached_w[key]
    cols_per_row = sorted_neigh

    col_order  = np.argsort(cols_per_row, axis=1, kind="stable")
    indices_2d = np.take_along_axis(cols_per_row, col_order, axis=1)
    data_2d    = np.take_along_axis(data_per_row, col_order, axis=1)

    indices = np.ascontiguousarray(indices_2d.reshape(-1))
    data    = np.ascontiguousarray(data_2d.reshape(-1))
    indptr  = np.arange(0, (N + 1) * k, k, dtype=indices.dtype)

    L = sp.csr_matrix((data, indices, indptr), shape=(N, M))
    L.has_sorted_indices = True

    n_unique = len(rep_for_key)
    print(
        f"[conv_cell_reuse] n_interior={N}, stencil_size={k}, "
        f"unique_patterns={n_unique}  (reuse {N / max(n_unique, 1):.1f}×, "
        f"solve_ratio={n_unique / max(N, 1):.3f})"
    )
    return L


def weight_matrix_ball_fingerprint(
    x: Array,
    p: Array,
    r: float,
    diffs,
    phi: str = "ga",
    eps: float = 1.0,
    order: int = 0,
    fingerprint_tol: float = 1e-4,
    min_stencil: int = 4,
    verbose: bool = True,
    inner_radius: float = 0.0,
    max_neighbors: int = 0,
    select_near_first: bool = True,
) -> tuple:
    """
    Build an RBF-FD weight matrix using ball-radius stencil with fingerprint reuse.

    For each interior node x_i, neighbours are all p_j with
    inner_radius <= ||x_i - p_j|| <= r  (center itself always kept).
    Rows with identical sorted relative-offset fingerprints are grouped;
    the RBF system is solved once per unique pattern and reused.
    """
    from rbf.pde.fd import weight_matrix as _rbf_wm

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n_int = x.shape[0]

    tree = cKDTree(p)
    raw_lists = tree.query_ball_point(x, r)

    inv_tol = 1.0 / fingerprint_tol
    groups_fp: Dict[tuple, list] = {}
    stencil_sizes: list[int] = []
    n_too_small = 0

    for i, raw_nbrs in enumerate(raw_lists):
        nbrs_arr = np.asarray(raw_nbrs, dtype=np.int64)
        offsets  = p[nbrs_arr] - x[i]
        dists    = np.linalg.norm(offsets, axis=1)

        if inner_radius > 0.0:
            keep     = (dists < 1e-14) | (dists >= inner_radius)
            nbrs_arr = nbrs_arr[keep]
            offsets  = offsets[keep]
            dists    = dists[keep]

        if max_neighbors > 0 and len(nbrs_arr) > max_neighbors:
            sort_idx = np.argsort(dists)
            if select_near_first:
                sel = sort_idx[:max_neighbors]
            else:
                center_mask = dists[sort_idx] < 1e-14
                center_idx  = sort_idx[center_mask]
                far_idx     = sort_idx[~center_mask][-(max_neighbors - len(center_idx)):]
                sel = np.concatenate([center_idx, far_idx])
            nbrs_arr = nbrs_arr[sel]
            offsets  = offsets[sel]

        k = len(nbrs_arr)
        stencil_sizes.append(k)
        if k < min_stencil:
            n_too_small += 1

        rounded   = np.round(offsets * inv_tol).astype(np.int64)
        lex_order = np.lexsort(rounded.T[::-1])
        key = tuple(map(tuple, rounded[lex_order]))
        if key not in groups_fp:
            groups_fp[key] = []
        groups_fp[key].append((i, nbrs_arr.tolist(), lex_order))

    _bin_edges  = [0, 8, 16, 32, 64, 128, 256]
    _bin_labels = ["<8", "8-15", "16-31", "32-63", "64-127", "128-255", ">=256"]
    sizes_arr   = np.array(stencil_sizes, dtype=np.int64)
    bin_counts: Dict[str, int] = {}
    for label, lo, hi in zip(_bin_labels, _bin_edges, _bin_edges[1:] + [2**31]):
        bin_counts[label] = int(np.sum((sizes_arr >= lo) & (sizes_arr < hi)))
    bin_pcts: Dict[str, float] = {
        k2: round(100.0 * v / max(n_int, 1), 2)
        for k2, v in bin_counts.items()
    }

    stencil_stats: Dict[str, Any] = {
        "n_interior":            n_int,
        "outer_radius_bohr":     float(r),
        "inner_radius_bohr":     float(inner_radius),
        "max_neighbors_cap":     int(max_neighbors),
        "select_near_first":     bool(select_near_first),
        "min_neighbors":         int(sizes_arr.min()) if n_int else 0,
        "max_neighbors_actual":  int(sizes_arr.max()) if n_int else 0,
        "mean_neighbors":        float(sizes_arr.mean()) if n_int else 0.0,
        "unique_patterns":       len(groups_fp),
        "neighbor_bins":         bin_counts,
        "neighbor_bins_pct":     bin_pcts,
    }

    if verbose:
        print(
            f"[ball_fingerprint] n_interior={n_int}, r={r:.4f} Bohr"
            + (f", r_inner={inner_radius:.4f}" if inner_radius > 0 else "")
            + (f", max_nbrs={max_neighbors}({'near' if select_near_first else 'far'})"
               if max_neighbors > 0 else "")
            + f"\n  stencil min/mean/max="
              f"{stencil_stats['min_neighbors']}/"
              f"{stencil_stats['mean_neighbors']:.1f}/"
              f"{stencil_stats['max_neighbors_actual']}, "
              f"unique_patterns={len(groups_fp)} "
              f"(reuse {n_int/max(len(groups_fp),1):.1f}×)"
        )
        parts = [f"{lbl}:{cnt}" for lbl, cnt in bin_counts.items() if cnt > 0]
        print(f"  neighbor distribution: {', '.join(parts)}")

    if n_too_small > 0:
        warnings.warn(
            f"[ball_fingerprint] {n_too_small} rows have <{min_stencil} neighbours "
            f"(r={r:.4f}, r_inner={inner_radius:.4f}) — check node layout.",
            RuntimeWarning, stacklevel=2,
        )

    row_idx: list[int]   = []
    col_idx: list[int]   = []
    wdata:   list[float] = []
    x_origin = np.zeros((1, 3), dtype=np.float64)

    for key, members in groups_fp.items():
        i0, nbrs0, lex0 = members[0]
        k = len(nbrs0)
        nbrs0_arr    = np.asarray(nbrs0)
        sorted_nbrs0 = nbrs0_arr[lex0]
        p_local      = p[sorted_nbrs0] - x[i0]

        try:
            wmat     = _rbf_wm(x=x_origin, p=p_local, n=k,
                               diffs=diffs, phi=phi, eps=eps, order=order)
            w_sorted = np.asarray(wmat.toarray()[0], dtype=np.float64)
        except Exception as exc:
            warnings.warn(
                f"[ball_fingerprint] weight_matrix failed (k={k}): {exc}",
                RuntimeWarning, stacklevel=2,
            )
            w_sorted = np.zeros(k, dtype=np.float64)

        for (i, nbrs, lex_i) in members:
            sorted_nbrs_i = np.asarray(nbrs)[lex_i]
            for j, j_global in enumerate(sorted_nbrs_i):
                row_idx.append(i)
                col_idx.append(int(j_global))
                wdata.append(float(w_sorted[j]))

    return (
        sp.csr_matrix(
            (wdata, (row_idx, col_idx)),
            shape=(n_int, p.shape[0]),
        ),
        stencil_stats,
    )
