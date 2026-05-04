"""rbf_code.nodes — node generation for RBF-FD problems.

Includes Poisson-disc nodes, sphere nodes, atom-augmented nodes, and the
hybrid conventional-cell tiling strategy for periodic crystals.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from rbf.pde.nodes import poisson_disc_nodes

from rbf_code.config import Array, _raise_on_nonfinite, _periodic_delta_frac
from rbf_code.periodic import (
    _wrap_frac, _unique_rows_mod1, _periodic_diff, _periodic_dist,
    _build_uniform_frac_grid, _estimate_grad_laplacian_uniform,
    _adaptive_accept_and_filter_periodic, _greedy_filter_by_dmin_periodic,
    _make_unit_cube_surface,
)


# ── Default zincblende atomic basis (fractional, wrt conventional cubic cell) ─

_CONV_CELL_IN_FRAC_DEFAULT: Array = np.array(
    [[0.0, 0.0, 0.0],
     [0.5, 0.5, 0.0],
     [0.5, 0.0, 0.5],
     [0.0, 0.5, 0.5]],
    dtype=np.float64,
)
_CONV_CELL_AS_FRAC_DEFAULT: Array = np.array(
    [[0.25, 0.25, 0.25],
     [0.75, 0.75, 0.25],
     [0.75, 0.25, 0.75],
     [0.25, 0.75, 0.75]],
    dtype=np.float64,
)

# 4 FCC lattice points inside one conventional cubic cell
_FCC_OFFSETS_FRAC: Array = np.array(
    [[0.0, 0.0, 0.0],
     [0.5, 0.5, 0.0],
     [0.5, 0.0, 0.5],
     [0.0, 0.5, 0.5]],
    dtype=np.float64,
)

# level-3 “FCC high-symmetry” base points: 8 corners of {1/3,2/3}^3
_LEVEL3_BASE_FRAC_DEFAULT: Array = np.array(
    [[1/3, 1/3, 1/3],
     [2/3, 2/3, 2/3],
     [1/3, 1/3, 2/3],
     [1/3, 2/3, 1/3],
     [2/3, 1/3, 1/3],
     [2/3, 2/3, 1/3],
     [2/3, 1/3, 2/3],
     [1/3, 2/3, 2/3]],
    dtype=np.float64,
)


# ─────────────────────────────────────────────────────────────────────────────

def generate_nodes(spacing: float = 0.5, L: float = 5.0):
    vert = np.array(
        [
            [-L, -L, -L],
            [L, -L, -L],
            [L, L, -L],
            [-L, L, -L],
            [-L, -L, L],
            [L, -L, L],
            [L, L, L],
            [-L, L, L],
        ]
    )
    smp = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ]
    )
    nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp))
    return nodes, groups


def make_grid_points(L: float, N: int) -> Tuple[Array, Tuple[int, int, int]]:
    X, Y, Z = np.meshgrid(
        np.linspace(-L, L, N),
        np.linspace(-L, L, N),
        np.linspace(-L, L, N),
        indexing="ij",
    )
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts, X.shape


def _make_icosphere(R: float, n_subdivide: int = 3) -> Tuple[Array, Array]:
    """Triangulated sphere surface of radius R for use with poisson_disc_nodes."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array(
        [[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
         [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
         [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]],
        dtype=float,
    )
    verts: list[Array] = list(raw / np.linalg.norm(raw[0]) * R)
    faces: list[list[int]] = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]

    for _ in range(n_subdivide):
        midpoints: Dict[Tuple[int, int], int] = {}

        def _mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                m = (np.asarray(verts[a]) + np.asarray(verts[b])) / 2.0
                m = m / np.linalg.norm(m) * R
                midpoints[key] = len(verts)
                verts.append(m)
            return midpoints[key]

        new_faces: list[list[int]] = []
        for f in faces:
            a, b, c = f
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

    return np.array(verts), np.array(faces, dtype=int)


def generate_sphere_nodes(
    spacing: float, R: float = 20.0, n_subdivide: int = 3,
    min_dist: float = 0.0,
):
    """Poisson disc nodes inside a sphere of radius R."""
    vert, smp = _make_icosphere(R, n_subdivide)
    nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp))
    if min_dist > 0.0 and len(nodes) > 1:
        keep = []
        tree = None
        for i in range(len(nodes)):
            x = nodes[i]
            if tree is not None and tree.query_ball_point(x, min_dist):
                continue
            keep.append(i)
            tree = cKDTree(nodes[np.array(keep, dtype=np.int64)])
        keep = np.array(keep, dtype=np.int64)
        remap = -np.ones(len(nodes), dtype=np.int64)
        remap[keep] = np.arange(len(keep), dtype=np.int64)
        groups = {
            k: remap[np.asarray(v, dtype=np.int64)][remap[np.asarray(v, dtype=np.int64)] >= 0]
            for k, v in groups.items()
        }
        nodes = nodes[keep]
    return nodes, groups


def _filter_close_points(candidates: Array, pinned: Array, min_dist: float) -> Array:
    """Drop rows of `candidates` within `min_dist` of any row of `pinned`."""
    if len(pinned) == 0 or min_dist <= 0.0:
        return candidates
    tree = cKDTree(pinned)
    d, _ = tree.query(candidates, k=1)
    return candidates[d > min_dist]


def generate_atom_augmented_nodes(
    atom_positions: Array,
    domain: str = "sphere",
    R: float = 20.0,
    spacing: float = 0.8,
    augment: str = "poisson_disc",
    exclude_radius: float = 0.0,
    sphere_subdivide: int = 3,
    cube_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Build RBF-FD nodes pinned at atom positions, optionally augmented with
    Poisson-disc fill of the surrounding domain.
    """
    atom_positions = np.asarray(atom_positions, dtype=np.float64).reshape(-1, 3)
    n_atoms = len(atom_positions)

    if augment not in ("poisson_disc", "none"):
        raise ValueError(f"augment must be 'poisson_disc' or 'none', got {augment!r}")

    if augment == "none":
        nodes = atom_positions.copy()
        groups = {
            "interior": np.arange(n_atoms, dtype=np.int64),
            "boundary": np.empty(0, dtype=np.int64),
            "atoms":    np.arange(n_atoms, dtype=np.int64),
        }
        return nodes, groups

    if domain == "sphere":
        vert, smp = _make_icosphere(R, sphere_subdivide)
    elif domain == "box":
        if cube_bounds is None:
            raise ValueError("domain='box' requires cube_bounds=(lo, hi)")
        lo, hi = cube_bounds
        vert = np.array([
            [lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
            [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi],
        ], dtype=np.float64)
        smp = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ], dtype=int)
    else:
        raise ValueError(f"domain must be 'sphere' or 'box', got {domain!r}")

    try:
        aug_nodes, aug_groups, _ = poisson_disc_nodes(
            spacing, (vert, smp), pinned_nodes=atom_positions,
        )
    except TypeError:
        aug_nodes, aug_groups, _ = poisson_disc_nodes(spacing, (vert, smp))
        keep_interior = _filter_close_points(
            aug_nodes[aug_groups["interior"]], atom_positions,
            min_dist=max(exclude_radius, 0.5 * spacing),
        )
        keep_boundary = aug_nodes[aug_groups["boundary"]]
        nodes = np.vstack([atom_positions, keep_interior, keep_boundary])
        n_int_aug = len(keep_interior)
        n_bd = len(keep_boundary)
        groups = {
            "atoms":    np.arange(n_atoms, dtype=np.int64),
            "interior": np.arange(n_atoms + n_int_aug, dtype=np.int64),
            "boundary": np.arange(n_atoms + n_int_aug,
                                   n_atoms + n_int_aug + n_bd, dtype=np.int64),
        }
        return nodes, groups

    interior = aug_groups["interior"]
    boundary = aug_groups.get("boundary", np.empty(0, dtype=np.int64))
    if exclude_radius > 0.0 and n_atoms > 0:
        non_atom_mask = np.ones(len(interior), dtype=bool)
        non_atom_mask[:n_atoms] = False
        non_atom_interior = aug_nodes[interior[non_atom_mask]]
        kept = _filter_close_points(non_atom_interior, atom_positions, exclude_radius)
        atom_interior = aug_nodes[interior[:n_atoms]]
        nodes = np.vstack([atom_interior, kept, aug_nodes[boundary]])
        n_int_aug = len(kept)
        groups = {
            "atoms":    np.arange(n_atoms, dtype=np.int64),
            "interior": np.arange(n_atoms + n_int_aug, dtype=np.int64),
            "boundary": np.arange(n_atoms + n_int_aug,
                                   n_atoms + n_int_aug + len(boundary),
                                   dtype=np.int64),
        }
        return nodes, groups

    groups_out = {
        "atoms":    np.arange(n_atoms, dtype=np.int64),
        "interior": np.asarray(interior, dtype=np.int64),
        "boundary": np.asarray(boundary, dtype=np.int64),
    }
    return aug_nodes, groups_out


def _generate_shifted_refined_fcc_frac(
    scale_factor: int,
    origin_frac: Optional[Array] = None,
) -> Array:
    """Deterministic refined FCC nodes in one conventional cell, in [0,1)^3."""
    if int(scale_factor) < 1:
        raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
    sf = int(scale_factor)
    r0 = (np.zeros(3, dtype=np.float64) if origin_frac is None
          else np.asarray(origin_frac, dtype=np.float64).reshape(3))
    pts: list[Array] = []
    for i in range(sf):
        for j in range(sf):
            for k in range(sf):
                cell = np.array([i, j, k], dtype=np.float64) / sf
                for off in _FCC_OFFSETS_FRAC:
                    pts.append(r0 + cell + off / sf)
    return _unique_rows_mod1(np.asarray(pts, dtype=np.float64))


def _poisson_like_periodic(
    skeleton_frac: Array,
    n_target: int,
    d_min: float,
    max_trials: int,
    rng: np.random.Generator,
) -> Tuple[Array, int]:
    """Fallback periodic rejection sampler (no pinned_nodes API required)."""
    accepted: list[Array] = []
    trials = 0
    while len(accepted) < n_target and trials < max_trials:
        trials += 1
        x = rng.random(3)
        ok = True
        for y in skeleton_frac:
            if _periodic_dist(x, y) < d_min:
                ok = False
                break
        if ok:
            for y in accepted:
                if _periodic_dist(x, y) < d_min:
                    ok = False
                    break
        if ok:
            accepted.append(x)
    if not accepted:
        return np.empty((0, 3), dtype=np.float64), trials
    return np.stack(accepted, axis=0), trials


def generate_conv_cell_nodes(
    bbox_min: Array,
    bbox_max: Array,
    a: float,
    d_min_frac: float = 0.06,
    n_random_target: int = 120,
    seed: int = 42,
    include_parity: bool = True,
    template_mode: str = "hybrid",
    fcc_scale_factor: int = 8,
    fcc_origin_frac: Optional[Array] = None,
    fcc_atom_refine_factor: int = 0,
    fcc_atom_radius_frac: float = 0.0,
    atom_frac: Optional[Array] = None,
    level3_base_frac: Optional[Array] = None,
    use_rbf_poisson: bool = True,
    boundary_margin_frac: float = 0.06,
    domain_shape: str = "cube",
    sphere_radius: Optional[float] = None,
    sphere_subdivide: int = 3,
    adaptive_random: bool = False,
    adaptive_grid_n: int = 36,
    adaptive_lambda_grad: float = 0.0,
    adaptive_lambda_lap: float = 0.0,
    adaptive_candidate_multiplier: float = 8.0,
    adaptive_gaussian_builder: Optional[Any] = None,
    include_skeleton: bool = True,
    include_interior: bool = True,
    include_boundary: bool = True,
    min_dist_cart: float = 0.0,
    verbose: bool = True,
) -> Tuple[Array, Dict[str, Array], Dict[str, Any]]:
    """
    Hybrid node generator on a conventional cubic cell (zincblende by default),
    tiled to fill an axis-aligned box or sphere domain.
    """
    t_start = time.perf_counter()
    t_prev = t_start

    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)

    if not include_skeleton:
        atom_frac = np.empty((0, 3), dtype=np.float64)
        level3_base_frac = np.empty((0, 3), dtype=np.float64)
    else:
        if atom_frac is None:
            atom_frac = np.vstack([_CONV_CELL_IN_FRAC_DEFAULT,
                                    _CONV_CELL_AS_FRAC_DEFAULT])
        else:
            atom_frac = np.asarray(atom_frac, dtype=np.float64).reshape(-1, 3)

        if level3_base_frac is None:
            level3_base_frac = _LEVEL3_BASE_FRAC_DEFAULT
        else:
            level3_base_frac = np.asarray(level3_base_frac, dtype=np.float64).reshape(-1, 3)

    if len(level3_base_frac):
        level3_frac = np.vstack([
            _wrap_frac(r0 + _FCC_OFFSETS_FRAC) for r0 in level3_base_frac
        ])
        level3_frac = _unique_rows_mod1(level3_frac)
    else:
        level3_frac = np.empty((0, 3), dtype=np.float64)

    if template_mode not in ("hybrid", "fcc_refined"):
        raise ValueError(f"template_mode must be 'hybrid' or 'fcc_refined', got {template_mode!r}")

    if template_mode == "fcc_refined":
        base_fcc_frac = _generate_shifted_refined_fcc_frac(
            scale_factor=fcc_scale_factor, origin_frac=fcc_origin_frac)
        cell_nodes_frac = base_fcc_frac
        cell_role_idx = np.full(len(cell_nodes_frac), 4, dtype=np.int64)

        n_base_fcc = int(len(cell_nodes_frac))
        n_enh1_added = 0
        n_enh2_added = 0

        if int(fcc_atom_refine_factor) > fcc_scale_factor and float(fcc_atom_radius_frac) > 0.0:
            local_fcc_frac = _generate_shifted_refined_fcc_frac(
                scale_factor=int(fcc_atom_refine_factor),
                origin_frac=fcc_origin_frac,
            )
            rad = float(fcc_atom_radius_frac)
            if len(local_fcc_frac) > 0 and len(atom_frac) > 0:
                near_atom = np.zeros(len(local_fcc_frac), dtype=bool)
                for af in atom_frac:
                    d = np.abs(_periodic_delta_frac(local_fcc_frac, af))
                    near_atom |= np.sqrt(np.sum(d * d, axis=1)) < rad
                local_fcc_frac = local_fcc_frac[near_atom]

            if len(local_fcc_frac) > 0 and len(base_fcc_frac) > 0:
                _shifts = np.array(
                    [[i, j, k] for i in (-1., 0., 1.)
                     for j in (-1., 0., 1.)
                     for k in (-1., 0., 1.)], dtype=np.float64)
                base_aug = np.vstack([base_fcc_frac + s for s in _shifts])
                dists_fine, _ = cKDTree(base_aug).query(local_fcc_frac, k=1)
                local_fcc_frac = local_fcc_frac[dists_fine >= d_min_frac]

            n_enh1_added = int(len(local_fcc_frac))
            if n_enh1_added > 0:
                cell_nodes_frac = np.vstack([base_fcc_frac, local_fcc_frac])
                cell_role_idx = np.concatenate([
                    np.full(len(base_fcc_frac), 4, dtype=np.int64),
                    np.full(n_enh1_added, 5, dtype=np.int64),
                ])
            if verbose:
                print(f"[conv_cell] Enhancement 1 (fine FCC near atoms): "
                      f"refine_factor={fcc_atom_refine_factor}, "
                      f"radius_frac={fcc_atom_radius_frac:.3f}, "
                      f"added {n_enh1_added} nodes")

        if adaptive_random:
            if adaptive_gaussian_builder is None:
                raise ValueError("adaptive_random=True requires adaptive_gaussian_builder")
            rng_enh2 = np.random.default_rng(seed)
            n_grid_enh2 = int(adaptive_grid_n)
            cand_frac_enh2 = _build_uniform_frac_grid(n_grid_enh2)
            V_cand_enh2 = np.asarray(
                adaptive_gaussian_builder.evaluate_at_points(cand_frac_enh2 * a),
                dtype=np.float64,
            )
            gnorm_enh2, lap_enh2 = _estimate_grad_laplacian_uniform(
                V_cand_enh2, n_grid_enh2, a)
            denom_enh2 = (1.0
                          + float(adaptive_lambda_grad) * gnorm_enh2
                          + float(adaptive_lambda_lap) * lap_enh2)
            h_w_enh2 = 1.0 / np.maximum(denom_enh2, 1e-12)
            n_cands_enh2 = int(max(1, round(float(adaptive_candidate_multiplier)
                                             * n_random_target)))
            n_take_enh2 = min(n_cands_enh2, len(h_w_enh2))
            take_enh2 = rng_enh2.choice(len(h_w_enh2), size=n_take_enh2, replace=False)
            random_frac_enh2 = _adaptive_accept_and_filter_periodic(
                candidates_frac=cand_frac_enh2[take_enh2],
                weights_h=h_w_enh2[take_enh2],
                d_min_frac=d_min_frac,
                n_target=n_random_target,
                rng=rng_enh2,
                pinned_frac=cell_nodes_frac,
            )
            n_enh2_added = int(len(random_frac_enh2))
            if n_enh2_added > 0:
                cell_nodes_frac = np.vstack([cell_nodes_frac, random_frac_enh2])
                cell_role_idx = np.concatenate([
                    cell_role_idx,
                    np.full(n_enh2_added, 2, dtype=np.int64),
                ])
            if verbose:
                print(f"[conv_cell] Enhancement 2 (adaptive random): "
                      f"λ_grad={adaptive_lambda_grad}, λ_lap={adaptive_lambda_lap}, "
                      f"added {n_enh2_added} nodes")

        cell_stats: Dict[str, Any] = {
            "cell_skeleton":         0,
            "cell_random":           n_enh2_added,
            "cell_accepted_parity":  0,
            "cell_after_filter":     int(len(cell_nodes_frac)),
            "cell_template_nodes_frac": cell_nodes_frac.copy(),
            "cell_template_nodes_cart": (cell_nodes_frac * a).copy(),
            "cell_template_roles":      cell_role_idx.copy(),
            "template_mode": "fcc_refined",
            "fcc_scale_factor": int(fcc_scale_factor),
            "fcc_base_nodes": n_base_fcc,
            "fcc_enh1_atom_refine_added": n_enh1_added,
            "fcc_enh2_adaptive_random_added": n_enh2_added,
        }
    else:
        skeleton_frac = _unique_rows_mod1(np.vstack([atom_frac, level3_frac]))

        rng = np.random.default_rng(seed)
        if adaptive_random:
            if adaptive_gaussian_builder is None:
                raise ValueError("adaptive_random=True requires adaptive_gaussian_builder")
            n_grid = int(adaptive_grid_n)
            cand_frac = _build_uniform_frac_grid(n_grid)
            cand_cart = cand_frac * a
            V_cand = np.asarray(adaptive_gaussian_builder.evaluate_at_points(cand_cart),
                                dtype=np.float64)
            _raise_on_nonfinite("conv_cell adaptive V_cand", V_cand)
            gnorm, lap_abs = _estimate_grad_laplacian_uniform(V_cand, n_grid, a)
            _raise_on_nonfinite("conv_cell adaptive |grad V|", gnorm)
            _raise_on_nonfinite("conv_cell adaptive |lap V|", lap_abs)
            denom = 1.0 + float(adaptive_lambda_grad) * gnorm + float(adaptive_lambda_lap) * lap_abs
            _raise_on_nonfinite("conv_cell adaptive denom", denom)
            if np.any(denom <= 0.0):
                dmin = float(np.min(denom))
                n_nonpos = int(np.count_nonzero(denom <= 0.0))
                raise ValueError(
                    "conv_cell adaptive denom has non-positive entries "
                    f"({n_nonpos}/{denom.size}); min={dmin:.6e}. "
                    "Please reduce adaptive lambdas."
                )
            h_w = 1.0 / np.maximum(denom, 1e-12)
            _raise_on_nonfinite("conv_cell adaptive h_w", h_w)
            n_candidates = int(max(1, round(float(adaptive_candidate_multiplier) * n_random_target)))
            n_take = min(n_candidates, len(h_w))
            take = rng.choice(len(h_w), size=n_take, replace=False)
            random_frac = _adaptive_accept_and_filter_periodic(
                candidates_frac=cand_frac[take],
                weights_h=h_w[take],
                d_min_frac=d_min_frac,
                n_target=n_random_target,
                rng=rng,
                pinned_frac=skeleton_frac,
            )
            _raise_on_nonfinite("conv_cell adaptive random_frac", random_frac)
        elif use_rbf_poisson:
            vert, smp = _make_unit_cube_surface(a)
            pinned_cart = skeleton_frac * a
            try:
                rbf_nodes_cart, _rbf_groups, _ = poisson_disc_nodes(
                    d_min_frac * a, (vert, smp), pinned_nodes=pinned_cart,
                )
                n_pin = len(pinned_cart)
                extra_cart = rbf_nodes_cart[n_pin:]
                face_tol = 1e-6 * a
                in_interior = np.all(
                    (extra_cart > face_tol) & (extra_cart < a - face_tol), axis=1)
                extra_cart = extra_cart[in_interior]
                random_frac = extra_cart / a
                if len(random_frac) > n_random_target:
                    perm = rng.permutation(len(random_frac))[:n_random_target]
                    random_frac = random_frac[perm]
            except TypeError:
                use_rbf_poisson = False

        if not use_rbf_poisson:
            random_frac, _trials = _poisson_like_periodic(
                skeleton_frac, n_random_target, d_min_frac,
                max_trials=max(200_000, 2000 * n_random_target), rng=rng,
            )

        if include_parity and len(random_frac) > 0:
            parity_frac = _wrap_frac(1.0 - random_frac)
        else:
            parity_frac = np.empty((0, 3), dtype=np.float64)

        parts = [
            ("atoms",  atom_frac),
            ("level3", level3_frac),
            ("random", random_frac),
            ("parity", parity_frac),
        ]
        ordered_frac = np.vstack([p[1] for p in parts if len(p[1])])
        ordered_frac = _unique_rows_mod1(ordered_frac)
        role_map: Dict[str, Array] = {}
        for name, arr in parts:
            role_map[name] = _wrap_frac(arr) if len(arr) else np.empty((0, 3), dtype=np.float64)

        cell_nodes_frac = _greedy_filter_by_dmin_periodic(ordered_frac, d_min_frac)
        cell_nodes_frac = _unique_rows_mod1(cell_nodes_frac)

        cell_role_idx = []
        role_priority = ["atoms", "level3", "random", "parity"]
        for node in cell_nodes_frac:
            assigned = 3
            for ri, rname in enumerate(role_priority):
                src = role_map[rname]
                if len(src) and np.any(np.all(np.abs(_periodic_diff(src, node)) < 1e-9, axis=1)):
                    assigned = ri
                    break
            cell_role_idx.append(assigned)
        cell_role_idx = np.asarray(cell_role_idx, dtype=np.int64)

        cell_stats = {
            "cell_skeleton":         int(len(skeleton_frac)),
            "cell_random":           int(len(random_frac)),
            "cell_accepted_parity":  int(len(parity_frac)),
            "cell_after_filter":     int(len(cell_nodes_frac)),
            "cell_template_nodes_frac": cell_nodes_frac.copy(),
            "cell_template_nodes_cart": (cell_nodes_frac * a).copy(),
            "cell_template_roles":      cell_role_idx.copy(),
            "template_mode": "hybrid",
        }

    t_after_cell_template = time.perf_counter()

    cell_nodes_cart = cell_nodes_frac * a
    if len(cell_nodes_cart) == 0:
        return (np.empty((0, 3), dtype=np.float64),
                {k: np.empty(0, dtype=np.int64)
                 for k in ("interior", "boundary", "atoms", "level3", "random", "parity",
                           "fcc", "fcc_local")},
                {**cell_stats, "tiled_total": 0})

    nmin = np.floor((bbox_min - cell_nodes_cart.max(axis=0)) / a).astype(int) - 1
    nmax = np.ceil((bbox_max - cell_nodes_cart.min(axis=0)) / a).astype(int) + 1

    tiled_nodes: list[Array] = []
    tiled_roles: list[Array] = []
    for nx in range(nmin[0], nmax[0] + 1):
        for ny in range(nmin[1], nmax[1] + 1):
            for nz in range(nmin[2], nmax[2] + 1):
                shift = np.array([nx, ny, nz], dtype=np.float64) * a
                pts = cell_nodes_cart + shift
                m = np.all(pts >= bbox_min, axis=1) & np.all(pts < bbox_max, axis=1)
                if np.any(m):
                    tiled_nodes.append(pts[m])
                    tiled_roles.append(cell_role_idx[m])

    if not tiled_nodes:
        return (np.empty((0, 3), dtype=np.float64),
                {k: np.empty(0, dtype=np.int64)
                 for k in ("interior", "boundary", "atoms", "level3", "random", "parity",
                           "fcc", "fcc_local")},
                {**cell_stats, "tiled_total": 0})

    nodes = np.vstack(tiled_nodes)
    roles = np.concatenate(tiled_roles)
    nodes_u, idx_u = np.unique(
        np.round(nodes / 1e-8).astype(np.int64), axis=0, return_index=True)
    idx_u = np.sort(idx_u)
    nodes = nodes[idx_u]
    roles = roles[idx_u]
    nodes_tiled_raw = nodes.copy()
    roles_tiled_raw = roles.copy()
    t_after_tiling = time.perf_counter()

    if domain_shape not in ("cube", "sphere"):
        raise ValueError(
            f"domain_shape must be 'cube' or 'sphere', got {domain_shape!r}")

    if domain_shape == "cube":
        margin = boundary_margin_frac * a
        near_face = (
            (nodes[:, 0] < bbox_min[0] + margin) | (nodes[:, 0] > bbox_max[0] - margin) |
            (nodes[:, 1] < bbox_min[1] + margin) | (nodes[:, 1] > bbox_max[1] - margin) |
            (nodes[:, 2] < bbox_min[2] + margin) | (nodes[:, 2] > bbox_max[2] - margin)
        )
        interior_idx = np.where(~near_face)[0].astype(np.int64)
        boundary_idx = np.where(near_face)[0].astype(np.int64)
        sphere_center = None
        sphere_radius_eff = None
    else:
        sphere_center = 0.5 * (bbox_min + bbox_max)
        if sphere_radius is None:
            sphere_radius_eff = 0.5 * float(np.min(bbox_max - bbox_min))
        else:
            sphere_radius_eff = float(sphere_radius)
        if sphere_radius_eff <= 0.0:
            raise ValueError(f"sphere_radius must be > 0, got {sphere_radius_eff}")

        r = np.linalg.norm(nodes - sphere_center[None, :], axis=1)
        rad_tol = max(1e-8, 1e-6 * a)
        interior_margin = max(d_min_frac * a, boundary_margin_frac * a)
        inside = r <= (sphere_radius_eff - interior_margin + rad_tol)
        nodes = nodes[inside]
        roles = roles[inside]

        ico_verts_raw, _ = _make_icosphere(sphere_radius_eff, sphere_subdivide)
        ico_verts = ico_verts_raw + sphere_center[None, :]
        n_interior = len(nodes)
        n_ico = len(ico_verts)

        nodes = np.vstack([nodes, ico_verts])
        roles = np.concatenate([roles, np.full(n_ico, 6, dtype=np.int64)])

        interior_idx = np.arange(n_interior, dtype=np.int64)
        boundary_idx = np.arange(n_interior, n_interior + n_ico, dtype=np.int64)
        margin = 0.0

    nodes_after_domain = nodes.copy()
    roles_after_domain = roles.copy()
    t_after_domain = time.perf_counter()

    groups: Dict[str, Array] = {
        "interior":  interior_idx,
        "boundary":  boundary_idx,
        "atoms":     np.where(roles == 0)[0].astype(np.int64),
        "level3":    np.where(roles == 1)[0].astype(np.int64),
        "random":    np.where(roles == 2)[0].astype(np.int64),
        "parity":    np.where(roles == 3)[0].astype(np.int64),
        "fcc":       np.where(roles == 4)[0].astype(np.int64),
        "fcc_local": np.where(roles == 5)[0].astype(np.int64),
        "icosphere": np.where(roles == 6)[0].astype(np.int64),
    }

    if min_dist_cart > 0.0 and len(nodes) > 1:
        protected = np.zeros(len(nodes), dtype=bool)
        protected[groups["boundary"]] = True
        protected[groups["atoms"]] = True
        order_idx = np.concatenate([np.where(protected)[0], np.where(~protected)[0]])
        keep: list[int] = []
        tree = None
        for i in order_idx:
            p = nodes[i]
            if tree is not None and tree.query_ball_point(p, min_dist_cart):
                continue
            keep.append(int(i))
            tree = cKDTree(nodes[np.array(keep, dtype=np.int64)])
        keep = np.array(sorted(keep), dtype=np.int64)
        remap = -np.ones(len(nodes), dtype=np.int64)
        remap[keep] = np.arange(len(keep), dtype=np.int64)
        for k in list(groups.keys()):
            mapped = remap[groups[k]]
            groups[k] = mapped[mapped >= 0].astype(np.int64)
        nodes = nodes[keep]
    nodes_after_close_filter = nodes.copy()
    t_after_close_filter = time.perf_counter()

    if not include_interior or not include_boundary:
        keep_mask = np.zeros(len(nodes), dtype=bool)
        if include_interior:
            keep_mask[groups["interior"]] = True
        if include_boundary:
            keep_mask[groups["boundary"]] = True
        keep = np.where(keep_mask)[0].astype(np.int64)
        remap = -np.ones(len(nodes), dtype=np.int64)
        remap[keep] = np.arange(len(keep), dtype=np.int64)
        for k in list(groups.keys()):
            mapped = remap[groups[k]]
            groups[k] = mapped[mapped >= 0].astype(np.int64)
        nodes = nodes[keep]
    t_after_group_select = time.perf_counter()

    stats = {
        **cell_stats,
        "tiled_total": int(len(nodes)),
        "domain_shape": domain_shape,
        "sphere_radius": (float(sphere_radius_eff) if sphere_radius_eff is not None else None),
        "adaptive_random": bool(adaptive_random),
        "step_cell_template_nodes_cart": (cell_nodes_frac * a).copy(),
        "step_tiled_raw_nodes_cart": nodes_tiled_raw,
        "step_tiled_raw_roles": roles_tiled_raw.copy(),
        "step_after_domain_nodes_cart": nodes_after_domain,
        "step_after_domain_roles": roles_after_domain.copy(),
        "step_after_close_filter_nodes_cart": nodes_after_close_filter,
        "timings_seconds": {
            "cell_template": float(t_after_cell_template - t_prev),
            "tiling": float(t_after_tiling - t_after_cell_template),
            "domain_tagging": float(t_after_domain - t_after_tiling),
            "close_filter": float(t_after_close_filter - t_after_domain),
            "group_select": float(t_after_group_select - t_after_close_filter),
            "total": float(t_after_group_select - t_start),
        },
    }

    if verbose:
        tile_range = (nmax - nmin + 1).tolist()
        a_bbox_min = nodes.min(axis=0)
        a_bbox_max = nodes.max(axis=0)
        i_bbox_min = (nodes[interior_idx].min(axis=0)
                      if len(interior_idx) else np.full(3, np.nan))
        i_bbox_max = (nodes[interior_idx].max(axis=0)
                      if len(interior_idx) else np.full(3, np.nan))
        print(f"[conv_cell] a={a:.4f}  bbox={bbox_min.tolist()} → {bbox_max.tolist()}")
        tmpl_extra = ""
        if cell_stats.get("fcc_base_nodes") is not None:
            tmpl_extra = (
                f"  [fcc_base={cell_stats['fcc_base_nodes']}"
                f"  enh1_fine={cell_stats['fcc_enh1_atom_refine_added']}"
                f"  enh2_rand={cell_stats['fcc_enh2_adaptive_random_added']}]"
            )
        print(f"[conv_cell]   cell template: {cell_stats['cell_after_filter']} nodes "
              f"(skeleton={cell_stats['cell_skeleton']}, random={cell_stats['cell_random']}, "
              f"parity={cell_stats['cell_accepted_parity']}){tmpl_extra}")
        print(f"[conv_cell]   tile shifts: {tile_range[0]}×{tile_range[1]}×{tile_range[2]} "
              f"→ {len(nodes)} tiled nodes")
        n_fcc_local = int(len(groups.get("fcc_local", [])))
        if n_fcc_local > 0:
            print(f"[conv_cell]   fcc_local (enh1) tiled: {n_fcc_local} nodes")
        print(f"[conv_cell]   all-node bbox     : {np.round(a_bbox_min, 3).tolist()} → "
              f"{np.round(a_bbox_max, 3).tolist()}")
        if domain_shape == "cube":
            print(f"[conv_cell]   margin={margin:.3f} Bohr  (frac={boundary_margin_frac})")
        else:
            n_ico = int(len(groups.get("icosphere", [])))
            print(f"[conv_cell]   sphere center      : {np.round(sphere_center, 3).tolist()}")
            print(f"[conv_cell]   sphere radius      : {sphere_radius_eff:.3f} Bohr")
            print(f"[conv_cell]   sphere subdivide   : {sphere_subdivide}  "
                  f"(icosphere boundary nodes: {n_ico})")
        print(f"[conv_cell]   interior / boundary = "
              f"{len(interior_idx)} / {len(boundary_idx)}")
        if len(interior_idx):
            print(f"[conv_cell]   interior bbox     : {np.round(i_bbox_min, 3).tolist()} → "
                  f"{np.round(i_bbox_max, 3).tolist()}")
    return nodes, groups, stats
