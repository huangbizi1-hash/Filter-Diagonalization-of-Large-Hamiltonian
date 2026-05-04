"""rbf_code.io_qd — Gaussian cube I/O and QD problem builder.

Exposes:
    read_cube_file    — parse .cube file into coordinate arrays + potential grid
    read_cube_atoms   — parse atom positions / atomic numbers from .cube file
    build_qd_problem  — high-level entry point: cube file → RBFProblem
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from rbf.pde.fd import weight_matrix

from rbf_code.config import Array, RBFConfig, RBFProblem, _raise_on_nonfinite
from rbf_code.nodes import (
    generate_sphere_nodes,
    generate_atom_augmented_nodes,
    generate_conv_cell_nodes,
)
from rbf_code.laplacian import (
    weight_matrix_ball_fingerprint,
    weight_matrix_conv_cell_reuse,
)


def read_cube_file(path: str) -> Tuple[Array, Array, Array, Array]:
    """
    Read a Gaussian cube file (orthogonal grid assumed).

    Returns
    -------
    x, y, z  : 1-D coordinate arrays (Bohr)
    potential : shape (nx, ny, nz) in Hartree
    """
    with open(path, "r") as f:
        lines = f.readlines()
    n_atoms = abs(int(lines[2].split()[0]))
    ox, oy, oz = (float(v) for v in lines[2].split()[1:4])
    nx, dx = int(lines[3].split()[0]), float(lines[3].split()[1])
    ny, dy = int(lines[4].split()[0]), float(lines[4].split()[2])
    nz, dz = int(lines[5].split()[0]), float(lines[5].split()[3])
    data_start = 6 + n_atoms
    vals: list[float] = []
    for line in lines[data_start:]:
        vals.extend(map(float, line.split()))
    potential = np.array(vals[: nx * ny * nz], dtype=np.float64).reshape(nx, ny, nz)
    x = ox + np.arange(nx) * dx
    y = oy + np.arange(ny) * dy
    z = oz + np.arange(nz) * dz
    return x, y, z, potential


def read_cube_atoms(path: str) -> Tuple[Array, Array]:
    """
    Read atom positions (and atomic numbers) from a Gaussian cube file.

    Returns
    -------
    positions      : (n_atoms, 3) in Bohr (same units as the cube grid)
    atomic_numbers : (n_atoms,) int
    """
    with open(path, "r") as f:
        lines = f.readlines()
    n_atoms = abs(int(lines[2].split()[0]))
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    numbers   = np.zeros(n_atoms, dtype=np.int64)
    for i in range(n_atoms):
        parts = lines[6 + i].split()
        numbers[i]   = int(parts[0])
        positions[i] = [float(parts[2]), float(parts[3]), float(parts[4])]
    return positions, numbers


def build_qd_problem(
    cube_file: str,
    domain: str = "cube",
    spacing: float = 0.5,
    R: float = 20.0,
    stencil_size: int = 80,
    phi: str = "phs3",
    eps: float = 0.5,
    order: int = 2,
    sphere_subdivide: int = 3,
    augment: str = "poisson_disc",
    exclude_radius: float = 0.0,
    v_clip_percentile: float = 99.9,
    # conv_cell-only knobs
    conv_cell_a: float = 11.4523,
    conv_cell_d_min_frac: float = 0.06,
    conv_cell_n_random: int = 120,
    conv_cell_seed: int = 42,
    conv_cell_parity: bool = True,
    conv_cell_template_mode: str = "hybrid",
    conv_cell_fcc_scale_factor: int = 8,
    conv_cell_fcc_origin_frac: Optional[Array] = None,
    conv_cell_fcc_atom_refine_factor: int = 0,
    conv_cell_fcc_atom_radius_frac: float = 0.0,
    conv_cell_boundary_margin_frac: float = 0.06,
    conv_cell_use_rbf_poisson: bool = True,
    conv_cell_domain_shape: str = "cube",
    conv_cell_sphere_radius: Optional[float] = None,
    conv_cell_sphere_subdivide: int = 3,
    conv_cell_adaptive_random: bool = False,
    conv_cell_adaptive_grid_n: int = 36,
    conv_cell_adaptive_lambda_grad: float = 0.0,
    conv_cell_adaptive_lambda_lap: float = 0.0,
    conv_cell_adaptive_candidate_multiplier: float = 8.0,
    conv_cell_include_skeleton: bool = True,
    conv_cell_reuse_weights: bool = True,
    stencil_radius: float = 0.0,
    stencil_fingerprint_tol: float = 1e-4,
    stencil_inner_radius: float = 0.0,
    stencil_max_neighbors: int = 0,
    stencil_select_near_first: bool = True,
    include_interior: bool = True,
    include_boundary: bool = True,
    node_min_dist: float = 0.0,
    v_source: str = "grid_interp",
    gaussian_params_file: Optional[str] = None,
    r_cut: float = 7.0,
) -> RBFProblem:
    """
    Build RBFProblem with QD potential from a Gaussian cube file.

    domain='cube'     — exact regular grid from the cube file
    domain='sphere'   — Poisson disc nodes inside a sphere of radius R
    domain='atoms'    — nodes at atom positions, optionally augmented
    domain='conv_cell'— hybrid conventional-cell tiled template (zincblende)
    """
    from scipy.interpolate import RegularGridInterpolator

    x_grid, y_grid, z_grid, pot_3d = read_cube_file(cube_file)
    interp_fn = RegularGridInterpolator(
        (x_grid, y_grid, z_grid), pot_3d,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    if domain == "cube":
        nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        i_idx, j_idx, k_idx = np.unravel_index(np.arange(len(nodes)), (nx, ny, nz))
        on_boundary = (
            (i_idx == 0) | (i_idx == nx - 1) |
            (j_idx == 0) | (j_idx == ny - 1) |
            (k_idx == 0) | (k_idx == nz - 1)
        )
        interior_idx = np.where(~on_boundary)[0]
        groups = {
            "interior": interior_idx,
            "boundary": np.where(on_boundary)[0],
        }
        if (not include_interior) or (not include_boundary):
            keep_mask = np.zeros(len(nodes), dtype=bool)
            if include_interior:
                keep_mask[groups["interior"]] = True
            if include_boundary:
                keep_mask[groups["boundary"]] = True
            keep = np.where(keep_mask)[0].astype(np.int64)
            remap = -np.ones(len(nodes), dtype=np.int64)
            remap[keep] = np.arange(len(keep), dtype=np.int64)
            groups["interior"] = remap[groups["interior"]]
            groups["interior"] = groups["interior"][groups["interior"] >= 0]
            groups["boundary"] = remap[groups["boundary"]]
            groups["boundary"] = groups["boundary"][groups["boundary"] >= 0]
            nodes = nodes[keep]
        cfg_L = float(np.max(np.abs(nodes)))

    elif domain == "sphere":
        nodes, groups = generate_sphere_nodes(
            spacing, R, sphere_subdivide, min_dist=node_min_dist)
        interior_idx = groups["interior"]
        cfg_L = R

    elif domain == "atoms":
        atom_pos, _ = read_cube_atoms(cube_file)
        nodes, groups = generate_atom_augmented_nodes(
            atom_positions=atom_pos,
            domain="sphere",
            R=R,
            spacing=spacing,
            augment=augment,
            exclude_radius=exclude_radius,
            sphere_subdivide=sphere_subdivide,
            include_interior=include_interior,
            include_boundary=include_boundary,
            min_dist=node_min_dist,
        )
        interior_idx = groups["interior"]
        cfg_L = R

    elif domain == "conv_cell":
        bbox_min = np.array([x_grid[0], y_grid[0], z_grid[0]], dtype=np.float64)
        bbox_max = np.array([x_grid[-1] + (x_grid[1] - x_grid[0]),
                              y_grid[-1] + (y_grid[1] - y_grid[0]),
                              z_grid[-1] + (z_grid[1] - z_grid[0])],
                             dtype=np.float64)
        adaptive_builder = None
        if conv_cell_adaptive_random:
            if gaussian_params_file is None:
                raise ValueError(
                    "conv_cell_adaptive_random=True requires gaussian_params_file "
                    "(to evaluate one-cell dense-grid potential).")
            from gaussian_potential_builder import GaussianPotentialBuilder
            adaptive_builder = GaussianPotentialBuilder(
                cube_file=cube_file,
                params_file=gaussian_params_file,
                r_cut=r_cut,
            )
        nodes, groups, _cell_stats = generate_conv_cell_nodes(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            a=conv_cell_a,
            d_min_frac=conv_cell_d_min_frac,
            n_random_target=conv_cell_n_random,
            seed=conv_cell_seed,
            include_parity=conv_cell_parity,
            template_mode=conv_cell_template_mode,
            fcc_scale_factor=conv_cell_fcc_scale_factor,
            fcc_origin_frac=conv_cell_fcc_origin_frac,
            fcc_atom_refine_factor=conv_cell_fcc_atom_refine_factor,
            fcc_atom_radius_frac=conv_cell_fcc_atom_radius_frac,
            boundary_margin_frac=conv_cell_boundary_margin_frac,
            use_rbf_poisson=conv_cell_use_rbf_poisson,
            domain_shape=conv_cell_domain_shape,
            sphere_radius=conv_cell_sphere_radius,
            sphere_subdivide=conv_cell_sphere_subdivide,
            adaptive_random=conv_cell_adaptive_random,
            adaptive_grid_n=conv_cell_adaptive_grid_n,
            adaptive_lambda_grad=conv_cell_adaptive_lambda_grad,
            adaptive_lambda_lap=conv_cell_adaptive_lambda_lap,
            adaptive_candidate_multiplier=conv_cell_adaptive_candidate_multiplier,
            adaptive_gaussian_builder=adaptive_builder,
            include_skeleton=conv_cell_include_skeleton,
            include_interior=include_interior,
            include_boundary=include_boundary,
            min_dist_cart=node_min_dist,
        )
        groups["conv_cell_template_nodes_frac"] = _cell_stats["cell_template_nodes_frac"]
        groups["conv_cell_template_nodes_cart"] = _cell_stats["cell_template_nodes_cart"]
        groups["conv_cell_template_roles"]      = _cell_stats["cell_template_roles"]
        groups["conv_cell_step_cell_template_nodes_cart"]    = _cell_stats.get("step_cell_template_nodes_cart")
        groups["conv_cell_step_tiled_raw_nodes_cart"]        = _cell_stats.get("step_tiled_raw_nodes_cart")
        groups["conv_cell_step_tiled_raw_roles"]             = _cell_stats.get("step_tiled_raw_roles")
        groups["conv_cell_step_after_domain_nodes_cart"]     = _cell_stats.get("step_after_domain_nodes_cart")
        groups["conv_cell_step_after_domain_roles"]          = _cell_stats.get("step_after_domain_roles")
        groups["conv_cell_step_after_close_filter_nodes_cart"] = _cell_stats.get("step_after_close_filter_nodes_cart")
        groups["conv_cell_step_timings_seconds"]             = _cell_stats.get("timings_seconds")
        groups["_enh1_fcc_base"] = _cell_stats.get("fcc_base_nodes", 0)
        groups["_enh1_added"]    = _cell_stats.get("fcc_enh1_atom_refine_added", 0)
        groups["_enh2_added"]    = _cell_stats.get("fcc_enh2_adaptive_random_added", 0)
        interior_idx = groups["interior"]
        cfg_L = float(np.max(np.abs(nodes))) if len(nodes) else 0.0

    else:
        raise ValueError(
            "domain must be 'cube', 'sphere', 'atoms', or 'conv_cell', "
            f"got {domain!r}")

    # ── QD potential on interior nodes ─────────────────────────────────────
    if v_source == "gaussian_direct":
        if gaussian_params_file is None:
            raise ValueError(
                "v_source='gaussian_direct' requires gaussian_params_file")
        from gaussian_potential_builder import GaussianPotentialBuilder
        builder = GaussianPotentialBuilder(
            cube_file=cube_file,
            params_file=gaussian_params_file,
            r_cut=r_cut,
        )
        V_nodes = builder.evaluate_at_points(nodes[interior_idx])
    elif v_source == "grid_interp":
        V_nodes = interp_fn(nodes[interior_idx]).astype(np.float64)
    else:
        raise ValueError(
            "v_source must be 'grid_interp' or 'gaussian_direct', "
            f"got {v_source!r}")

    v_cap   = float(np.percentile(V_nodes, v_clip_percentile))
    V_nodes = np.clip(V_nodes, None, v_cap)
    _raise_on_nonfinite("V_nodes (after clip)", V_nodes)

    _lap_diffs = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    if stencil_radius > 0.0:
        laplacian_matrix, _ball_stats = weight_matrix_ball_fingerprint(
            x=nodes[interior_idx],
            p=nodes,
            r=float(stencil_radius),
            diffs=_lap_diffs,
            phi=phi,
            eps=eps,
            order=order,
            fingerprint_tol=float(stencil_fingerprint_tol),
            inner_radius=float(stencil_inner_radius),
            max_neighbors=int(stencil_max_neighbors),
            select_near_first=bool(stencil_select_near_first),
            verbose=True,
        )
        if domain == "conv_cell":
            groups["_ball_stencil_stats"] = _ball_stats
    elif domain == "conv_cell" and conv_cell_reuse_weights:
        laplacian_matrix = weight_matrix_conv_cell_reuse(
            x=nodes[interior_idx],
            p=nodes,
            n=stencil_size,
            diffs=_lap_diffs,
            phi=phi,
            eps=eps,
            order=order,
        )
    else:
        laplacian_matrix = weight_matrix(
            x=nodes[interior_idx],
            p=nodes,
            n=stencil_size,
            diffs=_lap_diffs,
            phi=phi,
            eps=eps,
            order=order,
        )
    _raise_on_nonfinite("laplacian_matrix.data", laplacian_matrix.data)

    cfg = RBFConfig(
        spacing=spacing, L=cfg_L,
        stencil_size=stencil_size, phi=phi, eps=eps, order=order,
    )
    return RBFProblem(
        config=cfg,
        nodes=nodes,
        groups=groups,
        interior_idx=interior_idx,
        laplacian_matrix=laplacian_matrix,
        psi_interp_matrix=None,
        lap_interp_matrix=None,
        grid_points=None,
        grid_shape=None,
        V_nodes=V_nodes,
    )
