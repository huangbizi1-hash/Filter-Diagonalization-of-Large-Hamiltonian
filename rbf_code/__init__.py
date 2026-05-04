"""rbf_code — modular RBF-FD solver for quantum Hamiltonians.

All public names are re-exported here so callers can either import from
`rbf_code` directly or from the individual sub-modules.
"""
from rbf_code.config import (
    Array,
    _raise_on_nonfinite,
    _periodic_delta_frac,
    KERNELS_ALL,
    KERNEL_GROUPS,
    RBFConfig,
    RBFProblem,
    IterationRecord,
)
from rbf_code.periodic import (
    _wrap_frac,
    _unique_rows_mod1,
    _unique_cart_rows,
    _periodic_diff,
    _periodic_dist,
    _build_uniform_frac_grid,
    _estimate_grad_laplacian_uniform,
    _adaptive_accept_and_filter_periodic,
    _greedy_filter_by_dmin_periodic,
    _make_unit_cube_surface,
)
from rbf_code.nodes import (
    _CONV_CELL_IN_FRAC_DEFAULT,
    _CONV_CELL_AS_FRAC_DEFAULT,
    _FCC_OFFSETS_FRAC,
    _LEVEL3_BASE_FRAC_DEFAULT,
    generate_nodes,
    make_grid_points,
    _make_icosphere,
    generate_sphere_nodes,
    _filter_close_points,
    generate_atom_augmented_nodes,
    _generate_shifted_refined_fcc_frac,
    _poisson_like_periodic,
    generate_conv_cell_nodes,
)
from rbf_code.laplacian import (
    relative_laplacian_error,
    build_hamiltonian_matrix,
    compute_node_quality,
    weight_matrix_conv_cell_reuse,
    weight_matrix_ball_fingerprint,
)
from rbf_code.io_qd import (
    read_cube_file,
    read_cube_atoms,
    build_qd_problem,
)
from rbf_code.eigensolve import (
    build_problem,
    solve_lowest_eigenvalues,
    _ho_exact_levels,
    sweep_rbf_kernels,
    sweep_stencil_eps,
    iterate_hamiltonian,
)

__all__ = [
    # config
    "Array", "_raise_on_nonfinite", "_periodic_delta_frac",
    "KERNELS_ALL", "KERNEL_GROUPS",
    "RBFConfig", "RBFProblem", "IterationRecord",
    # periodic
    "_wrap_frac", "_unique_rows_mod1", "_unique_cart_rows",
    "_periodic_diff", "_periodic_dist",
    "_build_uniform_frac_grid", "_estimate_grad_laplacian_uniform",
    "_adaptive_accept_and_filter_periodic", "_greedy_filter_by_dmin_periodic",
    "_make_unit_cube_surface",
    # nodes
    "_CONV_CELL_IN_FRAC_DEFAULT", "_CONV_CELL_AS_FRAC_DEFAULT",
    "_FCC_OFFSETS_FRAC", "_LEVEL3_BASE_FRAC_DEFAULT",
    "generate_nodes", "make_grid_points",
    "_make_icosphere", "generate_sphere_nodes",
    "_filter_close_points", "generate_atom_augmented_nodes",
    "_generate_shifted_refined_fcc_frac", "_poisson_like_periodic",
    "generate_conv_cell_nodes",
    # laplacian
    "relative_laplacian_error", "build_hamiltonian_matrix",
    "compute_node_quality",
    "weight_matrix_conv_cell_reuse", "weight_matrix_ball_fingerprint",
    # io_qd
    "read_cube_file", "read_cube_atoms", "build_qd_problem",
    # eigensolve
    "build_problem", "solve_lowest_eigenvalues",
    "_ho_exact_levels", "sweep_rbf_kernels", "sweep_stencil_eps",
    "iterate_hamiltonian",
]
