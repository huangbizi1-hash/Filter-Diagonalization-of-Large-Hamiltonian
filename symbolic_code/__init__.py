"""symbolic_code – modular symbolic Hamiltonian filter pipeline.

Workflow
--------
1. space_partition  : divide 3-D space into cubes, build V(x,y,z) per cube
2. expr_loader      : load per-cube potential pickles (CubicExpressionManager)
3. h_powers         : recursively compute (aH+b)^n * psi per cube
4. chebyshev_filter : assemble f(H)*psi via Chebyshev expansion
5. filter_plot      : visualise |T_m(aE+b)| before running the full pipeline
6. julia_codegen    : convert sympy expressions -> Julia source for fast eval
"""

from .space_partition import CubicSpacePartition, read_cube_atoms, load_cube_expression
from .expr_loader import CubicExpressionManager
from .h_powers import (
    laplacian,
    directional_derivative,
    apply_H_on_pair,
    generate_scaled_H_powers,
    process_all_cubes,
    process_specific_cubes,
)
from .chebyshev_filter import (
    load_expr_srepr,
    load_H_powers,
    chebyshev_coeffs_transformed,
    apply_f_of_H_on_psi,
    group_by_exp_combined,
    extract_cos_sin_coeffs,
    apply_horner,
    svd_H,
)
from .filter_plot import filter_response, plot_chebyshev_filter, print_filter_summary

__all__ = [
    # space partition
    "CubicSpacePartition", "read_cube_atoms", "load_cube_expression",
    # expression loader
    "CubicExpressionManager",
    # H^n generation
    "laplacian", "directional_derivative", "apply_H_on_pair",
    "generate_scaled_H_powers", "process_all_cubes", "process_specific_cubes",
    # Chebyshev filter
    "load_expr_srepr", "load_H_powers", "chebyshev_coeffs_transformed",
    "apply_f_of_H_on_psi", "group_by_exp_combined",
    "extract_cos_sin_coeffs", "apply_horner", "svd_H",
    # filter plotting
    "filter_response", "plot_chebyshev_filter", "print_filter_summary",
]
