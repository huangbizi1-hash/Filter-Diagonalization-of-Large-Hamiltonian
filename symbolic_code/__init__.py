"""symbolic_code – modular symbolic Hamiltonian filter pipeline.

Workflow
--------
1. space_partition  : divide 3-D space into cubes, build V(x,y,z) per cube
2. expr_loader      : load per-cube potential pickles (CubicExpressionManager)
3. h_powers         : compute H^n (or (aH+b)^n) * psi per cube
4. chebyshev_filter : assemble f(H)*psi via Chebyshev expansion
5. filter_plot      : visualise |T_m(aE+b)| before running the full pipeline
6. julia_codegen    : convert sympy expressions → Julia source for fast eval

Recommended H-power strategy
-----------------------------
Use ``generate_H_powers`` + ``apply_f_of_H_from_raw_powers``:
  - H^n files have rational coefficients → compact expressions, fast sympy ops
  - Same files reusable for any energy window (a, b chosen at assembly time)
Legacy: ``generate_scaled_H_powers`` + ``apply_f_of_H_on_psi`` (a/b baked in).
"""

from .space_partition import CubicSpacePartition, read_cube_atoms, load_cube_expression
from .expr_loader import CubicExpressionManager
from .h_powers import (
    laplacian,
    directional_derivative,
    apply_H_on_pair,
    generate_H_powers,
    extend_H_powers_from_cache,
    generate_scaled_H_powers,
    process_all_cubes,
    process_specific_cubes,
)
from .chebyshev_filter import (
    load_expr_srepr,
    load_H_powers,
    load_H_raw_powers,
    chebyshev_coeffs_transformed,
    apply_f_of_H_from_raw_powers,
    apply_f_of_H_on_psi,
    group_by_exp_combined,
    extract_cos_sin_coeffs,
    apply_horner,
    svd_H,
)
from .filter_plot import filter_response, plot_chebyshev_filter, print_filter_summary
from .julia_codegen import (
    expr_to_julia_code,
    build_julia_scalar_function,
    build_julia_batch_script,
    build_julia_benchmark_script,
)

__all__ = [
    # space partition
    "CubicSpacePartition", "read_cube_atoms", "load_cube_expression",
    # expression loader
    "CubicExpressionManager",
    # H-power generation (both strategies)
    "laplacian", "directional_derivative", "apply_H_on_pair",
    "generate_H_powers", "extend_H_powers_from_cache", "generate_scaled_H_powers",
    "process_all_cubes", "process_specific_cubes",
    # Chebyshev filter assembly
    "load_expr_srepr", "load_H_powers", "load_H_raw_powers",
    "chebyshev_coeffs_transformed",
    "apply_f_of_H_from_raw_powers", "apply_f_of_H_on_psi",
    "group_by_exp_combined", "extract_cos_sin_coeffs", "apply_horner", "svd_H",
    # filter plotting
    "filter_response", "plot_chebyshev_filter", "print_filter_summary",
    # Julia code generation
    "expr_to_julia_code", "build_julia_scalar_function",
    "build_julia_batch_script", "build_julia_benchmark_script",
]
