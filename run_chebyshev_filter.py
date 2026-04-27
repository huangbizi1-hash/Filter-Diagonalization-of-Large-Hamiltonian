"""Entry point – Step 3: assemble f(H)*psi and export Julia code.

Loads H^n*psi files for a single cube subdirectory, assembles the
Chebyshev-filtered state f(H)*psi, groups terms by Gaussian factor,
applies Horner form, and optionally writes a Julia scalar function ready
for compilation and benchmarking.

Usage
-----
python run_chebyshev_filter.py \\
    --indir  H_powers_partitioned/cube_2.000_2.000_2.000 \\
    [--E_lower 0.0] [--E_upper 20.0] [--N 5] \\
    [--julia-out eval_fH.jl] [--benchmark]

The --indir must point to a single cube subdirectory that contains
H_scaled_power_n.sym.gz files produced by run_h_powers.py.
"""

import argparse
from symbolic_code.chebyshev_filter import (
    chebyshev_coeffs_transformed,
    apply_f_of_H_on_psi,
    extract_cos_sin_coeffs,
    group_by_exp_combined,
    apply_horner,
)
from symbolic_code.julia_codegen import (
    build_julia_scalar_function,
    build_julia_benchmark_script,
)


def main():
    p = argparse.ArgumentParser(
        description='Assemble Chebyshev f(H)*psi and optionally export Julia code.'
    )
    p.add_argument('--indir',     required=True,
                   help='Cube subdirectory with H_scaled_power_n.sym.gz files')
    p.add_argument('--E_lower',   type=float, default=0.0)
    p.add_argument('--E_upper',   type=float, default=20.0)
    p.add_argument('--N',         type=int,   default=5,
                   help='Chebyshev expansion order')
    p.add_argument('--fmt',       default='sym.gz', choices=['sym.gz', 'pkl'])
    p.add_argument('--julia-out', default=None,
                   help='If given, write Julia scalar function to this .jl file')
    p.add_argument('--benchmark', action='store_true',
                   help='Wrap Julia function in a benchmark harness')
    args = p.parse_args()

    # --- Chebyshev scaling ---
    a =  2.0 / (args.E_upper - args.E_lower)
    b = -(args.E_upper + args.E_lower) / (args.E_upper - args.E_lower)
    coeffs = chebyshev_coeffs_transformed(args.N, a=a, b=b)
    print(f'Chebyshev order N={args.N}  E=[{args.E_lower}, {args.E_upper}]')
    print(f'Coefficients: {coeffs}')

    # --- Assemble f(H)*psi ---
    psi_fH = apply_f_of_H_on_psi(args.indir, coeffs, args.N,
                                  file_type=args.fmt)
    expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)

    terms_cos = apply_horner(group_by_exp_combined(expr_cos))
    terms_sin = apply_horner(group_by_exp_combined(expr_sin))
    print(f'cos-phase terms: {len(terms_cos)}')
    print(f'sin-phase terms: {len(terms_sin)}')

    # --- Optional Julia export ---
    if args.julia_out:
        jl_func = build_julia_scalar_function(terms_cos, terms_sin)
        if args.benchmark:
            src = build_julia_benchmark_script(jl_func)
        else:
            src = jl_func
        with open(args.julia_out, 'w', encoding='utf-8') as f:
            f.write(src)
        print(f'Julia code written to {args.julia_out}')


if __name__ == '__main__':
    main()
