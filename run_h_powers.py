"""Entry point – Step 2: generate symbolic H^n * psi per spatial cube.

Loads the per-cube potential expressions produced by run_space_partition.py
and recursively applies H = -0.5*∇² + V to compute power files for each cube.

Two strategies (--method)
--------------------------
H_powers (default)
    Computes pure H^n without energy-window scaling.  Files named
    ``H_power_{n}.pkl`` and can be reused for any energy window.  E_lower /
    E_upper are NOT required for this mode.
scaled
    Computes (aH+b)^n with a = 2/(E_upper-E_lower), b = -(E_upper+E_lower)/
    (E_upper-E_lower) baked into the expressions.  Legacy behaviour.

Usage
-----
# Recommended: generate reusable H^n files (no energy window needed)
python run_h_powers.py \\
    --indir  cubic_space_partition_rcut5 \\
    --outdir H_powers_partitioned \\
    --N 6

# Legacy: (aH+b)^n for a fixed energy window
python run_h_powers.py \\
    --indir  cubic_space_partition_rcut5 \\
    --outdir H_powers_partitioned \\
    --N 6 --method scaled --E_lower -1.0 --E_upper 1.0
"""

import argparse
from symbolic_code import CubicExpressionManager, process_all_cubes


def main():
    p = argparse.ArgumentParser(
        description='Generate symbolic H^n * psi for each spatial cube.'
    )
    p.add_argument('--indir',   required=True,
                   help='Directory of per-cube potential pickles')
    p.add_argument('--outdir',  required=True,
                   help='Output root for H^n expression files')
    p.add_argument('--N',       type=int,   default=6,
                   help='Maximum power (default 6)')
    p.add_argument('--method',  default='H_powers', choices=['H_powers', 'scaled'],
                   help='H_powers=pure H^n reusable (default); '
                        'scaled=(aH+b)^n with energy window baked in')
    p.add_argument('--E_lower', type=float, default=None,
                   help='Lower energy bound (required for --method scaled)')
    p.add_argument('--E_upper', type=float, default=None,
                   help='Upper energy bound (required for --method scaled)')
    p.add_argument('--fmt',     default='pkl', choices=['pkl', 'sym.gz'],
                   help='Output file format (default pkl)')
    args = p.parse_args()

    if args.method == 'scaled':
        if args.E_lower is None or args.E_upper is None:
            p.error('--method scaled requires --E_lower and --E_upper')
        a =  2.0 / (args.E_upper - args.E_lower)
        b = -(args.E_upper + args.E_lower) / (args.E_upper - args.E_lower)
        print(f'Scaling: a={a:.4f}, b={b:.4f}  '
              f'(E in [{args.E_lower}, {args.E_upper}])')
    else:
        a, b = None, None
        print('Method: H_powers (no energy-window scaling; reusable files)')

    manager = CubicExpressionManager(args.indir)
    print(f'Loaded {len(manager)} cubes from {args.indir}')

    process_all_cubes(
        manager=manager,
        N=args.N,
        a=a,
        b=b,
        base_outdir=args.outdir,
        file_format=args.fmt,
        method='raw' if args.method == 'H_powers' else 'scaled',
    )


if __name__ == '__main__':
    main()
