"""Entry point – Step 2: generate symbolic (aH+b)^n * psi per spatial cube.

Loads the per-cube potential expressions produced by run_space_partition.py
and recursively applies H = -0.5*∇² + V to compute H^n * psi for
n = 1 … N, saving each result as a compressed .sym.gz (or .pkl) file.

Usage
-----
python run_h_powers.py \\
    --indir  cubic_space_partition_rcut5 \\
    --outdir H_powers_partitioned \\
    [--E_lower -1.0] [--E_upper 1.0] [--N 6] [--fmt sym.gz]

The scaling parameters are computed automatically:
    a = 2 / (E_upper - E_lower)
    b = -(E_upper + E_lower) / (E_upper - E_lower)
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
    p.add_argument('--E_lower', type=float, default=-1.0,
                   help='Lower energy bound for Chebyshev rescaling')
    p.add_argument('--E_upper', type=float, default=1.0,
                   help='Upper energy bound for Chebyshev rescaling')
    p.add_argument('--N',       type=int,   default=6,
                   help='Maximum Chebyshev power (default 6)')
    p.add_argument('--fmt',     default='sym.gz', choices=['sym.gz', 'pkl'],
                   help='Output file format (default sym.gz)')
    args = p.parse_args()

    a =  2.0 / (args.E_upper - args.E_lower)
    b = -(args.E_upper + args.E_lower) / (args.E_upper - args.E_lower)

    manager = CubicExpressionManager(args.indir)
    print(f'Loaded {len(manager)} cubes from {args.indir}')
    print(f'Scaling: a={a:.4f}, b={b:.4f}  (E in [{args.E_lower}, {args.E_upper}])')

    process_all_cubes(
        manager=manager,
        N=args.N,
        a=a,
        b=b,
        base_outdir=args.outdir,
        file_format=args.fmt,
    )


if __name__ == '__main__':
    main()
