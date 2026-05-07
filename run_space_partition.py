"""Entry point – Step 1: generate per-cube symbolic potential expressions.

Divides 3-D space into n_divisions^3 cubes, builds a symbolic Gaussian
potential V(x,y,z) for each cube from nearby atoms, and saves the
expressions as pickle files for later H^n computation.

Usage
-----
python run_space_partition.py \\
    --cube   localPot.cube \\
    --params gaussian_fitting/fitting_params.json \\
    --outdir cubic_space_partition_rcut5 \\
    [--L_s 20.0] [--n_divisions 10] [--r_cut 5.0]
"""

import argparse
from symbolic_code import CubicSpacePartition, read_cube_atoms


def main():
    p = argparse.ArgumentParser(
        description='Generate per-cube symbolic potential expressions.'
    )
    p.add_argument('--cube',        required=True,
                   help='.cube file with atomic structure')
    p.add_argument('--params',      required=True,
                   help='Gaussian fitting params JSON')
    p.add_argument('--outdir',      required=True,
                   help='Output directory for cube expression pickles')
    p.add_argument('--L_s',         type=float, default=20.0,
                   help='Space half-size in Bohr (default 20.0)')
    p.add_argument('--n_divisions', type=int,   default=10,
                   help='Divisions per dimension (default 10)')
    p.add_argument('--r_cut',       type=float, default=5.0,
                   help='Atom cutoff radius in Bohr (default 5.0)')
    args = p.parse_args()

    atoms = read_cube_atoms(args.cube)
    print(f'Read {len(atoms)} atoms from {args.cube}')

    partition = CubicSpacePartition(
        params_file=args.params,
        atoms=atoms,
        L_s=args.L_s,
        n_divisions=args.n_divisions,
        r_cut=args.r_cut,
    )
    partition.process_all_cubes(args.outdir)


if __name__ == '__main__':
    main()
