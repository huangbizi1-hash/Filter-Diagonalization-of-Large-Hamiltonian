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

Continuation (--start_power)
-----------------------------
If H^0 … H^{start_power} already exist in the output directory, pass
``--start_power K`` to load H^K from disk and compute only H^{K+1} … H^N.
This avoids recomputing powers that are already saved.

Single-cube mode (--cube_dirname)
----------------------------------
Pass ``--cube_dirname cube_neg15.944_neg11.389_6.833`` to process only that
one cube subdirectory instead of all cubes in the manager.

Usage
-----
# Full run for all cubes
python run_h_powers.py \\
    --indir  cubic_space_partition_rcut5 \\
    --outdir H_powers_partitioned \\
    --N 8

# Continue from H^8 to H^12 for a specific cube
python run_h_powers.py \\
    --indir  QD_R17_Vexpr_partition_custom_l3p0_rcut4p2 \\
    --outdir QD_R17_Julia_exp/custom_l3p0_rcut4p2 \\
    --method H_powers --fmt pkl \\
    --cube_dirname cube_neg15.944_neg11.389_6.833 \\
    --start_power 8 --N 12
"""

import argparse
import pickle
from pathlib import Path

from symbolic_code import CubicExpressionManager, process_all_cubes
from symbolic_code.h_powers import (
    apply_H_on_pair, _save_power, _cube_dirname,
)
import sympy as sp


# ---------------------------------------------------------------------------
# Continuation helper
# ---------------------------------------------------------------------------

def generate_H_powers_continued(
    start_power, N, outdir,
    file_format='pkl',
    expand=True,
    V=None, kvec=None, k2=None, pref=0.5,
    x=None, y=None, z=None,
):
    """Load H^{start_power} from disk and compute H^{start_power+1} … H^N.

    Parameters
    ----------
    start_power : int
        The power whose pkl already exists.  We load H^{start_power} as the
        initial (Ps, Pc) state and compute the next powers.
    N : int
        Maximum power to generate (inclusive).
    outdir : str or Path
        Directory containing H_power_{start_power}.pkl; new files are written
        to the same directory.
    file_format, expand, V, kvec, k2, pref, x, y, z
        Same as generate_H_powers.

    Returns
    -------
    dict  n -> {'Ps': expr, 'Pc': expr}  (only newly generated powers)
    """
    outdir = Path(outdir)
    pkl_path = outdir / f'H_power_{start_power}.pkl'
    if not pkl_path.exists():
        raise FileNotFoundError(
            f'H_power_{start_power}.pkl not found in {outdir}.\n'
            f'Run without --start_power to generate powers from scratch, '
            f'or check that H_power_{start_power}.pkl is present.'
        )

    print(f'  Loading H^{start_power} from {pkl_path.name} ...', end=' ', flush=True)
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)
    Ps = data.get('Ps', sp.Integer(1))
    Pc = data.get('Pc', sp.Integer(0))
    print('done')

    if N <= start_power:
        print(f'  N={N} <= start_power={start_power}: nothing to compute.')
        return {}

    results = {}
    for n in range(start_power + 1, N + 1):
        print(f'    H^{n} ...', end=' ', flush=True)
        Ps_H, Pc_H = apply_H_on_pair(Ps, Pc, V, kvec, k2, pref, x, y, z)
        if expand:
            Ps = sp.expand(Ps_H)
            Pc = sp.expand(Pc_H)
        else:
            Ps = Ps_H
            Pc = Pc_H
        results[n] = {'Ps': Ps, 'Pc': Pc}
        _save_power(outdir, n, results[n], file_format, prefix='H_power')
        print('✓')

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Generate symbolic H^n * psi for each spatial cube.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    p.add_argument('--start_power', type=int, default=None,
                   help='If set, load H^start_power from outdir and compute '
                        'H^{start_power+1} … H^N.  H^0…H^start_power must '
                        'already exist in the cube output directory.')
    p.add_argument('--cube_dirname', default=None,
                   help='If set, process only this specific cube subdirectory '
                        '(e.g. cube_neg15.944_neg11.389_6.833) instead of '
                        'all cubes in the manager.')
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

    if args.start_power is not None and args.method == 'scaled':
        p.error('--start_power is only supported with --method H_powers')

    manager = CubicExpressionManager(args.indir)
    print(f'Loaded {len(manager)} cubes from {args.indir}')

    x,  y,  z  = sp.symbols('x y z')
    kx, ky, kz = sp.symbols('kx ky kz')
    kvec = (kx, ky, kz)
    k2   = kx**2 + ky**2 + kz**2
    pref = 0.5

    # ------------------------------------------------------------------ #
    # Single-cube continuation mode                                        #
    # ------------------------------------------------------------------ #
    if args.cube_dirname is not None:
        # Find the matching cube in the manager
        found = None
        for idx, info in manager.cube_info.items():
            cx, cy, cz = info['center']
            dname = _cube_dirname(cx, cy, cz)
            if dname == args.cube_dirname:
                found = (idx, info)
                break

        if found is None:
            p.error(f'Cube directory "{args.cube_dirname}" not found in manager '
                    f'(indir={args.indir}).')

        idx, info = found
        V_expr = manager.get_expression(*idx)
        if V_expr is None:
            p.error(f'No potential expression for cube {args.cube_dirname}.')

        cube_dir = Path(args.outdir) / args.cube_dirname
        cube_dir.mkdir(parents=True, exist_ok=True)
        print(f'Processing single cube: {args.cube_dirname}  '
              f'(n_atoms={info["n_atoms"]})')

        if args.start_power is not None:
            print(f'Continuation mode: H^{args.start_power} → H^{args.N}')
            generate_H_powers_continued(
                start_power=args.start_power,
                N=args.N,
                outdir=cube_dir,
                file_format=args.fmt,
                expand=True,
                V=V_expr, kvec=kvec, k2=k2, pref=pref, x=x, y=y, z=z,
            )
        else:
            from symbolic_code.h_powers import generate_H_powers
            print(f'Full generation: H^0 → H^{args.N}')
            generate_H_powers(
                N=args.N,
                outdir=cube_dir,
                file_format=args.fmt,
                expand=True,
                V=V_expr, kvec=kvec, k2=k2, pref=pref, x=x, y=y, z=z,
            )
        print(f'Done.  Files in {cube_dir}')
        return

    # ------------------------------------------------------------------ #
    # All-cubes mode (original behaviour)                                  #
    # ------------------------------------------------------------------ #
    if args.start_power is not None:
        p.error('--start_power requires --cube_dirname to identify which cube '
                'to continue.')

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
