"""Entry-point: plot the Chebyshev filter function used in the symbolic pipeline.

The filter is  f(E) = T_m(aE + b)  where

    a =  2 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)

so that [E_lo, E_hi] maps to [-1, 1].

── Choosing parameters ──────────────────────────────────────────────────────

Explosion filter  (--mode explosion):
    Target: eigenvalues with  E < E_lo.
    E_lo   = energy threshold.  States below are amplified exponentially.
    E_hi   = upper bound, set to max(spectrum) + margin  (>> E_lo).
    m      = polynomial order.  Amplification at distance delta below E_lo:
                |T_m(aE+b)| ~ cosh(m * acosh(1 + 2*delta/(E_hi-E_lo)))
             Doubling m roughly doubles the exponent.

Bandpass filter  (--mode bandpass):
    Target: eigenvalues inside [E_lo, E_hi].
    E_lo, E_hi  = bracket only the target window.
    States inside have |T_m| <= 1; states outside grow.
    SVD filter diagonalisation recovers eigenvalues in [E_lo, E_hi].
    For sharp boundaries increase m.

── Usage examples ──────────────────────────────────────────────────────

# Explosion: target E < 4.0, full spectrum up to 70
python plot_symbolic_filter.py --E_lo 4.0 --E_hi 70.0 --m 20 40 --mode explosion

# Bandpass: window [0.5, 6.5] (3D harmonic oscillator ground band)
python plot_symbolic_filter.py --E_lo 0.5 --E_hi 6.5 --m 4 6 10 --mode bandpass

# Mark exact HO eigenvalues on the plot
python plot_symbolic_filter.py --E_lo 0.5 --E_hi 6.5 --m 4 10 --mode bandpass \\
    --eigenvalues 1.5 2.5 3.5 4.5 5.5 6.5

# Custom axis range
python plot_symbolic_filter.py --E_lo 1.5 --E_hi 30 --m 20 --E_min 0 --E_max 35
"""

import argparse
import sys

import numpy as np

from symbolic_code.filter_plot import (
    filter_response,
    plot_chebyshev_filter,
    print_filter_summary,
)


def build_parser():
    p = argparse.ArgumentParser(
        description='Plot Chebyshev filter |T_m(aE+b)| vs energy.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--E_lo', type=float, required=True,
                   help='Lower energy window boundary (maps to -1)')
    p.add_argument('--E_hi', type=float, required=True,
                   help='Upper energy window boundary (maps to +1)')
    p.add_argument('--m', type=int, nargs='+', default=[20],
                   help='Chebyshev order(s) to compare (default: 20)')
    p.add_argument('--mode', choices=['explosion', 'bandpass'],
                   default='explosion',
                   help=(
                       'explosion: target E < E_lo; '
                       'bandpass: target window [E_lo, E_hi]  (default: explosion)'
                   ))
    p.add_argument('--E_min', type=float, default=None,
                   help='Left edge of x-axis')
    p.add_argument('--E_max', type=float, default=None,
                   help='Right edge of x-axis')
    p.add_argument('--clip', type=float, default=100.0,
                   help='Clip |T_m| at this value in the linear panel (default: 100)')
    p.add_argument('--n_pts', type=int, default=3000,
                   help='Number of energy sample points (default: 3000)')
    p.add_argument('--eigenvalues', type=float, nargs='+', default=None,
                   help='Known eigenvalues to mark on plot (optional)')
    p.add_argument('--out', type=str, default='filter_response.png',
                   help='Output figure path (default: filter_response.png)')
    p.add_argument('--no-summary', action='store_true',
                   help='Skip printing the text summary table')
    return p


def main():
    args = build_parser().parse_args()

    if args.E_lo >= args.E_hi:
        sys.exit('ERROR: E_lo must be strictly less than E_hi')

    a =  2.0 / (args.E_hi - args.E_lo)
    b = -(args.E_hi + args.E_lo) / (args.E_hi - args.E_lo)

    print(f"\nChebyshev filter: |T_m(aE + b)|")
    print(f"  E_lo  = {args.E_lo}")
    print(f"  E_hi  = {args.E_hi}")
    print(f"  m     = {args.m}")
    print(f"  mode  = {args.mode}")
    print(f"  a     = {a:.6f}   window: {args.E_lo} -> -1,  {args.E_hi} -> +1")
    print(f"  b     = {b:.6f}")

    if not args.no_summary:
        if args.mode == 'explosion':
            hw = args.E_hi - args.E_lo
            probes = [
                args.E_lo - 0.05 * hw,
                args.E_lo - 0.20 * hw,
                args.E_lo - 0.50 * hw,
                args.E_lo - 1.00 * hw,
            ]
        else:
            hw = args.E_hi - args.E_lo
            mid = 0.5 * (args.E_lo + args.E_hi)
            probes = [
                args.E_lo - 0.3 * hw,
                args.E_lo,
                mid,
                args.E_hi,
                args.E_hi + 0.3 * hw,
            ]
        print_filter_summary(args.m, args.E_lo, args.E_hi,
                             probe_energies=probes)

    plot_chebyshev_filter(
        m_list=args.m,
        E_lo=args.E_lo,
        E_hi=args.E_hi,
        E_min=args.E_min,
        E_max=args.E_max,
        n_pts=args.n_pts,
        clip=args.clip,
        mode=args.mode,
        eigenvalues=args.eigenvalues,
        out_path=args.out,
    )


if __name__ == '__main__':
    main()
