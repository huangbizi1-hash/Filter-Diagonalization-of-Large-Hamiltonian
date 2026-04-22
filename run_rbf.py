from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rbf_core import RBFConfig, build_problem, iterate_hamiltonian, relative_laplacian_error



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RBF-FD harmonic-oscillator tests.")
    parser.add_argument("--spacing", type=float, default=0.5, help="Poisson-disc node spacing.")
    parser.add_argument("--L", type=float, default=5.0, help="Half box size, domain is [-L, L]^3.")
    parser.add_argument("--stencil-size", type=int, default=80, help="Stencil size n in weight_matrix.")
    parser.add_argument("--phi", type=str, default="phs3", help="RBF basis name, e.g. phs3, phs5.")
    parser.add_argument("--eps", type=float, default=0.5, help="RBF epsilon parameter.")
    parser.add_argument("--order", type=int, default=2, help="Polynomial augmentation order.")
    parser.add_argument("--grid-N", type=int, default=60, help="Regular grid size for interpolation matrices.")
    parser.add_argument("--n-max", type=int, default=10, help="Number of H^n iterations.")
    parser.add_argument(
        "--no-interp",
        action="store_true",
        help="Do not build interpolation matrices if you only need node-space quantities.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV output path for iteration summary.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    config = RBFConfig(
        spacing=args.spacing,
        L=args.L,
        stencil_size=args.stencil_size,
        phi=args.phi,
        eps=args.eps,
        order=args.order,
        grid_N=args.grid_N,
    )

    problem = build_problem(config=config, build_interpolation=not args.no_interp)

    err = relative_laplacian_error(problem)
    print("=" * 72)
    print("RBF-FD problem summary")
    print(f"nodes total        : {problem.nodes.shape[0]}")
    print(f"interior nodes     : {problem.interior_idx.shape[0]}")
    print(f"spacing            : {config.spacing}")
    print(f"L                  : {config.L}")
    print(f"stencil_size       : {config.stencil_size}")
    print(f"phi                : {config.phi}")
    print(f"eps                : {config.eps}")
    print(f"order              : {config.order}")
    print(f"grid_N             : {config.grid_N}")
    print("-" * 72)
    print("Laplacian error on interior nodes")
    for k, v in err.items():
        print(f"{k:18s}: {v:.6e}")

    records = iterate_hamiltonian(problem, n_max=args.n_max, normalize_each_step=True)

    print("-" * 72)
    print(f"{'n':>3s} {'E_H':>14s} {'E_T':>14s} {'E_V':>14s} {'exact':>14s} {'rel_err':>12s}")
    for r in records:
        print(
            f"{r.n:3d} {r.E_H:14.6e} {r.E_T:14.6e} {r.E_V:14.6e} "
            f"{r.exact:14.6e} {r.rel_err:12.4e}"
        )

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "E_H", "E_T", "E_V", "exact", "rel_err", "scale_factor"])
            for r in records:
                writer.writerow([r.n, r.E_H, r.E_T, r.E_V, r.exact, r.rel_err, r.scale_factor])
        print("-" * 72)
        print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
