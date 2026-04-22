from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rbf_core import (RBFConfig, build_problem, build_qd_problem,
                       iterate_hamiltonian, relative_laplacian_error,
                       solve_lowest_eigenvalues)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RBF-FD harmonic-oscillator tests.")
    parser.add_argument("--spacing", type=float, default=0.5, help="Poisson-disc node spacing.")
    parser.add_argument("--L", type=float, default=5.0, help="Half box size, domain is [-L, L]^3.")
    parser.add_argument("--stencil-size", type=int, default=80, help="Stencil size n in weight_matrix.")
    parser.add_argument("--phi", type=str, default="ga", help="RBF basis name, e.g. ga (Gaussian), phs3, phs5.")
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
        "--n-eigs",
        type=int,
        default=10,
        help="Number of lowest eigenvalues to compute via sparse eigensolver.",
    )
    parser.add_argument(
        "--no-eigs",
        action="store_true",
        help="Skip sparse eigenvalue solve (faster for large grids).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for eigenvalue solve: 'cpu' (scipy/ARPACK) or 'cuda' (cupy).",
    )

    # ── QD mode ──────────────────────────────────────────────────────────────
    qd = parser.add_argument_group("QD mode (activated when --qd-cube is given)")
    qd.add_argument(
        "--qd-cube", type=str, default="",
        help="Path to Gaussian .cube file for QD potential.",
    )
    qd.add_argument(
        "--qd-domain", type=str, default="cube", choices=["cube", "sphere"],
        help="Node placement: 'cube' uses the exact cube-file grid; "
             "'sphere' uses Poisson disc nodes inside a sphere of radius --qd-R.",
    )
    qd.add_argument(
        "--qd-R", type=float, default=20.0,
        help="Sphere radius in Bohr for --qd-domain sphere (default 20.0).",
    )
    qd.add_argument(
        "--qd-sphere-subdivide", type=int, default=3,
        help="Icosphere subdivision count (higher = smoother sphere boundary).",
    )

    parser.add_argument(
        "--csv", type=str, default="",
        help="Optional CSV output path for iteration summary.",
    )
    return parser.parse_args()



def _print_eigs(vals: "np.ndarray", n_eigs: int, is_ho: bool) -> None:
    """Print eigenvalue table; compares to 3-D HO exact levels when is_ho=True."""
    if is_ho:
        exact_levels: list[float] = []
        for s in range(30):
            e = s + 1.5
            deg = (s + 1) * (s + 2) // 2
            exact_levels.extend([e] * deg)
            if len(exact_levels) >= n_eigs:
                break
        print(f"{'#':>4}  {'E_rbf':>12}  {'E_exact':>12}  {'abs_err':>12}  {'rel_err':>10}")
        for i, ev in enumerate(vals):
            ex = exact_levels[i] if i < len(exact_levels) else float("nan")
            ae = abs(ev - ex)
            re = ae / abs(ex) if ex != 0 else float("nan")
            print(f"{i:4d}  {ev:12.6f}  {ex:12.6f}  {ae:12.2e}  {re:10.2e}")
    else:
        print(f"{'#':>4}  {'E_rbf':>14}")
        for i, ev in enumerate(vals):
            print(f"{i:4d}  {ev:14.6f}")


def main() -> None:
    args = parse_args()

    # ── QD mode ──────────────────────────────────────────────────────────────
    if args.qd_cube:
        print("=" * 72)
        print(f"QD mode  cube={args.qd_cube}  domain={args.qd_domain}")
        if args.qd_domain == "sphere":
            print(f"  sphere R={args.qd_R} Bohr  spacing={args.spacing}"
                  f"  sphere_subdivide={args.qd_sphere_subdivide}")
        print(f"  stencil_size={args.stencil_size}  phi={args.phi}"
              f"  eps={args.eps}  order={args.order}")
        print("Building problem (may take a while for large grids)...")
        problem = build_qd_problem(
            cube_file=args.qd_cube,
            domain=args.qd_domain,
            spacing=args.spacing,
            R=args.qd_R,
            stencil_size=args.stencil_size,
            phi=args.phi,
            eps=args.eps,
            order=args.order,
            sphere_subdivide=args.qd_sphere_subdivide,
        )
        print(f"nodes total    : {problem.nodes.shape[0]}")
        print(f"interior nodes : {problem.interior_idx.shape[0]}")
        print(f"V range        : [{problem.V_nodes.min():.4f}, {problem.V_nodes.max():.4f}] Ha")
        if not args.no_eigs:
            print("-" * 72)
            print(f"Sparse eigenvalue solve  n_eigs={args.n_eigs}  device={args.device}")
            vals, _ = solve_lowest_eigenvalues(problem, n_eigs=args.n_eigs,
                                               device=args.device)
            _print_eigs(vals, args.n_eigs, is_ho=False)
        return

    # ── HO mode (default) ────────────────────────────────────────────────────
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

    if not args.no_eigs:
        print("-" * 72)
        print(f"Sparse eigenvalue solve  n_eigs={args.n_eigs}"
              f"  interior={problem.interior_idx.shape[0]}  device={args.device}")
        vals, _ = solve_lowest_eigenvalues(problem, n_eigs=args.n_eigs,
                                           device=args.device)
        _print_eigs(vals, args.n_eigs, is_ho=True)

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
