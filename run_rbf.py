from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rbf_core import (RBFConfig, build_problem, build_qd_problem,
                       iterate_hamiltonian, relative_laplacian_error,
                       solve_lowest_eigenvalues, sweep_rbf_kernels,
                       generate_nodes, KERNELS_ALL, KERNEL_GROUPS)



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

    # ── Kernel sweep mode ─────────────────────────────────────────────────────
    sw = parser.add_argument_group("Kernel sweep (activated by --sweep-kernels)")
    sw.add_argument(
        "--sweep-kernels", action="store_true",
        help="Sweep all RBF kernels on fixed nodes, rank by mean spectral error.",
    )
    sw.add_argument(
        "--sweep-description", type=str,
        default="RBF kernel sweep on 3D harmonic oscillator, fixed Poisson-disc nodes",
        help="Description string stored in the output JSON.",
    )
    sw.add_argument(
        "--sweep-output-dir", type=str, default=".",
        help="Directory for JSON and PNG output (default: current dir).",
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



def _run_sweep_kernels(args) -> None:
    import datetime
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    print("=" * 72)
    print("Kernel sweep — 3D harmonic oscillator (nodes fixed)")
    print(f"  spacing={args.spacing}  L={args.L}  stencil_size={args.stencil_size}")
    print(f"  eps={args.eps}  order={args.order}  n_eigs={args.n_eigs}  device={args.device}")
    print("  Generating nodes once (fixed for all kernels)...")
    nodes, groups = generate_nodes(spacing=args.spacing, L=args.L)
    interior_idx = groups["interior"]
    print(f"  nodes total={nodes.shape[0]}  interior={interior_idx.shape[0]}")
    print("-" * 72)

    results = sweep_rbf_kernels(
        nodes=nodes, groups=groups,
        stencil_size=args.stencil_size,
        eps=args.eps,
        order=args.order,
        n_eigs=args.n_eigs,
        device=args.device,
    )

    ok     = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    print("-" * 72)
    print("Ranking (best → worst):")
    for i, r in enumerate(ok):
        print(f"  {i+1:2d}. {r['phi']:<10}  mean_rel_err={r['mean_rel_err']:.4e}")
    for r in failed:
        print(f"   FAILED: {r['phi']}  — {r.get('error','')}")

    # ── JSON ─────────────────────────────────────────────────────────────────
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.sweep_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "description": args.sweep_description,
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "spacing":       args.spacing,
            "L":             args.L,
            "n_nodes_total": int(nodes.shape[0]),
            "n_interior":    int(interior_idx.shape[0]),
            "stencil_size":  args.stencil_size,
            "eps":           args.eps,
            "order":         args.order,
            "n_eigs":        args.n_eigs,
            "device":        args.device,
        },
        "results": results,
        "ranking": [
            {"rank": i + 1, "phi": r["phi"], "mean_rel_err": r["mean_rel_err"]}
            for i, r in enumerate(ok)
        ],
        "best":  ok[0]["phi"] if ok else None,
        "worst": ok[-1]["phi"] if ok else None,
    }
    json_path = out_dir / f"sweep_kernels_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  JSON → {json_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if not ok:
        return

    group_colors = {
        "PHS":         "#4C72B0",
        "Global":      "#DD8452",
        "LengthScale": "#55A868",
        "Wendland":    "#C44E52",
    }
    phi_to_group = {p: g for g, ps in KERNEL_GROUPS.items() for p in ps}

    phis   = [r["phi"]          for r in ok]
    errs   = [r["mean_rel_err"] for r in ok]
    colors = [group_colors.get(phi_to_group.get(p, ""), "#888888") for p in phis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: sorted bar chart, log-scale
    ax = axes[0]
    ax.bar(range(len(phis)), errs, color=colors)
    ax.set_xticks(range(len(phis)))
    ax.set_xticklabels(phis, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Mean relative spectral error")
    ax.set_title(f"RBF kernel sweep — 3D HO "
                 f"(spacing={args.spacing}, n={args.stencil_size}, eps={args.eps}, order={args.order})")
    ax.grid(True, axis="y", alpha=0.3)
    legend_elems = [Patch(facecolor=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper left")

    # Right: per-eigenvalue relative error for top 5 kernels
    ax2 = axes[1]
    for r in ok[:5]:
        ax2.plot(range(args.n_eigs), r["rel_errs"],
                 marker="o", linewidth=1.5, markersize=4, label=r["phi"])
    ax2.set_xlabel("Eigenvalue index")
    ax2.set_ylabel("Relative error |E_rbf − E_exact| / |E_exact|")
    ax2.set_yscale("log")
    ax2.set_title("Per-eigenvalue error (top 5 kernels)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / f"sweep_kernels_{ts}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot → {plot_path}")


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

    # ── Kernel sweep mode ────────────────────────────────────────────────────
    if args.sweep_kernels:
        _run_sweep_kernels(args)
        return

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
