from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from rbf_core import (RBFConfig, build_problem, build_qd_problem,
                       build_hamiltonian_matrix,
                       iterate_hamiltonian, relative_laplacian_error,
                       solve_lowest_eigenvalues, sweep_rbf_kernels,
                       sweep_stencil_eps,
                       generate_nodes, KERNELS_ALL, KERNEL_GROUPS)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RBF-FD harmonic-oscillator tests.")
    parser.add_argument("--spacing", type=float, default=0.5, help="Poisson-disc node spacing.")
    parser.add_argument("--L", type=float, default=5.0, help="Half box size, domain is [-L, L]^3.")
    parser.add_argument("--stencil-size", type=int, default=80, help="Stencil size n in weight_matrix.")
    parser.add_argument("--phi", type=str, default="ga", help="RBF basis name, e.g. ga (Gaussian), phs3, phs5.")
    parser.add_argument("--eps", type=float, default=0.5, help="RBF epsilon parameter.")
    parser.add_argument(
        "--order", type=int, nargs="+", default=[2],
        help="Polynomial augmentation order(s). "
             "Use one value for normal runs (e.g. --order 2), "
             "or multiple values with --sweep-order (e.g. --order 0 1 2 3).",
    )
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
        help="Device for eigenvalue solve: 'cpu' (scipy ARPACK sparse) or 'cuda' (torch dense).",
    )
    parser.add_argument(
        "--symmetrize", action="store_true",
        help="Symmetrize H=(H+Hᵀ)/2 before solving (enables real eigsh). "
             "Default: off — uses non-symmetric eigs, returns complex eigenvalues.",
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

    # ── Stencil × eps 2D sweep ───────────────────────────────────────────────
    s2 = parser.add_argument_group(
        "Stencil×eps sweep (activated by --sweep-stencil-eps)")
    s2.add_argument(
        "--sweep-stencil-eps", action="store_true",
        help="2D sweep over stencil_size and eps for a fixed phi.",
    )
    s2.add_argument(
        "--sweep2-phi", type=str, default="ga",
        help="RBF kernel for the 2D sweep (default: ga).",
    )
    s2.add_argument(
        "--sweep2-stencil-sizes", type=int, nargs="+",
        default=[8, 16, 32, 64, 128, 256],
        help="Stencil sizes to sweep (default: 8 16 32 64 128 256).",
    )
    s2.add_argument(
        "--sweep2-eps-list", type=float, nargs="+",
        default=[round(0.1 * i, 10) for i in range(1, 11)],
        help="eps values to sweep (default: 0.1 0.2 … 1.0).",
    )
    s2.add_argument(
        "--sweep2-description", type=str,
        default="2D sweep of stencil_size × eps on 3D HO (ga kernel, fixed nodes)",
        help="Description stored in output JSON.",
    )
    s2.add_argument(
        "--sweep2-output-dir", type=str, default=".",
        help="Output directory for 2D sweep JSON and PNG.",
    )

    # ── Order sweep mode ────────────────────────────────────────────────────
    so = parser.add_argument_group("Order sweep (activated by --sweep-order)")
    so.add_argument(
        "--sweep-order", action="store_true",
        help="Sweep multiple polynomial augmentation orders with fixed "
             "spacing/phi/eps/stencil_size.",
    )
    so.add_argument(
        "--sweep-order-output-dir", type=str, default=".",
        help="Output directory for order-sweep JSON.",
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
    parser.add_argument(
        "--save-artifacts", action="store_true",
        help="Save generated nodes and sparse Hamiltonian matrix to files.",
    )
    parser.add_argument(
        "--artifacts-dir", type=str, default="rbf_artifacts",
        help="Output directory for saved nodes/matrix/metadata files.",
    )
    parser.add_argument(
        "--artifact-tag", type=str, default="",
        help="Optional extra tag appended to artifact filenames.",
    )
    return parser.parse_args()


def _single_order(args) -> int:
    """Return the first order value for modes that require exactly one order."""
    if not args.order:
        return 2
    return int(args.order[0])


def _ho_exact_levels(n: int) -> list[float]:
    levels: list[float] = []
    for s in range(100):
        e = s + 1.5
        deg = (s + 1) * (s + 2) // 2
        levels.extend([e] * deg)
        if len(levels) >= n:
            break
    return levels[:n]


def _run_sweep_order(args) -> None:
    import datetime
    import json
    import numpy as np
    from pathlib import Path

    orders = sorted(set(int(o) for o in args.order))
    print("=" * 72)
    print("Order sweep — 3D harmonic oscillator")
    print(f"  orders={orders}")
    print(f"  spacing={args.spacing}  L={args.L}  stencil_size={args.stencil_size}")
    print(f"  phi={args.phi}  eps={args.eps}  n_eigs={args.n_eigs}"
          f"  device={args.device}  symmetrize={args.symmetrize}")
    print("-" * 72)

    exact = _ho_exact_levels(args.n_eigs)
    results = []
    for order in orders:
        print(f"[order={order}] building problem...")
        config = RBFConfig(
            spacing=args.spacing,
            L=args.L,
            stencil_size=args.stencil_size,
            phi=args.phi,
            eps=args.eps,
            order=order,
            grid_N=args.grid_N,
        )
        problem = build_problem(config=config, build_interpolation=not args.no_interp)
        err = relative_laplacian_error(problem)

        row = {
            "order": order,
            "n_nodes_total": int(problem.nodes.shape[0]),
            "n_interior": int(problem.interior_idx.shape[0]),
            "laplacian_error": err,
            "status": "ok",
        }

        if not args.no_eigs:
            try:
                vals, _ = solve_lowest_eigenvalues(
                    problem, n_eigs=args.n_eigs, device=args.device,
                    symmetrize=args.symmetrize,
                )
                vals_real = [float(v.real) for v in vals]
                rel_errs = [
                    abs(vals_real[i] - exact[i]) / abs(exact[i])
                    if exact[i] != 0 else float("nan")
                    for i in range(min(len(vals_real), len(exact)))
                ]
                row.update({
                    "eigenvalues": vals_real,
                    "mean_rel_err_eigs": float(np.mean(rel_errs)) if rel_errs else None,
                    "max_rel_err_eigs": float(np.max(rel_errs)) if rel_errs else None,
                })
                print(f"  mean_rel_err_eigs={row['mean_rel_err_eigs']:.4e}")
            except Exception as e:
                row["status"] = "failed"
                row["error"] = str(e)
                print(f"  FAILED: {e}")
        results.append(row)

    ok = [r for r in results if r["status"] == "ok" and "mean_rel_err_eigs" in r]
    if ok:
        ok_sorted = sorted(ok, key=lambda r: r["mean_rel_err_eigs"])
        best_order = ok_sorted[0]["order"]
        print("-" * 72)
        print(f"Best order by mean_rel_err_eigs: {best_order}")
    else:
        best_order = None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.sweep_order_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "description": "Polynomial-order sweep on 3D harmonic oscillator",
        "datetime": datetime.datetime.now().isoformat(),
        "config": {
            "orders": orders,
            "spacing": args.spacing,
            "L": args.L,
            "stencil_size": args.stencil_size,
            "phi": args.phi,
            "eps": args.eps,
            "grid_N": args.grid_N,
            "n_eigs": args.n_eigs,
            "device": args.device,
            "symmetrize": args.symmetrize,
            "no_interp": args.no_interp,
            "no_eigs": args.no_eigs,
        },
        "best_order": best_order,
        "results": results,
    }
    json_path = out_dir / f"sweep_order_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f"JSON → {json_path}")


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
    order = _single_order(args)
    print(f"  eps={args.eps}  order={order}  n_eigs={args.n_eigs}"
          f"  device={args.device}  symmetrize={args.symmetrize}")
    print("  Generating nodes once (fixed for all kernels)...")
    nodes, groups = generate_nodes(spacing=args.spacing, L=args.L)
    interior_idx = groups["interior"]
    print(f"  nodes total={nodes.shape[0]}  interior={interior_idx.shape[0]}")
    print("-" * 72)

    results = sweep_rbf_kernels(
        nodes=nodes, groups=groups,
        stencil_size=args.stencil_size,
        eps=args.eps,
        order=order,
        n_eigs=args.n_eigs,
        device=args.device,
        symmetrize=args.symmetrize,
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
            "order":         order,
            "n_eigs":        args.n_eigs,
            "device":        args.device,
            "symmetrize":    args.symmetrize,
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
                 f"(spacing={args.spacing}, n={args.stencil_size}, eps={args.eps}, order={order})")
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


def _run_sweep_stencil_eps(args) -> None:
    import datetime
    import json
    import numpy as np
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    stencil_sizes = args.sweep2_stencil_sizes
    eps_values    = args.sweep2_eps_list
    phi           = args.sweep2_phi

    print("=" * 72)
    order = _single_order(args)
    print(f"2D sweep: stencil_size × eps   phi={phi}  order={order}")
    print(f"  stencil_sizes = {stencil_sizes}")
    print(f"  eps_values    = {[round(e, 2) for e in eps_values]}")
    print(f"  spacing={args.spacing}  L={args.L}  n_eigs={args.n_eigs}"
          f"  device={args.device}  symmetrize={args.symmetrize}")
    print("  Generating nodes (fixed)...")
    nodes, groups = generate_nodes(spacing=args.spacing, L=args.L)
    interior_idx = groups["interior"]
    print(f"  nodes total={nodes.shape[0]}  interior={interior_idx.shape[0]}")
    print("-" * 72)

    results = sweep_stencil_eps(
        nodes=nodes, groups=groups,
        phi=phi,
        stencil_sizes=stencil_sizes,
        eps_values=eps_values,
        order=order,
        n_eigs=args.n_eigs,
        device=args.device,
        symmetrize=args.symmetrize,
    )

    # best result
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        best = ok[0]
        print("-" * 72)
        print(f"Best: stencil={best['stencil_size']}  eps={best['eps']:.2f}"
              f"  mean_rel_err={best['mean_rel_err']:.4e}")

    # ── JSON ─────────────────────────────────────────────────────────────────
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.sweep2_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build 2D grid arrays for the JSON (rows=stencil_sizes, cols=eps_values)
    err_grid = []
    for n_st in stencil_sizes:
        row = []
        for eps in eps_values:
            match = next(
                (r for r in results
                 if r["stencil_size"] == n_st and abs(r["eps"] - eps) < 1e-9),
                None,
            )
            val = match["mean_rel_err"] if match and match["status"] == "ok" else None
            row.append(val)
        err_grid.append(row)

    output = {
        "description": args.sweep2_description,
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "spacing":        args.spacing,
            "L":              args.L,
            "n_nodes_total":  int(nodes.shape[0]),
            "n_interior":     int(interior_idx.shape[0]),
            "phi":            phi,
            "order":          order,
            "n_eigs":         args.n_eigs,
            "device":         args.device,
            "symmetrize":     args.symmetrize,
            "stencil_sizes":  stencil_sizes,
            "eps_values":     [round(e, 10) for e in eps_values],
        },
        "err_grid": err_grid,       # [stencil_idx][eps_idx]
        "results":  results,        # flat list sorted by mean_rel_err
        "best": {
            "stencil_size": best["stencil_size"],
            "eps":          best["eps"],
            "mean_rel_err": best["mean_rel_err"],
        } if ok else None,
    }
    json_path = out_dir / f"sweep_stencil_eps_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  JSON → {json_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if not ok:
        return

    # build numpy grid (NaN for failed)
    Z = np.full((len(stencil_sizes), len(eps_values)), np.nan)
    for r in ok:
        ri = stencil_sizes.index(r["stencil_size"])
        ci = min(range(len(eps_values)),
                 key=lambda j: abs(eps_values[j] - r["eps"]))
        Z[ri, ci] = r["mean_rel_err"]

    eps_arr = np.array([round(e, 2) for e in eps_values])
    n_st_arr = np.array(stencil_sizes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: 2D heat-map (log colour scale)
    ax = axes[0]
    vmin = np.nanmin(Z[Z > 0])
    vmax = np.nanmax(Z[np.isfinite(Z)])
    im = ax.pcolormesh(
        eps_arr, n_st_arr, Z,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis_r", shading="nearest",
    )
    fig.colorbar(im, ax=ax, label="Mean relative spectral error")
    ax.set_xlabel("eps (shape parameter)")
    ax.set_ylabel("Stencil size")
    ax.set_yscale("log")
    ax.set_yticks(n_st_arr)
    ax.set_yticklabels(n_st_arr)
    ax.set_title(f"3D HO spectral error  phi={phi}  spacing={args.spacing}"
                 f"  order={order}")
    # mark best
    if ok:
        ax.scatter([best["eps"]], [best["stencil_size"]],
                   marker="*", s=200, color="red", zorder=5, label="best")
        ax.legend(fontsize=9)

    # Right: line plot per stencil_size
    ax2 = axes[1]
    cmap_lines = plt.get_cmap("tab10")
    for idx, n_st in enumerate(stencil_sizes):
        row_errs = Z[idx]
        mask = np.isfinite(row_errs)
        if mask.any():
            ax2.semilogy(eps_arr[mask], row_errs[mask],
                         marker="o", markersize=4, linewidth=1.5,
                         color=cmap_lines(idx % 10),
                         label=f"n={n_st}")
    ax2.set_xlabel("eps")
    ax2.set_ylabel("Mean relative spectral error")
    ax2.set_title("Error vs eps per stencil size")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / f"sweep_stencil_eps_{ts}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot → {plot_path}")


def _print_eigs(vals: "np.ndarray", n_eigs: int, is_ho: bool) -> None:
    """Print eigenvalue table; compares to 3-D HO exact levels when is_ho=True.
    Handles both real (float64) and complex (complex128) eigenvalue arrays."""
    import numpy as np
    is_complex = np.iscomplexobj(vals)

    if is_ho:
        exact_levels: list[float] = []
        for s in range(30):
            e = s + 1.5
            deg = (s + 1) * (s + 2) // 2
            exact_levels.extend([e] * deg)
            if len(exact_levels) >= n_eigs:
                break
        if is_complex:
            print(f"{'#':>4}  {'Re(E)':>14}  {'Im(E)':>12}  "
                  f"{'E_exact':>10}  {'|Re-ex|/ex':>12}")
            for i, ev in enumerate(vals):
                ex = exact_levels[i] if i < len(exact_levels) else float("nan")
                re_err = abs(ev.real - ex) / abs(ex) if ex != 0 else float("nan")
                print(f"{i:4d}  {ev.real:14.6f}  {ev.imag:12.4e}  "
                      f"{ex:10.4f}  {re_err:12.4e}")
        else:
            print(f"{'#':>4}  {'E_rbf':>12}  {'E_exact':>12}  "
                  f"{'abs_err':>12}  {'rel_err':>10}")
            for i, ev in enumerate(vals):
                ex = exact_levels[i] if i < len(exact_levels) else float("nan")
                ae = abs(float(ev) - ex)
                re = ae / abs(ex) if ex != 0 else float("nan")
                print(f"{i:4d}  {float(ev):12.6f}  {ex:12.6f}  "
                      f"{ae:12.2e}  {re:10.2e}")
    else:
        if is_complex:
            print(f"{'#':>4}  {'Re(E)':>16}  {'Im(E)':>14}")
            for i, ev in enumerate(vals):
                print(f"{i:4d}  {ev.real:16.6f}  {ev.imag:14.4e}")
        else:
            print(f"{'#':>4}  {'E_rbf':>14}")
            for i, ev in enumerate(vals):
                print(f"{i:4d}  {float(ev):14.6f}")


def main() -> None:
    args = parse_args()

    # ── Order sweep mode ────────────────────────────────────────────────────
    if args.sweep_order:
        _run_sweep_order(args)
        return

    # ── Kernel sweep mode ────────────────────────────────────────────────────
    if args.sweep_kernels:
        _run_sweep_kernels(args)
        return

    # ── Stencil × eps 2D sweep ───────────────────────────────────────────────
    if args.sweep_stencil_eps:
        _run_sweep_stencil_eps(args)
        return

    # ── QD mode ──────────────────────────────────────────────────────────────
    if args.qd_cube:
        order = _single_order(args)
        print("=" * 72)
        print(f"QD mode  cube={args.qd_cube}  domain={args.qd_domain}")
        if args.qd_domain == "sphere":
            print(f"  sphere R={args.qd_R} Bohr  spacing={args.spacing}"
                  f"  sphere_subdivide={args.qd_sphere_subdivide}")
        print(f"  stencil_size={args.stencil_size}  phi={args.phi}"
              f"  eps={args.eps}  order={order}")
        print("Building problem (may take a while for large grids)...")
        problem = build_qd_problem(
            cube_file=args.qd_cube,
            domain=args.qd_domain,
            spacing=args.spacing,
            R=args.qd_R,
            stencil_size=args.stencil_size,
            phi=args.phi,
            eps=args.eps,
            order=order,
            sphere_subdivide=args.qd_sphere_subdivide,
        )
        print(f"nodes total    : {problem.nodes.shape[0]}")
        print(f"interior nodes : {problem.interior_idx.shape[0]}")
        print(f"V range        : [{problem.V_nodes.min():.4f}, {problem.V_nodes.max():.4f}] Ha")
        if args.save_artifacts:
            _save_problem_artifacts(problem, mode="qd", order=order, args=args)
        if not args.no_eigs:
            print("-" * 72)
            print(f"Sparse eigenvalue solve  n_eigs={args.n_eigs}  device={args.device}"
                  f"  symmetrize={args.symmetrize}")
            vals, _ = solve_lowest_eigenvalues(problem, n_eigs=args.n_eigs,
                                               device=args.device,
                                               symmetrize=args.symmetrize)
            _print_eigs(vals, args.n_eigs, is_ho=False)
        return

    # ── HO mode (default) ────────────────────────────────────────────────────
    config = RBFConfig(
        spacing=args.spacing,
        L=args.L,
        stencil_size=args.stencil_size,
        phi=args.phi,
        eps=args.eps,
        order=_single_order(args),
        grid_N=args.grid_N,
    )

    problem = build_problem(config=config, build_interpolation=not args.no_interp)
    if args.save_artifacts:
        _save_problem_artifacts(problem, mode="ho", order=config.order, args=args)

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
              f"  interior={problem.interior_idx.shape[0]}"
              f"  device={args.device}  symmetrize={args.symmetrize}")
        vals, _ = solve_lowest_eigenvalues(problem, n_eigs=args.n_eigs,
                                           device=args.device,
                                           symmetrize=args.symmetrize)
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
