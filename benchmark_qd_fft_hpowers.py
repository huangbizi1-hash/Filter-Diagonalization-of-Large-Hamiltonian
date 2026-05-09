#!/usr/bin/env python3
"""benchmark_qd_fft_hpowers.py
Benchmark pure-FFT Hamiltonian applications on QD R=17.

This script applies H^n (n=1..n_max) to random trial states using the FFT
Hamiltonian backend from fft_code, and records per-order timings.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from gaussian_potential_builder import GaussianPotentialBuilder
from fft_code.grid import build_k_diagonal
from fft_code.hamiltonian import apply_H


def build_qd17_potential(cube_file: str, params_file: str, r_cut: float, N: int | None,
                         L_s: float, d_grid: float):
    """Build potential for QD and return (x_grid, V)."""
    builder = GaussianPotentialBuilder(cube_file=cube_file, params_file=params_file, r_cut=r_cut)

    if N is None:
        N = int(np.ceil(2 * L_s / d_grid))
    x, y, z, V = builder.build_potential(N)
    return [x, y, z], V


def benchmark_hpowers(V: np.ndarray, x_grid, n_max: int, n_waves: int, seed: int):
    """Apply H repeatedly and measure per-order cost."""
    rng = np.random.default_rng(seed)
    N = V.shape[0]
    T_k = build_k_diagonal(x_grid)

    psis = rng.normal(size=(n_waves, N, N, N)) + 1j * rng.normal(size=(n_waves, N, N, N))
    psis = psis.astype(np.complex128, copy=False)

    rows = []
    current = psis

    for n in range(1, n_max + 1):
        t0 = time.perf_counter()
        nxt = np.empty_like(current)
        for i in range(n_waves):
            nxt[i] = apply_H(current[i], V, T_k)
        dt = time.perf_counter() - t0

        pts = N**3
        h_applies = n_waves
        ms_total = dt * 1000.0
        ms_per_wave = ms_total / n_waves
        ns_per_pt_per_wave = ms_per_wave / pts * 1e6

        rows.append({
            "n": n,
            "N": N,
            "grid_points": pts,
            "n_waves": n_waves,
            "total_ms": round(ms_total, 3),
            "ms_per_wave": round(ms_per_wave, 6),
            "ns_per_pt_per_wave": round(ns_per_pt_per_wave, 6),
            "h_applies": h_applies,
        })

        print(
            f"n={n:2d} | total={ms_total:9.3f} ms | "
            f"per_wave={ms_per_wave:9.4f} ms | "
            f"{ns_per_pt_per_wave:9.3f} ns/pt/wave"
        )

        current = nxt

    return rows


def main():
    parser = argparse.ArgumentParser(description="Benchmark FFT-only H^n timing on QD17")
    parser.add_argument("--cube_file", default="localPot.cube", help="Cube file for QD potential")
    parser.add_argument("--params_file", default="gaussian_fit_params.json", help="Gaussian parameter JSON")
    parser.add_argument("--r_cut", type=float, default=7.0, help="Atom cutoff in Bohr")
    parser.add_argument("--N", type=int, default=None, help="Grid size per axis; default from L_s/d_grid")
    parser.add_argument("--L_s", type=float, default=22.0, help="Half-box size in Bohr (QD17 default)")
    parser.add_argument("--d_grid", type=float, default=0.625, help="Grid spacing in Bohr")
    parser.add_argument("--n_max", type=int, default=8, help="Max H power")
    parser.add_argument("--n_waves", type=int, default=20, help="Number of trial waves")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--outdir", default="figs_qd_timing", help="Output directory")
    args = parser.parse_args()

    print("Building QD potential from files...")
    x_grid, V = build_qd17_potential(args.cube_file, args.params_file, args.r_cut, args.N, args.L_s, args.d_grid)
    N = V.shape[0]
    print(f"  Grid: N={N}, points={N**3:,}")
    print(f"  V range: [{V.min():.6f}, {V.max():.6f}]")

    print(f"\nBenchmark FFT H^n with n_max={args.n_max}, n_waves={args.n_waves}")
    rows = benchmark_hpowers(V, x_grid, args.n_max, args.n_waves, args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"fft_hpowers_qd17_nmax{args.n_max}_nw{args.n_waves}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "cube_file": args.cube_file,
        "params_file": args.params_file,
        "N": N,
        "grid_points": N**3,
        "n_max": args.n_max,
        "n_waves": args.n_waves,
        "seed": args.seed,
        "results_csv": str(csv_path),
    }
    json_path = outdir / f"fft_hpowers_qd17_nmax{args.n_max}_nw{args.n_waves}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved CSV  -> {csv_path}")
    print(f"Saved JSON -> {json_path}")


if __name__ == "__main__":
    main()
