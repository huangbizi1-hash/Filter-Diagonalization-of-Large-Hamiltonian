#!/usr/bin/env python3
"""Compare Gaussian-trap energies from JDQMR (FFT operator) vs symbolic-expression potential.

Potential:
    V(r) = -A * exp(-B * r^2)
Default: A=10.0, B=0.5

This script builds the same FFT-DVR Hamiltonian twice:
1) numeric potential path
2) symbolic-expression path (SymPy -> NumPy grid values)

Then solves low-lying energies with PRIMME JDQMR and stores all results to JSON.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import sympy as sp

import ho3d_solvers_v2 as solver
from gaussian_potential_builder import PotentialGrid


@dataclass
class SolveSummary:
    method: str
    n: int
    L: float
    n_levels: int
    A: float
    B: float
    energies: list[float]


def make_uniform_grid(n: int, L: float) -> np.ndarray:
    """Grid points on [-L, L) with spacing d = 2L/n."""
    return np.linspace(-L, L, n, endpoint=False)


def gaussian_numeric_grid(x: np.ndarray, A: float, B: float) -> np.ndarray:
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = X**2 + Y**2 + Z**2
    return -A * np.exp(-B * r2)


def gaussian_symbolic_grid(x: np.ndarray, A: float, B: float) -> np.ndarray:
    xs, ys, zs, As, Bs = sp.symbols("x y z A B", real=True)
    expr = -As * sp.exp(-Bs * (xs**2 + ys**2 + zs**2))
    fn = sp.lambdify((xs, ys, zs, As, Bs), expr, modules="numpy")
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.asarray(fn(X, Y, Z, A, B), dtype=float)


def solve_jdqmr_fft(potential: np.ndarray, x: np.ndarray, n_levels: int) -> SolveSummary:
    grid = PotentialGrid(x=x, y=x, z=x, potential=potential)
    # Pass L from grid so FFT-DVR uses consistent box size.
    L = float(max(abs(x[0]), abs(x[-1])))
    res = solver.solve_ho3d(
        "fft_dvr:jdqmr",
        N=len(x),
        L=L,
        n_levels=n_levels,
        potential_grid=grid,
        tol=1e-10,
    )
    return SolveSummary(
        method="fft_dvr:jdqmr",
        n=len(x),
        L=L,
        n_levels=n_levels,
        A=np.nan,
        B=np.nan,
        energies=[float(v) for v in res.evals[:n_levels]],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gaussian trap: JDQMR(FFT) vs symbolic-expression potential")
    p.add_argument("--A", type=float, default=10.0)
    p.add_argument("--B", type=float, default=0.5)
    p.add_argument("--d", type=float, default=0.5, help="Grid spacing")
    p.add_argument("--n", type=int, default=32, help="Grid size per axis")
    p.add_argument("--n_levels", type=int, default=8)
    p.add_argument("--output", type=str, default="results_gaussian_symbolic_vs_jdqmr.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    L = args.n * args.d / 2.0
    x = make_uniform_grid(args.n, L)

    V_num = gaussian_numeric_grid(x, args.A, args.B)
    V_sym = gaussian_symbolic_grid(x, args.A, args.B)
    max_abs_v_diff = float(np.max(np.abs(V_num - V_sym)))

    s_num = solve_jdqmr_fft(V_num, x, args.n_levels)
    s_sym = solve_jdqmr_fft(V_sym, x, args.n_levels)

    # annotate A/B into summaries
    s_num.A = s_sym.A = args.A
    s_num.B = s_sym.B = args.B

    e_num = np.array(s_num.energies)
    e_sym = np.array(s_sym.energies)

    payload = {
        "config": {
            "potential": "V(r) = -A * exp(-B*r^2)",
            "A": args.A,
            "B": args.B,
            "d": args.d,
            "n": args.n,
            "L": L,
            "n_levels": args.n_levels,
        },
        "jdqmr_fft_numeric_potential": asdict(s_num),
        "jdqmr_fft_symbolic_potential": asdict(s_sym),
        "energy_abs_diff": [float(v) for v in np.abs(e_num - e_sym)],
        "energy_max_abs_diff": float(np.max(np.abs(e_num - e_sym))),
        "potential_max_abs_diff": max_abs_v_diff,
    }

    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    print(f"potential max |ΔV| = {max_abs_v_diff:.3e}")
    print(f"energy max |ΔE|    = {payload['energy_max_abs_diff']:.3e}")


if __name__ == "__main__":
    main()
