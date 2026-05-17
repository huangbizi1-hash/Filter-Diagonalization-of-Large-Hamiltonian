#!/usr/bin/env python3
"""Compare FFT and FD discretizations on 3D HO using PRIMME_JDQMR.

Outputs JSON with:
- first n_levels eigenvalues
- accuracy against reference sequence (n + 0.5)
- average single H-apply time estimated by repeated matvec calls
"""

from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import primme

from ho3d_solvers_v2 import build_3d_fft_operator, build_3d_fd_operator


def _benchmark_matvec(H_op, dim: int, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)

    t0 = time.perf_counter()
    for _ in range(repeats):
        v = H_op @ v
    total = time.perf_counter() - t0
    return {
        "repeats": int(repeats),
        "total_seconds": float(total),
        "avg_seconds": float(total / repeats),
    }


def _solve_jdqmr(H_op, n_levels: int, tol: float, ncv: int, max_matvecs: int) -> dict:
    t0 = time.perf_counter()
    evals, _, stats = primme.eigsh(
        H_op,
        k=n_levels,
        which="SA",
        method="PRIMME_JDQMR",
        ncv=ncv,
        tol=tol,
        maxMatvecs=max_matvecs,
        return_stats=True,
        return_history=False,
    )
    wall = time.perf_counter() - t0
    evals = np.sort(np.asarray(evals, dtype=float))
    return {
        "eigenvalues": evals.tolist(),
        "solve_seconds": float(wall),
        "numMatvecs": int(stats.get("numMatvecs", -1)),
    }


def _accuracy_against_n_half(evals: np.ndarray) -> dict:
    ref = np.arange(len(evals), dtype=float) + 0.5
    abs_err = np.abs(evals - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-15)
    return {
        "reference": ref.tolist(),
        "abs_error": abs_err.tolist(),
        "rel_error": rel_err.tolist(),
        "mae": float(np.mean(abs_err)),
        "max_abs_error": float(np.max(abs_err)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FFT vs FD on HO3D with JDQMR")
    ap.add_argument("--N", type=int, default=32, help="Grid size per axis")
    ap.add_argument("--L", type=float, default=8.0, help="Half box length for HO")
    ap.add_argument("--n-levels", type=int, default=20, help="Number of eigenvalues")
    ap.add_argument("--fd-order", type=int, default=8, help="FD stencil order")
    ap.add_argument("--tol", type=float, default=1e-8, help="PRIMME tolerance")
    ap.add_argument("--ncv", type=int, default=120, help="PRIMME ncv")
    ap.add_argument("--max-matvecs", type=int, default=200000, help="PRIMME max matvecs")
    ap.add_argument("--timing-repeats", type=int, default=500, help="H applies for timing")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--out", type=Path, default=Path("results/fft_fd_ho3d_jdqmr.json"), help="Output JSON path")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    H_fft, dim, _ = build_3d_fft_operator(args.N, potential_grid=None, L=args.L)
    H_fd, _, _ = build_3d_fd_operator(args.N, potential_grid=None, fd_order=args.fd_order, L=args.L)

    fft_solve = _solve_jdqmr(H_fft, args.n_levels, args.tol, args.ncv, args.max_matvecs)
    fd_solve = _solve_jdqmr(H_fd, args.n_levels, args.tol, args.ncv, args.max_matvecs)

    fft_e = np.array(fft_solve["eigenvalues"], dtype=float)
    fd_e = np.array(fd_solve["eigenvalues"], dtype=float)

    out = {
        "script": "cli_compare_fft_fd_ho3d_jdqmr.py",
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "N": args.N,
            "L": args.L,
            "n_levels": args.n_levels,
            "fd_order": args.fd_order,
            "tol": args.tol,
            "ncv": args.ncv,
            "max_matvecs": args.max_matvecs,
            "timing_repeats": args.timing_repeats,
            "seed": args.seed,
            "reference": "n + 0.5",
        },
        "fft": {
            **fft_solve,
            "accuracy_vs_n_plus_0_5": _accuracy_against_n_half(fft_e),
            "h_apply_timing": _benchmark_matvec(H_fft, dim, args.timing_repeats, args.seed),
        },
        "fd": {
            **fd_solve,
            "accuracy_vs_n_plus_0_5": _accuracy_against_n_half(fd_e),
            "h_apply_timing": _benchmark_matvec(H_fd, dim, args.timing_repeats, args.seed),
        },
    }

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
