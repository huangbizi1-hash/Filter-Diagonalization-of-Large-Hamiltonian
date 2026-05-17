#!/usr/bin/env python3
"""compare_symbolic_fft_ho3d.py

Compare two implementations of the Chebyshev filter f(H) = T_m(aH+b)
on the 3-D harmonic oscillator H = -½∇² + ½(x²+y²+z²).

Methods
-------
symbolic : Julia evaluates Σ_n c_n · H^n · sin(k·r+b) symbolically.
           H^n expressions are built by sympy, cached as .pkl, then
           compiled to a Julia batch script via the baseline pipeline.
fft      : Python applies T_m(aH+b) via the 3-term Chebyshev recurrence
           with H_FFT (exact kinetic energy via FFT) as the matvec.

Both methods use the same random plane waves {k, b} and the same
Rayleigh-Ritz step (H_FFT).  Their Ritz eigenvalues should agree to
well within the discretisation error, which validates the symbolic path.

Usage
-----
  python compare_symbolic_fft_ho3d.py \\
      --N_sweep 10:14 --box_L 5.0 \\
      --cheb_m 5 --E_lo -3.0 --E_hi 20.0 \\
      --n_random 400 --n_print 20

  # Wider sweep, finer grids
  python compare_symbolic_fft_ho3d.py \\
      --N_sweep 8,10,12,14,16 --box_L 5.0 \\
      --cheb_m 5 --E_lo -3.0 --E_hi 20.0 \\
      --n_random 600 --n_print 30 --svd_tol 1e-4
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# Reuse pipeline functions from the existing symbolic script
from compare_symbolic_explosion_ho3d import (
    ensure_H_powers_cache,
    ensure_julia_script,
    julia_eval_filter,
    ho3d_exact_levels,
    make_grid,
    make_T_k,
    apply_H_fft,
    parse_N_sweep,
)
from symbolic_code.chebyshev_filter import chebyshev_coeffs_transformed
from filter_core import svd_rayleigh_ritz_op


# ── Chebyshev 3-term recurrence (FFT path) ────────────────────────────────────

def chebyshev_recurrence(H_apply, psi0: np.ndarray,
                         a: float, b_sc: float, m: int) -> np.ndarray:
    """Apply T_m(a*H + b_sc) to psi0 via the 3-term recurrence.

    T_0(x) = 1  →  y_0 = psi0
    T_1(x) = x  →  y_1 = (a*H + b_sc) * psi0
    T_j(x) = 2x*T_{j-1} - T_{j-2}  →  y_j = 2*(a*H+b_sc)*y_{j-1} - y_{j-2}
    """
    def Hs(phi):
        return a * H_apply(phi) + b_sc * phi

    if m == 0:
        return psi0.copy()
    y_prev = psi0.copy()
    y_curr = Hs(psi0)
    for _ in range(2, m + 1):
        y_next = 2.0 * Hs(y_curr) - y_prev
        y_prev, y_curr = y_curr, y_next
    return y_curr


# ── one N sweep step ───────────────────────────────────────────────────────────

def run_one_N(
    N: int,
    box_L: float,
    cheb_m: int,
    a: float,
    b_sc: float,
    n_random: int,
    k_max_arg: float,
    seed: int,
    svd_tol: float,
    n_print: int,
    jl_path: Path,
    julia_exe: str,
    exact_all: np.ndarray,
) -> dict:
    # ── grid ──────────────────────────────────────────────────────────────────
    # Correct convention:  d = 2*box_L/N  (NOT 2*box_L/(N-1))
    # x1d has N points with spacing exactly d.
    # Using np.linspace(-box_L, box_L, N) would give spacing 2*box_L/(N-1),
    # making the FFT k-vectors wrong and the kinetic energy systematically off.
    d, x1d = make_grid(N, box_L)
    assert abs((x1d[1] - x1d[0]) - d) < 1e-12, \
        f"linspace spacing {x1d[1]-x1d[0]:.6e} != d={d:.6e} — grid bug!"

    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    V3  = 0.5 * (X**2 + Y**2 + Z**2)
    T_k = make_T_k(N, d)
    N3  = N ** 3
    k_max = k_max_arg if k_max_arg > 0.0 else np.pi / d

    print(f"\n── N={N}  d={d:.5f}  L={(N-1)*d/2:.4f}  N³={N3}  "
          f"k_max={k_max:.3f} ──")

    # ── H_FFT matvec ──────────────────────────────────────────────────────────
    def H_matvec(v):
        return apply_H_fft(v, V3, T_k)

    H_linop = LinearOperator((N3, N3), matvec=H_matvec, dtype=float)

    # ── reference eigenvalues (eigsh on H_FFT) ────────────────────────────────
    n_ref = min(10, N3 - 2)
    print(f"  [ref] eigsh (k={n_ref}) ...", end=' ', flush=True)
    t0 = time.perf_counter()
    E_ref, _ = eigsh(H_linop, k=n_ref, which='SA')
    t_ref = time.perf_counter() - t0
    E_ref = np.sort(E_ref.real)
    print(f"{t_ref:.2f}s  E_ref[:5]={np.round(E_ref[:5], 5).tolist()}")

    # ── random plane waves ────────────────────────────────────────────────────
    rng    = np.random.default_rng(seed)
    k_vals = rng.uniform(-k_max, k_max, (n_random, 3))
    b_vals = rng.uniform(0.0, 2 * np.pi, n_random)

    # ══════════════════════════════════════════════════════════════════════════
    # Path A: Julia symbolic
    # ══════════════════════════════════════════════════════════════════════════
    print(f"  [symbolic] Julia ({n_random} waves) ...", flush=True)
    t_julia0 = time.perf_counter()
    try:
        C_f_sym, julia_timing = julia_eval_filter(
            jl_path, x1d, k_vals, b_vals, julia_exe)
        t_julia = time.perf_counter() - t_julia0
        ws = julia_timing.get('warmup_s')
        es = julia_timing.get('eval_s')
        nw = julia_timing.get('n_eval_waves')
        if es and nw:
            print(f"    warmup={ws:.2f}s  eval={es:.2f}s/{nw} waves"
                  f"  ({es/nw*1e3:.2f} ms/wave)  wall={t_julia:.1f}s")
        else:
            print(f"    wall={t_julia:.1f}s")
        julia_ok = True
    except Exception as exc:
        print(f"    ERROR: {exc}")
        julia_ok = False
        C_f_sym = None
        julia_timing = {}
        t_julia = 0.0

    sym_result = {"ok": julia_ok, "julia_timing": julia_timing,
                  "t_julia_s": t_julia, "ritz_evals": [],
                  "rank": 0, "t_ritz_s": 0.0}

    if julia_ok and C_f_sym is not None:
        basis_sym = C_f_sym.reshape(n_random, N3).T   # (N3, n_random)
        print(f"  [symbolic] Ritz  basis={basis_sym.shape} ...", end=' ', flush=True)
        t0 = time.perf_counter()
        E_sym, _, rank_sym = svd_rayleigh_ritz_op(
            basis_sym, H_matvec, svd_tol=svd_tol,
            max_energies=n_print, hermitian=True)
        t_ritz_sym = time.perf_counter() - t0
        sym_result.update({"ritz_evals": E_sym.tolist(),
                           "rank": int(rank_sym), "t_ritz_s": t_ritz_sym})
        print(f"{t_ritz_sym:.2f}s  rank={rank_sym}  "
              f"E[0]={E_sym[0]:.6f}")
        print(f"    Ritz: {np.round(E_sym[:n_print], 6).tolist()}")

    # ══════════════════════════════════════════════════════════════════════════
    # Path B: FFT 3-term recurrence
    # ══════════════════════════════════════════════════════════════════════════
    print(f"  [fft] Chebyshev recurrence ({n_random} waves) ...", end=' ', flush=True)
    t_fft0 = time.perf_counter()
    C_f_fft = np.empty((n_random, N3), dtype=float)
    for i in range(n_random):
        kx, ky, kz = k_vals[i]
        phase = kx * X + ky * Y + kz * Z + b_vals[i]
        psi0  = np.sin(phase).ravel()
        C_f_fft[i] = chebyshev_recurrence(H_matvec, psi0, a, b_sc, cheb_m)
    t_filter_fft = time.perf_counter() - t_fft0
    print(f"{t_filter_fft:.2f}s")

    basis_fft = C_f_fft.T   # (N3, n_random)
    print(f"  [fft] Ritz  basis={basis_fft.shape} ...", end=' ', flush=True)
    t0 = time.perf_counter()
    E_fft, _, rank_fft = svd_rayleigh_ritz_op(
        basis_fft, H_matvec, svd_tol=svd_tol,
        max_energies=n_print, hermitian=True)
    t_ritz_fft = time.perf_counter() - t0
    fft_result = {"ritz_evals": E_fft.tolist(), "rank": int(rank_fft),
                  "t_filter_s": t_filter_fft, "t_ritz_s": t_ritz_fft}
    print(f"{t_ritz_fft:.2f}s  rank={rank_fft}  "
          f"E[0]={E_fft[0]:.6f}")
    print(f"    Ritz: {np.round(E_fft[:n_print], 6).tolist()}")

    # ── cross-check & error vs exact ─────────────────────────────────────────
    n_cmp = min(n_print, len(E_fft))
    err_fft_exact = np.array([float(np.min(np.abs(exact_all - e)))
                               for e in E_fft[:n_cmp]])
    max_fft_exact  = float(np.max(err_fft_exact))
    mean_fft_exact = float(np.mean(err_fft_exact))

    max_sym_fft  = float('nan')
    max_sym_exact = float('nan')
    mean_sym_exact = float('nan')
    if sym_result["ritz_evals"]:
        E_sym_arr = np.array(sym_result["ritz_evals"])
        n_cmp2 = min(n_cmp, len(E_sym_arr))
        diff_sf = np.abs(E_sym_arr[:n_cmp2] - E_fft[:n_cmp2])
        max_sym_fft = float(np.max(diff_sf))
        err_sym_exact = np.array([float(np.min(np.abs(exact_all - e)))
                                   for e in E_sym_arr[:n_cmp2]])
        max_sym_exact  = float(np.max(err_sym_exact))
        mean_sym_exact = float(np.mean(err_sym_exact))
        print(f"  max|E_sym − E_fft|    = {max_sym_fft:.3e}  "
              f"(symbolic vs FFT internal consistency)")

    print(f"  max|E_fft − exact|    = {max_fft_exact:.3e}  "
          f"mean={mean_fft_exact:.3e}")
    if not np.isnan(max_sym_exact):
        print(f"  max|E_sym − exact|    = {max_sym_exact:.3e}  "
              f"mean={mean_sym_exact:.3e}")

    return {
        "N":            N,
        "d":            float(d),
        "k_max":        float(k_max),
        "E_ref":        E_ref.tolist(),
        "t_ref_s":      float(t_ref),
        "symbolic":     sym_result,
        "fft":          fft_result,
        "max_sym_vs_fft":     max_sym_fft,
        "max_fft_vs_exact":   max_fft_exact,
        "mean_fft_vs_exact":  mean_fft_exact,
        "max_sym_vs_exact":   max_sym_exact,
        "mean_sym_vs_exact":  mean_sym_exact,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Symbolic vs FFT Chebyshev filter on 3D HO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('--N_sweep',   type=str,   default='10:14')
    ap.add_argument('--box_L',     type=float, default=5.0)
    ap.add_argument('--cheb_m',    type=int,   default=5)
    ap.add_argument('--E_lo',      type=float, default=-3.0)
    ap.add_argument('--E_hi',      type=float, default=20.0)
    ap.add_argument('--n_random',  type=int,   default=400)
    ap.add_argument('--n_print',   type=int,   default=20)
    ap.add_argument('--seed',      type=int,   default=42)
    ap.add_argument('--k_max',     type=float, default=0.0,
                    help='0 = auto (π/d)')
    ap.add_argument('--svd_tol',   type=float, default=1e-4)
    ap.add_argument('--cache_dir', type=str,   default='ho3d_symbolic_cache')
    ap.add_argument('--julia_exe', type=str,   default='julia')
    ap.add_argument('--out_json',  type=str,   default='symbolic_fft_ho3d.json')
    args = ap.parse_args()

    N_list    = parse_N_sweep(args.N_sweep)
    cache_dir = Path(args.cache_dir)
    a    =  2.0 / (args.E_hi - args.E_lo)
    b_sc = -(args.E_hi + args.E_lo) / (args.E_hi - args.E_lo)

    print("=== Symbolic vs FFT Chebyshev filter — 3D HO ===")
    print(f"H = -½∇² + ½(x²+y²+z²)   (no ω/m parameters)")
    print(f"box_L={args.box_L}  N_list={N_list}")
    print(f"cheb_m={args.cheb_m}  E_lo={args.E_lo}  E_hi={args.E_hi}")
    print(f"a={a:.8f}  b_sc={b_sc:.8f}")
    print(f"n_random={args.n_random}  seed={args.seed}  svd_tol={args.svd_tol}")
    print()

    # ── Step 1: H^n pkl ───────────────────────────────────────────────────────
    print("── Step 1: H^n symbolic cache ──")
    ensure_H_powers_cache(cache_dir, args.cheb_m)
    print()

    # ── Step 2: Chebyshev coefficients ────────────────────────────────────────
    coeffs = chebyshev_coeffs_transformed(args.cheb_m, a=a, b=b_sc)
    print("── Step 2: Chebyshev coefficients ──")
    print(f"  T_{args.cheb_m}(aH+b) = Σ c_n H^n")
    for n, c in enumerate(coeffs):
        print(f"    c_{n} = {float(c):.10g}")
    print()

    # ── Step 3: Julia .jl script ──────────────────────────────────────────────
    print("── Step 3: Julia filter script ──")
    jl_path = ensure_julia_script(cache_dir, args.cheb_m, args.E_lo, args.E_hi)
    print()

    # ── exact HO levels ───────────────────────────────────────────────────────
    exact_all = ho3d_exact_levels(omega=1.0, n_max_shell=20)

    # ── sweep ─────────────────────────────────────────────────────────────────
    t_wall0 = time.perf_counter()
    sweep   = []
    for N in N_list:
        res = run_one_N(
            N=N, box_L=args.box_L,
            cheb_m=args.cheb_m, a=a, b_sc=b_sc,
            n_random=args.n_random, k_max_arg=args.k_max,
            seed=args.seed, svd_tol=args.svd_tol, n_print=args.n_print,
            jl_path=jl_path, julia_exe=args.julia_exe,
            exact_all=exact_all,
        )
        sweep.append(res)
    wall_total = time.perf_counter() - t_wall0

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'N':>4}  {'d':>7}  {'N³':>7}  "
          f"{'sym_vs_fft':>12}  {'fft_vs_exact':>13}  {'sym_vs_exact':>13}")
    for r in sweep:
        print(f"{r['N']:>4}  {r['d']:>7.4f}  {r['N']**3:>7d}  "
              f"{r['max_sym_vs_fft']:>12.3e}  "
              f"{r['max_fft_vs_exact']:>13.3e}  "
              f"{r['max_sym_vs_exact']:>13.3e}")
    print(f"\nTotal wall time: {wall_total:.1f}s")

    # ── JSON ──────────────────────────────────────────────────────────────────
    output = {
        "script":   "compare_symbolic_fft_ho3d.py",
        "datetime": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "params": {
            "N_list":    N_list,
            "box_L":     args.box_L,
            "cheb_m":    args.cheb_m,
            "E_lo":      args.E_lo,
            "E_hi":      args.E_hi,
            "a":         a,
            "b_sc":      b_sc,
            "coeffs":    [float(c) for c in coeffs],
            "n_random":  args.n_random,
            "n_print":   args.n_print,
            "seed":      args.seed,
            "svd_tol":   args.svd_tol,
            "k_max_arg": args.k_max,
            "cache_dir": str(cache_dir.resolve()),
            "julia_exe": args.julia_exe,
            "jl_script": str(jl_path.resolve()),
        },
        "sweep":        sweep,
        "wall_total_s": wall_total,
    }
    Path(args.out_json).write_text(json.dumps(output, indent=2))
    print(f"JSON → {args.out_json}")


if __name__ == "__main__":
    main()
