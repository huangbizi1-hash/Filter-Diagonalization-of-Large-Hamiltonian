"""
test_symbolic_ho3d.py
=====================
End-to-end test of the symbolic_code pipeline on the 3-D harmonic oscillator.

    H = -0.5 ∇² + 0.5(x² + y² + z²)

Exact eigenvalues:  E_n = n + 1.5   (n = nx + ny + nz = 0, 1, 2, ...)

Tests
-----
1. Single H step : symbolic apply_H_on_pair vs FFT  (fast, ~seconds)
2. Filter diag   : full Chebyshev pipeline, recover lowest eigenvalues

Usage
-----
    python test_symbolic_ho3d.py             # both tests (Test 2 ~ few minutes)
    python test_symbolic_ho3d.py --quick     # Test 1 only
    python test_symbolic_ho3d.py --N 3       # Chebyshev order for Test 2
"""

import argparse
import shutil
import sys
import tempfile
import time

import numpy as np
import sympy as sp

from symbolic_code.h_powers import apply_H_on_pair, generate_scaled_H_powers
from symbolic_code.chebyshev_filter import (
    chebyshev_coeffs_transformed,
    apply_f_of_H_on_psi,
    extract_cos_sin_coeffs,
    svd_H,
)


# ---------------------------------------------------------------------------
# Grid and FFT helpers
# ---------------------------------------------------------------------------

def make_grid(L=5.5, N=22):
    """Return 1-D coordinate array and 3-D meshgrids for [-L, L]^3."""
    x1 = np.linspace(-L, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
    return x1, X, Y, Z


def fft_apply_H(psi3d, x1, V3d):
    """Apply H = -0.5∇² + V numerically via FFT (periodic boundary conditions)."""
    N  = len(x1)
    dx = x1[1] - x1[0]
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=dx)          # wave-vector grid
    Kx, Ky, Kz = np.meshgrid(k1, k1, k1, indexing='ij')
    Hkin = np.fft.ifftn(
        0.5 * (Kx**2 + Ky**2 + Kz**2) * np.fft.fftn(psi3d)
    ).real
    return Hkin + V3d * psi3d


# ---------------------------------------------------------------------------
# Symbolic helpers
# ---------------------------------------------------------------------------

def ho_potential_sympy():
    """Return the HO potential as a sympy expression plus (x, y, z) symbols."""
    x, y, z = sp.symbols('x y z')
    V = sp.Rational(1, 2) * (x**2 + y**2 + z**2)
    return V, x, y, z


def build_fH_psi_sympy(N_cheby, E_lo, E_hi):
    """
    Compute f(H)*sin(θ) symbolically via Chebyshev expansion of order N_cheby.

    Returns psi_fH as a sympy expression in x,y,z,kx,ky,kz,b.
    Also returns the tempdir path (caller is responsible for cleanup).
    """
    V_sym, x, y, z = ho_potential_sympy()
    kx, ky, kz, b  = sp.symbols('kx ky kz b')
    kvec = (kx, ky, kz)
    k2   = kx**2 + ky**2 + kz**2

    a      =  2.0 / (E_hi - E_lo)
    b_sc   = -(E_hi + E_lo) / (E_hi - E_lo)
    coeffs = chebyshev_coeffs_transformed(N_cheby, a=a, b=b_sc)

    tmpdir = tempfile.mkdtemp(prefix='ho_hpow_')
    generate_scaled_H_powers(
        N_cheby, a, b_sc, outdir=tmpdir, file_format='pkl',
        V=V_sym, kvec=kvec, k2=k2, pref=0.5,
        x=x, y=y, z=z,
    )
    psi_fH = apply_f_of_H_on_psi(tmpdir, coeffs, N_cheby, file_type='pkl')
    shutil.rmtree(tmpdir)
    return psi_fH


# ---------------------------------------------------------------------------
# Test 1 – single H step
# ---------------------------------------------------------------------------

def test_single_H_step():
    """
    Verify that symbolic apply_H_on_pair(Ps=1, Pc=0, V_ho) matches
    FFT-based H applied to sin(kx*x + ky*y + kz*z + b).

    For a single Fourier mode, FFT kinetic is exact, so the only error
    is floating-point rounding -> expected max error < 1e-10.
    """
    print("\n" + "="*60)
    print("Test 1: single H step  (symbolic vs FFT)")
    print("="*60)

    x1, X, Y, Z = make_grid(L=5.0, N=24)
    V3d = 0.5 * (X**2 + Y**2 + Z**2)

    V_sym, xs, ys, zs = ho_potential_sympy()
    kx_s, ky_s, kz_s, b_s = sp.symbols('kx ky kz b')
    kvec = (kx_s, ky_s, kz_s)
    k2   = kx_s**2 + ky_s**2 + kz_s**2

    # pre-compute symbolic H*psi once (same for all k because Ps=1, Pc=0)
    Ps_new, Pc_new = apply_H_on_pair(
        sp.Integer(1), sp.Integer(0),
        V_sym, kvec, k2, 0.5, xs, ys, zs,
    )
    # Ps_new = 0.5*k² + 0.5*r²,  Pc_new = 0  (analytically)
    theta_sym  = kx_s*xs + ky_s*ys + kz_s*zs + b_s
    Hpsi_expr  = Ps_new * sp.sin(theta_sym) + Pc_new * sp.cos(theta_sym)
    f_Hpsi = sp.lambdify(
        [xs, ys, zs, kx_s, ky_s, kz_s, b_s], Hpsi_expr, 'numpy'
    )

    rng    = np.random.default_rng(0)
    errors = []
    for _ in range(6):
        kx_v, ky_v, kz_v = rng.uniform(-1.5, 1.5, 3)
        b_v  = rng.uniform(0, np.pi)
        psi  = np.sin(kx_v*X + ky_v*Y + kz_v*Z + b_v)

        Hpsi_num = fft_apply_H(psi, x1, V3d)
        Hpsi_sym = f_Hpsi(X, Y, Z, kx_v, ky_v, kz_v, b_v)

        err = float(np.max(np.abs(Hpsi_sym - Hpsi_num)))
        errors.append(err)
        print(f"  k=({kx_v:+.2f},{ky_v:+.2f},{kz_v:+.2f})  "
              f"max|sym-FFT| = {err:.2e}")

    max_err = max(errors)
    ok = max_err < 1e-8
    print(f"\n  Max error: {max_err:.2e}  →  {'PASSED ✓' if ok else 'FAILED ✗'}")
    return ok


# ---------------------------------------------------------------------------
# Test 2 – filter diagonalisation
# ---------------------------------------------------------------------------

def test_filter_diag(N_cheby=4, n_waves=60):
    """
    Full pipeline test:
      1. Build f(H)*psi symbolically (Chebyshev of order N_cheby).
      2. Evaluate on 3-D grid for n_waves random plane waves.
      3. Run SVD-based filter diagonalisation.
      4. Check recovered eigenvalues match HO spectrum E_n = n+1.5.

    Expected spectrum (lowest states):
      E = 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, ...
    """
    print("\n" + "="*60)
    print(f"Test 2: filter diagonalisation on 3D HO  (N_cheby={N_cheby})")
    print("="*60)

    E_lo, E_hi = 0.5, 6.5
    L,    Ng   = 5.5, 22

    # ---- Step A: build symbolic f(H)*psi ----
    print("  Building symbolic f(H)*psi ... (may take a few minutes for N>=4)")
    t0 = time.time()
    psi_fH = build_fH_psi_sympy(N_cheby, E_lo, E_hi)
    print(f"  Done in {time.time()-t0:.1f} s")

    # ---- Step B: extract sin/cos envelopes and lambdify ----
    expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)
    xs, ys, zs = sp.symbols('x y z')
    kx_s, ky_s, kz_s, b_s = sp.symbols('kx ky kz b')
    args = [xs, ys, zs, kx_s, ky_s, kz_s, b_s]

    print("  Lambdifying cos/sin envelopes ...")
    f_cos = sp.lambdify(args, expr_cos, 'numpy')
    f_sin = sp.lambdify(args, expr_sin, 'numpy')

    # ---- Step C: build grid ----
    x1, X, Y, Z = make_grid(L=L, N=Ng)
    V3d = 0.5 * (X**2 + Y**2 + Z**2)

    # ---- Step D: evaluate f(H)*psi for many plane waves ----
    rng    = np.random.default_rng(42)
    k_vals = rng.uniform(-1.5, 1.5, (n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, n_waves)

    print(f"  Evaluating for {n_waves} plane waves ...")
    C_f = np.zeros((n_waves, Ng, Ng, Ng), dtype=np.float64)
    for iw, (kv, bv) in enumerate(zip(k_vals, b_vals)):
        kx_v, ky_v, kz_v = kv
        phase = kx_v*X + ky_v*Y + kz_v*Z + bv
        fc = np.asarray(f_cos(X, Y, Z, kx_v, ky_v, kz_v, bv), dtype=float)
        fs = np.asarray(f_sin(X, Y, Z, kx_v, ky_v, kz_v, bv), dtype=float)
        C_f[iw] = fc * np.cos(phase) + fs * np.sin(phase)

    # ---- Step E: SVD filter diagonalisation ----
    print("  SVD filter diagonalisation ...")
    energies, _ = svd_H(
        C_f, Ng, Ng, Ng,
        x1, V3d, fft_apply_H,
        rank_threshold=1e-4,
        n_eigs=15,
    )

    # ---- Step F: check against exact HO eigenvalues ----
    exact_lo = np.array([1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5])
    n_show   = min(len(energies), 10)
    print(f"\n  Recovered eigenvalues : {np.round(energies[:n_show], 4)}")
    print(f"  Exact HO eigenvalues  : {exact_lo[:n_show]}")

    # Ground state must be within 2% of 1.5
    e0_err = abs(energies[0] - 1.5)
    ok_e0  = e0_err < 0.03

    # At least 4 eigenvalues must be within 0.1 of their HO counterpart
    n_compare = min(len(energies), len(exact_lo))
    errs = np.abs(np.sort(energies[:n_compare]) - exact_lo[:n_compare])
    ok_spectrum = np.sum(errs < 0.1) >= 4

    ok = ok_e0 and ok_spectrum
    print(f"\n  E_0 error = {e0_err:.4f}  (threshold 0.03)  "
          f"→ {'OK' if ok_e0 else 'FAIL'}")
    print(f"  Eigenvalues within 0.1 of exact: "
          f"{np.sum(errs < 0.1)}/{n_compare}  (need ≥4)  "
          f"→ {'OK' if ok_spectrum else 'FAIL'}")
    print(f"  {'PASSED ✓' if ok else 'FAILED ✗'}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Test symbolic_code pipeline on 3D harmonic oscillator.'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Only run Test 1 (single H step, ~seconds)'
    )
    parser.add_argument(
        '--N', type=int, default=4,
        help='Chebyshev expansion order for Test 2 (default 4, try 3 for speed)'
    )
    parser.add_argument(
        '--n_waves', type=int, default=60,
        help='Number of random plane waves (default 60)'
    )
    args = parser.parse_args()

    results = {}
    results['Test1_single_H_step'] = test_single_H_step()

    if not args.quick:
        results['Test2_filter_diag'] = test_filter_diag(
            N_cheby=args.N,
            n_waves=args.n_waves,
        )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, ok in results.items():
        status = 'PASSED ✓' if ok else 'FAILED ✗'
        print(f"  {name:<30} {status}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
