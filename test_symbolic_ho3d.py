"""test_symbolic_ho3d.py
=====================
End-to-end test of the symbolic_code pipeline on the 3-D harmonic oscillator.

    H = -0.5 ∇² + 0.5(x² + y² + z²)

Exact eigenvalues:  E_n = n + 1.5   (n = nx + ny + nz = 0, 1, 2, ...)

On every run a timestamped sub-folder is created inside --outdir (default
``results/``).  It contains:

    filter_N<m>_E<lo>-<hi>.png  –  Chebyshev filter plot (no H^n needed)
    results.json                –  runtime, source info, recovered energies

Tests
-----
1. Single H step : symbolic apply_H_on_pair vs FFT  (grid-aligned k)
2. Filter diag   : full pipeline, recover lowest HO eigenvalues

H-power strategies (--method)
------------------------------
H_powers (default)
    Pure H^n with sp.expand().  Rational coefficients → compact expressions.
    a, b applied at assembly, so H^n files are reusable across energy windows.
scaled
    (aH+b)^n with a, b baked in (legacy, for comparison).

Usage
-----
    python test_symbolic_ho3d.py                              # both tests, H^n
    python test_symbolic_ho3d.py --quick                      # Test 1 only
    python test_symbolic_ho3d.py --N 6 --E_lo 0.5 --E_hi 6.5
    python test_symbolic_ho3d.py --method scaled              # legacy path
    python test_symbolic_ho3d.py --outdir /scratch/myresults  # custom output root
"""

import argparse
import datetime
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy as sp

from symbolic_code.h_powers import (
    apply_H_on_pair,
    generate_H_powers,
    generate_scaled_H_powers,
)
from symbolic_code.chebyshev_filter import (
    chebyshev_coeffs_transformed,
    apply_f_of_H_from_raw_powers,
    apply_f_of_H_on_psi,
    extract_cos_sin_coeffs,
    svd_H,
)
from symbolic_code.filter_plot import plot_chebyshev_filter


# ---------------------------------------------------------------------------
# Results-folder helpers
# ---------------------------------------------------------------------------

def make_run_dir(outdir, N, E_lo, E_hi, method):
    """Create and return a timestamped run directory inside *outdir*."""
    ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'N{N}_E{E_lo:.2f}-{E_hi:.2f}_{method}'
    run_dir = Path(outdir) / f'run_{ts}_{tag}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_filter_plot(run_dir, N, E_lo, E_hi, recovered_eigs=None):
    """Plot |T_N(aE+b)| and save PNG to *run_dir*.

    Exact HO eigenvalues are shown as tick markers.  If *recovered_eigs* is
    given (after Test 2), a second file is saved with both sets overlaid.
    Returns the filename of the primary plot.
    """
    exact_eigs = [1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
    fname = f'filter_N{N}_E{E_lo:.2f}-{E_hi:.2f}.png'
    out   = run_dir / fname
    plot_chebyshev_filter(
        m_list=[N],
        E_lo=E_lo, E_hi=E_hi,
        mode='bandpass',
        eigenvalues=exact_eigs,
        out_path=str(out),
    )
    import matplotlib.pyplot as plt
    plt.close('all')

    if recovered_eigs is not None:
        fname2 = f'filter_N{N}_E{E_lo:.2f}-{E_hi:.2f}_recovered.png'
        out2   = run_dir / fname2
        fig, ax = plot_chebyshev_filter(
            m_list=[N],
            E_lo=E_lo, E_hi=E_hi,
            mode='bandpass',
            eigenvalues=exact_eigs,
        )
        # overlay recovered eigenvalues in a different colour
        rec = np.asarray(recovered_eigs)
        ybot = ax.get_ylim()[0]
        ax.scatter(
            rec, np.full_like(rec, ybot),
            marker='v', s=40, color='steelblue', zorder=5,
            label='recovered',
        )
        ax.legend(fontsize=9, ncol=2, loc='upper right')
        fig.savefig(str(out2), dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f'  Saved -> {out2}')

    return fname


def write_json(run_dir, data):
    """Serialise *data* to run_dir/results.json (numpy-safe)."""
    def _cvt(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f'Not JSON-serialisable: {type(obj)}')

    out = run_dir / 'results.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=_cvt)
    print(f'  Results JSON -> {out}')


# ---------------------------------------------------------------------------
# Grid and FFT helpers
# ---------------------------------------------------------------------------

def make_grid(L=5.5, N=22):
    """Return 1-D coordinate array and 3-D meshgrids for [-L, L)."""
    x1 = np.linspace(-L, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
    return x1, X, Y, Z


def grid_kvecs(x1):
    """Return FFT-exact wave-vector array for coordinate grid x1."""
    dx = x1[1] - x1[0]
    return 2 * np.pi * np.fft.fftfreq(len(x1), d=dx)


def fft_apply_H(psi3d, x1, V3d):
    """Apply H = -0.5∇² + V via FFT (periodic boundary conditions)."""
    dx = x1[1] - x1[0]
    k1 = 2 * np.pi * np.fft.fftfreq(len(x1), d=dx)
    Kx, Ky, Kz = np.meshgrid(k1, k1, k1, indexing='ij')
    Hkin = np.fft.ifftn(
        0.5 * (Kx**2 + Ky**2 + Kz**2) * np.fft.fftn(psi3d)
    ).real
    return Hkin + V3d * psi3d


# ---------------------------------------------------------------------------
# Symbolic helpers
# ---------------------------------------------------------------------------

def ho_potential_sympy():
    """Return HO potential as sympy expression plus (x, y, z) symbols."""
    x, y, z = sp.symbols('x y z')
    V = sp.Rational(1, 2) * (x**2 + y**2 + z**2)
    return V, x, y, z


def build_fH_psi_sympy(N_cheby, E_lo, E_hi, method='H_powers'):
    """Compute f(H)*psi symbolically.  Returns (psi_fH, source_info dict).

    source_info records where / how the H-power expressions were built, so
    it can be written verbatim into results.json.
    """
    V_sym, x, y, z = ho_potential_sympy()
    kx, ky, kz, b  = sp.symbols('kx ky kz b')
    kvec = (kx, ky, kz)
    k2   = kx**2 + ky**2 + kz**2

    a    =  2.0 / (E_hi - E_lo)
    b_sc = -(E_hi + E_lo) / (E_hi - E_lo)

    tmpdir = tempfile.mkdtemp(prefix='ho_hpow_')
    source_info = {
        'h_powers_source' : 'computed_fresh',
        'h_powers_tmpdir' : tmpdir,   # cleaned up before function returns
        'h_powers_method' : method,
        'chebyshev_a'     : a,
        'chebyshev_b'     : b_sc,
    }
    try:
        if method == 'H_powers':
            generate_H_powers(
                N_cheby, tmpdir, file_format='pkl',
                V=V_sym, kvec=kvec, k2=k2, pref=0.5,
                x=x, y=y, z=z,
            )
            psi_fH = apply_f_of_H_from_raw_powers(tmpdir, N_cheby, a=a, b=b_sc)
        else:
            coeffs = chebyshev_coeffs_transformed(N_cheby, a=a, b=b_sc)
            generate_scaled_H_powers(
                N_cheby, a, b_sc, outdir=tmpdir, file_format='pkl',
                V=V_sym, kvec=kvec, k2=k2, pref=0.5,
                x=x, y=y, z=z,
            )
            psi_fH = apply_f_of_H_on_psi(tmpdir, coeffs, N_cheby, file_type='pkl')
    finally:
        shutil.rmtree(tmpdir)

    return psi_fH, source_info


# ---------------------------------------------------------------------------
# Test 1 – single H step
# ---------------------------------------------------------------------------

def test_single_H_step():
    """Verify symbolic H*psi matches FFT H*psi for grid-aligned k.

    Returns (ok, result_dict).
    """
    print("\n" + "="*60)
    print("Test 1: single H step  (symbolic vs FFT, grid-aligned k)")
    print("="*60)

    t0 = time.time()
    L, N = 5.0, 24
    x1, X, Y, Z = make_grid(L=L, N=N)
    V3d = 0.5 * (X**2 + Y**2 + Z**2)

    # Grid-aligned k: only these give an FFT kinetic energy that is exact.
    k_grid  = grid_kvecs(x1)
    small_k = k_grid[k_grid > 0][:4]   # e.g. ~[0.628, 1.257, 1.885, 2.513]

    V_sym, xs, ys, zs = ho_potential_sympy()
    kx_s, ky_s, kz_s, b_s = sp.symbols('kx ky kz b')
    kvec = (kx_s, ky_s, kz_s)
    k2   = kx_s**2 + ky_s**2 + kz_s**2

    # Analytical H*sin(k·r+b) = (0.5k² + 0.5r²)*sin(k·r+b)
    Ps_new, Pc_new = apply_H_on_pair(
        sp.Integer(1), sp.Integer(0),
        V_sym, kvec, k2, 0.5, xs, ys, zs,
    )
    theta_sym = kx_s*xs + ky_s*ys + kz_s*zs + b_s
    Hpsi_expr = Ps_new * sp.sin(theta_sym) + Pc_new * sp.cos(theta_sym)
    f_Hpsi = sp.lambdify(
        [xs, ys, zs, kx_s, ky_s, kz_s, b_s], Hpsi_expr, 'numpy'
    )

    rng = np.random.default_rng(0)
    b_vals = rng.uniform(0, np.pi, 6)
    k_triples = [
        (small_k[0], small_k[1], small_k[2]),
        (small_k[1], small_k[0], small_k[3]),
        (small_k[2], small_k[3], small_k[0]),
        (small_k[0], small_k[3], small_k[1]),
        (small_k[1], small_k[2], small_k[0]),
        (small_k[3], small_k[0], small_k[2]),
    ]

    errors = []
    for (kx_v, ky_v, kz_v), b_v in zip(k_triples, b_vals):
        psi      = np.sin(kx_v*X + ky_v*Y + kz_v*Z + b_v)
        Hpsi_num = fft_apply_H(psi, x1, V3d)
        Hpsi_sym = f_Hpsi(X, Y, Z, kx_v, ky_v, kz_v, b_v)
        err = float(np.max(np.abs(Hpsi_sym - Hpsi_num)))
        errors.append(err)
        print(f"  k=({kx_v:+.3f},{ky_v:+.3f},{kz_v:+.3f})  "
              f"max|sym-FFT| = {err:.2e}")

    max_err = max(errors)
    ok      = max_err < 1e-8
    runtime = time.time() - t0
    print(f"\n  Max error: {max_err:.2e}  →  {'PASSED ✓' if ok else 'FAILED ✗'}")

    return ok, {
        'name'       : 'single_H_step',
        'status'     : 'PASSED' if ok else 'FAILED',
        'max_error'  : max_err,
        'all_errors' : [float(e) for e in errors],
        'runtime_s'  : runtime,
    }


# ---------------------------------------------------------------------------
# Test 2 – filter diagonalisation
# ---------------------------------------------------------------------------

def test_filter_diag(N_cheby=4, E_lo=0.5, E_hi=6.5,
                     n_waves=60, method='H_powers'):
    """Full Chebyshev pipeline: build, evaluate, diagonalise, compare.

    Returns (ok, result_dict).
    """
    print("\n" + "="*60)
    print(f"Test 2: filter diagonalisation on 3D HO")
    print(f"        N_cheby={N_cheby}  E=[{E_lo}, {E_hi}]  method={method}")
    print("="*60)

    t_total = time.time()
    L, Ng = 5.5, 22

    # ---- Step A: build f(H)*psi symbolically ----
    print(f"  Building symbolic f(H)*psi (method={method}) ...")
    t0 = time.time()
    psi_fH, source_info = build_fH_psi_sympy(N_cheby, E_lo, E_hi, method=method)
    build_time = time.time() - t0
    print(f"  Done in {build_time:.1f} s")

    # ---- Step B: lambdify ----
    expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)
    xs, ys, zs = sp.symbols('x y z')
    kx_s, ky_s, kz_s, b_s = sp.symbols('kx ky kz b')
    sym_args = [xs, ys, zs, kx_s, ky_s, kz_s, b_s]
    print("  Lambdifying cos/sin envelopes ...")
    f_cos = sp.lambdify(sym_args, expr_cos, 'numpy')
    f_sin = sp.lambdify(sym_args, expr_sin, 'numpy')

    # ---- Step C: numerical grid ----
    x1, X, Y, Z = make_grid(L=L, N=Ng)
    V3d = 0.5 * (X**2 + Y**2 + Z**2)

    # ---- Step D: evaluate f(H)*psi for n_waves plane waves ----
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

    # ---- Step F: compare to exact HO eigenvalues ----
    exact_lo  = np.array([1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5])
    n_show    = min(len(energies), 10)
    n_compare = min(len(energies), len(exact_lo))
    errs      = np.abs(np.sort(energies[:n_compare]) - exact_lo[:n_compare])

    e0_err      = abs(float(energies[0]) - 1.5)
    ok_e0       = e0_err < 0.03
    n_good      = int(np.sum(errs < 0.1))
    ok_spectrum = n_good >= 4
    ok          = ok_e0 and ok_spectrum
    runtime     = time.time() - t_total

    print(f"\n  Recovered eigenvalues : {np.round(energies[:n_show], 4)}")
    print(f"  Exact HO eigenvalues  : {exact_lo[:n_show]}")
    print(f"\n  E_0 error = {e0_err:.4f}  (threshold 0.03)  "
          f"→ {'OK' if ok_e0 else 'FAIL'}")
    print(f"  Eigenvalues within 0.1 of exact: "
          f"{n_good}/{n_compare}  (need ≥4)  "
          f"→ {'OK' if ok_spectrum else 'FAIL'}")
    print(f"  {'PASSED ✓' if ok else 'FAILED ✗'}")

    return ok, {
        'name'                : 'filter_diagonalisation',
        'status'              : 'PASSED' if ok else 'FAILED',
        'runtime_s'           : runtime,
        'build_time_s'        : build_time,
        **source_info,
        'energies_recovered'  : [float(e) for e in energies[:n_show]],
        'energies_exact'      : exact_lo[:n_show].tolist(),
        'e0_error'            : e0_err,
        'n_within_0.1'        : n_good,
        'n_compare'           : n_compare,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Test symbolic_code pipeline on 3D harmonic oscillator.'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Only run Test 1 (single H step, ~seconds)')
    parser.add_argument('--N', type=int, default=4,
                        help='Chebyshev expansion order (default 4)')
    parser.add_argument('--E_lo', type=float, default=0.5,
                        help='Lower energy bound (default 0.5)')
    parser.add_argument('--E_hi', type=float, default=6.5,
                        help='Upper energy bound (default 6.5)')
    parser.add_argument('--n_waves', type=int, default=60,
                        help='Number of random plane waves (default 60)')
    parser.add_argument('--method', default='H_powers',
                        choices=['H_powers', 'scaled'],
                        help='H_powers=pure H^n (default), scaled=(aH+b)^n')
    parser.add_argument('--outdir', default='results',
                        help='Root folder for timestamped run dirs (default: results/)')
    args = parser.parse_args()

    # ---- create run directory ----
    run_dir = make_run_dir(args.outdir, args.N, args.E_lo, args.E_hi, args.method)
    print(f"\nRun directory: {run_dir}\n")

    # ---- plot filter immediately (no H^n computation needed) ----
    print("Plotting Chebyshev filter ...")
    filter_fname = save_filter_plot(run_dir, args.N, args.E_lo, args.E_hi)

    # ---- initialise JSON payload ----
    run_meta = {
        'timestamp' : datetime.datetime.now().isoformat(),
        'run_dir'   : str(run_dir),
        'args'      : {
            'N'       : args.N,
            'E_lo'    : args.E_lo,
            'E_hi'    : args.E_hi,
            'n_waves' : args.n_waves,
            'method'  : args.method,
            'quick'   : args.quick,
        },
        'filter_plot' : filter_fname,
    }

    # ---- run tests ----
    all_ok = {}

    ok1, data1 = test_single_H_step()
    all_ok['Test1_single_H_step'] = ok1
    run_meta['test1'] = data1

    if not args.quick:
        ok2, data2 = test_filter_diag(
            N_cheby=args.N,
            E_lo=args.E_lo,
            E_hi=args.E_hi,
            n_waves=args.n_waves,
            method=args.method,
        )
        all_ok['Test2_filter_diag'] = ok2
        run_meta['test2'] = data2

        # second plot: overlay recovered eigenvalues
        print("\nSaving filter plot with recovered eigenvalues ...")
        save_filter_plot(
            run_dir, args.N, args.E_lo, args.E_hi,
            recovered_eigs=data2.get('energies_recovered'),
        )

    run_meta['overall_status'] = 'PASSED' if all(all_ok.values()) else 'FAILED'

    # ---- write JSON ----
    print()
    write_json(run_dir, run_meta)

    # ---- summary ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, ok in all_ok.items():
        print(f"  {name:<30} {'PASSED ✓' if ok else 'FAILED ✗'}")
    print(f"\n  Results saved in: {run_dir}")

    sys.exit(0 if all(all_ok.values()) else 1)


if __name__ == '__main__':
    main()
