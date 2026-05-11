#!/usr/bin/env python3
"""test_gaussian_symbolic_energy.py

Test accuracy of the symbolic Julia H-operator against a FFT+eigsh reference
on a 3-D Gaussian potential well  V(r) = -A * exp(-B * r²).

Two evaluation paths
--------------------
[ref]  FFT H (LinearOperator via numpy.fft) + scipy eigsh
         → reference eigenvalues E_ref[i] and eigenvectors ψ_ref[i]

[sym]  Symbolic H^1 → Julia batch evaluation
         For each ψ_ref[i]:
           1. Decompose in plane-wave basis: ĉ_k = FFT(ψ_i)
           2. Batch-evaluate H|k_cos⟩ and H|k_sin⟩ at all grid points using Julia
           3. Reconstruct H_sym ψ_i = (1/N³) Σ_k [Re(ĉ_k)·H_cos_k − Im(ĉ_k)·H_sin_k]
           4. Rayleigh quotient E_sym_i = ⟨ψ_i|H_sym|ψ_i⟩
           5. Residual ‖H_sym ψ_i − E_ref_i ψ_i‖ / ‖ψ_i‖

Random-state Chebyshev filter test (--n_random, --E_lo, --E_hi, --cheb_m)
---------------------------------------------------------------------------
Generates --n_random random real states, applies the Chebyshev explosion
filter T_{cheb_m}(aH+b) (with a,b mapping [E_lo,E_hi]→[-1,1]) using the
FFT Hamiltonian, then compares FFT and symbolic Rayleigh quotients for each
filtered state.  This mirrors the chebyshev_explosion path in main.py.

Symbolic pipeline
-----------------
  1. SymPy apply_H_on_pair → Pc_new, Ps_new for cos-start and sin-start
  2. julia_codegen.build_julia_hn_cse_script → two Julia source files
       h1_cos.jl : out[i] = cos_p[i] * (k²/2 + V(r_i))   [cos-start]
       h1_sin.jl : out[i] = sin_p[i] * (k²/2 + V(r_i))   [sin-start]
  3. Julia is called in batches (--batch_size k-vectors at a time)

Usage
-----
  python test_gaussian_symbolic_energy.py
  python test_gaussian_symbolic_energy.py --A 10 --B 0.5 --d 0.5 --box_L 6
  python test_gaussian_symbolic_energy.py --n_levels 8 --batch_size 512 --save_jl
  python test_gaussian_symbolic_energy.py --n_random 4 --E_lo -9.0 --E_hi -6.0 --cheb_m 20
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, str(Path(__file__).parent))
from symbolic_code.h_powers import apply_H_on_pair
from symbolic_code.julia_codegen import build_julia_hn_cse_script


# ── grid helpers ─────────────────────────────────────────────────────────────

def make_grid(d: float, box_L: float):
    """Uniform periodic grid on [-box_L, box_L) with spacing d.

    Returns N (points per axis) and x1d (1-D coordinate array, length N).
    The grid does NOT include +box_L; this is consistent with periodic FFT.
    """
    N = int(round(2.0 * box_L / d))
    x1d = np.arange(N) * d - box_L
    return N, x1d


# ── FFT reference operator ───────────────────────────────────────────────────

def _make_T_k(N: int, x1d: np.ndarray) -> np.ndarray:
    """Kinetic-energy diagonal in k-space: T_k[kx,ky,kz] = (kx²+ky²+kz²)/2."""
    d = float(x1d[1] - x1d[0])
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    return (k1d[:, None, None] ** 2 +
            k1d[None, :, None] ** 2 +
            k1d[None, None, :] ** 2) / 2.0


def _apply_H_fft(psi: np.ndarray, V_num: np.ndarray, T_k: np.ndarray) -> np.ndarray:
    """Apply H = -½∇² + V to psi (3-D array) via FFT."""
    return np.fft.ifftn(T_k * np.fft.fftn(psi)).real + V_num * psi


def build_fft_H_op(N: int, x1d: np.ndarray, V_num: np.ndarray):
    """Build scipy LinearOperator for H = -½∇² + V using numpy.fft.

    T̂ψ = IFFT[ (k²/2) · FFT(ψ) ]   (periodic BCs, pseudospectral)
    """
    T_k = _make_T_k(N, x1d)

    def matvec(v):
        return _apply_H_fft(v.reshape(N, N, N), V_num, T_k).ravel()

    return LinearOperator((N ** 3, N ** 3), matvec=matvec, dtype=float)


def apply_chebyshev_filter_fft(
        psi: np.ndarray, V_num: np.ndarray, T_k: np.ndarray,
        m: int, E_lo: float, E_hi: float) -> np.ndarray:
    """Apply T_m(aH+b) to psi using the three-term Chebyshev recurrence.

    Maps [E_lo, E_hi] → [-1, 1] via a = 2/(E_hi-E_lo), b = -(E_hi+E_lo)/(E_hi-E_lo).
    Identical to apply_chebyshev_explosion in fft_code/hamiltonian.py but
    self-contained (no external imports).
    """
    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)

    def _apply_Hs(phi: np.ndarray) -> np.ndarray:
        return a * _apply_H_fft(phi, V_num, T_k) + b * phi

    y_prev = psi.copy()
    if m == 0:
        return y_prev
    y_curr = _apply_Hs(psi)
    for _ in range(2, m + 1):
        y_next = 2.0 * _apply_Hs(y_curr) - y_prev
        y_prev = y_curr
        y_curr = y_next
    return y_curr


# ── symbolic H^1 codegen ─────────────────────────────────────────────────────

def build_h1_symbolic(A: float, B: float):
    """Return symbolic H^1 envelopes for V = -A*exp(-B*r²).

    Two starting conditions:
      cos-start (Pc=1, Ps=0): H maps cos(k·r) → Pc_new*cos(k·r) + Ps_new*sin(k·r)
      sin-start (Pc=0, Ps=1): H maps sin(k·r) → Pc_new*cos(k·r) + Ps_new*sin(k·r)

    For this simple Gaussian:
      cos-start → Pc_new = k²/2 + V,  Ps_new = 0
      sin-start → Pc_new = 0,          Ps_new = k²/2 + V

    Returns (Pc_cos, Ps_cos, Pc_sin, Ps_sin) and symbol dict.
    """
    x, y, z = sp.symbols('x y z', real=True)
    kx, ky, kz = sp.symbols('kx ky kz', real=True)
    V_sym = -A * sp.exp(-B * (x ** 2 + y ** 2 + z ** 2))
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    kvec = (kx, ky, kz)

    # cos-start: initial Ps=0, Pc=1
    Ps_cos, Pc_cos = apply_H_on_pair(
        sp.Integer(0), sp.Integer(1), V_sym, kvec, k2, 0.5, x, y, z)
    Pc_cos = sp.expand(Pc_cos)
    Ps_cos = sp.Integer(0) if sp.simplify(Ps_cos) == 0 else sp.expand(Ps_cos)

    # sin-start: initial Ps=1, Pc=0
    Ps_sin, Pc_sin = apply_H_on_pair(
        sp.Integer(1), sp.Integer(0), V_sym, kvec, k2, 0.5, x, y, z)
    Ps_sin = sp.expand(Ps_sin)
    Pc_sin = sp.Integer(0) if sp.simplify(Pc_sin) == 0 else sp.expand(Pc_sin)

    return Pc_cos, Ps_cos, Pc_sin, Ps_sin


# ── Julia batch runner ───────────────────────────────────────────────────────

def call_julia_batch(jl_file, Xf, Yf, Zf, kx_b, ky_b, kz_b,
                     work_dir, julia_exe):
    """Call Julia for one batch of k-vectors; return (n_waves, N_grid) array.

    k-vectors are passed with b=0 (no phase offset).
    out[j, i] = eval_one_wave! output for wave j at grid point i.
    """
    N_grid = Xf.size
    n_waves = len(kx_b)
    wd = Path(work_dir)

    grid_bin = wd / '_sg.bin'
    k_bin    = wd / '_sk.bin'
    out_bin  = wd / '_so.bin'

    with open(grid_bin, 'wb') as f:
        Xf.astype('<f8').tofile(f)
        Yf.astype('<f8').tofile(f)
        Zf.astype('<f8').tofile(f)

    b_zeros = np.zeros(n_waves)
    kb = np.column_stack([kx_b, ky_b, kz_b, b_zeros]).astype('<f8')
    kb.ravel().tofile(str(k_bin))

    proc = subprocess.run(
        [julia_exe, str(jl_file),
         str(grid_bin), str(k_bin), str(out_bin),
         str(N_grid), str(n_waves)],
        capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f'Julia failed (rc={proc.returncode}):\n{proc.stderr[:2000]}')

    raw = np.fromfile(str(out_bin), dtype='<f8')
    # Julia writes: out[1:N], out[N+1:2N], ... (wave-major order)
    return raw.reshape(n_waves, N_grid)


# ── symbolic H application ───────────────────────────────────────────────────

def apply_H_sym(jl_cos_file, jl_sin_file, psi, N, x1d,
                julia_exe, work_dir, batch_size):
    """Apply symbolic H to psi via plane-wave decomposition + Julia batch eval.

    H ψ(r_i) = (1/N³) Σ_k [ Re(ĉ_k) · H_cos_k(r_i) − Im(ĉ_k) · H_sin_k(r_i) ]

    where:
      H_cos_k(r) = (k²/2 + V(r)) · cos(k·r)   ← jl_cos_file output
      H_sin_k(r) = (k²/2 + V(r)) · sin(k·r)   ← jl_sin_file output

    Both files use b=0, so cos_p[i]=cos(k·r_i) and sin_p[i]=sin(k·r_i).
    """
    N3 = N ** 3
    d = float(x1d[1] - x1d[0])
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    Xf = X.ravel().astype(float)
    Yf = Y.ravel().astype(float)
    Zf = Z.ravel().astype(float)

    # k-grid frequencies matching numpy.fft convention
    kfreq = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    Kx, Ky, Kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    kx_all = Kx.ravel()
    ky_all = Ky.ravel()
    kz_all = Kz.ravel()

    c_hat  = np.fft.fftn(psi)          # (N,N,N) complex
    c_flat = c_hat.ravel()              # (N³,) complex, ĉ_k = c_flat[k]

    Hpsi_flat = np.zeros(N3)
    n_batches = (N3 + batch_size - 1) // batch_size

    for bi in range(n_batches):
        s = bi * batch_size
        e = min(s + batch_size, N3)

        kx_b = kx_all[s:e]
        ky_b = ky_all[s:e]
        kz_b = kz_all[s:e]
        c_b  = c_flat[s:e]             # shape (batch,) complex

        # H_cos_k[j,i] = Pc_new(k_j, r_i) * cos(k_j·r_i)
        H_cos = call_julia_batch(
            jl_cos_file, Xf, Yf, Zf, kx_b, ky_b, kz_b, work_dir, julia_exe)

        # H_sin_k[j,i] = Ps_new(k_j, r_i) * sin(k_j·r_i)
        H_sin = call_julia_batch(
            jl_sin_file, Xf, Yf, Zf, kx_b, ky_b, kz_b, work_dir, julia_exe)

        # Accumulate: Hψ_i += (1/N³) Σ_j [Re(ĉ_j) * H_cos[j,i] - Im(ĉ_j) * H_sin[j,i]]
        Hpsi_flat += (H_cos.T @ c_b.real - H_sin.T @ c_b.imag) / N3

    return Hpsi_flat.reshape(N, N, N)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Compare FFT+eigsh vs symbolic Julia H on Gaussian potential.')
    ap.add_argument('--A', type=float, default=10.0,
                    help='Gaussian depth (default 10.0 Ha)')
    ap.add_argument('--B', type=float, default=0.5,
                    help='Gaussian width exponent (default 0.5 Bohr⁻²)')
    ap.add_argument('--d', type=float, default=0.5,
                    help='Grid spacing in Bohr (default 0.5)')
    ap.add_argument('--box_L', type=float, default=6.0,
                    help='Half-box size in Bohr (default 6.0)')
    ap.add_argument('--n_levels', type=int, default=5,
                    help='Number of eigenstates to compare (default 5)')
    ap.add_argument('--batch_size', type=int, default=256,
                    help='k-vectors per Julia batch call (default 256)')
    ap.add_argument('--julia', default='julia',
                    help='Julia executable (default: julia)')
    ap.add_argument('--out_json', default='gaussian_symbolic_test.json',
                    help='Output JSON path (default: gaussian_symbolic_test.json)')
    ap.add_argument('--save_jl', action='store_true',
                    help='Save generated .jl files as h1_cos.jl / h1_sin.jl')
    # ── Chebyshev filter / random-state test ────────────────────────────────
    ap.add_argument('--n_random', type=int, default=0,
                    help='Number of random initial states to filter and test '
                         '(default 0 = skip random-state test). '
                         'Requires --E_lo and --E_hi.')
    ap.add_argument('--E_lo', type=float, default=None,
                    help='Lower energy bound for Chebyshev explosion filter.')
    ap.add_argument('--E_hi', type=float, default=None,
                    help='Upper energy bound for Chebyshev explosion filter.')
    ap.add_argument('--cheb_m', type=int, default=20,
                    help='Chebyshev filter order m (default 20). '
                         'T_m(aH+b) maps [E_lo,E_hi]→[-1,1].')
    ap.add_argument('--seed', type=int, default=42,
                    help='RNG seed for random initial states (default 42)')
    args = ap.parse_args()

    if args.n_random > 0 and (args.E_lo is None or args.E_hi is None):
        ap.error('--n_random requires both --E_lo and --E_hi')

    # ── Grid ────────────────────────────────────────────────────────────────
    N, x1d = make_grid(args.d, args.box_L)
    d      = float(x1d[1] - x1d[0])     # actual spacing (may differ slightly from args.d)
    N3     = N ** 3
    print(f'Grid: N={N} per axis  N³={N3}  d={d:.4f} Bohr  '
          f'box_L={args.box_L} Bohr')
    print(f'Potential: V = -{args.A} * exp(-{args.B} * r²)')

    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    V_num   = -args.A * np.exp(-args.B * (X ** 2 + Y ** 2 + Z ** 2))
    T_k     = _make_T_k(N, x1d)    # kinetic diagonal in k-space (shared)

    # ── Reference: FFT + eigsh ───────────────────────────────────────────────
    print('\n[ref] Building FFT H operator and solving with eigsh ...')
    H_fft  = build_fft_H_op(N, x1d, V_num)   # uses pre-built T_k internally
    t0_ref = time.perf_counter()
    E_ref, psi_ref = eigsh(H_fft, k=args.n_levels, which='SA')
    t_ref  = time.perf_counter() - t0_ref
    idx    = np.argsort(E_ref)
    E_ref  = E_ref[idx]
    psi_ref = psi_ref[:, idx]
    print(f'  Done in {t_ref:.2f}s')
    for i, e in enumerate(E_ref):
        print(f'  E_ref[{i}] = {e:.8f} Ha')

    # ── Symbolic H^1 codegen ─────────────────────────────────────────────────
    print('\n[sym] Building symbolic H^1 envelopes ...')
    t0 = time.perf_counter()
    Pc_cos, Ps_cos, Pc_sin, Ps_sin = build_h1_symbolic(args.A, args.B)
    t_sym_build = time.perf_counter() - t0
    print(f'  cos-start: Pc={Pc_cos}  Ps={Ps_cos}')
    print(f'  sin-start: Pc={Pc_sin}  Ps={Ps_sin}')
    print(f'  ops(Pc_cos)={sp.count_ops(Pc_cos)}  '
          f'ops(Ps_sin)={sp.count_ops(Ps_sin)}  '
          f'build_time={t_sym_build:.2f}s')

    print('\n[sym] Generating Julia code ...')
    # cos-start file: out[i] = cos_p[i]*(k²/2+V(r)) -- Pc_cos is the cos envelope
    jl_src_cos, ops_cos = build_julia_hn_cse_script(Pc_cos, Ps_cos)
    # sin-start file: out[i] = sin_p[i]*(k²/2+V(r)) -- Ps_sin is the sin envelope
    jl_src_sin, ops_sin = build_julia_hn_cse_script(Pc_sin, Ps_sin)
    print(f'  cos .jl: {len(jl_src_cos):,} bytes  ops={ops_cos}')
    print(f'  sin .jl: {len(jl_src_sin):,} bytes  ops={ops_sin}')

    # ── Rayleigh-quotient test for each eigenvector ──────────────────────────
    results = []
    with tempfile.TemporaryDirectory(prefix='gsym_') as td:
        jl_cos = Path(td) / 'h1_cos.jl'
        jl_sin = Path(td) / 'h1_sin.jl'
        jl_cos.write_text(jl_src_cos)
        jl_sin.write_text(jl_src_sin)

        if args.save_jl:
            Path('h1_cos.jl').write_text(jl_src_cos)
            Path('h1_sin.jl').write_text(jl_src_sin)
            print('  Saved h1_cos.jl  h1_sin.jl')

        for i in range(args.n_levels):
            psi_i = psi_ref[:, i].reshape(N, N, N)

            # Normalise on the grid (trapezoidal / Riemann sum)
            norm2  = float(np.sum(psi_i ** 2) * d ** 3)
            psi_n  = psi_i / np.sqrt(norm2)

            print(f'\n[sym] Level {i}  E_ref={E_ref[i]:.8f}  '
                  f'||ψ||={np.sqrt(norm2):.6f}  '
                  f'batches={( N3 + args.batch_size - 1) // args.batch_size}')

            t0_s = time.perf_counter()
            Hpsi_sym = apply_H_sym(
                jl_cos, jl_sin, psi_n, N, x1d,
                args.julia, td, args.batch_size)
            t_sym_i = time.perf_counter() - t0_s

            # Rayleigh quotient: E_sym = ⟨ψ|H|ψ⟩ (ψ already normalised)
            E_sym_i  = float(np.sum(psi_n * Hpsi_sym) * d ** 3)

            # Residual: ||H_sym ψ − E_ref ψ|| / ||ψ||   (ψ normalised, so ||ψ||=1)
            residual = float(np.sqrt(np.sum((Hpsi_sym - E_ref[i] * psi_n) ** 2) * d ** 3))

            abs_err  = abs(E_sym_i - E_ref[i])
            print(f'  E_sym={E_sym_i:.8f}  |ΔE|={abs_err:.2e}  '
                  f'residual={residual:.2e}  t_sym={t_sym_i:.1f}s')

            results.append({
                'level'          : i,
                'E_ref'          : float(E_ref[i]),
                'E_sym'          : E_sym_i,
                'abs_energy_err' : abs_err,
                'residual'       : residual,
                'time_sym_s'     : t_sym_i,
            })

    # ── Random-state Chebyshev filter test ───────────────────────────────────
    random_results = []
    if args.n_random > 0:
        rng = np.random.default_rng(args.seed)
        print(f'\n[rand] Chebyshev explosion filter test:')
        print(f'  E_lo={args.E_lo}  E_hi={args.E_hi}  cheb_m={args.cheb_m}  '
              f'n_random={args.n_random}  seed={args.seed}')
        a_cheb = 2.0 / (args.E_hi - args.E_lo)
        b_cheb = -(args.E_hi + args.E_lo) / (args.E_hi - args.E_lo)
        print(f'  Chebyshev scaling: a={a_cheb:.4f}  b={b_cheb:.4f}')

        with tempfile.TemporaryDirectory(prefix='gsym_rand_') as td_rand:
            jl_cos_r = Path(td_rand) / 'h1_cos.jl'
            jl_sin_r = Path(td_rand) / 'h1_sin.jl'
            jl_cos_r.write_text(jl_src_cos)
            jl_sin_r.write_text(jl_src_sin)

            for i in range(args.n_random):
                # Random real state, Gaussian-modulated for faster decay at boundaries
                psi_r = rng.standard_normal((N, N, N))
                norm2_r = float(np.sum(psi_r ** 2) * d ** 3)
                psi_r /= np.sqrt(norm2_r)

                # Apply Chebyshev explosion filter with FFT H
                t0_f = time.perf_counter()
                psi_f = apply_chebyshev_filter_fft(
                    psi_r, V_num, T_k, args.cheb_m, args.E_lo, args.E_hi)
                t_filter = time.perf_counter() - t0_f

                # Normalize filtered state
                norm2_f = float(np.sum(psi_f ** 2) * d ** 3)
                if norm2_f < 1e-30:
                    print(f'  random[{i}]: filtered state has negligible norm; skipping.')
                    continue
                psi_fn = psi_f / np.sqrt(norm2_f)

                # FFT Rayleigh quotient
                Hpsi_fft_r = _apply_H_fft(psi_fn, V_num, T_k)
                E_fft_r    = float(np.sum(psi_fn * Hpsi_fft_r) * d ** 3)

                # Symbolic Rayleigh quotient via Julia
                t0_s = time.perf_counter()
                Hpsi_sym_r = apply_H_sym(
                    jl_cos_r, jl_sin_r, psi_fn, N, x1d,
                    args.julia, td_rand, args.batch_size)
                t_sym_r = time.perf_counter() - t0_s
                E_sym_r = float(np.sum(psi_fn * Hpsi_sym_r) * d ** 3)

                abs_err_r = abs(E_sym_r - E_fft_r)
                in_window = args.E_lo <= E_fft_r <= args.E_hi
                print(f'  random[{i:03d}]: E_fft={E_fft_r:.6f}  E_sym={E_sym_r:.6f}  '
                      f'|ΔE|={abs_err_r:.2e}  in_window={in_window}  '
                      f't_filter={t_filter:.1f}s  t_sym={t_sym_r:.1f}s')

                random_results.append({
                    'idx'            : i,
                    'E_fft'          : E_fft_r,
                    'E_sym'          : E_sym_r,
                    'abs_energy_err' : abs_err_r,
                    'in_window'      : in_window,
                    'time_filter_s'  : t_filter,
                    'time_sym_s'     : t_sym_r,
                })

        n_in = sum(1 for r in random_results if r['in_window'])
        if random_results:
            max_err_r = max(r['abs_energy_err'] for r in random_results)
            print(f'\n  In-window: {n_in}/{len(random_results)}  '
                  f'Max |ΔE(sym−fft)|={max_err_r:.2e} Ha')

    # ── Save JSON ────────────────────────────────────────────────────────────
    summary = {
        'params': {
            'A'         : args.A,
            'B'         : args.B,
            'd'         : d,
            'box_L'     : args.box_L,
            'N'         : N,
            'N3'        : N3,
            'n_levels'  : args.n_levels,
            'batch_size': args.batch_size,
            'n_random'  : args.n_random,
            'E_lo'      : args.E_lo,
            'E_hi'      : args.E_hi,
            'cheb_m'    : args.cheb_m,
            'seed'      : args.seed,
        },
        'ops_cos'            : ops_cos,
        'ops_sin'            : ops_sin,
        'time_ref_s'         : t_ref,
        'time_sym_build_s'   : t_sym_build,
        'results'            : results,
        'max_abs_energy_err' : max(r['abs_energy_err'] for r in results),
        'max_residual'       : max(r['residual'] for r in results),
        'random_results'     : random_results,
    }
    if random_results:
        summary['random_max_abs_energy_err'] = max(
            r['abs_energy_err'] for r in random_results)
        summary['random_in_window_fraction'] = (
            sum(1 for r in random_results if r['in_window']) / len(random_results))
    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    print(f'\n{"="*55}')
    print(f'Results → {args.out_json}')
    print(f'Max |ΔE|     = {summary["max_abs_energy_err"]:.2e} Ha  (eigsh states)')
    print(f'Max residual = {summary["max_residual"]:.2e}')
    if random_results:
        print(f'Max |ΔE|     = {summary["random_max_abs_energy_err"]:.2e} Ha  '
              f'(random states, sym vs fft)')
        print(f'In window    = {summary["random_in_window_fraction"]*100:.0f}%  '
              f'of {len(random_results)} random states')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
