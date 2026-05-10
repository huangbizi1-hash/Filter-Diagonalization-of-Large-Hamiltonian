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

def build_fft_H_op(N: int, x1d: np.ndarray, V_num: np.ndarray):
    """Build scipy LinearOperator for H = -½∇² + V using numpy.fft.

    T̂ψ = IFFT[ (k²/2) · FFT(ψ) ]   (periodic BCs, pseudospectral)
    """
    d = float(x1d[1] - x1d[0])
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    T_k = (k1d[:, None, None] ** 2 +
           k1d[None, :, None] ** 2 +
           k1d[None, None, :] ** 2) / 2.0

    def matvec(v):
        psi = v.reshape(N, N, N)
        Tpsi = np.fft.ifftn(T_k * np.fft.fftn(psi)).real
        return (Tpsi + V_num * psi).ravel()

    return LinearOperator((N ** 3, N ** 3), matvec=matvec, dtype=float)


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
    args = ap.parse_args()

    # ── Grid ────────────────────────────────────────────────────────────────
    N, x1d = make_grid(args.d, args.box_L)
    d      = float(x1d[1] - x1d[0])     # actual spacing (may differ slightly from args.d)
    N3     = N ** 3
    print(f'Grid: N={N} per axis  N³={N3}  d={d:.4f} Bohr  '
          f'box_L={args.box_L} Bohr')
    print(f'Potential: V = -{args.A} * exp(-{args.B} * r²)')

    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    V_num   = -args.A * np.exp(-args.B * (X ** 2 + Y ** 2 + Z ** 2))

    # ── Reference: FFT + eigsh ───────────────────────────────────────────────
    print('\n[ref] Building FFT H operator and solving with eigsh ...')
    H_fft  = build_fft_H_op(N, x1d, V_num)
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

    # ── Save JSON ────────────────────────────────────────────────────────────
    summary = {
        'params': {
            'A'        : args.A,
            'B'        : args.B,
            'd'        : d,
            'box_L'    : args.box_L,
            'N'        : N,
            'N3'       : N3,
            'n_levels' : args.n_levels,
            'batch_size': args.batch_size,
        },
        'ops_cos'            : ops_cos,
        'ops_sin'            : ops_sin,
        'time_ref_s'         : t_ref,
        'time_sym_build_s'   : t_sym_build,
        'results'            : results,
        'max_abs_energy_err' : max(r['abs_energy_err'] for r in results),
        'max_residual'       : max(r['residual'] for r in results),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))

    print(f'\n{"="*55}')
    print(f'Results → {args.out_json}')
    print(f'Max |ΔE|     = {summary["max_abs_energy_err"]:.2e} Ha')
    print(f'Max residual = {summary["max_residual"]:.2e}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
