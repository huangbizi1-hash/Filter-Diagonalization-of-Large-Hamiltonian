#!/usr/bin/env python3
"""test_gaussian_symbolic_energy.py

Chebyshev explosion filter on a 3-D Gaussian potential well
  V(r) = -A * exp(-B * r²)

Three paths are compared:
  [ref] scipy eigsh on FFT Hamiltonian – reference eigenvalues
  [fft] T_m(aH+b) applied to n_random states via FFT three-term recurrence,
        then Rayleigh-Ritz diagonalisation of the filtered subspace
  [sym] Same but the filter f(H) is assembled from symbolic H^n expressions
        (stored as sympy pkl) and evaluated at every grid point via Julia

Symbolic pipeline (one-time cost per (A,B,m,E_lo,E_hi)):
  1. Generate H^n pkl files   → H_powers_gaussian/A{A}_B{B}/H_power_{n}.pkl
  2. Assemble f(H) from H^n  → Pc_total, Ps_total (envelope functions)
  3. Build Julia scripts      → H_powers_gaussian/A{A}_B{B}/*.jl  (cached)
  4. One Julia call per batch, all random states processed per batch
     (minimises per-call JIT overhead)

Per-batch formula:
  f(H)|ψ⟩(r_i) = (1/N³) Σ_k [Re(c_k) fH_cos(k,r_i) − Im(c_k) fH_sin(k,r_i)]
  where fH_cos = f(H)|cos(k·r)⟩, fH_sin = f(H)|sin(k·r)⟩ (Julia outputs)

Output:
  • Per-state Rayleigh-quotient energies after f(H) normalisation
  • Ritz eigenvalues for both [fft] and [sym] subspaces
  • JSON with full results (--out_json)

Usage:
  python test_gaussian_symbolic_energy.py --n_random 4 --E_lo -3.0 --E_hi 20.0 --cheb_m 5
  python test_gaussian_symbolic_energy.py --A 10 --B 0.5 --d 0.5 --box_L 6 \\
      --n_random 8 --E_lo -6.0 --E_hi 20.0 --cheb_m 8 --n_levels 5
"""

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).parent))
from symbolic_code.h_powers import apply_H_on_pair, generate_H_powers
from symbolic_code.chebyshev_filter import apply_f_of_H_from_raw_powers
from symbolic_code.julia_codegen import build_julia_hn_cse_script
from filter_core import svd_rayleigh_ritz_op


# ── grid ──────────────────────────────────────────────────────────────────────

def make_grid(d: float, box_L: float):
    N = int(round(2.0 * box_L / d))
    x1d = np.arange(N) * d - box_L
    return N, x1d


def make_T_k(N: int, d: float) -> np.ndarray:
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    return (k1d[:, None, None]**2 +
            k1d[None, :, None]**2 +
            k1d[None, None, :]**2) / 2.0


# ── FFT Hamiltonian ───────────────────────────────────────────────────────────

def apply_H_fft(psi: np.ndarray, V_num: np.ndarray, T_k: np.ndarray) -> np.ndarray:
    """H|ψ⟩ via FFT (periodic BCs). psi may be 3-D or 1-D (auto-reshapes)."""
    sh = psi.shape
    p3 = psi.reshape(T_k.shape)
    res = np.fft.ifftn(T_k * np.fft.fftn(p3)).real + V_num * p3
    return res.reshape(sh)


def apply_chebyshev_fft(psi: np.ndarray, V_num: np.ndarray, T_k: np.ndarray,
                         m: int, E_lo: float, E_hi: float) -> np.ndarray:
    """T_m(aH+b)|ψ⟩ via three-term Chebyshev recurrence (FFT path)."""
    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)

    def Hs(phi):
        return a * apply_H_fft(phi, V_num, T_k) + b * phi

    y0 = psi.copy()
    if m == 0:
        return y0
    y1 = Hs(y0)
    for _ in range(2, m + 1):
        y1, y0 = 2.0 * Hs(y1) - y0, y1
    return y1


# ── H^n sympy cache ───────────────────────────────────────────────────────────

def _sympy_ingredients(A: float, B: float):
    x, y, z = sp.symbols('x y z', real=True)
    kx, ky, kz = sp.symbols('kx ky kz', real=True)
    V_sym = -A * sp.exp(-B * x**2) * sp.exp(-B * y**2) * sp.exp(-B * z**2)
    k2 = kx**2 + ky**2 + kz**2
    return x, y, z, kx, ky, kz, V_sym, k2


def ensure_h_powers(n_max: int, A: float, B: float, cache_base: str) -> Path:
    """Generate (or load from cache) H^0 … H^n_max pkl files.

    Cache directory: {cache_base}/A{A:g}_B{B:g}/
    Files: H_power_0.pkl … H_power_{n_max}.pkl
    """
    outdir = Path(cache_base) / f'A{A:g}_B{B:g}'
    outdir.mkdir(parents=True, exist_ok=True)

    missing = [n for n in range(n_max + 1)
               if not (outdir / f'H_power_{n}.pkl').exists()]
    if not missing:
        print(f'  [cache] H^0..H^{n_max} already cached in {outdir}')
        return outdir

    # Find the highest power already on disk to extend from
    cached = sorted(
        int(p.stem.split('_')[-1])
        for p in outdir.glob('H_power_*.pkl')
        if p.stem.split('_')[-1].isdigit()
    )

    x, y, z, kx, ky, kz, V_sym, k2 = _sympy_ingredients(A, B)
    kvec = (kx, ky, kz)

    if cached and cached[-1] < n_max:
        # Extend from the last cached power
        start = cached[-1]
        print(f'  [cache] Extending from H^{start} up to H^{n_max} in {outdir}')
        with open(outdir / f'H_power_{start}.pkl', 'rb') as fh:
            d = pickle.load(fh)
        Ps, Pc = d['Ps'], d['Pc']
        for n in range(start + 1, n_max + 1):
            print(f'    H^{n} ...', end=' ', flush=True)
            t0 = time.perf_counter()
            Ps_H, Pc_H = apply_H_on_pair(Ps, Pc, V_sym, kvec, k2, 0.5, x, y, z)
            Ps = sp.expand(Ps_H)
            Pc = sp.expand(Pc_H)
            with open(outdir / f'H_power_{n}.pkl', 'wb') as fh:
                pickle.dump({'Ps': Ps, 'Pc': Pc}, fh)
            print(f'✓  ({time.perf_counter()-t0:.1f}s)')
    else:
        # Generate from scratch
        print(f'  [cache] Generating H^0..H^{n_max} in {outdir}')
        generate_H_powers(
            n_max, outdir, file_format='pkl', expand=True,
            V=V_sym, kvec=kvec, k2=k2, pref=0.5, x=x, y=y, z=z,
        )

    return outdir


def assemble_fH_julia(cheb_m: int, E_lo: float, E_hi: float,
                       cache_dir: Path) -> tuple[Path, Path]:
    """Assemble f(H) from H^n pkl and return paths to Julia .jl files (cached).

    Returns (jl_sin_path, jl_cos_path):
      jl_sin evaluates f(H)|sin(k·r)⟩ = Pc_total*cos_p + Ps_total*sin_p
      jl_cos evaluates f(H)|cos(k·r)⟩ = Ps_total*cos_p − Pc_total*sin_p
        (derived from symmetry: Pc_cos_n = Ps_sin_n, Ps_cos_n = −Pc_sin_n)
    """
    tag = f'm{cheb_m}_Elo{E_lo:g}_Ehi{E_hi:g}'
    jl_sin = cache_dir / f'fH_sin_{tag}.jl'
    jl_cos = cache_dir / f'fH_cos_{tag}.jl'

    if jl_sin.exists() and jl_cos.exists():
        print(f'  [cache] Julia scripts found: {jl_sin.name}, {jl_cos.name}')
        return jl_sin, jl_cos

    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)
    print(f'  [sym] Assembling f(H) (m={cheb_m}, a={a:.4f}, b={b:.4f}) ...', flush=True)
    t0 = time.perf_counter()
    Pc_total, Ps_total = apply_f_of_H_from_raw_powers(
        cache_dir, cheb_m, a, b, file_type='pkl', return_envelopes=True)
    print(f'  [sym] Assembly done in {time.perf_counter()-t0:.1f}s  '
          f'ops_Pc={sp.count_ops(Pc_total)}  ops_Ps={sp.count_ops(Ps_total)}')

    print('  [sym] Building Julia scripts (CSE) ...', flush=True)
    t0 = time.perf_counter()
    # sin-start: out = Pc*cos_p + Ps*sin_p  →  f(H)|sin(k·r)⟩
    src_sin, ops_sin = build_julia_hn_cse_script(Pc_total, Ps_total)
    # cos-start (by symmetry): out = Ps*cos_p + (-Pc)*sin_p  →  f(H)|cos(k·r)⟩
    src_cos, ops_cos = build_julia_hn_cse_script(Ps_total, -Pc_total)
    print(f'  [sym] Codegen done in {time.perf_counter()-t0:.1f}s  '
          f'ops_sin={ops_sin}  ops_cos={ops_cos}')

    jl_sin.write_text(src_sin)
    jl_cos.write_text(src_cos)
    print(f'  [sym] Saved {jl_sin.name}  ({len(src_sin):,} bytes)')
    print(f'  [sym] Saved {jl_cos.name}  ({len(src_cos):,} bytes)')
    return jl_sin, jl_cos


# ── Julia batch runner ────────────────────────────────────────────────────────

def _call_julia(jl_file: Path, Xf, Yf, Zf, kx_b, ky_b, kz_b,
                work_dir: str, julia_exe: str) -> np.ndarray:
    """Call Julia script for one batch of k-vectors.

    Returns out (n_waves, N_grid) float64.
    out[j, i] = evaluated Julia expression for k-vector j at grid point i.
    """
    N_grid = Xf.size
    n_waves = len(kx_b)
    wd = Path(work_dir)
    grid_f = wd / '_grid.bin'
    k_f    = wd / '_kvals.bin'
    out_f  = wd / '_out.bin'

    with open(grid_f, 'wb') as fh:
        Xf.astype('<f8').tofile(fh)
        Yf.astype('<f8').tofile(fh)
        Zf.astype('<f8').tofile(fh)
    kb = np.column_stack([kx_b, ky_b, kz_b, np.zeros(n_waves)]).astype('<f8')
    kb.ravel().tofile(str(k_f))

    proc = subprocess.run(
        [julia_exe, str(jl_file),
         str(grid_f), str(k_f), str(out_f), str(N_grid), str(n_waves)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'Julia failed (rc={proc.returncode}):\n'
                           f'{proc.stderr[:3000]}')

    return np.fromfile(str(out_f), dtype='<f8').reshape(n_waves, N_grid)


# ── symbolic f(H) applied to multiple states ──────────────────────────────────

def apply_fH_sym_all(jl_sin: Path, jl_cos: Path,
                     psi_list: list, N: int, x1d: np.ndarray,
                     julia_exe: str, work_dir: str,
                     batch_size: int) -> tuple[list, float]:
    """Apply symbolic f(H) to all states in psi_list simultaneously.

    f(H)|ψ⟩(r_i) = (1/N³) Σ_k [Re(c_k) fH_cos(k,r_i) − Im(c_k) fH_sin(k,r_i)]

    Processes all states in one sweep over k-vector batches, so Julia is called
    only 2 * n_batches times (sin + cos scripts) regardless of len(psi_list).

    Returns (fHpsi_list, julia_time_s).
    """
    N3  = N**3
    d   = float(x1d[1] - x1d[0])
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    Xf  = X.ravel().astype(np.float64)
    Yf  = Y.ravel().astype(np.float64)
    Zf  = Z.ravel().astype(np.float64)

    kfreq = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    Kx, Ky, Kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    kx_all = Kx.ravel()
    ky_all = Ky.ravel()
    kz_all = Kz.ravel()

    n_states   = len(psi_list)
    n_batches  = (N3 + batch_size - 1) // batch_size

    # FFT all states up front: c_all[s, k] = FFT(psi_s).ravel()[k]
    c_all  = np.array([np.fft.fftn(p).ravel() for p in psi_list],
                      dtype=np.complex128)              # (n_states, N3)
    c_real = c_all.real                                  # (n_states, N3)
    c_imag = c_all.imag                                  # (n_states, N3)

    fHpsi = np.zeros((n_states, N3), dtype=np.float64)
    t_julia = 0.0

    for bi in range(n_batches):
        s = bi * batch_size
        e = min(s + batch_size, N3)
        kx_b = kx_all[s:e]
        ky_b = ky_all[s:e]
        kz_b = kz_all[s:e]

        t0 = time.perf_counter()
        # fH_sin[j, i] = f(H)|sin(k_j·r)⟩(r_i)  shape (batch, N3)
        fH_sin = _call_julia(jl_sin, Xf, Yf, Zf, kx_b, ky_b, kz_b, work_dir, julia_exe)
        # fH_cos[j, i] = f(H)|cos(k_j·r)⟩(r_i)  shape (batch, N3)
        fH_cos = _call_julia(jl_cos, Xf, Yf, Zf, kx_b, ky_b, kz_b, work_dir, julia_exe)
        t_julia += time.perf_counter() - t0

        # Accumulate for all states simultaneously:
        # fHpsi[s_i, :] += Σ_j [Re(c_j^{s_i}) fH_cos[j,:] - Im(c_j^{s_i}) fH_sin[j,:]] / N3
        # = (c_real[s_i, s:e] @ fH_cos - c_imag[s_i, s:e] @ fH_sin) / N3
        fHpsi += (c_real[:, s:e] @ fH_cos - c_imag[:, s:e] @ fH_sin) / N3

    return [fHpsi[i].reshape(N, N, N) for i in range(n_states)], t_julia


# ── normalisation / Rayleigh quotient ─────────────────────────────────────────

def normalize(psi: np.ndarray, d: float):
    norm2 = float(np.sum(psi**2) * d**3)
    if norm2 < 1e-30:
        return None, 0.0
    return psi / np.sqrt(norm2), norm2


def rayleigh_quotient(psi_n: np.ndarray, V_num: np.ndarray,
                      T_k: np.ndarray, d: float) -> float:
    Hpsi = apply_H_fft(psi_n, V_num, T_k)
    return float(np.sum(psi_n * Hpsi) * d**3)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Gaussian potential Chebyshev explosion filter: '
                    'symbolic Julia vs FFT Hamiltonian.')
    ap.add_argument('--A',         type=float, default=10.0,
                    help='Gaussian depth (Ha)  [default 10.0]')
    ap.add_argument('--B',         type=float, default=0.5,
                    help='Gaussian exponent (Bohr⁻²)  [default 0.5]')
    ap.add_argument('--d',         type=float, default=0.5,
                    help='Grid spacing (Bohr)  [default 0.5]')
    ap.add_argument('--box_L',     type=float, default=6.0,
                    help='Half-box length (Bohr)  [default 6.0]')
    ap.add_argument('--n_levels',  type=int,   default=5,
                    help='Eigsh levels for reference  [default 5]')
    ap.add_argument('--n_random',  type=int,   default=4,
                    help='Number of random initial states  [default 4]')
    ap.add_argument('--E_lo',      type=float, default=-3.0,
                    help='Lower energy bound for Chebyshev filter  [default -3.0]')
    ap.add_argument('--E_hi',      type=float, default=20.0,
                    help='Upper energy bound for Chebyshev filter  [default 20.0]')
    ap.add_argument('--cheb_m',    type=int,   default=8,
                    help='Chebyshev polynomial order  [default 8]')
    ap.add_argument('--seed',      type=int,   default=42,
                    help='RNG seed  [default 42]')
    ap.add_argument('--batch_size', type=int,  default=0,
                    help='k-vectors per Julia subprocess call. '
                         '0 = auto (N³ for small grids, 4096 otherwise)  [default 0]')
    ap.add_argument('--julia',     type=str,   default='julia',
                    help='Julia executable  [default julia]')
    ap.add_argument('--cache_base', type=str,  default='H_powers_gaussian',
                    help='Root directory for H^n / Julia caches  [default H_powers_gaussian]')
    ap.add_argument('--svd_tol',   type=float, default=1e-4,
                    help='SVD truncation threshold for Ritz  [default 1e-4]')
    ap.add_argument('--out_json',  type=str,   default='gaussian_explosion_test.json',
                    help='Output JSON path  [default gaussian_explosion_test.json]')
    args = ap.parse_args()

    t_total = time.perf_counter()

    # ── grid ──────────────────────────────────────────────────────────────────
    N, x1d = make_grid(args.d, args.box_L)
    d = float(x1d[1] - x1d[0])
    N3 = N**3
    print(f'Grid: N={N}  N³={N3}  d={d:.4f} Bohr  box_L={args.box_L} Bohr')
    print(f'Potential: V = -{args.A} * exp(-{args.B} * r²)')
    print(f'Filter: T_{args.cheb_m}(aH+b)  E_lo={args.E_lo}  E_hi={args.E_hi}')
    print(f'        a={2/(args.E_hi-args.E_lo):.4f}  '
          f'b={-(args.E_hi+args.E_lo)/(args.E_hi-args.E_lo):.4f}')

    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    V_num = -args.A * np.exp(-args.B * (X**2 + Y**2 + Z**2))
    T_k   = make_T_k(N, d)

    batch_size = args.batch_size if args.batch_size > 0 else (N3 if N3 <= 30000 else 4096)
    n_batches  = (N3 + batch_size - 1) // batch_size
    print(f'Julia batch_size={batch_size}  n_batches={n_batches}')

    # ── reference: eigsh ─────────────────────────────────────────────────────
    from scipy.sparse.linalg import LinearOperator
    print(f'\n[ref] eigsh ({args.n_levels} lowest levels) ...')
    t0 = time.perf_counter()
    H_linop = LinearOperator(
        (N3, N3),
        matvec=lambda v: apply_H_fft(v, V_num, T_k),
        dtype=float,
    )
    E_ref, psi_ref = eigsh(H_linop, k=args.n_levels, which='SA')
    t_ref = time.perf_counter() - t0
    idx = np.argsort(E_ref)
    E_ref = E_ref[idx]
    print(f'  Done in {t_ref:.2f}s')
    for i, e in enumerate(E_ref):
        print(f'  E_ref[{i}] = {e:.8f} Ha')

    # ── generate random states ─────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    psi_rand_list = [
        rng.standard_normal((N, N, N)).astype(np.float64)
        for _ in range(args.n_random)
    ]
    # normalise each
    psi_rand_list = [
        p / np.sqrt(float(np.sum(p**2) * d**3)) for p in psi_rand_list
    ]

    # ── FFT explosion filter ──────────────────────────────────────────────────
    print(f'\n[fft] Chebyshev explosion filter (m={args.cheb_m}) ...')
    t0 = time.perf_counter()
    fft_filtered = []
    fft_energies = []
    for i, psi in enumerate(psi_rand_list):
        psi_f = apply_chebyshev_fft(psi, V_num, T_k, args.cheb_m, args.E_lo, args.E_hi)
        psi_fn, _ = normalize(psi_f, d)
        if psi_fn is None:
            print(f'  [fft] state[{i}]: negligible norm after filter, skipping')
            continue
        E_f = rayleigh_quotient(psi_fn, V_num, T_k, d)
        fft_filtered.append(psi_fn)
        fft_energies.append(E_f)
        in_win = args.E_lo <= E_f <= args.E_hi
        print(f'  [fft] state[{i:03d}]: E = {E_f:.8f} Ha  in_window={in_win}')
    t_fft_filter = time.perf_counter() - t0
    print(f'  Filter time: {t_fft_filter:.2f}s')

    # Ritz on FFT-filtered subspace
    print('\n[fft] Rayleigh-Ritz ...')
    t0 = time.perf_counter()
    basis_fft = np.column_stack([p.ravel() for p in fft_filtered])  # (N3, n)
    E_ritz_fft, _, rank_fft = svd_rayleigh_ritz_op(
        basis_fft,
        lambda v: apply_H_fft(v, V_num, T_k),
        svd_tol=args.svd_tol,
        max_energies=args.n_levels + 5,
        hermitian=True,
    )
    t_ritz_fft = time.perf_counter() - t0
    print(f'  rank={rank_fft}  time={t_ritz_fft:.2f}s')
    print(f'  Ritz eigenvalues: {np.round(E_ritz_fft[:args.n_levels], 6).tolist()}')

    # ── symbolic H^n cache ───────────────────────────────────────────────────
    print(f'\n[sym] Ensuring H^0..H^{args.cheb_m} pkl cache ...')
    t0 = time.perf_counter()
    cache_dir = ensure_h_powers(args.cheb_m, args.A, args.B, args.cache_base)
    t_cache = time.perf_counter() - t0
    print(f'  Cache ready in {t_cache:.1f}s  ({cache_dir})')

    # ── build / load Julia scripts ────────────────────────────────────────────
    print('\n[sym] Building Julia f(H) scripts ...')
    t0 = time.perf_counter()
    jl_sin, jl_cos = assemble_fH_julia(args.cheb_m, args.E_lo, args.E_hi, cache_dir)
    t_codegen = time.perf_counter() - t0
    print(f'  Codegen total time: {t_codegen:.1f}s')

    # ── symbolic explosion filter ─────────────────────────────────────────────
    print(f'\n[sym] Applying symbolic f(H) to {args.n_random} states '
          f'(batch_size={batch_size}, n_batches={n_batches}) ...')
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='gexpl_') as td:
        fH_sym_list, t_julia = apply_fH_sym_all(
            jl_sin, jl_cos, psi_rand_list, N, x1d,
            args.julia, td, batch_size,
        )
    t_sym_filter = time.perf_counter() - t0
    print(f'  Julia time: {t_julia:.1f}s  total: {t_sym_filter:.1f}s')

    sym_filtered = []
    sym_energies = []
    sym_results  = []
    print()
    for i, psi_f in enumerate(fH_sym_list):
        psi_fn, _ = normalize(psi_f, d)
        if psi_fn is None:
            print(f'  [sym] state[{i}]: negligible norm after filter, skipping')
            sym_results.append({'idx': i, 'skipped': True})
            continue
        E_s = rayleigh_quotient(psi_fn, V_num, T_k, d)
        sym_filtered.append(psi_fn)
        sym_energies.append(E_s)
        in_win = args.E_lo <= E_s <= args.E_hi
        dE = abs(E_s - fft_energies[i]) if i < len(fft_energies) else float('nan')
        print(f'  [sym] state[{i:03d}]: E_sym={E_s:.8f} Ha  '
              f'E_fft={fft_energies[i]:.8f} Ha  '
              f'|ΔE|={dE:.2e}  in_window={in_win}')
        sym_results.append({
            'idx': i,
            'E_sym': float(E_s),
            'E_fft': float(fft_energies[i]) if i < len(fft_energies) else None,
            'abs_dE': float(dE),
            'in_window': in_win,
        })

    # Ritz on symbolic-filtered subspace
    if sym_filtered:
        print('\n[sym] Rayleigh-Ritz ...')
        t0 = time.perf_counter()
        basis_sym = np.column_stack([p.ravel() for p in sym_filtered])
        E_ritz_sym, _, rank_sym = svd_rayleigh_ritz_op(
            basis_sym,
            lambda v: apply_H_fft(v, V_num, T_k),
            svd_tol=args.svd_tol,
            max_energies=args.n_levels + 5,
            hermitian=True,
        )
        t_ritz_sym = time.perf_counter() - t0
        print(f'  rank={rank_sym}  time={t_ritz_sym:.2f}s')
        print(f'  Ritz eigenvalues: {np.round(E_ritz_sym[:args.n_levels], 6).tolist()}')
    else:
        E_ritz_sym = np.array([])
        t_ritz_sym = 0.0
        rank_sym   = 0

    t_wall = time.perf_counter() - t_total

    # ── summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Reference eigenvalues (eigsh):')
    for i, e in enumerate(E_ref):
        print(f'  E_ref[{i}] = {e:.8f} Ha')
    print(f'\n[fft] Per-state energies:  '
          + '  '.join(f'{e:.5f}' for e in fft_energies))
    print(f'[fft] Ritz: ' + '  '.join(f'{e:.5f}' for e in E_ritz_fft[:args.n_levels]))
    if sym_energies:
        print(f'[sym] Per-state energies:  '
              + '  '.join(f'{e:.5f}' for e in sym_energies))
        print(f'[sym] Ritz: ' + '  '.join(f'{e:.5f}' for e in E_ritz_sym[:args.n_levels]))
        max_dE = max(r['abs_dE'] for r in sym_results if not r.get('skipped'))
        print(f'\nMax |ΔE(sym−fft)| per state = {max_dE:.2e} Ha')
    print(f'\nTotal wall time: {t_wall:.1f}s')
    print(f'{"="*60}')

    # ── JSON output ───────────────────────────────────────────────────────────
    summary = {
        'params': {
            'A': args.A, 'B': args.B, 'd': d, 'box_L': args.box_L,
            'N': N, 'N3': N3,
            'n_random': args.n_random, 'n_levels': args.n_levels,
            'E_lo': args.E_lo, 'E_hi': args.E_hi, 'cheb_m': args.cheb_m,
            'seed': args.seed, 'batch_size': batch_size,
        },
        'timings': {
            'ref_eigsh_s'  : t_ref,
            'h_power_cache_s': t_cache,
            'codegen_s'    : t_codegen,
            'fft_filter_s' : t_fft_filter,
            'sym_filter_s' : t_sym_filter,
            'julia_total_s': t_julia,
            'wall_total_s' : t_wall,
        },
        'E_ref'             : E_ref.tolist(),
        'fft_state_energies': fft_energies,
        'fft_ritz'          : E_ritz_fft.tolist(),
        'fft_ritz_rank'     : int(rank_fft),
        'sym_state_results' : sym_results,
        'sym_ritz'          : E_ritz_sym.tolist(),
        'sym_ritz_rank'     : int(rank_sym),
    }
    if sym_results and any(not r.get('skipped') for r in sym_results):
        summary['max_abs_dE_state'] = max(
            r['abs_dE'] for r in sym_results if not r.get('skipped'))
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f'Results → {args.out_json}')


if __name__ == '__main__':
    main()
