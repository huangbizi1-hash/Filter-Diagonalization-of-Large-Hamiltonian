#!/usr/bin/env python3
"""compare_symbolic_explosion_ho3d.py

Chebyshev explosion filter on the 3D harmonic oscillator (ω=1, m=1)
using the Julia-compiled symbolic f(H) expression.

Pipeline
--------
Once per (cheb_m, E_lo, E_hi):
  1. Generate H^n.pkl (n = 0..cheb_m) by symbolic differentiation of
       H = -0.5 ∇² + 0.5(x²+y²+z²)    [V inserted as numeric constant]
     Cached in --cache_dir.
  2. Compute Chebyshev monomial coefficients  c_n  so that
       f(H) = T_m(aH+b) = Σ_n c_n H^n
  3. Assemble the symbolic f(H)·sin(k·r+b) expression and generate a
     Julia batch script (baseline: group_by_exp_combined + apply_horner +
     build_julia_batch_script).  Saved in --cache_dir as
       julia_filter_m{m}_Elo{E_lo}_Ehi{E_hi}.jl

Per N in --N_sweep:
  4. Build a symmetric grid (same convention as compare_fd_fft_explosion.py).
  5. Sample n_random random plane waves (kx, ky, kz, b).
  6. Call Julia to evaluate  psi_f = f(H)·sin(k·r+b)  for every wave.
  7. Rayleigh-Ritz on the filtered basis using H_FFT (exact kinetic energy).
  8. Collect and store Ritz eigenvalues.

Note: Ritz always uses H_FFT because the symbolic route can only provide
f(H)|ψ⟩ — it cannot apply H again to the filtered state numerically.

Usage
-----
  python compare_symbolic_explosion_ho3d.py \\
      --potential harmonic \\
      --N_sweep 10:12 --box_L 5.0 \\
      --cheb_m 5 --E_lo 8.0 --E_hi 80.0 \\
      --n_random 800 \\
      --n_print 30

  # Override Julia binary
  python compare_symbolic_explosion_ho3d.py ... --julia_exe /usr/local/bin/julia
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sympy as sp

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from symbolic_code.h_powers import generate_H_powers
from symbolic_code.chebyshev_filter import (
    chebyshev_coeffs_transformed,
    apply_f_of_H_from_raw_powers,
    extract_cos_sin_coeffs,
    group_by_exp_combined,
    apply_horner,
)
from symbolic_code.julia_codegen import build_julia_batch_script
from filter_core import svd_rayleigh_ritz_op


# ── exact HO levels ────────────────────────────────────────────────────────────

def ho3d_exact_levels(omega: float = 1.0, n_max_shell: int = 20) -> np.ndarray:
    evals = []
    for nx in range(n_max_shell + 1):
        for ny in range(n_max_shell + 1):
            for nz in range(n_max_shell + 1):
                evals.append(omega * (nx + ny + nz + 1.5))
    return np.sort(evals)


# ── grid (symmetric, matches compare_fd_fft_explosion.py) ─────────────────────

def make_grid(N: int, box_L: float):
    d   = 2.0 * box_L / N
    L   = (N - 1) * d / 2
    x1d = np.linspace(-L, L, N)
    return d, x1d


def make_T_k(N: int, d: float) -> np.ndarray:
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    return (k1d[:, None, None]**2 +
            k1d[None, :, None]**2 +
            k1d[None, None, :]**2) / 2.0


def apply_H_fft(psi_flat, V, T_k):
    N = T_k.shape[0]
    p3  = psi_flat.reshape(N, N, N)
    Tp3 = np.fft.ifftn(T_k * np.fft.fftn(p3)).real
    return (Tp3 + V * p3).ravel()


# ── H^n symbolic cache ─────────────────────────────────────────────────────────

def ensure_H_powers_cache(cache_dir: Path, m: int, omega: float = 1.0):
    """Generate H^0..H^m.pkl in cache_dir if not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check which powers already exist
    existing = set()
    for f in cache_dir.glob('H_power_*.pkl'):
        try:
            existing.add(int(f.stem.split('_')[-1]))
        except (ValueError, IndexError):
            pass

    needed = set(range(m + 1))
    missing = sorted(needed - existing)
    if not missing:
        print(f"  H^n cache: H^0..{m} all present in {cache_dir}")
        return

    print(f"  H^n cache: computing powers {missing} → {cache_dir}")
    x, y, z = sp.symbols('x y z')
    kx, ky, kz = sp.symbols('kx ky kz')
    kvec = (kx, ky, kz)
    k2   = kx**2 + ky**2 + kz**2
    # V as numeric constant (omega=1 baked in)
    V_sym = sp.Rational(1, 2) * (sp.Integer(1)**2) * (x**2 + y**2 + z**2)

    if 0 not in existing:
        generate_H_powers(
            m, cache_dir, file_format='pkl',
            V=V_sym, kvec=kvec, k2=k2, pref=0.5, x=x, y=y, z=z,
        )
    else:
        # Extend from the highest cached power
        import pickle
        max_cached = max(existing)
        start_from = max_cached
        with open(cache_dir / f'H_power_{start_from}.pkl', 'rb') as fh:
            data = pickle.load(fh)
        Ps, Pc = data['Ps'], data['Pc']
        for n in range(start_from + 1, m + 1):
            print(f"    extending H^{n} ...", end=' ', flush=True)
            from symbolic_code.h_powers import apply_H_on_pair
            Ps_H, Pc_H = apply_H_on_pair(Ps, Pc, V_sym, kvec, k2, 0.5, x, y, z)
            Ps = sp.expand(Ps_H)
            Pc = sp.expand(Pc_H)
            with open(cache_dir / f'H_power_{n}.pkl', 'wb') as fh:
                pickle.dump({'Ps': Ps, 'Pc': Pc}, fh)
            print('✓')


# ── Julia .jl cache ────────────────────────────────────────────────────────────

def ensure_julia_script(cache_dir: Path, m: int, E_lo: float, E_hi: float) -> Path:
    """Build and cache the Julia batch script for f(H) = T_m(aH+b)."""
    a    =  2.0 / (E_hi - E_lo)
    b_sc = -(E_hi + E_lo) / (E_hi - E_lo)

    # Key encodes window so different windows get different .jl files
    a_key  = f"{a:.8g}".replace('.', 'p').replace('-', 'm')
    b_key  = f"{b_sc:.8g}".replace('.', 'p').replace('-', 'm')
    jl_path = cache_dir / f'julia_filter_m{m}_a{a_key}_b{b_key}.jl'

    if jl_path.exists():
        print(f"  Julia script cache hit → {jl_path.name}")
        return jl_path

    print(f"  Assembling symbolic f(H)·ψ  (m={m}, E_lo={E_lo}, E_hi={E_hi}) ...")
    t0 = time.perf_counter()
    psi_fH = apply_f_of_H_from_raw_powers(cache_dir, m, a=a, b=b_sc)
    print(f"    f(H)·ψ assembled in {time.perf_counter()-t0:.1f}s")

    print("  Extracting cos/sin envelopes ...")
    t0 = time.perf_counter()
    expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)
    print(f"    done in {time.perf_counter()-t0:.1f}s")

    print("  Grouping by exp, applying Horner, generating Julia source ...")
    t0 = time.perf_counter()
    terms_cos = apply_horner(group_by_exp_combined(expr_cos))
    terms_sin = apply_horner(group_by_exp_combined(expr_sin))
    jl_src    = build_julia_batch_script(terms_cos, terms_sin)
    elapsed   = time.perf_counter() - t0
    jl_path.write_text(jl_src, encoding='utf-8')
    print(f"    done in {elapsed:.1f}s → {jl_path.name}")

    return jl_path


# ── run Julia batch evaluation ─────────────────────────────────────────────────

def julia_eval_filter(jl_path: Path, x1d: np.ndarray, k_vals: np.ndarray,
                      b_vals: np.ndarray, julia_exe: str) -> tuple:
    """Call Julia to evaluate f(H)*sin(k·r+b) for all waves on 3D grid.

    Returns
    -------
    C_f : np.ndarray  (n_waves, N, N, N)
    timing : dict
    """
    N       = len(x1d)
    n_waves = len(k_vals)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    with tempfile.TemporaryDirectory(prefix='sym_ho3d_') as tmp:
        tmp = Path(tmp)
        grid_bin  = tmp / 'grid.bin'
        kvals_bin = tmp / 'kvals.bin'
        out_bin   = tmp / 'out.bin'

        # Write grid (X_flat, Y_flat, Z_flat)
        with open(grid_bin, 'wb') as f:
            X.ravel().astype('<f8').tofile(f)
            Y.ravel().astype('<f8').tofile(f)
            Z.ravel().astype('<f8').tofile(f)

        # Write k-values: [kx0, ky0, kz0, b0, kx1, ...]
        kb = np.column_stack([k_vals, b_vals.reshape(-1, 1)]).astype('<f8')
        kb.ravel().tofile(str(kvals_bin))

        cmd = [julia_exe, str(jl_path),
               str(grid_bin), str(kvals_bin), str(out_bin),
               str(N**3), str(n_waves)]
        t_wall0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        total_wall_s = time.perf_counter() - t_wall0

        if proc.returncode != 0:
            raise RuntimeError(
                f"Julia exited with code {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        # Parse JSON timing from last stdout line
        timing = {'warmup_s': None, 'eval_s': None,
                  'n_eval_waves': None, 'total_wall_s': total_wall_s}
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    timing.update(json.loads(line))
                except json.JSONDecodeError:
                    pass
                break

        raw = np.fromfile(str(out_bin), dtype='<f8')

    C_f = raw.reshape(n_waves, N, N, N)
    return C_f, timing


# ── N sweep parser (same as compare_fd_fft_explosion.py) ──────────────────────

def parse_N_sweep(s: str) -> list:
    s = s.strip()
    if ':' in s:
        parts = [int(x) for x in s.split(':')]
        if len(parts) == 2: return list(range(parts[0], parts[1]))
        if len(parts) == 3: return list(range(parts[0], parts[1], parts[2]))
    return [int(x) for x in s.split(',') if x.strip()]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Symbolic Chebyshev explosion on 3D harmonic oscillator via Julia')
    ap.add_argument('--potential', choices=['harmonic'], default='harmonic',
                    help='Potential type (only harmonic supported)')
    ap.add_argument('--omega',    type=float, default=1.0,
                    help='Harmonic oscillator frequency (default 1.0; baked into pkl)')
    ap.add_argument('--N_sweep',  type=str,   default='10:12',
                    help='N values: "10:12" → [10,11], "10,12,14" → list')
    ap.add_argument('--box_L',    type=float, default=5.0,
                    help='Half-box length in Bohr (default 5.0)')
    ap.add_argument('--cheb_m',   type=int,   default=5,
                    help='Chebyshev polynomial order (default 5)')
    ap.add_argument('--E_lo',     type=float, default=8.0,
                    help='Chebyshev filter lower bound (default 8.0)')
    ap.add_argument('--E_hi',     type=float, default=80.0,
                    help='Chebyshev filter upper bound (default 80.0)')
    ap.add_argument('--n_random', type=int,   default=400,
                    help='Number of random plane waves (default 400)')
    ap.add_argument('--n_print',  type=int,   default=0,
                    help='Ritz eigenvalues to print/store per N (0 = all)')
    ap.add_argument('--seed',     type=int,   default=42)
    ap.add_argument('--k_max',    type=float, default=0.0,
                    help='Abs range for random k per axis; 0=auto (π/d_max)')
    ap.add_argument('--svd_tol',  type=float, default=1e-4)
    ap.add_argument('--cache_dir', type=str,  default='ho3d_symbolic_cache',
                    help='Directory for H^n.pkl and .jl cache (default ho3d_symbolic_cache/)')
    ap.add_argument('--julia_exe', type=str,  default='julia',
                    help='Julia executable (default: julia)')
    ap.add_argument('--out_json', type=str,   default='symbolic_ho3d_sweep.json')
    ap.add_argument('--fd_order', type=str,   default='',
                    help='(Unused; accepted for CLI compatibility with other scripts)')
    args = ap.parse_args()

    # ── parse N list ──────────────────────────────────────────────────────────
    N_list = parse_N_sweep(args.N_sweep)
    if not N_list:
        print("ERROR: --N_sweep produced empty list")
        sys.exit(1)

    n_print = args.n_print if args.n_print > 0 else None
    cache_dir = Path(args.cache_dir)

    print(f"=== Symbolic explosion on 3D HO ===")
    print(f"omega={args.omega}  box_L={args.box_L}  N_list={N_list}")
    print(f"cheb_m={args.cheb_m}  E_lo={args.E_lo}  E_hi={args.E_hi}")
    print(f"n_random={args.n_random}  seed={args.seed}")
    print(f"cache_dir={cache_dir}")
    print()

    t_wall_start = time.perf_counter()

    # ── Step 1: generate H^n.pkl ──────────────────────────────────────────────
    print("── Step 1: H^n symbolic cache ──")
    if args.omega != 1.0:
        print("WARNING: omega != 1.0 is not reflected in existing pkl files.")
        print("  The pkl files use V = 0.5*(x²+y²+z²) with omega=1 hardcoded.")
    ensure_H_powers_cache(cache_dir, args.cheb_m, omega=args.omega)
    print()

    # ── Step 2: Chebyshev coefficients ───────────────────────────────────────
    a    =  2.0 / (args.E_hi - args.E_lo)
    b_sc = -(args.E_hi + args.E_lo) / (args.E_hi - args.E_lo)
    coeffs = chebyshev_coeffs_transformed(args.cheb_m, a=a, b=b_sc)
    print(f"── Step 2: Chebyshev coefficients ──")
    print(f"  a={a:.6f}  b={b_sc:.6f}")
    print(f"  c = {[float(c) for c in coeffs]}")
    print()

    # ── Step 3: Julia .jl script ──────────────────────────────────────────────
    print("── Step 3: Julia filter script ──")
    jl_path = ensure_julia_script(cache_dir, args.cheb_m, args.E_lo, args.E_hi)
    print()

    # ── exact HO levels (reference) ───────────────────────────────────────────
    exact_all = ho3d_exact_levels(args.omega, n_max_shell=20)

    # ── Step 4-7: sweep over N ────────────────────────────────────────────────
    sweep = []

    for N in N_list:
        d, x1d = make_grid(N, args.box_L)
        X3, Y3, Z3 = np.meshgrid(x1d, x1d, x1d, indexing='ij')
        V3  = 0.5 * args.omega**2 * (X3**2 + Y3**2 + Z3**2)
        T_k = make_T_k(N, d)
        N3  = N**3

        # auto k_max: Nyquist of the finest grid in the sweep
        k_max = args.k_max if args.k_max > 0 else (np.pi / d)

        print(f"── N={N:3d}  d={d:.5f}  N³={N3}  k_max={k_max:.3f} ──")

        # reference eigsh
        print(f"  [ref] eigsh ...", end=' ', flush=True)
        from scipy.sparse.linalg import eigsh, LinearOperator
        H_linop = LinearOperator(
            (N3, N3),
            matvec=lambda v, _V=V3, _T=T_k: apply_H_fft(v, _V, _T),
            dtype=float,
        )
        t0 = time.perf_counter()
        n_ref = min(10, N3 - 2)
        E_ref, _ = eigsh(H_linop, k=n_ref, which='SA')
        t_ref = time.perf_counter() - t0
        E_ref = np.sort(E_ref.real)
        print(f"{t_ref:.2f}s  E_ref={np.round(E_ref[:5], 4).tolist()}")

        # random plane waves
        rng    = np.random.default_rng(args.seed)
        k_vals = rng.uniform(-k_max, k_max, (args.n_random, 3))
        b_vals = rng.uniform(0.0, 2 * np.pi, args.n_random)

        # Julia evaluation
        print(f"  [symbolic] calling Julia ({args.n_random} waves) ...", flush=True)
        t_julia0 = time.perf_counter()
        try:
            C_f, julia_timing = julia_eval_filter(
                jl_path, x1d, k_vals, b_vals, args.julia_exe)
            t_julia = time.perf_counter() - t_julia0
            ws = julia_timing.get('warmup_s')
            es = julia_timing.get('eval_s')
            nw = julia_timing.get('n_eval_waves')
            if es is not None and nw:
                per_wave_ms = es / nw * 1000
                print(f"    warmup={ws:.2f}s  eval={es:.2f}s/{nw} waves"
                      f"  ({per_wave_ms:.2f} ms/wave)  wall={t_julia:.1f}s")
            else:
                print(f"    wall={t_julia:.1f}s")
            julia_ok = True
        except Exception as exc:
            print(f"    ERROR: {exc}")
            julia_ok = False
            C_f = None
            julia_timing = {}
            t_julia = 0.0

        # Ritz with H_FFT
        ritz_evals = []
        svd_rank   = 0
        t_ritz     = 0.0
        if julia_ok and C_f is not None:
            # basis: columns are filtered wavefunctions (flat)
            basis = C_f.reshape(args.n_random, N3).T   # (N3, n_random)
            n_want = max(10, n_print or 0)
            print(f"  [ritz] SVD+Ritz  basis shape={basis.shape} ...", end=' ', flush=True)
            t0 = time.perf_counter()
            H_matvec = lambda v: apply_H_fft(v, V3, T_k)
            E_ritz, _, svd_rank = svd_rayleigh_ritz_op(
                basis, H_matvec,
                svd_tol=args.svd_tol, max_energies=n_want, hermitian=True)
            t_ritz = time.perf_counter() - t0
            ritz_evals = E_ritz.tolist()
            print(f"{t_ritz:.2f}s  rank={svd_rank}  E[0]={ritz_evals[0]:.6f}")
            ritz_show = ritz_evals[:n_print] if n_print else ritz_evals
            print(f"    Ritz ({len(ritz_show)} evals): {np.round(ritz_show, 6).tolist()}")

        # errors vs exact
        max_err  = float('nan')
        mean_err = float('nan')
        if ritz_evals:
            rv   = np.array(ritz_evals[:n_print] if n_print else ritz_evals)
            errs = [float(np.min(np.abs(exact_all - e))) for e in rv]
            max_err  = float(np.max(errs))
            mean_err = float(np.mean(errs))
            print(f"    max|ΔE vs exact|={max_err:.3e}  "
                  f"mean|ΔE|={mean_err:.3e}")

        sweep.append({
            'N'           : N,
            'd'           : d,
            'k_max'       : k_max,
            'E_ref'       : E_ref.tolist(),
            't_ref_s'     : t_ref,
            'julia_ok'    : julia_ok,
            'julia_timing': julia_timing,
            't_julia_s'   : t_julia,
            't_ritz_s'    : t_ritz,
            'svd_rank'    : svd_rank,
            'ritz_evals'  : ritz_evals[:n_print] if n_print else ritz_evals,
            'max_err_exact' : max_err,
            'mean_err_exact': mean_err,
        })
        print()

    wall_total = time.perf_counter() - t_wall_start

    # ── JSON output ───────────────────────────────────────────────────────────
    output = {
        'script'  : 'compare_symbolic_explosion_ho3d.py',
        'datetime': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'params'  : {
            'omega'     : args.omega,
            'box_L'     : args.box_L,
            'N_list'    : N_list,
            'cheb_m'    : args.cheb_m,
            'E_lo'      : args.E_lo,
            'E_hi'      : args.E_hi,
            'n_random'  : args.n_random,
            'n_print'   : args.n_print,
            'seed'      : args.seed,
            'svd_tol'   : args.svd_tol,
            'cache_dir' : str(cache_dir.resolve()),
            'julia_exe' : args.julia_exe,
            'cheb_a'    : a,
            'cheb_b'    : b_sc,
            'cheb_coeffs': [float(c) for c in coeffs],
            'jl_script' : str(jl_path.resolve()),
        },
        'sweep'       : sweep,
        'wall_total_s': wall_total,
    }
    Path(args.out_json).write_text(json.dumps(output, indent=2))
    print(f"JSON → {args.out_json}")
    print(f"Total wall time: {wall_total:.1f}s")


if __name__ == '__main__':
    main()
