#!/usr/bin/env python3
"""compare_symbolic_fft_ho3d_old_but_correct_and_fast.py

Self-contained restoration of the original working symbolic vs FFT
Chebyshev filter comparison.  Does NOT import from
compare_symbolic_explosion_ho3d.py (which has been heavily modified).
All pipeline functions are defined inline.

Key differences from the current (broken) version:
  * julia_eval_filter: original call (no -t auto, no per-thread buffers)
  * ensure_julia_script: supports --force_rebuild to delete stale .jl cache
  * _load_H_raw_powers_safe: numeric sort + filters out _horner.pkl files

Usage
-----
  python compare_symbolic_fft_ho3d_old_but_correct_and_fast.py \\
      --N_sweep 7:15 --box_L 4.0 \\
      --cheb_m 10 --E_lo 7.0 --E_hi 70.0 \\
      --n_random 400 --n_print 20

  # Force rebuild the Julia script (use this if you get NaN):
  python compare_symbolic_fft_ho3d_old_but_correct_and_fast.py \\
      --N_sweep 7:15 --box_L 4.0 \\
      --cheb_m 10 --E_lo 7.0 --E_hi 70.0 \\
      --n_random 400 --n_print 20 --force_rebuild
"""

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.sparse.linalg import eigsh, LinearOperator

sys.path.insert(0, str(Path(__file__).parent))

from symbolic_code.chebyshev_filter import (
    chebyshev_coeffs_transformed,
    extract_cos_sin_coeffs,
    group_by_exp_combined,
    apply_horner,
)
from symbolic_code.julia_codegen import build_julia_batch_script
from symbolic_code.h_powers import generate_H_powers
from filter_core import svd_rayleigh_ritz_op


# ── safe H^n loader (numeric sort + skip _horner.pkl) ─────────────────────────

def _load_H_raw_powers_safe(folder: Path, n_max: int) -> dict:
    """Load H_power_n.pkl files, sorted numerically, skipping *_horner.pkl."""
    folder = Path(folder)

    def _n(p):
        return int(p.name.replace('.pkl', '').split('_')[-1])

    def _is_plain(p):
        try:
            _n(p)
            return True
        except ValueError:
            return False

    results = {}
    for fpath in sorted(filter(_is_plain, folder.glob('H_power_*.pkl')), key=_n):
        n = _n(fpath)
        with open(fpath, 'rb') as fh:
            results[n] = pickle.load(fh)
        if n >= n_max:
            break
    return results


def _apply_f_of_H_from_raw_powers(cache_dir: Path, m: int, a: float, b: float):
    """Assemble f(H)*psi = Σ c_n H^n*psi using safe loader."""
    coeffs  = chebyshev_coeffs_transformed(m, a=a, b=b)
    results = _load_H_raw_powers_safe(cache_dir, m)

    kx, ky, kz, bsym = sp.symbols('kx ky kz b')
    x, y, z = sp.symbols('x y z')
    theta = kx * x + ky * y + kz * z + bsym

    Ps_total = sp.Integer(0)
    Pc_total = sp.Integer(0)
    for n, c in enumerate(coeffs):
        if c == 0:
            continue
        if n == 0:
            Ps_total += c
        else:
            entry = results.get(n)
            if entry is None:
                raise KeyError(
                    f"H^{n} not found in {cache_dir!r}; "
                    f"run with cheb_m >= {n} first")
            Ps_total += c * entry['Ps']
            Pc_total += c * entry['Pc']

    return Ps_total * sp.sin(theta) + Pc_total * sp.cos(theta)


# ── H^n symbolic cache ─────────────────────────────────────────────────────────

def ensure_H_powers_cache(cache_dir: Path, m: int):
    """Generate H^0..H^m.pkl in cache_dir if not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing = set()
    for f in cache_dir.glob('H_power_*.pkl'):
        try:
            n = int(f.stem.replace('H_power_', '').split('_')[0])
            existing.add(n)
        except (ValueError, IndexError):
            pass

    needed  = set(range(m + 1))
    missing = sorted(needed - existing)
    if not missing:
        print(f"  H^n cache: H^0..{m} all present in {cache_dir}")
        return

    print(f"  H^n cache: computing powers {missing} → {cache_dir}")
    x, y, z = sp.symbols('x y z')
    kx, ky, kz = sp.symbols('kx ky kz')
    V_sym = sp.Rational(1, 2) * (x**2 + y**2 + z**2)
    generate_H_powers(
        m, cache_dir, file_format='pkl',
        V=V_sym, kvec=(kx, ky, kz), k2=kx**2 + ky**2 + kz**2,
        pref=0.5, x=x, y=y, z=z,
    )


# ── Julia script cache ─────────────────────────────────────────────────────────

def ensure_julia_script(cache_dir: Path, m: int,
                        E_lo: float, E_hi: float,
                        force_rebuild: bool = False) -> Path:
    """Build and cache the Julia batch script for f(H) = T_m(aH+b).

    Parameters
    ----------
    force_rebuild : bool
        If True, delete any existing .jl and rebuild from scratch.
        Use this when you suspect the cached script is stale/wrong (NaN results).
    """
    a    =  2.0 / (E_hi - E_lo)
    b_sc = -(E_hi + E_lo) / (E_hi - E_lo)

    a_key  = f"{a:.8g}".replace('.', 'p').replace('-', 'm')
    b_key  = f"{b_sc:.8g}".replace('.', 'p').replace('-', 'm')
    jl_path = cache_dir / f'julia_filter_m{m}_a{a_key}_b{b_key}.jl'

    if force_rebuild and jl_path.exists():
        print(f"  --force_rebuild: deleting stale {jl_path.name}")
        jl_path.unlink()

    if jl_path.exists():
        print(f"  Julia script cache hit → {jl_path.name}")
        return jl_path

    print(f"  Assembling symbolic f(H)*psi  (m={m}, E_lo={E_lo}, E_hi={E_hi}) ...")
    t0 = time.perf_counter()
    psi_fH = _apply_f_of_H_from_raw_powers(cache_dir, m, a=a, b=b_sc)
    print(f"    f(H)*psi assembled in {time.perf_counter()-t0:.1f}s")

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


# ── Julia batch evaluation (original call, no -t auto) ────────────────────────

def julia_eval_filter(jl_path: Path, x1d: np.ndarray,
                      k_vals: np.ndarray, b_vals: np.ndarray,
                      julia_exe: str) -> tuple:
    """Call Julia to evaluate f(H)*sin(k*r+b) for all waves on 3D grid.

    Returns
    -------
    C_f : np.ndarray  (n_waves, N, N, N)
    timing : dict
    """
    N       = len(x1d)
    n_waves = len(k_vals)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    with tempfile.TemporaryDirectory(prefix='sym_ho3d_') as tmp:
        tmp       = Path(tmp)
        grid_bin  = tmp / 'grid.bin'
        kvals_bin = tmp / 'kvals.bin'
        out_bin   = tmp / 'out.bin'

        with open(grid_bin, 'wb') as f:
            X.ravel().astype('<f8').tofile(f)
            Y.ravel().astype('<f8').tofile(f)
            Z.ravel().astype('<f8').tofile(f)

        kb = np.column_stack([k_vals, b_vals.reshape(-1, 1)]).astype('<f8')
        kb.ravel().tofile(str(kvals_bin))

        cmd = [julia_exe, str(jl_path),
               str(grid_bin), str(kvals_bin), str(out_bin),
               str(N**3), str(n_waves)]

        t_wall0 = time.perf_counter()
        proc    = subprocess.run(cmd, capture_output=True, text=True)
        total_wall_s = time.perf_counter() - t_wall0

        if proc.returncode != 0:
            raise RuntimeError(
                f"Julia exited with code {proc.returncode}.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

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


# ── grid helpers ──────────────────────────────────────────────────────────────

def ho3d_exact_levels(omega: float = 1.0, n_max_shell: int = 20) -> np.ndarray:
    evals = []
    for nx in range(n_max_shell + 1):
        for ny in range(n_max_shell + 1):
            for nz in range(n_max_shell + 1):
                evals.append(omega * (nx + ny + nz + 1.5))
    return np.sort(evals)


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
    N   = T_k.shape[0]
    p3  = psi_flat.reshape(N, N, N)
    Tp3 = np.fft.ifftn(T_k * np.fft.fftn(p3)).real
    return (Tp3 + V * p3).ravel()


def parse_N_sweep(s: str) -> list:
    s = s.strip()
    if ':' in s:
        lo, hi = (int(x) for x in s.split(':'))
        return list(range(lo, hi))
    return [int(x) for x in s.split(',')]


# ── Chebyshev 3-term recurrence (FFT path) ────────────────────────────────────

def chebyshev_recurrence(H_apply, psi0: np.ndarray,
                         a: float, b_sc: float, m: int) -> np.ndarray:
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


# ── one N sweep step ──────────────────────────────────────────────────────────

def run_one_N(N, box_L, cheb_m, a, b_sc,
              n_random, k_max_arg, seed, svd_tol, n_print,
              jl_path, julia_exe, exact_all) -> dict:

    d, x1d = make_grid(N, box_L)
    assert abs((x1d[1] - x1d[0]) - d) < 1e-12

    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    V3  = 0.5 * (X**2 + Y**2 + Z**2)
    T_k = make_T_k(N, d)
    N3  = N ** 3
    k_max = k_max_arg if k_max_arg > 0.0 else np.pi / d

    print(f"\n-- N={N}  d={d:.5f}  L={(N-1)*d/2:.4f}  N3={N3}  "
          f"k_max={k_max:.3f} --")

    def H_matvec(v):
        return apply_H_fft(v, V3, T_k)

    H_linop = LinearOperator((N3, N3), matvec=H_matvec, dtype=float)

    n_ref = min(10, N3 - 2)
    print(f"  [ref] eigsh (k={n_ref}) ...", end=' ', flush=True)
    t0 = time.perf_counter()
    E_ref, _ = eigsh(H_linop, k=n_ref, which='SA')
    t_ref = time.perf_counter() - t0
    E_ref = np.sort(E_ref.real)
    print(f"{t_ref:.2f}s  E_ref[:5]={np.round(E_ref[:5], 5).tolist()}")

    rng    = np.random.default_rng(seed)
    k_vals = rng.uniform(-k_max, k_max, (n_random, 3))
    b_vals = rng.uniform(0.0, 2 * np.pi, n_random)

    # ── symbolic path ─────────────────────────────────────────────────────────
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
            print(f"    warmup={ws:.4f}s  eval={es:.4f}s/{nw} waves"
                  f"  ({es/nw*1e3:.3f} ms/wave)  wall={t_julia:.1f}s")
        else:
            print(f"    wall={t_julia:.1f}s")
        nan_frac = float(np.isnan(C_f_sym).mean())
        if nan_frac > 0:
            print(f"    WARNING: {nan_frac*100:.1f}% NaN in Julia output "
                  f"-- rerun with --force_rebuild to rebuild the .jl script")
        julia_ok = nan_frac < 0.5
    except Exception as exc:
        print(f"    ERROR: {exc}")
        julia_ok    = False
        C_f_sym     = None
        julia_timing = {}
        t_julia     = 0.0

    sym_result = {"ok": julia_ok, "julia_timing": julia_timing,
                  "t_julia_s": t_julia, "ritz_evals": [],
                  "rank": 0, "t_ritz_s": 0.0}

    if julia_ok and C_f_sym is not None:
        good = ~np.isnan(C_f_sym).any(axis=(1, 2, 3))
        C_f_clean = C_f_sym[good]
        if C_f_clean.shape[0] < 10:
            print(f"    too few clean waves ({C_f_clean.shape[0]}), skipping Ritz")
        else:
            basis_sym = C_f_clean.reshape(C_f_clean.shape[0], N3).T
            print(f"  [symbolic] Ritz  basis={basis_sym.shape} ...",
                  end=' ', flush=True)
            t0 = time.perf_counter()
            E_sym, _, rank_sym = svd_rayleigh_ritz_op(
                basis_sym, H_matvec, svd_tol=svd_tol,
                max_energies=n_print, hermitian=True)
            t_ritz_sym = time.perf_counter() - t0
            sym_result.update({"ritz_evals": E_sym.tolist(),
                               "rank": int(rank_sym), "t_ritz_s": t_ritz_sym})
            print(f"{t_ritz_sym:.2f}s  rank={rank_sym}  E[0]={E_sym[0]:.6f}")
            print(f"    Ritz: {np.round(E_sym[:n_print], 6).tolist()}")

    # ── FFT path ──────────────────────────────────────────────────────────────
    print(f"  [fft] Chebyshev recurrence ({n_random} waves) ...",
          end=' ', flush=True)
    t_fft0  = time.perf_counter()
    C_f_fft = np.empty((n_random, N3), dtype=float)
    for i in range(n_random):
        kx, ky, kz = k_vals[i]
        phase      = kx * X + ky * Y + kz * Z + b_vals[i]
        psi0       = np.sin(phase).ravel()
        C_f_fft[i] = chebyshev_recurrence(H_matvec, psi0, a, b_sc, cheb_m)
    t_filter_fft = time.perf_counter() - t_fft0
    print(f"{t_filter_fft:.2f}s")

    basis_fft = C_f_fft.T
    print(f"  [fft] Ritz  basis={basis_fft.shape} ...", end=' ', flush=True)
    t0 = time.perf_counter()
    E_fft, _, rank_fft = svd_rayleigh_ritz_op(
        basis_fft, H_matvec, svd_tol=svd_tol,
        max_energies=n_print, hermitian=True)
    t_ritz_fft = time.perf_counter() - t0
    fft_result = {"ritz_evals": E_fft.tolist(), "rank": int(rank_fft),
                  "t_filter_s": t_filter_fft, "t_ritz_s": t_ritz_fft}
    print(f"{t_ritz_fft:.2f}s  rank={rank_fft}  E[0]={E_fft[0]:.6f}")
    print(f"    Ritz: {np.round(E_fft[:n_print], 6).tolist()}")

    n_cmp = min(n_print, len(E_fft))
    err_fft_exact  = np.array([float(np.min(np.abs(exact_all - e)))
                                for e in E_fft[:n_cmp]])
    max_fft_exact  = float(np.max(err_fft_exact))
    mean_fft_exact = float(np.mean(err_fft_exact))

    max_sym_fft    = float('nan')
    max_sym_exact  = float('nan')
    mean_sym_exact = float('nan')
    if sym_result["ritz_evals"]:
        E_sym_arr = np.array(sym_result["ritz_evals"])
        n_cmp2    = min(n_cmp, len(E_sym_arr))
        diff_sf   = np.abs(E_sym_arr[:n_cmp2] - E_fft[:n_cmp2])
        max_sym_fft = float(np.max(diff_sf))
        err_sym_exact = np.array([float(np.min(np.abs(exact_all - e)))
                                   for e in E_sym_arr[:n_cmp2]])
        max_sym_exact  = float(np.max(err_sym_exact))
        mean_sym_exact = float(np.mean(err_sym_exact))
        print(f"  max|E_sym - E_fft|    = {max_sym_fft:.3e}")

    print(f"  max|E_fft - exact|    = {max_fft_exact:.3e}  "
          f"mean={mean_fft_exact:.3e}")
    if not np.isnan(max_sym_exact):
        print(f"  max|E_sym - exact|    = {max_sym_exact:.3e}  "
              f"mean={mean_sym_exact:.3e}")

    return {
        "N": N, "d": float(d), "k_max": float(k_max),
        "E_ref": E_ref.tolist(), "t_ref_s": float(t_ref),
        "symbolic": sym_result, "fft": fft_result,
        "max_sym_vs_fft":    max_sym_fft,
        "max_fft_vs_exact":  max_fft_exact,
        "mean_fft_vs_exact": mean_fft_exact,
        "max_sym_vs_exact":  max_sym_exact,
        "mean_sym_vs_exact": mean_sym_exact,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Symbolic vs FFT Chebyshev filter (self-contained restore)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('--N_sweep',       type=str,   default='10:14')
    ap.add_argument('--box_L',         type=float, default=5.0)
    ap.add_argument('--cheb_m',        type=int,   default=5)
    ap.add_argument('--E_lo',          type=float, default=-3.0)
    ap.add_argument('--E_hi',          type=float, default=20.0)
    ap.add_argument('--n_random',      type=int,   default=400)
    ap.add_argument('--n_print',       type=int,   default=20)
    ap.add_argument('--seed',          type=int,   default=42)
    ap.add_argument('--k_max',         type=float, default=0.0,
                    help='0 = auto (pi/d)')
    ap.add_argument('--svd_tol',       type=float, default=1e-4)
    ap.add_argument('--cache_dir',     type=str,   default='ho3d_symbolic_cache')
    ap.add_argument('--julia_exe',     type=str,   default='julia')
    ap.add_argument('--out_json',      type=str,
                    default='symbolic_fft_ho3d_restored.json')
    ap.add_argument('--force_rebuild', action='store_true',
                    help='Delete and rebuild the cached .jl Julia script. '
                         'Use this when you get NaN results.')
    args = ap.parse_args()

    N_list    = parse_N_sweep(args.N_sweep)
    cache_dir = Path(args.cache_dir)
    a    =  2.0 / (args.E_hi - args.E_lo)
    b_sc = -(args.E_hi + args.E_lo) / (args.E_hi - args.E_lo)

    print("=== Symbolic vs FFT Chebyshev filter -- 3D HO (restored) ===")
    print(f"H = -half*nabla^2 + half*(x^2+y^2+z^2)")
    print(f"box_L={args.box_L}  N_list={N_list}")
    print(f"cheb_m={args.cheb_m}  E_lo={args.E_lo}  E_hi={args.E_hi}")
    print(f"a={a:.8f}  b_sc={b_sc:.8f}")
    print(f"n_random={args.n_random}  seed={args.seed}  svd_tol={args.svd_tol}")
    print(f"force_rebuild={args.force_rebuild}")
    print()

    print("-- Step 1: H^n symbolic cache --")
    ensure_H_powers_cache(cache_dir, args.cheb_m)
    print()

    coeffs = chebyshev_coeffs_transformed(args.cheb_m, a=a, b=b_sc)
    print("-- Step 2: Chebyshev coefficients --")
    for n, c in enumerate(coeffs):
        print(f"    c_{n} = {float(c):.10g}")
    print()

    print("-- Step 3: Julia filter script --")
    jl_path = ensure_julia_script(
        cache_dir, args.cheb_m, args.E_lo, args.E_hi,
        force_rebuild=args.force_rebuild)
    print()

    exact_all = ho3d_exact_levels(omega=1.0, n_max_shell=20)

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

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'N':>4}  {'d':>7}  {'N3':>7}  "
          f"{'sym_vs_fft':>12}  {'fft_vs_exact':>13}  {'sym_vs_exact':>13}")
    for r in sweep:
        print(f"{r['N']:>4}  {r['d']:>7.4f}  {r['N']**3:>7d}  "
              f"{r['max_sym_vs_fft']:>12.3e}  "
              f"{r['max_fft_vs_exact']:>13.3e}  "
              f"{r['max_sym_vs_exact']:>13.3e}")
    print(f"\nTotal wall time: {wall_total:.1f}s")

    output = {
        "script":   Path(__file__).name,
        "datetime": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "params": {
            "N_list": N_list, "box_L": args.box_L,
            "cheb_m": args.cheb_m, "E_lo": args.E_lo, "E_hi": args.E_hi,
            "a": a, "b_sc": b_sc,
            "coeffs": [float(c) for c in coeffs],
            "n_random": args.n_random, "n_print": args.n_print,
            "seed": args.seed, "svd_tol": args.svd_tol,
            "k_max_arg": args.k_max,
            "cache_dir": str(cache_dir.resolve()),
            "julia_exe": args.julia_exe,
            "jl_script": str(jl_path.resolve()),
            "force_rebuild": args.force_rebuild,
        },
        "sweep":        sweep,
        "wall_total_s": wall_total,
    }
    Path(args.out_json).write_text(json.dumps(output, indent=2))
    print(f"JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
