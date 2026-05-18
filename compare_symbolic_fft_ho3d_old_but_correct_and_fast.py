#!/usr/bin/env python3
"""compare_symbolic_fft_ho3d_old_but_correct_and_fast.py

Symbolic vs FFT Chebyshev filter comparison on the 3D harmonic oscillator.
Completely self-contained — imports ONLY from symbolic_code.* and filter_core.

Pipeline (baseline method)
--------------------------
Given H_power_n.pkl files (pre-computed, n=0..m):
  1. Load raw Ps_n, Pc_n from each pkl.
  2. Compute Chebyshev monomial coefficients c_n for T_m(aH+b).
  3. Assemble expr_cos = sum(c_n * Pc_n), expr_sin = sum(c_n * Ps_n).
  4. group_by_exp_combined + apply_horner  ->  one group (trivial exp=1).
  5. Write a clean Julia script that evaluates the Horner polynomial at each
     grid point and accumulates  out[i] = cos_p[i]*sum_cos + sin_p[i]*sum_sin.
  6. Cache the .jl file; re-use it for every N in the sweep.

Usage
-----
  python compare_symbolic_fft_ho3d_old_but_correct_and_fast.py \
      --N_sweep 7:15 --box_L 4.0 \
      --cheb_m 10 --E_lo 7.0 --E_hi 70.0 \
      --n_random 400 --n_print 20

  # Force-rebuild the .jl if you get NaN or Julia errors:
  python compare_symbolic_fft_ho3d_old_but_correct_and_fast.py ... --force_rebuild
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
    group_by_exp_combined,
    apply_horner,
)
from symbolic_code.h_powers import generate_H_powers
from filter_core import svd_rayleigh_ritz_op


# ── H^n pkl loader (numeric sort, skip *_horner.pkl) ──────────────────────────

def _load_H_powers(cache_dir: Path, n_max: int) -> dict:
    """Load H_power_0.pkl .. H_power_{n_max}.pkl in numeric order."""
    def _n(p):
        return int(p.name.replace('.pkl', '').split('_')[-1])
    def _is_plain(p):
        try: _n(p); return True
        except ValueError: return False
    result = {}
    for fpath in sorted(filter(_is_plain, cache_dir.glob('H_power_*.pkl')), key=_n):
        n = _n(fpath)
        with open(fpath, 'rb') as f:
            result[n] = pickle.load(f)
        if n >= n_max:
            break
    return result


# ── Assemble expr_cos, expr_sin = Σ c_n Pc_n / Ps_n ──────────────────────────

def _assemble_filter_envelopes(cache_dir: Path, m: int,
                               a: float, b: float) -> tuple:
    """Return (expr_cos, expr_sin) = (Σ c_n Pc_n, Σ c_n Ps_n).

    expr_cos multiplies cos(k·r+b); expr_sin multiplies sin(k·r+b).
    """
    coeffs  = chebyshev_coeffs_transformed(m, a=a, b=b)
    H_data  = _load_H_powers(cache_dir, m)

    Pc_total = sp.Integer(0)
    Ps_total = sp.Integer(0)
    for n, c in enumerate(coeffs):
        if c == 0:
            continue
        if n == 0:
            # H^0 · sin(θ) = sin(θ): only sin envelope
            Ps_total = Ps_total + c
        else:
            entry = H_data.get(n)
            if entry is None:
                raise KeyError(
                    f"H^{n} pkl not found in {cache_dir}. "
                    f"Run with --cheb_m >= {n} first.")
            Pc_total = Pc_total + c * entry['Pc']
            Ps_total = Ps_total + c * entry['Ps']

    return Pc_total, Ps_total


# ── Julia code generator ───────────────────────────────────────────────────────

def _sympy_to_julia(expr) -> str:
    """Convert sympy scalar expression to a Julia code string."""
    s = sp.julia_code(expr)
    # Strip broadcast-dot operators emitted by sympy for array context
    for op in ('.+', '.-', '.*', './', '.^'):
        s = s.replace(op, op[1:])
    return s


def _build_julia_script(poly_cos, poly_sin) -> str:
    """Generate a complete Julia batch script from Horner-reduced polynomials.

    Parameters
    ----------
    poly_cos : sympy expression (the cos envelope polynomial)
    poly_sin : sympy expression (the sin envelope polynomial)

    The script has this call convention::

        julia <script>.jl  grid.bin  kvals.bin  out.bin  N  n_waves

    where N = total number of 3-D grid points (= Nx^3).
    Stdout last line: JSON {"warmup_s": ..., "eval_s": ..., "n_eval_waves": ...}
    """
    jl_cos = _sympy_to_julia(poly_cos)
    jl_sin = _sympy_to_julia(poly_sin)

    return f"""\
# Auto-generated Julia filter script — DO NOT EDIT
# Evaluates f(H)·sin(k·r+b) via the Chebyshev monomial expansion.
#
# Usage:  julia <script>.jl  grid.bin  kvals.bin  out.bin  N  n_waves
# Stdout: JSON {{warmup_s, eval_s, n_eval_waves}}

function eval_filter_wave!(
        out  :: AbstractVector{{Float64}},
        X    :: Vector{{Float64}},
        Y    :: Vector{{Float64}},
        Z    :: Vector{{Float64}},
        kx   :: Float64,
        ky   :: Float64,
        kz   :: Float64,
        b    :: Float64)

    N = length(X)

    # Phase + trig (broadcasted, uses SLEEF SIMD)
    phase = @. kx * X + ky * Y + kz * Z + b
    cos_p = cos.(phase)
    sin_p = sin.(phase)

    @inbounds for i in 1:N
        x = X[i]; y = Y[i]; z = Z[i]

        # Horner-reduced polynomial envelopes
        p_cos = ({jl_cos})
        p_sin = ({jl_sin})

        out[i] = cos_p[i] * p_cos + sin_p[i] * p_sin
    end
end


function main()
    if length(ARGS) != 5
        error("Usage: julia script.jl grid.bin kvals.bin out.bin N n_waves")
    end

    grid_file  = ARGS[1]
    kvals_file = ARGS[2]
    out_file   = ARGS[3]
    N          = parse(Int, ARGS[4])
    n_waves    = parse(Int, ARGS[5])

    # Read grid (X_flat, Y_flat, Z_flat concatenated)
    grid_bytes = read(grid_file)
    grid_data  = reinterpret(Float64, grid_bytes)
    X = Vector{{Float64}}(grid_data[1:N])
    Y = Vector{{Float64}}(grid_data[N+1:2N])
    Z = Vector{{Float64}}(grid_data[2N+1:3N])

    # Read k-values: [kx0, ky0, kz0, b0, kx1, ky1, kz1, b1, ...]
    kb_bytes = read(kvals_file)
    kb       = reinterpret(Float64, kb_bytes)

    # Output buffer
    out_all = Vector{{Float64}}(undef, N * n_waves)

    # Warmup: evaluate wave 0 to trigger JIT compilation
    t_warmup = @elapsed eval_filter_wave!(
        view(out_all, 1:N), X, Y, Z,
        Float64(kb[1]), Float64(kb[2]), Float64(kb[3]), Float64(kb[4]))

    # Timed eval: waves 1 .. n_waves-1
    t_eval = @elapsed begin
        for iw in 1:n_waves-1
            ofs = iw * N + 1
            kid = iw * 4 + 1
            eval_filter_wave!(
                view(out_all, ofs:ofs+N-1), X, Y, Z,
                Float64(kb[kid]),   Float64(kb[kid+1]),
                Float64(kb[kid+2]), Float64(kb[kid+3]))
        end
    end

    # Write output
    open(out_file, "w") do io
        write(io, out_all)
    end

    n_eval = max(n_waves - 1, 1)
    println(\"{{\\\"warmup_s\\\": $t_warmup, \\\"eval_s\\\": $t_eval, \" *
            \"\\\"n_eval_waves\\\": $n_eval}}\")
end

main()
"""


# ── H^n cache & Julia script cache ────────────────────────────────────────────

def ensure_H_powers_cache(cache_dir: Path, m: int):
    """Generate H_power_0.pkl .. H_power_m.pkl if not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = set()
    for f in cache_dir.glob('H_power_*.pkl'):
        try:
            n = int(f.stem.split('_')[-1])
            existing.add(n)
        except ValueError:
            pass
    missing = sorted(set(range(m + 1)) - existing)
    if not missing:
        print(f"  H^n cache OK: H^0..H^{m} present in {cache_dir}")
        return
    print(f"  H^n cache: computing {missing} ...")
    x, y, z   = sp.symbols('x y z')
    kx, ky, kz = sp.symbols('kx ky kz')
    V_sym = sp.Rational(1, 2) * (x**2 + y**2 + z**2)
    generate_H_powers(
        m, cache_dir, file_format='pkl',
        V=V_sym, kvec=(kx, ky, kz), k2=kx**2 + ky**2 + kz**2,
        pref=0.5, x=x, y=y, z=z)


def ensure_julia_filter_script(cache_dir: Path, m: int,
                                E_lo: float, E_hi: float,
                                force_rebuild: bool = False) -> Path:
    """Build (or load from cache) the Julia filter script.

    The script is cached as  julia_filter_m{m}_a{a_key}_b{b_key}.jl
    in cache_dir.  Pass force_rebuild=True to delete and rebuild.
    """
    a    =  2.0 / (E_hi - E_lo)
    b_sc = -(E_hi + E_lo) / (E_hi - E_lo)

    def _key(v):
        return f'{v:.8g}'.replace('.', 'p').replace('-', 'm').replace('+', '')

    jl_path = cache_dir / f'julia_filter_m{m}_a{_key(a)}_b{_key(b_sc)}.jl'

    if force_rebuild and jl_path.exists():
        print(f"  --force_rebuild: deleting {jl_path.name}")
        jl_path.unlink()

    if jl_path.exists():
        print(f"  Julia script cache hit: {jl_path.name}")
        return jl_path

    print(f"  Building Julia filter script"
          f"  (m={m}, E_lo={E_lo}, E_hi={E_hi}) ...")

    # 1. Assemble cos/sin envelope polynomials
    t0 = time.perf_counter()
    print("    1/3  assembling Σ c_n H^n envelopes ...", end=' ', flush=True)
    expr_cos, expr_sin = _assemble_filter_envelopes(cache_dir, m, a, b_sc)
    print(f"{time.perf_counter()-t0:.1f}s")

    # 2. Group by Gaussian factor + Horner reduction
    t0 = time.perf_counter()
    print("    2/3  group_by_exp + apply_horner ...", end=' ', flush=True)
    terms_cos = apply_horner(group_by_exp_combined(expr_cos))
    terms_sin = apply_horner(group_by_exp_combined(expr_sin))
    print(f"{time.perf_counter()-t0:.1f}s")
    print(f"         cos groups={len(terms_cos)}  sin groups={len(terms_sin)}")

    # For the plane-wave + HO case there should be exactly 1 group each
    # with trivial exp_part = 1.  If there are multiple groups, sum them.
    def _sum_groups(terms) -> sp.Expr:
        total = sp.Integer(0)
        for ep, pp in terms:
            total = total + ep * pp
        return total

    poly_cos = _sum_groups(terms_cos) if terms_cos else sp.Integer(0)
    poly_sin = _sum_groups(terms_sin) if terms_sin else sp.Integer(0)

    # 3. Generate and cache the Julia script
    t0 = time.perf_counter()
    print("    3/3  generating Julia source ...", end=' ', flush=True)
    jl_src = _build_julia_script(poly_cos, poly_sin)
    jl_path.write_text(jl_src, encoding='utf-8')
    print(f"{time.perf_counter()-t0:.1f}s  ->  {jl_path.name}")

    return jl_path


# ── Julia batch evaluation ────────────────────────────────────────────────────

def julia_eval_filter(jl_path: Path, x1d: np.ndarray,
                      k_vals: np.ndarray, b_vals: np.ndarray,
                      julia_exe: str) -> tuple:
    """Run Julia to evaluate f(H)·sin(k·r+b) for all plane waves.

    Returns
    -------
    C_f    : ndarray  (n_waves, N, N, N)
    timing : dict
    """
    N_grid  = len(x1d)
    n_waves = len(k_vals)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    with tempfile.TemporaryDirectory(prefix='sym_filter_') as tmp:
        tmp       = Path(tmp)
        grid_bin  = tmp / 'grid.bin'
        kvals_bin = tmp / 'kvals.bin'
        out_bin   = tmp / 'out.bin'

        # Write grid as flat float64 little-endian: [X_flat, Y_flat, Z_flat]
        np.concatenate([X.ravel(), Y.ravel(), Z.ravel()]).astype('<f8').tofile(
            str(grid_bin))

        # Write k-values: [kx0,ky0,kz0,b0, kx1,...]
        kb = np.column_stack([k_vals, b_vals.reshape(-1, 1)]).astype('<f8')
        kb.ravel().tofile(str(kvals_bin))

        cmd = [julia_exe, str(jl_path),
               str(grid_bin), str(kvals_bin), str(out_bin),
               str(N_grid**3), str(n_waves)]

        t0   = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"Julia exited {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}")

        timing = {'warmup_s': None, 'eval_s': None,
                  'n_eval_waves': None, 'total_wall_s': wall}
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    timing.update(json.loads(line))
                except json.JSONDecodeError:
                    pass
                break

        raw = np.fromfile(str(out_bin), dtype='<f8')

    return raw.reshape(n_waves, N_grid, N_grid, N_grid), timing


# ── grid helpers ──────────────────────────────────────────────────────────────

def make_grid(N: int, box_L: float):
    d   = 2.0 * box_L / N
    L   = (N - 1) * d / 2.0
    return d, np.linspace(-L, L, N)


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


def ho3d_exact_levels(n_max_shell: int = 20) -> np.ndarray:
    evals = [nx + ny + nz + 1.5
             for nx in range(n_max_shell + 1)
             for ny in range(n_max_shell + 1)
             for nz in range(n_max_shell + 1)]
    return np.sort(evals)


def parse_N_sweep(s: str) -> list:
    s = s.strip()
    if ':' in s:
        lo, hi = (int(x) for x in s.split(':'))
        return list(range(lo, hi))
    return [int(x) for x in s.split(',')]


# ── Chebyshev FFT recurrence ──────────────────────────────────────────────────

def chebyshev_recurrence(H_apply, psi0, a, b_sc, m):
    if m == 0:
        return psi0.copy()
    Hs = lambda v: a * H_apply(v) + b_sc * v
    y_prev, y_curr = psi0.copy(), Hs(psi0)
    for _ in range(2, m + 1):
        y_next      = 2.0 * Hs(y_curr) - y_prev
        y_prev, y_curr = y_curr, y_next
    return y_curr


# ── per-N comparison ──────────────────────────────────────────────────────────

def run_one_N(N, box_L, cheb_m, a, b_sc,
              n_random, k_max_arg, seed, svd_tol, n_print,
              jl_path, julia_exe, exact_all) -> dict:

    d, x1d = make_grid(N, box_L)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    V3   = 0.5 * (X**2 + Y**2 + Z**2)
    T_k  = make_T_k(N, d)
    N3   = N**3
    k_max = k_max_arg if k_max_arg > 0 else np.pi / d

    print(f"\n-- N={N}  d={d:.4f}  N3={N3}  k_max={k_max:.3f} --")

    def H_mv(v):
        return apply_H_fft(v, V3, T_k)
    H_op = LinearOperator((N3, N3), matvec=H_mv, dtype=float)

    # Reference eigenvalues
    nref = min(10, N3 - 2)
    t0   = time.perf_counter()
    Eref, _ = eigsh(H_op, k=nref, which='SA')
    Eref    = np.sort(Eref.real)
    print(f"  [ref]  eigsh  {time.perf_counter()-t0:.2f}s  "
          f"E[:5]={np.round(Eref[:5],5).tolist()}")

    rng    = np.random.default_rng(seed)
    k_vals = rng.uniform(-k_max, k_max, (n_random, 3))
    b_vals = rng.uniform(0.0, 2*np.pi, n_random)

    # ── symbolic path ─────────────────────────────────────────────────────────
    sym_result = {"ok": False, "julia_timing": {}, "t_julia_s": 0.0,
                  "ritz_evals": [], "rank": 0, "t_ritz_s": 0.0}
    print(f"  [sym]  Julia  ({n_random} waves) ...", flush=True)
    t0 = time.perf_counter()
    try:
        C_sym, jtiming = julia_eval_filter(jl_path, x1d, k_vals, b_vals, julia_exe)
        t_julia = time.perf_counter() - t0
        es = jtiming.get('eval_s') or 0.0
        nw = jtiming.get('n_eval_waves') or 1
        print(f"    warmup={jtiming.get('warmup_s','?'):.4f}s  "
              f"eval={es:.4f}s/{nw} waves  "
              f"({es/nw*1e3:.3f} ms/wave)  wall={t_julia:.1f}s")

        nan_count = int(np.isnan(C_sym).any(axis=(1,2,3)).sum())
        if nan_count:
            print(f"    WARNING: {nan_count}/{n_random} waves have NaN  "
                  f"-- rerun with --force_rebuild")
        good = ~np.isnan(C_sym).any(axis=(1, 2, 3))
        C_clean = C_sym[good]

        sym_result['ok']           = True
        sym_result['julia_timing'] = jtiming
        sym_result['t_julia_s']    = t_julia

        if C_clean.shape[0] >= 5:
            basis = C_clean.reshape(C_clean.shape[0], N3).T
            print(f"  [sym]  Ritz  basis={basis.shape} ...", end=' ', flush=True)
            t0 = time.perf_counter()
            E_sym, _, rank = svd_rayleigh_ritz_op(
                basis, H_mv, svd_tol=svd_tol, max_energies=n_print, hermitian=True)
            t_ritz = time.perf_counter() - t0
            sym_result.update({"ritz_evals": E_sym.tolist(),
                               "rank": int(rank), "t_ritz_s": t_ritz})
            print(f"{t_ritz:.2f}s  rank={rank}  E[0]={E_sym[0]:.6f}")
            print(f"    Ritz: {np.round(E_sym[:n_print],6).tolist()}")
        else:
            print(f"    too few clean waves ({C_clean.shape[0]}), skipping Ritz")

    except Exception as exc:
        print(f"    ERROR: {exc}")
        sym_result['t_julia_s'] = time.perf_counter() - t0

    # ── FFT recurrence ────────────────────────────────────────────────────────
    print(f"  [fft]  Chebyshev  ({n_random} waves) ...", end=' ', flush=True)
    t0 = time.perf_counter()
    C_fft = np.empty((n_random, N3))
    for i in range(n_random):
        kx, ky, kz = k_vals[i]
        psi0 = np.sin(kx*X + ky*Y + kz*Z + b_vals[i]).ravel()
        C_fft[i] = chebyshev_recurrence(H_mv, psi0, a, b_sc, cheb_m)
    t_fft = time.perf_counter() - t0
    print(f"{t_fft:.2f}s")

    basis_fft = C_fft.T
    print(f"  [fft]  Ritz  basis={basis_fft.shape} ...", end=' ', flush=True)
    t0 = time.perf_counter()
    E_fft, _, rank_fft = svd_rayleigh_ritz_op(
        basis_fft, H_mv, svd_tol=svd_tol, max_energies=n_print, hermitian=True)
    t_rfft = time.perf_counter() - t0
    fft_result = {"ritz_evals": E_fft.tolist(), "rank": int(rank_fft),
                  "t_filter_s": t_fft, "t_ritz_s": t_rfft}
    print(f"{t_rfft:.2f}s  rank={rank_fft}  E[0]={E_fft[0]:.6f}")
    print(f"    Ritz: {np.round(E_fft[:n_print],6).tolist()}")

    # ── error metrics ─────────────────────────────────────────────────────────
    nc = min(n_print, len(E_fft))
    err_fe = np.array([np.min(np.abs(exact_all - e)) for e in E_fft[:nc]])
    max_fe = float(err_fe.max());  mean_fe = float(err_fe.mean())

    max_sf = float('nan');  max_se = float('nan');  mean_se = float('nan')
    if sym_result["ritz_evals"]:
        Es  = np.array(sym_result["ritz_evals"])
        nc2 = min(nc, len(Es))
        max_sf = float(np.abs(Es[:nc2] - E_fft[:nc2]).max())
        err_se = np.array([np.min(np.abs(exact_all - e)) for e in Es[:nc2]])
        max_se = float(err_se.max());  mean_se = float(err_se.mean())
        print(f"  max|E_sym-E_fft| = {max_sf:.3e}")

    print(f"  max|E_fft-exact| = {max_fe:.3e}  mean={mean_fe:.3e}")
    if not np.isnan(max_se):
        print(f"  max|E_sym-exact| = {max_se:.3e}  mean={mean_se:.3e}")

    return {"N": N, "d": float(d), "k_max": float(k_max),
            "E_ref": Eref.tolist(), "t_ref_s": float(time.perf_counter()-t0),
            "symbolic": sym_result, "fft": fft_result,
            "max_sym_vs_fft": max_sf, "max_fft_vs_exact": max_fe,
            "mean_fft_vs_exact": mean_fe, "max_sym_vs_exact": max_se,
            "mean_sym_vs_exact": mean_se}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--N_sweep',       default='10:14')
    ap.add_argument('--box_L',         type=float, default=5.0)
    ap.add_argument('--cheb_m',        type=int,   default=5)
    ap.add_argument('--E_lo',          type=float, default=-3.0)
    ap.add_argument('--E_hi',          type=float, default=20.0)
    ap.add_argument('--n_random',      type=int,   default=400)
    ap.add_argument('--n_print',       type=int,   default=20)
    ap.add_argument('--seed',          type=int,   default=42)
    ap.add_argument('--k_max',         type=float, default=0.0)
    ap.add_argument('--svd_tol',       type=float, default=1e-4)
    ap.add_argument('--cache_dir',     default='ho3d_symbolic_cache')
    ap.add_argument('--julia_exe',     default='julia')
    ap.add_argument('--out_json',      default='symbolic_fft_ho3d_restored.json')
    ap.add_argument('--force_rebuild', action='store_true',
                    help='Delete and rebuild the cached .jl script')
    args = ap.parse_args()

    N_list    = parse_N_sweep(args.N_sweep)
    cache_dir = Path(args.cache_dir)
    a    =  2.0 / (args.E_hi - args.E_lo)
    b_sc = -(args.E_hi + args.E_lo) / (args.E_hi - args.E_lo)

    print("=== Symbolic vs FFT Chebyshev filter  3D HO ===")
    print(f"  cheb_m={args.cheb_m}  E_lo={args.E_lo}  E_hi={args.E_hi}")
    print(f"  a={a:.8f}  b_sc={b_sc:.8f}")
    print(f"  box_L={args.box_L}  N_sweep={N_list}  n_random={args.n_random}")
    print(f"  force_rebuild={args.force_rebuild}")
    print()

    # Step 1: H^n pkl cache
    print("-- Step 1: H^n pkl cache --")
    ensure_H_powers_cache(cache_dir, args.cheb_m)
    print()

    # Step 2: Chebyshev coefficients
    coeffs = chebyshev_coeffs_transformed(args.cheb_m, a=a, b=b_sc)
    print("-- Step 2: Chebyshev coefficients --")
    for n, c in enumerate(coeffs):
        print(f"  c_{n:2d} = {float(c):+.10g}")
    print()

    # Step 3: Julia script
    print("-- Step 3: Julia filter script --")
    jl_path = ensure_julia_filter_script(
        cache_dir, args.cheb_m, args.E_lo, args.E_hi,
        force_rebuild=args.force_rebuild)
    print()

    exact_all = ho3d_exact_levels()

    # Step 4: sweep
    t0    = time.perf_counter()
    sweep = []
    for N in N_list:
        sweep.append(run_one_N(
            N=N, box_L=args.box_L,
            cheb_m=args.cheb_m, a=a, b_sc=b_sc,
            n_random=args.n_random, k_max_arg=args.k_max,
            seed=args.seed, svd_tol=args.svd_tol, n_print=args.n_print,
            jl_path=jl_path, julia_exe=args.julia_exe,
            exact_all=exact_all))
    wall = time.perf_counter() - t0

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'N':>4}  {'d':>7}  {'N3':>7}  "
          f"{'sym_vs_fft':>12}  {'fft_vs_exact':>13}  {'sym_vs_exact':>13}")
    for r in sweep:
        print(f"{r['N']:>4}  {r['d']:>7.4f}  {r['N']**3:>7d}  "
              f"{r['max_sym_vs_fft']:>12.3e}  "
              f"{r['max_fft_vs_exact']:>13.3e}  "
              f"{r['max_sym_vs_exact']:>13.3e}")
    print(f"\nTotal wall time: {wall:.1f}s")

    out = {"script": Path(__file__).name,
           "datetime": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "params": {"N_list": N_list, "box_L": args.box_L,
                      "cheb_m": args.cheb_m, "E_lo": args.E_lo, "E_hi": args.E_hi,
                      "a": a, "b_sc": b_sc, "coeffs": [float(c) for c in coeffs],
                      "n_random": args.n_random, "n_print": args.n_print,
                      "seed": args.seed, "svd_tol": args.svd_tol,
                      "cache_dir": str(cache_dir.resolve()),
                      "julia_exe": args.julia_exe,
                      "jl_script": str(jl_path.resolve()),
                      "force_rebuild": args.force_rebuild},
           "sweep": sweep, "wall_total_s": wall}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
