"""test_symbolic_ho3d.py
=====================
End-to-end test of the symbolic_code pipeline on the 3-D harmonic oscillator.

    H = -0.5 ∇² + 0.5(x² + y² + z²)

Exact eigenvalues:  E_n = n + 1.5   (n = nx + ny + nz = 0, 1, 2, ...)

On every run a timestamped sub-folder is created inside --outdir (default
``results/``).  It contains:

    filter_m<m>_E<lo>-<hi>.png          – filter plot (generated immediately)
    filter_m<m>_E<lo>-<hi>_recovered.png – filter + recovered eigenvalues overlay
    results.json                         – runtime, source info, energies

H-power strategies (--method)
------------------------------
H_powers (default)
    Pure H^n via sp.expand().  Rational coefficients → compact expressions.
    Files saved in --cache_dir (default ho3d_h_powers_cache/) and reused
    across runs.  a, b applied only at assembly time.
scaled
    (aH+b)^n with a, b baked in (legacy, no caching).

Filter modes (--filter_mode)
-----------------------------
explosion (default)
    Amplifies E < E_lo, suppresses E ∈ [E_lo, E_hi].
    Use when you want eigenvalues BELOW a threshold, e.g. --E_lo 5.
bandpass
    Narrow window; Test 2 checks eigenvalues inside [E_lo, E_hi].

E_hi
----
Omit --E_hi to auto-compute:  E_hi = max_kinetic + max_HO_potential on grid.
  max_kinetic   = 0.5 * 3 * (π/dx)²   (3D Nyquist corner)
  max_potential = 0.5 * 3 * (L-dx)²   (HO at farthest grid point)

Usage
-----
    python test_symbolic_ho3d.py                              # explosion, auto E_hi
    python test_symbolic_ho3d.py --quick                      # Test 1 only
    python test_symbolic_ho3d.py --m 8 --E_lo 5.0            # target E < 5
    python test_symbolic_ho3d.py --filter_mode bandpass --E_lo 0.5 --E_hi 6.5
    python test_symbolic_ho3d.py --no_cache                   # always recompute H^n
    python test_symbolic_ho3d.py --n_waves 120 --k_max 2.0   # more waves, wider k range
    python test_symbolic_ho3d.py --eval_backend julia         # use Julia JIT (julia must be on PATH)
    python test_symbolic_ho3d.py --eval_backend julia --julia_exe /path/to/julia
"""

import argparse
import datetime
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy as sp

from symbolic_code.h_powers import (
    apply_H_on_pair, generate_H_powers, generate_scaled_H_powers,
)
from symbolic_code.chebyshev_filter import (
    chebyshev_coeffs_transformed,
    apply_f_of_H_from_raw_powers, apply_f_of_H_on_psi,
    extract_cos_sin_coeffs, svd_H,
)
from symbolic_code.filter_plot import plot_chebyshev_filter


# ============================================================
# Lambdify helper
# ============================================================

def _safe_lambdify(sym_args, expr, modules='numpy', chunk_size=250):
    """Lambdify that handles large flat polynomials.

    sp.lambdify generates Python source code whose arithmetic tree mirrors the
    expression structure.  For a flat expanded polynomial with N terms the
    source is a single left-recursive sum of depth N-1.  Python's compiler
    hits its default recursion limit (~1000) once N exceeds roughly 1000.

    This helper checks the number of top-level summands.  If they fit in one
    chunk, it falls through to ordinary lambdify.  Otherwise it splits into
    chunks of at most *chunk_size* terms, lambdifies each chunk separately,
    and returns a wrapper that sums them at evaluation time.
    """
    terms = sp.Add.make_args(expr)          # (expr,) if not a sum
    if len(terms) <= chunk_size:
        return sp.lambdify(sym_args, expr, modules)

    n_chunks = (len(terms) + chunk_size - 1) // chunk_size
    print(f"    [safe_lambdify: {len(terms)} terms → {n_chunks} chunks of ≤{chunk_size}]")
    chunk_funcs = []
    for i in range(0, len(terms), chunk_size):
        chunk_expr = sp.Add(*terms[i : i + chunk_size])
        chunk_funcs.append(sp.lambdify(sym_args, chunk_expr, modules))

    def _f(*args):
        result = chunk_funcs[0](*args)
        for fn in chunk_funcs[1:]:
            result = result + fn(*args)
        return result

    return _f


# ============================================================
# Julia evaluation backend
# ============================================================

def _julia_eval_filter(expr_cos, expr_sin, X, Y, Z, k_vals, b_vals,
                       work_dir, julia_exe='julia', jl_cache_path=None):
    """Evaluate f(H)*psi on the 3-D grid using Julia JIT compilation.

    Optimisations vs plain subprocess approach:
    - Separate @inline poly/exp functions per group (JIT specialisation)
    - Precomputed exp terms (once per grid, reused across waves)
    - Vectorised trig via Julia's @. broadcast (SLEEF SIMD)
    - Pre-allocated working buffers (no per-wave heap allocation)
    - Warmup wave triggers JIT; remaining waves are timed separately

    If jl_cache_path is given and the file exists, the Julia source is
    loaded from cache (skipping sp.julia_code which can be slow for large
    polynomials).  Otherwise the script is generated and saved there.

    Returns
    -------
    C_f   : numpy array (n_waves, Ng, Ng, Ng)
    timing: dict with warmup_s, eval_s, n_eval_waves, total_wall_s
    """
    from symbolic_code.chebyshev_filter import group_by_exp_combined, apply_horner
    from symbolic_code.julia_codegen import build_julia_batch_script

    work_dir = Path(work_dir)
    Ng = X.shape[0]
    N  = Ng ** 3
    n_waves = len(k_vals)

    # ---- Julia source (cache-aware) ----
    jl_cache = Path(jl_cache_path) if jl_cache_path else None
    if jl_cache is not None and jl_cache.exists():
        jl_file = jl_cache
        print(f"  Julia script (cache hit) -> {jl_file}")
    else:
        print("  Building Julia source (group+Horner) ...")
        t_codegen = time.time()
        terms_cos = apply_horner(group_by_exp_combined(expr_cos))
        terms_sin = apply_horner(group_by_exp_combined(expr_sin))
        jl_src    = build_julia_batch_script(terms_cos, terms_sin)
        t_codegen = time.time() - t_codegen

        jl_file = jl_cache if jl_cache else (work_dir / 'eval_filter.jl')
        if jl_cache:
            jl_cache.parent.mkdir(parents=True, exist_ok=True)
        jl_file.write_text(jl_src, encoding='utf-8')
        tag = '(saved to cache)' if jl_cache else '(in run dir)'
        print(f"  Julia source built in {t_codegen:.1f}s, saved {tag} -> {jl_file}")

    # ---- binary I/O files ----
    grid_bin  = work_dir / '_grid.bin'
    kvals_bin = work_dir / '_kvals.bin'
    out_bin   = work_dir / '_cfilt.bin'

    with open(grid_bin, 'wb') as f:
        X.ravel().astype('<f8').tofile(f)
        Y.ravel().astype('<f8').tofile(f)
        Z.ravel().astype('<f8').tofile(f)

    kb = np.column_stack([k_vals, b_vals]).astype('<f8')
    kb.ravel().tofile(str(kvals_bin))

    # ---- run Julia ----
    cmd = [julia_exe, str(jl_file),
           str(grid_bin), str(kvals_bin), str(out_bin),
           str(N), str(n_waves)]
    print(f"  Running Julia (warmup+eval, n_waves={n_waves}) ...")
    t_wall0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    total_wall_s = time.time() - t_wall0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Julia exited with code {proc.returncode}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    # ---- parse JSON timing from last stdout line ----
    timing = {'warmup_s': None, 'eval_s': None, 'n_eval_waves': None,
              'total_wall_s': total_wall_s}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                t = json.loads(line)
                timing.update(t)
            except json.JSONDecodeError:
                pass
            break

    eval_s   = timing['eval_s']
    warmup_s = timing['warmup_s']
    n_eval   = timing['n_eval_waves']
    if eval_s is not None and n_eval:
        ms_per_wave = eval_s / n_eval * 1000
        print(f"  Julia  warmup  {warmup_s:.2f}s  |  "
              f"eval {eval_s:.2f}s / {n_eval} waves  "
              f"({ms_per_wave:.2f} ms/wave)  |  wall {total_wall_s:.1f}s")
    else:
        print(f"  Julia wall time: {total_wall_s:.1f}s")
        if proc.stdout.strip():
            print(f"  Julia stdout: {proc.stdout.strip()}")

    # ---- read results ----
    raw = np.fromfile(str(out_bin), dtype='<f8')
    C_f = raw.reshape(n_waves, Ng, Ng, Ng)

    grid_bin.unlink(missing_ok=True)
    kvals_bin.unlink(missing_ok=True)
    out_bin.unlink(missing_ok=True)

    return C_f, timing


# ============================================================
# Grid helpers
# ============================================================

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


def estimate_E_hi(L, N_grid, pref=0.5):
    """Upper bound on H eigenvalues representable on the grid.

    T_max = pref * 3 * (pi/dx)^2   – kinetic energy at 3-D Nyquist corner
    V_max = pref * 3 * (L-dx)^2    – HO potential at farthest grid point
    """
    dx    = 2.0 * L / N_grid
    k_max = np.pi / dx
    T_max = pref * 3.0 * k_max ** 2
    x_max = L - dx
    V_max = pref * 3.0 * x_max ** 2
    return T_max + V_max


# ============================================================
# Exact 3D HO spectrum
# ============================================================

def ho3d_spectrum(E_max):
    """Sorted list of exact 3D HO eigenvalues (with degeneracy) <= E_max.

    E_n = n + 1.5,  deg(n) = (n+1)(n+2)/2,  n = nx+ny+nz.
    """
    eigs = []
    for n in range(int(E_max) + 2):
        E = n + 1.5
        if E > E_max + 1e-10:
            break
        deg = (n + 1) * (n + 2) // 2
        eigs.extend([float(E)] * deg)
    return sorted(eigs)


# ============================================================
# H^n cache management (HO 3D specific)
# ============================================================

def _max_cached_power(cache_dir):
    """Return the highest n for which H_power_n.pkl exists, or -1."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return -1
    ns = []
    for f in cache_dir.glob('H_power_*.pkl'):
        try:
            ns.append(int(f.stem.split('_')[-1]))
        except (ValueError, IndexError):
            pass
    return max(ns) if ns else -1


def _extend_H_powers_cache(cache_dir, m_target, V, kvec, k2, pref, x, y, z):
    """Ensure H_power_0.pkl … H_power_{m_target}.pkl exist in cache_dir.

    If cache already has powers up to some n < m_target, extends from there
    so that previously computed work is not repeated.
    If cache is empty, computes from scratch via generate_H_powers.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_n = _max_cached_power(cache_dir)

    if max_n >= m_target:
        return  # already sufficient

    if max_n < 0:
        # Empty cache – compute everything from scratch
        generate_H_powers(
            m_target, cache_dir, file_format='pkl',
            V=V, kvec=kvec, k2=k2, pref=pref, x=x, y=y, z=z,
        )
        return

    # Extend H^(max_n+1) … H^m_target continuing from cached H^max_n
    print(f"  Extending cache: H^{max_n+1} … H^{m_target} ...")
    with open(cache_dir / f'H_power_{max_n}.pkl', 'rb') as fh:
        data = pickle.load(fh)
    Ps, Pc = data['Ps'], data['Pc']

    for n in range(max_n + 1, m_target + 1):
        print(f"    H^{n} ...", end=' ', flush=True)
        Ps_H, Pc_H = apply_H_on_pair(Ps, Pc, V, kvec, k2, pref, x, y, z)
        Ps = sp.expand(Ps_H)
        Pc = sp.expand(Pc_H)
        with open(cache_dir / f'H_power_{n}.pkl', 'wb') as fh:
            pickle.dump({'Ps': Ps, 'Pc': Pc}, fh)
        print('✓')

    # Guard: ensure n=0 entry exists (always trivial but required by loader)
    p0 = cache_dir / 'H_power_0.pkl'
    if not p0.exists():
        with open(p0, 'wb') as fh:
            pickle.dump({'Ps': sp.Integer(1), 'Pc': sp.Integer(0)}, fh)


# ============================================================
# Symbolic helpers
# ============================================================

def ho_potential_sympy():
    """Return HO potential as sympy expression + (x, y, z) symbols."""
    x, y, z = sp.symbols('x y z')
    V = sp.Rational(1, 2) * (x**2 + y**2 + z**2)
    return V, x, y, z


def build_fH_psi_sympy(m, E_lo, E_hi, method='H_powers', cache_dir=None):
    """Compute f(H)*psi symbolically.  Returns (psi_fH, source_info dict).

    For method='H_powers' with a cache_dir:
      - Checks cache for existing H_power_n.pkl files
      - Loads from cache if all 0..m are present; extends/computes otherwise
      - Saves newly computed powers to cache_dir for future reuse

    source_info is written verbatim into results.json.
    """
    V_sym, x, y, z = ho_potential_sympy()
    kx, ky, kz, b  = sp.symbols('kx ky kz b')
    kvec = (kx, ky, kz)
    k2   = kx**2 + ky**2 + kz**2
    a    =  2.0 / (E_hi - E_lo)
    b_sc = -(E_hi + E_lo) / (E_hi - E_lo)

    if method == 'H_powers' and cache_dir is not None:
        cache_dir  = Path(cache_dir)
        max_n      = _max_cached_power(cache_dir)
        from_cache = (max_n >= m)

        if not from_cache:
            print(f"  H^n cache: max_n={max(max_n, 0)}, need {m} → computing …")
            _extend_H_powers_cache(cache_dir, m, V_sym, kvec, k2, 0.5, x, y, z)
        else:
            print(f"  H^n cache hit: loading H^0..{m} from {cache_dir}")

        psi_fH = apply_f_of_H_from_raw_powers(cache_dir, m, a=a, b=b_sc)
        source_info = {
            'h_powers_source'     : str(cache_dir.resolve()),
            'h_powers_from_cache' : from_cache,
            'h_powers_method'     : method,
            'chebyshev_a'         : a,
            'chebyshev_b'         : b_sc,
        }

    else:
        # No cache: compute in a temp directory, delete after use
        tmpdir = tempfile.mkdtemp(prefix='ho_hpow_')
        source_info = {
            'h_powers_source'     : 'computed_fresh',
            'h_powers_from_cache' : False,
            'h_powers_method'     : method,
            'chebyshev_a'         : a,
            'chebyshev_b'         : b_sc,
        }
        try:
            if method == 'H_powers':
                generate_H_powers(
                    m, tmpdir, file_format='pkl',
                    V=V_sym, kvec=kvec, k2=k2, pref=0.5, x=x, y=y, z=z,
                )
                psi_fH = apply_f_of_H_from_raw_powers(tmpdir, m, a=a, b=b_sc)
            else:
                coeffs = chebyshev_coeffs_transformed(m, a=a, b=b_sc)
                generate_scaled_H_powers(
                    m, a, b_sc, outdir=tmpdir, file_format='pkl',
                    V=V_sym, kvec=kvec, k2=k2, pref=0.5, x=x, y=y, z=z,
                )
                psi_fH = apply_f_of_H_on_psi(tmpdir, coeffs, m, file_type='pkl')
        finally:
            shutil.rmtree(tmpdir)

    return psi_fH, source_info


# ============================================================
# Results-folder helpers
# ============================================================

def make_run_dir(outdir, m, E_lo, E_hi, method):
    """Create and return a timestamped run directory inside *outdir*."""
    ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag     = f'm{m}_E{E_lo:.2f}-{E_hi:.2f}_{method}'
    run_dir = Path(outdir) / f'run_{ts}_{tag}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_filter_plot(run_dir, m, E_lo, E_hi, filter_mode, recovered_eigs=None):
    """Plot |T_m(aE+b)| and save PNG.  Returns primary filename.

    Exact HO eigenvalues appropriate to the filter mode are shown as markers.
    If *recovered_eigs* is given, a second '_recovered' file is also written.
    """
    import matplotlib.pyplot as plt

    # Choose eigenvalue markers based on filter mode
    if filter_mode == 'explosion':
        markers = sorted(set(ho3d_spectrum(E_lo - 1e-10)))  # E < E_lo
    else:
        markers = sorted(set(
            e for e in ho3d_spectrum(E_hi) if E_lo <= e <= E_hi
        ))

    fname = f'filter_m{m}_E{E_lo:.2f}-{E_hi:.2f}.png'
    plot_chebyshev_filter(
        m_list=[m],
        E_lo=E_lo, E_hi=E_hi,
        mode=filter_mode,
        eigenvalues=markers or None,
        out_path=str(run_dir / fname),
    )
    plt.close('all')

    if recovered_eigs is not None:
        fname2 = f'filter_m{m}_E{E_lo:.2f}-{E_hi:.2f}_recovered.png'
        fig, ax = plot_chebyshev_filter(
            m_list=[m],
            E_lo=E_lo, E_hi=E_hi,
            mode=filter_mode,
            eigenvalues=markers or None,
        )
        rec  = np.asarray(recovered_eigs)
        ybot = ax.get_ylim()[0]
        ax.scatter(rec, np.full_like(rec, ybot),
                   marker='v', s=40, color='steelblue', zorder=5,
                   label='recovered')
        ax.legend(fontsize=9, ncol=2, loc='upper right')
        fig.savefig(str(run_dir / fname2), dpi=150, bbox_inches='tight')
        print(f'  Saved -> {run_dir / fname2}')
        plt.close('all')

    return fname


def write_json(run_dir, data):
    """Serialise *data* to run_dir/results.json (numpy-safe)."""
    def _cvt(obj):
        if isinstance(obj, np.ndarray):              return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        if isinstance(obj, Path):                    return str(obj)
        raise TypeError(f'Not JSON-serialisable: {type(obj)}')

    out = run_dir / 'results.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=_cvt)
    print(f'  Results JSON -> {out}')


# ============================================================
# Test 1 – single H step
# ============================================================

def test_single_H_step():
    """Verify symbolic H*psi matches FFT H*psi for grid-aligned k.

    Only grid-aligned k vectors (k ∈ 2π/L·ℤ) give an exact FFT kinetic
    energy.  Expected max error < 1e-8 (floating-point only).

    Returns (ok, result_dict).
    """
    print("\n" + "="*60)
    print("Test 1: single H step  (symbolic vs FFT, grid-aligned k)")
    print("="*60)

    t0 = time.time()
    L, N = 5.0, 24
    x1, X, Y, Z = make_grid(L=L, N=N)
    V3d = 0.5 * (X**2 + Y**2 + Z**2)

    k_grid  = grid_kvecs(x1)
    small_k = k_grid[k_grid > 0][:4]   # e.g. ~[0.628, 1.257, 1.885, 2.513]

    V_sym, xs, ys, zs = ho_potential_sympy()
    kx_s, ky_s, kz_s, b_s = sp.symbols('kx ky kz b')
    kvec = (kx_s, ky_s, kz_s)
    k2   = kx_s**2 + ky_s**2 + kz_s**2

    # Analytical result: H*sin(k·r+b) = (0.5k² + 0.5r²)*sin(k·r+b)
    Ps_new, Pc_new = apply_H_on_pair(
        sp.Integer(1), sp.Integer(0),
        V_sym, kvec, k2, 0.5, xs, ys, zs,
    )
    theta_sym = kx_s*xs + ky_s*ys + kz_s*zs + b_s
    Hpsi_expr = Ps_new * sp.sin(theta_sym) + Pc_new * sp.cos(theta_sym)
    f_Hpsi    = sp.lambdify(
        [xs, ys, zs, kx_s, ky_s, kz_s, b_s], Hpsi_expr, 'numpy'
    )

    rng = np.random.default_rng(0)
    b_vals    = rng.uniform(0, np.pi, 6)
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
        print(f"  k=({kx_v:+.3f},{ky_v:+.3f},{kz_v:+.3f})  max|sym-FFT| = {err:.2e}")

    max_err = max(errors)
    ok      = max_err < 1e-8
    runtime = time.time() - t0
    print(f"\n  Max error: {max_err:.2e}  → {'PASSED ✓' if ok else 'FAILED ✗'}")

    return ok, {
        'name'       : 'single_H_step',
        'status'     : 'PASSED' if ok else 'FAILED',
        'max_error'  : max_err,
        'all_errors' : [float(e) for e in errors],
        'runtime_s'  : runtime,
    }


# ============================================================
# Test 2 – filter diagonalisation
# ============================================================

def test_filter_diag(m=6, E_lo=4.5, E_hi=None, L=5.5, Ng=22,
                     n_waves=60, k_max=1.5, method='H_powers', cache_dir=None,
                     filter_mode='explosion', eval_backend='numpy', run_dir=None,
                     julia_exe='julia'):
    """Full Chebyshev pipeline: build, evaluate, diagonalise, compare.

    filter_mode='explosion': target eigenvalues < E_lo (amplified region).
    filter_mode='bandpass':  target eigenvalues in [E_lo, E_hi].

    eval_backend='numpy'  – use _safe_lambdify + NumPy (default).
    eval_backend='julia'  – generate a .jl script and call Julia via subprocess;
                            run_dir is used to store eval_filter.jl and I/O.

    Returns (ok, result_dict).
    """
    if E_hi is None:
        E_hi = estimate_E_hi(L, Ng)

    # Exact target eigenvalues depend on filter mode
    if filter_mode == 'explosion':
        exact = np.array(ho3d_spectrum(E_lo - 1e-10))   # E strictly < E_lo
    else:
        exact = np.array([
            e for e in ho3d_spectrum(E_hi) if E_lo <= e <= E_hi
        ])

    print("\n" + "="*60)
    print(f"Test 2: filter diagonalisation on 3D HO")
    print(f"        m={m}  E_lo={E_lo}  E_hi={E_hi:.4f}  "
          f"mode={filter_mode}  method={method}  k_max={k_max}")
    target_desc = f'E < {E_lo}' if filter_mode == 'explosion' else f'E ∈ [{E_lo}, {E_hi}]'
    print(f"        target: {len(exact)} eigenvalue(s) with {target_desc}")
    print("="*60)

    t_total = time.time()

    # ---- build symbolic f(H)*psi ----
    print("  Building symbolic f(H)*psi ...")
    t0 = time.time()
    psi_fH, source_info = build_fH_psi_sympy(
        m, E_lo, E_hi, method=method, cache_dir=cache_dir
    )
    build_time = time.time() - t0
    print(f"  Done in {build_time:.1f} s")

    # ---- extract cos/sin envelopes ----
    expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)

    # ---- numerical grid ----
    x1, X, Y, Z = make_grid(L=L, N=Ng)
    V3d = 0.5 * (X**2 + Y**2 + Z**2)

    # ---- random plane waves ----
    rng    = np.random.default_rng(42)
    k_vals = rng.uniform(-k_max, k_max, (n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, n_waves)

    # ---- evaluate f(H)*psi (backend-dependent) ----
    julia_timing = {}
    print(f"  Evaluating for {n_waves} plane waves  [backend={eval_backend}] ...")
    if eval_backend == 'julia':
        _jdir = Path(run_dir) if run_dir is not None else Path(tempfile.mkdtemp(prefix='ho3d_jl_'))
        # Julia source cache: skip sp.julia_code on re-runs with same m/a/b
        _jl_cache = None
        if cache_dir is not None:
            a_key = f"{source_info['chebyshev_a']:.8g}".replace('.','p').replace('-','m')
            b_key = f"{source_info['chebyshev_b']:.8g}".replace('.','p').replace('-','m')
            _jl_cache = Path(cache_dir) / f"julia_eval_m{m}_{a_key}_{b_key}.jl"
        C_f, julia_timing = _julia_eval_filter(
            expr_cos, expr_sin, X, Y, Z, k_vals, b_vals,
            work_dir=_jdir, julia_exe=julia_exe, jl_cache_path=_jl_cache)
    else:
        xs, ys, zs = sp.symbols('x y z')
        kx_s, ky_s, kz_s, b_s = sp.symbols('kx ky kz b')
        sym_args = [xs, ys, zs, kx_s, ky_s, kz_s, b_s]
        print("  Lambdifying cos/sin envelopes ...")
        f_cos = _safe_lambdify(sym_args, expr_cos)
        f_sin = _safe_lambdify(sym_args, expr_sin)
        C_f = np.zeros((n_waves, Ng, Ng, Ng), dtype=np.float64)
        for iw, (kv, bv) in enumerate(zip(k_vals, b_vals)):
            kx_v, ky_v, kz_v = kv
            phase = kx_v*X + ky_v*Y + kz_v*Z + bv
            fc = np.asarray(f_cos(X, Y, Z, kx_v, ky_v, kz_v, bv), dtype=float)
            fs = np.asarray(f_sin(X, Y, Z, kx_v, ky_v, kz_v, bv), dtype=float)
            C_f[iw] = fc * np.cos(phase) + fs * np.sin(phase)

    # ---- SVD filter diagonalisation ----
    print("  SVD filter diagonalisation ...")
    energies, _ = svd_H(
        C_f, Ng, Ng, Ng, x1, V3d, fft_apply_H,
        rank_threshold=1e-4, n_eigs=max(len(exact) + 5, 15),
    )

    # ---- compare recovered vs exact ----
    n_show    = min(len(energies), max(len(exact), 10))
    n_compare = min(len(energies), len(exact))
    recovered = np.sort(energies[:n_show])

    if n_compare > 0:
        errs   = np.abs(recovered[:n_compare] - exact[:n_compare])
        e0_err = abs(float(recovered[0]) - float(exact[0]))
    else:
        errs, e0_err = np.array([]), float('inf')

    ok_e0       = e0_err < 0.03
    n_good      = int(np.sum(errs < 0.1))
    ok_spectrum = n_good >= min(4, len(exact))
    ok          = ok_e0 and ok_spectrum
    runtime     = time.time() - t_total

    print(f"\n  Recovered eigenvalues : {np.round(recovered[:10], 4)}")
    print(f"  Exact eigenvalues     : {np.round(exact[:10], 4)}")
    print(f"\n  E_0 error = {e0_err:.4f}  (threshold 0.03)  "
          f"→ {'OK' if ok_e0 else 'FAIL'}")
    print(f"  Eigenvalues within 0.1 of exact: "
          f"{n_good}/{n_compare}  (need ≥{min(4, len(exact))})  "
          f"→ {'OK' if ok_spectrum else 'FAIL'}")
    print(f"  {'PASSED ✓' if ok else 'FAILED ✗'}")

    return ok, {
        'name'               : 'filter_diagonalisation',
        'status'             : 'PASSED' if ok else 'FAILED',
        'runtime_s'          : runtime,
        'build_time_s'       : build_time,
        'filter_mode'        : filter_mode,
        'eval_backend'       : eval_backend,
        'julia_timing'       : julia_timing if julia_timing else None,
        'E_hi'               : E_hi,
        'k_max'              : k_max,
        **source_info,
        'energies_recovered' : [float(e) for e in recovered],
        'energies_exact'     : [float(e) for e in exact],
        'e0_error'           : e0_err,
        'n_within_0.1'       : n_good,
        'n_compare'          : n_compare,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test symbolic_code pipeline on 3D harmonic oscillator.'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Only run Test 1 (single H step, ~seconds)')
    parser.add_argument('--m', type=int, default=6,
                        help='Chebyshev order m (default 6)')
    parser.add_argument('--E_lo', type=float, default=4.5,
                        help='Energy threshold: amplify E < E_lo (default 4.5)')
    parser.add_argument('--E_hi', type=float, default=None,
                        help='Upper energy bound; auto-computed from grid if omitted')
    parser.add_argument('--L', type=float, default=5.5,
                        help='Half-box size for grid (default 5.5)')
    parser.add_argument('--Ng', type=int, default=22,
                        help='Grid points per axis (default 22)')
    parser.add_argument('--n_waves', type=int, default=60,
                        help='Number of random plane waves (default 60)')
    parser.add_argument('--k_max', type=float, default=1.5,
                        help='Abs. range for random k vectors: uniform(-k_max, k_max) '
                             'per axis (default 1.5)')
    parser.add_argument('--method', default='H_powers',
                        choices=['H_powers', 'scaled'],
                        help='H_powers=pure H^n (default); scaled=(aH+b)^n')
    parser.add_argument('--filter_mode', default='explosion',
                        choices=['explosion', 'bandpass'],
                        help='explosion: find E<E_lo (default); '
                             'bandpass: find E in [E_lo,E_hi]')
    parser.add_argument('--cache_dir', default='ho3d_h_powers_cache',
                        help='Persistent cache for H^n pkl files '
                             '(default ho3d_h_powers_cache/)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable H^n cache (always recompute, no saving)')
    parser.add_argument('--outdir', default='results',
                        help='Root folder for timestamped run dirs (default results/)')
    parser.add_argument('--eval_backend', default='numpy',
                        choices=['numpy', 'julia'],
                        help='numpy: lambdify+NumPy (default); '
                             'julia: generate .jl and call Julia via subprocess')
    parser.add_argument('--julia_exe', default='julia',
                        help='Julia executable name or full path (default: julia)')
    args = parser.parse_args()

    # Always compute the reference E_hi from grid parameters and print it
    dx        = 2.0 * args.L / args.Ng
    T_max     = 0.5 * 3 * (np.pi / dx) ** 2
    V_max     = 0.5 * 3 * (args.L - dx) ** 2
    E_hi_ref  = T_max + V_max
    print(f"E_hi_ref = {E_hi_ref:.4f}  "
          f"(T_max={T_max:.2f} + V_max={V_max:.2f}, "
          f"grid {args.Ng}³, L={args.L})")

    E_hi = args.E_hi if args.E_hi is not None else E_hi_ref

    cache_dir = None if args.no_cache else args.cache_dir

    # ---- create run directory ----
    run_dir = make_run_dir(args.outdir, args.m, args.E_lo, E_hi, args.method)
    print(f"\nRun directory: {run_dir}\n")

    # ---- plot filter immediately (no H^n computation required) ----
    print("Plotting Chebyshev filter ...")
    filter_fname = save_filter_plot(
        run_dir, args.m, args.E_lo, E_hi, args.filter_mode
    )

    # ---- initialise JSON payload ----
    run_meta = {
        'timestamp'   : datetime.datetime.now().isoformat(),
        'run_dir'     : str(run_dir),
        'args'        : {
            'm'           : args.m,
            'E_lo'        : args.E_lo,
            'E_hi'        : E_hi,
            'E_hi_ref'    : E_hi_ref,
            'E_hi_auto'   : args.E_hi is None,
            'L'           : args.L,
            'Ng'          : args.Ng,
            'n_waves'     : args.n_waves,
            'k_max'       : args.k_max,
            'method'       : args.method,
            'filter_mode'  : args.filter_mode,
            'eval_backend' : args.eval_backend,
            'julia_exe'    : args.julia_exe,
            'cache_dir'    : str(cache_dir) if cache_dir else None,
            'quick'        : args.quick,
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
            m=args.m,
            E_lo=args.E_lo,
            E_hi=E_hi,
            L=args.L,
            Ng=args.Ng,
            n_waves=args.n_waves,
            k_max=args.k_max,
            method=args.method,
            cache_dir=cache_dir,
            filter_mode=args.filter_mode,
            eval_backend=args.eval_backend,
            run_dir=run_dir,
            julia_exe=args.julia_exe,
        )
        all_ok['Test2_filter_diag'] = ok2
        run_meta['test2'] = data2

        # Second plot: overlay recovered eigenvalues
        print("\nSaving filter plot with recovered eigenvalues ...")
        save_filter_plot(
            run_dir, args.m, args.E_lo, E_hi, args.filter_mode,
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
    if cache_dir:
        print(f"  H^n cache:        {Path(cache_dir).resolve()}")

    sys.exit(0 if all(all_ok.values()) else 1)


if __name__ == '__main__':
    main()
