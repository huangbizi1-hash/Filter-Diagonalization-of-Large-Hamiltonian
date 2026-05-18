#!/usr/bin/env python3
"""compare_fd_fft_explosion.py

Compare Chebyshev explosion filter accuracy and speed when the kinetic energy
operator is evaluated via FFT vs. various orders of finite difference (FD).

Potentials supported:
  gaussian : V(r) = -A * exp(-B * r²)
  harmonic : V(r) = ½ ω² r²

For each discretisation method (FFT + selected FD orders) the script:
  1. Applies the Chebyshev filter T_m(aH+b) to n_random random states
  2. Computes per-state Rayleigh quotients
  3. Runs Rayleigh-Ritz to extract approximate eigenvalues
  4. Records wall time for f(H) application

Reference eigenvalues come from scipy eigsh on the FFT Hamiltonian.

Finite-difference orders available: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
(central-difference stencils from ho3d_solvers_v2.FD_STENCILS)

Grid sweep:
  --N_sweep START:STOP[:STEP]  — run for each N in range(START, STOP, STEP)
                                  STOP is exclusive, default STEP=1
  --N_sweep N1,N2,N3,...       — run for each listed N
  All sweep results go into a single JSON under the "sweep" key.

Usage examples:
  # Single N
  python compare_fd_fft_explosion.py --potential harmonic --omega 1.0 \\
      --N 20 --box_L 5.0 --cheb_m 20 --E_lo 7.0 --E_hi 60.0 \\
      --n_random 400 --fd_order 2,4,6,8,10,12 --n_print 30

  # Use H_FFT for Ritz for all methods (original behaviour — "cross" comparison)
  python compare_fd_fft_explosion.py ... --ritz_h fft

  # Use each method's own H for Ritz (default — measures true discretisation error)
  python compare_fd_fft_explosion.py ... --ritz_h consistent

  # Sweep N=14..20
  python compare_fd_fft_explosion.py --potential harmonic --omega 1.0 \\
      --N_sweep 14:21 --box_L 5.0 --cheb_m 20 --E_lo 7.0 --E_hi 60.0 \\
      --n_random 400 --fd_order 2,4,6,8,10,12 --n_print 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d
from scipy.sparse.linalg import eigsh, LinearOperator

try:
    import primme
    HAS_PRIMME = True
except ImportError:
    HAS_PRIMME = False

sys.path.insert(0, str(Path(__file__).parent))
from ho3d_solvers_v2 import FD_STENCILS
from filter_core import svd_rayleigh_ritz_op

# ── grid helpers ───────────────────────────────────────────────────────────────

def make_grid_from_N(N: int, box_L: float):
    """Return (d, x1d) matching main.py's build_grid convention.

    d  = 2*box_L / N  (periodic spacing)
    x1d = linspace(-(N-1)*d/2, (N-1)*d/2, N)  — symmetric, both endpoints kept.

    The grid is symmetric about 0 so V(x_0) = V(x_{N-1}) for even-symmetric
    potentials (HO, Gaussian).  This avoids a large potential discontinuity at
    the FFT wrap-around boundary which would otherwise amplify high-k modes.
    """
    d = 2.0 * box_L / N
    L = (N - 1) * d / 2          # = (1 - 1/N) * box_L
    x1d = np.linspace(-L, L, N)
    return d, x1d


def make_grid_from_d(d: float, box_L: float):
    """Return (N, x1d), N = round(2*box_L/d), using symmetric linspace grid."""
    N = int(round(2.0 * box_L / d))
    L = (N - 1) * d / 2
    x1d = np.linspace(-L, L, N)
    return N, x1d


def make_T_k(N: int, d: float) -> np.ndarray:
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    return (k1d[:, None, None]**2 +
            k1d[None, :, None]**2 +
            k1d[None, None, :]**2) / 2.0


# ── FFT H application ──────────────────────────────────────────────────────────

def apply_H_fft(psi: np.ndarray, V: np.ndarray, T_k: np.ndarray) -> np.ndarray:
    sh = psi.shape
    p3 = psi.reshape(T_k.shape)
    return (np.fft.ifftn(T_k * np.fft.fftn(p3)).real + V * p3).reshape(sh)


def apply_chebyshev_fft(psi, V, T_k, m, E_lo, E_hi):
    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)

    def Hs(phi):
        return a * apply_H_fft(phi, V, T_k) + b * phi

    y0 = psi.copy()
    if m == 0:
        return y0
    y1 = Hs(y0)
    for _ in range(2, m + 1):
        y1, y0 = 2.0 * Hs(y1) - y0, y1
    return y1


# ── FD H application ───────────────────────────────────────────────────────────

def apply_H_fd(psi: np.ndarray, V: np.ndarray,
               stencil: np.ndarray, inv_d2: float) -> np.ndarray:
    """H|ψ⟩ via finite difference (periodic wrap BCs)."""
    Tpsi = (convolve1d(psi, stencil, axis=0, mode='wrap') +
            convolve1d(psi, stencil, axis=1, mode='wrap') +
            convolve1d(psi, stencil, axis=2, mode='wrap')) * inv_d2
    return Tpsi + V * psi


def apply_chebyshev_fd(psi, V, stencil, inv_d2, m, E_lo, E_hi):
    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)

    def Hs(phi):
        return a * apply_H_fd(phi, V, stencil, inv_d2) + b * phi

    y0 = psi.copy()
    if m == 0:
        return y0
    y1 = Hs(y0)
    for _ in range(2, m + 1):
        y1, y0 = 2.0 * Hs(y1) - y0, y1
    return y1


# ── normalisation / Rayleigh quotient ─────────────────────────────────────────

def normalize(psi: np.ndarray, d: float):
    norm2 = float(np.sum(psi**2) * d**3)
    if norm2 < 1e-30:
        return None, 0.0
    return psi / np.sqrt(norm2), norm2


def rayleigh_quotient(psi_n, V, T_k, d):
    return float(np.sum(psi_n * apply_H_fft(psi_n, V, T_k)) * d**3)


# ── explosion filter on a list of random states ────────────────────────────────

def run_explosion(psi_list, V, apply_H_matvec, apply_cheb,
                  T_k_for_rq, d, m, E_lo, E_hi, n_levels, svd_tol,
                  label: str, max_ritz: int = 0):
    """
    Run Chebyshev explosion on psi_list, return result dict.

    apply_cheb(psi, m, E_lo, E_hi) → filtered 3-D array
    apply_H_matvec(v_flat) → flat H|v⟩  (for Ritz)
    T_k_for_rq  used only for Rayleigh quotient (always FFT-accurate)
    """
    N3 = psi_list[0].size

    filtered    = []
    energies    = []
    state_times = []

    t_filter_start = time.perf_counter()
    for psi in psi_list:
        t0 = time.perf_counter()
        psi_f = apply_cheb(psi, m, E_lo, E_hi)
        psi_fn, _ = normalize(psi_f, d)
        t_state = time.perf_counter() - t0
        if psi_fn is None:
            continue
        E = rayleigh_quotient(psi_fn, V, T_k_for_rq, d)
        filtered.append(psi_fn)
        energies.append(float(E))
        state_times.append(t_state)
    t_filter = time.perf_counter() - t_filter_start

    n_kept      = len(filtered)
    t_per_state = t_filter / max(n_kept, 1)
    t_per_pt_us = t_per_state / N3 * 1e6

    E_ritz = np.array([])
    rank   = 0
    t_ritz = 0.0
    if filtered:
        basis = np.column_stack([p.ravel() for p in filtered])
        t0 = time.perf_counter()
        n_want = max(n_levels + 5, max_ritz)
        E_ritz, _, rank = svd_rayleigh_ritz_op(
            basis, apply_H_matvec,
            svd_tol=svd_tol, max_energies=n_want, hermitian=True)
        t_ritz = time.perf_counter() - t0

    return {
        'label'              : label,
        'n_kept'             : n_kept,
        'state_energies'     : energies,
        'ritz_evals'         : E_ritz.tolist(),
        'ritz_rank'          : int(rank),
        'filter_time_s'      : t_filter,
        'filter_per_state_s' : t_per_state,
        'filter_per_pt_us'   : t_per_pt_us,
        'ritz_time_s'        : t_ritz,
    }


# ── single-N computation ───────────────────────────────────────────────────────

def run_single_N(N: int, args, fd_orders: list, n_print, ritz_h: str = 'consistent') -> dict:
    """Run the full explosion comparison for one value of N.

    Returns a dict suitable for embedding in the sweep JSON.
    n_print: int or None (None → keep all Ritz evals).
    ritz_h:  'fft'        — always use H_FFT for Rayleigh-Ritz
             'fd'         — use H_FD for FD methods, H_FFT for FFT method
             'consistent' — same as 'fd' (each method uses its own H)
    max_ritz passed to run_explosion ensures at least n_print evals are computed.
    """
    max_ritz = n_print if n_print else 0
    d, x1d = make_grid_from_N(N, args.box_L)
    N3 = N**3
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    # ── potential ──────────────────────────────────────────────────────────────
    if args.potential == 'gaussian':
        V_num    = -args.A * np.exp(-args.B * (X**2 + Y**2 + Z**2))
        pot_desc = f'Gaussian V=-{args.A}*exp(-{args.B}*r²)'
    else:
        V_num    = 0.5 * args.omega**2 * (X**2 + Y**2 + Z**2)
        pot_desc = f'Harmonic V=½×{args.omega}²×r²'

    # ── kinetic operators ──────────────────────────────────────────────────────
    T_k_exact = make_T_k(N, d)
    V_max     = float(np.max(V_num))
    if args.kinetic_cut > 0:
        kinetic_cut = args.kinetic_cut
    else:
        kinetic_cut = max(args.E_hi - V_max, args.E_hi * 0.3)
    T_k_filt = np.minimum(T_k_exact, kinetic_cut)

    print(f'\n{"─"*68}')
    print(f'N={N}  N³={N3}  d={d:.5f} Bohr  eff_L={(N-1)*d/2:.4f}  '
          f'V_max={V_max:.2f} Ha  kinetic_cut={kinetic_cut:.2f} Ha')

    # ── reference: eigsh ──────────────────────────────────────────────────────
    print(f'  [ref] eigsh ({args.n_levels} levels) ...', end=' ', flush=True)
    t0 = time.perf_counter()
    H_linop = LinearOperator(
        (N3, N3),
        matvec=lambda v: apply_H_fft(v, V_num, T_k_exact),
        dtype=float,
    )
    E_ref, _ = eigsh(H_linop, k=args.n_levels, which='SA')
    t_ref = time.perf_counter() - t0
    E_ref = np.sort(E_ref)
    print(f'{t_ref:.2f}s  E_ref={np.round(E_ref, 5).tolist()}')

    # ── random initial states ─────────────────────────────────────────────────
    rng      = np.random.default_rng(args.seed)
    psi_list = [rng.standard_normal((N, N, N)).astype(np.float64)
                for _ in range(args.n_random)]
    psi_list = [p / np.sqrt(float(np.sum(p**2) * d**3)) for p in psi_list]

    # H_FFT matvec (always exact kinetic energy, used for reference eigsh and
    # optionally for Ritz when ritz_h='fft')
    def H_matvec_fft(v):
        return apply_H_fft(v, V_num, T_k_exact)

    method_results = []

    # ── FFT explosion ──────────────────────────────────────────────────────────
    def cheb_fft(psi, m, E_lo, E_hi):
        return apply_chebyshev_fft(psi, V_num, T_k_filt, m, E_lo, E_hi)

    res = run_explosion(
        psi_list, V_num, H_matvec_fft, cheb_fft,
        T_k_exact, d, args.cheb_m, args.E_lo, args.E_hi,
        args.n_levels, args.svd_tol, label='fft', max_ritz=max_ritz,
    )
    _print_result(res, n_print)
    method_results.append(res)

    # ── FD explosion ───────────────────────────────────────────────────────────
    for order in fd_orders:
        stencil = FD_STENCILS[order].astype(np.float64)
        inv_d2  = -0.5 / (d ** 2)
        label   = f'fd{order}'

        def cheb_fd(psi, m, E_lo, E_hi, _s=stencil, _id2=inv_d2):
            return apply_chebyshev_fd(psi, V_num, _s, _id2, m, E_lo, E_hi)

        # Choose which H to use for Rayleigh-Ritz
        if ritz_h == 'fft':
            H_ritz = H_matvec_fft
        else:   # 'fd' or 'consistent': use this method's own H
            def H_ritz(v, _s=stencil, _id2=inv_d2, _N=N):
                return apply_H_fd(v.reshape(_N, _N, _N), V_num, _s, _id2).ravel()

        res = run_explosion(
            psi_list, V_num, H_ritz, cheb_fd,
            T_k_exact, d, args.cheb_m, args.E_lo, args.E_hi,
            args.n_levels, args.svd_tol, label=label, max_ritz=max_ritz,
        )
        method_results.append(res)
        _print_result(res, n_print)

    # ── compact summary for this N ─────────────────────────────────────────────
    E_ref0 = E_ref[0]
    print(f'  {"Method":<10} {"Ritz[0]":>12} {"ΔE":>12} '
          f'{"filter(s)":>10} {"per_pt(µs)":>11}')
    for r in method_results:
        ritz0 = r['ritz_evals'][0] if r['ritz_evals'] else float('nan')
        print(f'  {r["label"]:<10} {ritz0:12.6f} {ritz0-E_ref0:+12.2e} '
              f'{r["filter_time_s"]:10.3f} {r["filter_per_pt_us"]:11.3f}')

    return {
        'N'          : N,
        'N3'         : N3,
        'd'          : d,
        'kinetic_cut': kinetic_cut,
        'V_max'      : V_max,
        't_ref_s'    : t_ref,
        'E_ref'      : E_ref.tolist(),
        'methods'    : [
            {**r, 'ritz_evals': r['ritz_evals'][:n_print]}
            for r in method_results
        ],
    }




def exact_ho3d_evals(n_levels: int, omega: float = 1.0) -> np.ndarray:
    """First n_levels eigenvalues of 3D isotropic HO: E = omega*(nx+ny+nz+3/2).

    Each principal quantum number N = nx+ny+nz has degeneracy (N+1)*(N+2)//2.
    Returns sorted eigenvalues with correct degeneracy.
    """
    evals = []
    for N in range(n_levels + 20):
        degen = (N + 1) * (N + 2) // 2
        evals.extend([omega * (N + 1.5)] * degen)
        if len(evals) >= n_levels:
            break
    return np.array(evals[:n_levels])


def benchmark_h_apply_time(H_matvec, N3: int, n_repeat: int = 500) -> float:
    """Average wall-time (seconds) per H application over n_repeat calls.

    Applies H to the same fixed unit vector each time to avoid overflow from
    repeated power-iteration amplification of large eigenvalues.
    """
    rng = np.random.default_rng(12345)
    v = rng.standard_normal(N3).astype(np.float64)
    v /= np.linalg.norm(v)
    H_matvec(v)  # warm-up (not timed)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        H_matvec(v)
    return (time.perf_counter() - t0) / max(n_repeat, 1)


def _get_num_matvecs(stats) -> int:
    """Extract numMatvecs from primme stats (dict or object with attributes)."""
    if isinstance(stats, dict):
        return int(stats.get('numMatvecs', -1))
    return int(getattr(stats, 'numMatvecs', -1))


def run_jdqmr_single_N(N: int, args, fd_orders: list) -> dict:
    if not HAS_PRIMME:
        raise RuntimeError('primme is required for --solver jdqmr. Please install primme.')

    d, x1d = make_grid_from_N(N, args.box_L)
    N3 = N**3
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    if args.potential == 'gaussian':
        V_num = -args.A * np.exp(-args.B * (X**2 + Y**2 + Z**2))
    else:
        V_num = 0.5 * args.omega**2 * (X**2 + Y**2 + Z**2)

    T_k_exact = make_T_k(N, d)

    # Reference exact eigenvalues (3D HO only; gaussian has no closed form here)
    if args.potential == 'harmonic':
        exact = exact_ho3d_evals(args.n_levels, omega=args.omega)
    else:
        exact = None

    def _make_entry(label, evals, t_avg, stats):
        evals = np.sort(np.asarray(evals).real)
        entry = {
            'label'        : label,
            'evals'        : evals.tolist(),
            'avg_h_apply_s': float(t_avg),
            'num_matvecs'  : _get_num_matvecs(stats),
        }
        if exact is not None:
            n = min(len(evals), len(exact))
            err = np.abs(evals[:n] - exact[:n])
            entry['exact_ho3d']    = exact[:n].tolist()
            entry['abs_err']       = err.tolist()
            entry['mean_abs_err']  = float(np.mean(err))
            entry['max_abs_err']   = float(np.max(err))
        return entry

    def _run_jdqmr(H_linop):
        return primme.eigsh(
            H_linop,
            k=args.n_levels, which='SA', method='PRIMME_JDQMR',
            tol=args.jdqmr_tol, maxMatvecs=args.max_matvecs,
            return_stats=True, return_history=False,
        )

    methods = []

    # FFT method
    def H_fft(v):
        return apply_H_fft(v, V_num, T_k_exact)

    t_avg = benchmark_h_apply_time(H_fft, N3, n_repeat=args.h_repeat)
    evals, _, stats = _run_jdqmr(LinearOperator((N3, N3), matvec=H_fft, dtype=float))
    methods.append(_make_entry('fft', evals, t_avg, stats))
    print(f'  [fft]  avg_H={t_avg*1e3:.3f}ms  matvecs={_get_num_matvecs(stats)}'
          + (f'  mean_err={methods[-1]["mean_abs_err"]:.2e}' if exact is not None else ''))

    # FD methods
    for order in fd_orders:
        stencil = FD_STENCILS[order].astype(np.float64)
        inv_d2  = -0.5 / (d ** 2)

        def H_fd(v, _s=stencil, _id2=inv_d2, _N=N):
            return apply_H_fd(v.reshape(_N, _N, _N), V_num, _s, _id2).ravel()

        t_avg = benchmark_h_apply_time(H_fd, N3, n_repeat=args.h_repeat)
        evals, _, stats = _run_jdqmr(LinearOperator((N3, N3), matvec=H_fd, dtype=float))
        methods.append(_make_entry(f'fd{order}', evals, t_avg, stats))
        print(f'  [fd{order}] avg_H={t_avg*1e3:.3f}ms  matvecs={_get_num_matvecs(stats)}'
              + (f'  mean_err={methods[-1]["mean_abs_err"]:.2e}' if exact is not None else ''))

    return {'N': N, 'N3': N3, 'd': d, 'solver': 'jdqmr',
            'potential': args.potential, 'methods': methods}

# ── CLI helpers ────────────────────────────────────────────────────────────────

def _parse_N_sweep(s: str) -> list:
    """Parse --N_sweep string into a list of N values.

    Formats accepted:
      "14:21"      → list(range(14, 21))     = [14,15,16,17,18,19,20]
      "14:21:2"    → list(range(14, 21, 2))  = [14,16,18,20]
      "14,16,20"   → [14, 16, 20]
    """
    s = s.strip()
    if ':' in s:
        parts = [int(x) for x in s.split(':')]
        if len(parts) == 2:
            return list(range(parts[0], parts[1]))
        if len(parts) == 3:
            return list(range(parts[0], parts[1], parts[2]))
        raise ValueError(f'Cannot parse --N_sweep {s!r}')
    return [int(x) for x in s.split(',') if x.strip()]


def _print_result(res: dict, n_print):
    """Print filter timing and Ritz eigenvalues.  n_print=None → all."""
    n    = res['n_kept']
    tf   = res['filter_time_s']
    tps  = res['filter_per_state_s']
    tpp  = res['filter_per_pt_us']
    ritz = res['ritz_evals'][:n_print]
    print(f'  [{res["label"]}] kept={n}  filter={tf:.3f}s  '
          f'per_state={tps:.3f}s  per_pt={tpp:.3f}µs')
    print(f'    Ritz ({len(ritz)} evals): {np.round(ritz, 6).tolist()}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Compare FFT vs FD Chebyshev explosion filter.')
    ap.add_argument('--potential', choices=['gaussian', 'harmonic'],
                    default='harmonic',
                    help='Potential type  [default harmonic]')
    ap.add_argument('--A',     type=float, default=10.0,
                    help='Gaussian depth (Ha)  [default 10.0]')
    ap.add_argument('--B',     type=float, default=0.5,
                    help='Gaussian exponent (Bohr⁻²)  [default 0.5]')
    ap.add_argument('--omega', type=float, default=1.0,
                    help='Harmonic oscillator frequency ω  [default 1.0]')

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--N',       type=int,  default=None,
                     help='Grid points per axis (single run)')
    grp.add_argument('--d',       type=float, default=None,
                     help='Grid spacing in Bohr (single run)')
    grp.add_argument('--N_sweep', type=str,  default=None,
                     help='Sweep over N values: "14:21" (range, exclusive stop) '
                          'or "14:21:2" (with step) or "14,16,18,20" (list). '
                          'Overrides --N and --d.')
    ap.add_argument('--box_L',   type=float, default=5.0,
                    help='Half-box length (Bohr)  [default 5.0]')

    ap.add_argument('--n_levels',  type=int,   default=5,
                    help='Number of reference eigsh levels  [default 5]')
    ap.add_argument('--n_random',  type=int,   default=4,
                    help='Number of random initial states  [default 4]')
    ap.add_argument('--E_lo',      type=float, default=1.0,
                    help='Chebyshev filter lower bound  [default 1.0]')
    ap.add_argument('--E_hi',      type=float, default=60.0,
                    help='Chebyshev filter upper bound  [default 60.0]')
    ap.add_argument('--cheb_m',    type=int,   default=20,
                    help='Chebyshev polynomial order  [default 20]')
    ap.add_argument('--seed',      type=int,   default=42,
                    help='RNG seed  [default 42]')
    ap.add_argument('--fd_order',  type=str,   default='2,4,6,8,10,12',
                    help='Comma-separated FD orders to test '
                         f'(available: {sorted(FD_STENCILS.keys())})  '
                         '[default 2,4,6,8,10,12]')
    ap.add_argument('--kinetic_cut', type=float, default=0.0,
                    help='Kinetic energy cut-off (Ha) for the explosion filter T_k. '
                         '0 = auto (E_hi - V_max).  Does NOT affect eigsh / Ritz.  '
                         '[default 0 = auto]')
    ap.add_argument('--n_print',   type=int,   default=0,
                    help='Ritz eigenvalues to print and store per method. '
                         '0 = all  [default 0]')
    ap.add_argument('--svd_tol',   type=float, default=1e-4,
                    help='SVD truncation threshold for Ritz  [default 1e-4]')
    ap.add_argument('--ritz_h',   choices=['fft', 'fd', 'consistent'],
                    default='consistent',
                    help='Which H to use for Rayleigh-Ritz extraction. '
                         '"fft" = always H_FFT (exact kinetic energy); '
                         '"fd"/"consistent" = each FD method uses its own H_FD '
                         '(measures true discretisation error).  [default consistent]')
    ap.add_argument('--solver', choices=['explosion','jdqmr'], default='explosion',
                    help='Eigen solver mode: explosion (Chebyshev filter) or jdqmr')
    ap.add_argument('--jdqmr_tol', type=float, default=1e-8,
                    help='JDQMR tolerance [default 1e-8]')
    ap.add_argument('--max_matvecs', type=int, default=50000,
                    help='JDQMR max matvecs [default 50000]')
    ap.add_argument('--h_repeat', type=int, default=500,
                    help='Average H-apply time over this many repeats [default 500]')
    ap.add_argument('--out_json',  type=str,   default='fd_fft_explosion.json',
                    help='Output JSON path  [default fd_fft_explosion.json]')
    args = ap.parse_args()

    t_wall_start = time.perf_counter()

    # ── parse FD orders ────────────────────────────────────────────────────────
    try:
        fd_orders = [int(x.strip()) for x in args.fd_order.split(',') if x.strip()]
    except ValueError:
        ap.error(f'--fd_order must be comma-separated integers, got: {args.fd_order!r}')
    bad = [o for o in fd_orders if o not in FD_STENCILS]
    if bad:
        ap.error(f'FD orders {bad} not available. Choose from {sorted(FD_STENCILS.keys())}')

    n_print = args.n_print if args.n_print > 0 else None

    # ── determine N list ───────────────────────────────────────────────────────
    if args.N_sweep is not None:
        try:
            N_list = _parse_N_sweep(args.N_sweep)
        except (ValueError, TypeError) as e:
            ap.error(f'--N_sweep parse error: {e}')
        if not N_list:
            ap.error('--N_sweep produced an empty list of N values')
    elif args.N is not None:
        N_list = [args.N]
    elif args.d is not None:
        N_tmp, _ = make_grid_from_d(args.d, args.box_L)
        N_list = [N_tmp]
    else:
        N_list = [20]   # fallback default

    is_sweep = len(N_list) > 1

    print(f'Potential:  {args.potential}  '
          + (f'omega={args.omega}' if args.potential == 'harmonic'
             else f'A={args.A}  B={args.B}'))
    print(f'box_L={args.box_L}  n_random={args.n_random}  '
          f'cheb_m={args.cheb_m}  E_lo={args.E_lo}  E_hi={args.E_hi}')
    print(f'FD orders:  {fd_orders}')
    if is_sweep and args.solver == 'explosion':
        print(f'N sweep:    {N_list}')

    # ── run ───────────────────────────────────────────────────────────────────
    sweep_results = []
    for N in N_list:
        if args.solver == 'jdqmr':
            result = run_jdqmr_single_N(N, args, fd_orders)
        else:
            result = run_single_N(N, args, fd_orders, n_print, ritz_h=args.ritz_h)
        sweep_results.append(result)

    t_wall = time.perf_counter() - t_wall_start

    # ── final summary (sweep mode) ─────────────────────────────────────────────
    if is_sweep and args.solver == 'explosion':
        method_labels = ['fft'] + [f'fd{o}' for o in fd_orders]
        print(f'\n{"="*78}')
        print(f'Sweep summary  (Ritz[0] vs exact E_ref[0]):')
        hdr = f'{"N":>4} {"d":>8}'
        for lbl in method_labels:
            hdr += f'  {lbl:>10}'
        print(hdr)
        print('-' * 78)
        for res in sweep_results:
            E0 = res['E_ref'][0]
            row = f'{res["N"]:>4} {res["d"]:>8.5f}'
            mmap = {m['label']: m for m in res['methods']}
            for lbl in method_labels:
                m = mmap.get(lbl)
                if m and m['ritz_evals']:
                    dE = m['ritz_evals'][0] - E0
                    row += f'  {dE:>+10.2e}'
                else:
                    row += f'  {"—":>10}'
            print(row)
        print(f'{"="*78}')
    print(f'Total wall time: {t_wall:.2f}s')

    # ── JSON output ───────────────────────────────────────────────────────────
    summary = {
        'params': {
            'potential'  : args.potential,
            'A'          : args.A,
            'B'          : args.B,
            'omega'      : args.omega,
            'box_L'      : args.box_L,
            'N_list'     : N_list,
            'n_random'   : args.n_random,
            'n_levels'   : args.n_levels,
            'E_lo'       : args.E_lo,
            'E_hi'       : args.E_hi,
            'cheb_m'     : args.cheb_m,
            'seed'       : args.seed,
            'fd_orders'  : fd_orders,
            'n_print'    : args.n_print,
            'ritz_h'     : args.ritz_h,
            'solver'     : args.solver,
            'jdqmr_tol'  : args.jdqmr_tol,
            'max_matvecs': args.max_matvecs,
            'h_repeat'   : args.h_repeat,
        },
        'sweep'         : sweep_results,
        'wall_total_s'  : t_wall,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f'Results → {args.out_json}')


if __name__ == '__main__':
    main()
