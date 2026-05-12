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

Usage examples:
  python compare_fd_fft_explosion.py --potential harmonic --omega 1.0 \\
      --N 20 --cheb_m 20 --E_lo 7.0 --E_hi 60.0 --n_random 400 \\
      --fd_order 2,4,6,8,10

  python compare_fd_fft_explosion.py --potential gaussian --A 10 --B 0.5 \\
      --box_L 4.0 --d 0.5 --cheb_m 8 --E_lo -3.0 --E_hi 20.0 \\
      --fd_order 2,4,6,8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d
from scipy.sparse.linalg import eigsh, LinearOperator

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
                  label: str):
    """
    Run Chebyshev explosion on psi_list, return result dict.

    apply_cheb(psi, m, E_lo, E_hi) → filtered 3-D array
    apply_H_matvec(v_flat) → flat H|v⟩  (for Ritz)
    T_k_for_rq  used only for Rayleigh quotient (always FFT-accurate)
    """
    N3  = psi_list[0].size
    d3  = d**3

    filtered = []
    energies = []
    state_times = []

    t_filter_start = time.perf_counter()
    for psi in psi_list:
        t0 = time.perf_counter()
        psi_f = apply_cheb(psi, m, E_lo, E_hi)
        psi_fn, norm2 = normalize(psi_f, d)
        t_state = time.perf_counter() - t0
        if psi_fn is None:
            continue
        E = rayleigh_quotient(psi_fn, V, T_k_for_rq, d)
        filtered.append(psi_fn)
        energies.append(float(E))
        state_times.append(t_state)
    t_filter = time.perf_counter() - t_filter_start

    n_kept = len(filtered)
    t_per_state = t_filter / max(n_kept, 1)
    t_per_pt_us = t_per_state / N3 * 1e6

    # Rayleigh-Ritz
    E_ritz = np.array([])
    rank   = 0
    t_ritz = 0.0
    if filtered:
        basis = np.column_stack([p.ravel() for p in filtered])
        t0 = time.perf_counter()
        E_ritz, _, rank = svd_rayleigh_ritz_op(
            basis, apply_H_matvec,
            svd_tol=svd_tol, max_energies=n_levels + 5, hermitian=True)
        t_ritz = time.perf_counter() - t0

    return {
        'label'          : label,
        'n_kept'         : n_kept,
        'state_energies' : energies,
        'ritz_evals'     : E_ritz.tolist(),
        'ritz_rank'      : int(rank),
        'filter_time_s'  : t_filter,
        'filter_per_state_s' : t_per_state,
        'filter_per_pt_us'   : t_per_pt_us,
        'ritz_time_s'    : t_ritz,
    }


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

    # grid: specify N or d (not both); box_L always required with d
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--N',     type=int,   default=None,
                     help='Grid points per axis (overrides --d)')
    grp.add_argument('--d',     type=float, default=0.5,
                     help='Grid spacing (Bohr)  [default 0.5]')
    ap.add_argument('--box_L', type=float, default=4.0,
                    help='Half-box length (Bohr)  [default 4.0]')

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
                    help='Kinetic energy cut-off (Ha) applied to the explosion '
                         'filter T_k to prevent high-k spurious modes from being '
                         'explosively amplified.  0 = auto (E_hi - V_max).  '
                         'Does NOT affect eigsh reference or Ritz H.  [default 0]')
    ap.add_argument('--svd_tol',   type=float, default=1e-4,
                    help='SVD truncation threshold for Ritz  [default 1e-4]')
    ap.add_argument('--out_json',  type=str,   default='fd_fft_explosion.json',
                    help='Output JSON path  [default fd_fft_explosion.json]')
    args = ap.parse_args()

    t_wall_start = time.perf_counter()

    # ── parse FD orders ────────────────────────────────────────────────────────
    try:
        fd_orders = [int(x.strip()) for x in args.fd_order.split(',') if x.strip()]
    except ValueError:
        ap.error(f'--fd_order must be comma-separated integers, got: {args.fd_order}')
    bad = [o for o in fd_orders if o not in FD_STENCILS]
    if bad:
        ap.error(f'FD orders {bad} not available. Choose from {sorted(FD_STENCILS.keys())}')

    # ── grid ──────────────────────────────────────────────────────────────────
    if args.N is not None:
        N = args.N
        d, x1d = make_grid_from_N(N, args.box_L)
    else:
        N, x1d = make_grid_from_d(args.d, args.box_L)
        d = float(x1d[1] - x1d[0])
    N3 = N**3

    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    # ── potential ──────────────────────────────────────────────────────────────
    if args.potential == 'gaussian':
        V_num = -args.A * np.exp(-args.B * (X**2 + Y**2 + Z**2))
        pot_desc = f'Gaussian  V = -{args.A} * exp(-{args.B} * r²)'
    else:
        V_num = 0.5 * args.omega**2 * (X**2 + Y**2 + Z**2)
        pot_desc = f'Harmonic  V = ½ × {args.omega}² × r²'

    # T_k_exact: full (unclipped) kinetic operator — used for eigsh reference and Ritz.
    # T_k_filt:  clipped at kinetic_cut — used ONLY for the Chebyshev explosion filter.
    #
    # Why clip?  Without a cut, grid-corner modes have T_k ~ 59 Ha (for d=0.5).
    # Combined with V, their total energy >> E_hi, so T_m amplifies them by factors
    # 10^7–10^13 relative to the physical target states, completely swamping the
    # subspace.  Matching main.py's build_k_diagonal (default kinetic_cut=30 Ha).
    T_k_exact = make_T_k(N, d)
    V_max      = float(np.max(V_num))
    if args.kinetic_cut > 0:
        kinetic_cut = args.kinetic_cut
    else:
        # auto: ensure T_k + V_max ≤ E_hi so high-k modes stay inside the window
        kinetic_cut = max(args.E_hi - V_max, args.E_hi * 0.3)
    T_k_filt = np.minimum(T_k_exact, kinetic_cut)

    print(f'Grid:          N={N}  N³={N3}  d={d:.4f} Bohr  '
          f'eff_L={(N-1)*d/2:.4f} Bohr  (box_L={args.box_L})')
    print(f'Potential:     {pot_desc}  V_max={V_max:.2f} Ha')
    print(f'Filter:        T_{args.cheb_m}(aH+b)  '
          f'E_lo={args.E_lo}  E_hi={args.E_hi}')
    print(f'               a={2/(args.E_hi-args.E_lo):.4f}  '
          f'b={-(args.E_hi+args.E_lo)/(args.E_hi-args.E_lo):.4f}')
    print(f'kinetic_cut:   {kinetic_cut:.2f} Ha  '
          f'(T_k_max={float(np.max(T_k_exact)):.2f} Ha)')
    print(f'FD orders:     {fd_orders}  (max available: {max(FD_STENCILS.keys())})')

    # ── reference: eigsh with exact (unclipped) T_k ───────────────────────────
    print(f'\n[ref] eigsh ({args.n_levels} lowest levels) ...')
    t0 = time.perf_counter()
    H_linop = LinearOperator(
        (N3, N3),
        matvec=lambda v: apply_H_fft(v, V_num, T_k_exact),
        dtype=float,
    )
    E_ref, _ = eigsh(H_linop, k=args.n_levels, which='SA')
    t_ref = time.perf_counter() - t0
    E_ref = np.sort(E_ref)
    print(f'  Done in {t_ref:.2f}s')
    for i, e in enumerate(E_ref):
        print(f'  E_ref[{i}] = {e:.8f} Ha')

    # ── random initial states ─────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    psi_list = [rng.standard_normal((N, N, N)).astype(np.float64)
                for _ in range(args.n_random)]
    psi_list = [p / np.sqrt(float(np.sum(p**2) * d**3)) for p in psi_list]

    # Ritz H always uses exact (unclipped) T_k for unbiased eigenvalues
    def H_matvec(v):
        return apply_H_fft(v, V_num, T_k_exact)

    method_results = []

    # ── FFT explosion (uses T_k_filt for filter, T_k_exact for Ritz) ──────────
    print(f'\n[fft] Chebyshev explosion (m={args.cheb_m}, kinetic_cut={kinetic_cut:.1f}) ...')
    def cheb_fft(psi, m, E_lo, E_hi):
        return apply_chebyshev_fft(psi, V_num, T_k_filt, m, E_lo, E_hi)

    res_fft = run_explosion(
        psi_list, V_num, H_matvec, cheb_fft,
        T_k_exact, d, args.cheb_m, args.E_lo, args.E_hi,
        args.n_levels, args.svd_tol, label='fft',
    )
    _print_result(res_fft, args.n_levels)
    method_results.append(res_fft)

    # ── FD explosion for each order ────────────────────────────────────────────
    for order in fd_orders:
        stencil = FD_STENCILS[order].astype(np.float64)
        inv_d2  = -0.5 / (d ** 2)
        label   = f'fd{order}'

        print(f'\n[{label}] Chebyshev explosion (FD order {order}, m={args.cheb_m}) ...')

        def cheb_fd(psi, m, E_lo, E_hi, _s=stencil, _id2=inv_d2):
            return apply_chebyshev_fd(psi, V_num, _s, _id2, m, E_lo, E_hi)

        res = run_explosion(
            psi_list, V_num, H_matvec, cheb_fd,
            T_k_exact, d, args.cheb_m, args.E_lo, args.E_hi,
            args.n_levels, args.svd_tol, label=label,
        )
        method_results.append(res)
        _print_result(res, args.n_levels)

    t_wall = time.perf_counter() - t_wall_start

    # ── summary table ─────────────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'Reference eigenvalues (eigsh / FFT):')
    for i, e in enumerate(E_ref):
        print(f'  E_ref[{i}] = {e:.8f} Ha')
    print()
    hdr = f'{"Method":<10} {"Ritz[0]":>12} {"ΔE vs ref":>12} '
    hdr += f'{"f(H) total":>12} {"per state":>11} {"per pt µs":>10}'
    print(hdr)
    print('-' * 72)
    E_ref0 = E_ref[0]
    for r in method_results:
        ritz0  = r['ritz_evals'][0] if r['ritz_evals'] else float('nan')
        dE     = ritz0 - E_ref0
        print(f'{r["label"]:<10} {ritz0:12.8f} {dE:+12.2e} '
              f'{r["filter_time_s"]:12.3f}s '
              f'{r["filter_per_state_s"]:10.3f}s '
              f'{r["filter_per_pt_us"]:9.3f}µs')
    print(f'{"="*72}')
    print(f'Total wall time: {t_wall:.2f}s')

    # ── JSON output ───────────────────────────────────────────────────────────
    summary = {
        'params': {
            'potential'  : args.potential,
            'A'          : args.A,
            'B'          : args.B,
            'omega'      : args.omega,
            'N'          : N,
            'N3'         : N3,
            'd'          : d,
            'box_L'      : args.box_L,
            'n_random'   : args.n_random,
            'n_levels'   : args.n_levels,
            'E_lo'       : args.E_lo,
            'E_hi'       : args.E_hi,
            'cheb_m'     : args.cheb_m,
            'seed'       : args.seed,
            'fd_orders'  : fd_orders,
            'kinetic_cut': kinetic_cut,
            'V_max'      : V_max,
        },
        'timings': {
            'ref_eigsh_s' : t_ref,
            'wall_total_s': t_wall,
        },
        'E_ref'         : E_ref.tolist(),
        'methods'       : method_results,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f'Results → {args.out_json}')


def _print_result(res: dict, n_levels: int):
    n = res['n_kept']
    tf = res['filter_time_s']
    tps = res['filter_per_state_s']
    tpp = res['filter_per_pt_us']
    ritz = res['ritz_evals'][:n_levels]
    print(f'  kept={n}  filter={tf:.3f}s  per_state={tps:.3f}s  per_pt={tpp:.3f}µs')
    print(f'  Ritz: {np.round(ritz, 6).tolist()}')


if __name__ == '__main__':
    main()
