#!/usr/bin/env python3
"""compare_fd_fft_jdqmr_ho3d.py

Directly compare FFT vs finite-difference discretisations of the 3-D
harmonic oscillator by solving for the lowest eigenvalues with JDQMR
(PRIMME), bypassing the Chebyshev filter entirely.

Each discrete Hamiltonian H_disc = T_disc + V is solved independently.
Results are compared against the exact 3-D HO levels
  E(nx,ny,nz) = omega * (nx + ny + nz + 3/2)

Grid convention (matches compare_fd_fft_explosion.py):
  d   = 2 * box_L / N
  x1d = linspace(-(N-1)*d/2, (N-1)*d/2, N)   # symmetric

Usage:
  python compare_fd_fft_jdqmr_ho3d.py --N_sweep 10:25 --omega 1.0 \\
      --box_L 5.0 --n_levels 10 --fd_order 2,4,6,8,10,12

  python compare_fd_fft_jdqmr_ho3d.py --N_sweep 10:25 --omega 1.0 \\
      --box_L 5.0 --n_levels 10 --fd_order 2,4,6,8,10,12 \\
      --out_json jdqmr_ho3d_sweep.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from scipy.sparse.linalg import LinearOperator

try:
    import primme
    HAS_PRIMME = True
except ImportError:
    HAS_PRIMME = False

sys.path.insert(0, str(Path(__file__).parent))
from ho3d_solvers_v2 import FD_STENCILS


# ── grid ──────────────────────────────────────────────────────────────────────

def make_grid(N: int, box_L: float):
    d   = 2.0 * box_L / N
    L   = (N - 1) * d / 2
    x1d = np.linspace(-L, L, N)
    return d, x1d


# ── exact HO levels ───────────────────────────────────────────────────────────

def ho3d_exact_levels(omega: float, n_max_shell: int = 20) -> np.ndarray:
    evals = []
    for nx in range(n_max_shell + 1):
        for ny in range(n_max_shell + 1):
            for nz in range(n_max_shell + 1):
                evals.append(omega * (nx + ny + nz + 1.5))
    return np.sort(evals)


# ── Hamiltonian builders ───────────────────────────────────────────────────────

def make_H_fft(N: int, d: float, V: np.ndarray) -> LinearOperator:
    """FFT kinetic energy, no kinetic_cut (exact for JDQMR)."""
    k1d   = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    K2    = k1d ** 2
    T_k   = (K2[:, None, None] + K2[None, :, None] + K2[None, None, :]) / 2.0

    def matvec(v):
        psi  = v.reshape(N, N, N)
        Tpsi = np.fft.ifftn(T_k * np.fft.fftn(psi)).real
        return (Tpsi + V * psi).ravel()

    return LinearOperator((N**3, N**3), matvec=matvec, dtype=float)


def make_H_fd(N: int, d: float, V: np.ndarray, fd_order: int) -> LinearOperator:
    """FD kinetic energy using central-difference stencil from FD_STENCILS."""
    stencil = FD_STENCILS[fd_order].astype(float)
    inv_d2  = -0.5 / d**2

    def matvec(v):
        psi  = v.reshape(N, N, N)
        Tpsi = (convolve1d(psi, stencil, axis=0, mode='wrap') +
                convolve1d(psi, stencil, axis=1, mode='wrap') +
                convolve1d(psi, stencil, axis=2, mode='wrap')) * inv_d2
        return (Tpsi + V * psi).ravel()

    return LinearOperator((N**3, N**3), matvec=matvec, dtype=float)


# ── JDQMR runner ──────────────────────────────────────────────────────────────

def run_jdqmr(H_op, n_levels: int, tol: float, max_matvecs: int, label: str):
    ncv = max(4 * n_levels, 40)
    print(f"  [{label}]  k={n_levels} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        evals, _, stats = primme.eigsh(
            H_op,
            k              = n_levels,
            which          = 'SA',
            method         = 'PRIMME_JDQMR',
            ncv            = ncv,
            tol            = tol,
            maxMatvecs     = max_matvecs,
            return_stats   = True,
            return_history = False,
        )
        t_wall  = time.perf_counter() - t0
        n_mv    = int(stats['numMatvecs'])
        evals   = np.sort(evals.real)
        success = True
        err_msg = ""
    except Exception as exc:
        t_wall  = time.perf_counter() - t0
        n_mv    = -1
        evals   = np.full(n_levels, np.nan)
        success = False
        err_msg = str(exc)

    if success:
        print(f"E0={evals[0]:.6f}  T={t_wall:.2f}s  N_H={n_mv}")
    else:
        print(f"FAILED: {err_msg}")

    return dict(label=label, evals=evals.tolist(), t_wall=t_wall,
                n_mv=n_mv, success=success, err_msg=err_msg)


# ── N sweep parser ─────────────────────────────────────────────────────────────

def parse_N_sweep(s: str) -> list:
    s = s.strip()
    if ':' in s:
        parts = [int(x) for x in s.split(':')]
        if len(parts) == 2: return list(range(parts[0], parts[1]))
        if len(parts) == 3: return list(range(parts[0], parts[1], parts[2]))
    return [int(x) for x in s.split(',') if x.strip()]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not HAS_PRIMME:
        print("ERROR: primme not installed.  pip install primme")
        sys.exit(1)

    ap = argparse.ArgumentParser(description='JDQMR FD vs FFT on 3-D harmonic oscillator')
    ap.add_argument('--omega',       type=float, default=1.0)
    ap.add_argument('--box_L',       type=float, default=5.0)
    ap.add_argument('--n_levels',    type=int,   default=10,
                    help='Number of lowest eigenvalues to solve for')
    ap.add_argument('--fd_order',    type=str,   default='2,4,6,8,10,12',
                    help='Comma-separated FD orders, e.g. 2,4,6,8')
    ap.add_argument('--N_sweep',     type=str,   default='10:25',
                    help='N values: "10:25" → range(10,25), "10,15,20" → list')
    ap.add_argument('--tol',         type=float, default=1e-8)
    ap.add_argument('--max_matvecs', type=int,   default=50000)
    ap.add_argument('--dpi',         type=int,   default=150)
    ap.add_argument('--out_json',    type=str,   default='jdqmr_ho3d_sweep.json')
    args = ap.parse_args()

    fd_orders = [int(x) for x in args.fd_order.split(',') if x.strip()]
    for o in fd_orders:
        if o not in FD_STENCILS:
            print(f"ERROR: fd_order={o} not in FD_STENCILS {sorted(FD_STENCILS)}")
            sys.exit(1)

    N_list       = parse_N_sweep(args.N_sweep)
    method_labels = ['fft'] + [f'fd{o}' for o in fd_orders]
    exact_all    = ho3d_exact_levels(args.omega, n_max_shell=25)

    print(f"=== JDQMR HO3D sweep ===")
    print(f"omega={args.omega}  box_L={args.box_L}  n_levels={args.n_levels}")
    print(f"N_list={N_list}")
    print(f"fd_orders={fd_orders}")
    print()

    t_total0 = time.perf_counter()
    sweep = []

    for N in N_list:
        d, x1d = make_grid(N, args.box_L)
        X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
        V = 0.5 * args.omega**2 * (X**2 + Y**2 + Z**2)
        print(f"── N={N:3d}  d={d:.4f}  N³={N**3} ──")

        method_results = []

        # FFT
        H_fft = make_H_fft(N, d, V)
        r = run_jdqmr(H_fft, args.n_levels, args.tol, args.max_matvecs, 'fft')
        method_results.append(r)

        # FD orders
        for order in fd_orders:
            H_fd = make_H_fd(N, d, V, order)
            r = run_jdqmr(H_fd, args.n_levels, args.tol, args.max_matvecs, f'fd{order}')
            method_results.append(r)

        # compute errors vs exact HO
        for r in method_results:
            if r['success']:
                evals = np.array(r['evals'])
                errs  = [float(np.min(np.abs(exact_all - e))) for e in evals]
                r['errs_vs_exact'] = errs
                r['max_err'] = float(np.max(errs))
                r['mean_err'] = float(np.mean(errs))
            else:
                r['errs_vs_exact'] = []
                r['max_err'] = float('nan')
                r['mean_err'] = float('nan')

        sweep.append({'N': N, 'd': d, 'methods': method_results})
        print()

    wall_total = time.perf_counter() - t_total0

    # ── save JSON ──────────────────────────────────────────────────────────────
    output = {
        'script'  : 'compare_fd_fft_jdqmr_ho3d.py',
        'datetime': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'params'  : {
            'omega'       : args.omega,
            'box_L'       : args.box_L,
            'n_levels'    : args.n_levels,
            'fd_orders'   : fd_orders,
            'N_list'      : N_list,
            'tol'         : args.tol,
            'max_matvecs' : args.max_matvecs,
        },
        'sweep'       : sweep,
        'wall_total_s': wall_total,
    }
    out_path = Path(args.out_json)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"JSON saved: {out_path}")

    # ── figures ────────────────────────────────────────────────────────────────
    stem    = out_path.stem
    out_dir = out_path.parent

    d_list  = [s['d'] for s in sweep]

    # colour / marker helpers (same as plot_fd_fft_sweep.py)
    cmap_fd = plt.cm.plasma
    fd_labels_all = [f'fd{o}' for o in fd_orders]

    def mcolor(lbl):
        if lbl == 'fft': return 'tab:blue'
        idx = fd_labels_all.index(lbl) if lbl in fd_labels_all else 0
        return cmap_fd(0.15 + 0.7 * idx / max(len(fd_labels_all) - 1, 1))

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    def mmarker(lbl):
        return markers[method_labels.index(lbl) % len(markers)]

    # ── Figure 1: max|ΔE| and mean|ΔE| vs N ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lbl in method_labels:
        max_errs  = [next((m['max_err']  for m in s['methods'] if m['label'] == lbl), np.nan)
                     for s in sweep]
        mean_errs = [next((m['mean_err'] for m in s['methods'] if m['label'] == lbl), np.nan)
                     for s in sweep]
        c, mk = mcolor(lbl), mmarker(lbl)
        axes[0].semilogy(N_list, max_errs,  marker=mk, color=c, label=lbl,
                         linewidth=1.4, markersize=6)
        axes[1].semilogy(N_list, mean_errs, marker=mk, color=c, label=lbl,
                         linewidth=1.4, markersize=6)

    for ax, title in zip(axes, ['max |ΔE|', 'mean |ΔE|']):
        ax.set_xlabel('N  (grid points per axis)', fontsize=11)
        ax.set_ylabel('Energy error vs exact HO (Ha)', fontsize=11)
        ax.set_title(f'{title}  vs  N  (JDQMR, first {args.n_levels} levels)', fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which='both', alpha=0.35)
        ax.set_xticks(N_list)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(N_list)
        ax2.set_xticklabels([f'{d:.3f}' for d in d_list], fontsize=7, rotation=45)
        ax2.set_xlabel('d  (Bohr)', fontsize=9)

    fig.suptitle(
        f'JDQMR: FFT vs FD — 3D HO  ω={args.omega}  box_L={args.box_L}\n'
        f'(eigenvalues of each discrete H, compared to exact)',
        fontsize=11)
    fig.tight_layout()
    acc_path = out_dir / f'{stem}_accuracy.png'
    fig.savefig(acc_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {acc_path}')

    # ── Figure 2: per-eigenvalue error heatmap per method ─────────────────────
    n_methods = len(method_labels)
    ncols = min(n_methods, 4)
    nrows = (n_methods + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols,
                               figsize=(4.5 * ncols, 3.5 * nrows),
                               squeeze=False)

    for mi, lbl in enumerate(method_labels):
        ax = axes2[mi // ncols][mi % ncols]
        mat = np.full((len(N_list), args.n_levels), np.nan)
        for i, s in enumerate(sweep):
            m = next((m for m in s['methods'] if m['label'] == lbl), None)
            if m and m['success'] and m['errs_vs_exact']:
                errs = np.array(m['errs_vs_exact'])
                mat[i, :len(errs)] = errs
        im = ax.pcolormesh(
            np.arange(args.n_levels + 1),
            np.array(N_list + [N_list[-1] + 1]) - 0.5,
            np.log10(np.clip(mat, 1e-14, None)),
            cmap='RdYlGn_r', vmin=-10, vmax=0,
        )
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('Eigenvalue index', fontsize=9)
        ax.set_ylabel('N', fontsize=9)
        ax.set_yticks(N_list)
        cb = fig2.colorbar(im, ax=ax, pad=0.02)
        cb.set_label('log₁₀|ΔE|', fontsize=8)

    for mi in range(n_methods, nrows * ncols):
        axes2[mi // ncols][mi % ncols].set_visible(False)

    fig2.suptitle(
        f'Per-eigenvalue error  log₁₀|ΔE vs exact HO|\n'
        f'ω={args.omega}  box_L={args.box_L}  JDQMR  (no filter)',
        fontsize=11)
    fig2.tight_layout()
    evals_path = out_dir / f'{stem}_evals.png'
    fig2.savefig(evals_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {evals_path}')

    # ── Figure 3: wall time vs N ───────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for lbl in method_labels:
        times = [next((m['t_wall'] for m in s['methods'] if m['label'] == lbl), np.nan)
                 for s in sweep]
        ax3.plot(N_list, times, marker=mmarker(lbl), color=mcolor(lbl),
                 label=lbl, linewidth=1.4, markersize=6)
    ax3.set_xlabel('N  (grid points per axis)', fontsize=11)
    ax3.set_ylabel('Wall time (s)', fontsize=11)
    ax3.set_title(f'JDQMR wall time vs N\nω={args.omega}  box_L={args.box_L}  k={args.n_levels}',
                  fontsize=10)
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.35)
    ax3.set_xticks(N_list)
    fig3.tight_layout()
    tim_path = out_dir / f'{stem}_timing.png'
    fig3.savefig(tim_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved: {tim_path}')

    print(f'\nTotal wall time: {wall_total:.1f}s')


if __name__ == '__main__':
    main()
