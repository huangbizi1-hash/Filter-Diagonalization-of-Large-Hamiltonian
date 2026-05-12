#!/usr/bin/env python3
"""plot_fd_fft_sweep.py

Read a JSON produced by compare_fd_fft_explosion.py and generate figures
comparing FFT vs finite-difference explosion filter accuracy and timing.

Three figures are saved next to the JSON file:
  <stem>_accuracy.png   — max|ΔE| and mean|ΔE| vs N for each method
  <stem>_evals.png      — individual Ritz eigenvalue errors vs exact level
  <stem>_timing.png     — filter time per grid-point (µs) vs N

Usage:
  python plot_fd_fft_sweep.py fd_fft_explosion.json
  python plot_fd_fft_sweep.py fd_fft_explosion.json --n_evals 20 --dpi 150
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── HO exact eigenvalues ───────────────────────────────────────────────────────

def ho3d_exact_levels(omega: float, n_max_shell: int = 15):
    """Return sorted array of exact 3-D HO eigenvalues (with degeneracy).

    E_{nx,ny,nz} = omega * (nx + ny + nz + 3/2), all nx,ny,nz >= 0.
    """
    evals = []
    for nx in range(n_max_shell + 1):
        for ny in range(n_max_shell + 1):
            for nz in range(n_max_shell + 1):
                evals.append(omega * (nx + ny + nz + 1.5))
    return np.sort(evals)


# ── helpers ────────────────────────────────────────────────────────────────────

def match_ritz_to_exact(ritz: np.ndarray, exact: np.ndarray):
    """Nearest-exact match for each Ritz value; return absolute errors."""
    errors = []
    for r in ritz:
        errors.append(float(np.min(np.abs(exact - r))))
    return np.array(errors)


def method_color(label: str, all_labels: list):
    cmap_fd = plt.cm.plasma
    fd_labels = [l for l in all_labels if l.startswith('fd')]
    if label == 'fft':
        return 'tab:blue'
    idx = fd_labels.index(label) if label in fd_labels else 0
    return cmap_fd(0.15 + 0.7 * idx / max(len(fd_labels) - 1, 1))


def method_marker(label: str):
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    all_methods = ['fft', 'fd2', 'fd4', 'fd6', 'fd8', 'fd10', 'fd12',
                   'fd14', 'fd16', 'fd18', 'fd20']
    idx = all_methods.index(label) if label in all_methods else 0
    return markers[idx % len(markers)]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Plot FD vs FFT sweep results.')
    ap.add_argument('json', help='Path to fd_fft_explosion.json')
    ap.add_argument('--n_evals', type=int, default=0,
                    help='Number of eigenvalues to use for accuracy metrics. '
                         '0 = all stored  [default 0]')
    ap.add_argument('--dpi', type=int, default=150,
                    help='Figure DPI  [default 150]')
    ap.add_argument('--exact_ref', action='store_true',
                    help='Compare against exact HO eigenvalues instead of eigsh E_ref')
    args = ap.parse_args()

    json_path = Path(args.json)
    with open(json_path) as fh:
        data = json.load(fh)

    params  = data['params']
    sweep   = data['sweep']
    out_dir = json_path.parent
    stem    = json_path.stem

    potential = params['potential']
    omega     = params.get('omega', 1.0)
    fd_orders = params['fd_orders']
    E_lo      = params['E_lo']
    E_hi      = params['E_hi']

    N_list = [s['N'] for s in sweep]
    method_labels = ['fft'] + [f'fd{o}' for o in fd_orders]

    # exact reference (HO only)
    if args.exact_ref and potential == 'harmonic':
        exact_all = ho3d_exact_levels(omega, n_max_shell=20)
        ref_label = 'exact HO'
    else:
        exact_all = None
        ref_label = 'eigsh (FFT)'

    # ── collect per-method arrays ──────────────────────────────────────────────
    # max_dE[method][i_N], mean_dE[method][i_N], per_pt_us[method][i_N]
    max_dE   = {lbl: [] for lbl in method_labels}
    mean_dE  = {lbl: [] for lbl in method_labels}
    per_pt   = {lbl: [] for lbl in method_labels}
    all_ritz = {lbl: [] for lbl in method_labels}   # list-of-arrays per N

    for s in sweep:
        E_ref  = np.array(s['E_ref'])
        mmap   = {m['label']: m for m in s['methods']}

        for lbl in method_labels:
            m = mmap.get(lbl)
            if m is None or not m['ritz_evals']:
                max_dE[lbl].append(np.nan)
                mean_dE[lbl].append(np.nan)
                per_pt[lbl].append(np.nan)
                all_ritz[lbl].append(np.array([]))
                continue

            ritz = np.array(m['ritz_evals'])
            n_use = args.n_evals if args.n_evals > 0 else len(ritz)
            ritz  = ritz[:n_use]

            if exact_all is not None:
                errs = match_ritz_to_exact(ritz, exact_all)
            else:
                # match each Ritz to nearest E_ref
                errs = np.array([float(np.min(np.abs(E_ref - r))) for r in ritz])

            max_dE[lbl].append(float(np.max(errs)) if len(errs) else np.nan)
            mean_dE[lbl].append(float(np.mean(errs)) if len(errs) else np.nan)
            per_pt[lbl].append(m['filter_per_pt_us'])
            all_ritz[lbl].append(ritz)

    d_list = [s['d'] for s in sweep]

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1: accuracy vs N
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lbl in method_labels:
        c  = method_color(lbl, method_labels)
        mk = method_marker(lbl)
        axes[0].semilogy(N_list, max_dE[lbl],  marker=mk, color=c,
                         label=lbl, linewidth=1.4, markersize=6)
        axes[1].semilogy(N_list, mean_dE[lbl], marker=mk, color=c,
                         label=lbl, linewidth=1.4, markersize=6)

    for ax, title in zip(axes, ['max |ΔE|', 'mean |ΔE|']):
        ax.set_xlabel('N  (grid points per axis)', fontsize=11)
        ax.set_ylabel('Energy error (Ha)', fontsize=11)
        ax.set_title(f'{title}  vs  N\n'
                     f'(ref: {ref_label},  first {args.n_evals or "all"} Ritz evals)',
                     fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which='both', alpha=0.35)
        ax.set_xticks(N_list)
        # secondary x-axis showing d
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(N_list)
        ax2.set_xticklabels([f'{d:.3f}' for d in d_list], fontsize=7, rotation=45)
        ax2.set_xlabel('d  (Bohr)', fontsize=9)

    fig.suptitle(
        f'Chebyshev explosion: FFT vs FD  |  {potential}  '
        f'(ω={omega})  box_L={params["box_L"]}  m={params["cheb_m"]}  '
        f'E_lo={E_lo}',
        fontsize=11)
    fig.tight_layout()
    acc_path = out_dir / f'{stem}_accuracy.png'
    fig.savefig(acc_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {acc_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2: per-eigenvalue error heatmap (N × eval index) per method
    # ══════════════════════════════════════════════════════════════════════════
    n_methods = len(method_labels)
    ncols = min(n_methods, 4)
    nrows = (n_methods + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols,
                                figsize=(4.5 * ncols, 3.5 * nrows),
                                squeeze=False)

    n_evals_max = max(
        (len(all_ritz[lbl][i]) for lbl in method_labels
         for i in range(len(N_list))),
        default=1,
    )

    for mi, lbl in enumerate(method_labels):
        ax = axes2[mi // ncols][mi % ncols]
        mat = np.full((len(N_list), n_evals_max), np.nan)
        for i, ritz in enumerate(all_ritz[lbl]):
            if len(ritz) == 0:
                continue
            if exact_all is not None:
                errs = match_ritz_to_exact(ritz, exact_all)
            else:
                E_ref = np.array(sweep[i]['E_ref'])
                errs  = np.array([float(np.min(np.abs(E_ref - r))) for r in ritz])
            mat[i, :len(errs)] = errs

        im = ax.pcolormesh(
            np.arange(n_evals_max + 1),
            np.array(N_list + [N_list[-1] + 1]) - 0.5,
            np.log10(np.clip(mat, 1e-14, None)),
            cmap='RdYlGn_r', vmin=-6, vmax=0,
        )
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('Ritz eigenvalue index', fontsize=9)
        ax.set_ylabel('N', fontsize=9)
        ax.set_yticks(N_list)
        cb = fig2.colorbar(im, ax=ax, pad=0.02)
        cb.set_label('log₁₀|ΔE|', fontsize=8)

    # hide unused subplots
    for mi in range(n_methods, nrows * ncols):
        axes2[mi // ncols][mi % ncols].set_visible(False)

    fig2.suptitle(
        f'Per-eigenvalue error  log₁₀|ΔE|  (ref: {ref_label})\n'
        f'{potential}  box_L={params["box_L"]}  m={params["cheb_m"]}',
        fontsize=11)
    fig2.tight_layout()
    evals_path = out_dir / f'{stem}_evals.png'
    fig2.savefig(evals_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {evals_path}')

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 3: timing per grid-point vs N
    # ══════════════════════════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for lbl in method_labels:
        c  = method_color(lbl, method_labels)
        mk = method_marker(lbl)
        ax3.plot(N_list, per_pt[lbl], marker=mk, color=c,
                 label=lbl, linewidth=1.4, markersize=6)
    ax3.set_xlabel('N  (grid points per axis)', fontsize=11)
    ax3.set_ylabel('Filter time per grid-point (µs)', fontsize=11)
    ax3.set_title(
        f'f(H) cost per grid-point\n'
        f'{potential}  box_L={params["box_L"]}  '
        f'm={params["cheb_m"]}  n_random={params["n_random"]}',
        fontsize=10)
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.35)
    ax3.set_xticks(N_list)
    fig3.tight_layout()
    tim_path = out_dir / f'{stem}_timing.png'
    fig3.savefig(tim_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved: {tim_path}')


if __name__ == '__main__':
    main()
