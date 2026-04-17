"""
test_filter.py — Filter diagonalization on the GNN sparse grid.

Compares GNN H operator vs pure FD H operator as the Hamiltonian inside
the Newton-filter recursion.  Outputs a Markdown summary and a comparison
plot saved to run_dir/filter_test.png.

Usage (via run_gnn.py):
    python run_gnn.py --mode test_filter --run_dir gnn_models/XXXXXXXX \\
        --filter_nc 200 --filter_el -0.17 --filter_n_random 30

Physics
-------
Runs on the GNN's native sparse grid: N_sparse³ nodes, step d_sparse.
Spectral range is estimated automatically from the physics constants.
"""

import os
import time

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from .physics  import N_sparse, d_sparse, V_sparse
from .gnn_operator import build_gnn_operator

# ── fft_code imports (filter coefficients live there) ────────────────────────
try:
    from fft_code.params       import PhysParams
    from fft_code.filter_coeff import build_filter_coefficients, _filt_func_gaussian
    _HAS_FILTER = True
except ImportError:
    _HAS_FILTER = False


# ── Newton filter + Rayleigh-Ritz (self-contained, same logic as
#    compare_fd_filter.py but independent of that script) ─────────────────────

def _apply_filter_all(H_op, psi_flat, nodes, an, par):
    """
    f_i(H)|ψ⟩ for all ms filter centres, sharing Newton basis vectors.
    H-apply count = nc.  Returns (ms, n_grid).
    """
    ms, nc = an.shape
    results  = an[:, 0:1] * psi_flat[None, :]
    psi_prev = psi_flat.copy()
    for j in range(1, nc):
        H_psi    = H_op.matvec(psi_prev)
        psi_curr = ((4.0 / par.dE) * (H_psi - par.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        results += an[:, j:j+1] * psi_curr[None, :]
        psi_prev = psi_curr
    return results


def _rayleigh_ritz(basis_mat, H_op, svd_tol=1e-3, max_energies=20):
    """SVD + Rayleigh-Ritz on columns of basis_mat.  Returns (energies, rank)."""
    norms = np.linalg.norm(basis_mat, axis=0)
    mask  = norms > 1e-15
    B     = basis_mat[:, mask] / norms[None, mask]

    Q, R         = np.linalg.qr(B, mode='reduced')
    _, sigma, _  = np.linalg.svd(R, full_matrices=False)
    # Use U1 from a second SVD to get the orthonormal Ritz basis
    U1, _, _     = np.linalg.svd(R, full_matrices=False)
    r            = max(1, int(np.sum(sigma > svd_tol)))
    Ur           = (Q @ U1)[:, :r]

    H_tilde = np.zeros((r, r))
    for j in range(r):
        H_tilde[:, j] = Ur.T @ H_op.matvec(Ur[:, j])

    evals, _ = eigh(H_tilde)
    return np.sort(evals.real)[:max_energies], r


# ── spectral parameter estimation ────────────────────────────────────────────

def _estimate_spectral_params():
    """
    Estimate Vmin and dE for the GNN's sparse-grid physics.

    Vmin: min(V_sparse) with a small margin.
    dE:   |Vmin| + 3 * T_max_1D, where T_max_1D = 2/d_sparse² is the 2nd-order
          FD kinetic energy ceiling per axis.
    """
    vmin  = float(V_sparse.min()) - 0.5          # margin below potential well
    t_max = 3.0 * (2.0 / d_sparse**2)            # crude 3D FD kinetic ceiling
    d_e   = abs(vmin) + t_max + 2.0              # extra 2 Ha safety margin
    return vmin, d_e


# ── main test function ────────────────────────────────────────────────────────

def test_gnn_filter(
    run_dir:         str,
    nc:              int   = 200,
    el:              float = -0.17,
    n_random:        int   = 20,
    n_max_energies:  int   = 10,
    svd_tol:         float = 1e-3,
    output_root:     str   = ".",
    device:          str   = 'cpu',
):
    """
    Run filter diagonalization with GNN H and FD H, and compare results.

    Parameters
    ----------
    run_dir        : GNN run directory
    nc             : number of Newton steps (filter order)
    el             : target energy (Ha); filter centres here
    n_random       : number of random starting vectors
    n_max_energies : max Ritz values to report
    svd_tol        : SVD rank truncation threshold in Rayleigh-Ritz
    output_root    : directory for filter_test.png (default: run_dir)
    device         : torch device for GNN evaluations
    """
    if not _HAS_FILTER:
        raise ImportError(
            "fft_code not found.  Run from the repo root so that "
            "fft_code/ is on sys.path.")

    print(f"\n{'='*60}")
    print(f"  GNN Filter Test   run_dir={run_dir}")
    print(f"  nc={nc}  El={el}  n_random={n_random}")
    print(f"{'='*60}")

    # ── spectral parameters ───────────────────────────────────────────────
    vmin, d_e = _estimate_spectral_params()
    dt        = (nc / (d_e * 2.5)) ** 2
    par       = PhysParams(dE=d_e, Vmin=vmin, dt=dt)
    print(f"  Vmin={vmin:.3f}  dE={d_e:.3f}  dt={dt:.4f}")

    # ── Newton filter coefficients (shared between GNN and FD) ────────────
    El_list     = np.array([el])
    filter_func = lambda x, el_: _filt_func_gaussian(x, el_, dt)

    print("  Building Newton filter coefficients...")
    t0 = time.perf_counter()
    an, samp = build_filter_coefficients(
        El_list, par, nc,
        filter_func=filter_func,
        samp_method="ashkenazy",
        interpolation_tolerance=1e-6,
        enhance_step=10,
        max_enhance_iters=30,
    )
    nc_true = len(samp)
    print(f"  nc_true={nc_true}  ({time.perf_counter()-t0:.1f}s)")
    nodes = samp   # Newton nodes (scaled coordinates)

    # ── operators ────────────────────────────────────────────────────────
    n_grid = N_sparse ** 3
    rng    = np.random.default_rng(42)

    print("  Building GNN operator...")
    gnn_op = build_gnn_operator(run_dir, use_fd=False, device=device)
    print("  Building FD operator...")
    fd_op  = build_gnn_operator(run_dir, use_fd=True,  device=device)

    # ── run filter for each operator ──────────────────────────────────────
    results = {}
    for label, H_op in [("GNN", gnn_op), ("FD", fd_op)]:
        print(f"\n  [{label}] filtering {n_random} random vectors  (nc={nc_true} H-applies each)...")
        t0 = time.perf_counter()

        filtered = np.zeros((n_random, n_grid))
        for i in range(n_random):
            psi_flat = rng.standard_normal(n_grid)
            psi_flat /= np.linalg.norm(psi_flat)
            out = _apply_filter_all(H_op, psi_flat, nodes, an, par)
            # out shape: (ms, n_grid) with ms=1
            v = out[0]
            nrm = np.linalg.norm(v)
            if nrm > 0:
                filtered[i] = v / nrm
            if (i + 1) % 10 == 0:
                print(f"    filtered {i+1}/{n_random}", flush=True)

        t_filter = time.perf_counter() - t0
        n_H_filter = nc_true * n_random
        print(f"  Filtering done: {t_filter:.2f}s  (N_H={n_H_filter})")

        print(f"  [{label}] Rayleigh-Ritz...")
        t0 = time.perf_counter()
        energies, rank = _rayleigh_ritz(filtered.T, H_op, svd_tol, n_max_energies)
        t_rr = time.perf_counter() - t0
        n_H_rr = rank
        print(f"  RR done: {t_rr:.2f}s  rank={rank}")

        results[label] = {
            'energies':   energies,
            'rank':       rank,
            't_filter':   t_filter,
            't_rr':       t_rr,
            'n_H_filter': n_H_filter,
            'n_H_total':  n_H_filter + n_H_rr,
        }

    # ── print comparison table ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Method':<6}  {'E[0] (Ha)':>12}  {'ΔE vs FD':>12}  "
          f"{'T_wall (s)':>10}  {'N_H':>8}  {'rank':>5}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*5}")
    fd_e0 = results['FD']['energies'][0] if len(results['FD']['energies']) > 0 else float('nan')
    for label in ("GNN", "FD"):
        r = results[label]
        e0 = r['energies'][0] if len(r['energies']) > 0 else float('nan')
        de = e0 - fd_e0
        print(f"  {label:<6}  {e0:>12.6f}  {de:>+12.2e}  "
              f"{r['t_filter']+r['t_rr']:>10.2f}  {r['n_H_total']:>8}  {r['rank']:>5}")
    print(f"{'─'*60}")

    # ── plot ─────────────────────────────────────────────────────────────
    out_dir = run_dir if output_root == "." else output_root
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: energy levels found
    ax = axes[0]
    for i, (label, color) in enumerate([("GNN", "steelblue"), ("FD", "tomato")]):
        ev = results[label]['energies']
        ax.scatter(range(len(ev)), ev, color=color, label=label,
                   s=30, zorder=3+i, alpha=0.85)
    ax.axhline(el, ls='--', color='gray', lw=1, alpha=0.7, label=f"El={el}")
    ax.set_xlabel("Level index")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(f"Filter eigenvalues  (nc={nc_true}, El={el})")
    ax.legend(fontsize=9)

    # Right: GNN vs FD energy difference per level
    ax = axes[1]
    gnn_ev = results['GNN']['energies']
    fd_ev  = results['FD']['energies']
    n_common = min(len(gnn_ev), len(fd_ev))
    if n_common > 0:
        de = gnn_ev[:n_common] - fd_ev[:n_common]
        ax.bar(range(n_common), de, color="mediumpurple", alpha=0.8)
        ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel("Level index")
    ax.set_ylabel("E_GNN − E_FD (Ha)")
    ax.set_title("GNN correction to eigenvalues")

    fig.suptitle(f"Filter Diagonalization — GNN vs FD  (run: {os.path.basename(run_dir)})",
                 fontsize=10)
    fig.tight_layout()
    save_path = os.path.join(out_dir, "filter_test.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {save_path}")

    return results
