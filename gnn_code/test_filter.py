"""
test_filter.py — Filter diagonalization on the real QD potential.

Runs the Newton-filter recursion using the FD Hamiltonian (and optionally
the GNN Hamiltonian) on the real QD potential loaded from localPot.cube /
gaussian_fit_params.json.

Grid spacing is taken from the GNN training config (d_sparse field).
N is derived as round(box_extent / d_sparse) per axis.

If the QD grid size differs from the GNN's training grid size, the GNN
comparison is skipped (GNN must be retrained on the real QD potential first).

Usage (via run_gnn.py):
    python run_gnn.py --mode test_filter --run_dir gnn_models/XXXXXXXX \\
        --filter_nc 5000 \\
        --filter_el_list -0.24 -0.22 -0.20 -0.18 \\
        --filter_n_random 64

Physics parameters (real QD)
-----------------------------
  VMIN = -5.0 Ha   (true minimum of the InAs/GaAs QD potential)
  DE   = 50.0 Ha   (spectral width; same as compare_fd_filter.py)
"""

import os
import json
import time

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from .physics  import N_sparse, d_sparse
from .gnn_operator import build_gnn_operator

# ── fft_code imports (filter coefficients live there) ────────────────────────
try:
    from fft_code.params       import PhysParams
    from fft_code.filter_coeff import build_filter_coefficients, _filt_func_gaussian
    _HAS_FILTER = True
except ImportError:
    _HAS_FILTER = False

# ── real QD potential builder ─────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CUBE   = os.path.join(_REPO_ROOT, "localPot.cube")
_DEFAULT_PARAMS = os.path.join(_REPO_ROOT, "gaussian_fit_params.json")

# Fixed spectral parameters for the real QD (consistent with compare_fd_filter.py)
_QD_VMIN = -5.0   # Ha
_QD_DE   = 50.0   # Ha
_QD_RCUT = 7.0    # Å


def _load_qd_potential(d: float,
                       cube_file:   str = _DEFAULT_CUBE,
                       params_file: str = _DEFAULT_PARAMS,
                       r_cut:       float = _QD_RCUT):
    """
    Load the real QD potential at grid spacing d (Bohr).

    Uses GaussianPotentialBuilder to reconstruct the potential from Gaussian
    fit parameters, evaluated on a grid with spacing d derived from the cube
    file's spatial extent.

    Returns
    -------
    V_flat : np.ndarray, shape (N³,)   potential values (Ha)
    N      : int                        grid points per axis
    x      : np.ndarray, shape (N,)    coordinate array (Å)
    """
    try:
        from gaussian_potential_builder import GaussianPotentialBuilder
    except ImportError:
        raise ImportError(
            "gaussian_potential_builder not found.  "
            "Run from the repo root so it is on sys.path.")

    builder = GaussianPotentialBuilder(cube_file=cube_file,
                                       params_file=params_file,
                                       r_cut=r_cut)

    # Derive N from the cube file extent and desired d
    # The cube spatial extent is in Å; d is in Bohr (1 Bohr ≈ 0.529177 Å)
    # The cube file uses Bohr internally (origin and vectors are in Bohr)
    # builder.x is in Bohr already (set from cube header directly)
    box_extent_bohr = builder.x[-1] - builder.x[0]   # Bohr
    N = max(2, round(box_extent_bohr / d) + 1)

    print(f"  QD grid: box_extent={box_extent_bohr:.3f} Bohr, "
          f"d={d:.4f} Bohr → N={N} ({N**3:,} points)")

    x_arr, y_arr, z_arr, V = builder.build_potential(N)
    return V.ravel().astype(np.float64), N, x_arr


# ── Newton filter + Rayleigh-Ritz ─────────────────────────────────────────────

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
    if not mask.any():
        print("    WARNING: all filtered vectors are zero — no eigenvalues found")
        return np.array([]), 0

    B = basis_mat[:, mask] / norms[None, mask]

    Q, R         = np.linalg.qr(B, mode='reduced')
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    r            = max(1, int(np.sum(sigma > svd_tol)))
    Ur           = (Q @ U1)[:, :r]

    if Ur.shape[1] == 0:
        print("    WARNING: Ritz basis has 0 columns after truncation")
        return np.array([]), 0

    H_tilde = np.zeros((r, r))
    for j in range(r):
        H_tilde[:, j] = Ur.T @ H_op.matvec(Ur[:, j])

    evals, _ = eigh(H_tilde)
    return np.sort(evals.real)[:max_energies], r


# ── Gaussian window plot ──────────────────────────────────────────────────────

def _plot_filter_windows(El_list, vmin, d_e, dt, nc_true, out_path):
    """Plot Gaussian filter windows f(E; El) over the physical energy range."""
    e_min  = vmin
    e_max  = vmin + d_e
    x_phys = np.linspace(e_min, e_max, 2000)

    fig, ax = plt.subplots(figsize=(9, 4))
    cmap    = plt.cm.viridis
    colors  = [cmap(i / max(len(El_list) - 1, 1)) for i in range(len(El_list))]

    for el, color in zip(El_list, colors):
        w = np.array([_filt_func_gaussian(e, el, dt) for e in x_phys])
        ax.plot(x_phys, w, color=color, lw=1.5, label=f"El={el:.3f}")

    ax.set_xlabel("Energy (Ha)")
    ax.set_ylabel("Filter weight")
    ax.set_title(f"Gaussian filter windows  (nc={nc_true}, dt={dt:.4f})")
    ax.legend(fontsize=7, ncol=3)
    ax.axhline(0, color='k', lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Filter windows → {out_path}")


# ── main test function ────────────────────────────────────────────────────────

def test_gnn_filter(
    run_dir:         str,
    nc:              int        = 5000,
    el_list:         list       = None,
    n_random:        int        = 64,
    n_max_energies:  int        = 20,
    svd_tol:         float      = 1e-3,
    output_root:     str        = ".",
    device:          str        = 'cpu',
    cube_file:       str        = _DEFAULT_CUBE,
    params_file:     str        = _DEFAULT_PARAMS,
):
    """
    Run filter diagonalization on the real QD potential.

    The FD Hamiltonian always uses the real QD potential at d_sparse resolution
    (d_sparse read from run_dir/config.json).

    The GNN Hamiltonian is included only when the GNN was trained on the same
    grid size as the QD grid; otherwise a warning is printed and GNN is skipped.

    Parameters
    ----------
    run_dir        : GNN run directory (config.json + epoch_*.pt)
    nc             : Newton filter order (H-applies per random vector)
    el_list        : list of target energies (Ha)
    n_random       : number of random starting vectors
    n_max_energies : max Ritz values to report
    svd_tol        : SVD rank truncation threshold in Rayleigh-Ritz
    output_root    : directory for output plots (default: run_dir)
    device         : torch device for GNN evaluations
    cube_file      : path to the QD cube file (default: localPot.cube)
    params_file    : path to the Gaussian fit params JSON (default: gaussian_fit_params.json)
    """
    if not _HAS_FILTER:
        raise ImportError(
            "fft_code not found.  Run from the repo root so that "
            "fft_code/ is on sys.path.")

    if el_list is None:
        el_list = [-0.17]
    El_list = np.array(el_list)
    ms      = len(El_list)

    # ── load config to get d_sparse ───────────────────────────────────────
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    d  = config.get('d_sparse', d_sparse)
    N_gnn = config.get('N_sparse', N_sparse)   # GNN was trained on this grid

    print(f"\n{'='*60}")
    print(f"  GNN Filter Test (real QD)   run_dir={run_dir}")
    print(f"  nc={nc}  El_list={El_list.tolist()}  n_random={n_random}")
    print(f"  d_sparse={d:.4f} Bohr (from config)")
    print(f"{'='*60}")

    # ── load real QD potential ────────────────────────────────────────────
    print("  Loading real QD potential...")
    V_qd, N_qd, _ = _load_qd_potential(d, cube_file=cube_file,
                                        params_file=params_file)
    print(f"  V_qd: min={V_qd.min():.4f}  max={V_qd.max():.4f} Ha")

    # ── spectral parameters (fixed for real QD) ───────────────────────────
    vmin = _QD_VMIN
    d_e  = _QD_DE
    dt   = (nc / (d_e * 2.5)) ** 2
    par  = PhysParams(dE=d_e, Vmin=vmin, dt=dt)
    print(f"  Vmin={vmin:.3f}  dE={d_e:.3f}  dt={dt:.6f}")

    # ── GNN applicability check ───────────────────────────────────────────
    gnn_applicable = (N_qd == N_gnn)
    if not gnn_applicable:
        print(f"  WARNING: QD grid N={N_qd} ≠ GNN training grid N={N_gnn}.")
        print(f"           GNN comparison skipped — retrain GNN on real QD first.")

    # ── Newton filter coefficients ────────────────────────────────────────
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
    nodes   = samp
    print(f"  nc_true={nc_true}  ms={ms}  ({time.perf_counter()-t0:.1f}s)")

    # ── plot filter windows ───────────────────────────────────────────────
    out_dir = run_dir if output_root == "." else output_root
    _plot_filter_windows(
        El_list, vmin, d_e, dt, nc_true,
        os.path.join(out_dir, "filter_windows.png"),
    )

    # ── build operators ───────────────────────────────────────────────────
    print("  Building FD operator (real QD)...")
    fd_op = build_gnn_operator(run_dir, use_fd=True, device=device,
                               V_ext=V_qd, N_grid=N_qd)

    operators = [("FD", fd_op)]

    if gnn_applicable:
        print("  Building GNN operator (real QD)...")
        gnn_op = build_gnn_operator(run_dir, use_fd=False, device=device,
                                    V_ext=V_qd, N_grid=N_qd)
        operators = [("GNN", gnn_op)] + operators

    # ── run filter ────────────────────────────────────────────────────────
    n_grid = N_qd ** 3
    rng    = np.random.default_rng(42)
    results = {}

    for label, H_op in operators:
        print(f"\n  [{label}] filtering {n_random} random vectors  "
              f"(nc={nc_true} H-applies each, ms={ms} El centres)...")
        t0 = time.perf_counter()

        filtered = np.zeros((ms * n_random, n_grid))
        for i in range(n_random):
            psi_flat = rng.standard_normal(n_grid)
            psi_flat /= np.linalg.norm(psi_flat)
            out = _apply_filter_all(H_op, psi_flat, nodes, an, par)
            for ie in range(ms):
                v   = out[ie]
                nrm = np.linalg.norm(v)
                if nrm > 0:
                    filtered[ie * n_random + i] = v / nrm
            if (i + 1) % 10 == 0:
                print(f"    filtered {i+1}/{n_random}", flush=True)

        t_filter = time.perf_counter() - t0
        n_H_filter = nc_true * n_random
        print(f"  Filtering done: {t_filter:.2f}s  (N_H={n_H_filter}, "
              f"basis_cols={ms * n_random})")

        print(f"  [{label}] Rayleigh-Ritz...")
        t0 = time.perf_counter()
        energies, rank = _rayleigh_ritz(filtered.T, H_op, svd_tol, n_max_energies)
        t_rr = time.perf_counter() - t0
        print(f"  RR done: {t_rr:.2f}s  rank={rank}  n_energies={len(energies)}")

        results[label] = {
            'energies':   energies,
            'rank':       rank,
            't_filter':   t_filter,
            't_rr':       t_rr,
            'n_H_filter': n_H_filter,
            'n_H_total':  n_H_filter + rank,
        }

    # ── print comparison table ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Method':<6}  {'E[0] (Ha)':>12}  {'T_wall (s)':>10}  "
          f"{'N_H':>8}  {'rank':>5}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*8}  {'─'*5}")
    fd_ev = results.get('FD', {}).get('energies', np.array([]))
    fd_e0 = fd_ev[0] if len(fd_ev) > 0 else float('nan')
    for label, _ in operators:
        r  = results[label]
        ev = r['energies']
        e0 = ev[0] if len(ev) > 0 else float('nan')
        print(f"  {label:<6}  {e0:>12.6f}  "
              f"{r['t_filter']+r['t_rr']:>10.2f}  {r['n_H_total']:>8}  {r['rank']:>5}")
    print(f"{'─'*60}")

    # ── plot ─────────────────────────────────────────────────────────────
    colors = {"GNN": "steelblue", "FD": "tomato"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for i, (label, _) in enumerate(operators):
        ev = results[label]['energies']
        if len(ev) > 0:
            ax.scatter(range(len(ev)), ev, color=colors.get(label, "gray"),
                       label=label, s=30, zorder=3+i, alpha=0.85)
    for el in El_list:
        ax.axhline(el, ls='--', color='gray', lw=0.8, alpha=0.6)
    ax.axhline(El_list[0], ls='--', color='gray', lw=0.8, alpha=0.6,
               label=f"El targets ({ms})")
    ax.set_xlabel("Level index")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(f"Filter eigenvalues — real QD  (nc={nc_true}, ms={ms})")
    ax.legend(fontsize=9)

    ax = axes[1]
    if gnn_applicable and 'GNN' in results and 'FD' in results:
        gnn_ev = results['GNN']['energies']
        n_common = min(len(gnn_ev), len(fd_ev))
        if n_common > 0:
            de = gnn_ev[:n_common] - fd_ev[:n_common]
            ax.bar(range(n_common), de, color="mediumpurple", alpha=0.8)
            ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel("E_GNN − E_FD (Ha)")
        ax.set_title("GNN correction to eigenvalues")
    else:
        if len(fd_ev) > 0:
            ax.bar(range(len(fd_ev)), fd_ev, color="tomato", alpha=0.8)
        ax.set_ylabel("Energy (Ha)")
        ax.set_title("FD eigenvalues (GNN not trained on real QD)")
    ax.set_xlabel("Level index")

    fig.suptitle(
        f"Filter Diagonalization — real QD  (run: {os.path.basename(run_dir)})",
        fontsize=10)
    fig.tight_layout()
    save_path = os.path.join(out_dir, "filter_test.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {save_path}")

    return results
