"""
test_filter.py — Filter diagonalization on the real QD potential.

Runs Newton-filter recursion using:
  · FD Hamiltonian  — build_3d_fd_operator from ho3d_solvers_v2 (any N)
  · GNN Hamiltonian — build_gnn_operator (only when N_qd == GNN training N)

The QD potential is reconstructed via GaussianPotentialBuilder at the grid
spacing d_sparse read from the GNN run's config.json.  N_qd is then derived
as round(box_extent / d_sparse), making the grid consistent with the training.

The GNN is a local operator (MLP on nearest-neighbour edges) whose weights
depend only on the grid spacing d_sparse.  build_gnn_operator now rebuilds
the graph for the actual N_qd, so the GNN can run on any grid size as long
as d_sparse matches the training value.

Results are stored as:
  <out_dir>/filter_test_<TIMESTAMP>.json    — full metadata + energies
  <out_dir>/filter_test_<TIMESTAMP>.md     — Markdown comparison table
  <out_dir>/filter_windows.png             — Gaussian filter windows
  <out_dir>/filter_test.png               — eigenvalue comparison plot

Usage (via run_gnn.py):
    python run_gnn.py --mode test_filter --run_dir gnn_models/XXXXXXXX \\
        --filter_nc 5000 \\
        --filter_el_list -0.24 -0.22 -0.20 -0.18 \\
        --filter_n_random 64

Spectral parameters (real QD, fixed)
-------------------------------------
  VMIN = -5.0 Ha   (same as compare_fd_filter.py)
  DE   = 50.0 Ha
"""

import json
import os
import datetime
import time
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator

from .physics  import N_sparse, d_sparse
from .gnn_operator import build_gnn_operator

# ── fft_code imports ──────────────────────────────────────────────────────────
try:
    from fft_code.params       import PhysParams
    from fft_code.filter_coeff import build_filter_coefficients, _filt_func_gaussian
    from fft_code.hamiltonian  import apply_H as _fft_apply_H
    _HAS_FILTER = True
except ImportError:
    _HAS_FILTER = False

# ── FD operator builder (works for arbitrary N) ───────────────────────────────
try:
    from ho3d_solvers_v2 import build_3d_fd_operator
    _HAS_HO3D = True
except ImportError:
    _HAS_HO3D = False

# ── real QD potential builder ─────────────────────────────────────────────────
_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CUBE   = os.path.join(_REPO_ROOT, "localPot.cube")
_DEFAULT_PARAMS = os.path.join(_REPO_ROOT, "gaussian_fit_params.json")

# Fixed spectral parameters for the real QD (matches compare_fd_filter.py)
_QD_VMIN = -5.0   # Ha
_QD_DE   = 50.0   # Ha
_QD_RCUT = 7.0    # Å (cutoff radius for Gaussian reconstruction)


# ── QD potential loader ───────────────────────────────────────────────────────

def _load_qd_potential(d, cube_file=_DEFAULT_CUBE, params_file=_DEFAULT_PARAMS,
                       r_cut=_QD_RCUT):
    """
    Load the real QD potential at grid spacing d (Bohr).

    N is derived from the cube file's spatial extent divided by d, so the
    resulting grid has spacing exactly d (within rounding).

    Returns
    -------
    pot_grid : PotentialGrid  — N×N×N potential ready for build_3d_fd_operator
    N        : int            — grid points per axis
    d_actual : float          — actual grid spacing (Bohr), ≈ d
    """
    try:
        from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid
    except ImportError:
        raise ImportError(
            "gaussian_potential_builder not found — run from repo root.")

    builder = GaussianPotentialBuilder(cube_file=cube_file,
                                       params_file=params_file,
                                       r_cut=r_cut)

    # Cube file positions are in Bohr. Derive N from box extent and d.
    box_extent = float(builder.x[-1] - builder.x[0])   # Bohr
    N          = max(2, round(box_extent / d) + 1)

    x, y, z, V = builder.build_potential(N)
    d_actual    = float(x[1] - x[0])

    print(f"  QD grid: box={box_extent:.3f} Bohr, d_req={d:.4f} → "
          f"N={N} ({N**3:,} pts), d_actual={d_actual:.4f} Bohr")
    print(f"  V_qd:   min={V.min():.4f}  max={V.max():.4f} Ha")

    return PotentialGrid(x, y, z, V, source="QD"), N, d_actual


# ── Newton filter ─────────────────────────────────────────────────────────────

def _apply_filter_all(H_op, psi_flat, nodes, an, par):
    """
    f_i(H)|ψ⟩ for all ms filter centres, sharing Newton basis vectors.
    H is applied nc times total.  Returns (ms, n_grid).
    """
    ms, nc   = an.shape
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


# ── Rayleigh-Ritz ─────────────────────────────────────────────────────────────

def _rayleigh_ritz(basis_mat, H_op, svd_tol=1e-3, max_energies=20):
    """SVD + Rayleigh-Ritz on columns of basis_mat.  Returns (energies, rank)."""
    norms = np.linalg.norm(basis_mat, axis=0)
    mask  = norms > 1e-15
    if not mask.any():
        print("    WARNING: all filtered vectors are zero — no eigenvalues found")
        return np.array([]), 0

    B            = basis_mat[:, mask] / norms[None, mask]
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


# ── FFT operator ──────────────────────────────────────────────────────────────

def _build_fft_qd_operator(V_qd_3d: np.ndarray, N_qd: int,
                            d_actual: float) -> LinearOperator:
    """
    Wrap apply_H (pyfftw, PBC) as a scipy LinearOperator for the QD grid.
    No kinetic cutoff: full continuous-space kinetic spectrum.
    """
    kx_1d = 2 * np.pi * np.fft.fftfreq(N_qd, d=d_actual)
    Kx, Ky, Kz = np.meshgrid(kx_1d, kx_1d, kx_1d, indexing='ij')
    T_k_diag = ((Kx**2 + Ky**2 + Kz**2) / 2.0).astype(np.float64)
    V_3d  = V_qd_3d.reshape(N_qd, N_qd, N_qd).astype(np.float64)
    n_grid = N_qd ** 3

    def _matvec(psi_flat: np.ndarray) -> np.ndarray:
        psi_3d = np.asarray(psi_flat, dtype=np.complex128).reshape(N_qd, N_qd, N_qd)
        return _fft_apply_H(psi_3d, V_3d, T_k_diag).real.ravel().astype(np.float64)

    op = LinearOperator(shape=(n_grid, n_grid), matvec=_matvec, dtype=np.float64)
    op.label = "FFT"
    return op


# ── k=0 保真度测试 ────────────────────────────────────────────────────────────

def _k0_fidelity_test(operators: list, nodes: np.ndarray, an: np.ndarray,
                      par, n_grid: int, El_list: np.ndarray) -> dict:
    """
    用 k=0（常数归一化态）作为初始滤波态，比较三种算符的结果。

    对每个 El 中心返回：
      energies   — 滤波后态的 Rayleigh 商
      fidelities — |<ψ_FFT | ψ_method>|（以 FFT 为基准，FFT 自身为 1.0）

    Parameters
    ----------
    operators : list of (label, H_op)，FFT 须排在首位以作参照
    """
    ms     = an.shape[0]
    psi_k0 = np.ones(n_grid, dtype=np.float64) / np.sqrt(n_grid)

    # 先对所有算符滤波，归一化
    filtered_by_label: dict[str, list] = {}
    for label, H_op in operators:
        out  = _apply_filter_all(H_op, psi_k0, nodes, an, par)  # (ms, n_grid)
        vecs = []
        for ie in range(ms):
            v   = out[ie]
            nrm = np.linalg.norm(v)
            vecs.append(v / nrm if nrm > 1e-15 else None)
        filtered_by_label[label] = vecs

    # FFT 滤波态作为参照
    fft_vecs = filtered_by_label.get("FFT")

    results = {}
    for label, H_op in operators:
        vecs       = filtered_by_label[label]
        energies   = []
        fidelities = []
        for ie in range(ms):
            v = vecs[ie]
            if v is None:
                energies.append(float('nan'))
                fidelities.append(float('nan'))
                continue
            # Rayleigh 商
            Hv = H_op.matvec(v)
            energies.append(float(np.dot(v, Hv)))
            # 保真度
            if fft_vecs is not None and fft_vecs[ie] is not None:
                fidelities.append(float(abs(np.dot(v, fft_vecs[ie]))))
            else:
                fidelities.append(float('nan'))
        results[label] = {"energies": energies, "fidelities": fidelities}

    # 打印结果
    print(f"\n  k=0 fidelity test  (El_list={El_list.tolist()})")
    header = f"  {'Method':<8}  " + "  ".join(
        f"El={el:.3f}:E/Fid" for el in El_list)
    print(header)
    for label, res in results.items():
        cols = []
        for ie in range(ms):
            e   = res["energies"][ie]
            fid = res["fidelities"][ie]
            cols.append(f"{e:+.4f}/{fid:.4f}")
        print(f"  {label:<8}  " + "  ".join(cols))

    return results


# ── filter window plot ────────────────────────────────────────────────────────

def _plot_filter_windows(El_list, vmin, d_e, dt, nc_true, out_path):
    """Plot Gaussian filter windows over the physical energy range."""
    e_min  = vmin
    e_max  = vmin + d_e
    x_phys = np.linspace(e_min, e_max, 2000)
    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(El_list) - 1, 1)) for i in range(len(El_list))]

    fig, ax = plt.subplots(figsize=(9, 4))
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


# ── result storage ────────────────────────────────────────────────────────────

def _save_results(results_list, config_meta, out_dir, ts,
                  k0_results: dict = None):
    """
    Save filter results to JSON and Markdown.

    results_list : list of dicts with keys
                   label, energies, El_list, E_mean, E_std,
                   rank, t_filter, t_rr, n_H_filter, n_H_total
    k0_results   : dict returned by _k0_fidelity_test (optional)
    """
    out_dir = Path(out_dir)

    # ── JSON ─────────────────────────────────────────────────────────────────
    def _to_list(arr):
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        if arr is None:
            return None
        return list(arr)

    json_results = []
    for r in results_list:
        ev = r["energies"]
        json_results.append({
            "label":     r["label"],
            "El_list":   _to_list(r.get("El_list", [])),
            "E0":        float(ev[0]) if len(ev) > 0 else None,
            "E_mean":    _to_list(r.get("E_mean")),   # per-El mean Rayleigh quotient
            "E_std":     _to_list(r.get("E_std")),    # per-El std
            "t_wall":    r["t_filter"] + r["t_rr"],
            "n_H_filter": r["n_H_filter"],
            "n_H_total": r["n_H_total"],
            "rr_rank":   r["rank"],
            "energies":  _to_list(ev),
            "success":   len(ev) > 0,
        })

    output = {
        "script":   "gnn_code/test_filter.py",
        "datetime": datetime.datetime.now().isoformat(),
        "config":   config_meta,
        "results":  json_results,
        "k0_test":  k0_results,   # None if not requested
    }
    json_path = out_dir / f"filter_test_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  JSON → {json_path}")

    # ── Markdown table ────────────────────────────────────────────────────────
    fd_E0 = next(
        (r["E0"] for r in json_results if r["label"].startswith("FD") and r["E0"] is not None),
        float("nan"))

    lines = [
        f"# GNN Filter Test — real QD\n",
        f"run_dir: `{config_meta.get('run_dir', '?')}`  "
        f"| datetime: {output['datetime']}\n",
        f"d_sparse={config_meta.get('d_sparse', '?')} Bohr  "
        f"| N_qd={config_meta.get('N_qd', '?')}  "
        f"| nc_true={config_meta.get('nc_true', '?')}  "
        f"| n_random={config_meta.get('n_random', '?')}  "
        f"| El_list={config_meta.get('El_list', '?')}\n",
        "| Method | E[0] (Ha) | E_mean (Ha) | E_std (Ha) | ΔE vs FD | T_wall (s) | N_H | RR rank |",
        "|--------|-----------|-------------|------------|----------|------------|-----|---------|",
    ]
    for r in json_results:
        e0    = r["E0"] if r["E0"] is not None else float("nan")
        de    = (e0 - fd_E0) if r["E0"] is not None else float("nan")
        emean = r["E_mean"]
        emean_s = f"{float(emean[0]):.4f}" if emean else "  nan  "
        estd  = r["E_std"]
        estd_s  = f"{float(estd[0]):.4f}"  if estd  else "  nan  "
        lines.append(
            f"| {r['label']:8s} | {e0:12.6f} | {emean_s:>11s} | {estd_s:>10s} "
            f"| {de:+.2e} | {r['t_wall']:10.2f} | {r['n_H_total']:8d} | {r['rr_rank']:7d} |"
        )

    # k=0 fidelity table
    if k0_results:
        El_cfg = config_meta.get('El_list', [])
        lines += ["", "## k=0 fidelity test\n",
                  "| Method | " + " | ".join(f"El={el:.3f} E(Ha)" for el in El_cfg) +
                  " | " + " | ".join(f"El={el:.3f} Fid" for el in El_cfg) + " |",
                  "|--------|" + "|".join("--------" for _ in El_cfg) + "|" +
                  "|".join("--------" for _ in El_cfg) + "|"]
        for label, res in k0_results.items():
            e_cols   = " | ".join(f"{e:+.4f}" for e in res["energies"])
            fid_cols = " | ".join(
                f"{f:.4f}" if not np.isnan(f) else "  nan  "
                for f in res["fidelities"])
            lines.append(f"| {label:<8} | {e_cols} | {fid_cols} |")

    md_text = "\n".join(lines) + "\n"
    print("\n" + md_text)
    md_path = out_dir / f"filter_test_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"  MD   → {md_path}")


# ── main function ─────────────────────────────────────────────────────────────

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
    vmin:            float      = None,
    d_e:             float      = None,
):
    """
    Run filter diagonalization on the real QD potential.

    Grid spacing d is read from run_dir/config.json (d_sparse field).
    N_qd = round(QD_box_extent / d_sparse).

    FD runs on the N_qd grid via build_3d_fd_operator (ho3d_solvers_v2).
    GNN runs only when N_qd == GNN training N_sparse (grids match).

    Parameters
    ----------
    run_dir        : GNN run directory (config.json + epoch_*.pt)
    nc             : Newton filter order (H-applies per random vector)
    el_list        : list of target energies (Ha)
    n_random       : number of random starting vectors
    n_max_energies : max Ritz values to report
    svd_tol        : SVD rank truncation threshold
    output_root    : output directory for results (default: run_dir)
    device         : torch device for GNN ('cpu' or 'cuda')
    cube_file      : path to QD cube file (default: localPot.cube)
    params_file    : path to Gaussian fit params JSON
    """
    if not _HAS_FILTER:
        raise ImportError("fft_code not found — run from repo root.")
    if not _HAS_HO3D:
        raise ImportError("ho3d_solvers_v2 not found — run from repo root.")

    if el_list is None:
        el_list = [-0.17]
    El_list = np.array(el_list)
    ms      = len(El_list)
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(run_dir if output_root == "." else output_root)

    # ── GNN config ────────────────────────────────────────────────────────────
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)
    d         = float(config.get('d_sparse', d_sparse))
    N_gnn     = int(config.get('N_sparse',   N_sparse))
    fd_order  = int(config.get('fd_order',   4))
    model_type = config.get('model_type', 'gnn')

    print(f"\n{'='*60}")
    print(f"  GNN Filter Test (real QD)")
    print(f"  run_dir    = {run_dir}")
    print(f"  El_list    = {El_list.tolist()}")
    print(f"  nc={nc}  n_random={n_random}  d_sparse={d:.4f} Bohr")
    print(f"{'='*60}")

    # ── QD potential at GNN grid spacing ─────────────────────────────────────
    # Always use d_sparse from the GNN config, not the cube file's natural d.
    # The Gaussian-based QD potential can be re-evaluated at any spacing,
    # so we rebuild the grid at d_sparse to match the trained GNN stencil.
    print("\n  Loading real QD potential...")
    print(f"  Using d_sparse={d:.4f} Bohr from GNN config to build QD grid")
    pot_grid, N_qd, d_actual = _load_qd_potential(
        d, cube_file=cube_file, params_file=params_file)

    # ── spectral parameters ───────────────────────────────────────────────────
    if vmin is None:
        vmin = _QD_VMIN
    if d_e is None:
        d_e = _QD_DE
    dt   = (nc / (d_e * 2.5)) ** 2
    par  = PhysParams(dE=d_e, Vmin=vmin, dt=dt)
    print(f"  Spectral: Vmin={vmin}  dE={d_e}  dt={dt:.6f}")

    # ── Newton filter coefficients ────────────────────────────────────────────
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

    # ── filter window plot ────────────────────────────────────────────────────
    _plot_filter_windows(El_list, vmin, d_e, dt, nc_true,
                         out_dir / "filter_windows.png")

    # ── build operators (order: FFT → FD → GNN) ──────────────────────────────
    V_qd_3d   = pot_grid.potential.reshape(N_qd, N_qd, N_qd)
    V_qd_flat = pot_grid.potential.ravel().astype(np.float32)
    operators: list = []

    # FFT: 精确参照（pyfftw, PBC, 无截断），始终排第一
    print(f"\n  Building FFT operator (N={N_qd}, pyfftw)...")
    try:
        fft_h_op = _build_fft_qd_operator(V_qd_3d, N_qd, d_actual)
        operators.append(("FFT", fft_h_op))
        print(f"  FFT operator ready.")
    except Exception as exc:
        print(f"  WARNING: FFT operator failed ({exc}) → FFT skipped.")

    # FD: uses build_3d_fd_operator so it works for any N_qd
    print(f"\n  Building FD operator (order={fd_order}, N={N_qd})...")
    fd_h_op, _, _ = build_3d_fd_operator(N_qd, pot_grid, fd_order=fd_order)
    fd_h_op.label = f"FD-{fd_order}"
    operators.append(("FD", fd_h_op))

    # GNN: rebuild graph for N_qd; the GNN is a local operator whose MLP weights
    # depend only on d_sparse, so it generalises to any grid size.
    gnn_applicable = True
    if N_qd != N_gnn:
        print(f"  NOTE: N_qd={N_qd} ≠ N_gnn={N_gnn}  → rebuilding graph for N={N_qd}.")
    print(f"  Building GNN operator (N={N_qd}, model={model_type})...")
    try:
        gnn_h_op = build_gnn_operator(run_dir, use_fd=False, device=device,
                                       V_ext=V_qd_flat, N_grid=N_qd)
        operators.append(("GNN", gnn_h_op))
    except Exception as exc:
        print(f"  WARNING: GNN operator failed ({exc}) → GNN skipped.")
        gnn_applicable = False

    # ── filter loop ───────────────────────────────────────────────────────────
    n_grid = N_qd ** 3
    results_list = []

    for label, H_op in operators:
        print(f"\n  [{label}] filtering {n_random} random vectors "
              f"(nc_true={nc_true}, H-applies/vec={nc_true-1}, ms={ms} El centres)...")
        rng = np.random.default_rng(42)   # same seed for all operators
        t0  = time.perf_counter()

        filtered   = np.zeros((ms * n_random, n_grid))
        # E_temp[ie] accumulates per-vector Rayleigh quotients for El centre ie
        E_temp = [[] for _ in range(ms)]

        for i in range(n_random):
            psi_flat = rng.standard_normal(n_grid)
            psi_flat /= np.linalg.norm(psi_flat)
            out = _apply_filter_all(H_op, psi_flat, nodes, an, par)
            for ie in range(ms):
                v   = out[ie]
                nrm = np.linalg.norm(v)
                if nrm > 1e-15:
                    v_n = v / nrm
                    filtered[ie * n_random + i] = v_n
                    # Rayleigh 商：<v|H|v>（参照 main.py 的 E_exp 计算）
                    Hv = H_op.matvec(v_n)
                    E_temp[ie].append(float(np.dot(v_n, Hv)))
            if (i + 1) % 10 == 0:
                print(f"    filtered {i+1}/{n_random}", flush=True)

        t_filter   = time.perf_counter() - t0
        # _apply_filter_all 循环 range(1, nc_true) = nc_true-1 次 H·apply
        n_H_filter = (nc_true - 1) * n_random
        # +ms*n_random 次来自 Rayleigh 商计算
        n_H_rq     = sum(len(v) for v in E_temp)
        print(f"  Filtering done: {t_filter:.2f}s  N_H_filter={n_H_filter}  "
              f"N_H_RQ={n_H_rq}  basis_cols={ms * n_random}")

        # per-El E_mean / E_std
        E_mean_list = [float(np.mean(E_temp[ie])) if E_temp[ie] else float('nan')
                       for ie in range(ms)]
        E_std_list  = [float(np.std(E_temp[ie]))  if E_temp[ie] else float('nan')
                       for ie in range(ms)]
        for ie in range(ms):
            print(f"    El={El_list[ie]:.3f}  E_mean={E_mean_list[ie]:.4f}  "
                  f"E_std={E_std_list[ie]:.4f}")

        print(f"  [{label}] Rayleigh-Ritz...")
        t0 = time.perf_counter()
        energies, rank = _rayleigh_ritz(filtered.T, H_op, svd_tol, n_max_energies)
        t_rr = time.perf_counter() - t0
        print(f"  RR done: {t_rr:.2f}s  rank={rank}  n_energies={len(energies)}")

        results_list.append(dict(
            label      = label,
            energies   = energies,
            El_list    = El_list.tolist(),
            E_mean     = E_mean_list,
            E_std      = E_std_list,
            rank       = rank,
            t_filter   = t_filter,
            t_rr       = t_rr,
            n_H_filter = n_H_filter,
            n_H_total  = n_H_filter + n_H_rq + rank,
        ))

    # ── k=0 保真度测试 ────────────────────────────────────────────────────────
    k0_results = _k0_fidelity_test(operators, nodes, an, par, n_grid, El_list)

    # ── print comparison table ────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  {'Method':<8}  {'E[0] (Ha)':>12}  {'E_mean[0]':>10}  "
          f"{'E_std[0]':>9}  {'N_H':>8}  {'rank':>5}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*5}")
    for r in results_list:
        ev    = r["energies"]
        e0    = ev[0] if len(ev) > 0 else float("nan")
        em    = r["E_mean"][0] if r["E_mean"] else float("nan")
        es    = r["E_std"][0]  if r["E_std"]  else float("nan")
        print(f"  {r['label']:<8}  {e0:>12.6f}  {em:>10.4f}  "
              f"{es:>9.4f}  {r['n_H_total']:>8}  {r['rank']:>5}")
    print(f"{'─'*72}")

    # ── save results ──────────────────────────────────────────────────────────
    config_meta = dict(
        run_dir       = str(run_dir),
        cube_file     = str(cube_file),
        params_file   = str(params_file),
        d_sparse      = d,
        d_actual      = d_actual,
        N_qd          = N_qd,
        N_gnn         = N_gnn,
        gnn_applicable= gnn_applicable,
        nc            = nc,
        nc_true       = nc_true,
        ms            = ms,
        El_list       = El_list.tolist(),
        n_random      = n_random,
        svd_tol       = svd_tol,
        n_max_energies= n_max_energies,
        fd_order      = fd_order,
        model_type    = model_type,
        vmin          = vmin,
        d_e           = d_e,
        dt            = dt,
    )
    _save_results(results_list, config_meta, out_dir, ts, k0_results=k0_results)

    # ── eigenvalue plot ───────────────────────────────────────────────────────
    colors = {"GNN": "steelblue", "FD": "tomato", "FD-4": "tomato"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for i, r in enumerate(results_list):
        ev = r["energies"]
        if len(ev) > 0:
            color = next((v for k, v in colors.items() if r["label"].startswith(k)),
                         "gray")
            ax.scatter(range(len(ev)), ev, color=color, label=r["label"],
                       s=30, zorder=3+i, alpha=0.85)
    for el in El_list:
        ax.axhline(el, ls='--', color='gray', lw=0.8, alpha=0.6)
    ax.axhline(El_list[0], ls='--', color='gray', lw=0.8, alpha=0.6,
               label=f"El targets ({ms})")
    ax.set_xlabel("Level index")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(f"Filter eigenvalues — real QD  (nc={nc_true}, ms={ms})")
    ax.legend(fontsize=9)

    ax = axes[1]
    gnn_r = next((r for r in results_list if r["label"] == "GNN"), None)
    fd_r  = next((r for r in results_list if r["label"].startswith("FD")), None)
    if gnn_r is not None and fd_r is not None:
        gnn_ev   = gnn_r["energies"]
        fd_ev    = fd_r["energies"]
        n_common = min(len(gnn_ev), len(fd_ev))
        if n_common > 0:
            de = gnn_ev[:n_common] - fd_ev[:n_common]
            ax.bar(range(n_common), de, color="mediumpurple", alpha=0.8)
            ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel("E_GNN − E_FD (Ha)")
        ax.set_title("GNN correction to eigenvalues")
    elif fd_r is not None:
        fd_ev = fd_r["energies"]
        if len(fd_ev) > 0:
            ax.bar(range(len(fd_ev)), fd_ev, color="tomato", alpha=0.8)
        ax.set_ylabel("Energy (Ha)")
        ax.set_title(f"FD eigenvalues  (GNN skipped: N_qd={N_qd} ≠ N_gnn={N_gnn})")
    ax.set_xlabel("Level index")

    fig.suptitle(
        f"Filter Diagonalization — real QD  (run: {os.path.basename(run_dir)})",
        fontsize=10)
    fig.tight_layout()
    plot_path = out_dir / "filter_test.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {plot_path}")

    return results_list
