"""
main.py
=======
Quantum eigenvalue solver via FFT-based Hamiltonian filter + Rayleigh-Ritz.

Usage:
  python main.py                  # run with CONFIG below
  python main.py --cfg my.json    # override CONFIG from a JSON file

All results (energies, timings, config snapshot) are saved to
  results/<timestamp>_<tag>/res.json
All figures are saved to the same folder.
"""

# ============================================================
# Imports
# ============================================================
import argparse
import copy
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import pyfftw
from scipy.linalg import eigh


# ============================================================
# ✅  CONFIG  ― only change this block to run different setups
# ============================================================
#cccc
CONFIG: Dict[str, Any] = {
    # ---------- output ----------
    "out_root": "results",
    "tag": "gaussian_run",

    # ---------- potential ----------
    "potential": {
        "type": "gaussian_files",
        "d": 0.5,                               # grid spacing for resampling
        "cube_file": "localPot.cube",
        "params_file": "gaussian_fit_params.json",
        "r_cut": 7.0,
    },

    # ---------- grid ----------
    "N": 64,

    # ---------- filter ----------
    # ⚠️  先用小参数跑一次，看启动时打印的 Spectrum check 里的 H_max，再调这两个值
    "nc": 1000,
    "dE": 50.0,             # 根据实际 H_max 调整
    "Vmin": -5.0,           # 根据实际 V_min 调整
    "El_list": list(np.arange(-0.7, -0.6, 0.1).tolist()),   # 根据你关心的能级范围设置

    # ---------- random states ----------
    "n_random": 40,
    "seed": 42,

    # ---------- SVD / Rayleigh-Ritz ----------
    "svd_tol": 1e-3,
    "max_energies": 200,

    # ---------- kinetic cutoff ----------
    "kinetic_cut": 30.0,

    # ---------- misc ----------
    "print_every_filter": 1,
}
CONFIG["dt"] = (CONFIG["nc"] / (CONFIG["dE"] * 2.5)) ** 2
'''
CONFIG: Dict[str, Any] = {
    # ---------- output ----------
    "out_root": "results",
    "tag": "ho3d_test",

    # ---------- potential ----------
    # type = "ho3d"          : pure V = 0.5*omega^2*r^2 (unbounded)
    # type = "harmonic_well" : truncated harmonic well (same as refactored_code)
    # type = "gaussian_blob" : single Gaussian dip
    # type = "gaussian_files": load from .cube + .json
    #
    # ⚠️  Vmin and dE MUST satisfy:
    #       Vmin + dE  >=  V_max(grid) + kinetic_cut   (covers full spectrum)
    #       Vmin       <=  V_min(grid)
    #
    #   harmonic_well (d=0.5, width=4): V in [0,8],  H_max~38  → Vmin=-5, dE=50  ✓
    #   ho3d          (d=0.5, N=20)   : V in [0,34], H_max~64  → Vmin=-1, dE=70
    "potential": {
        "type": "harmonic_well",   # ← switch potential type here

        # ── shared: grid spacing (used by all types) ──
        "d": 0.5,

        # ── ho3d ──
        "omega": 1.0,

        # ── harmonic_well ──
        "center_locations": [[0.0, 0.0, 0.0]],
        "width": 4.0,

        # ── gaussian_blob ──
        "h": -5.0,
        "mu": 0.1,

        # ── gaussian_files ──
        "cube_file": "localPot.cube",
        "params_file": "gaussian_fit_params.json",
        "r_cut": 7.0,
    },

    # ---------- grid ----------
    "N": 20,                # grid points per axis

    # ---------- filter ----------
    # ⚠️  Adjust Vmin/dE when switching potential type (see note above)
    "nc": 200,              # Newton interpolation nodes
    "dE": 50.0,             # energy window width  (harmonic_well default)
    "Vmin": -5.0,           # lower bound          (harmonic_well default)
    # For ho3d use:  "dE": 70.0, "Vmin": -1.0
    "El_list": list(np.arange(0.5, 5.5, 1.0).tolist()),  # filter centres

    # ---------- random states ----------
    "n_random": 50,
    "seed": 42,

    # ---------- SVD / Rayleigh-Ritz ----------
    "svd_tol": 1e-3,
    "max_energies": 200,

    # ---------- kinetic cutoff ----------
    "kinetic_cut": 30.0,

    # ---------- misc ----------
    "print_every_filter": 1,   # print progress every N filter centres
}
# dt is derived, not set manually
CONFIG["dt"] = (CONFIG["nc"] / (CONFIG["dE"] * 2.5)) ** 2

'''
# ============================================================
# JSON helper
# ============================================================
def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python for JSON."""
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj.real)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


# ============================================================
# Data classes
# ============================================================
class IstParams:
    """Discretisation parameters for the Newton interpolation."""
    def __init__(self, nc: int, ms: int):
        self.nc = nc   # number of Newton nodes
        self.ms = ms   # number of filter energy centres


class PhysParams:
    """Physical / scaling parameters."""
    def __init__(self, dE: float, Vmin: float, dt: float):
        self.dE   = dE
        self.Vmin = Vmin
        self.dt   = dt


# ============================================================
# Grid utilities
# ============================================================
def grid_to_vec(y_grid: np.ndarray) -> np.ndarray:
    """(n, Nx, Ny, Nz) → (Nx*Ny*Nz, n)"""
    return y_grid.reshape(y_grid.shape[0], -1).T


def vec_to_grid(y_vec: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """(Nx*Ny*Nz, n) → (n, Nx, Ny, Nz)"""
    return y_vec.T.reshape(y_vec.shape[-1], Nx, Ny, Nz)


def build_grid(N: int, d: float) -> Tuple:
    """
    Uniform grid: spacing *d*, *N* points per axis, centred at 0.
    Returns x, y, z, X, Y, Z, x_grid=[x,y,z].
    """
    L = (N - 1) * d / 2
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    z = np.linspace(-L, L, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return x, y, z, X, Y, Z, [x, y, z]


def build_grid_box(N: int, L: float) -> Tuple:
    """
    Uniform grid: N points per axis, x ∈ [-L/2, L/2).
    Spacing d = L/N  (periodic-box convention).
    Returns x, y, z, X, Y, Z, x_grid=[x,y,z].
    """
    d = L / N
    x = (np.arange(N) - N / 2) * d
    y = x.copy()
    z = x.copy()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return x, y, z, X, Y, Z, [x, y, z]


def build_k_diagonal(x_grid, kinetic_cut: float = 30.0) -> np.ndarray:
    """T(k) = |k|²/2, capped at *kinetic_cut*."""
    x, y, z = x_grid
    d = x[1] - x[0]
    kx = 2 * np.pi * np.fft.fftfreq(len(x), d=d)
    ky = 2 * np.pi * np.fft.fftfreq(len(y), d=d)
    kz = 2 * np.pi * np.fft.fftfreq(len(z), d=d)
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    T_k = (Kx**2 + Ky**2 + Kz**2) / 2
    return np.minimum(T_k, kinetic_cut)


# ============================================================
# Potential builders
# ============================================================
def build_ho3d(X, Y, Z, omega: float = 1.0) -> np.ndarray:
    """V = ½ ω² r²   (3-D harmonic oscillator)."""
    return 0.5 * omega**2 * (X**2 + Y**2 + Z**2)


def build_harmonic_well(X, Y, Z, center_locations: List, width: float) -> np.ndarray:
    """Truncated harmonic well(s)."""
    V = np.zeros(X.shape)
    V_max = 0.5 * width**2
    for center in center_locations:
        c = np.asarray(center)
        r2 = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2
        inside = np.sqrt(r2) < width
        V[inside] += 0.5 * r2[inside]
    V[(V == 0) | (V >= V_max)] = V_max
    return V


def build_gaussian_blob(X, Y, Z, h: float = -5.0, mu: float = 0.1) -> np.ndarray:
    """V = h * exp(-μ r²)."""
    return h * np.exp(-mu * (X**2 + Y**2 + Z**2))


def build_gaussian_files(X, Y, Z, cube_file: str, params_file: str,
                          r_cut: float = 7.0) -> np.ndarray:
    """
    Reconstruct potential from ab-initio Gaussian fit parameters.
    Reads atom positions from *cube_file* and fit coefficients from *params_file*.
    """
    # ---- read Gaussian fit parameters ----
    with open(params_file, 'r') as f:
        params = json.load(f)

    # ---- read atom positions from cube file ----
    atoms, atom_types = _read_cube_atoms(cube_file)

    V = np.zeros(X.shape)
    for atom, atype in zip(atoms, atom_types):
        if atype == 'Unknown':
            continue
        atom_params = params['atoms'][atype]
        amplitudes  = atom_params['amplitudes']
        exponents   = atom_params['exponents']
        pos = atom['position']
        r2  = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
        r   = np.sqrt(r2)
        mask = r <= r_cut
        V_atom = np.zeros_like(r)
        for A, alpha in zip(amplitudes, exponents):
            V_atom[mask] += A * np.exp(-alpha * r2[mask])
        V += V_atom
    return V


def _read_cube_atoms(filename: str) -> Tuple[List[Dict], List[str]]:
    """Parse atom positions and types from a Gaussian cube file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    idx = 2
    parts    = lines[idx].split()
    n_atoms  = int(parts[0])
    idx += 1 + 3   # skip origin + 3 grid-vector lines

    atoms      = []
    atom_types = []
    _atomic_number_to_type = {33: 'As', 49: 'In'}

    for _ in range(n_atoms):
        parts = lines[idx].split()
        atomic_number = int(parts[0])
        charge        = float(parts[1])
        position      = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
        atoms.append({'atomic_number': atomic_number,
                      'charge': charge,
                      'position': position})
        if atomic_number in _atomic_number_to_type:
            atom_types.append(_atomic_number_to_type[atomic_number])
        elif atomic_number == 1:
            atom_types.append('P1' if charge > 0 else 'P2')
        else:
            atom_types.append('Unknown')
        idx += 1

    return atoms, atom_types


def build_potential_from_config(cfg: Dict, N: int) -> Tuple[np.ndarray, ...]:
    """
    Dispatch to the correct potential builder based on cfg['potential']['type'].

    Returns x, y, z, X, Y, Z, x_grid, V
    """
    pot  = cfg["potential"]
    ptype = pot["type"]
    kinetic_cut = cfg.get("kinetic_cut", 30.0)

    if ptype == "ho3d":
        x, y, z, X, Y, Z, x_grid = build_grid(N, pot["d"])
        V = build_ho3d(X, Y, Z, omega=pot["omega"])

    elif ptype == "harmonic_well":
        x, y, z, X, Y, Z, x_grid = build_grid(N, pot["d"])
        V = build_harmonic_well(X, Y, Z,
                                center_locations=pot["center_locations"],
                                width=pot["width"])

    elif ptype == "gaussian_blob":
        x, y, z, X, Y, Z, x_grid = build_grid(N, pot["d"])
        V = build_gaussian_blob(X, Y, Z, h=pot["h"], mu=pot["mu"])

    elif ptype == "gaussian_files":
        # Grid is read from cube file dimensions; we re-sample to N
        from gaussian_potential_builder import GaussianPotentialBuilder
        builder = GaussianPotentialBuilder(
            cube_file=pot["cube_file"],
            params_file=pot["params_file"],
            r_cut=pot["r_cut"],
        )
        x_raw, y_raw, z_raw, V = builder.build_potential(N)
        x = x_raw; y = y_raw; z = z_raw
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        x_grid  = [x, y, z]

    else:
        raise ValueError(f"Unknown potential type: '{ptype}'")

    return x, y, z, X, Y, Z, x_grid, V


# ============================================================
# Wavefunction utilities
# ============================================================
def normalize_psi(psi: np.ndarray) -> np.ndarray:
    return psi / np.sqrt(np.sum(np.abs(psi)**2))


def random_sine_psi(X, Y, Z, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Random sinusoidal wavefunction."""
    if rng is not None:
        vals = rng.uniform(-1.0, 1.0, 4)
        kx_val, ky_val, kz_val, b_val = vals
    else:
        kx_val = random.uniform(-1.0, 1.0)
        ky_val = random.uniform(-1.0, 1.0)
        kz_val = random.uniform(-1.0, 1.0)
        b_val  = random.uniform(-1.0, 1.0)
    psi = np.sin(kx_val * X + ky_val * Y + kz_val * Z + b_val) + 0.0j
    return normalize_psi(psi)


# ============================================================
# Hamiltonian (FFT-based)
# ============================================================
def eval_kinetic(psi: np.ndarray, T_k_diagonal: np.ndarray) -> np.ndarray:
    """T̂|ψ⟩ via pyfftw 3-D FFT."""
    shape    = psi.shape
    fft_in   = pyfftw.empty_aligned(shape, dtype='complex128')
    fft_out  = pyfftw.empty_aligned(shape, dtype='complex128')
    ifft_out = pyfftw.empty_aligned(shape, dtype='complex128')
    fft1  = pyfftw.FFTW(fft_in,  fft_out,  axes=(0, 1, 2),
                         direction='FFTW_FORWARD',  flags=('FFTW_MEASURE',))
    ifft1 = pyfftw.FFTW(fft_out, ifft_out, axes=(0, 1, 2),
                         direction='FFTW_BACKWARD', flags=('FFTW_MEASURE',))
    fft_in[:] = psi
    fft1()
    fft_out[:] = T_k_diagonal * fft_out
    ifft1()
    return ifft_out.copy()


def apply_H(psi: np.ndarray, V: np.ndarray,
            T_k_diagonal: np.ndarray) -> np.ndarray:
    """H|ψ⟩ = T̂|ψ⟩ + V|ψ⟩."""
    return eval_kinetic(psi, T_k_diagonal) + V * psi


def apply_filter_H(psi: np.ndarray, V: np.ndarray,
                   nodes: np.ndarray, coeffs: np.ndarray,
                   par: PhysParams,
                   T_k_diagonal: np.ndarray) -> np.ndarray:
    """
    f(H)|ψ⟩  via Newton polynomial in scaled Hamiltonian.
    Scaled variable:  x̃ = 4(H - Vmin)/dE - 2
    """
    n        = len(nodes)
    psi_prev = psi.copy()
    result   = coeffs[0] * psi

    for j in range(1, n):
        T_psi    = eval_kinetic(psi_prev, T_k_diagonal)
        H_psi    = T_psi + V * psi_prev
        psi_curr = ((4.0 / par.dE) * (H_psi - par.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        result  += coeffs[j] * psi_curr
        psi_prev = psi_curr

    return result


# ============================================================
# Newton interpolation for filter coefficients
# ============================================================
def _filt_func_values(x: np.ndarray, dt: float) -> np.ndarray:
    return np.sqrt(dt / np.pi) * np.exp(-x**2 * dt)


def _samp_points_ashkenazy(min_val: float, max_val: float, nc: int) -> np.ndarray:
    nc3  = 32 * nc
    samp = (min_val + np.arange(nc3) * (max_val - min_val) / (nc3 - 1)).astype(complex)
    point = np.zeros(nc, dtype=complex)
    point[0] = samp[0]
    dv = (samp.real - point[0].real)**2 + (samp.imag - point[0].imag)**2
    veca = np.where(dv < 1e-10, -1e30, np.log(dv))
    for j in range(1, nc):
        kmax     = np.argmax(veca)
        point[j] = samp[kmax]
        dv   = (samp.real - point[j].real)**2 + (samp.imag - point[j].imag)**2
        veca = np.where(dv < 1e-10, -1e30, veca + np.log(dv))
    return point.real


def _newton_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    f = np.zeros((n, n))
    f[:, 0] = y
    for j in range(1, n):
        for i in range(j, n):
            f[i, j] = (f[i, j - 1] - f[i - 1, j - 1]) / (x[i] - x[i - j])
    return f[np.arange(n), np.arange(n)]


def build_filter_coefficients(El_list: np.ndarray, par: PhysParams,
                               nc: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Newton interpolation coefficients for each filter centre in El_list.

    Returns
    -------
    an   : (ms, nc) coefficient array
    samp : (nc,)    Ashkenazy nodes in [-2, 2]
    """
    Smin, Smax = -2.0, 2.0
    scale  = (Smax - Smin) / par.dE
    samp   = _samp_points_ashkenazy(Smin, Smax, nc)
    x_phys = (samp + 2.0) / scale + par.Vmin

    an = np.zeros((len(El_list), nc))
    for ie, El in enumerate(El_list):
        y = _filt_func_values(x_phys - El, par.dt)
        an[ie] = _newton_coefficients(samp, y)
    return an, samp


# ============================================================
# SVD / Rayleigh-Ritz diagonalisation
# ============================================================
def svd_rayleigh_ritz(filtered_psi_matrix: np.ndarray,
                      x_grid, V: np.ndarray,
                      Nx: int, Ny: int, Nz: int,
                      T_k_diagonal: np.ndarray,
                      svd_tol: float = 1e-3,
                      max_energies: int = 200) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Rayleigh-Ritz diagonalisation in the filtered subspace.

    Returns energies, Ur, rank r.
    """
    C_f = grid_to_vec(filtered_psi_matrix)
    C_f = C_f / np.linalg.norm(C_f, axis=0)
    C_f = C_f[:, ~np.any(np.isinf(C_f), axis=0)]

    print("  QR decomposition ...")
    Q, R = np.linalg.qr(C_f, mode='reduced')

    print("  SVD on R ...")
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    U = Q @ U1

    r = int(np.sum(sigma > svd_tol))
    print(f"  Selected rank r = {r}  (tol = {svd_tol})")
    Ur = U[:, :r]

    Ur_grid  = vec_to_grid(Ur, Nx, Ny, Nz)
    HUr_grid = np.zeros_like(Ur_grid, dtype='complex128')
    for i in range(r):
        HUr_grid[i] = apply_H(Ur_grid[i], V, T_k_diagonal)

    H_tilde       = Ur.T.conj() @ grid_to_vec(HUr_grid)
    energies, _   = eigh(H_tilde)
    energies      = np.sort(energies.real)[:max_energies]
    return energies, Ur, r


# ============================================================
# Plotting  (all functions save to file; show() is never called)
# ============================================================
def _savefig(fig, path: Path, close: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved: {path}")
    if close:
        plt.close(fig)


def plot_filter_interpolation(El_list, an, samp, par, interval,
                               out_dir: Path) -> None:
    """True filter vs Newton interpolant."""
    def _scaled_eval(x_eval, nodes, coeffs):
        x = 4.0 * (x_eval - par.Vmin) / par.dE - 2.0
        result = coeffs[0]
        basis  = 1.0
        for j in range(1, len(nodes)):
            basis  *= (x - nodes[j - 1])
            result += coeffs[j] * basis
        return result

    x_plot = np.linspace(interval[0], interval[1], 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    for ie, El in enumerate(El_list):
        y_true   = _filt_func_values(x_plot - El, par.dt)
        y_interp = np.array([_scaled_eval(x, samp, an[ie]) for x in x_plot])
        mae = np.mean(np.abs(y_true - y_interp))
        print(f"  El={El:.2f}  MAE={mae:.6e}")
        ax.plot(x_plot, y_true,   color='blue',
                label='True filter' if ie == 0 else "")
        ax.plot(x_plot, y_interp, '--', color='red',
                label=f'Newton interp (nc={len(samp)})' if ie == 0 else "")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(f'Filter Function vs Newton Interpolation (nc={len(samp)})')
    ax.set_xlim(interval[0], interval[1])
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _savefig(fig, out_dir / "filter_interpolation.png")


def plot_filtered_energies(El_list, E_mean_all, E_std_all, N_values,
                            error_mean_all, n_random: int,
                            out_dir: Path) -> None:
    """Errorbar: filtered energy expectation vs filter centre El."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, N in enumerate(N_values):
        ax.errorbar(El_list, E_mean_all[i], yerr=E_std_all[i],
                    fmt='o-', capsize=5,
                    label=f'N={N}, MeanErr={error_mean_all[i]:.4f}')
    ax.set_xlabel('El')
    ax.set_ylabel('E_filtered')
    ax.set_title(f'Filtered Energy by FFT (Averaged over {n_random} Random States)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _savefig(fig, out_dir / "filtered_energies.png")


def plot_energy_levels(energies_filtered: np.ndarray,
                       exact_energies: np.ndarray,
                       out_dir: Path) -> None:
    """Side-by-side horizontal energy-level diagram."""
    energies_filtered = np.sort(energies_filtered)
    fig, ax = plt.subplots(figsize=(5, 6))
    for i, e in enumerate(energies_filtered):
        ax.hlines(e, 0, 0.45, color='blue', linestyle='-',
                  label='FD Energies' if i == 0 else "")
    for i, e in enumerate(exact_energies):
        ax.hlines(e, 0.55, 1.0, color='red', linestyle='-',
                  label='Exact Energies' if i == 0 else "")
    ax.set_xlabel('Category')
    ax.set_ylabel('Energy')
    ax.set_xticks([0.225, 0.775])
    ax.set_xticklabels(['FD Energies', 'Exact Energies'])
    ax.grid(True, axis='y')
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir / "energy_levels.png")


def plot_energy_errors(energies_dict: Dict[str, np.ndarray],
                       exact_energies: np.ndarray,
                       out_dir: Path) -> None:
    """Log-scale scatter: |E_FD - E_exact| vs nearest exact level."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(energies_dict)))
    for (label, energies), color in zip(energies_dict.items(), colors):
        energies = np.sort(energies)
        nearest  = exact_energies[
            np.argmin(np.abs(energies[:, None] - exact_energies[None, :]), axis=1)]
        errors   = np.abs(energies - nearest)
        ax.scatter(nearest, errors, color=color, alpha=0.5, s=30, label=label)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_yscale('log')
    ax.set_xlabel('Nearest Exact Energy Level')
    ax.set_ylabel('|FD - Exact|')
    ax.set_title('Energy Error per FD Level')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_dir / "energy_errors.png")


def plot_potential_slice(V: np.ndarray, x: np.ndarray, z: np.ndarray,
                          out_dir: Path) -> None:
    """2-D slice of potential at y = 0 (midplane)."""
    Ny  = V.shape[1]
    mid = Ny // 2
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(x, z, V[:, mid, :].T, shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='V (a.u.)')
    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('z (a.u.)')
    ax.set_title('Potential slice at y = 0')
    fig.tight_layout()
    _savefig(fig, out_dir / "potential_slice.png")


# ============================================================
# Main
# ============================================================
def run(cfg: Dict[str, Any]) -> None:
    # ---- output directory ----
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = cfg.get("tag", "")
    out_dir = Path(cfg["out_root"]) / (stamp + (f"_{tag}" if tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n")

    # ---- seeding ----
    seed = cfg.get("seed", 0)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    # ---- timers ----
    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    # ================================================================
    # 1. Build potential
    # ================================================================
    print("=" * 60)
    print("1. Building potential ...")
    t0 = time.perf_counter()

    N   = cfg["N"]
    x, y, z, X, Y, Z, x_grid, V = build_potential_from_config(cfg, N)
    Nx = Ny = Nz = N

    timings["build_potential"] = time.perf_counter() - t0
    print(f"   V shape: {V.shape},  min={V.min():.4f},  max={V.max():.4f}")
    print(f"   Grid spacing: {x[1]-x[0]:.4f} a.u.")
    print(f"   Time: {timings['build_potential']:.3f} s")

    # ---- sanity check: filter window must cover the full spectrum ----
    H_max_est = float(V.max()) + cfg.get("kinetic_cut", 30.0)
    H_min_est = float(V.min())
    win_lo    = cfg["Vmin"]
    win_hi    = cfg["Vmin"] + cfg["dE"]
    if win_lo > H_min_est or win_hi < H_max_est:
        print(f"\n  ⚠️  WARNING: filter window [{win_lo}, {win_hi}] does NOT cover "
              f"estimated spectrum [{H_min_est:.2f}, {H_max_est:.2f}]!")
        print(f"     Suggested fix: Vmin <= {H_min_est:.2f}, "
              f"dE >= {H_max_est - H_min_est:.2f}\n")
    else:
        print(f"   Spectrum check OK: [{H_min_est:.2f}, {H_max_est:.2f}] "
              f"⊆ window [{win_lo}, {win_hi}]")

    # potential slice plot
    plot_potential_slice(V, x, z, out_dir)

    # ================================================================
    # 2. Build kinetic diagonal
    # ================================================================
    kinetic_cut  = cfg.get("kinetic_cut", 30.0)
    T_k_diagonal = build_k_diagonal(x_grid, kinetic_cut=kinetic_cut)

    # ================================================================
    # 3. Build filter coefficients
    # ================================================================
    print("\n2. Building filter coefficients ...")
    t0 = time.perf_counter()

    El_list = np.array(cfg["El_list"], dtype=float)
    nc      = cfg["nc"]
    dt      = cfg["dt"]
    par     = PhysParams(dE=cfg["dE"], Vmin=cfg["Vmin"], dt=dt)
    ist     = IstParams(nc=nc, ms=len(El_list))

    print(f"   nc={nc},  dE={par.dE},  Vmin={par.Vmin},  dt={dt:.4f}")
    print(f"   sigma = {1/np.sqrt(2*dt):.4f}")
    print(f"   Number of filter centres: {len(El_list)}")

    an, samp = build_filter_coefficients(El_list, par, nc)

    timings["build_filter"] = time.perf_counter() - t0
    print(f"   Time: {timings['build_filter']:.3f} s")

    # filter interpolation plot
    interval = (float(El_list[0]) - 0.5, float(El_list[-1]) + 0.5)
    interval = (-4.5,3.0)
    if interval[0] < interval[1]:
        plot_filter_interpolation(El_list, an, samp, par, interval, out_dir)
        #print('\nplot_filter_interpolation\n')

    # ================================================================
    # 4. Filter random states
    # ================================================================
    print("\n3. Filtering random states ...")
    t0 = time.perf_counter()

    n_random           = cfg["n_random"]
    filtered_psi_matrix = np.zeros(
        (ist.ms * n_random, Nx, Ny, Nz), dtype='complex128')
    E_mean = []
    E_std  = []
    print_every = cfg.get("print_every_filter", 1)

    for ie in range(ist.ms):
        E_temp = []
        for i in range(n_random):
            psi_rand = random_sine_psi(X, Y, Z, rng=rng)
            psi_filt = apply_filter_H(psi_rand, V, samp, an[ie], par, T_k_diagonal)
            psi_filt = normalize_psi(psi_filt)

            H_psi  = apply_H(psi_filt, V, T_k_diagonal)
            E_exp  = float(np.sum(psi_filt.conj() * H_psi).real)
            E_temp.append(E_exp)
            filtered_psi_matrix[ie * n_random + i] = psi_filt

        mean_e = float(np.mean(E_temp))
        std_e  = float(np.std(E_temp))
        E_mean.append(mean_e)
        E_std.append(std_e)

        if ie % print_every == 0:
            print(f"   El={El_list[ie]:.2f}  mean={mean_e:.4f}  std={std_e:.4f}")

    timings["filter_states"] = time.perf_counter() - t0
    error_mean = float(np.mean(np.abs(np.array(E_mean) - El_list)))
    print(f"   Mean error vs El: {error_mean:.6f}")
    print(f"   Time: {timings['filter_states']:.3f} s")

    plot_filtered_energies(El_list, [E_mean], [E_std], [N], [error_mean],
                            n_random, out_dir)

    # ================================================================
    # 5. Rayleigh-Ritz diagonalisation
    # ================================================================
    print("\n4. Rayleigh-Ritz diagonalisation ...")
    t0 = time.perf_counter()

    energies, Ur, rank = svd_rayleigh_ritz(
        filtered_psi_matrix, x_grid, V, Nx, Ny, Nz,
        T_k_diagonal=T_k_diagonal,
        svd_tol=cfg.get("svd_tol", 1e-3),
        max_energies=cfg.get("max_energies", 200),
    )

    timings["rayleigh_ritz"] = time.perf_counter() - t0
    timings["total"]         = time.perf_counter() - t_total_start

    print(f"   Rank r = {rank}")
    print(f"   First 10 energies: {np.round(energies[:10], 4).tolist()}")
    print(f"   Time: {timings['rayleigh_ritz']:.3f} s")
    print(f"\n   Total wall time: {timings['total']:.3f} s")

    # ================================================================
    # 6. Plots for eigenvalues
    # ================================================================
    # exact energies for 3-D HO (only meaningful for ho3d potential)
    exact_energies = np.arange(1.5, 18.5, 1.0)

    plot_energy_levels(energies, exact_energies, out_dir)
    plot_energy_errors({f"N={N}": energies}, exact_energies, out_dir)

    # ================================================================
    # 7. Save results to JSON
    # ================================================================
    print("\n5. Saving results ...")
    results = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "config":           cfg,
        "timings_seconds":  timings,
        "grid": {
            "N":      N,
            "Nx":     Nx, "Ny": Ny, "Nz": Nz,
            "x_min":  float(x[0]), "x_max": float(x[-1]),
            "dx":     float(x[1] - x[0]),
        },
        "potential": {
            "V_min":  float(V.min()),
            "V_max":  float(V.max()),
            "V_mean": float(V.mean()),
        },
        "filter": {
            "nc":     nc,
            "dt":     dt,
            "sigma":  float(1 / np.sqrt(2 * dt)),
            "El_list": El_list.tolist(),
            "E_mean": E_mean,
            "E_std":  E_std,
            "error_mean_vs_El": error_mean,
        },
        "rayleigh_ritz": {
            "rank":      rank,
            "svd_tol":   cfg.get("svd_tol", 1e-3),
            "n_energies": len(energies),
            "energies":  energies.tolist(),
        },
    }
    save_json(results, out_dir / "res.json")
    print(f"   Saved: {out_dir / 'res.json'}")
    print(f"\nAll outputs in: {out_dir}")


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FFT filter-diagonalise solver")
    parser.add_argument("--cfg", type=str, default=None,
                        help="Path to a JSON file that overrides CONFIG")
    args = parser.parse_args()

    cfg = copy.deepcopy(CONFIG)

    if args.cfg:
        with open(args.cfg, "r") as f:
            override = json.load(f)
        # deep-merge: top-level keys override, nested dicts are merged
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        # recompute derived dt if nc/dE were overridden
        cfg["dt"] = (cfg["nc"] / (cfg["dE"] * 2.5)) ** 2

    run(cfg)


if __name__ == "__main__":
    main()