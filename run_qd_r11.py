#!/usr/bin/env python3
"""run_qd_r11.py
Full pipeline for InAs QD R=11 Bohr filter diagonalisation with Julia.

Stages
------
1  Build QD_Outputs/QD_R11.cube          – atom geometry (skip if exists)
2  Space partition  → <vexpr_dir>/        – V_expr pkl per cube
3  H^n expressions  → <expr_dir>/         – H_power_n.pkl per cube subdir
4  Julia eval scripts                     – eval_filter_m{m}_*.jl per cube
5  Filter diagonalisation                 – evaluate f(H)*psi, SVD, eigenvalues

Usage
-----
    python run_qd_r11.py                          # all stages, default params
    python run_qd_r11.py --stages 1,2             # only build cube + partition
    python run_qd_r11.py --stages 3,4,5 --m 8    # H^n + Julia + filter diag
    python run_qd_r11.py --stages 5 --n_waves 120 --julia_exe /usr/bin/julia
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.spatial import KDTree

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from symbolic_code.space_partition import CubicSpacePartition, read_cube_atoms
from symbolic_code.expr_loader import CubicExpressionManager
from symbolic_code.h_powers import generate_H_powers, process_all_cubes as hpow_process_all
from symbolic_code.chebyshev_filter import (
    apply_f_of_H_from_raw_powers,
    extract_cos_sin_coeffs,
    group_by_exp_combined,
    apply_horner,
    svd_H,
)
from symbolic_code.julia_codegen import build_julia_batch_script

# ---------------------------------------------------------------------------
# Fixed QD / grid parameters
# ---------------------------------------------------------------------------
QD_RADIUS   = 11.0      # Bohr
BOX_HALF    = 16.0      # L_s = radius + 5 buffer (Bohr)
D_GRID      = 0.625     # grid spacing (Bohr)
N_DIVISIONS = 8         # cubes per axis  → cube side = 2*16/8 = 4 Bohr
R_CUT       = 5.0       # atom contribution cutoff per cube (Bohr)

# Lattice parameters for zincblende InAs
_A_INAS    = 11.4523    # InAs lattice constant (Bohr)
_IN_H_BOND = 2.7291     # In-H passivation bond length (Bohr)
_AS_H_BOND = 1.2652     # As-H passivation bond length (Bohr)


# ===========================================================================
# Stage 1 – build QD cube file
# ===========================================================================

def _get_ideal_neighbors(a):
    v = a / 4.0
    return np.array([[v, v, v], [v, -v, -v], [-v, v, -v], [-v, -v, v]])


def build_qd_cube(radius_bohr, output_path, d=0.625):
    """Build InAs QD + H passivation, write Gaussian cube file.

    Returns (in_pos, as_pos, h_pos, N_grid, origin).
    Skips writing if *output_path* already exists.
    """
    output_path = Path(output_path)
    if output_path.exists():
        print(f"  Stage 1: cube exists, skipping → {output_path}")
        return None

    a = _A_INAS
    n_cells = int(np.ceil(radius_bohr / a)) + 1
    base_in = np.array([[0, 0, 0], [0.5, 0.5, 0],
                        [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    base_as = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                        [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]) * a

    in_pos, as_pos = [], []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                off = np.array([i, j, k]) * a
                for p in base_in:
                    if np.linalg.norm(p + off) <= radius_bohr:
                        in_pos.append(p + off)
                for p in base_as:
                    if np.linalg.norm(p + off) <= radius_bohr:
                        as_pos.append(p + off)

    in_pos = np.array(in_pos) if in_pos else np.empty((0, 3))
    as_pos = np.array(as_pos) if as_pos else np.empty((0, 3))
    all_pos = np.vstack([in_pos, as_pos]) if len(in_pos) and len(as_pos) else np.empty((0, 3))

    h_pos = []
    if len(all_pos):
        tree = KDTree(all_pos)
        dirs_in  =  _get_ideal_neighbors(a)
        dirs_as  = -_get_ideal_neighbors(a)
        for pos in in_pos:
            for dv in dirs_in:
                if tree.query(pos + dv)[0] > 0.1 * a:
                    h_pos.append(pos + dv / np.linalg.norm(dv) * _IN_H_BOND)
        for pos in as_pos:
            for dv in dirs_as:
                if tree.query(pos + dv)[0] > 0.1 * a:
                    h_pos.append(pos + dv / np.linalg.norm(dv) * _AS_H_BOND)

    h_pos    = np.array(h_pos) if h_pos else np.empty((0, 3))
    box_half = radius_bohr + 5.0
    origin   = -box_half
    N        = int(np.ceil(2 * box_half / d))
    natoms   = len(in_pos) + len(as_pos) + len(h_pos)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"Generated InAs Quantum Dot (Radius={radius_bohr})\n")
        f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        f.write(f"{natoms:5d} {origin:12.6f} {origin:12.6f} {origin:12.6f}\n")
        f.write(f"{N:5d} {d:12.6f}     0.000000     0.000000\n")
        f.write(f"{N:5d}     0.000000 {d:12.6f}     0.000000\n")
        f.write(f"{N:5d}     0.000000     0.000000 {d:12.6f}\n")
        for pos in in_pos:
            f.write(f"   49     0.000000 {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        for pos in as_pos:
            f.write(f"   33     0.000000 {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        for pos in h_pos:
            f.write(f"    1     0.000000 {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        # zero-filled volumetric data (not used; atoms are what matter)
        zero_str = "  0.00000E+00"
        total = N ** 3
        full, rem = divmod(total, 6)
        for _ in range(full):
            f.write(zero_str * 6 + "\n")
        if rem:
            f.write(zero_str * rem + "\n")

    print(f"  Stage 1: wrote {output_path}  "
          f"({len(in_pos)} In + {len(as_pos)} As + {len(h_pos)} H = {natoms} atoms, "
          f"grid {N}^3)")
    return in_pos, as_pos, h_pos, N, origin


# ===========================================================================
# Stage 2 – space partition
# ===========================================================================

def run_space_partition(cube_file, params_file, vexpr_dir,
                        L_s=BOX_HALF, n_divisions=N_DIVISIONS, r_cut=R_CUT,
                        partition_mode="uniform", cell_edge=None, anchor_atom=True):
    """Build per-cube V_expr pkl files into *vexpr_dir*.

    Skip if the directory already contains .pkl files.
    """
    vexpr_dir = Path(vexpr_dir)
    existing = list(vexpr_dir.glob('*.pkl')) if vexpr_dir.exists() else []
    if existing:
        print(f"  Stage 2: {len(existing)} pkl files exist in {vexpr_dir}, skipping")
        return

    atoms = read_cube_atoms(str(cube_file))
    print(f"  Stage 2: read {len(atoms)} atoms from {cube_file}")

    if partition_mode == "cell":
        if cell_edge is None:
            cell_edge = _A_INAS / 2.0
        anchor = None
        if anchor_atom and atoms:
            arr = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
            anchor = arr[np.argmin(arr.sum(axis=1))]
            print(f"  Stage 2: cell partition anchor atom at ({anchor[0]:.3f}, {anchor[1]:.3f}, {anchor[2]:.3f})")
        print(f"  Stage 2: using crystal-cell partition, cube edge={cell_edge:.6f} Bohr")
        partition = CubicSpacePartition(
            params_file=str(params_file), atoms=atoms, L_s=L_s,
            n_divisions=n_divisions, r_cut=r_cut, cube_size=cell_edge, anchor_corner=anchor,
        )
    else:
        partition = CubicSpacePartition(
            params_file=str(params_file),
            atoms=atoms,
            L_s=L_s,
            n_divisions=n_divisions,
            r_cut=r_cut,
        )
    n_with = partition.process_all_cubes(str(vexpr_dir))
    summary_path = Path(vexpr_dir) / "partition_summary.txt"
    summary_path.write_text(
        f"mode={partition_mode}\nL_s={L_s}\nr_cut={r_cut}\ncube_edge={partition.l0}\ntotal_cubes={len(partition.cube_centers)}\nnon_empty={n_with}\n",
        encoding="utf-8",
    )
    print(f"  Stage 2: generated {n_with} non-empty cube expressions → {vexpr_dir}")
    print(f"  Stage 2: summary saved → {summary_path}")


# ===========================================================================
# Stage 3 – H^n expressions
# ===========================================================================

def run_h_powers(vexpr_dir, expr_dir, m_max, file_format='pkl', expand=False):
    """Generate H^n pkl files for every cube in *vexpr_dir*.

    expand=False (default) preserves the product-tree structure of each H^n
    expression — smaller pkl files, faster generation, and the trees carry
    factored sub-expressions that sp.cse() can exploit when building Julia code.
    The assembly step in Stage 4 still calls sp.expand() to flatten before
    group_by_exp_combined, so the final Julia code is identical either way.
    """
    print(f"\n  Stage 3: generating H^0…{m_max} for each cube "
          f"(expand={expand}) → {expr_dir}")
    manager = CubicExpressionManager(str(vexpr_dir))
    if not manager.expressions:
        raise RuntimeError(f"No cube expressions found in {vexpr_dir}")
    hpow_process_all(
        manager=manager,
        N=m_max,
        base_outdir=str(expr_dir),
        file_format=file_format,
        method='raw',
        expand=expand,
    )


# ===========================================================================
# Stage 4 – Julia eval scripts
# ===========================================================================

def _cube_dirname(cx, cy, cz):
    """Must match the naming convention in h_powers.py."""
    def fmt(v):
        return f"{v:.3f}".replace("-", "neg")
    return f"cube_{fmt(cx)}_{fmt(cy)}_{fmt(cz)}"


def run_julia_codegen(vexpr_dir, expr_dir, m, E_lo, E_hi, expand=False):
    """Build and cache eval_filter_m{m}_*.jl in every cube subdir of expr_dir.

    When expand=False (default), the H^n pkl files were generated without
    sp.expand().  Assembly uses return_envelopes=True to avoid the sin/cos
    wrapping, then sp.expand() is called on the envelopes directly — this is
    faster than wrapping+extracting and yields identical Julia code.
    """
    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)
    a_key = f"{a:.8g}".replace('.', 'p').replace('-', 'm')
    b_key = f"{b:.8g}".replace('.', 'p').replace('-', 'm')
    jl_name = f"eval_filter_m{m}_{a_key}_{b_key}.jl"

    print(f"\n  Stage 4: Julia codegen for m={m}, E_lo={E_lo}, E_hi={E_hi:.4f} "
          f"(expand={expand})")
    print(f"    a={a:.6f}  b={b:.6f}  → {jl_name}")

    expr_dir  = Path(expr_dir)
    vexpr_dir = Path(vexpr_dir)

    # Load cube centers from vexpr pkl files (to get the cube dirs)
    manager   = CubicExpressionManager(str(vexpr_dir))
    done = skip = 0

    for idx in sorted(manager.cube_info.keys()):
        info = manager.cube_info[idx]
        cx, cy, cz = info['center']
        cube_dir = expr_dir / _cube_dirname(cx, cy, cz)

        if not cube_dir.exists():
            print(f"    Cube {idx}: H^n dir missing ({cube_dir.name}) – skipped")
            skip += 1
            continue

        jl_path = cube_dir / jl_name
        if jl_path.exists():
            done += 1
            continue

        # Assemble f(H)*psi and build Julia script
        try:
            t0 = time.time()
            if not expand:
                # Faster path: get envelopes directly (no sin/cos wrapping),
                # then expand them here so group_by_exp_combined can decompose
                # the Gaussian factors correctly.
                expr_cos_raw, expr_sin_raw = apply_f_of_H_from_raw_powers(
                    str(cube_dir), m, a=a, b=b,
                    file_type='pkl', return_envelopes=True)
                expr_cos = sp.expand(expr_cos_raw)
                expr_sin = sp.expand(expr_sin_raw)
            else:
                psi_fH = apply_f_of_H_from_raw_powers(str(cube_dir), m, a=a, b=b,
                                                       file_type='pkl')
                expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)
            terms_cos = apply_horner(group_by_exp_combined(expr_cos))
            terms_sin = apply_horner(group_by_exp_combined(expr_sin))
            jl_src = build_julia_batch_script(terms_cos, terms_sin)
            jl_path.write_text(jl_src, encoding='utf-8')
            dt = time.time() - t0
            print(f"    [{done+skip+1}] Cube {idx} ({cube_dir.name}): "
                  f"built in {dt:.1f}s  cos_groups={len(terms_cos)}  "
                  f"sin_groups={len(terms_sin)}")
            done += 1
        except Exception as exc:
            print(f"    [{done+skip+1}] Cube {idx}: ERROR – {exc}")
            skip += 1

    print(f"  Stage 4: done={done}  skipped/error={skip}")
    return a, b, jl_name


# ===========================================================================
# Stage 5 – filter diagonalisation
# ===========================================================================

def _build_grid(L_s=BOX_HALF, d=D_GRID):
    """Return (x1, X, Y, Z) for a uniform grid with spacing d.

    Uses ceil so Ng matches generate_QD_cubes.py / build_qd_cube convention.
    """
    N = int(np.ceil(2 * L_s / d))
    x1 = np.linspace(-L_s, L_s, N, endpoint=False)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
    return x1, X, Y, Z


def _build_V_grid(atoms, params_data, X, Y, Z):
    """Evaluate full Gaussian potential V(x,y,z) on the grid numerically.

    V = sum_atoms sum_gaussians A_i * exp(-a_i * r^2)
    """
    ap = params_data['atoms']
    V  = np.zeros_like(X)
    for atom in atoms:
        atype = atom['type']
        if atype not in ap:
            continue
        p   = ap[atype]
        dx  = X - atom['x']
        dy  = Y - atom['y']
        dz  = Z - atom['z']
        r2  = dx**2 + dy**2 + dz**2
        for A, a in zip(p['amplitudes'], p['exponents']):
            V += A * np.exp(-a * r2)
    return V


def _fft_apply_H(psi3d, x1, V3d):
    """Apply H = -0.5∇² + V via FFT (periodic BC)."""
    N  = len(x1)
    dx = x1[1] - x1[0]
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    Kx, Ky, Kz = np.meshgrid(k1, k1, k1, indexing='ij')
    Hkin = np.fft.ifftn(
        0.5 * (Kx**2 + Ky**2 + Kz**2) * np.fft.fftn(psi3d)
    ).real
    return Hkin + V3d * psi3d


def _run_julia_batch(jl_file, Xf, Yf, Zf, k_vals, b_vals, work_dir, julia_exe='julia'):
    """Run a pre-built Julia batch script on the given (flat) grid coordinates.

    Parameters
    ----------
    jl_file  : Path   .jl script built by build_julia_batch_script
    Xf, Yf, Zf : 1-D arrays  flat grid coordinates (length N_pts)
    k_vals   : (n_waves, 3)
    b_vals   : (n_waves,)
    work_dir : Path   directory for temporary binary I/O files

    Returns
    -------
    out      : ndarray shape (n_waves, N_pts)
    timing   : dict  {warmup_s, eval_s, n_eval_waves, total_wall_s}
    """
    work_dir  = Path(work_dir)
    n_waves   = len(k_vals)
    N_pts     = len(Xf)
    grid_bin  = work_dir / '_grid.bin'
    kvals_bin = work_dir / '_kvals.bin'
    out_bin   = work_dir / '_cfilt.bin'

    with open(grid_bin, 'wb') as f:
        Xf.astype('<f8').tofile(f)
        Yf.astype('<f8').tofile(f)
        Zf.astype('<f8').tofile(f)

    kb = np.column_stack([k_vals, b_vals]).astype('<f8')
    kb.ravel().tofile(str(kvals_bin))

    cmd = [julia_exe, str(jl_file),
           str(grid_bin), str(kvals_bin), str(out_bin),
           str(N_pts), str(n_waves)]
    t0   = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    if proc.returncode != 0:
        raise RuntimeError(
            f"Julia failed (code {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    timing = {'warmup_s': None, 'eval_s': None, 'n_eval_waves': None,
              'total_wall_s': wall}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                timing.update(json.loads(line))
            except json.JSONDecodeError:
                pass
            break

    raw = np.fromfile(str(out_bin), dtype='<f8')
    out = raw.reshape(n_waves, N_pts)

    grid_bin.unlink(missing_ok=True)
    kvals_bin.unlink(missing_ok=True)
    out_bin.unlink(missing_ok=True)

    return out, timing


def _estimate_E_hi(L_s, N_grid, pref=0.5, V_peak=12.0):
    """Rough E_hi from grid Nyquist kinetic + potential peak.

    V_peak is an approximate upper bound on |V| at an atomic core
    (As peak from Gaussian fit: ~10 Hartree).
    """
    dx    = 2.0 * L_s / N_grid
    k_max = np.pi / dx
    T_max = pref * 3.0 * k_max ** 2
    return T_max + V_peak


def run_filter_diag(vexpr_dir, expr_dir, m, E_lo, E_hi, n_waves, k_max,
                    julia_exe='julia', rank_threshold=1e-3, n_eigs=30,
                    params_file='gaussian_fit_params.json',
                    cube_file='QD_Outputs/QD_R11.cube'):
    """Evaluate f(H)*psi on full QD grid using per-cube Julia scripts, then SVD.

    For each cube:
      - Determine which full-grid points lie inside
      - Run the pre-built Julia .jl with all n_waves
      - Accumulate results into C_f[n_waves, Ng, Ng, Ng]
    Then run SVD filter diagonalisation to extract eigenvalues.
    """
    vexpr_dir = Path(vexpr_dir)
    expr_dir  = Path(expr_dir)

    a    = 2.0 / (E_hi - E_lo)
    b    = -(E_hi + E_lo) / (E_hi - E_lo)
    a_key = f"{a:.8g}".replace('.', 'p').replace('-', 'm')
    b_key = f"{b:.8g}".replace('.', 'p').replace('-', 'm')
    jl_name = f"eval_filter_m{m}_{a_key}_{b_key}.jl"

    print(f"\n  Stage 5: filter diagonalisation")
    print(f"    m={m}  E_lo={E_lo}  E_hi={E_hi:.4f}  n_waves={n_waves}  k_max={k_max}")

    # ---- build grid ----
    x1, X, Y, Z = _build_grid(L_s=BOX_HALF, d=D_GRID)
    Ng = len(x1)
    dx = x1[1] - x1[0]
    l0 = 2.0 * BOX_HALF / N_DIVISIONS     # cube side length
    print(f"    grid: {Ng}^3 = {Ng**3:,} pts   dx={dx:.4f} Bohr   cube side={l0:.3f} Bohr")

    # ---- build numerical V on grid ----
    print("    Building V_grid (numerical) ...")
    t0 = time.time()
    atoms = read_cube_atoms(str(cube_file))
    with open(params_file, 'r') as f:
        params_data = json.load(f)
    V3d = _build_V_grid(atoms, params_data, X, Y, Z)
    print(f"    V_grid done in {time.time()-t0:.1f}s  "
          f"V range=[{V3d.min():.3f}, {V3d.max():.3f}] Hartree")

    # ---- random plane waves ----
    rng    = np.random.default_rng(42)
    k_vals = rng.uniform(-k_max, k_max, (n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, n_waves)

    # ---- per-cube grid point assignment ----
    # Grid index ix belongs to cube ci where ci = floor(ix * N_DIVISIONS / Ng)
    # Flat grid assignment arrays
    ix_all = np.arange(Ng, dtype=np.int32)
    cube_ax = np.minimum((ix_all * N_DIVISIONS) // Ng, N_DIVISIONS - 1)

    # ---- load cube metadata ----
    manager = CubicExpressionManager(str(vexpr_dir))
    if not manager.cube_info:
        raise RuntimeError(f"No cube info in {vexpr_dir}")

    # ---- accumulate C_f ----
    C_f_flat = np.zeros((n_waves, Ng**3), dtype=np.float64)
    work_dir = Path(tempfile.mkdtemp(prefix='qd_jl_'))
    total_eval_s = 0.0
    cubes_done = 0
    cubes_skip = 0

    t_eval_all = time.time()
    for idx in sorted(manager.cube_info.keys()):
        ci, cj, ck = idx
        info = manager.cube_info[idx]
        cx, cy, cz = info['center']
        cube_dir = expr_dir / _cube_dirname(cx, cy, cz)
        jl_path  = cube_dir / jl_name

        if not jl_path.exists():
            cubes_skip += 1
            continue

        # Grid indices owned by this cube along each axis
        xi_mask = (cube_ax == ci)
        yi_mask = (cube_ax == cj)
        zi_mask = (cube_ax == ck)

        ix_c = np.where(xi_mask)[0]
        iy_c = np.where(yi_mask)[0]
        iz_c = np.where(zi_mask)[0]

        # Meshgrid of flat 3D indices for this cube
        IXc, IYc, IZc = np.meshgrid(ix_c, iy_c, iz_c, indexing='ij')
        flat_idx = (IXc * Ng * Ng + IYc * Ng + IZc).ravel()

        Xf = X[IXc, IYc, IZc].ravel()
        Yf = Y[IXc, IYc, IZc].ravel()
        Zf = Z[IXc, IYc, IZc].ravel()

        if len(Xf) == 0:
            cubes_skip += 1
            continue

        try:
            out, timing = _run_julia_batch(
                jl_path, Xf, Yf, Zf, k_vals, b_vals, work_dir, julia_exe)
            # out shape: (n_waves, N_pts_cube)
            C_f_flat[:, flat_idx] = out
            cubes_done += 1
            es = timing.get('eval_s')
            if es:
                total_eval_s += es
            if cubes_done % 10 == 0:
                print(f"    {cubes_done} cubes done  "
                      f"({len(Xf)} pts/cube)  total_eval_s={total_eval_s:.1f}s")
        except Exception as exc:
            print(f"    Cube {idx}: Julia error – {exc}")
            cubes_skip += 1

    import shutil
    wall_total = time.time() - t_eval_all
    shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\n    Evaluation complete: {cubes_done} cubes  |  "
          f"{cubes_skip} skipped  |  wall={wall_total:.1f}s  "
          f"Julia eval_s={total_eval_s:.1f}s")

    # ---- reshape to (n_waves, Ng, Ng, Ng) ----
    C_f = C_f_flat.reshape(n_waves, Ng, Ng, Ng)

    # ---- SVD filter diagonalisation ----
    print("    Running SVD filter diagonalisation ...")
    t0 = time.time()

    def apply_H_func(psi3d, x_grid, V_on_grid):
        return _fft_apply_H(psi3d, x_grid, V_on_grid)

    energies, _ = svd_H(
        C_f, Ng, Ng, Ng, x1, V3d, apply_H_func,
        rank_threshold=rank_threshold,
        n_eigs=n_eigs,
    )
    print(f"    SVD done in {time.time()-t0:.1f}s")

    print(f"\n  === Filter diag result (m={m}, E_lo={E_lo}) ===")
    print(f"  First {min(n_eigs, len(energies))} eigenvalues (Hartree):")
    for i, E in enumerate(energies[:n_eigs]):
        print(f"    [{i:3d}]  {E:+.6f}")

    return energies


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QD R=11 filter diagonalisation pipeline (Julia-accelerated).'
    )
    parser.add_argument('--stages', default='1,2,3,4,5',
                        help='Comma-separated stages to run (default 1,2,3,4,5)')
    parser.add_argument('--m', type=int, default=8,
                        help='Chebyshev order (default 8)')
    parser.add_argument('--E_lo', type=float, default=0.0,
                        help='Explosion filter threshold: amplify E < E_lo (default 0.0)')
    parser.add_argument('--E_hi', type=float, default=None,
                        help='Upper energy bound (auto-estimated from grid if omitted)')
    parser.add_argument('--n_waves', type=int, default=100,
                        help='Number of random plane waves (default 100)')
    parser.add_argument('--k_max', type=float, default=1.0,
                        help='Abs. range for random k vectors (default 1.0 Bohr^-1)')
    parser.add_argument('--n_eigs', type=int, default=30,
                        help='Max eigenvalues to report (default 30)')
    parser.add_argument('--rank_threshold', type=float, default=1e-3,
                        help='SVD rank cutoff (default 1e-3)')
    parser.add_argument('--julia_exe', default='julia',
                        help='Julia executable (default: julia)')
    parser.add_argument('--cube_file', default='QD_Outputs/QD_R11.cube',
                        help='QD cube file path')
    parser.add_argument('--params_file', default='gaussian_fit_params.json',
                        help='Gaussian fit parameters JSON')
    parser.add_argument('--vexpr_dir', default='QD_R11_Vexpr',
                        help='Output dir for V_expr pkl files (stage 2)')
    parser.add_argument('--partition_mode', choices=['uniform', 'cell'], default='uniform',
                        help='Stage-2 partition mode: uniform (original) or cell (晶胞, edge default a/2).')
    parser.add_argument('--cell_edge', type=float, default=None,
                        help='Cube edge length for --partition_mode cell (Bohr). Default a/2.')
    parser.add_argument('--expr_dir', default=None,
                        help='Output dir for H^n + Julia scripts (stages 3-4). '
                             'Default: QD_R11_Julia_exp/no_expansion (no-expand) '
                             'or QD_R11_expressions (--expand)')
    parser.add_argument('--expand', action='store_true', default=False,
                        help='Use sp.expand() at each H^n step (legacy; default: off). '
                             'Off = faster generation, smaller pkl, identical Julia output.')
    args = parser.parse_args()

    stages = set(int(s.strip()) for s in args.stages.split(','))
    expand = args.expand

    # Auto-select expr_dir based on expand flag if not explicitly set
    if args.expr_dir is not None:
        expr_dir_path = args.expr_dir
    elif expand:
        expr_dir_path = 'QD_R11_expressions_cell' if args.partition_mode == 'cell' else 'QD_R11_expressions'
    else:
        expr_dir_path = 'QD_R11_Julia_exp/no_expansion_cell' if args.partition_mode == 'cell' else 'QD_R11_Julia_exp/no_expansion'

    # Auto E_hi from grid parameters (use ceil to match build_qd_cube)
    Ng = int(np.ceil(2 * BOX_HALF / D_GRID))
    E_hi_ref = _estimate_E_hi(BOX_HALF, Ng)
    print(f"QD R=11 pipeline  |  grid {Ng}^3  |  E_hi_ref={E_hi_ref:.2f} Hartree")
    E_hi = args.E_hi if args.E_hi is not None else E_hi_ref
    print(f"Using E_hi={E_hi:.4f}  E_lo={args.E_lo}  m={args.m}  "
          f"expand={expand}  expr_dir={expr_dir_path}\n")

    t_start = time.time()

    if 1 in stages:
        print("=" * 60)
        print("Stage 1: Build QD cube file")
        print("=" * 60)
        build_qd_cube(QD_RADIUS, args.cube_file, d=D_GRID)

    if 2 in stages:
        print("\n" + "=" * 60)
        print("Stage 2: Space partition → V_expr pkl files")
        print("=" * 60)
        run_space_partition(
            args.cube_file, args.params_file, args.vexpr_dir,
            L_s=BOX_HALF, n_divisions=N_DIVISIONS, r_cut=R_CUT,
            partition_mode=args.partition_mode, cell_edge=args.cell_edge,
        )

    if 3 in stages:
        print("\n" + "=" * 60)
        print("Stage 3: H^n expressions")
        print("=" * 60)
        run_h_powers(args.vexpr_dir, expr_dir_path, args.m, expand=expand)

    if 4 in stages:
        print("\n" + "=" * 60)
        print("Stage 4: Julia eval scripts")
        print("=" * 60)
        run_julia_codegen(
            args.vexpr_dir, expr_dir_path, args.m, args.E_lo, E_hi, expand=expand
        )

    if 5 in stages:
        print("\n" + "=" * 60)
        print("Stage 5: Filter diagonalisation")
        print("=" * 60)
        energies = run_filter_diag(
            vexpr_dir=args.vexpr_dir,
            expr_dir=expr_dir_path,
            m=args.m,
            E_lo=args.E_lo,
            E_hi=E_hi,
            n_waves=args.n_waves,
            k_max=args.k_max,
            julia_exe=args.julia_exe,
            rank_threshold=args.rank_threshold,
            n_eigs=args.n_eigs,
            params_file=args.params_file,
            cube_file=args.cube_file,
        )

        # Save eigenvalues to JSON
        out_json = Path(expr_dir_path) / f'eigenvalues_m{args.m}_Elo{args.E_lo}.json'
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump({
                'm': args.m, 'E_lo': args.E_lo, 'E_hi': E_hi,
                'n_waves': args.n_waves, 'k_max': args.k_max,
                'energies_hartree': [float(e) for e in energies],
            }, f, indent=2)
        print(f"\n  Eigenvalues saved → {out_json}")

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
