"""Symbolic generation of H^n or (aH+b)^n * psi for plane-wave basis states.

H = -pref * laplacian + V(x,y,z)   (pref = 0.5 in atomic units)

Starting from psi = sin(kx*x + ky*y + kz*z + b) we write
    psi = Ps * sin(theta) + Pc * cos(theta)
and recursively apply H to update (Ps, Pc), keeping only the
envelopes (no trig factors). Each application expands the polynomial
degree in x,y,z by 2.

Two generation strategies
--------------------------
generate_H_powers        – computes pure H^n (recommended).  Files are
                           reusable for any energy window; sp.expand() keeps
                           expressions compact with rational coefficients.
generate_scaled_H_powers – computes (aH+b)^n with a/b baked in (legacy).
                           Use only if you need a single fixed energy window.

Public API
----------
laplacian, directional_derivative, apply_H_on_pair
generate_H_powers
generate_scaled_H_powers
process_all_cubes, process_specific_cubes
"""

import gzip
import pickle
from pathlib import Path

import sympy as sp


# ---------------------------------------------------------------------------
# Differential operators
# ---------------------------------------------------------------------------

def laplacian(f, x, y, z):
    """返回 ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²."""
    return sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)


def directional_derivative(f, kvec, x, y, z):
    """Return k · ∇f = kx ∂f/∂x + ky ∂f/∂y + kz ∂f/∂z."""
    kx, ky, kz = kvec
    return kx * sp.diff(f, x) + ky * sp.diff(f, y) + kz * sp.diff(f, z)


# ---------------------------------------------------------------------------
# One H-application step
# ---------------------------------------------------------------------------

def apply_H_on_pair(Ps, Pc, V, kvec, k2, pref, x, y, z):
    """Apply H = -pref*∇² + V to Ps*sin(θ) + Pc*cos(θ).

    Uses the identities
        ∇²[f sinθ] = (∇²f) sinθ + 2(∇f·k) cosθ - k² f sinθ
        ∇²[f cosθ] = (∇²f) cosθ - 2(∇f·k) sinθ - k² f cosθ
    so that the result is still a linear combination of sinθ and cosθ.

    Parameters
    ----------
    Ps, Pc : sympy expressions
        Envelope functions of (x,y,z) only – no trig factors.
    V : sympy expression
        Local potential V(x,y,z).
    kvec : tuple (kx_sym, ky_sym, kz_sym)
    k2 : sympy expression   kx²+ky²+kz²
    pref : float            kinetic prefactor (0.5 in a.u.)
    x, y, z : sympy symbols

    Returns
    -------
    (Ps_new, Pc_new) where H*psi = Ps_new*sinθ + Pc_new*cosθ
    """
    lap_Ps = laplacian(Ps, x, y, z)
    dir_Ps = directional_derivative(Ps, kvec, x, y, z)
    lap_Pc = laplacian(Pc, x, y, z)
    dir_Pc = directional_derivative(Pc, kvec, x, y, z)

    Ps_new = sp.Add(pref * k2 * Ps, -pref * lap_Ps,  dir_Pc, V * Ps, evaluate=False)
    Pc_new = sp.Add(pref * k2 * Pc, -pref * lap_Pc, -dir_Ps, V * Pc, evaluate=False)
    return Ps_new, Pc_new


# ---------------------------------------------------------------------------
# File I/O helper
# ---------------------------------------------------------------------------

def _save_power(outdir, n, data, file_format, prefix):
    """Write one power dict {Ps, Pc} to disk."""
    stem = outdir / f'{prefix}_{n}'
    if file_format == 'sym.gz':
        with gzip.open(str(stem) + '.sym.gz', 'wt', encoding='utf8') as f:
            f.write(str(data))
    else:
        with open(str(stem) + '.pkl', 'wb') as f:
            pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Strategy A: pure H^n  (recommended default)
# ---------------------------------------------------------------------------

def generate_H_powers(
    N, outdir,
    file_format='pkl',
    expand=True,
    V=None, kvec=None, k2=None, pref=0.5,
    x=None, y=None, z=None,
):
    """Compute and save H^n * psi for n = 0 … N (no energy-window scaling).

    Files named ``H_power_{n}.pkl`` / ``H_power_{n}.sym.gz`` can be reused
    for any Chebyshev filter by calling
    ``chebyshev_filter.apply_f_of_H_from_raw_powers`` with the desired a, b.

    Parameters
    ----------
    N : int
        Maximum power.
    outdir : str or Path
        Output directory.
    file_format : 'pkl' | 'sym.gz'
        Binary pickle (faster I/O) or compressed text (human-readable).
    expand : bool
        If True (default), call sp.expand() at each step so sympy works with
        compact flat-polynomial form (combined like terms, rational coefficients).
        If False, skip sp.expand() to preserve the product/sum tree structure —
        e.g. ``V * Ps`` stays as-is instead of being distributed.  The
        unexpanded trees are larger in sympy but carry factored structure that
        sp.cse() can exploit when generating Julia code.
    V, kvec, k2, pref, x, y, z
        Symbolic Hamiltonian ingredients (see apply_H_on_pair).

    Returns
    -------
    dict  n -> {'Ps': expr, 'Pc': expr}
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Ps = sp.Integer(1)
    Pc = sp.Integer(0)
    results = {0: {'Ps': Ps, 'Pc': Pc}}
    _save_power(outdir, 0, results[0], file_format, prefix='H_power')

    for n in range(1, N + 1):
        print(f"    H^{n} ...", end=' ', flush=True)
        Ps_H, Pc_H = apply_H_on_pair(Ps, Pc, V, kvec, k2, pref, x, y, z)
        if expand:
            Ps = sp.expand(Ps_H)
            Pc = sp.expand(Pc_H)
        else:
            Ps = Ps_H
            Pc = Pc_H
        results[n] = {'Ps': Ps, 'Pc': Pc}
        _save_power(outdir, n, results[n], file_format, prefix='H_power')
        print('✓')

    return results


# ---------------------------------------------------------------------------
# Strategy B: (aH+b)^n  (legacy)
# ---------------------------------------------------------------------------

def generate_scaled_H_powers(
    N, a, b, outdir,
    file_format='sym.gz',
    V=None, kvec=None, k2=None, pref=0.5,
    x=None, y=None, z=None,
):
    """Compute and save (aH+b)^n * psi for n = 0 … N.

    The energy-window scaling (a, b) is baked into the expressions, so
    the files are specific to one energy window.  Prefer generate_H_powers
    when you may need to change E_lo / E_hi without recomputing.

    Files named ``H_scaled_power_{n}.pkl`` / ``.sym.gz``.

    Parameters
    ----------
    N : int
    a, b : float
        Chebyshev rescaling: a = 2/(E_hi-E_lo), b = -(E_hi+E_lo)/(E_hi-E_lo).
    outdir : str or Path
    file_format : 'sym.gz' | 'pkl'
    V, kvec, k2, pref, x, y, z

    Returns
    -------
    dict  n -> {'Ps': expr, 'Pc': expr}
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Ps = sp.Integer(1)
    Pc = sp.Integer(0)
    results = {0: {'Ps': Ps, 'Pc': Pc}}

    for n in range(1, N + 1):
        print(f"    (aH+b)^{n} ...", end=' ', flush=True)
        Ps_H, Pc_H = apply_H_on_pair(Ps, Pc, V, kvec, k2, pref, x, y, z)
        Ps = a * Ps_H + b * Ps
        Pc = a * Pc_H + b * Pc
        results[n] = {'Ps': Ps, 'Pc': Pc}
        _save_power(outdir, n, results[n], file_format, prefix='H_scaled_power')
        print('✓')

    return results


# ---------------------------------------------------------------------------
# Bulk helpers
# ---------------------------------------------------------------------------

def process_all_cubes(
    manager, N, a=None, b=None,
    base_outdir='H_powers_basis_partitioned',
    file_format='sym.gz',
    method='raw',
):
    """Generate H^n or (aH+b)^n * psi for every cube in *manager*.

    Parameters
    ----------
    manager : CubicExpressionManager
    N : int
    a, b : float
        Chebyshev scaling; required only when method='scaled'.
    base_outdir : str
    file_format : str
    method : 'raw' | 'scaled'
        'raw'    – pure H^n; files reusable for any energy window (default).
        'scaled' – (aH+b)^n with a,b baked in.
    """
    if method == 'scaled' and (a is None or b is None):
        raise ValueError("method='scaled' requires a and b (from E_lo, E_hi)")

    base_outdir = Path(base_outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    x, y, z = sp.symbols('x y z')
    kx, ky, kz = sp.symbols('kx ky kz')
    kvec = (kx, ky, kz)
    k2   = kx**2 + ky**2 + kz**2
    pref = 0.5

    total = len(manager)
    done = skipped = 0

    print(f"\n{'='*70}")
    print(f"PROCESSING ALL CUBES  method={method}  N={N}", end='')
    if method == 'scaled':
        print(f"  a={a:.4f}  b={b:.4f}")
    else:
        print()
    print(f"Output → {base_outdir}")
    print(f"{'='*70}\n")

    for idx in sorted(manager.expressions.keys()):
        i, j, k = idx
        print(f"[{done+skipped+1}/{total}] Cube ({i},{j},{k})", end=' ... ')

        V_expr = manager.get_expression(i, j, k)
        if V_expr is None:
            print('no expr – skipped')
            skipped += 1
            continue

        info = manager.cube_info[idx]
        cx, cy, cz = info['center']
        print(f"center=({cx:.2f},{cy:.2f},{cz:.2f}) atoms={info['n_atoms']}")

        cube_dir = base_outdir / _cube_dirname(cx, cy, cz)
        cube_dir.mkdir(parents=True, exist_ok=True)
        _write_cube_info(cube_dir, idx, info, N, a, b)

        try:
            if method == 'scaled':
                generate_scaled_H_powers(
                    N, a, b, outdir=cube_dir, file_format=file_format,
                    V=V_expr, kvec=kvec, k2=k2, pref=pref, x=x, y=y, z=z,
                )
            else:
                generate_H_powers(
                    N, outdir=cube_dir, file_format=file_format,
                    V=V_expr, kvec=kvec, k2=k2, pref=pref, x=x, y=y, z=z,
                )
            print(f"  ✓ powers 0…{N} in {cube_dir.name}/")
            done += 1
        except Exception as exc:
            print(f"  ✗ Error: {exc}")
            skipped += 1

    print(f"\nDone. processed={done}  skipped={skipped}")


def process_specific_cubes(
    manager, cube_indices, N, a=None, b=None,
    base_outdir='H_powers_basis_partitioned',
    file_format='sym.gz',
    method='raw',
):
    """Like process_all_cubes but for a subset of cube indices."""
    if method == 'scaled' and (a is None or b is None):
        raise ValueError("method='scaled' requires a and b")

    base_outdir = Path(base_outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    x, y, z = sp.symbols('x y z')
    kx, ky, kz = sp.symbols('kx ky kz')
    kvec = (kx, ky, kz)
    k2   = kx**2 + ky**2 + kz**2
    pref = 0.5

    print(f"Processing {len(cube_indices)} specific cubes (method={method}) ...")

    for idx in cube_indices:
        i, j, k = idx
        if idx not in manager:
            print(f"  Cube {idx}: not in manager – skipped")
            continue

        V_expr = manager.get_expression(i, j, k)
        info   = manager.cube_info[idx]
        cx, cy, cz = info['center']
        print(f"  Cube {idx}: center=({cx:.2f},{cy:.2f},{cz:.2f})")

        cube_dir = base_outdir / _cube_dirname(cx, cy, cz)
        cube_dir.mkdir(parents=True, exist_ok=True)
        _write_cube_info(cube_dir, idx, info, N, a, b)

        try:
            if method == 'scaled':
                generate_scaled_H_powers(
                    N, a, b, outdir=cube_dir, file_format=file_format,
                    V=V_expr, kvec=kvec, k2=k2, pref=pref, x=x, y=y, z=z,
                )
            else:
                generate_H_powers(
                    N, outdir=cube_dir, file_format=file_format,
                    V=V_expr, kvec=kvec, k2=k2, pref=pref, x=x, y=y, z=z,
                )
            print(f"    ✓ complete")
        except Exception as exc:
            print(f"    ✗ Error: {exc}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cube_dirname(cx, cy, cz):
    """Build a filesystem-safe subdirectory name from cube centre coords."""
    def fmt(v):
        return f"{v:.3f}".replace("-", "neg")
    return f"cube_{fmt(cx)}_{fmt(cy)}_{fmt(cz)}"


def _write_cube_info(cube_dir, idx, info, N, a, b):
    with open(cube_dir / 'cube_info.txt', 'w', encoding='utf-8') as f:
        f.write(f"Cube index  : {idx}\n")
        f.write(f"Center      : {info['center']} Bohr\n")
        f.write(f"Atoms       : {info['n_atoms']}\n")
        f.write(f"Cube size   : {info['cube_size']:.6f} Bohr\n")
        f.write(f"r_cut       : {info['r_cut']:.6f} Bohr\n")
        f.write(f"a           : {a if a is not None else 'N/A (raw H^n mode)'}\n")
        f.write(f"b           : {b if b is not None else 'N/A (raw H^n mode)'}\n")
        f.write(f"N (max pow) : {N}\n")
