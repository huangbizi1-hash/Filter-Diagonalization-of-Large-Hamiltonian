"""Chebyshev filter f(H)*psi assembly from symbolic H^n*psi expressions.

Workflow
--------
1. ``load_H_powers``             – read (aH+b)^n files produced by h_powers.py
2. ``chebyshev_coeffs_transformed`` – coefficients of T_n(aH+b)
3. ``apply_f_of_H_on_psi``       – assemble f(H)*psi = Σ c_n (aH+b)^n * psi
4. ``extract_cos_sin_coeffs``    – split into cos-phase / sin-phase envelopes
5. ``group_by_exp_combined``     – group terms by their Gaussian factor
6. ``apply_horner``              – Horner form for polynomial parts
7. ``svd_H``                     – filter-diagonalisation via QR+SVD (numerical)

Note: ``fH_func.py`` and ``fH_func_AsIn.py`` have been merged here;
      the only difference was the default folder path, which is now a
      parameter of ``apply_f_of_H_on_psi``.
"""

import gzip
import pickle
from pathlib import Path

import sympy as sp
from sympy import symbols, exp, Add, Mul, powsimp


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_expr_srepr(path):
    """Load a sympy expression from a ``.sym.gz`` or plain-text file."""
    path = str(path)
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf8') as f:
            s = f.read()
    else:
        with open(path, 'r', encoding='utf8') as f:
            s = f.read()
    return sp.sympify(s, evaluate=False)


def load_H_powers(folder, n_max, file_type='sym.gz'):
    """Load H^n*psi dicts from a cube subdirectory.

    Parameters
    ----------
    folder : str or Path
    n_max  : int   maximum order to load
    file_type : 'sym.gz' | 'pkl'

    Returns
    -------
    dict  n (int) → {'Ps': expr, 'Pc': expr}
    """
    folder = Path(folder)
    if file_type == 'pkl':
        pattern = 'H_scaled_power_*.pkl'
    elif file_type == 'sym.gz':
        pattern = 'H_scaled_power_*.sym.gz'
    else:
        raise ValueError(f"Unsupported file_type: {file_type!r}")

    results = {}
    for fpath in sorted(folder.glob(pattern)):
        # parse n from filenames like H_scaled_power_3.sym.gz
        stem = fpath.name.replace('.sym.gz', '').replace('.pkl', '')
        n = int(stem.split('_')[-1])
        if file_type == 'pkl':
            with open(fpath, 'rb') as fh:
                data = pickle.load(fh)
        else:
            data = load_expr_srepr(fpath)
        results[n] = data
        if n >= n_max:
            break
    return results


# ---------------------------------------------------------------------------
# Chebyshev coefficients
# ---------------------------------------------------------------------------

def chebyshev_coeffs_transformed(n, a=1.0, b=0.0):
    """Return polynomial coefficients of T_n(a*x + b) in ascending degree.

    These rescale T_n so that the energy window [E_lo, E_hi] maps to [-1,1]:
        a = 2 / (E_hi - E_lo)
        b = -(E_hi + E_lo) / (E_hi - E_lo)

    Returns list [c_0, c_1, …, c_n].
    """
    x = sp.Symbol('x')
    Tn = sp.chebyshevt(n, x)
    Tn_shifted = sp.expand(Tn.subs(x, a * x + b))
    poly = sp.Poly(Tn_shifted, x)
    return [poly.coeff_monomial(x**i) for i in range(n + 1)]


# ---------------------------------------------------------------------------
# f(H)*psi assembly
# ---------------------------------------------------------------------------

def apply_f_of_H_on_psi(folder, f_coeffs, n_max, file_type='sym.gz'):
    """Assemble  f(H)*sin(θ)  symbolically via Chebyshev expansion.

    f is expanded as  f(H) = c_0 + c_1*H + … + c_N*H^N  where the H^n*psi
    files are loaded from *folder*.  The result is
        psi_fH = A_sin(x,y,z,...)*sin(θ) + A_cos(x,y,z,...)*cos(θ)

    Parameters
    ----------
    folder     : str or Path  directory with ``H_scaled_power_n`` files
    f_coeffs   : list         [c_0, c_1, …, c_N] (from chebyshev_coeffs_transformed)
    n_max      : int          must equal len(f_coeffs) - 1
    file_type  : str

    Returns
    -------
    psi_fH : sympy expression in x, y, z, kx, ky, kz, b
    """
    results = load_H_powers(folder, n_max, file_type=file_type)
    x, y, z, kx, ky, kz, b = sp.symbols('x y z kx ky kz b')
    theta = kx * x + ky * y + kz * z + b

    Ps_total = sp.Integer(0)
    Pc_total = sp.Integer(0)
    for n, c in enumerate(f_coeffs[1:], start=1):
        Ps_total += c * results[n]['Ps']
        Pc_total += c * results[n]['Pc']

    # c_0 term: H^0 * psi = psi = 1*sin(θ), so Ps=1, Pc=0
    psi_fH = (
        Ps_total * sp.sin(theta)
        + Pc_total * sp.cos(theta)
        + f_coeffs[0] * sp.sin(theta)
    )
    return psi_fH


# ---------------------------------------------------------------------------
# Expression analysis
# ---------------------------------------------------------------------------

def extract_cos_sin_coeffs(psi_fH):
    """Extract the coefficients of cos(θ) and sin(θ) from f(H)*psi.

    Returns
    -------
    (expr_cos, expr_sin) : sympy expressions in x, y, z, kx, ky, kz, b
    """
    x, y, z, kx, ky, kz, b = symbols('x y z kx ky kz b')
    phase = b + kx * x + ky * y + kz * z
    expanded = psi_fH.expand()
    expr_cos = expanded.coeff(sp.cos(phase), 1) or sp.Integer(0)
    expr_sin = expanded.coeff(sp.sin(phase), 1) or sp.Integer(0)
    return expr_cos, expr_sin


def group_by_exp_combined(expr):
    """Group terms by shared exponential (Gaussian) factor.

    After grouping, each Gaussian envelope is evaluated only once during
    numerical computation.

    Returns
    -------
    list of (exp_part, poly_part) tuples
        ``exp_part`` is a product of ``exp(…)`` factors (or sp.Integer(1));
        ``poly_part`` is the remaining polynomial coefficient.
    """
    if expr == 0:
        return []

    expr = powsimp(expr, combine='all')
    grouped = {}
    for term in Add.make_args(expr):
        exp_part  = sp.Integer(1)
        poly_part = sp.Integer(1)
        for factor in Mul.make_args(term):
            if factor.has(exp):
                exp_part  = exp_part  * factor
            else:
                poly_part = poly_part * factor
        grouped[exp_part] = grouped.get(exp_part, sp.Integer(0)) + poly_part

    return list(grouped.items())


def apply_horner(terms):
    """Apply Horner's method to the polynomial part of each term.

    Reduces multiplications when evaluating the expression numerically.

    Parameters
    ----------
    terms : list of (exp_part, poly_part)

    Returns
    -------
    list of (exp_part, horner_poly_part)
    """
    return [(ep, sp.horner(pp)) for ep, pp in terms]


# ---------------------------------------------------------------------------
# Numerical diagonalisation (requires FFT_func / fft_code)
# ---------------------------------------------------------------------------

def svd_H(
    filtered_psi_matrix, Nx, Ny, Nz,
    x_grid, V_on_grid, apply_H_func,
    rank_threshold=1e-3, n_eigs=200,
):
    """Filter diagonalisation: QR + SVD on a filtered basis.

    Takes a stack of f(H)*psi states, orthogonalises them via QR+SVD,
    builds a small projected Hamiltonian H_tilde, and diagonalises it.

    Parameters
    ----------
    filtered_psi_matrix : ndarray, shape (n_filters, Nx, Ny, Nz)
    Nx, Ny, Nz : int
    x_grid : ndarray
    V_on_grid : ndarray
    apply_H_func : callable
        ``apply_H_func(psi_3d, x_grid, V_on_grid)`` → H*psi_3d.
        Typically ``fft_code.hamiltonian.apply_H`` or
        ``FFT_func.scaled_fft_eval_H_psi``.
    rank_threshold : float
        Singular-value cutoff for effective-rank determination.
    n_eigs : int
        Maximum number of eigenvalues to return.

    Returns
    -------
    energies : ndarray  sorted eigenvalues
    Ur       : ndarray  reduced basis (flat index)
    """
    import numpy as np
    from scipy.linalg import eigh

    def _to_vec(g):
        return g.reshape(g.shape[0], -1).T

    def _to_grid(v):
        return v.T.reshape(v.shape[-1], Nx, Ny, Nz)

    C_f = _to_vec(filtered_psi_matrix)
    C_f = C_f / np.linalg.norm(C_f, axis=0)
    C_f = C_f[:, ~np.any(np.isinf(C_f), axis=0)]

    Q, R = np.linalg.qr(C_f, mode='reduced')
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    U = Q @ U1

    r = int(np.sum(sigma > rank_threshold))
    Ur = U[:, :r]
    Ur_grid = _to_grid(Ur)

    H_Ur_grid = np.zeros_like(Ur_grid, dtype='complex128')
    for i in range(r):
        H_Ur_grid[i] = apply_H_func(Ur_grid[i], x_grid, V_on_grid)

    H_Ur    = _to_vec(H_Ur_grid)
    H_tilde = Ur.conj().T @ H_Ur
    energies, _ = eigh(H_tilde)
    return np.sort(energies)[:n_eigs], Ur
