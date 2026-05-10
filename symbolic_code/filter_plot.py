"""Chebyshev explosion/bandpass filter visualisation for the symbolic pipeline.

The filter function is

    f(E) = T_m(a*E + b)

where
    a =  2 / (E_hi - E_lo)          (reciprocal of half-window width)
    b = -(E_hi + E_lo) / (E_hi - E_lo)   (shift to centre at 0)

This maps the window [E_lo, E_hi] -> [-1, 1], so:
  * eigenvalues INSIDE  [E_lo, E_hi]  ->  |T_m| <= 1   (passband / suppressed)
  * eigenvalues OUTSIDE [E_lo, E_hi]  ->  |T_m| grows polynomially with m

Two common configurations
--------------------------
Explosion  (asymmetric window):
    E_lo = <energy threshold>
    E_hi = <max eigenvalue of H + margin>     (>> E_lo)
    -> eigenvalues with E < E_lo are amplified;  [E_lo, E_hi] is suppressed.

Bandpass  (narrow symmetric window):
    E_lo, E_hi bracket only the eigenvalues of interest.
    -> inside states have bounded coefficients, outside states are amplified.
    SVD filter diagonalisation then recovers eigenvalues in [E_lo, E_hi].

Public API
----------
filter_response          -- evaluate |T_m(aE+b)| on an energy array
plot_chebyshev_filter    -- produce a two-panel (linear + log) plot
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev as _Cheb


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def filter_response(E_arr, m, E_lo, E_hi):
    """Return |T_m(aE+b)| evaluated on *E_arr*.

    Uses the stable Chebyshev evaluation (not the monomial expansion which
    overflows for large m and |aE+b| >> 1).

    Parameters
    ----------
    E_arr : array-like        Energy values
    m : int                   Chebyshev polynomial order
    E_lo, E_hi : float        Window boundaries (mapped to [-1, 1])

    Returns
    -------
    ndarray  same shape as E_arr
    """
    E_arr = np.asarray(E_arr, dtype=float)
    a = 2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)
    coef = np.zeros(m + 1)
    coef[m] = 1.0          # T_m in Chebyshev-coefficient basis
    return np.abs(_Cheb(coef)(a * E_arr + b))


def _ab(E_lo, E_hi):
    """Return (a, b) rescaling coefficients."""
    a =  2.0 / (E_hi - E_lo)
    b = -(E_hi + E_lo) / (E_hi - E_lo)
    return a, b


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_chebyshev_filter(
    m_list, E_lo, E_hi,
    E_min=None, E_max=None,
    n_pts=3000, clip=100.0,
    mode='explosion',
    eigenvalues=None,
    out_path=None,
    ax=None,
):
    """Plot |T_m(aE+b)| for one or more polynomial orders m.

    Produces a two-panel figure:
      * top   : linear-scale plot (clipped at *clip*) with shaded regions
      * bottom: log-scale plot showing the full dynamic range

    Parameters
    ----------
    m_list : int or list of int
        Chebyshev order(s) to overlay on the same axes.
    E_lo, E_hi : float
        Energy window boundaries.  [E_lo, E_hi] maps to [-1, 1].
    E_min, E_max : float, optional
        x-axis range.  Defaults extend the window by ~30%.
    n_pts : int
        Number of sample points (default 3000).
    clip : float
        Clamp |T_m| at this value in the linear panel (default 100).
    mode : 'explosion' | 'bandpass'
        Controls shading:
        * 'explosion' - red shading for E < E_lo (amplification zone)
        * 'bandpass'  - green shading only for the target window
    eigenvalues : array-like, optional
        If given, draw vertical ticks at these energies (e.g. exact eigenvalues).
    out_path : str or Path, optional
        Save figure to this path if given.
    ax : matplotlib.Axes, optional
        Draw on this axes (top panel only; log panel is skipped).

    Returns
    -------
    (fig, ax_main)
    """
    if isinstance(m_list, int):
        m_list = [m_list]

    a, b = _ab(E_lo, E_hi)
    hw = E_hi - E_lo

    if E_min is None:
        E_min = E_lo - 0.35 * hw
    if E_max is None:
        E_max = E_hi + 0.15 * hw if mode == 'explosion' else E_hi + 0.35 * hw

    E_arr = np.linspace(E_min, E_max, n_pts)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(m_list), 1)))

    own_fig = (ax is None)
    if own_fig:
        fig, axes = plt.subplots(
            2, 1, figsize=(11, 8),
            gridspec_kw={'height_ratios': [3, 2]},
            sharex=True,
        )
        ax_main, ax_log = axes
    else:
        fig = ax.get_figure()
        ax_main = ax
        ax_log  = None

    # -- draw curves --
    for m, col in zip(m_list, colors):
        resp    = filter_response(E_arr, m, E_lo, E_hi)
        clipped = np.clip(resp, 0.0, clip)
        ax_main.plot(E_arr, clipped, color=col, lw=1.8, label=f"m = {m}")
        if ax_log is not None:
            safe = np.where(resp > 1e-15, resp, 1e-15)
            ax_log.semilogy(E_arr, safe, color=col, lw=1.5, label=f"m = {m}")

    # -- region shading --
    lo_x = max(E_min, E_lo)
    hi_x = min(E_max, E_hi)
    for axi in ([ax_main, ax_log] if ax_log else [ax_main]):
        if lo_x < hi_x:
            axi.axvspan(lo_x, hi_x, alpha=0.12, color='green',
                        label=f"passband [{E_lo}, {E_hi}]")
        if mode == 'explosion' and E_min < E_lo:
            axi.axvspan(E_min, E_lo, alpha=0.12, color='red',
                        label=f"explosion  E < {E_lo}")
        axi.axvline(E_lo, color='green',     ls='--', lw=1.2)
        axi.axvline(E_hi, color='darkgreen', ls='--', lw=1.2)
        axi.axhline(1.0,  color='gray',      ls=':',  lw=1.0)
        axi.grid(True, ls='--', alpha=0.35)
        axi.set_xlim(E_min, E_max)

    # -- optional eigenvalue ticks --
    if eigenvalues is not None:
        eigenvalues = np.asarray(eigenvalues)
        ax_main.scatter(
            eigenvalues,
            np.full_like(eigenvalues, -clip * 0.04),
            marker='^', s=40, color='black', zorder=5,
            clip_on=False, label='eigenvalues',
        )

    ax_main.set_ylim(-clip * 0.06, clip * 1.05)
    ax_main.set_ylabel(f"|T_m(aE + b)|  (clipped at {clip})")
    ax_main.set_title(
        f"Chebyshev filter response   [{E_lo}, {E_hi}] -> [-1, 1]\n"
        f"a = {a:.5f}   b = {b:.5f}   "
        f"mode = {mode}"
    )
    ax_main.legend(fontsize=9, ncol=2, loc='upper right')

    if ax_log is not None:
        ax_log.axhline(1.0, color='gray', ls=':', lw=1.0)
        ax_log.set_ylabel("|T_m(aE + b)|  (log scale)")
        ax_log.set_xlabel("Energy")
        ax_log.legend(fontsize=8, ncol=2, loc='upper right')
    else:
        ax_main.set_xlabel("Energy")

    if own_fig:
        fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"  Saved -> {out_path}")

    return fig, ax_main


# ---------------------------------------------------------------------------
# Quick informational summary
# ---------------------------------------------------------------------------

def print_filter_summary(m_list, E_lo, E_hi, probe_energies=None):
    """Print a text table of filter parameters and response values.

    Parameters
    ----------
    m_list : int or list of int
    E_lo, E_hi : float
    probe_energies : list of float, optional
        Extra energies to evaluate (e.g. known eigenvalues).
    """
    if isinstance(m_list, int):
        m_list = [m_list]
    a, b = _ab(E_lo, E_hi)

    print("\nChebyshev filter summary")
    print(f"  E_lo = {E_lo},  E_hi = {E_hi}")
    print(f"  a    = {a:.6f}   (scaling: E -> aE+b)")
    print(f"  b    = {b:.6f}")
    print(f"  window width = {E_hi - E_lo:.4f}")
    print()

    probe = [E_lo - (E_hi - E_lo) * 0.1,
             E_lo - (E_hi - E_lo) * 0.5,
             E_lo - (E_hi - E_lo) * 1.0]
    if probe_energies:
        probe = list(probe_energies)

    header = f"  {'E':>10}  {'aE+b':>8}  " + "".join(f"  |T_{m}|" for m in m_list)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for E in probe:
        x = a * E + b
        vals = " ".join(
            f"{filter_response([E], m, E_lo, E_hi)[0]:>8.3e}" for m in m_list
        )
        print(f"  {E:>10.4f}  {x:>8.4f}  {vals}")
    print()
    print(f"  Peak |T_m| inside passband (should be <= 1):")
    E_in = np.linspace(E_lo, E_hi, 500)
    for m in m_list:
        peak = filter_response(E_in, m, E_lo, E_hi).max()
        print(f"    m={m:>3}: {peak:.6f}")
