#!/usr/bin/env python3
"""benchmark_cse_threshold.py
Test CSE occurrence-count thresholds to find the optimal register-pressure / FLOP tradeoff.

Two CSE strategies are compared
--------------------------------
flat_cse   (existing)
    sp.cse() is applied to the raw flat-expanded polynomial groups returned by
    group_by_exp_combined.  The 2037 shared sub-expressions are mostly low-level
    monomials (x^2, x^2*y^2, ...) — dense-but-small, causing register pressure
    when all kept.

horner_cse (new, "baseline + CSE")
    apply_horner() is called first (same as the Horner baseline), then sp.cse()
    is applied to the resulting compact nested expressions.  The hope is that
    Horner already factors out r2 = x^2+y^2+z^2 and k2 = kx^2+ky^2+kz^2 as
    natural repeated structures, so CSE finds only a handful of high-level temps.
    If n_kept drops to << 30, all temps fit in registers → performance comparable
    to or better than baseline.

Expected outcome
----------------
  flat_cse T=20   : ~116 kept → 100 ms/wave (register pressure)
  horner_cse T=1  : few kept (r2, k2, H?) → close to 0.642 ms/wave?

Usage
-----
  python benchmark_cse_threshold.py --m 8 --n_waves 20
  python benchmark_cse_threshold.py --m 8 --distribution_only   # fast: just histograms
  python benchmark_cse_threshold.py --m 8 --n_waves 20 --thresholds 1,5,20,50
  python benchmark_cse_threshold.py --m 8 --n_waves 20 --mode flat   # skip horner_cse
  python benchmark_cse_threshold.py --m 8 --n_waves 20 --mode horner # skip flat_cse
"""

import argparse
import json
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_strategies import (
    _jl, _exp_funcs, _precompute_exp, _main_func,
    _EVAL_SIG, _TRIG_BLOCK, run_julia, prepare_expressions,
    build_baseline,
)
from symbolic_code.chebyshev_filter import (
    apply_horner,
    apply_f_of_H_from_raw_powers,
    extract_cos_sin_coeffs,
    group_by_exp_combined,
)


# ---------------------------------------------------------------------------
# Occurrence counting
# ---------------------------------------------------------------------------

def count_cse_occurrences(repl, reduced):
    """For each sym in repl, count how many times it appears after its definition.

    Counts occurrences in:
      - repl[i+1:] (later replacement expressions)
      - reduced     (the final output expressions)

    This equals the number of times the temp saves a repeated computation.
    A count of 1 means the sub-expression is used exactly once — the CSE
    only introduces a local variable with no arithmetic savings.
    """
    counts = {}
    for i, (sym, _) in enumerate(repl):
        n = 0
        for _, later_expr in repl[i + 1:]:
            n += later_expr.count(sym)
        for r in reduced:
            n += r.count(sym)
        counts[sym] = n
    return counts


def show_count_distribution(counts, thresholds=None, title='CSE occurrence-count distribution'):
    """Print a text histogram of CSE occurrence counts and a threshold table."""
    vals = list(counts.values())
    hist = Counter(vals)
    total = len(vals)

    if thresholds is None:
        thresholds = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000]

    print(f"\n{'─'*55}")
    print(f"{title}  ({total} total temps)")
    print(f"{'─'*55}")
    print(f"{'count':>7}  {'n_temps':>7}  {'cumul%':>7}  histogram")
    print(f"{'─'*55}")

    cumulative = 0
    shown = 0
    sorted_counts = sorted(hist.keys())
    for c in sorted_counts:
        n = hist[c]
        cumulative += n
        pct_cum = cumulative / total * 100
        bar = '█' * max(1, int(n / total * 60))
        print(f"  {c:>5}  {n:>7}  {pct_cum:>6.1f}%  {bar}")
        shown += 1
        if shown >= 25 and c < sorted_counts[-1]:
            rest_count = total - cumulative
            if rest_count > 0:
                print(f"  ...    (remaining {rest_count} temps with count > {c})")
            break

    # Threshold table
    print(f"\n{'─'*55}")
    print(f"Effect of threshold T  (keep temp if count >= T):")
    print(f"{'─'*55}")
    print(f"{'T':>6}  {'kept':>6}  {'kept%':>7}  {'inlined':>8}  "
          f"{'savings_est':>12}")
    print(f"{'─'*55}")

    for t in thresholds:
        n_kept   = sum(1 for v in vals if v >= t)
        n_inline = total - n_kept
        saved_count = sum(v for v in vals if v >= t)
        print(f"  {t:>4}  {n_kept:>6}  {n_kept/total*100:>6.1f}%  "
              f"{n_inline:>8}  {saved_count:>12} (count sum)")


# ---------------------------------------------------------------------------
# Selective CSE builder
# ---------------------------------------------------------------------------

def selective_cse_build(terms_cos, terms_sin, min_count, repl, reduced, counts,
                        strategy_label=None):
    """Build a Julia cse_inlined script keeping only temps with count >= min_count.

    Temps below the threshold are inlined back into downstream expressions
    (both later kept-temp definitions and the final reduced expressions).
    """
    n_cos = len(terms_cos)
    n_sin = len(terms_sin)

    if strategy_label is None:
        strategy_label = f'cse_inlined(min_count={min_count})'

    # Forward pass: resolve inline substitutions as we go.
    # Because repl is topologically ordered (si only references sj for j<i),
    # processing forward guarantees that inline_map always contains fully
    # resolved expressions (no pending chain substitutions).
    inline_map = {}
    kept = []

    for sym, expr in repl:
        expr_resolved = expr.xreplace(inline_map) if inline_map else expr
        if counts.get(sym, 0) < min_count:
            inline_map[sym] = expr_resolved
        else:
            kept.append((sym, expr_resolved))

    final_reduced = ([r.xreplace(inline_map) for r in reduced]
                     if inline_map else list(reduced))

    ops_kept    = sum(int(sp.count_ops(expr)) for _, expr in kept)
    ops_reduced = sum(int(sp.count_ops(r)) for r in final_reduced)
    ops_post    = ops_kept + ops_reduced

    # Build Julia source
    L = [f'# strategy: {strategy_label}']
    L += _exp_funcs(terms_cos, terms_sin)
    L += _precompute_exp(n_cos, n_sin)

    cse_lines = [f'        {sym} = {_jl(expr)}' for sym, expr in kept]
    cse_block = ('\n'.join(cse_lines)
                 if cse_lines else '        # no CSE temps (all inlined)')

    cos_lines = '\n'.join(
        f'        _sum_cos += exp_cos[i,{k+1}] * ({_jl(r)})'
        for k, r in enumerate(final_reduced[:n_cos]))
    sin_lines = '\n'.join(
        f'        _sum_sin += exp_sin[i,{k+1}] * ({_jl(r)})'
        for k, r in enumerate(final_reduced[n_cos:]))

    L += [_EVAL_SIG, _TRIG_BLOCK,
          '    @inbounds for i in 1:N',
          '        x, y, z = X[i], Y[i], Z[i]',
          cse_block,
          '        _sum_cos = 0.0',
          '        _sum_sin = 0.0',
          cos_lines or '        # no cos terms',
          sin_lines or '        # no sin terms',
          '        out[i] = cos_p[i] * _sum_cos + sin_p[i] * _sum_sin',
          '    end', 'end', '']
    L += _main_func(n_cos, n_sin)

    return '\n'.join(L), len(kept), ops_post


# ---------------------------------------------------------------------------
# One threshold sweep
# ---------------------------------------------------------------------------

def run_threshold_sweep(terms_cos, terms_sin, thresholds, repl, reduced, counts,
                        label_prefix, work_dir, X, Y, Z, k_vals, b_vals,
                        outdir, julia_exe, n_reps, save_jl, reference_out):
    """Time selective_cse_build for each threshold; return results dict."""
    results = {}
    for min_count in thresholds:
        print(f'\n{"="*60}')
        print(f'[{label_prefix}] min_count={min_count}', flush=True)

        t_build = time.time()
        jl_src, n_kept, ops_post = selective_cse_build(
            terms_cos, terms_sin, min_count, repl, reduced, counts,
            strategy_label=f'{label_prefix}(min_count={min_count})')
        t_build = time.time() - t_build

        jl_file = outdir / f'eval_{label_prefix}_t{min_count}_m8.jl'
        jl_file.write_text(jl_src)
        print(f'  n_kept={n_kept}  ops_post={ops_post}  '
              f'.jl={len(jl_src):,} bytes  build={t_build:.1f}s')

        try:
            timing, raw_out = run_julia(
                jl_file, X, Y, Z, k_vals, b_vals, work_dir, julia_exe,
                n_reps=n_reps)
        except RuntimeError as exc:
            print(f'  JULIA FAILED: {exc}')
            results[f'{label_prefix}_t{min_count}'] = {
                'error': str(exc), 'min_count': min_count,
                'label': f'{label_prefix}_t{min_count}',
            }
            if not save_jl:
                jl_file.unlink(missing_ok=True)
            continue

        mspw = timing['ms_per_wave']
        print(f'  warmup={timing["warmup_s"]*1000:.1f}ms  '
              f'eval={mspw:.3f}ms/wave')

        max_err = None
        if reference_out is not None and raw_out is not None:
            max_err = float(np.abs(raw_out - reference_out).max())
            status = 'OK' if max_err < 1e-6 else 'WARNING'
            print(f'  vs baseline: max|diff|={max_err:.2e}  [{status}]')

        results[f'{label_prefix}_t{min_count}'] = {
            'label': f'{label_prefix}_t{min_count}',
            'min_count': min_count,
            'n_kept': n_kept, 'ops_post': ops_post,
            'ms_per_wave': mspw,
            'warmup_ms': timing['warmup_s'] * 1000,
            'max_err': max_err, 'build_time_s': t_build,
        }

        if not save_jl:
            jl_file.unlink(missing_ok=True)

    return results


# ---------------------------------------------------------------------------
# QD-cube helpers  (used when --cube_dir is given instead of HO cache)
# ---------------------------------------------------------------------------

def _parse_cube_info_txt(cube_dir):
    """Read cube_info.txt → dict with 'idx' (i,j,k), 'cube_size', 'a', 'b'."""
    info_path = Path(cube_dir) / 'cube_info.txt'
    if not info_path.exists():
        return {}
    result = {}
    with open(info_path, encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                result[k.strip().lower().replace(' ', '_')] = v.strip()
    if 'cube_index' in result:
        m = re.match(r'\((\d+),\s*(\d+),\s*(\d+)\)', result['cube_index'])
        if m:
            result['idx'] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if 'cube_size' in result:
        try:
            result['cube_size'] = float(result['cube_size'].split()[0])
        except (ValueError, IndexError):
            pass
    for key in ('a', 'b', 'atoms'):
        if key in result:
            try:
                result[key] = float(result[key].split()[0])
            except (ValueError, IndexError):
                result[key] = None
    return result


def _infer_ab_for_cube(cube_dir, m):
    """Read (a, b) from cube_info.txt or parse from an existing .jl filename."""
    info = _parse_cube_info_txt(cube_dir)
    a_raw = info.get('a');  b_raw = info.get('b')
    try:
        if a_raw not in (None, 'N/A') and b_raw not in (None, 'N/A'):
            return float(a_raw), float(b_raw)
    except (TypeError, ValueError):
        pass

    for jl_path in sorted(Path(cube_dir).glob(f'eval_filter_m{m}_*.jl')):
        parts = jl_path.stem.split('_')
        m_idx = next((i for i, p in enumerate(parts) if re.match(r'^m\d+$', p)), None)
        if m_idx is not None and m_idx + 2 < len(parts):
            try:
                a = float(parts[m_idx + 1].replace('p', '.').replace('m', '-'))
                b = float(parts[m_idx + 2].replace('p', '.').replace('m', '-'))
                return a, b
            except ValueError:
                pass
    return None, None


def prepare_from_qd_cube(cube_dir, m, a, b, expand=False):
    """Load H^n pkl files from cube_dir; return (terms_cos, terms_sin) pre-Horner.

    Same assembly path as run_qd_r11.py Stage 4, up to group_by_exp_combined.
    terms_cos / terms_sin are flat (pre-Horner) polynomial groups, matching
    the format expected by build_baseline() and selective_cse_build().
    """
    print(f'  Loading H^0..{m} pkl from {Path(cube_dir).name} ...',
          end=' ', flush=True)
    t0 = time.time()
    if not expand:
        expr_cos_raw, expr_sin_raw = apply_f_of_H_from_raw_powers(
            str(cube_dir), m, a=a, b=b, file_type='pkl', return_envelopes=True)
        expr_cos = sp.expand(expr_cos_raw)
        expr_sin = sp.expand(expr_sin_raw)
    else:
        psi_fH = apply_f_of_H_from_raw_powers(
            str(cube_dir), m, a=a, b=b, file_type='pkl')
        expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)

    terms_cos = group_by_exp_combined(expr_cos)
    terms_sin = group_by_exp_combined(expr_sin)
    print(f'done ({time.time()-t0:.1f}s)  '
          f'cos_groups={len(terms_cos)}  sin_groups={len(terms_sin)}')
    return terms_cos, terms_sin


def _cube_interior_grid(cube_dir, L_s, d_grid):
    """Return flat (Xf, Yf, Zf) arrays for the interior points of the cube.

    Reads cube_info.txt for the cube index and cube_size, then replicates
    the same grid-assignment logic as run_filter_diag / benchmark_qd_julia.
    """
    info = _parse_cube_info_txt(cube_dir)
    if 'idx' not in info or 'cube_size' not in info:
        raise RuntimeError(
            f"cube_info.txt missing idx or cube_size in {cube_dir}")

    ci, cj, ck = info['idx']
    cube_size   = info['cube_size']
    N_DIVISIONS = max(1, round(2 * L_s / cube_size))
    Ng          = int(np.ceil(2 * L_s / d_grid))
    x1 = np.linspace(-L_s, L_s, Ng, endpoint=False)
    X3, Y3, Z3 = np.meshgrid(x1, x1, x1, indexing='ij')

    ix_all  = np.arange(Ng, dtype=np.int32)
    cube_ax = np.minimum((ix_all * N_DIVISIONS) // Ng, N_DIVISIONS - 1)

    ix_c = np.where(cube_ax == ci)[0]
    iy_c = np.where(cube_ax == cj)[0]
    iz_c = np.where(cube_ax == ck)[0]
    IXc, IYc, IZc = np.meshgrid(ix_c, iy_c, iz_c, indexing='ij')
    flat_idx = (IXc * Ng * Ng + IYc * Ng + IZc).ravel()

    print(f'  Cube interior: {ci},{cj},{ck}  N_DIVISIONS={N_DIVISIONS}  '
          f'Ng={Ng}  n_pts={len(flat_idx)}')
    return X3.ravel()[flat_idx], Y3.ravel()[flat_idx], Z3.ravel()[flat_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Test CSE occurrence-count thresholds for f(H)*psi Julia eval.\n'
                    'Supports both 3-D HO (default) and real QD cube (--cube_dir).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--m', type=int, default=8)
    # HO-mode parameters (ignored when --cube_dir is given)
    parser.add_argument('--E_lo', type=float, default=4.5,
                        help='HO mode: lower energy (default 4.5). '
                             'QD mode: auto-read from .jl file (override with --E_lo 0)')
    parser.add_argument('--E_hi', type=float, default=None)
    parser.add_argument('--L', type=float, default=5.5,
                        help='HO mode: half-box size (default 5.5)')
    parser.add_argument('--Ng', type=int, default=22,
                        help='HO mode: grid points per axis (default 22)')
    # QD-cube mode parameters
    parser.add_argument('--cube_dir', default=None,
                        help='QD mode: path to cube subdirectory containing '
                             'H_power_n.pkl files. Activates QD mode.')
    parser.add_argument('--L_s', type=float, default=22.0,
                        help='QD mode: half-box size in Bohr (default 22.0)')
    parser.add_argument('--d_grid', type=float, default=0.625,
                        help='QD mode: grid spacing in Bohr (default 0.625)')
    parser.add_argument('--expand', action='store_true', default=False,
                        help='QD mode: use expanded pkl path (default False)')
    # Common parameters
    parser.add_argument('--n_waves', type=int, default=20,
                        help='Number of plane waves for Julia timing (default 20; >= 10)')
    parser.add_argument('--n_reps', type=int, default=1,
                        help='Julia timing repetitions (default 1)')
    parser.add_argument('--thresholds', default='1,2,3,5,8,10,15,20,30,50,100,200',
                        help='Comma-separated min_count thresholds to test')
    parser.add_argument('--mode', default='both',
                        choices=['flat', 'horner', 'both'],
                        help='Which CSE strategy to sweep')
    parser.add_argument('--cache_dir', default='ho3d_h_powers_cache',
                        help='HO mode: H^n cache directory (default ho3d_h_powers_cache)')
    parser.add_argument('--julia_exe', default='julia')
    parser.add_argument('--distribution_only', action='store_true',
                        help='Only print CSE count distributions; skip Julia runs')
    parser.add_argument('--save_jl', action='store_true',
                        help='Keep generated .jl files in outdir')
    parser.add_argument('--outdir', default='benchmark_results',
                        help='Output directory (default benchmark_results/)')
    args = parser.parse_args()

    if args.n_waves < 2 and not args.distribution_only:
        print('WARNING: n_waves < 2 → eval_s = 0, timings meaningless. '
              'Use --n_waves 20.\n')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    thresholds = sorted(int(t) for t in args.thresholds.split(','))
    do_flat   = args.mode in ('flat', 'both')
    do_horner = args.mode in ('horner', 'both')

    # ---- Detect mode and build symbolic expressions ----
    qd_mode = args.cube_dir is not None

    if qd_mode:
        # --- QD cube mode: load H^n pkl ---
        cube_dir = Path(args.cube_dir)
        if not cube_dir.exists():
            print(f'ERROR: --cube_dir not found: {cube_dir}')
            sys.exit(1)

        a, b = _infer_ab_for_cube(cube_dir, args.m)
        if a is None:
            if args.E_hi is not None:
                E_hi_qd = args.E_hi
            else:
                dx_qd = 2 * args.L_s / int(np.ceil(2 * args.L_s / args.d_grid))
                E_hi_qd = 0.5 * 3 * (np.pi / dx_qd)**2 + 12.0
            E_lo_qd = args.E_lo if args.E_lo != 4.5 else 0.0  # QD default E_lo=0
            a = 2.0 / (E_hi_qd - E_lo_qd)
            b = -(E_hi_qd + E_lo_qd) / (E_hi_qd - E_lo_qd)
            print(f'  Filter a/b estimated from grid: a={a:.8g}  b={b:.8g}')
        else:
            E_hi_qd = (1.0 - b) / a
            E_lo_qd = (-1.0 - b) / a
            print(f'  Filter: E_lo={E_lo_qd:.4f}  E_hi={E_hi_qd:.4f}  '
                  f'a={a:.8g}  b={b:.8g}')

        terms_cos, terms_sin = prepare_from_qd_cube(
            cube_dir, args.m, a, b, expand=args.expand)

        # Build interior grid (flat 1D arrays)
        Xf, Yf, Zf = _cube_interior_grid(cube_dir, args.L_s, args.d_grid)
        X, Y, Z = Xf, Yf, Zf
        E_hi = E_hi_qd
        mode_tag = f'qd_{cube_dir.name}'

    else:
        # --- HO mode: existing behaviour ---
        dx   = 2 * args.L / args.Ng
        E_hi = args.E_hi or (0.5*3*(np.pi/dx)**2 + 0.5*3*(args.L-dx)**2)
        print(f'E_hi = {E_hi:.4f}  (auto)\n')

        terms_cos, terms_sin, _ = prepare_expressions(
            args.m, args.E_lo, E_hi, args.cache_dir, expand=True)

        x1 = np.linspace(-args.L, args.L, args.Ng, endpoint=False)
        X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
        mode_tag = 'ho3d'

    # ---- Flat-poly CSE ----
    repl = reduced = counts = None
    if do_flat:
        all_polys = [pp for _, pp in terms_cos] + [pp for _, pp in terms_sin]
        print('Running sp.cse() on flat poly ...', flush=True)
        t0 = time.time()
        repl, reduced = sp.cse(all_polys, symbols=sp.numbered_symbols('_s'))
        print(f'  done in {time.time()-t0:.1f}s  ({len(repl)} replacements)')

        print('Counting occurrences per flat-CSE temp ...', flush=True)
        t0 = time.time()
        counts = count_cse_occurrences(repl, reduced)
        print(f'  done in {time.time()-t0:.1f}s')

        show_count_distribution(counts, thresholds,
                                title='Flat-poly CSE  (CSE on pre-Horner poly)')

    # ---- Horner-first CSE ----
    repl_h = reduced_h = counts_h = None
    terms_cos_h = terms_sin_h = None
    if do_horner:
        print('\nApplying Horner to all groups (same as baseline) ...', flush=True)
        t0 = time.time()
        terms_cos_h = apply_horner(terms_cos)
        terms_sin_h = apply_horner(terms_sin)
        print(f'  done in {time.time()-t0:.1f}s  '
              f'({len(terms_cos_h)} cos groups, {len(terms_sin_h)} sin groups)')

        all_horner_polys = ([pp for _, pp in terms_cos_h]
                            + [pp for _, pp in terms_sin_h])
        print(f'Running sp.cse() on Horner polys ({len(all_horner_polys)} polys) ...',
              flush=True)
        t0 = time.time()
        repl_h, reduced_h = sp.cse(all_horner_polys,
                                    symbols=sp.numbered_symbols('_h'))
        print(f'  done in {time.time()-t0:.1f}s  ({len(repl_h)} replacements)')

        print('Counting occurrences per Horner-CSE temp ...', flush=True)
        t0 = time.time()
        counts_h = count_cse_occurrences(repl_h, reduced_h)
        print(f'  done in {time.time()-t0:.1f}s')

        show_count_distribution(counts_h, thresholds,
                                title='Horner-first CSE  (CSE on Horner poly = "baseline+CSE")')

    if args.distribution_only:
        dist_file = outdir / f'cse_distribution_{mode_tag}_m{args.m}.json'
        payload = {'args': {'m': args.m, 'E_hi': E_hi,
                            'thresholds': thresholds, 'mode': args.mode,
                            'source': 'qd_cube' if qd_mode else 'ho3d'}}
        if do_flat and counts is not None:
            flat_vals = list(counts.values())
            payload['flat_cse'] = {
                'total_temps': len(flat_vals),
                'count_histogram': dict(Counter(flat_vals)),
                'threshold_n_kept': {str(t): sum(1 for v in flat_vals if v >= t)
                                     for t in thresholds},
            }
        if do_horner and counts_h is not None:
            h_vals = list(counts_h.values())
            payload['horner_cse'] = {
                'total_temps': len(h_vals),
                'count_histogram': dict(Counter(h_vals)),
                'threshold_n_kept': {str(t): sum(1 for v in h_vals if v >= t)
                                     for t in thresholds},
            }
        with open(dist_file, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\nDistribution → {dist_file}')
        return

    # ---- k-vectors ----
    rng    = np.random.default_rng(42)
    k_vals = rng.uniform(-1.0, 1.0, (args.n_waves, 3))
    b_vals = rng.uniform(0, 2*np.pi, args.n_waves)

    results = {}
    reference_out = None

    with tempfile.TemporaryDirectory(prefix='bench_thresh_') as work_dir:

        # ---- Baseline (Horner, no CSE) ----
        print(f'\n{"="*60}')
        print('Baseline: Horner (no explicit CSE)')
        jl_src, _, _, extra = build_baseline(terms_cos, terms_sin)
        jl_file = outdir / f'eval_baseline_m{args.m}.jl'
        jl_file.write_text(jl_src)
        print(f'  ops_post={extra["ops_post"]}  '
              f'.jl size={len(jl_src):,} bytes')
        timing, raw_out = run_julia(jl_file, X, Y, Z, k_vals, b_vals, work_dir,
                                    args.julia_exe, n_reps=args.n_reps)
        mspw = timing['ms_per_wave']
        print(f'  warmup={timing["warmup_s"]*1000:.1f}ms  eval={mspw:.3f}ms/wave')
        reference_out = raw_out.copy() if raw_out is not None else None
        results['baseline_horner'] = {
            'label': 'baseline_horner', 'min_count': None,
            'n_kept': None, 'ops_post': extra['ops_post'],
            'ms_per_wave': mspw,
            'warmup_ms': timing['warmup_s'] * 1000,
            'max_err': 0.0,
        }
        if not args.save_jl:
            jl_file.unlink(missing_ok=True)

        # ---- Flat-CSE threshold sweep ----
        if do_flat:
            print(f'\n{"─"*60}')
            print('Flat-poly CSE threshold sweep  (CSE on pre-Horner poly)')
            flat_results = run_threshold_sweep(
                terms_cos, terms_sin, thresholds,
                repl, reduced, counts,
                label_prefix='cse_t',
                work_dir=work_dir,
                X=X, Y=Y, Z=Z, k_vals=k_vals, b_vals=b_vals,
                outdir=outdir, julia_exe=args.julia_exe,
                n_reps=args.n_reps, save_jl=args.save_jl,
                reference_out=reference_out)
            results.update(flat_results)

        # ---- Horner-CSE threshold sweep ----
        if do_horner:
            print(f'\n{"─"*60}')
            print('Horner-first CSE threshold sweep  ("baseline + CSE")')
            horner_results = run_threshold_sweep(
                terms_cos, terms_sin, thresholds,
                repl_h, reduced_h, counts_h,
                label_prefix='hcse_t',
                work_dir=work_dir,
                X=X, Y=Y, Z=Z, k_vals=k_vals, b_vals=b_vals,
                outdir=outdir, julia_exe=args.julia_exe,
                n_reps=args.n_reps, save_jl=args.save_jl,
                reference_out=reference_out)
            results.update(horner_results)

    # ---- Summary table ----
    ref_ms = results.get('baseline_horner', {}).get('ms_per_wave')

    print(f'\n{"="*76}')
    src_tag = f'cube={args.cube_dir}' if qd_mode else f'E_lo={args.E_lo}  Ng={args.Ng}'
    print(f'SUMMARY  m={args.m}  E_hi={E_hi:.2f}  {src_tag}  '
          f'n_waves={args.n_waves}  mode={args.mode}')
    print(f'{"="*76}')

    header = (f'{"strategy":<22} {"type":<10} {"min_cnt":>8} {"n_kept":>7} '
              f'{"ops_post":>9} {"ms/wave":>9} {"vs_ref":>10} {"warmup_ms":>10}')
    print(header)
    print('-' * len(header))

    def _row(r, type_label):
        if 'error' in r:
            print(f'{r.get("label","?"):<22}  ERROR: {r["error"][:30]}')
            return
        mspw   = r.get('ms_per_wave')
        n_kept = r.get('n_kept', '-')
        min_c  = r.get('min_count', '-')
        ops_p  = r.get('ops_post', '-')
        wmup   = f'{r["warmup_ms"]:.0f}' if r.get('warmup_ms') else '-'
        if ref_ms and mspw:
            ratio  = ref_ms / mspw
            vs_ref = f'{ratio:.2f}x' if ratio >= 0.1 else f'1/{1/ratio:.0f}x'
        else:
            vs_ref = '-'
        print(f'{r["label"]:<22} {type_label:<10} {str(min_c):>8} {str(n_kept):>7} '
              f'{str(ops_p):>9} {(mspw or 0):>9.3f} {vs_ref:>10} {wmup:>10}')

    _row(results['baseline_horner'], 'baseline')

    if do_flat:
        print(f'{"─── flat-CSE (pre-Horner) ─"*2}')
        for key in results:
            if key.startswith('cse_t'):
                _row(results[key], 'flat_cse')

    if do_horner:
        print(f'{"─── horner-CSE (post-Horner) ─"*2}')
        for key in results:
            if key.startswith('hcse_t'):
                _row(results[key], 'horner_cse')

    # ---- Save JSON ----
    result_file = outdir / f'cse_threshold_{mode_tag}_m{args.m}.json'
    payload = {
        'args': {
            'M': args.m, 'E_hi': E_hi,
            'source': 'qd_cube' if qd_mode else 'ho3d',
            'cube_dir': str(args.cube_dir) if qd_mode else None,
            'n_waves': args.n_waves, 'n_reps': args.n_reps,
            'thresholds': thresholds, 'mode': args.mode,
        },
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'error'}
                    for k, v in results.items() if 'error' not in v},
    }
    if do_flat and counts is not None:
        flat_vals = list(counts.values())
        payload['flat_cse_info'] = {
            'total_temps': len(repl),
            'count_histogram': dict(sorted(Counter(flat_vals).items())),
            'threshold_n_kept': {str(t): sum(1 for v in flat_vals if v >= t)
                                 for t in thresholds},
        }
    if do_horner and counts_h is not None:
        h_vals = list(counts_h.values())
        payload['horner_cse_info'] = {
            'total_temps': len(repl_h),
            'count_histogram': dict(sorted(Counter(h_vals).items())),
            'threshold_n_kept': {str(t): sum(1 for v in h_vals if v >= t)
                                 for t in thresholds},
        }

    with open(result_file, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'\nResults → {result_file}')
    if args.save_jl:
        print(f'.jl files → {outdir}/')


if __name__ == '__main__':
    main()
