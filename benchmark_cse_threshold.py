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
from symbolic_code.chebyshev_filter import apply_horner


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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Test CSE occurrence-count thresholds for f(H)*psi Julia eval.'
    )
    parser.add_argument('--m', type=int, default=8)
    parser.add_argument('--E_lo', type=float, default=4.5)
    parser.add_argument('--E_hi', type=float, default=None)
    parser.add_argument('--L', type=float, default=5.5)
    parser.add_argument('--Ng', type=int, default=22)
    parser.add_argument('--n_waves', type=int, default=20,
                        help='Number of plane waves for Julia timing (default 20; >= 10)')
    parser.add_argument('--n_reps', type=int, default=1,
                        help='Julia timing repetitions (default 1)')
    parser.add_argument('--thresholds', default='1,2,3,5,8,10,15,20,30,50,100,200',
                        help='Comma-separated min_count thresholds to test '
                             '(default: 1,2,3,5,8,10,15,20,30,50,100,200)')
    parser.add_argument('--mode', default='both',
                        choices=['flat', 'horner', 'both'],
                        help='Which CSE strategy to sweep: '
                             'flat=CSE on flat poly (existing), '
                             'horner=CSE on Horner poly (new, "baseline+CSE"), '
                             'both=run both (default)')
    parser.add_argument('--cache_dir', default='ho3d_h_powers_cache')
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

    dx   = 2 * args.L / args.Ng
    E_hi = args.E_hi or (0.5*3*(np.pi/dx)**2 + 0.5*3*(args.L-dx)**2)
    print(f'E_hi = {E_hi:.4f}  (auto)\n')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    thresholds = sorted(int(t) for t in args.thresholds.split(','))
    do_flat   = args.mode in ('flat', 'both')
    do_horner = args.mode in ('horner', 'both')

    # ---- Build symbolic expressions ----
    terms_cos, terms_sin, _ = prepare_expressions(
        args.m, args.E_lo, E_hi, args.cache_dir, expand=True)

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
        dist_file = outdir / f'cse_distribution_m{args.m}.json'
        payload = {'args': {'m': args.m, 'E_lo': args.E_lo, 'E_hi': E_hi,
                            'thresholds': thresholds, 'mode': args.mode}}
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

    # ---- Grid and k-vectors ----
    x1 = np.linspace(-args.L, args.L, args.Ng, endpoint=False)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
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
    print(f'SUMMARY  m={args.m}  E_lo={args.E_lo}  E_hi={E_hi:.2f}  '
          f'Ng={args.Ng}  n_waves={args.n_waves}  mode={args.mode}')
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
    result_file = outdir / f'cse_threshold_m{args.m}.json'
    payload = {
        'args': {
            'M': args.m, 'E_lo': args.E_lo, 'E_hi': E_hi,
            'Ng': args.Ng, 'n_waves': args.n_waves, 'n_reps': args.n_reps,
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
