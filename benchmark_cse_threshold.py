#!/usr/bin/env python3
"""benchmark_cse_threshold.py
Test CSE occurrence-count thresholds to find the optimal register-pressure / FLOP tradeoff.

Background
----------
sp.cse() creates a replacement temp _si = expr_i whenever expr_i appears
more than once in the input polynomials.  The "occurrence count" of _si is
how many times it appears in subsequent expressions (later temps + reduced
expressions).  All temps have count >= 1.

Keeping ALL 2330 temps produces register pressure (CPU has ~16-32 float
registers) → stack spills → factor-350x slowdown vs Horner.

A threshold T keeps only temps with count >= T, inlining the rest.
This trades fewer local variables (less register pressure) against some
repeated computations (the inlined expressions are evaluated wherever they
were used).

Expected outcome
----------------
  T = 1     : keep everything → same as cse_inlined, ~220ms/wave
  T = 2-5   : inline "chain" temps (used once) → fewer vars, still large
  T = ~20+  : keep only genuinely shared sub-monomials → O(100) vars
  T = large : almost nothing kept → degenerates toward no_horner / flat poly

The sweet spot is where n_kept drops below ~30-50 so all temps fit in
registers without spilling.

Usage
-----
  python benchmark_cse_threshold.py --m 8 --n_waves 20
  python benchmark_cse_threshold.py --m 8 --n_waves 20 --thresholds 2,5,10,30,100
  python benchmark_cse_threshold.py --m 8 --distribution_only   # just show histogram
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


def show_count_distribution(counts, thresholds=None):
    """Print a text histogram of CSE occurrence counts and a threshold table."""
    vals = list(counts.values())
    hist = Counter(vals)
    total = len(vals)

    if thresholds is None:
        thresholds = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000]

    print(f"\n{'─'*55}")
    print(f"CSE occurrence-count distribution  ({total} total temps)")
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
        # bar width proportional to n/total
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
        # Rough savings estimate: each inlined temp is computed
        # (count) times instead of 1 → extra ops = (count - 1) * cost
        # We don't have cost here, so just show count sum
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
        # Resolve any previously inlined syms that appear in this expression
        expr_resolved = expr.xreplace(inline_map) if inline_map else expr

        if counts.get(sym, 0) < min_count:
            inline_map[sym] = expr_resolved   # inline: substitute everywhere it's used
        else:
            kept.append((sym, expr_resolved))  # keep: emit as Julia local variable

    # Apply inline substitutions to the final output expressions
    final_reduced = ([r.xreplace(inline_map) for r in reduced]
                     if inline_map else list(reduced))

    # Post-CSE op count: ops in kept-temp assignments + ops in final exprs
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
    parser.add_argument('--cache_dir', default='ho3d_h_powers_cache')
    parser.add_argument('--julia_exe', default='julia')
    parser.add_argument('--distribution_only', action='store_true',
                        help='Only print CSE count distribution; skip Julia runs')
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

    # ---- Build symbolic expressions ----
    terms_cos, terms_sin, _ = prepare_expressions(
        args.m, args.E_lo, E_hi, args.cache_dir, expand=True)

    all_polys = [pp for _, pp in terms_cos] + [pp for _, pp in terms_sin]

    print('Running sp.cse() on flat poly ...', flush=True)
    t0 = time.time()
    repl, reduced = sp.cse(all_polys, symbols=sp.numbered_symbols('_s'))
    print(f'  done in {time.time()-t0:.1f}s  ({len(repl)} replacements)', flush=True)

    print('Counting occurrences per temp (may take ~30-60s) ...', flush=True)
    t0 = time.time()
    counts = count_cse_occurrences(repl, reduced)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    show_count_distribution(counts, thresholds)

    if args.distribution_only:
        # Save distribution JSON even when not running Julia
        dist_vals = list(counts.values())
        dist_file = outdir / f'cse_distribution_m{args.m}.json'
        with open(dist_file, 'w') as f:
            json.dump({
                'total_temps': len(dist_vals),
                'count_histogram': dict(Counter(dist_vals)),
                'threshold_analysis': {
                    str(t): sum(1 for v in dist_vals if v >= t)
                    for t in thresholds
                },
            }, f, indent=2)
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

        # ---- Baseline (Horner) ----
        print(f'\n{"="*60}')
        print('Baseline: Horner')
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
            'max_err': 0.0
        }
        if not args.save_jl:
            jl_file.unlink(missing_ok=True)

        # ---- Each threshold ----
        for min_count in thresholds:
            print(f'\n{"="*60}')
            print(f'min_count={min_count}', flush=True)

            t_build = time.time()
            jl_src, n_kept, ops_post = selective_cse_build(
                terms_cos, terms_sin, min_count, repl, reduced, counts)
            t_build = time.time() - t_build

            jl_file = outdir / f'eval_cse_t{min_count}_m{args.m}.jl'
            jl_file.write_text(jl_src)
            print(f'  n_kept={n_kept}  ops_post={ops_post}  '
                  f'.jl={len(jl_src):,} bytes  build={t_build:.1f}s')

            try:
                timing, raw_out = run_julia(
                    jl_file, X, Y, Z, k_vals, b_vals, work_dir,
                    args.julia_exe, n_reps=args.n_reps)
            except RuntimeError as exc:
                print(f'  JULIA FAILED: {exc}')
                results[f'cse_t{min_count}'] = {'error': str(exc), 'min_count': min_count}
                if not args.save_jl:
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

            results[f'cse_t{min_count}'] = {
                'label': f'cse_t{min_count}', 'min_count': min_count,
                'n_kept': n_kept, 'ops_post': ops_post,
                'ms_per_wave': mspw,
                'warmup_ms': timing['warmup_s'] * 1000,
                'max_err': max_err, 'build_time_s': t_build,
            }

            if not args.save_jl:
                jl_file.unlink(missing_ok=True)

    # ---- Summary table ----
    ref_ms = results.get('baseline_horner', {}).get('ms_per_wave')

    print(f'\n{"="*72}')
    print(f'SUMMARY  m={args.m}  E_lo={args.E_lo}  E_hi={E_hi:.2f}  '
          f'Ng={args.Ng}  n_waves={args.n_waves}')
    print(f'{"="*72}')

    header = (f'{"strategy":<20} {"min_cnt":>8} {"n_kept":>7} {"ops_post":>9} '
              f'{"ms/wave":>9} {"vs_ref":>10} {"warmup_ms":>10}')
    print(header)
    print('-' * len(header))

    for r in results.values():
        if 'error' in r:
            print(f'{r.get("label","?"):<20}  ERROR: {r["error"][:30]}')
            continue
        mspw    = r.get('ms_per_wave')
        n_kept  = r.get('n_kept', '-')
        min_c   = r.get('min_count', '-')
        ops_p   = r.get('ops_post', '-')
        wmup    = f'{r["warmup_ms"]:.0f}' if r.get('warmup_ms') else '-'
        if ref_ms and mspw:
            ratio  = ref_ms / mspw
            vs_ref = f'{ratio:.2f}x' if ratio >= 0.1 else f'1/{1/ratio:.0f}x'
        else:
            vs_ref = '-'
        print(f'{r["label"]:<20} {str(min_c):>8} {str(n_kept):>7} {str(ops_p):>9} '
              f'{(mspw or 0):>9.3f} {vs_ref:>10} {wmup:>10}')

    # ---- Save JSON ----
    dist_vals = list(counts.values())
    result_file = outdir / f'cse_threshold_m{args.m}.json'
    with open(result_file, 'w') as f:
        json.dump({
            'args': {
                'M': args.m, 'E_lo': args.E_lo, 'E_hi': E_hi,
                'Ng': args.Ng, 'n_waves': args.n_waves, 'n_reps': args.n_reps,
                'thresholds': thresholds,
            },
            'cse_total_temps': len(repl),
            'count_histogram': dict(sorted(Counter(dist_vals).items())),
            'threshold_n_kept': {
                str(t): sum(1 for v in dist_vals if v >= t) for t in thresholds
            },
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'error'}
                        for k, v in results.items() if 'error' not in v},
        }, f, indent=2, default=str)
    print(f'\nResults → {result_file}')
    if args.save_jl:
        print(f'.jl files → {outdir}/')


if __name__ == '__main__':
    main()
