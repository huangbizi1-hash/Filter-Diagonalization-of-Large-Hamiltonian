#!/usr/bin/env python3
"""benchmark_hn_cse.py
Find the optimal CSE threshold for the H^n Julia evaluation code.

Unlike benchmark_cse_threshold.py (which operates on the Chebyshev filter
f(H)·ψ = Σ c_k H^k·ψ), this script operates on a single H^n power loaded
directly from H_power_{n1}.pkl without any Chebyshev combination.

The unexpanded SymPy expression is fed to sp.cse().  Each CSE temp is
counted by how many times it is reused.  The threshold sweep keeps only
temps with reuse count >= T and inlines the rest, then times Julia.

Usage
-----
  python benchmark_hn_cse.py \\
      --vexpr_dir QD_R17_Vexpr_partition_uniform \\
      --expr_dir  QD_R17_Julia_exp/partition_uniform_no_expansion \\
      --n1 3 --n_atoms 5 --n_waves 20

  # distribution only (no Julia):
  python benchmark_hn_cse.py ... --distribution_only

  # custom thresholds:
  python benchmark_hn_cse.py ... --thresholds 1,2,3,5,10,20
"""

import argparse
import json
import pickle
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_strategies import _jl, run_julia
from benchmark_cse_threshold import count_cse_occurrences, show_count_distribution
from symbolic_code.expr_loader import CubicExpressionManager


# ---------------------------------------------------------------------------
# Julia codegen for H^n CSE (no grouped exp factors)
# ---------------------------------------------------------------------------

_HN_EVAL_SIG = '''\
function eval_one_wave!(out::AbstractVector{Float64},
                        X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64},
                        kx::Float64, ky::Float64, kz::Float64, b::Float64,
                        phase::Vector{Float64}, cos_p::Vector{Float64},
                        sin_p::Vector{Float64})'''

_HN_TRIG_BLOCK = '''\
    @. phase = kx * X + ky * Y + kz * Z + b
    @. cos_p = cos(phase)
    @. sin_p = sin(phase)
    N = length(X)'''


def _hn_main_func():
    return [
        'function main()',
        '    length(ARGS) == 5 || error(',
        '        "Usage: julia script.jl grid.bin kvals.bin out.bin N n_waves")',
        '    N       = parse(Int, ARGS[4])',
        '    n_waves = parse(Int, ARGS[5])',
        '    buf = Vector{Float64}(undef, 3 * N)',
        '    open(ARGS[1], "r") do io; read!(io, buf); end',
        '    X = buf[1:N]; Y = buf[N+1:2N]; Z = buf[2N+1:3N]',
        '    kb = Vector{Float64}(undef, 4 * n_waves)',
        '    open(ARGS[2], "r") do io; read!(io, kb); end',
        '    out   = Vector{Float64}(undef, N * n_waves)',
        '    phase = Vector{Float64}(undef, N)',
        '    cos_p = Vector{Float64}(undef, N)',
        '    sin_p = Vector{Float64}(undef, N)',
        '    t_warmup = @elapsed eval_one_wave!(',
        '        @view(out[1:N]), X, Y, Z, kb[1], kb[2], kb[3], kb[4],',
        '        phase, cos_p, sin_p)',
        '    t_eval = @elapsed for iw in 1:n_waves-1',
        '        ofs = iw * N + 1; kid = iw * 4 + 1',
        '        eval_one_wave!(@view(out[ofs:ofs+N-1]), X, Y, Z,',
        '                       kb[kid], kb[kid+1], kb[kid+2], kb[kid+3],',
        '                       phase, cos_p, sin_p)',
        '    end',
        '    open(ARGS[3], "w") do io; write(io, out); end',
        '    n_eval = max(n_waves - 1, 1)',
        '    println("{\\\"warmup_s\\\": $t_warmup, \\\"eval_s\\\": $t_eval,'
        ' \\\"n_eval_waves\\\": $n_eval}")',
        'end',
        '',
        'main()',
    ]


def selective_hn_cse_build(min_count, repl, red_cos, red_sin, counts,
                            strategy_label=None, keep_scores=None):
    """Build H^n Julia keeping only CSE temps with occurrence count >= min_count.

    Same forward-pass inlining logic as selective_cse_build in
    benchmark_cse_threshold.py, but uses the flat H^n eval structure
    (no grouped Gaussian factor precompute).

    Returns (jl_src, n_kept, ops_post).
    """
    if strategy_label is None:
        strategy_label = f'hn_cse(min_count={min_count})'

    inline_map = {}
    kept = []
    for sym, expr in repl:
        expr_resolved = expr.xreplace(inline_map) if inline_map else expr
        score = keep_scores.get(sym, counts.get(sym, 0)) if keep_scores else counts.get(sym, 0)
        if score < min_count:
            inline_map[sym] = expr_resolved
        else:
            kept.append((sym, expr_resolved))

    final_red_cos = red_cos.xreplace(inline_map) if inline_map else red_cos
    final_red_sin = red_sin.xreplace(inline_map) if inline_map else red_sin

    ops_kept    = sum(int(sp.count_ops(e)) for _, e in kept)
    ops_reduced = int(sp.count_ops(final_red_cos)) + int(sp.count_ops(final_red_sin))
    ops_post    = ops_kept + ops_reduced

    temp_block = (
        '\n'.join(f'        {_jl(s)} = {_jl(e)}' for s, e in kept)
        if kept else '        # no CSE temps (all inlined)'
    )

    L = [f'# strategy: {strategy_label}']
    L += [_HN_EVAL_SIG, _HN_TRIG_BLOCK,
          '    @inbounds for i in 1:N',
          '        x, y, z = X[i], Y[i], Z[i]',
          temp_block,
          f'        out[i] = cos_p[i] * ({_jl(final_red_cos)})'
          f' + sin_p[i] * ({_jl(final_red_sin)})',
          '    end', 'end', '']
    L += _hn_main_func()

    return '\n'.join(L), len(kept), ops_post


# ---------------------------------------------------------------------------
# Cube selection
# ---------------------------------------------------------------------------

def _cube_dirname(cx, cy, cz):
    def fmt(v):
        return f'{v:.3f}'.replace('-', 'neg')
    return f'cube_{fmt(cx)}_{fmt(cy)}_{fmt(cz)}'


def find_cube_by_atoms(vexpr_dir, expr_dir, n_atoms, n1, seed=0):
    """Return a cube_dir Path with exactly n_atoms atoms that has H_power_{n1}.pkl."""
    manager = CubicExpressionManager(str(vexpr_dir))
    candidates = []
    for info in manager.cube_info.values():
        if info.get('n_atoms') != n_atoms:
            continue
        cx, cy, cz = info['center']
        cube_dir = Path(expr_dir) / _cube_dirname(cx, cy, cz)
        if (cube_dir / f'H_power_{n1}.pkl').exists():
            candidates.append(cube_dir)

    if not candidates:
        return None
    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(len(candidates)))]


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def run_hn_threshold_sweep(thresholds, repl, red_cos, red_sin, counts,
                            work_dir, X, Y, Z, k_vals, b_vals,
                            outdir, julia_exe, n_reps, save_jl,
                            reference_out, n1, keep_scores=None,
                            score_label='count'):
    """Sweep selective_hn_cse_build thresholds; return results dict."""
    results = {}
    for min_count in thresholds:
        print(f'\n{"="*60}')
        print(f'[hn_cse] min_count={min_count}', flush=True)

        t_build = time.time()
        jl_src, n_kept, ops_post = selective_hn_cse_build(
            min_count, repl, red_cos, red_sin, counts,
            strategy_label=f'hn_cse_t{min_count}_{score_label}',
            keep_scores=keep_scores)
        t_build = time.time() - t_build

        jl_file = outdir / f'eval_hn_cse_n{n1}_t{min_count}.jl'
        jl_file.write_text(jl_src)
        print(f'  n_kept={n_kept}  ops_post={ops_post}  '
              f'.jl={len(jl_src):,} bytes  build={t_build:.1f}s')

        try:
            timing, raw_out = run_julia(
                jl_file, X, Y, Z, k_vals, b_vals, work_dir, julia_exe,
                n_reps=n_reps)
        except RuntimeError as exc:
            print(f'  JULIA FAILED: {exc}')
            results[f'hn_cse_t{min_count}'] = {
                'error': str(exc), 'min_count': min_count,
                'label': f'hn_cse_t{min_count}',
            }
            if not save_jl:
                jl_file.unlink(missing_ok=True)
            continue

        mspw = timing['ms_per_wave']
        print(f'  warmup={timing["warmup_s"]*1000:.1f}ms  eval={mspw:.3f}ms/wave')

        max_err = None
        if reference_out is not None and raw_out is not None:
            diff = np.abs(raw_out - reference_out)
            n_nan = int(np.isnan(diff).sum())
            valid = ~np.isnan(diff)
            max_err = float(diff[valid].max()) if valid.any() else float('nan')
            status = 'OK' if (not np.isnan(max_err) and max_err < 1e-6) else 'WARNING'
            nan_str = f'  nan_pts={n_nan}/{diff.size}' if n_nan > 0 else ''
            print(f'  vs baseline: max|diff|={max_err:.2e}  [{status}]{nan_str}')

        results[f'hn_cse_t{min_count}'] = {
            'label': f'hn_cse_t{min_count}', 'min_count': min_count,
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
        description='Optimal CSE threshold search for H^n Julia evaluation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--vexpr_dir', required=True,
                        help='Path to vexpr pkl directory (for atom-count lookup)')
    parser.add_argument('--expr_dir', required=True,
                        help='Path to expr directory (contains cube_* subdirs with pkl)')
    parser.add_argument('--n1', type=int, required=True,
                        help='H^n1 order to benchmark (loads H_power_{n1}.pkl)')
    parser.add_argument('--n_atoms', type=int, default=5,
                        help='Target atom count for cube selection (default 5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for cube selection (default 0)')
    parser.add_argument('--n_waves', type=int, default=20,
                        help='Number of plane waves for Julia timing (default 20)')
    parser.add_argument('--n_reps', type=int, default=1,
                        help='Julia timing repetitions (default 1)')
    parser.add_argument('--thresholds', default='1,2,3,5,8,10,15,20,30,50,100,200',
                        help='Comma-separated min_count thresholds to sweep')
    parser.add_argument('--L_s', type=float, default=22.0,
                        help='Half-box size in Bohr for grid (default 22.0)')
    parser.add_argument('--d_grid', type=float, default=0.625,
                        help='Grid spacing in Bohr (default 0.625)')
    parser.add_argument('--julia_exe', default='julia')
    parser.add_argument('--distribution_only', action='store_true',
                        help='Only print CSE count distribution; skip Julia timing')
    parser.add_argument('--save_jl', action='store_true',
                        help='Keep generated .jl files in outdir')
    parser.add_argument('--outdir', default='benchmark_results_hn',
                        help='Output directory (default benchmark_results_hn/)')
    parser.add_argument('--score_mode', choices=['count', 'savings'], default='count',
                        help=('Threshold score mode: count keeps CSE temps by reuse count; '
                              'savings keeps by estimated net op savings = count*(ops-1)-1'))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    thresholds = sorted(int(t) for t in args.thresholds.split(','))

    # ---- Find cube ----
    print(f'Searching for cube with n_atoms={args.n_atoms} '
          f'and H_power_{args.n1}.pkl ...')
    cube_dir = find_cube_by_atoms(
        args.vexpr_dir, args.expr_dir, args.n_atoms, args.n1, seed=args.seed)
    if cube_dir is None:
        print(f'ERROR: no cube with n_atoms={args.n_atoms} '
              f'and H_power_{args.n1}.pkl found.')
        sys.exit(1)
    print(f'  Selected: {cube_dir.name}')

    # ---- Load H^n1 pkl and run CSE ----
    pkl_path = cube_dir / f'H_power_{args.n1}.pkl'
    print(f'Loading {pkl_path.name} ...', end=' ', flush=True)
    t0 = time.time()
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)
    expr_cos = data.get('Pc', sp.Integer(0))
    expr_sin = data.get('Ps', sp.Integer(0))
    print(f'done ({time.time()-t0:.1f}s)')

    print('Running sp.cse([Pc, Ps]) ...', end=' ', flush=True)
    t0 = time.time()
    repl, (red_cos, red_sin) = sp.cse(
        [expr_cos, expr_sin], symbols=sp.numbered_symbols('_h'))
    print(f'done ({time.time()-t0:.1f}s)  {len(repl)} replacements')

    print('Counting occurrences per CSE temp ...', end=' ', flush=True)
    t0 = time.time()
    counts = count_cse_occurrences(repl, [red_cos, red_sin])
    print(f'done ({time.time()-t0:.1f}s)')
    expr_ops = {sym: int(sp.count_ops(expr)) for sym, expr in repl}
    savings_scores = {
        sym: (counts.get(sym, 0) * max(expr_ops[sym] - 1, 0) - 1)
        for sym in expr_ops
    }

    score_map = counts if args.score_mode == 'count' else savings_scores
    score_title = 'reuse-count' if args.score_mode == 'count' else 'estimated net-op savings'
    show_count_distribution(
        score_map, thresholds,
        title=(f'H^{args.n1} CSE {score_title}  '
               f'(n_atoms={args.n_atoms}  cube={cube_dir.name})'))

    if args.distribution_only:
        dist_file = outdir / f'hn_cse_dist_n{args.n1}_atoms{args.n_atoms}.json'
        vals = list(counts.values())
        payload = {
            'n1': args.n1, 'n_atoms': args.n_atoms, 'cube': cube_dir.name,
            'total_temps': len(vals),
            'count_histogram': dict(Counter(vals)),
            'threshold_n_kept': {str(t): sum(1 for v in vals if v >= t)
                                  for t in thresholds},
        }
        with open(dist_file, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\nDistribution → {dist_file}')
        return

    # ---- Build flat grid inside cube (same as benchmark_cse_threshold.py) ----
    from benchmark_cse_threshold import _parse_cube_info_txt
    info = _parse_cube_info_txt(cube_dir)
    if 'idx' in info and 'cube_size' in info:
        from benchmark_cse_threshold import _cube_interior_grid
        X, Y, Z = _cube_interior_grid(cube_dir, args.L_s, args.d_grid)
        print(f'  Using cube interior: n_pts={X.size}')
    else:
        Ng = int(np.ceil(2 * args.L_s / args.d_grid))
        x1 = np.linspace(-args.L_s, args.L_s, Ng, endpoint=False)
        X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
        print(f'  No cube_info.txt — using full grid: Ng={Ng}^3={Ng**3:,} pts')

    rng    = np.random.default_rng(42)
    k_vals = rng.uniform(-1.0, 1.0, (args.n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, args.n_waves)

    results = {}

    with tempfile.TemporaryDirectory(prefix='bench_hn_cse_') as work_dir:

        # ---- Baseline: all CSE temps kept (min_count=1) ----
        print(f'\n{"="*60}')
        print('Baseline: all CSE temps kept (min_count=1)')
        t_build = time.time()
        jl_src_base, n_kept_base, ops_base = selective_hn_cse_build(
            1, repl, red_cos, red_sin, counts, strategy_label='hn_cse_baseline',
            keep_scores=score_map)
        t_build = time.time() - t_build
        jl_file = outdir / f'eval_hn_cse_n{args.n1}_baseline.jl'
        jl_file.write_text(jl_src_base)
        print(f'  n_kept={n_kept_base}  ops_post={ops_base}  '
              f'.jl={len(jl_src_base):,} bytes  build={t_build:.1f}s')
        timing, ref_out = run_julia(
            jl_file, X, Y, Z, k_vals, b_vals, work_dir,
            args.julia_exe, n_reps=args.n_reps)
        mspw_base = timing['ms_per_wave']
        print(f'  warmup={timing["warmup_s"]*1000:.1f}ms  eval={mspw_base:.3f}ms/wave')
        reference_out = ref_out.copy() if ref_out is not None else None
        results['baseline'] = {
            'label': 'baseline (T=1)', 'min_count': 1,
            'n_kept': n_kept_base, 'ops_post': ops_base,
            'ms_per_wave': mspw_base,
            'warmup_ms': timing['warmup_s'] * 1000, 'max_err': 0.0,
        }
        if not args.save_jl:
            jl_file.unlink(missing_ok=True)

        # ---- Threshold sweep (T >= 2) ----
        sweep_thresholds = [t for t in thresholds if t > 1]
        sweep_results = run_hn_threshold_sweep(
            sweep_thresholds, repl, red_cos, red_sin, counts,
            work_dir, X, Y, Z, k_vals, b_vals,
            outdir, args.julia_exe, args.n_reps, args.save_jl,
            reference_out, args.n1, keep_scores=score_map,
            score_label=args.score_mode)
        results.update(sweep_results)

    # ---- Summary table ----
    ref_ms = mspw_base
    print(f'\n{"="*76}')
    print(f'SUMMARY  H^{args.n1}  n_atoms={args.n_atoms}  '
          f'cube={cube_dir.name}  n_waves={args.n_waves}')
    print(f'{"="*76}')
    header = (f'{"strategy":<22} {"min_cnt":>8} {"n_kept":>7} '
              f'{"ops_post":>9} {"ms/wave":>9} {"vs_T1":>8} {"warmup_ms":>10}')
    print(header)
    print('-' * len(header))

    def _row(r):
        if 'error' in r:
            print(f'{r.get("label","?"):<22}  ERROR: {r["error"][:40]}')
            return
        mspw = r.get('ms_per_wave') or 0
        vs = f'{ref_ms/mspw:.2f}x' if mspw else '-'
        wmup = f'{r["warmup_ms"]:.0f}' if r.get('warmup_ms') else '-'
        print(f'{r["label"]:<22} {str(r.get("min_count","-")):>8} '
              f'{str(r.get("n_kept","-")):>7} {str(r.get("ops_post","-")):>9} '
              f'{mspw:>9.3f} {vs:>8} {wmup:>10}')

    for key in ['baseline'] + [f'hn_cse_t{t}' for t in sweep_thresholds]:
        if key in results:
            _row(results[key])

    # ---- Save JSON ----
    result_file = outdir / f'hn_cse_n{args.n1}_atoms{args.n_atoms}.json'
    vals = list(counts.values())
    payload = {
        'args': {'n1': args.n1, 'n_atoms': args.n_atoms, 'cube': cube_dir.name,
                 'n_waves': args.n_waves, 'thresholds': thresholds},
        'cse_info': {
            'total_temps': len(repl),
            'count_histogram': dict(sorted(Counter(vals).items())),
            'threshold_n_kept': {str(t): sum(1 for v in vals if v >= t)
                                  for t in thresholds},
        },
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'error'}
                    for k, v in results.items() if 'error' not in v},
    }
    with open(result_file, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'\nResults → {result_file}')


if __name__ == '__main__':
    main()
