#!/usr/bin/env python3
import argparse, json, pickle, tempfile
from pathlib import Path
import sympy as sp
import numpy as np

from benchmark_strategies import run_julia, _jl, build_baseline
from benchmark_hn_cse import selective_hn_cse_build
from benchmark_cse_threshold import count_cse_occurrences
from symbolic_code.chebyshev_filter import group_by_exp_combined, apply_horner

# Function signature for H^n eval (no exp_cos/exp_sin buffers needed)
EVAL_SIG = '''\
function eval_one_wave!(out::AbstractVector{Float64},
                        X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64},
                        kx::Float64, ky::Float64, kz::Float64, b::Float64,
                        phase::Vector{Float64}, cos_p::Vector{Float64},
                        sin_p::Vector{Float64})'''

TRIG = '''\
    @. phase = kx * X + ky * Y + kz * Z + b
    @. cos_p = cos(phase)
    @. sin_p = sin(phase)
    N = length(X)'''


def main_func():
    return [
        'function main()',
        '    length(ARGS) == 5 || error(',
        '        "Usage: julia script.jl grid.bin kvals.bin out.bin N n_waves")',
        '    N = parse(Int, ARGS[4]); n_waves = parse(Int, ARGS[5])',
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


def build_raw(expr_c, expr_s):
    """Strategy 1: direct SymPy tree -> Julia (no CSE, no Horner)."""
    ops = int(sp.count_ops(expr_c)) + int(sp.count_ops(expr_s))
    L = ['# strategy: raw_tree', EVAL_SIG, TRIG,
         '    @inbounds for i in 1:N',
         '        x, y, z = X[i], Y[i], Z[i]',
         f'        out[i] = cos_p[i] * ({_jl(expr_c)}) + sin_p[i] * ({_jl(expr_s)})',
         '    end', 'end', ''] + main_func()
    return '\n'.join(L), ops


def build_horner_inloop(expr_c, expr_s):
    """Strategy 3: expand + group_by_exp + Horner, inline in loop body (no precompute)."""
    tc = apply_horner(group_by_exp_combined(sp.expand(expr_c)))
    ts = apply_horner(group_by_exp_combined(sp.expand(expr_s)))

    # Sequential += accumulation avoids a single giant joined expression (JIT-friendly)
    cos_lines = '\n'.join(
        f'        _sum_cos += ({_jl(ep)}) * ({_jl(pp)})' for ep, pp in tc
    ) or '        # no cos terms'
    sin_lines = '\n'.join(
        f'        _sum_sin += ({_jl(ep)}) * ({_jl(pp)})' for ep, pp in ts
    ) or '        # no sin terms'

    ops = sum(int(sp.count_ops(pp)) + int(sp.count_ops(ep)) for ep, pp in tc + ts)
    L = ['# strategy: horner_inloop', EVAL_SIG, TRIG,
         '    @inbounds for i in 1:N',
         '        x, y, z = X[i], Y[i], Z[i]',
         '        _sum_cos = 0.0',
         '        _sum_sin = 0.0',
         cos_lines,
         sin_lines,
         '        out[i] = cos_p[i] * _sum_cos + sin_p[i] * _sum_sin',
         '    end', 'end', ''] + main_func()
    return '\n'.join(L), ops


def load_expr(cube_dir, n):
    with open(Path(cube_dir) / f'H_power_{n}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data.get('Pc', sp.Integer(0)), data.get('Ps', sp.Integer(0))


def run(args):
    expr_c, expr_s = load_expr(args.cube_dir, args.n_select)
    strategies = {}

    # Strategy 1: raw SymPy tree -> Julia
    jl1, ops1 = build_raw(expr_c, expr_s)
    strategies['1_raw_tree_no_cse'] = (jl1, ops1)

    # Strategy 2: raw SymPy tree + CSE (keep temps used > 20 times)
    repl, (redc, reds) = sp.cse([expr_c, expr_s], symbols=sp.numbered_symbols('_t'))
    # fix: count_cse_occurrences requires both repl and the reduced expressions
    counts = count_cse_occurrences(repl, [redc, reds])
    jl2, nk, ops2 = selective_hn_cse_build(21, repl, redc, reds, counts, 'raw_tree_cse_gt20')
    strategies['2_raw_tree_cse_gt20'] = (jl2, ops2)

    # Strategy 3: expand + group_by_exp + Horner, inlined in loop (no precompute buffers)
    jl3, ops3 = build_horner_inloop(expr_c, expr_s)
    strategies['3_horner_inloop'] = (jl3, ops3)

    # Strategies 4, 5, 6: build_baseline (group_by_exp + Horner + precomputed exp buffers)
    # fix: build_baseline calls apply_horner internally, so do NOT pre-apply it here
    # fix: capture 4-tuple return to get post-optimisation ops_post
    jl5, _tc5, _ts5, extra5 = build_baseline(
        group_by_exp_combined(sp.expand(expr_c)),
        group_by_exp_combined(sp.expand(expr_s)),
    )
    ops5 = extra5['ops_post']
    # 4 and 6 currently reuse the same baseline codegen path as 5
    strategies['4_horner_precompute_xyz_terms'] = (jl5, ops5)
    strategies['5_horner_precompute_exp_only']  = (jl5, ops5)
    strategies['6_current_baseline']            = (jl5, ops5)

    out = {'cube_dir': str(args.cube_dir), 'n_select': args.n_select, 'results': {}}

    if args.no_timing:
        for k, (_, ops) in strategies.items():
            out['results'][k] = {'Nop': int(ops)}
    else:
        rng    = np.random.default_rng(args.seed)
        N      = args.n_pts
        X      = rng.uniform(-1, 1, size=N)
        Y      = rng.uniform(-1, 1, size=N)
        Z      = rng.uniform(-1, 1, size=N)
        k_vals = rng.normal(size=(args.n_waves, 3))
        b_vals = rng.uniform(-np.pi, np.pi, size=args.n_waves)

        with tempfile.TemporaryDirectory(prefix='hn_methods_') as td:
            for k, (jl, ops) in strategies.items():
                p = Path(td) / f'{k}.jl'
                p.write_text(jl)
                timing, _ = run_julia(
                    p, X, Y, Z, k_vals, b_vals, Path(td),
                    args.julia, n_reps=args.n_reps)
                # fix: warmup_s can be None if Julia JSON parsing fails
                warmup_ms = timing['warmup_s'] * 1000 if timing['warmup_s'] is not None else None
                out['results'][k] = {
                    'Nop': int(ops),
                    'ms_per_wave': timing['ms_per_wave'],
                    'warmup_ms': warmup_ms,
                }

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f'wrote {args.out_json}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cube_dir', required=True)
    ap.add_argument('--n_select', type=int, default=1)
    ap.add_argument('--out_json', default='hn_method_benchmark.json')
    ap.add_argument('--no_timing', action='store_true')
    ap.add_argument('--n_pts', type=int, default=4000)
    ap.add_argument('--n_waves', type=int, default=20)
    ap.add_argument('--n_reps', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--julia', default='julia')
    run(ap.parse_args())
