#!/usr/bin/env python3
"""benchmark_hn_methods.py

Compare six Julia code-generation strategies for evaluating H^n plane-wave
envelopes P_c(r;k) and P_s(r;k) defined by:

    H^n e^{i(k·r+b)} = P_c(r;k) cos(k·r+b) + P_s(r;k) sin(k·r+b)

Strategies
----------
1  raw_tree_no_cse          : direct SymPy tree → Julia, no CSE, no Horner
2  raw_tree_cse_gt20        : raw tree + global CSE, keep temps used ≥ 21 times
3  horner_inloop            : expand + group-by-exp + Horner, all inline (no precompute)
4  horner_precompute_xyz    : expand + group-by-exp, decompose by k-monomial,
                              precompute G_j(r)*c_{j,m}(r) for each k-monomial m
5  horner_precompute_exp    : expand + group-by-exp + Horner, precompute only G_j(r)
                              (= exp-only precompute; same code path as build_baseline)
6  current_baseline         : build_baseline from benchmark_strategies.py
                              (identical to strategy 5; listed separately for reference)

Strategy 4 design
-----------------
Group P_c = Σ_j G_j(r) Q_j(k;r) by exp factor, then expand each Q_j as a
polynomial in (kx,ky,kz):  Q_j = Σ_m c_{j,m}(r) k^m.

Collect by k-monomial:  S_m(r) = Σ_j G_j(r) c_{j,m}(r)  (xyz-only expression).

Precompute S_m(r_i) for all i and all k-monomials m into matrix xyz_cos[N, n_monomials].

eval_one_wave!: per-wave, evaluate k-monomial scalars k^m once, then inner loop is
    out[i] = (Σ_m  xyz_cos[i,m] * k^m) * cos_p[i] + (…) * sin_p[i]
No xyz computation at all in the hot loop.

Strategies 5 and 6
-------------------
Both use build_baseline (precompute G_j matrix, Horner polynomial per wave).
They are identical in implementation; strategy 6 is the production baseline label.

Usage
-----
python benchmark_hn_methods.py \\
    --cube_dir QD_R17_Julia_exp/custom_l3p0_rcut4p2/cube_neg15.944_neg11.389_6.833 \\
    --n_select 1,2,3,5,8,10,12 \\
    --n_pts 100 --n_waves 50 --n_reps 3 \\
    --out_json hn_methods.json

# No Julia timing (just op counts):
python benchmark_hn_methods.py ... --no_timing
"""

import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import sympy as sp

from benchmark_strategies import run_julia, _jl, build_baseline
from benchmark_hn_cse import selective_hn_cse_build
from benchmark_cse_threshold import count_cse_occurrences
from symbolic_code.chebyshev_filter import group_by_exp_combined, apply_horner


# ---------------------------------------------------------------------------
# Julia code helpers shared across strategies 1-3
# ---------------------------------------------------------------------------

_EVAL_SIG_PLAIN = '''\
function eval_one_wave!(out::AbstractVector{Float64},
                        X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64},
                        kx::Float64, ky::Float64, kz::Float64, b::Float64,
                        phase::Vector{Float64}, cos_p::Vector{Float64},
                        sin_p::Vector{Float64})'''

_TRIG_PLAIN = '''\
    @. phase = kx * X + ky * Y + kz * Z + b
    @. cos_p = cos(phase)
    @. sin_p = sin(phase)
    N = length(X)'''


def _plain_main():
    """main() for strategies 1-3 (no precompute matrix)."""
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


# ---------------------------------------------------------------------------
# Strategy 1: raw SymPy tree, no CSE, no Horner
# ---------------------------------------------------------------------------

def build_raw(expr_c, expr_s):
    """Direct SymPy tree → Julia; no CSE, no Horner."""
    ops = int(sp.count_ops(expr_c)) + int(sp.count_ops(expr_s))
    L = ['# strategy: raw_tree_no_cse', _EVAL_SIG_PLAIN, _TRIG_PLAIN,
         '    @inbounds for i in 1:N',
         '        x, y, z = X[i], Y[i], Z[i]',
         f'        out[i] = cos_p[i] * ({_jl(expr_c)}) + sin_p[i] * ({_jl(expr_s)})',
         '    end', 'end', ''] + _plain_main()
    return '\n'.join(L), ops


# ---------------------------------------------------------------------------
# Strategy 2: raw tree + selective CSE (threshold T = 21)
# ---------------------------------------------------------------------------

def build_cse_gt20(expr_c, expr_s):
    """Global CSE on [P_c, P_s]; keep temps used ≥ 21 times."""
    repl, (redc, reds) = sp.cse([expr_c, expr_s], symbols=sp.numbered_symbols('_t'))
    counts = count_cse_occurrences(repl, [redc, reds])
    jl, _nk, ops = selective_hn_cse_build(21, repl, redc, reds, counts,
                                           strategy_label='raw_tree_cse_gt20')
    return jl, ops


# ---------------------------------------------------------------------------
# Strategy 3: expand + group-by-exp + Horner, fully inline (no precompute)
# ---------------------------------------------------------------------------

def build_horner_inloop(expr_c, expr_s):
    """Horner in-loop: both G_j(r) and Q_j(k;r) computed per grid point."""
    tc = apply_horner(group_by_exp_combined(sp.expand(expr_c)))
    ts = apply_horner(group_by_exp_combined(sp.expand(expr_s)))

    cos_lines = '\n'.join(
        f'        _sum_cos += ({_jl(ep)}) * ({_jl(pp)})' for ep, pp in tc
    ) or '        # no cos terms'
    sin_lines = '\n'.join(
        f'        _sum_sin += ({_jl(ep)}) * ({_jl(pp)})' for ep, pp in ts
    ) or '        # no sin terms'

    ops = sum(int(sp.count_ops(pp)) + int(sp.count_ops(ep)) for ep, pp in tc + ts)
    L = ['# strategy: horner_inloop', _EVAL_SIG_PLAIN, _TRIG_PLAIN,
         '    @inbounds for i in 1:N',
         '        x, y, z = X[i], Y[i], Z[i]',
         '        _sum_cos = 0.0',
         '        _sum_sin = 0.0',
         cos_lines,
         sin_lines,
         '        out[i] = cos_p[i] * _sum_cos + sin_p[i] * _sum_sin',
         '    end', 'end', ''] + _plain_main()
    return '\n'.join(L), ops


# ---------------------------------------------------------------------------
# Strategy 4: group-by-exp, decompose by k-monomial, precompute all xyz terms
# ---------------------------------------------------------------------------

def _k_mono_jl(d1, d2, d3):
    """Julia expression for kx^d1 * ky^d2 * kz^d3."""
    parts = []
    for sym, d in [('kx', d1), ('ky', d2), ('kz', d3)]:
        if d == 1:
            parts.append(sym)
        elif d > 1:
            parts.append(f'{sym} ^ {d}')
    return ' * '.join(parts) if parts else '1.0'


def _decompose_by_k_monom(terms):
    """Group (G_j, Q_j) pairs by k-monomial.

    For each group j, expands Q_j as a polynomial in (kx,ky,kz) and
    accumulates  S_m(r) = Σ_j  G_j(r) * c_{j,m}(r)  for each monomial m.

    Returns sorted list of  (k_monom_tuple, S_m_expr)  pairs.
    """
    kx_s, ky_s, kz_s = sp.symbols('kx ky kz')
    monom_map = {}
    for G_j, Q_j in terms:
        try:
            q_poly = sp.Poly(sp.expand(Q_j), kx_s, ky_s, kz_s)
            for monom, coeff in zip(q_poly.monoms(), q_poly.coeffs()):
                spatial = G_j * coeff
                monom_map[monom] = monom_map.get(monom, sp.Integer(0)) + spatial
        except (sp.PolynomialError, sp.GeneratorsNeeded):
            # Q_j not a polynomial in k — treat as k^0 term
            m = (0, 0, 0)
            monom_map[m] = monom_map.get(m, sp.Integer(0)) + G_j * Q_j
    return sorted(monom_map.items())   # [(monom, spatial_expr), ...]


def build_precompute_xyz(terms_cos, terms_sin):
    """Strategy 4: precompute S_m(r_i) = Σ_j G_j * c_{j,m} for each k-monomial.

    eval_one_wave! inner loop:
        out[i] = (Σ_m  xyz_cos[i,m] * km_val_m) * cos_p[i] + … * sin_p[i]
    where km_val_m = kx^d1*ky^d2*kz^d3 are scalars precomputed once per wave.

    ops_post counts only what is inside eval_one_wave! (excludes precompute).
    """
    cos_items = _decompose_by_k_monom(terms_cos)   # [(monom, S_m), ...]
    sin_items = _decompose_by_k_monom(terms_sin)

    n_cos = len(cos_items)
    n_sin = len(sin_items)
    nc, ns = max(n_cos, 1), max(n_sin, 1)

    # ops_post: per-wave ops in eval_one_wave! only
    unique_monoms = sorted({m for m, _ in cos_items + sin_items})
    # k-monomial scalar evaluations (once per wave, before the loop)
    ops_k = sum(max(d1 + d2 + d3 - 1, 0) for d1, d2, d3 in unique_monoms)
    # inner loop: n_cos multiply-adds for cos, n_sin for sin, + 2 final muls
    ops_inner = (2 * n_cos - 1 if n_cos > 0 else 0) + \
                (2 * n_sin - 1 if n_sin > 0 else 0) + 2
    ops_post = ops_k + ops_inner

    L = ['# strategy: horner_precompute_xyz_terms']

    # @inline functions for S_m(r) — one per (cos, k-monomial) and (sin, k-monomial)
    for idx, (monom, spatial) in enumerate(cos_items):
        L += [f'@inline function _xyz_cos_{idx}'
              f'(x::Float64, y::Float64, z::Float64)::Float64',
              f'    return {_jl(spatial)}', 'end', '']
    for idx, (monom, spatial) in enumerate(sin_items):
        L += [f'@inline function _xyz_sin_{idx}'
              f'(x::Float64, y::Float64, z::Float64)::Float64',
              f'    return {_jl(spatial)}', 'end', '']

    # precompute_xyz! — fills matrix once before wave loop
    L += [
        'function precompute_xyz!(xyz_cos::Matrix{Float64}, xyz_sin::Matrix{Float64},',
        '                         X::Vector{Float64}, Y::Vector{Float64},'
        ' Z::Vector{Float64})',
        '    N = length(X)',
        '    @inbounds for i in 1:N',
    ]
    for idx in range(n_cos):
        L.append(f'        xyz_cos[i,{idx+1}] = _xyz_cos_{idx}(X[i], Y[i], Z[i])')
    for idx in range(n_sin):
        L.append(f'        xyz_sin[i,{idx+1}] = _xyz_sin_{idx}(X[i], Y[i], Z[i])')
    L += ['    end', 'end', '']

    # Per-wave k-monomial scalars (computed once before the grid loop)
    monom_var = {}
    k_pre_lines = []
    for d1, d2, d3 in unique_monoms:
        if d1 == 0 and d2 == 0 and d3 == 0:
            monom_var[(d1, d2, d3)] = '1.0'
        else:
            var = f'_km_{d1}_{d2}_{d3}'
            monom_var[(d1, d2, d3)] = var
            k_pre_lines.append(f'    {var} = {_k_mono_jl(d1, d2, d3)}')
    k_block = '\n'.join(k_pre_lines) if k_pre_lines else '    # (k^0 only)'

    _EVAL_SIG_4 = '''\
function eval_one_wave!(out::AbstractVector{Float64},
                        X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64},
                        kx::Float64, ky::Float64, kz::Float64, b::Float64,
                        xyz_cos::Matrix{Float64}, xyz_sin::Matrix{Float64},
                        phase::Vector{Float64}, cos_p::Vector{Float64},
                        sin_p::Vector{Float64})'''

    _TRIG_4 = '''\
    @. phase = kx * X + ky * Y + kz * Z + b
    @. cos_p = cos(phase)
    @. sin_p = sin(phase)
    N = length(X)'''

    cos_sum = ' + '.join(
        f'xyz_cos[i,{idx+1}] * {monom_var[m]}'
        for idx, (m, _) in enumerate(cos_items)
    ) or '0.0'
    sin_sum = ' + '.join(
        f'xyz_sin[i,{idx+1}] * {monom_var[m]}'
        for idx, (m, _) in enumerate(sin_items)
    ) or '0.0'

    L += [_EVAL_SIG_4, k_block, _TRIG_4,
          '    @inbounds for i in 1:N',
          f'        out[i] = ({cos_sum}) * cos_p[i] + ({sin_sum}) * sin_p[i]',
          '    end', 'end', '']

    # main() with xyz precompute matrix
    L += [
        'function main()',
        '    length(ARGS) == 5 || error(',
        '        "Usage: julia script.jl grid.bin kvals.bin out.bin N n_waves")',
        '    N = parse(Int, ARGS[4]); n_waves = parse(Int, ARGS[5])',
        '    buf = Vector{Float64}(undef, 3 * N)',
        '    open(ARGS[1], "r") do io; read!(io, buf); end',
        '    X = buf[1:N]; Y = buf[N+1:2N]; Z = buf[2N+1:3N]',
        '    kb = Vector{Float64}(undef, 4 * n_waves)',
        '    open(ARGS[2], "r") do io; read!(io, kb); end',
        '    out     = Vector{Float64}(undef, N * n_waves)',
        f'    xyz_cos = ones(Float64, N, {nc})',
        f'    xyz_sin = ones(Float64, N, {ns})',
        '    precompute_xyz!(xyz_cos, xyz_sin, X, Y, Z)',
        '    phase   = Vector{Float64}(undef, N)',
        '    cos_p   = Vector{Float64}(undef, N)',
        '    sin_p   = Vector{Float64}(undef, N)',
        '    t_warmup = @elapsed eval_one_wave!(',
        '        @view(out[1:N]), X, Y, Z, kb[1], kb[2], kb[3], kb[4],',
        '        xyz_cos, xyz_sin, phase, cos_p, sin_p)',
        '    t_eval = @elapsed for iw in 1:n_waves-1',
        '        ofs = iw * N + 1; kid = iw * 4 + 1',
        '        eval_one_wave!(@view(out[ofs:ofs+N-1]), X, Y, Z,',
        '                       kb[kid], kb[kid+1], kb[kid+2], kb[kid+3],',
        '                       xyz_cos, xyz_sin, phase, cos_p, sin_p)',
        '    end',
        '    open(ARGS[3], "w") do io; write(io, out); end',
        '    n_eval = max(n_waves - 1, 1)',
        '    println("{\\\"warmup_s\\\": $t_warmup, \\\"eval_s\\\": $t_eval,'
        ' \\\"n_eval_waves\\\": $n_eval}")',
        'end',
        '',
        'main()',
    ]

    return '\n'.join(L), ops_post


# ---------------------------------------------------------------------------
# Strategy 5 & 6: group-by-exp + Horner, precompute only G_j (= build_baseline)
# ---------------------------------------------------------------------------

def build_precompute_exp(terms_cos, terms_sin):
    """Strategy 5: precompute only G_j(r_i) (Gaussian envelope).

    Wraps build_baseline from benchmark_strategies.py.
    Q_j(k;r) is evaluated in Horner form inside eval_one_wave! per wave.
    ops_post counts only the Horner polynomial phase (excludes precompute).
    """
    jl, tc, ts, extra = build_baseline(terms_cos, terms_sin)
    return jl, extra['ops_post']


# ---------------------------------------------------------------------------
# Expression loader
# ---------------------------------------------------------------------------

def load_expr(cube_dir, n):
    """Load H^n envelope pair (Pc, Ps) from H_power_{n}.pkl."""
    with open(Path(cube_dir) / f'H_power_{n}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data.get('Pc', sp.Integer(0)), data.get('Ps', sp.Integer(0))


# ---------------------------------------------------------------------------
# Per-order benchmark
# ---------------------------------------------------------------------------

def run_one_n(cube_dir, n, args):
    """Build and time all six strategies for a single H^n order."""
    expr_c, expr_s = load_expr(cube_dir, n)
    terms_cos = group_by_exp_combined(sp.expand(expr_c))
    terms_sin = group_by_exp_combined(sp.expand(expr_s))

    strategies = {}

    # Strategy 1
    jl1, ops1 = build_raw(expr_c, expr_s)
    strategies['1_raw_tree_no_cse'] = (jl1, ops1)

    # Strategy 2
    jl2, ops2 = build_cse_gt20(expr_c, expr_s)
    strategies['2_raw_tree_cse_gt20'] = (jl2, ops2)

    # Strategy 3
    jl3, ops3 = build_horner_inloop(expr_c, expr_s)
    strategies['3_horner_inloop'] = (jl3, ops3)

    # Strategy 4: precompute all xyz terms (G_j * c_{j,m}(r) per k-monomial)
    jl4, ops4 = build_precompute_xyz(terms_cos, terms_sin)
    strategies['4_horner_precompute_xyz_terms'] = (jl4, ops4)

    # Strategy 5: precompute only G_j (Gaussian), Horner for Q_j per wave
    jl5, ops5 = build_precompute_exp(terms_cos, terms_sin)
    strategies['5_horner_precompute_exp_only'] = (jl5, ops5)

    # Strategy 6: current_baseline (identical code path to strategy 5)
    jl6, ops6 = build_precompute_exp(terms_cos, terms_sin)
    strategies['6_current_baseline'] = (jl6, ops6)

    results = {}

    if args.no_timing:
        for k, (_, ops) in strategies.items():
            results[k] = {'Nop': int(ops)}
        return results

    rng    = np.random.default_rng(args.seed)
    N      = args.n_pts
    X      = rng.uniform(-1, 1, size=N)
    Y      = rng.uniform(-1, 1, size=N)
    Z      = rng.uniform(-1, 1, size=N)
    k_vals = rng.normal(size=(args.n_waves, 3))
    b_vals = rng.uniform(-np.pi, np.pi, size=args.n_waves)

    with tempfile.TemporaryDirectory(prefix='hn_methods_') as td:
        for k, (jl, ops) in strategies.items():
            jl_path = Path(td) / f'{k}.jl'
            jl_path.write_text(jl)
            if args.save_jl:
                out_jl = Path(args.save_jl) / f'n{n}_{k}.jl'
                out_jl.parent.mkdir(parents=True, exist_ok=True)
                out_jl.write_text(jl)

            try:
                timing, _ = run_julia(
                    jl_path, X, Y, Z, k_vals, b_vals, Path(td),
                    args.julia, n_reps=args.n_reps)
                warmup_ms = (timing['warmup_s'] * 1000
                             if timing['warmup_s'] is not None else None)
                results[k] = {
                    'Nop': int(ops),
                    'ms_per_wave': timing['ms_per_wave'],
                    'warmup_ms': warmup_ms,
                }
            except RuntimeError as exc:
                results[k] = {'Nop': int(ops), 'error': str(exc)}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Benchmark 6 Julia code-generation strategies for H^n envelopes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--cube_dir', required=True,
                    help='Directory containing H_power_n.pkl files')
    ap.add_argument('--n_select', default='1',
                    help='Comma-separated list of H^n orders to benchmark '
                         '(e.g. 1,2,3,5,8,10,12)')
    ap.add_argument('--out_json', default='hn_method_benchmark.json')
    ap.add_argument('--no_timing', action='store_true',
                    help='Only report op counts, skip Julia timing')
    ap.add_argument('--n_pts',   type=int, default=100,
                    help='Grid points for Julia timing (default 100)')
    ap.add_argument('--n_waves', type=int, default=50,
                    help='Plane waves for Julia timing (default 50)')
    ap.add_argument('--n_reps',  type=int, default=3,
                    help='Julia timing repetitions (default 3)')
    ap.add_argument('--seed',    type=int, default=0)
    ap.add_argument('--julia',   default='julia')
    ap.add_argument('--save_jl', default=None, metavar='DIR',
                    help='If set, save generated .jl files to this directory')
    args = ap.parse_args()

    n_list = sorted(int(x) for x in args.n_select.split(','))
    cube_dir = Path(args.cube_dir)

    out = {
        'cube_dir': str(cube_dir),
        'n_select': n_list,
        'results_by_n': {},
    }

    for n in n_list:
        pkl = cube_dir / f'H_power_{n}.pkl'
        if not pkl.exists():
            print(f'[n={n}] H_power_{n}.pkl not found — skipping')
            continue
        print(f'\n{"="*60}')
        print(f'n = {n}')
        print(f'{"="*60}')

        results = run_one_n(cube_dir, n, args)
        out['results_by_n'][str(n)] = {'n_select': n, 'results': results}

        # Print summary row
        header = f'  {"strategy":<32} {"Nop":>7}  {"ms/wave":>9}  {"warmup_ms":>10}'
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for k, r in results.items():
            if 'error' in r:
                print(f'  {k:<32}  ERROR: {r["error"][:40]}')
                continue
            mspw = r.get('ms_per_wave')
            wmup = r.get('warmup_ms')
            mspw_s = f'{mspw:.4f}' if mspw is not None else '-'
            wmup_s = f'{wmup:.2f}'  if wmup  is not None else '-'
            print(f'  {k:<32} {r["Nop"]:>7}  {mspw_s:>9}  {wmup_s:>10}')

    return out


def run(args):
    out = {'cube_dir': str(args.cube_dir), 'n_select': args.n_select, 'results_by_n': {}}
    for n in args.n_select:
        out['results_by_n'][str(n)] = benchmark_for_n(args.cube_dir, n, args)

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f'\nwrote {args.out_json}')


if __name__ == '__main__':
    main()
