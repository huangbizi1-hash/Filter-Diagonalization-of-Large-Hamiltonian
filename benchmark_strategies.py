#!/usr/bin/env python3
"""benchmark_strategies.py
Compare Julia evaluation strategies for f(H)*psi, with and without sp.expand().

Strategies
----------
baseline      group_by_exp + Horner + @inline per group  (current code)
no_horner     flat poly, no Horner, no CSE               (worst-case reference)
cse_group     group_by_exp + sp.cse() per group
cse_inlined   global sp.cse() across all groups, shared temps in @inbounds loop
k_separate    separate k-polynomial from spatial-polynomial; k-coefficients
              precomputed once per wave; inner loop is k-free

Expand modes (--compare_expand)
---------------------------------
Each strategy can be run on two expression sets:
  E  – standard: sp.expand() at every H^n step → flat monomial sum
  N  – noexpand: skip sp.expand() → preserves V*Ps, k2*Ps product structure

With --compare_expand, all requested strategies are timed on BOTH sets, and
the table shows rows like  baseline_E, baseline_N, cse_group_E, cse_group_N.

Why --n_waves matters
---------------------
The Julia script separates a warmup wave (triggers JIT compilation) from the
remaining n_waves-1 timed waves.  With n_waves=1 the timed section is empty
and eval_s=0 — all ms/wave values are 0.  Use --n_waves >= 20 for meaningful
timing; --n_waves 40 is recommended.

Why the ops column now shows post-optimisation counts
------------------------------------------------------
  baseline   : sp.count_ops of the Horner-transformed poly
  no_horner  : sp.count_ops of the flat expanded poly (no reduction)
  cse_group  : total ops in CSE temp assignments + reduced expressions
  cse_inlined: same, but across all groups jointly
  k_separate : spatial poly ops + k-coefficient ops per wave
Lower ops → fewer FLOPs per grid point per wave.

Usage
-----
  python benchmark_strategies.py                           # m=6, all strategies
  python benchmark_strategies.py --m 8 --n_waves 40
  python benchmark_strategies.py --m 8 --n_waves 40 --compare_expand
  python benchmark_strategies.py --strategies baseline,cse_group --n_waves 40
  python benchmark_strategies.py --save_jl                # keep .jl for inspection
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Julia code helper
# ---------------------------------------------------------------------------

def _jl(expr):
    s = sp.julia_code(expr)
    for op in ('.+', '.-', '.*', './', '.^'):
        s = s.replace(op, op[1:])
    return s


# ---------------------------------------------------------------------------
# Complexity metrics
# ---------------------------------------------------------------------------

def total_ops(terms_list):
    """Sum sp.count_ops over all poly parts (pre-optimisation baseline)."""
    return sum(int(sp.count_ops(pp)) for _, pp in terms_list)

def total_terms(terms_list):
    """Sum number of top-level summands over all poly parts."""
    return sum(len(sp.Add.make_args(pp)) for _, pp in terms_list)


# ---------------------------------------------------------------------------
# Shared Julia building blocks
# ---------------------------------------------------------------------------
_EVAL_SIG = '''\
function eval_one_wave!(out::AbstractVector{Float64},
                        X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64},
                        kx::Float64, ky::Float64, kz::Float64, b::Float64,
                        exp_cos::Matrix{Float64}, exp_sin::Matrix{Float64},
                        phase::Vector{Float64}, cos_p::Vector{Float64},
                        sin_p::Vector{Float64})'''

_TRIG_BLOCK = '''\
    @. phase = kx * X + ky * Y + kz * Z + b
    @. cos_p = cos(phase)
    @. sin_p = sin(phase)
    N = length(X)'''


def _exp_funcs(terms_cos, terms_sin):
    L = []
    for i, (ep, _) in enumerate(terms_cos):
        L += [f'@inline function _exp_cos_{i}(x::Float64, y::Float64, z::Float64)::Float64',
              f'    return {_jl(ep)}', 'end', '']
    for i, (ep, _) in enumerate(terms_sin):
        L += [f'@inline function _exp_sin_{i}(x::Float64, y::Float64, z::Float64)::Float64',
              f'    return {_jl(ep)}', 'end', '']
    return L


def _precompute_exp(n_cos, n_sin):
    L = ['function precompute_exp!(exp_cos::Matrix{Float64}, exp_sin::Matrix{Float64},',
         '                         X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64})',
         '    N = length(X)', '    @inbounds for i in 1:N']
    for i in range(n_cos):
        L.append(f'        exp_cos[i,{i+1}] = _exp_cos_{i}(X[i], Y[i], Z[i])')
    for i in range(n_sin):
        L.append(f'        exp_sin[i,{i+1}] = _exp_sin_{i}(X[i], Y[i], Z[i])')
    L += ['    end', 'end', '']
    return L


def _main_func(n_cos, n_sin):
    nc = max(n_cos, 1); ns = max(n_sin, 1)
    return [
        'function main()',
        '    length(ARGS) == 5 || error("Usage: julia script.jl grid.bin kvals.bin out.bin N n_waves")',
        '    grid_file = ARGS[1]; kvals_file = ARGS[2]; out_file = ARGS[3]',
        '    N = parse(Int, ARGS[4]); n_waves = parse(Int, ARGS[5])',
        '    buf = Vector{Float64}(undef, 3 * N)',
        '    open(grid_file, "r") do io; read!(io, buf); end',
        '    X = buf[1:N]; Y = buf[N+1:2N]; Z = buf[2N+1:3N]',
        '    kb = Vector{Float64}(undef, 4 * n_waves)',
        '    open(kvals_file, "r") do io; read!(io, kb); end',
        f'    exp_cos = ones(Float64, N, {nc}); exp_sin = ones(Float64, N, {ns})',
        '    precompute_exp!(exp_cos, exp_sin, X, Y, Z)',
        '    out = Vector{Float64}(undef, N * n_waves)',
        '    phase = Vector{Float64}(undef, N)',
        '    cos_p = Vector{Float64}(undef, N)',
        '    sin_p = Vector{Float64}(undef, N)',
        '    t_warmup = @elapsed eval_one_wave!(',
        '        @view(out[1:N]), X, Y, Z, kb[1], kb[2], kb[3], kb[4],',
        '        exp_cos, exp_sin, phase, cos_p, sin_p)',
        '    t_eval = @elapsed for iw in 1:n_waves-1',
        '        ofs = iw * N + 1; kid = iw * 4 + 1',
        '        eval_one_wave!(@view(out[ofs:ofs+N-1]), X, Y, Z,',
        '                       kb[kid], kb[kid+1], kb[kid+2], kb[kid+3],',
        '                       exp_cos, exp_sin, phase, cos_p, sin_p)',
        '    end',
        '    open(out_file, "w") do io; write(io, out); end',
        '    n_eval = max(n_waves - 1, 1)',
        '    println("{\\\"warmup_s\\\": $t_warmup, \\\"eval_s\\\": $t_eval,'
        ' \\\"n_eval_waves\\\": $n_eval}")',
        'end', '', 'main()',
    ]


# ---------------------------------------------------------------------------
# Strategy 1: baseline
# ---------------------------------------------------------------------------

def build_baseline(terms_cos, terms_sin):
    """group_by_exp + Horner + @inline per group."""
    from symbolic_code.chebyshev_filter import apply_horner
    tc = apply_horner(terms_cos)
    ts = apply_horner(terms_sin)

    n_cos, n_sin = len(tc), len(ts)
    L = ['# strategy: baseline']
    for i, (_, pp) in enumerate(tc):
        L += [f'@inline function _poly_cos_{i}(x::Float64, y::Float64, z::Float64,'
              f' kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}', 'end', '']
    for i, (_, pp) in enumerate(ts):
        L += [f'@inline function _poly_sin_{i}(x::Float64, y::Float64, z::Float64,'
              f' kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}', 'end', '']
    L += _exp_funcs(tc, ts) + _precompute_exp(n_cos, n_sin)

    cos_sum = ' + '.join(
        f'exp_cos[i,{i+1}] * _poly_cos_{i}(X[i], Y[i], Z[i], kx, ky, kz)'
        for i in range(n_cos)) or '0.0'
    sin_sum = ' + '.join(
        f'exp_sin[i,{i+1}] * _poly_sin_{i}(X[i], Y[i], Z[i], kx, ky, kz)'
        for i in range(n_sin)) or '0.0'

    L += [_EVAL_SIG, _TRIG_BLOCK,
          '    @inbounds for i in 1:N',
          f'        out[i] = ({cos_sum}) * cos_p[i] + ({sin_sum}) * sin_p[i]',
          '    end', 'end', '']
    L += _main_func(n_cos, n_sin)
    ops_post = total_ops(tc) + total_ops(ts)
    return '\n'.join(L), tc, ts, {'ops_post': ops_post}


# ---------------------------------------------------------------------------
# Strategy 2: no_horner
# ---------------------------------------------------------------------------

def build_no_horner(terms_cos, terms_sin):
    """Flat poly, no Horner, no CSE  (worst-case reference)."""
    n_cos, n_sin = len(terms_cos), len(terms_sin)
    L = ['# strategy: no_horner']
    for i, (_, pp) in enumerate(terms_cos):
        L += [f'@inline function _poly_cos_{i}(x::Float64, y::Float64, z::Float64,'
              f' kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}', 'end', '']
    for i, (_, pp) in enumerate(terms_sin):
        L += [f'@inline function _poly_sin_{i}(x::Float64, y::Float64, z::Float64,'
              f' kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}', 'end', '']
    L += _exp_funcs(terms_cos, terms_sin) + _precompute_exp(n_cos, n_sin)

    cos_sum = ' + '.join(
        f'exp_cos[i,{i+1}] * _poly_cos_{i}(X[i], Y[i], Z[i], kx, ky, kz)'
        for i in range(n_cos)) or '0.0'
    sin_sum = ' + '.join(
        f'exp_sin[i,{i+1}] * _poly_sin_{i}(X[i], Y[i], Z[i], kx, ky, kz)'
        for i in range(n_sin)) or '0.0'

    L += [_EVAL_SIG, _TRIG_BLOCK,
          '    @inbounds for i in 1:N',
          f'        out[i] = ({cos_sum}) * cos_p[i] + ({sin_sum}) * sin_p[i]',
          '    end', 'end', '']
    L += _main_func(n_cos, n_sin)
    ops_post = total_ops(terms_cos) + total_ops(terms_sin)
    return '\n'.join(L), terms_cos, terms_sin, {'ops_post': ops_post}


# ---------------------------------------------------------------------------
# Strategy 3: cse_group
# ---------------------------------------------------------------------------

def build_cse_group(terms_cos, terms_sin):
    """CSE per poly group; each @inline function gets local CSE temps.

    ops_post counts ops in CSE temp assignments + reduced expressions —
    the TRUE per-point FLOP count, not the pre-CSE flat-poly count.
    """
    n_cos, n_sin = len(terms_cos), len(terms_sin)
    L = ['# strategy: cse_group']
    total_cse = 0
    ops_post = 0

    def _poly_cse(name, pp):
        nonlocal total_cse, ops_post
        repl, (red,) = sp.cse([pp], symbols=sp.numbered_symbols('_s'))
        total_cse += len(repl)
        ops_post += sum(int(sp.count_ops(expr)) for _, expr in repl)
        ops_post += int(sp.count_ops(red))
        lines = [f'@inline function {name}(x::Float64, y::Float64, z::Float64,'
                 f' kx::Float64, ky::Float64, kz::Float64)::Float64']
        for sym, expr in repl:
            lines.append(f'    local {sym} = {_jl(expr)}')
        lines += [f'    return {_jl(red)}', 'end', '']
        return '\n'.join(lines)

    for i, (_, pp) in enumerate(terms_cos):
        L.append(_poly_cse(f'_poly_cos_{i}', pp))
    for i, (_, pp) in enumerate(terms_sin):
        L.append(_poly_cse(f'_poly_sin_{i}', pp))

    L += _exp_funcs(terms_cos, terms_sin) + _precompute_exp(n_cos, n_sin)

    cos_sum = ' + '.join(
        f'exp_cos[i,{i+1}] * _poly_cos_{i}(X[i], Y[i], Z[i], kx, ky, kz)'
        for i in range(n_cos)) or '0.0'
    sin_sum = ' + '.join(
        f'exp_sin[i,{i+1}] * _poly_sin_{i}(X[i], Y[i], Z[i], kx, ky, kz)'
        for i in range(n_sin)) or '0.0'

    L += [_EVAL_SIG, _TRIG_BLOCK,
          '    @inbounds for i in 1:N',
          f'        out[i] = ({cos_sum}) * cos_p[i] + ({sin_sum}) * sin_p[i]',
          '    end', 'end', '']
    L += _main_func(n_cos, n_sin)
    return '\n'.join(L), terms_cos, terms_sin, {
        'ops_post': ops_post, 'n_cse_temps': total_cse}


# ---------------------------------------------------------------------------
# Strategy 4: cse_inlined
# ---------------------------------------------------------------------------

def build_cse_inlined(terms_cos, terms_sin):
    """Global CSE across all groups; shared temps in a single @inbounds loop."""
    n_cos, n_sin = len(terms_cos), len(terms_sin)
    all_polys = [pp for _, pp in terms_cos] + [pp for _, pp in terms_sin]

    print('    [cse_inlined] running sp.cse() on all poly expressions ...',
          flush=True)
    t0 = time.time()
    repl, reduced = sp.cse(all_polys, symbols=sp.numbered_symbols('_s'))
    print(f'    [cse_inlined] done in {time.time()-t0:.1f}s  '
          f'({len(repl)} replacements)', flush=True)

    ops_post = (sum(int(sp.count_ops(expr)) for _, expr in repl)
                + sum(int(sp.count_ops(r)) for r in reduced))

    L = ['# strategy: cse_inlined']
    L += _exp_funcs(terms_cos, terms_sin) + _precompute_exp(n_cos, n_sin)

    cse_lines = [f'        {sym} = {_jl(expr)}' for sym, expr in repl]
    cse_block = '\n'.join(cse_lines) if cse_lines else '        # no shared CSE temps'

    cos_lines = '\n'.join(
        f'        _sum_cos += exp_cos[i,{k+1}] * ({_jl(red)})'
        for k, red in enumerate(reduced[:n_cos]))
    sin_lines = '\n'.join(
        f'        _sum_sin += exp_sin[i,{k+1}] * ({_jl(red)})'
        for k, red in enumerate(reduced[n_cos:]))

    if not cos_lines:
        cos_lines = '        # no cos terms'
    if not sin_lines:
        sin_lines = '        # no sin terms'

    L += [_EVAL_SIG, _TRIG_BLOCK,
          '    @inbounds for i in 1:N',
          '        x, y, z = X[i], Y[i], Z[i]',
          cse_block,
          '        _sum_cos = 0.0',
          '        _sum_sin = 0.0',
          cos_lines,
          sin_lines,
          '        out[i] = cos_p[i] * _sum_cos + sin_p[i] * _sum_sin',
          '    end', 'end', '']
    L += _main_func(n_cos, n_sin)
    return '\n'.join(L), terms_cos, terms_sin, {
        'ops_post': ops_post, 'n_cse_temps': len(repl)}


# ---------------------------------------------------------------------------
# Strategy 5: k_separate
# ---------------------------------------------------------------------------

def _mono_jl(mono):
    parts = [(v, e) for v, e in zip(('x', 'y', 'z'), mono) if e > 0]
    if not parts:
        return '1.0'
    terms = []
    for v, e in parts:
        terms.append(v if e == 1 else f'{v}^{e}')
    return ' * '.join(terms)


def _try_k_separate(pp):
    x, y, z = sp.symbols('x y z')
    try:
        poly = sp.Poly(sp.expand(pp), x, y, z)
        return list(zip(poly.monoms(), poly.coeffs()))
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return None


def build_k_separate(terms_cos, terms_sin):
    """Separate k-polynomial from spatial polynomial."""
    n_cos, n_sin = len(terms_cos), len(terms_sin)
    L = ['# strategy: k_separate']

    k_coeff_lines = []
    cos_inner = []
    sin_inner = []
    fallback_cos = []
    fallback_sin = []
    ops_post = 0

    for i, (_, pp) in enumerate(terms_cos):
        decomp = _try_k_separate(pp)
        if decomp is None:
            fallback_cos.append(i)
        else:
            parts = []
            for j, (mono, kc) in enumerate(decomp):
                vname = f'_kc_c{i}_{j}'
                k_coeff_lines.append(f'    {vname} = {_jl(kc)}')
                ops_post += int(sp.count_ops(kc))
                parts.append(f'{vname} * {_mono_jl(mono)}' if mono != (0,0,0)
                             else vname)
            # spatial loop ops: monomial mults + additions
            ops_post += sum(max(sum(1 for e in m if e > 0) - 1, 0)
                           for m, _ in decomp)
            ops_post += max(len(decomp) - 1, 0)
            expr = ' + '.join(parts) if parts else '0.0'
            cos_inner.append(f'        _sum_cos += exp_cos[i,{i+1}] * ({expr})')

    for i, (_, pp) in enumerate(terms_sin):
        decomp = _try_k_separate(pp)
        if decomp is None:
            fallback_sin.append(i)
        else:
            parts = []
            for j, (mono, kc) in enumerate(decomp):
                vname = f'_kc_s{i}_{j}'
                k_coeff_lines.append(f'    {vname} = {_jl(kc)}')
                ops_post += int(sp.count_ops(kc))
                parts.append(f'{vname} * {_mono_jl(mono)}' if mono != (0,0,0)
                             else vname)
            ops_post += sum(max(sum(1 for e in m if e > 0) - 1, 0)
                           for m, _ in decomp)
            ops_post += max(len(decomp) - 1, 0)
            expr = ' + '.join(parts) if parts else '0.0'
            sin_inner.append(f'        _sum_sin += exp_sin[i,{i+1}] * ({expr})')

    n_fallback = len(fallback_cos) + len(fallback_sin)
    if n_fallback:
        print(f'    [k_separate] {n_fallback} groups could not be separated '
              '(falling back to flat poly)', flush=True)

    for i in fallback_cos:
        _, pp = terms_cos[i]
        L += [f'@inline function _poly_cos_{i}(x::Float64, y::Float64, z::Float64,'
              f' kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}', 'end', '']
        cos_inner.append(
            f'        _sum_cos += exp_cos[i,{i+1}]'
            f' * _poly_cos_{i}(x, y, z, kx, ky, kz)')
        ops_post += int(sp.count_ops(pp))
    for i in fallback_sin:
        _, pp = terms_sin[i]
        L += [f'@inline function _poly_sin_{i}(x::Float64, y::Float64, z::Float64,'
              f' kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}', 'end', '']
        sin_inner.append(
            f'        _sum_sin += exp_sin[i,{i+1}]'
            f' * _poly_sin_{i}(x, y, z, kx, ky, kz)')
        ops_post += int(sp.count_ops(pp))

    L += _exp_funcs(terms_cos, terms_sin) + _precompute_exp(n_cos, n_sin)

    k_block = '\n'.join(k_coeff_lines) if k_coeff_lines else '    # (no separable groups)'
    cos_block = '\n'.join(cos_inner) or '        # no cos terms'
    sin_block = '\n'.join(sin_inner) or '        # no sin terms'

    L += [_EVAL_SIG,
          '    # Per-wave: precompute k-polynomial coefficients (scalars, no kx/ky/kz in loop)',
          k_block,
          _TRIG_BLOCK,
          '    @inbounds for i in 1:N',
          '        x, y, z = X[i], Y[i], Z[i]',
          '        _sum_cos = 0.0',
          '        _sum_sin = 0.0',
          cos_block,
          sin_block,
          '        out[i] = cos_p[i] * _sum_cos + sin_p[i] * _sum_sin',
          '    end', 'end', '']
    L += _main_func(n_cos, n_sin)
    return '\n'.join(L), terms_cos, terms_sin, {
        'ops_post': ops_post, 'n_k_coeffs': len(k_coeff_lines)}


# ---------------------------------------------------------------------------
# Julia runner
# ---------------------------------------------------------------------------

def run_julia(jl_path, X, Y, Z, k_vals, b_vals, work_dir, julia_exe, n_reps=3):
    """Run a pre-built .jl file, average over n_reps; return timing dict + output."""
    Ng = X.shape[0]
    N = Ng ** 3
    n_waves = len(k_vals)

    grid_bin  = Path(work_dir) / '_g.bin'
    kvals_bin = Path(work_dir) / '_k.bin'
    out_bin   = Path(work_dir) / '_o.bin'

    with open(grid_bin, 'wb') as f:
        X.ravel().astype('<f8').tofile(f)
        Y.ravel().astype('<f8').tofile(f)
        Z.ravel().astype('<f8').tofile(f)
    kb = np.column_stack([k_vals, b_vals]).astype('<f8')
    kb.ravel().tofile(str(kvals_bin))

    cmd = [julia_exe, str(jl_path),
           str(grid_bin), str(kvals_bin), str(out_bin),
           str(N), str(n_waves)]

    timings = []
    raw_out = None
    for rep in range(n_reps):
        t_wall = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        t_wall = time.time() - t_wall

        if proc.returncode != 0:
            raise RuntimeError(
                f'Julia failed (code {proc.returncode}).\n'
                f'stderr:\n{proc.stderr[:2000]}\nstdout:\n{proc.stdout[:1000]}')

        t = {'warmup_s': None, 'eval_s': None, 'n_eval_waves': None,
             'total_wall_s': t_wall}
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                try:
                    t.update(json.loads(line))
                except json.JSONDecodeError:
                    pass
                break
        timings.append(t)

        if rep == 0 and out_bin.exists():
            raw_out = np.fromfile(str(out_bin), dtype='<f8')

    grid_bin.unlink(missing_ok=True)
    kvals_bin.unlink(missing_ok=True)
    out_bin.unlink(missing_ok=True)

    eval_vals = [t['eval_s'] for t in timings if t['eval_s'] is not None]
    n_eval    = timings[0].get('n_eval_waves') or max(n_waves - 1, 1)
    avg_eval  = float(np.mean(eval_vals)) if eval_vals else None
    warmup    = timings[0].get('warmup_s')

    return {
        'warmup_s'     : warmup,
        'eval_s'       : avg_eval,
        'n_eval_waves' : n_eval,
        'ms_per_wave'  : (avg_eval / n_eval * 1000) if avg_eval else None,
        'total_wall_s' : timings[0]['total_wall_s'],
        'n_reps'       : n_reps,
    }, raw_out


# ---------------------------------------------------------------------------
# Expression preparation
# ---------------------------------------------------------------------------

def prepare_expressions(m, E_lo, E_hi, cache_dir, expand=True):
    """Build psi_fH for 3-D HO, return grouped (exp,poly) lists.

    expand=True  (default)
        Calls sp.expand() at every H^n step → flat monomial sum with rational
        coefficients.  Loads from / saves to cache_dir.

    expand=False
        Skips sp.expand() → preserves product structure (V*Ps, k2*Ps, etc.).
        Loads from / saves to cache_dir (use a separate noexpand cache dir).
        return_envelopes=True is used to bypass psi_fH.expand() in
        extract_cos_sin_coeffs, keeping the tree intact for sp.cse().
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from symbolic_code.chebyshev_filter import (
        extract_cos_sin_coeffs, group_by_exp_combined,
        apply_f_of_H_from_raw_powers)
    from symbolic_code.h_powers import generate_H_powers
    from test_symbolic_ho3d import _max_cached_power

    cache_dir = Path(cache_dir)
    a   = 2.0 / (E_hi - E_lo)
    b_sc = -(E_hi + E_lo) / (E_hi - E_lo)

    if expand:
        from test_symbolic_ho3d import build_fH_psi_sympy
        print(f'Building f(H)*psi  m={m}  E_lo={E_lo}  E_hi={E_hi:.2f}  '
              f'[expand=True] ...', flush=True)
        t0 = time.time()
        psi_fH, info = build_fH_psi_sympy(m, E_lo, E_hi, cache_dir=str(cache_dir))
        print(f'  done in {time.time()-t0:.1f}s', flush=True)

        print('Extracting cos/sin envelopes + grouping by exp ...', flush=True)
        t0 = time.time()
        expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)
    else:
        # No-expand path: generate H^n without sp.expand() if not cached, then
        # return envelopes directly (bypassing extract_cos_sin_coeffs which calls
        # .expand() and would destroy the preserved tree structure).
        max_n = _max_cached_power(cache_dir)
        if max_n < m:
            print(f'  Building noexpand H^n cache (m={m}) in {cache_dir} ...',
                  flush=True)
            x, y, z = sp.symbols('x y z')
            kx, ky, kz = sp.symbols('kx ky kz')
            V_sym = sp.Rational(1, 2) * (x**2 + y**2 + z**2)
            t0 = time.time()
            generate_H_powers(
                m, cache_dir, file_format='pkl', expand=False,
                V=V_sym, kvec=(kx, ky, kz), k2=kx**2+ky**2+kz**2,
                pref=0.5, x=x, y=y, z=z)
            print(f'  done in {time.time()-t0:.1f}s', flush=True)
        else:
            print(f'  Noexpand H^n cache hit (max_n={max_n}): {cache_dir}',
                  flush=True)

        print(f'Assembling f(H)*psi envelopes  [expand=False] ...', flush=True)
        t0 = time.time()
        # return_envelopes=True returns (Pc_total, Ps_total) without expanding
        expr_cos, expr_sin = apply_f_of_H_from_raw_powers(
            cache_dir, m, a=a, b=b_sc, return_envelopes=True)
        info = {'expand': False, 'cache_dir': str(cache_dir)}

    t0 = time.time()
    terms_cos = group_by_exp_combined(expr_cos)
    terms_sin = group_by_exp_combined(expr_sin)
    print(f'  done in {time.time()-t0:.1f}s  '
          f'cos_groups={len(terms_cos)}  sin_groups={len(terms_sin)}', flush=True)
    return terms_cos, terms_sin, info


# ---------------------------------------------------------------------------
# Build dispatch
# ---------------------------------------------------------------------------

BUILDERS = {
    'baseline'    : build_baseline,
    'no_horner'   : build_no_horner,
    'cse_group'   : build_cse_group,
    'cse_inlined' : build_cse_inlined,
    'k_separate'  : build_k_separate,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Julia evaluation strategies for f(H)*psi.'
    )
    parser.add_argument('--m', type=int, default=6,
                        help='Chebyshev order (default 6)')
    parser.add_argument('--E_lo', type=float, default=4.5,
                        help='Filter threshold (default 4.5)')
    parser.add_argument('--E_hi', type=float, default=None,
                        help='Upper bound (auto from grid if omitted)')
    parser.add_argument('--L', type=float, default=5.5,
                        help='Half-box size (default 5.5)')
    parser.add_argument('--Ng', type=int, default=22,
                        help='Grid points per axis (default 22)')
    parser.add_argument('--n_waves', type=int, default=40,
                        help='Plane waves (default 40; use >= 20 for valid timing)')
    parser.add_argument('--n_reps', type=int, default=3,
                        help='Julia timing repetitions (default 3)')
    parser.add_argument('--cache_dir', default='ho3d_h_powers_cache',
                        help='H^n cache dir for expand=True (default ho3d_h_powers_cache/)')
    parser.add_argument('--noexp_cache_dir', default='ho3d_h_powers_cache_noexpand',
                        help='H^n cache dir for expand=False '
                             '(default ho3d_h_powers_cache_noexpand/)')
    parser.add_argument('--compare_expand', action='store_true',
                        help='Run each strategy on BOTH expand=True (_E) and '
                             'expand=False (_N); shows combined table')
    parser.add_argument('--no_expand', action='store_true',
                        help='Run on expand=False only (separate from default expand=True)')
    parser.add_argument('--strategies', default=','.join(BUILDERS),
                        help=f'Comma-separated strategies (default: all)')
    parser.add_argument('--julia_exe', default='julia',
                        help='Julia executable (default: julia)')
    parser.add_argument('--no_check', action='store_true',
                        help='Skip correctness check vs baseline')
    parser.add_argument('--save_jl', action='store_true',
                        help='Keep generated .jl files after run')
    parser.add_argument('--outdir', default='benchmark_results',
                        help='Dir for results and .jl files (default benchmark_results/)')
    args = parser.parse_args()

    requested = [s.strip() for s in args.strategies.split(',')]
    unknown = [s for s in requested if s not in BUILDERS]
    if unknown:
        print(f'Unknown strategies: {unknown}')
        print(f'Valid: {list(BUILDERS.keys())}')
        sys.exit(1)

    if args.n_waves < 2:
        print(f'\nWARNING: --n_waves={args.n_waves} → the Julia timed section runs '
              f'n_waves-1=0 iterations, so eval_s=0 and ms/wave=0 for ALL strategies.  '
              f'Use --n_waves 40 for meaningful timing.\n')

    dx   = 2 * args.L / args.Ng
    E_hi = args.E_hi if args.E_hi is not None else (
        0.5 * 3 * (np.pi / dx) ** 2 + 0.5 * 3 * (args.L - dx) ** 2
    )
    print(f'\nE_hi = {E_hi:.4f}  ({"user" if args.E_hi else "auto"})\n')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Each entry: (label_suffix, cache_dir_path, expand_bool)
    if args.compare_expand:
        datasets = [
            ('_E', args.cache_dir,       True),
            ('_N', args.noexp_cache_dir, False),
        ]
    elif args.no_expand:
        datasets = [('', args.noexp_cache_dir, False)]
    else:
        datasets = [('', args.cache_dir, True)]

    x1 = np.linspace(-args.L, args.L, args.Ng, endpoint=False)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
    rng    = np.random.default_rng(42)
    k_vals = rng.uniform(-1.0, 1.0, (args.n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, args.n_waves)

    all_results = {}
    reference_out = None   # expand=True baseline used for correctness checks

    with tempfile.TemporaryDirectory(prefix='bench_jl_') as work_dir:
        for suffix, cache_dir, expand in datasets:
            print(f'\n{"─"*60}')
            print(f'Dataset: expand={expand}  cache={cache_dir}')
            print(f'{"─"*60}')
            terms_cos, terms_sin, _ = prepare_expressions(
                args.m, args.E_lo, E_hi, cache_dir, expand=expand)

            ops_flat    = total_ops(terms_cos)   + total_ops(terms_sin)
            nterms_flat = total_terms(terms_cos) + total_terms(terms_sin)
            print(f'\nFlat poly (pre-optim): {ops_flat} ops  {nterms_flat} terms')
            print(f'Grid: {args.Ng}^3={args.Ng**3} pts  '
                  f'n_waves={args.n_waves}  n_reps={args.n_reps}\n')

            for name in requested:
                full_name = name + suffix
                print(f'{"="*55}')
                print(f'Strategy: {full_name}  (expand={expand})')
                t_build = time.time()
                try:
                    jl_src, tc, ts, extra = BUILDERS[name](terms_cos, terms_sin)
                except Exception as exc:
                    print(f'  BUILD FAILED: {exc}')
                    all_results[full_name] = {'error': str(exc)}
                    continue
                t_build = time.time() - t_build

                ops_post = extra.get('ops_post', total_ops(tc) + total_ops(ts))

                jl_file = outdir / f'eval_{full_name}_m{args.m}.jl'
                jl_file.write_text(jl_src, encoding='utf-8')
                extra_str = '  '.join(
                    f'{k}={v}' for k, v in extra.items() if k != 'ops_post')
                print(f'  .jl written ({len(jl_src):,} bytes)  '
                      f'build_time={t_build:.1f}s')
                print(f'  ops_post={ops_post}  {extra_str}')

                try:
                    timing, raw_out = run_julia(
                        jl_file, X, Y, Z, k_vals, b_vals, work_dir,
                        args.julia_exe, n_reps=args.n_reps)
                except RuntimeError as exc:
                    print(f'  JULIA FAILED: {exc}')
                    all_results[full_name] = {'error': str(exc),
                                              'ops_post': ops_post}
                    if not args.save_jl:
                        jl_file.unlink(missing_ok=True)
                    continue

                mspw = timing['ms_per_wave']
                print(f'  warmup={timing["warmup_s"]*1000:.1f}ms  '
                      f'eval={mspw:.3f}ms/wave  wall={timing["total_wall_s"]:.1f}s')

                max_err = None
                if not args.no_check and raw_out is not None:
                    if name == 'baseline' and expand:
                        reference_out = raw_out.copy()
                        max_err = 0.0
                    elif reference_out is not None:
                        diff = np.abs(raw_out - reference_out)
                        max_err = float(diff.max())
                        status = 'OK' if max_err < 1e-6 else 'WARNING'
                        print(f'  vs reference: max|diff|={max_err:.2e}  [{status}]')

                all_results[full_name] = {
                    'ops_post'    : ops_post,
                    'ops_flat'    : ops_flat,
                    'expand'      : expand,
                    'ms_per_wave' : mspw,
                    'warmup_ms'   : timing['warmup_s'] * 1000 if timing['warmup_s'] else None,
                    'total_wall_s': timing['total_wall_s'],
                    'max_err'     : max_err,
                    'build_time_s': t_build,
                    **{k: v for k, v in extra.items() if k != 'ops_post'},
                }

                if not args.save_jl:
                    jl_file.unlink(missing_ok=True)

    # ---- Summary table ----
    print(f'\n{"="*80}')
    print(f'SUMMARY  m={args.m}  E_lo={args.E_lo}  E_hi={E_hi:.2f}  '
          f'Ng={args.Ng}  n_waves={args.n_waves}')
    if args.n_waves < 2:
        print('  *** n_waves<2: ms/wave=0, speedup meaningless — use --n_waves 40 ***')
    print(f'{"="*80}')

    ref_ms = None
    for cand in ('baseline', 'baseline_E'):
        if cand in all_results and all_results[cand].get('ms_per_wave'):
            ref_ms = all_results[cand]['ms_per_wave']
            break

    header = (f'{"Strategy":<18} {"exp":>4} {"ops_post":>9} {"ms/wave":>9} '
              f'{"vs_ref":>10} {"warmup_ms":>10} {"max_err":>10}')
    print(header)
    print('-' * len(header))
    for full_name in all_results:
        r = all_results[full_name]
        if 'error' in r:
            print(f'{full_name:<18}  ERROR: {r["error"][:45]}')
            continue
        mspw  = r.get('ms_per_wave')
        exp_s = 'T' if r.get('expand', True) else 'F'
        if ref_ms and mspw:
            ratio = ref_ms / mspw
            # Show as Nx faster (ratio>1) or 1/Nx slower (ratio<1) so the
            # sign of the comparison is always obvious in the output.
            spd = f'{ratio:.2f}x' if ratio >= 0.1 else f'1/{1/ratio:.0f}x'
        else:
            spd = '-'
        err   = f'{r["max_err"]:.1e}' if r.get('max_err') is not None else '-'
        wmup  = f'{r["warmup_ms"]:.1f}' if r.get('warmup_ms') else '-'
        print(f'{full_name:<18} {exp_s:>4} {r["ops_post"]:>9} '
              f'{(mspw or 0):>9.3f} {spd:>10} {wmup:>10} {err:>10}')

    result_file = outdir / f'benchmark_m{args.m}.json'
    with open(result_file, 'w') as f:
        json.dump({
            'args': {'m': args.m, 'E_lo': args.E_lo, 'E_hi': E_hi,
                     'Ng': args.Ng, 'n_waves': args.n_waves,
                     'n_reps': args.n_reps,
                     'compare_expand': args.compare_expand,
                     'no_expand': args.no_expand},
            'results': all_results,
        }, f, indent=2, default=str)
    print(f'\nResults → {result_file}')
    if args.save_jl:
        print(f'.jl files → {outdir}/')


if __name__ == '__main__':
    main()
