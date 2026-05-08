#!/usr/bin/env python3
"""benchmark_strategies.py
Compare different Julia expression strategies for f(H)*psi evaluation.

Strategies
----------
baseline      current: group_by_exp + Horner + @inline funcs per group
no_horner     group_by_exp only, flat expanded poly (worst-case reference)
cse_group     group_by_exp + sp.cse() per group  (replaces Horner with CSE)
cse_inlined   global sp.cse() across all groups + single monolithic inner loop
k_separate    separate k-polynomial from spatial-polynomial; precompute k-coefficients
              once per wave, evaluate pure spatial poly per point

Theory
------
baseline  : Horner rewrites f(x) = a0 + a1*x + a2*x^2 + ... as a0 + x*(a1 + x*(...))
            For single-variable polys this halves multiplications.  For 6-variable
            (x,y,z,kx,ky,kz) polys sympy.horner picks one ordering variable and
            only partially reuses cross-terms — expansion beforehand may discard
            useful factored structure.

cse_group : sp.cse() on the flat expanded poly finds ALL repeated sub-DAGs
            (e.g. x*y computed 7 times → 1 temp).  Always reduces or matches
            Horner for op-count; never makes things worse.

cse_inlined : same CSE but across ALL groups simultaneously, putting shared
              temps directly in the @inbounds inner loop.  Avoids @inline call
              overhead and cross-group redundancy.

k_separate : The polynomial poly(x,y,z,kx,ky,kz) with kx,ky,kz CONSTANT per wave
             is decomposed as  Σ_α c_α(kx,ky,kz)·xᵃyᵇzᶜ.  For each wave we
             compute the scalar k-coefficients once (O(Q·K) ops, Q monomials,
             K ops per k-coeff).  The inner loop then has only spatial multiplications
             (kx,ky,kz multiply-free).  For large N and small n_waves this wins;
             for small N or many cheap k-terms Horner/CSE may win.

lookup_blas : (future) precompute spatial-basis matrix B[N,Q], then per wave do
             BLAS GEMV: out = B @ c_vec.  Not implemented here; use k_separate as
             approximation since it has the same asymptotic cost.

Metrics reported
----------------
  ops_total   : sp.count_ops() summed over all poly parts (lower = fewer FLOPs)
  n_terms     : total summands across poly parts (before CSE/Horner)
  n_cse_temps : CSE replacement variables introduced (cse_* strategies only)
  warmup_ms   : Julia JIT + first-wave time
  eval_ms/w   : average ms per wave (excluding warmup)
  speedup     : relative to baseline eval_ms/w (>1 is faster)
  max_err     : max |output - baseline| (correctness check)

Usage
-----
  python benchmark_strategies.py                                 # m=6, all strategies
  python benchmark_strategies.py --m 8 --n_waves 40
  python benchmark_strategies.py --strategies baseline,cse_group,k_separate
  python benchmark_strategies.py --julia_exe /usr/bin/julia --n_reps 5
  python benchmark_strategies.py --no_check                      # skip correctness check
  python benchmark_strategies.py --save_jl                       # keep .jl files for inspection
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
# Julia code helper (same as in julia_codegen.py)
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
    """Sum sp.count_ops over all poly parts."""
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
    """group_by_exp + Horner + @inline per group  (current code)."""
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
    return '\n'.join(L), tc, ts, {}


# ---------------------------------------------------------------------------
# Strategy 2: no_horner
# ---------------------------------------------------------------------------

def build_no_horner(terms_cos, terms_sin):
    """Flat expanded poly, no Horner, no CSE  (worst-case reference)."""
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
    return '\n'.join(L), terms_cos, terms_sin, {}


# ---------------------------------------------------------------------------
# Strategy 3: cse_group
# ---------------------------------------------------------------------------

def build_cse_group(terms_cos, terms_sin):
    """CSE per poly group; each @inline function gets local CSE temps."""
    n_cos, n_sin = len(terms_cos), len(terms_sin)
    L = ['# strategy: cse_group']
    total_cse = 0

    def _poly_cse(name, pp):
        nonlocal total_cse
        repl, (red,) = sp.cse([pp], symbols=sp.numbered_symbols('_s'))
        total_cse += len(repl)
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
    return '\n'.join(L), terms_cos, terms_sin, {'n_cse_temps': total_cse}


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

    L = ['# strategy: cse_inlined']
    L += _exp_funcs(terms_cos, terms_sin) + _precompute_exp(n_cos, n_sin)

    # CSE preamble lines (computed once per point inside the inner loop)
    cse_lines = [f'        {sym} = {_jl(expr)}' for sym, expr in repl]
    cse_block = '\n'.join(cse_lines) if cse_lines else '        # no shared CSE temps'

    # Group contribution lines using reduced expressions
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
    return '\n'.join(L), terms_cos, terms_sin, {'n_cse_temps': len(repl)}


# ---------------------------------------------------------------------------
# Strategy 5: k_separate
# ---------------------------------------------------------------------------

def _mono_jl(mono):
    """Julia code for x^a * y^b * z^c (mono = (a,b,c) tuple)."""
    parts = [(v, e) for v, e in zip(('x', 'y', 'z'), mono) if e > 0]
    if not parts:
        return '1.0'
    terms = []
    for v, e in parts:
        terms.append(v if e == 1 else f'{v}^{e}')
    return ' * '.join(terms)


def _try_k_separate(pp):
    """Decompose pp(x,y,z,kx,ky,kz) → list of ((a,b,c), k_coeff_expr).

    Returns None if the polynomial cannot be expressed as a poly in (x,y,z).
    """
    x, y, z = sp.symbols('x y z')
    try:
        poly = sp.Poly(sp.expand(pp), x, y, z)
        return list(zip(poly.monoms(), poly.coeffs()))
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return None


def build_k_separate(terms_cos, terms_sin):
    """Separate k-polynomial from spatial polynomial.

    For each poly group pp(x,y,z,kx,ky,kz) we form
        pp = Σ_α  c_α(kx,ky,kz) · x^a · y^b · z^c
    The k-coefficients c_α are scalars precomputed once per wave.
    The inner loop only contains spatial multiplications.
    Groups that cannot be separated fall back to the flat-poly @inline form.
    """
    n_cos, n_sin = len(terms_cos), len(terms_sin)

    L = ['# strategy: k_separate']

    # ---- per-wave k-coefficient variables and inner-loop spatial sums ----
    k_coeff_lines = []   # declared before @inbounds loop
    cos_inner = []       # lines inside @inbounds loop for cos groups
    sin_inner = []       # lines inside @inbounds loop for sin groups
    fallback_cos = []    # group indices that couldn't be separated
    fallback_sin = []

    for i, (_, pp) in enumerate(terms_cos):
        decomp = _try_k_separate(pp)
        if decomp is None:
            fallback_cos.append(i)
        else:
            parts = []
            for j, (mono, kc) in enumerate(decomp):
                vname = f'_kc_c{i}_{j}'
                k_coeff_lines.append(f'    {vname} = {_jl(kc)}')
                parts.append(f'{vname} * {_mono_jl(mono)}' if mono != (0,0,0)
                             else vname)
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
                parts.append(f'{vname} * {_mono_jl(mono)}' if mono != (0,0,0)
                             else vname)
            expr = ' + '.join(parts) if parts else '0.0'
            sin_inner.append(f'        _sum_sin += exp_sin[i,{i+1}] * ({expr})')

    n_fallback = len(fallback_cos) + len(fallback_sin)
    if n_fallback:
        print(f'    [k_separate] {n_fallback} groups could not be separated '
              '(falling back to flat poly)', flush=True)

    # Fallback: add @inline functions for non-separable groups
    for i in fallback_cos:
        _, pp = terms_cos[i]
        L += [f'@inline function _poly_cos_{i}(x::Float64, y::Float64, z::Float64,'
              f' kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}', 'end', '']
        cos_inner.append(
            f'        _sum_cos += exp_cos[i,{i+1}]'
            f' * _poly_cos_{i}(x, y, z, kx, ky, kz)')
    for i in fallback_sin:
        _, pp = terms_sin[i]
        L += [f'@inline function _poly_sin_{i}(x::Float64, y::Float64, z::Float64,'
              f' kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}', 'end', '']
        sin_inner.append(
            f'        _sum_sin += exp_sin[i,{i+1}]'
            f' * _poly_sin_{i}(x, y, z, kx, ky, kz)')

    L += _exp_funcs(terms_cos, terms_sin) + _precompute_exp(n_cos, n_sin)

    k_block = '\n'.join(k_coeff_lines) if k_coeff_lines else '    # (no separable groups)'
    cos_block = '\n'.join(cos_inner) or '        # no cos terms'
    sin_block = '\n'.join(sin_inner) or '        # no sin terms'

    L += [_EVAL_SIG,
          '    # Per-wave: precompute k-polynomial coefficients (scalars, kx/ky/kz free in loop)',
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
    return '\n'.join(L), terms_cos, terms_sin, {'n_k_coeffs': len(k_coeff_lines)}


# ---------------------------------------------------------------------------
# Julia runner
# ---------------------------------------------------------------------------

def run_julia(jl_path, X, Y, Z, k_vals, b_vals, work_dir, julia_exe, n_reps=3):
    """Run a pre-built .jl file, average over n_reps; return timing dict + raw output."""
    Ng = X.shape[0]
    N = Ng ** 3
    n_waves = len(k_vals)

    grid_bin  = Path(work_dir) / '_g.bin'
    kvals_bin = Path(work_dir) / '_k.bin'
    out_bin   = Path(work_dir) / '_o.bin'

    # Write grid
    with open(grid_bin, 'wb') as f:
        X.ravel().astype('<f8').tofile(f)
        Y.ravel().astype('<f8').tofile(f)
        Z.ravel().astype('<f8').tofile(f)
    # Write k/b
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

    # Average eval_s over reps (warmup excluded — it always includes JIT)
    eval_vals = [t['eval_s'] for t in timings if t['eval_s'] is not None]
    n_eval    = timings[0].get('n_eval_waves') or max(n_waves - 1, 1)
    avg_eval  = float(np.mean(eval_vals)) if eval_vals else None
    warmup    = timings[0].get('warmup_s')

    return {
        'warmup_s'      : warmup,
        'eval_s'        : avg_eval,
        'n_eval_waves'  : n_eval,
        'ms_per_wave'   : (avg_eval / n_eval * 1000) if avg_eval else None,
        'total_wall_s'  : timings[0]['total_wall_s'],
        'n_reps'        : n_reps,
    }, raw_out


# ---------------------------------------------------------------------------
# Expression preparation  (shared across strategies)
# ---------------------------------------------------------------------------

def prepare_expressions(m, E_lo, E_hi, cache_dir):
    """Build psi_fH for 3-D harmonic oscillator, return grouped (exp,poly) lists."""
    sys.path.insert(0, str(Path(__file__).parent))
    from test_symbolic_ho3d import build_fH_psi_sympy
    from symbolic_code.chebyshev_filter import (
        extract_cos_sin_coeffs, group_by_exp_combined)

    print(f'Building f(H)*psi  m={m}  E_lo={E_lo}  E_hi={E_hi:.2f} ...', flush=True)
    t0 = time.time()
    psi_fH, info = build_fH_psi_sympy(m, E_lo, E_hi, cache_dir=cache_dir)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    print('Extracting cos/sin envelopes + grouping by exp ...', flush=True)
    t0 = time.time()
    expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)
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
                        help='Plane waves for timing (default 40)')
    parser.add_argument('--n_reps', type=int, default=3,
                        help='Julia timing repetitions (default 3)')
    parser.add_argument('--cache_dir', default='ho3d_h_powers_cache',
                        help='H^n cache dir (default ho3d_h_powers_cache/)')
    parser.add_argument('--strategies', default=','.join(BUILDERS),
                        help=f'Comma-separated strategies (default: all)')
    parser.add_argument('--julia_exe', default='julia',
                        help='Julia executable (default: julia)')
    parser.add_argument('--no_check', action='store_true',
                        help='Skip correctness check vs baseline')
    parser.add_argument('--save_jl', action='store_true',
                        help='Keep generated .jl files (default: delete after run)')
    parser.add_argument('--outdir', default='benchmark_results',
                        help='Dir for results JSON and .jl files (default benchmark_results/)')
    args = parser.parse_args()

    requested = [s.strip() for s in args.strategies.split(',')]
    unknown = [s for s in requested if s not in BUILDERS]
    if unknown:
        print(f'Unknown strategies: {unknown}')
        print(f'Valid: {list(BUILDERS.keys())}')
        sys.exit(1)

    # Auto E_hi
    dx = 2 * args.L / args.Ng
    E_hi = args.E_hi if args.E_hi is not None else (
        0.5 * 3 * (np.pi / dx) ** 2 + 0.5 * 3 * (args.L - dx) ** 2
    )
    print(f'\nE_hi = {E_hi:.4f}  ({"user" if args.E_hi else "auto"})\n')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Build expressions (shared across strategies)
    terms_cos, terms_sin, expr_info = prepare_expressions(
        args.m, args.E_lo, E_hi, args.cache_dir)

    # Grid
    x1 = np.linspace(-args.L, args.L, args.Ng, endpoint=False)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')

    rng = np.random.default_rng(42)
    k_vals = rng.uniform(-1.0, 1.0, (args.n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, args.n_waves)

    # Raw op counts on the FLAT (unmodified) expressions
    ops_flat = total_ops(terms_cos) + total_ops(terms_sin)
    nterms_flat = total_terms(terms_cos) + total_terms(terms_sin)

    print(f'\nFlat poly complexity: {ops_flat} total ops, {nterms_flat} terms')
    print(f'Grid: {args.Ng}^3={args.Ng**3} pts  n_waves={args.n_waves}  n_reps={args.n_reps}')
    print()

    # Run each strategy
    results = {}
    baseline_out = None

    with tempfile.TemporaryDirectory(prefix='bench_jl_') as work_dir:
        for name in requested:
            print(f'{"="*55}')
            print(f'Strategy: {name}')
            t_build = time.time()
            try:
                jl_src, tc, ts, extra = BUILDERS[name](terms_cos, terms_sin)
            except Exception as exc:
                print(f'  BUILD FAILED: {exc}')
                results[name] = {'error': str(exc)}
                continue
            t_build = time.time() - t_build

            # Compute ops on the strategy's actual poly terms
            ops_used = total_ops(tc) + total_ops(ts)
            nterms_used = total_terms(tc) + total_terms(ts)

            jl_file = outdir / f'eval_{name}_m{args.m}.jl'
            jl_file.write_text(jl_src, encoding='utf-8')
            print(f'  .jl written ({len(jl_src):,} bytes)  build_time={t_build:.1f}s')
            print(f'  ops={ops_used}  terms={nterms_used}  '
                  + ('  '.join(f'{k}={v}' for k, v in extra.items())))

            try:
                timing, raw_out = run_julia(
                    jl_file, X, Y, Z, k_vals, b_vals, work_dir,
                    args.julia_exe, n_reps=args.n_reps)
            except RuntimeError as exc:
                print(f'  JULIA FAILED: {exc}')
                results[name] = {'error': str(exc), 'ops': ops_used}
                if not args.save_jl:
                    jl_file.unlink(missing_ok=True)
                continue

            mspw = timing['ms_per_wave']
            print(f'  warmup={timing["warmup_s"]*1000:.1f}ms  '
                  f'eval={mspw:.3f}ms/wave  wall={timing["total_wall_s"]:.1f}s')

            # Correctness check
            max_err = None
            if not args.no_check and raw_out is not None:
                if name == 'baseline':
                    baseline_out = raw_out.copy()
                    max_err = 0.0
                elif baseline_out is not None:
                    diff = np.abs(raw_out - baseline_out)
                    max_err = float(diff.max())
                    status = 'OK' if max_err < 1e-8 else 'WARNING'
                    print(f'  correctness vs baseline: max|diff|={max_err:.2e}  [{status}]')

            results[name] = {
                'ops'        : ops_used,
                'n_terms'    : nterms_used,
                'ms_per_wave': mspw,
                'warmup_ms'  : timing['warmup_s'] * 1000 if timing['warmup_s'] else None,
                'total_wall_s': timing['total_wall_s'],
                'max_err'    : max_err,
                'build_time_s': t_build,
                **extra,
            }

            if not args.save_jl:
                jl_file.unlink(missing_ok=True)

    # Comparison table
    print(f'\n{"="*80}')
    print(f'SUMMARY  m={args.m}  E_lo={args.E_lo}  E_hi={E_hi:.2f}  '
          f'Ng={args.Ng}  n_waves={args.n_waves}')
    print(f'{"="*80}')

    baseline_ms = (results.get('baseline') or {}).get('ms_per_wave')

    header = f'{"Strategy":<14} {"ops":>7} {"terms":>6} {"ms/wave":>9} {"speedup":>8} {"max_err":>10}'
    print(header)
    print('-' * len(header))
    for name in requested:
        r = results.get(name, {})
        if 'error' in r:
            print(f'{name:<14}  ERROR: {r["error"][:40]}')
            continue
        mspw = r.get('ms_per_wave')
        spd  = f'{baseline_ms/mspw:.2f}x' if (baseline_ms and mspw) else '-'
        err  = f'{r["max_err"]:.1e}' if r.get('max_err') is not None else '-'
        print(f'{name:<14} {r["ops"]:>7} {r["n_terms"]:>6} '
              f'{(mspw or 0):>9.3f} {spd:>8} {err:>10}')

    # Save JSON
    result_file = outdir / f'benchmark_m{args.m}.json'
    with open(result_file, 'w') as f:
        json.dump({
            'args': {'m': args.m, 'E_lo': args.E_lo, 'E_hi': E_hi,
                     'Ng': args.Ng, 'n_waves': args.n_waves, 'n_reps': args.n_reps},
            'flat_ops': ops_flat, 'flat_terms': nterms_flat,
            'results': results,
        }, f, indent=2, default=str)
    print(f'\nResults → {result_file}')

    if args.save_jl:
        print(f'.jl files → {outdir}/')


if __name__ == '__main__':
    main()
