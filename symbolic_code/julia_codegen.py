"""Python-side utilities for generating Julia source code from sympy expressions.

Typical usage
-------------
1. Compute psi_fH via :func:`chebyshev_filter.apply_f_of_H_on_psi`.
2. Extract envelopes with :func:`chebyshev_filter.extract_cos_sin_coeffs`.
3. Group by Gaussian factor: :func:`chebyshev_filter.group_by_exp_combined`.
4. Apply Horner form: :func:`chebyshev_filter.apply_horner`.
5. Call :func:`build_julia_batch_script` to get a complete standalone ``.jl``.
6. Call ``julia script.jl grid.bin kvals.bin out.bin N n_waves`` to evaluate.

Legacy: :func:`build_julia_scalar_function` + :func:`build_julia_benchmark_script`
build a scalar function + standalone benchmark (useful for timing a single point).

Performance design
------------------
The generated batch script mirrors the optimizations from the old Julia notebook:

* **Separate @inline functions** per group (exp part + poly part) — lets Julia's
  JIT specialise and inline each piece independently.
* **Precompute exp terms once** (``precompute_exp!``) for the full grid.  For
  plane-wave bases (no Gaussian factors) this is trivial (stores 1.0); for
  Gaussian bases it avoids recomputing ``exp(...)`` per wave.
* **Vectorised trig** (``@.`` broadcast on pre-allocated buffers) so Julia uses
  its SLEEF SIMD math library for cos/sin rather than evaluating them one
  scalar at a time.
* **Warmup wave** triggers JIT compilation; the remaining waves are timed
  separately.  Timings are printed as JSON to stdout so Python can parse them.
"""

import sympy as sp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _jl(expr):
    """Convert sympy expression to Julia code, stripping broadcast operators."""
    s = sp.julia_code(expr)
    # sp.julia_code may emit .+, .*, etc. for some expressions; strip the dots
    # so the code is valid for scalar evaluation.
    for op in ('.+', '.-', '.*', './', '.^'):
        s = s.replace(op, op[1:])
    return s


def expr_to_julia_code(expr):
    """Convert a single sympy expression to a Julia code string."""
    return _jl(expr)


def build_julia_scalar_function(terms_cos, terms_sin, func_name='f_psi'):
    """Build a Julia function that evaluates f(H)*psi at a single grid point.

    The generated function computes
        f_psi(x,y,z,kx,ky,kz,b)
          = (Σ_i exp_i(x,y,z) * poly_i(x,y,z,kx,ky,kz)) * cos(kx*x+ky*y+kz*z+b)
          + (Σ_j exp_j(x,y,z) * poly_j(x,y,z,kx,ky,kz)) * sin(kx*x+ky*y+kz*z+b)

    Parameters
    ----------
    terms_cos, terms_sin : list of (exp_part, poly_part)
        Output of :func:`chebyshev_filter.group_by_exp_combined`
        (optionally post-processed with :func:`chebyshev_filter.apply_horner`).
    func_name : str

    Returns
    -------
    str  Julia source code
    """
    lines = [f'function {func_name}(x::Float64, y::Float64, z::Float64,',
             f'                    kx::Float64, ky::Float64, kz::Float64,',
             f'                    b::Float64)::Float64']

    cos_vars = []
    for i, (ep, pp) in enumerate(terms_cos):
        var = f'_ccos{i}'
        lines.append(f'    {var} = ({_jl(ep)}) * ({_jl(pp)})')
        cos_vars.append(var)

    sin_vars = []
    for i, (ep, pp) in enumerate(terms_sin):
        var = f'_csin{i}'
        lines.append(f'    {var} = ({_jl(ep)}) * ({_jl(pp)})')
        sin_vars.append(var)

    phase   = 'kx*x + ky*y + kz*z + b'
    cos_sum = ' + '.join(cos_vars) if cos_vars else '0.0'
    sin_sum = ' + '.join(sin_vars) if sin_vars else '0.0'
    lines.append(f'    return ({cos_sum}) * cos({phase}) + ({sin_sum}) * sin({phase})')
    lines.append('end')
    return '\n'.join(lines)


def build_julia_batch_script(terms_cos, terms_sin):
    """Generate a complete standalone Julia script for batch grid evaluation.

    Implements the optimisations from the original Julia notebook:

    * Separate ``@inline _poly_cos_i`` / ``_poly_sin_i`` functions per group
      (polynomial part only, no trig) so Julia's JIT can specialise each.
    * Separate ``@inline _exp_cos_i`` / ``_exp_sin_i`` for the Gaussian
      factor; ``precompute_exp!`` evaluates them once for the whole grid.
    * ``eval_one_wave!`` uses vectorised ``@.`` broadcasting for cos/sin
      (SLEEF SIMD) on pre-allocated buffers, then a scalar ``@inbounds``
      loop for the polynomial part.
    * The first wave is a **warmup** (triggers JIT); the remaining waves are
      timed.  Both times are printed as JSON to stdout.

    Call convention::

        julia eval_filter.jl grid.bin kvals.bin out.bin N n_waves

    * ``grid.bin``  – ``[X_flat; Y_flat; Z_flat]`` (3N float64, little-endian)
    * ``kvals.bin`` – ``[kx0,ky0,kz0,b0, kx1,…]``  (4*n_waves float64)
    * ``out.bin``   – result: n_waves*N float64, wave-major order
    * Stdout line (last): JSON with warmup_s, eval_s, n_eval_waves

    Parameters
    ----------
    terms_cos, terms_sin : list of (exp_part, poly_part)
        From :func:`chebyshev_filter.group_by_exp_combined`, optionally
        post-processed with :func:`chebyshev_filter.apply_horner`.

    Returns
    -------
    str  Complete Julia source ready to write to a ``.jl`` file.
    """
    n_cos = len(terms_cos)
    n_sin = len(terms_sin)

    L = ['# Auto-generated by symbolic_code.julia_codegen — DO NOT EDIT',
         '# Evaluates the Chebyshev-filtered plane-wave basis on a 3-D grid.',
         '#',
         '# Usage:',
         '#   julia eval_filter.jl grid.bin kvals.bin out.bin N n_waves',
         '#',
         '# Stdout (last line): JSON timing  {warmup_s, eval_s, n_eval_waves}',
         '']

    # --- per-group polynomial functions (no trig, no exp) ---
    for i, (ep, pp) in enumerate(terms_cos):
        L += [f'@inline function _poly_cos_{i}(x::Float64, y::Float64, z::Float64,',
              f'                               kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}',
              'end', '']

    for i, (ep, pp) in enumerate(terms_sin):
        L += [f'@inline function _poly_sin_{i}(x::Float64, y::Float64, z::Float64,',
              f'                               kx::Float64, ky::Float64, kz::Float64)::Float64',
              f'    return {_jl(pp)}',
              'end', '']

    # --- per-group Gaussian exp functions (trivial = 1 for plane-wave basis) ---
    for i, (ep, pp) in enumerate(terms_cos):
        L += [f'@inline function _exp_cos_{i}(x::Float64, y::Float64, z::Float64)::Float64',
              f'    return {_jl(ep)}',
              'end', '']

    for i, (ep, pp) in enumerate(terms_sin):
        L += [f'@inline function _exp_sin_{i}(x::Float64, y::Float64, z::Float64)::Float64',
              f'    return {_jl(ep)}',
              'end', '']

    # --- precompute_exp! (called once per grid, before the wave loop) ---
    L += ['function precompute_exp!(exp_cos::Matrix{Float64}, exp_sin::Matrix{Float64},',
          '                         X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64})',
          '    N = length(X)',
          '    @inbounds for i in 1:N']
    for i in range(n_cos):
        L.append(f'        exp_cos[i,{i+1}] = _exp_cos_{i}(X[i], Y[i], Z[i])')
    for i in range(n_sin):
        L.append(f'        exp_sin[i,{i+1}] = _exp_sin_{i}(X[i], Y[i], Z[i])')
    L += ['    end', 'end', '']

    # --- eval_one_wave! ---
    # Uses pre-allocated phase/cos_p/sin_p buffers to avoid per-wave allocation.
    # Vectorised trig via Julia's @. (SLEEF SIMD) then scalar poly loop.
    # IMPORTANT: use sequential += accumulation (not a single joined expression)
    # so that Julia's JIT can optimise each statement independently.
    cos_accum = '\n'.join(
        f'        _sum_cos += exp_cos[i,{i+1}]'
        f' * _poly_cos_{i}(X[i], Y[i], Z[i], kx, ky, kz)'
        for i in range(n_cos)
    )
    sin_accum = '\n'.join(
        f'        _sum_sin += exp_sin[i,{i+1}]'
        f' * _poly_sin_{i}(X[i], Y[i], Z[i], kx, ky, kz)'
        for i in range(n_sin)
    )

    L += ['function eval_one_wave!(out::AbstractVector{Float64},',
          '                        X::Vector{Float64}, Y::Vector{Float64}, Z::Vector{Float64},',
          '                        kx::Float64, ky::Float64, kz::Float64, b::Float64,',
          '                        exp_cos::Matrix{Float64}, exp_sin::Matrix{Float64},',
          '                        phase::Vector{Float64}, cos_p::Vector{Float64},',
          '                        sin_p::Vector{Float64})',
          '    # Vectorised phase + trig (SLEEF SIMD, no per-wave allocation)',
          '    @. phase = kx * X + ky * Y + kz * Z + b',
          '    @. cos_p = cos(phase)',
          '    @. sin_p = sin(phase)',
          '    N = length(X)',
          '    @inbounds for i in 1:N',
          '        _sum_cos = 0.0',
          (cos_accum if cos_accum else '        # no cos terms'),
          '        _sum_sin = 0.0',
          (sin_accum if sin_accum else '        # no sin terms'),
          '        out[i] = cos_p[i] * _sum_cos + sin_p[i] * _sum_sin',
          '    end',
          'end', '']

    # --- main ---
    nc_alloc = max(n_cos, 1)
    ns_alloc = max(n_sin, 1)
    L += ['function main()',
          '    length(ARGS) == 5 || error(',
          '        "Usage: julia eval_filter.jl grid.bin kvals.bin out.bin N n_waves")',
          '    grid_file  = ARGS[1]',
          '    kvals_file = ARGS[2]',
          '    out_file   = ARGS[3]',
          '    N          = parse(Int, ARGS[4])',
          '    n_waves    = parse(Int, ARGS[5])',
          '',
          '    # Read grid (X, Y, Z flat float64)',
          '    buf = Vector{Float64}(undef, 3 * N)',
          '    open(grid_file, "r") do io; read!(io, buf); end',
          '    X = buf[1:N]; Y = buf[N+1:2N]; Z = buf[2N+1:3N]',
          '',
          '    # Read k/b values: [kx0,ky0,kz0,b0, kx1,…]',
          '    kb = Vector{Float64}(undef, 4 * n_waves)',
          '    open(kvals_file, "r") do io; read!(io, kb); end',
          '',
          '    # Precompute Gaussian exp terms once for all waves',
          f'    exp_cos = ones(Float64, N, {nc_alloc})',
          f'    exp_sin = ones(Float64, N, {ns_alloc})',
          '    precompute_exp!(exp_cos, exp_sin, X, Y, Z)',
          '',
          '    # Pre-allocate working buffers (avoid per-wave heap allocation)',
          '    out   = Vector{Float64}(undef, N * n_waves)',
          '    phase = Vector{Float64}(undef, N)',
          '    cos_p = Vector{Float64}(undef, N)',
          '    sin_p = Vector{Float64}(undef, N)',
          '',
          '    # Warmup: evaluate first wave to trigger JIT compilation',
          '    t_warmup = @elapsed eval_one_wave!(',
          '        @view(out[1:N]), X, Y, Z, kb[1], kb[2], kb[3], kb[4],',
          '        exp_cos, exp_sin, phase, cos_p, sin_p)',
          '',
          '    # Timed eval: remaining waves (no JIT overhead)',
          '    t_eval = @elapsed for iw in 1:n_waves-1',
          '        ofs = iw * N + 1',
          '        kid = iw * 4 + 1',
          '        eval_one_wave!(@view(out[ofs:ofs+N-1]), X, Y, Z,',
          '                       kb[kid], kb[kid+1], kb[kid+2], kb[kid+3],',
          '                       exp_cos, exp_sin, phase, cos_p, sin_p)',
          '    end',
          '',
          '    open(out_file, "w") do io; write(io, out); end',
          '',
          '    n_eval = max(n_waves - 1, 1)',
          '    println("{\\\"warmup_s\\\": $t_warmup, \\\"eval_s\\\": $t_eval,' +
          ' \\\"n_eval_waves\\\": $n_eval}")',
          'end',
          '',
          'main()']

    return '\n'.join(L)


def build_julia_hn_cse_script(expr_cos, expr_sin):
    """Generate Julia script evaluating H^n via CSE on the raw SymPy tree.

    Unlike build_julia_batch_script, this does NOT expand or group by
    Gaussian factor.  Instead it applies sp.cse() to the unexpanded
    expressions from H_power_n.pkl, preserving the implicit sharing in
    the SymPy tree.  The result has far fewer FLOP when the expressions
    contain many repeated subexpressions (r^2, cross-terms, etc.).

    Parameters
    ----------
    expr_cos, expr_sin : sympy expressions
        H^n·ψ split into cos and sin envelopes (data['Pc'], data['Ps']
        from H_power_n.pkl).  Should NOT be pre-expanded.

    Returns
    -------
    (str, int)  Julia source, total_ops (ADD + MUL after CSE)

    Call convention (same as build_julia_batch_script)::

        julia eval_Hn_cse_m{n}.jl grid.bin kvals.bin out.bin N n_waves
    """
    import sympy as sp

    repl, (red_cos, red_sin) = sp.cse(
        [expr_cos, expr_sin], symbols=sp.numbered_symbols('_t'))

    # Count ops for reporting
    ops_temps   = sum(int(sp.count_ops(e)) for _, e in repl)
    ops_reduced = int(sp.count_ops(red_cos)) + int(sp.count_ops(red_sin))
    total_ops   = ops_temps + ops_reduced

    temp_block = (
        '\n'.join(f'        {_jl(sym)} = {_jl(e)}' for sym, e in repl)
        if repl else '        # no CSE temps'
    )

    L = ['# Auto-generated by symbolic_code.julia_codegen (H^n CSE) — DO NOT EDIT',
         '# Evaluates H^n·ψ preserving SymPy tree structure via sp.cse().',
         f'# total_ops={total_ops}  (temps={ops_temps}  reduced={ops_reduced})',
         '#',
         '# Usage: julia eval_Hn_cse_m{n}.jl grid.bin kvals.bin out.bin N n_waves',
         '']

    # eval_one_wave!  — no separate exp precompute (Gaussians inlined via CSE)
    L += ['function eval_one_wave!(out::AbstractVector{Float64},',
          '                        X::Vector{Float64}, Y::Vector{Float64},'
          ' Z::Vector{Float64},',
          '                        kx::Float64, ky::Float64, kz::Float64,'
          ' b::Float64,',
          '                        phase::Vector{Float64},'
          ' cos_p::Vector{Float64}, sin_p::Vector{Float64})',
          '    @. phase = kx * X + ky * Y + kz * Z + b',
          '    @. cos_p = cos(phase)',
          '    @. sin_p = sin(phase)',
          '    N = length(X)',
          '    @inbounds for i in 1:N',
          '        x, y, z = X[i], Y[i], Z[i]',
          temp_block,
          f'        out[i] = cos_p[i] * ({_jl(red_cos)})'
          f' + sin_p[i] * ({_jl(red_sin)})',
          '    end',
          'end', '']

    # main()  — same CLI as build_julia_batch_script, no exp_cos/exp_sin buffers
    L += ['function main()',
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
          'main()']

    return '\n'.join(L), total_ops


def build_julia_benchmark_script(julia_func_src, n_points=10_000, n_iter=40):
    """Wrap a Julia scalar function in a simple benchmark harness.

    Parameters
    ----------
    julia_func_src : str
        Output of :func:`build_julia_scalar_function`.
    n_points : int
        Number of random evaluation points.
    n_iter : int
        Number of timing iterations.

    Returns
    -------
    str  Complete Julia script ready to run with ``julia script.jl``.
    """
    return f"""\
# Auto-generated Julia benchmark script
{julia_func_src}

using Random
Random.seed!(42)
n = {n_points}
xs = randn(n); ys = randn(n); zs = randn(n)
kx0 = 1.0; ky0 = 0.0; kz0 = 0.0; b0 = 0.0

result = similar(xs)
println("Warming up ...")
for i in 1:n
    result[i] = f_psi(xs[i], ys[i], zs[i], kx0, ky0, kz0, b0)
end

println("Benchmarking ({n_iter} iterations x $n points) ...")
@time begin
    for _ in 1:{n_iter}
        for i in 1:n
            result[i] = f_psi(xs[i], ys[i], zs[i], kx0, ky0, kz0, b0)
        end
    end
end
GC.gc()
println("Done.")
"""
