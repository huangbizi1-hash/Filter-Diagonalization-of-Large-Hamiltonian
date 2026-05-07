"""Python-side utilities for generating Julia source code from sympy expressions.

Typical usage
-------------
1. Compute psi_fH via :func:`chebyshev_filter.apply_f_of_H_on_psi`.
2. Extract and group terms with :func:`chebyshev_filter.extract_cos_sin_coeffs`
   and :func:`chebyshev_filter.group_by_exp_combined`.
3. Optionally apply Horner form with :func:`chebyshev_filter.apply_horner`.
4. Call :func:`build_julia_scalar_function` to get a Julia source string.
5. Optionally wrap it with :func:`build_julia_benchmark_script`.
6. Write the string to a ``.jl`` file and load it from Julia via ``include``.
"""

import sympy as sp


def expr_to_julia_code(expr):
    """Convert a single sympy expression to a Julia code string."""
    return sp.julia_code(expr)


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
    lines = [f'function {func_name}(x, y, z, kx, ky, kz, b)']

    cos_vars = []
    for i, (ep, pp) in enumerate(terms_cos):
        var = f'_ccos{i}'
        lines.append(f'    {var} = ({sp.julia_code(ep)}) * ({sp.julia_code(pp)})')
        cos_vars.append(var)

    sin_vars = []
    for i, (ep, pp) in enumerate(terms_sin):
        var = f'_csin{i}'
        lines.append(f'    {var} = ({sp.julia_code(ep)}) * ({sp.julia_code(pp)})')
        sin_vars.append(var)

    phase   = 'kx*x + ky*y + kz*z + b'
    cos_sum = ' + '.join(cos_vars) if cos_vars else '0.0'
    sin_sum = ' + '.join(sin_vars) if sin_vars else '0.0'
    lines.append(f'    return ({cos_sum}) * cos({phase}) + ({sin_sum}) * sin({phase})')
    lines.append('end')
    return '\n'.join(lines)


def build_julia_benchmark_script(julia_func_src, n_points=10_000, n_iter=40):
    """Wrap a Julia function in a simple benchmark harness.

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
