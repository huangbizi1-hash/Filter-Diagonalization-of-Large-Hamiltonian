# eval_func.jl
# Julia script: convert a Python sympy expression to a compiled Julia function
# and benchmark its evaluation speed.
#
# Prerequisites (Julia packages): SymPy, PyCall
# Prerequisites (Python side):    symbolic_code package on PYTHONPATH
#
# Usage:
#   cd <project_root>
#   julia symbolic_code/eval_func.jl

using PyCall

# ---------------------------------------------------------------------------
# 1. Make the symbolic_code Python package importable
# ---------------------------------------------------------------------------
pushfirst!(PyCall.PyVector(PyCall.pyimport("sys")."path"), pwd())

importlib = pyimport("importlib")

# Reload helpers so edits in Python are picked up without restarting Julia
fH_func_mod = pyimport("symbolic_code.chebyshev_filter")
importlib.reload(fH_func_mod)

# ---------------------------------------------------------------------------
# 2. Load grid / wave-vector data from a companion Python module (grid_data.py)
#    That module must define: kx_val, ky_val, kz_val, b_val, x, y, z
# ---------------------------------------------------------------------------
grid_data = pyimport("grid_data")
importlib.reload(grid_data)

kx = convert(Float64, grid_data."kx_val")
ky = convert(Float64, grid_data."ky_val")
kz = convert(Float64, grid_data."kz_val")
b  = convert(Float64, grid_data."b_val")
x  = convert(Array{Float64}, grid_data."x")
y  = convert(Array{Float64}, grid_data."y")
z  = convert(Array{Float64}, grid_data."z")

# ---------------------------------------------------------------------------
# 3. Build the sympy expression for f(H)*psi and convert to Julia source
# ---------------------------------------------------------------------------
sympy = pyimport("sympy")

# chebyshev_filter.psi_fH must be set from outside, or computed here:
# example: psi_fH = fH_func_mod.apply_f_of_H_on_psi(folder, coeffs, n_max)
psi_fH    = fH_func_mod."psi_fH"   # sympy expression
julia_src = sympy.julia_code(psi_fH)

# ---------------------------------------------------------------------------
# 4. Strip broadcast dots (.+, .*, ...) so the expression is scalar
# ---------------------------------------------------------------------------
using MacroTools

function devectorize(expr)
    MacroTools.postwalk(expr) do x
        if x isa Expr && x.head == :call
            op = x.args[1]
            if op isa Symbol
                s = string(op)
                if startswith(s, ".")
                    new_op = Symbol(s[2:end])
                    return Expr(:call, new_op, x.args[2:end]...)
                end
            end
        end
        return x
    end
end

parsed_expr = Meta.parse(julia_src)
scalar_expr = devectorize(parsed_expr)

# ---------------------------------------------------------------------------
# 5. Compile a scalar Julia function f_eval(x,y,z,kx,ky,kz,b)
# ---------------------------------------------------------------------------
eval(quote
    function f_eval(x, y, z, kx, ky, kz, b)
        return $(scalar_expr)
    end
end)

# ---------------------------------------------------------------------------
# 6. Benchmark
# ---------------------------------------------------------------------------
n_iterations = 40
result = similar(x)

println("Warming up ...")
for i in eachindex(x)
    result[i] = f_eval(x[i], y[i], z[i], kx, ky, kz, b)
end

println("Benchmarking ($n_iterations iterations x $(length(x)) points) ...")
@time begin
    for _ in 1:n_iterations
        for i in eachindex(x)
            result[i] = f_eval(x[i], y[i], z[i], kx, ky, kz, b)
        end
    end
end
GC.gc()
println("Done.")
