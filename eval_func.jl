using SymPy
using PyCall
cd("c:/FD")   # 假设 fH_func.py 在这个文件夹
pushfirst!(PyCall.PyVector(PyCall.pyimport("sys")."path"), pwd())
fH_func = pyimport("fH_func")
using PyCall
importlib = pyimport("importlib")
fH_func = pyimport("fH_func")
importlib.reload(fH_func)
grid_data = pyimport("grid_data")
using PyCall
importlib = pyimport("importlib")
grid_data = pyimport("grid_data")
importlib.reload(grid_data)
kx = convert(Float64, grid_data."kx_val")
ky = convert(Float64, grid_data."ky_val")
kz = convert(Float64, grid_data."kz_val")
b  = convert(Float64, grid_data."b_val")
h  = convert(Float64, grid_data."h_val")
mu = convert(Float64, grid_data."mu_val")
a0 = convert(Float64, grid_data."a0_val")
b0 = convert(Float64, grid_data."b0_val")
x = convert(Array{Float64}, grid_data."x")
y = convert(Array{Float64}, grid_data."y")
z = convert(Array{Float64}, grid_data."z")
a0 = 1.0
b0 = 0.0
#result = f.(x, y, z, kx, ky, kz, b, h, mu)

using PyCall
using SymPy

sympy = pyimport("sympy")
julia_code = sympy.julia_code(fH_func.psi_fH)

using MacroTools

function devectorize(expr)
    MacroTools.postwalk(expr) do x
        # 将 .+ 转为 +, .- 转为 -, .* 转为 *, ./ 转为 / 等
        if @capture(x, a_.b_)
            return :($a.$b)  # 保留字段访问
        elseif x isa Expr && x.head == :.
            # 处理广播调用 f.(args...)
            return Expr(:call, x.args[1], x.args[2:end]...)
        elseif x isa Expr && x.head == :call
            # 处理点运算符 .+, .-, .*, ./ 等
            if x.args[1] isa Symbol
                op_str = string(x.args[1])
                if startswith(op_str, ".")
                    new_op = Symbol(op_str[2:end])  # 去掉开头的点
                    return Expr(:call, new_op, x.args[2:end]...)
                end
            end
        end
        return x
    end
end

# 使用示例
julia_code1 = "x .+ y .* z ./ 2.0"
expr = Meta.parse(julia_code)
println("原始表达式: ", expr)

devec_expr = devectorize(expr)
println("去向量化后: ", devec_expr)

# 生成函数
eval(quote
    function f1(x, y, z, kx, ky, kz, b, h, mu)
        return $(devec_expr)
    end
end)

n_iterations = 40
result = similar(x)
println("开始 $n_iterations 次迭代...")
@time begin
    for j in 1:n_iterations
        for i in 1:length(x)
            result[i] = f1(x[i], y[i], z[i], kx, ky, kz, b, h, mu)
            
        end
    end
end
GC.gc()
println("完成!")