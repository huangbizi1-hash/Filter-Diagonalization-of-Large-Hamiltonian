import pickle
from pathlib import Path
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import FFT_func
import random
import pickle
import gzip
from pathlib import Path
import sympy as sp
from sympy import symbols, exp, Add, Mul, collect, expand
from scipy.linalg import svd, eigh

# (n, N, N, N)->(N**3, n)
def grid_to_vec(y_grid):
    return y_grid.reshape(y_grid.shape[0], -1).T

# (N**3, n)->(n, N, N, N)
def vec_to_grid(y_grid, Nx, Ny, Nz):
    return y_grid.T.reshape(y_grid.shape[-1], Nx, Ny, Nz)



def svd_H(filtered_psi_matrix):
    C_f_grid = filtered_psi_matrix.copy()
    C_f = grid_to_vec(C_f_grid)
    C_f = C_f / np.linalg.norm(C_f, axis=0)
    inf_columns = np.any(np.isinf(C_f), axis=0)
    C_f = C_f[:, ~inf_columns]

    print("Performing QR decomposition...")
    Q_qr, R_qr = np.linalg.qr(C_f, mode='reduced')
    print(f"Q_qr dimensions: {Q_qr.shape}")
    print(f"R_qr dimensions: {R_qr.shape}")
    print("QR decomposition complete.")

    print("Performing SVD on R_qr...")
    U1_svd, Sigma_svd, V_svd_T = np.linalg.svd(R_qr, full_matrices=False)
    V_svd = V_svd_T.T
    print("SVD on R_qr complete.")

    print("Calculating U = Q_qr @ U1_svd...")
    U = Q_qr @ U1_svd
    print(f"Final U dimensions: {U.shape}")
    print(f"Sigma dimensions: {Sigma_svd.shape}")
    print(f"V_svd dimensions: {V_svd.shape}")

    rank_threshold = 1e-3
    r = np.sum(Sigma_svd > rank_threshold)
    print(f"Selected rank (r): {r}")

    Ur = U[:, :r]
    print(f"U_r dimensions: {Ur.shape}")

    Ur_grid = vec_to_grid(Ur, Nx, Ny, Nz)
    H_Ur_grid = np.zeros_like(Ur_grid, dtype='complex128')
    for i in range(H_Ur_grid.shape[0]):
        H_Ur_grid[i, :, :, :] = FFT_func.scaled_fft_eval_H_psi(Ur_grid[i, :, :, :], x_grid, V_on_grid)

    H_Ur = grid_to_vec(H_Ur_grid)
    H_tilde = Ur.T.conj() @ H_Ur
    energies_filtered, _ = eigh(H_tilde)
    energies_filtered = np.sort(energies_filtered)
    energies_filtered = energies_filtered[:200]
    return energies_filtered, Ur

def load_expr_srepr(path):
    """从 .srepr 或 .sym.gz 文件加载表达式"""
    if str(path).endswith(".gz"):
        with gzip.open(path, 'rt', encoding='utf8') as f:
            s = f.read()
    else:
        with open(path, 'r', encoding='utf8') as f:
            s = f.read()
    data = sp.sympify(s, evaluate=False)
    return data

def load_H_powers(folder, n_max, file_type='sym.gz'):
    folder = Path(folder)
    results = {}
    
    if file_type == 'pkl':
        pattern = "H_scaled_power_*.pkl"
    elif file_type == 'sym.gz':
        pattern = "H_scaled_power_*.sym.gz"
    else:
        raise ValueError(f"Unsupported file_type: {file_type}. Use 'pkl' or 'sym.gz'")
    
    for f in sorted(folder.glob(pattern)):
        n = int(f.stem.split('_')[-1] if file_type == 'pkl' 
                else f.name.split('_')[-1].split('.')[0])
        
        if file_type == 'pkl':
            with open(f, 'rb') as fh:
                data = pickle.load(fh)
        elif file_type == 'sym.gz':
            data = load_expr_srepr(f)
        
        results[n] = data
        if n == n_max:
            break
    
    return results

def chebyshev_coeffs_transformed(n, a=1.0, b=0.0):
    x = sp.Symbol('x')
    Tn = sp.chebyshevt(n, x)
    # 将 x 替换为 a*x + b
    Tn_transformed = Tn.subs(x, a*x + b)
    # 展开并提取系数
    Tn_expanded = sp.expand(Tn_transformed)
    poly = sp.Poly(Tn_expanded, x)
    coeffs = [poly.coeff_monomial(x**i) for i in range(n+1)]
    return coeffs

def apply_f_of_H_on_psi(folder, f_coeffs, n_max):
    results = load_H_powers(folder, n_max)
    N = len(f_coeffs) - 1
    x, y, z, kx, ky, kz, b = sp.symbols('x y z kx ky kz b')
    theta = kx*x + ky*y + kz*z + b

    # 初始化总和
    Ps_total = 0
    Pc_total = 0
    for n, c in enumerate(f_coeffs[1:]):  # 从 1 开始
        print('n',n)
        Ps_total += c * results[n+1]['Ps']
        Pc_total += c * results[n+1]['Pc']


    # 返回最终的符号表达式
    psi_fH = (Ps_total * sp.sin(theta) + Pc_total * sp.cos(theta)) + f_coeffs[0] * sp.sin(theta)
    return psi_fH

folder = "H_symbolic_basis_expression_gaussian"

n_max = 5
'''cheby_coeffs = chebyshev_coeffs_transformed(n_max)
print(cheby_coeffs)
psi_fH = apply_f_of_H_on_psi(folder, cheby_coeffs, n_max)'''
E_lower = 0.0
E_upper = 20.0
a0_val = 2 / (E_upper - E_lower)
b0_val = -(E_upper + E_lower) / (E_upper - E_lower)
cheby_coeffs = chebyshev_coeffs_transformed(n_max, a=a0_val, b=b0_val)
print(cheby_coeffs)
psi_fH = apply_f_of_H_on_psi(folder, cheby_coeffs, n_max)

from sympy import powsimp
from sympy import symbols, collect, exp, cos, sin, Add, Mul, simplify
from collections import defaultdict

# 第一步：提取 cos 和 sin 的系数（同上）
x, y, z, kx, ky, kz, b = symbols('x y z kx ky kz b')
phase = b + kx*x + ky*y + kz*z

expr_expanded = psi_fH.expand()
expr1 = expr_expanded.coeff(cos(phase), 1) or 0
expr2 = expr_expanded.coeff(sin(phase), 1) or 0

def group_by_exp_combined(expr):
    """
    将表达式按相同的 exp 项分组，先合并 exp
    返回列表：[(exp_part, poly_part), ...]
    """
    if expr == 0:
        return []
    
    # 先合并所有 exp
    expr = powsimp(expr, combine='all')
    
    grouped = {}
    
    for term in Add.make_args(expr):
        # 分离 exp 和非 exp 部分
        exp_part = 1
        poly_part = 1
        
        for factor in Mul.make_args(term):
            if factor.has(exp):
                exp_part *= factor
            else:
                poly_part *= factor
        
        # 累加相同 exp 项
        if exp_part in grouped:
            grouped[exp_part] += poly_part
        else:
            grouped[exp_part] = poly_part
    
    return list(grouped.items())

# 使用
terms1 = group_by_exp_combined(expr1)
terms2 = group_by_exp_combined(expr2)

term_cos = cos(phase)
term_sin = sin(phase)

use_horner = 1
if use_horner:
    terms1 = [(term[0], sp.horner(term[1])) for term in terms1]
    terms2 = [(term[0], sp.horner(term[1])) for term in terms2]      

'''print("expr1:")
for i, (exp_p, poly_p) in enumerate(terms1):
    if exp_p == 1:
        print(f"term1_{i} = {poly_p}")
    else:
        print(f"term1_{i} = {exp_p} * ({poly_p})")

print("\nexpr2:")
for i, (exp_p, poly_p) in enumerate(terms2):
    if exp_p == 1:
        print(f"term2_{i} = {poly_p}")
    else:
        print(f"term2_{i} = {exp_p} * ({poly_p})")'''
