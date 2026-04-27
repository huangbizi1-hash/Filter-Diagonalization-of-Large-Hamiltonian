# 用于为所有立方体生成H^n表达式，每个立方体有独立的子文件夹
# save_hn_scaled_partitioned_fixed.py
import sympy as sp
import pickle
import gzip
import numpy as np
import sys
from pathlib import Path
from load_partitioned_Vexpr import CubicExpressionManager


def laplacian(f, x, y, z):
    return sp.diff(f, x, 2) + sp.diff(f, y, 2) + sp.diff(f, z, 2)

def directional_derivative(f, kvec, x, y, z):
    kx, ky, kz = kvec
    return kx*sp.diff(f, x) + ky*sp.diff(f, y) + kz*sp.diff(f, z)

def apply_H_on_pair(Ps, Pc, V, kvec, k2, pref, x, y, z):
    """Compute (Ps_new, Pc_new) = H(Ps*sin + Pc*cos)."""
    lap_Ps = laplacian(Ps, x, y, z)
    dir_Ps = directional_derivative(Ps, kvec, x, y, z)
    lap_Pc = laplacian(Pc, x, y, z)
    dir_Pc = directional_derivative(Pc, kvec, x, y, z)

    Ps_new = sp.Add((pref*k2)*Ps, -pref*lap_Ps, dir_Pc, V*Ps, evaluate=False)
    Pc_new = sp.Add((pref*k2)*Pc, -pref*lap_Pc, -dir_Ps, V*Pc, evaluate=False)

    return Ps_new, Pc_new

# ---- 生成 (aH + b)^n psi ----
def generate_scaled_H_powers(N, a, b, outdir="H_scaled_powers", file_format='sym.gz', 
                            V=None, kvec=None, k2=None, pref=None, x=None, y=None, z=None):
    """
    生成 (aH+b)^n psi 并保存为字典格式的文件
    
    Parameters:
    -----------
    N : int
        最大幂次
    a : float
        缩放参数 a
    b : float
        缩放参数 b
    outdir : str or Path
        输出目录
    file_format : str
        文件格式，'sym.gz' 或 'pkl'
    V : sympy expression
        势能表达式
    kvec, k2, pref, x, y, z : 
        符号变量和参数
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 初始化 ψ₀ = sin(theta)
    Ps = sp.Integer(1)
    Pc = sp.Integer(0)

    results = {0: {'Ps': Ps, 'Pc': Pc}}  # (aH+b)^0 ψ = ψ

    for n in range(1, N+1):
        print(f"    Computing (aH+b)^{n}...", end=' ')
        Ps_H, Pc_H = apply_H_on_pair(Ps, Pc, V, kvec, k2, pref, x, y, z)
        Ps_new = a * Ps_H + b * Ps
        Pc_new = a * Pc_H + b * Pc
        Ps, Pc = Ps_new, Pc_new
        results[n] = {'Ps': Ps, 'Pc': Pc}
        
        # 保存为可加载的格式
        if file_format == 'sym.gz':
            # 保存为符号表达式字符串的压缩格式
            with gzip.open(outdir / f'H_scaled_power_{n}.sym.gz', 'wt', encoding='utf8') as f:
                data_str = str({'Ps': results[n]['Ps'], 'Pc': results[n]['Pc']})
                f.write(data_str)
        elif file_format == 'pkl':
            # 保存为pickle格式
            with open(outdir / f'H_scaled_power_{n}.pkl', 'wb') as f:
                pickle.dump(results[n], f)
        
        print(f"✓")

    return results


def process_all_cubes(manager, N, a, b, base_outdir="H_powers_basis_partitioned", 
                     file_format='sym.gz'):
    """
    为所有立方体生成H^n表达式
    
    Parameters:
    -----------
    manager : CubicExpressionManager
        立方体表达式管理器
    N : int
        最大幂次
    a, b : float
        缩放参数
    base_outdir : str
        基础输出目录
    file_format : str
        文件格式
    """
    base_outdir = Path(base_outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    
    # 符号定义
    x, y, z = sp.symbols('x y z')
    kx, ky, kz = sp.symbols('kx ky kz')
    b_sym = sp.symbols('b')
    
    kvec = (kx, ky, kz)
    theta = kx*x + ky*y + kz*z + b_sym
    k2 = kx**2 + ky**2 + kz**2
    pref = 0.5
    
    # 统计信息
    total_cubes = len(manager)
    processed = 0
    skipped = 0
    
    print(f"\n{'='*70}")
    print(f"PROCESSING ALL CUBES")
    print(f"{'='*70}")
    print(f"Total cubes to process: {total_cubes}")
    print(f"Maximum power N: {N}")
    print(f"Scaling parameters: a={a:.6f}, b={b:.6f}")
    print(f"Output directory: {base_outdir}")
    print(f"File format: {file_format}\n")
    
    # 遍历所有立方体
    for idx in sorted(manager.expressions.keys()):
        i, j, k = idx
        
        print(f"[{processed+1}/{total_cubes}] Cube ({i}, {j}, {k})...")
        
        # 获取势能表达式
        V_expr = manager.get_expression(i, j, k)
        
        if V_expr is None:
            print(f"  ✗ No expression found, skipping\n")
            skipped += 1
            continue
        
        # 获取立方体信息
        cube_info = manager.cube_info[idx]
        center = cube_info['center']
        n_atoms = cube_info['n_atoms']
        
        print(f"  Center: ({center[0]:6.2f}, {center[1]:6.2f}, {center[2]:6.2f}), Atoms: {n_atoms}")
        
        # 创建子文件夹：cube_i_j_k
        #cube_dir = base_outdir / f"cube_{i}_{j}_{k}"
        #cube_dir.mkdir(parents=True, exist_ok=True)

        # 创建子文件夹：使用中心坐标命名
        cx, cy, cz = center
        # 处理负号：将 - 替换为 neg
        cx_str = f"{cx:.3f}".replace("-", "neg")
        cy_str = f"{cy:.3f}".replace("-", "neg")
        cz_str = f"{cz:.3f}".replace("-", "neg")
        cube_dir = base_outdir / f"cube_{cx_str}_{cy_str}_{cz_str}"
        cube_dir.mkdir(parents=True, exist_ok=True)

        # 保存立方体信息到子文件夹
        info_file = cube_dir / "cube_info.txt"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Cube Index: ({i}, {j}, {k})\n")
            f.write(f"Center: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}) Bohr\n")
            f.write(f"Number of atoms: {n_atoms}\n")
            f.write(f"Cube size: {cube_info['cube_size']:.6f} Bohr\n")
            f.write(f"Cutoff radius: {cube_info['r_cut']:.6f} Bohr\n")
            f.write(f"\nScaling parameters:\n")
            f.write(f"  a = {a:.6f}\n")
            f.write(f"  b = {b:.6f}\n")
            f.write(f"\nGenerated H^n powers: 0 to {N}\n")
        
        # 生成H^n表达式
        try:
            results = generate_scaled_H_powers(
                N, a, b,
                outdir=cube_dir,
                file_format=file_format,
                V=V_expr,
                kvec=kvec,
                k2=k2,
                pref=pref,
                x=x, y=y, z=z
            )
            print(f"  ✓ Generated H^0 to H^{N} in {cube_dir.name}/\n")
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error generating H^n: {e}\n")
            skipped += 1
            continue
    
    # 总结
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total cubes: {total_cubes}")
    print(f"Successfully processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"\nOutput structure:")
    print(f"{base_outdir}/")
    print(f"  ├── cube_0_0_0/")
    print(f"  │   ├── cube_info.txt")
    print(f"  │   ├── H_scaled_power_1.{file_format}")
    print(f"  │   ├── H_scaled_power_2.{file_format}")
    print(f"  │   └── ...")
    print(f"  ├── cube_0_0_1/")
    print(f"  │   └── ...")
    print(f"  └── ...")


def process_specific_cubes(manager, cube_indices, N, a, b, 
                          base_outdir="H_powers_basis_partitioned",
                          file_format='sym.gz'):
    """
    只处理指定的立方体
    
    Parameters:
    -----------
    manager : CubicExpressionManager
        立方体表达式管理器
    cube_indices : list of tuples
        要处理的立方体索引列表，例如 [(0,0,0), (0,0,1), (1,1,1)]
    N : int
        最大幂次
    a, b : float
        缩放参数
    base_outdir : str
        基础输出目录
    file_format : str
        文件格式
    """
    base_outdir = Path(base_outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    
    # 符号定义
    x, y, z = sp.symbols('x y z')
    kx, ky, kz = sp.symbols('kx ky kz')
    b_sym = sp.symbols('b')
    
    kvec = (kx, ky, kz)
    k2 = kx**2 + ky**2 + kz**2
    pref = 0.5
    
    print(f"\n{'='*70}")
    print(f"PROCESSING SPECIFIC CUBES")
    print(f"{'='*70}")
    print(f"Cubes to process: {len(cube_indices)}")
    print(f"Indices: {cube_indices}")
    print(f"Maximum power N: {N}")
    print(f"\n")
    
    for idx in cube_indices:
        i, j, k = idx
        
        if idx not in manager:
            print(f"Cube ({i}, {j}, {k}): ✗ Not found in manager, skipping\n")
            continue
        
        print(f"Processing Cube ({i}, {j}, {k})...")
        
        # 获取势能表达式
        V_expr = manager.get_expression(i, j, k)
        cube_info = manager.cube_info[idx]
        
        print(f"  Center: {cube_info['center']}")
        print(f"  Atoms: {cube_info['n_atoms']}")
        
        # 创建子文件夹
        #cube_dir = base_outdir / f"cube_{i}_{j}_{k}"
        #cube_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子文件夹：使用中心坐标命名
        cx, cy, cz = cube_info['center']
        cx_str = f"{cx:.3f}".replace("-", "neg")
        cy_str = f"{cy:.3f}".replace("-", "neg")
        cz_str = f"{cz:.3f}".replace("-", "neg")
        cube_dir = base_outdir / f"cube_{cx_str}_{cy_str}_{cz_str}"
        cube_dir.mkdir(parents=True, exist_ok=True)

        # 保存立方体信息
        info_file = cube_dir / "cube_info.txt"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"Cube Index: ({i}, {j}, {k})\n")
            f.write(f"Center: {cube_info['center']}\n")
            f.write(f"Number of atoms: {cube_info['n_atoms']}\n")
            f.write(f"Scaling parameters: a={a}, b={b}\n")
            f.write(f"Generated H^n powers: 0 to {N}\n")
        
        # 生成H^n表达式
        try:
            results = generate_scaled_H_powers(
                N, a, b,
                outdir=cube_dir,
                file_format=file_format,
                V=V_expr,
                kvec=kvec,
                k2=k2,
                pref=pref,
                x=x, y=y, z=z
            )
            print(f"  ✓ Complete\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")


# ---- 主程序 ----
if __name__ == "__main__":
    print("="*70)
    print("PARTITIONED H^N EXPRESSION GENERATOR (FIXED)")
    print("="*70)
    
    # 加载立方体表达式管理器
    print("\nLoading cubic expressions...")
    #manager = CubicExpressionManager('C:/FD/cubic_space_partition_rcut=5.0_whole')
    manager = CubicExpressionManager('C:/FD//test_cubic_space_partition_rcut=5.0_whole')

    print(f"✓ Loaded {len(manager)} cubes")
    
    # 显示索引范围
    print(f"\nIndex ranges:")
    print(f"  i: {manager.i_range}")
    print(f"  j: {manager.j_range}")
    print(f"  k: {manager.k_range}")
    
    # 缩放参数
    E_lower = -1
    E_upper = 1
    a = 2 / (E_upper - E_lower)
    b = -(E_upper + E_lower) / (E_upper - E_lower)
    
    # 最大阶数
    N = 6
    
    # 选择处理模式
    print(f"\n{'='*70}")
    print("PROCESSING MODE")
    print(f"{'='*70}")
    print("1. Process ALL cubes")
    print("2. Process SPECIFIC cubes only")
    print("3. Process TEST cube (0,0,1) only")

    process_all_cubes(
        manager, N, a, b,
        base_outdir="test_H_powers_basis_partitioned_rcut=5.0_whole",
        file_format='sym.gz'
    )
   
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)