#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间分割势能符号表达式生成
  - 将 [-L_s, L_s]³ 空间分割成 4³ = 64 个小立方体
  - 为每个立方体生成独立的势能符号表达式
  - 文件名包含立方体中心坐标和原子统计信息
"""

import numpy as np
import json
import os
import pickle
from collections import Counter
import sympy as sp
from sympy import symbols, exp, sqrt, simplify, lambdify


# -------------------------
# 配置和参数
# -------------------------
class CubicSpacePartition:
    """空间分割势能分析器"""
    
    def __init__(self, params_file, atoms, L_s=8.0, n_divisions=4, r_cut=5.0):
        """
        参数:
            params_file: 拟合参数JSON文件
            atoms: 原子列表
            L_s: 总空间半径 (Bohr)，空间为 [-L_s, L_s]³
            n_divisions: 每个维度的分割数
            r_cut: 截断半径 (Bohr)
        """
        with open(params_file, 'r', encoding='utf-8') as f:
            self.params_data = json.load(f)
        self.atom_params = self.params_data['atoms']
        
        self.atoms = atoms
        self.L_s = L_s
        self.n_divisions = n_divisions
        self.r_cut = r_cut
        
        # 计算每个小立方体的边长
        self.l0 = 2 * L_s / n_divisions
        
        print(f"\n{'='*70}")
        print("CUBIC SPACE PARTITION ANALYZER SETUP")
        print(f"{'='*70}")
        print(f"✓ Loaded parameters for atom types: {list(self.atom_params.keys())}")
        print(f"✓ Total space: [{-L_s:.1f}, {L_s:.1f}]³ Bohr")
        print(f"✓ Divisions per dimension: {n_divisions}")
        print(f"✓ Number of cubes: {n_divisions**3}")
        print(f"✓ Each cube size: {self.l0:.3f} Bohr")
        print(f"✓ Cutoff radius: {self.r_cut:.2f} Bohr")
        print(f"✓ Total atoms: {len(self.atoms)}")
        
        # 为每个原子编号
        for i, atom in enumerate(self.atoms):
            atom['index'] = i
        
        # 生成所有小立方体的中心坐标
        self.cube_centers = self._generate_cube_centers()
        
    def _generate_cube_centers(self):
        """生成所有小立方体的中心坐标"""
        centers = []
        
        # 计算每个维度的中心坐标
        half_l0 = self.l0 / 2.0
        coords_1d = np.linspace(-self.L_s + half_l0, 
                                self.L_s - half_l0, 
                                self.n_divisions)
        
        # 生成所有组合
        for i, cx in enumerate(coords_1d):
            for j, cy in enumerate(coords_1d):
                for k, cz in enumerate(coords_1d):
                    centers.append({
                        'index': (i, j, k),
                        'center': np.array([cx, cy, cz]),
                        'id': len(centers)
                    })
        
        print(f"\n✓ Generated {len(centers)} cube centers")
        print(f"  Example centers:")
        for i in range(min(5, len(centers))):
            c = centers[i]['center']
            idx = centers[i]['index']
            print(f"    Cube {i:2d} at index {idx}: ({c[0]:7.3f}, {c[1]:7.3f}, {c[2]:7.3f})")
        if len(centers) > 5:
            print(f"    ...")
        
        return centers
    
    def find_relevant_atoms_for_cube(self, cube_center):
        """
        找出对指定立方体有影响的原子
        
        参数:
            cube_center: 立方体中心坐标 (numpy array)
        
        返回:
            相关原子列表
        """
        half_l0 = self.l0 / 2.0
        relevant_atoms = []
        
        for atom in self.atoms:
            atom_pos = np.array([atom['x'], atom['y'], atom['z']])
            
            # 计算原子到立方体的最短距离
            dist_to_cube = self._point_to_cube_distance(atom_pos, cube_center, half_l0)
            
            # 如果原子球与立方体相交，则该原子是relevant的
            if dist_to_cube > self.r_cut:
                continue
            
            # 检查是否在立方体的内部
            in_cube = self._point_in_cube(atom_pos, cube_center, half_l0)
            
            # 计算到立方体中心的距离
            dist_to_center = np.linalg.norm(atom_pos - cube_center)
            
            relevant_atoms.append({
                'atom': atom,
                'distance_to_center': dist_to_center,
                'distance_to_cube': dist_to_cube,
                'in_cube': in_cube,
            })
        
        # 按距离到立方体的距离排序
        relevant_atoms.sort(key=lambda x: x['distance_to_cube'])
        
        return relevant_atoms
    
    def _point_in_cube(self, point, cube_center, half_l0):
        """检查点是否在立方体内"""
        relative_pos = point - cube_center
        return (abs(relative_pos[0]) <= half_l0 and 
                abs(relative_pos[1]) <= half_l0 and 
                abs(relative_pos[2]) <= half_l0)
    
    def _point_to_cube_distance(self, point, cube_center, half_l0):
        """计算点到立方体的最短距离"""
        relative_pos = point - cube_center
        dx = max(abs(relative_pos[0]) - half_l0, 0)
        dy = max(abs(relative_pos[1]) - half_l0, 0)
        dz = max(abs(relative_pos[2]) - half_l0, 0)
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def generate_symbolic_potential(self, relevant_atoms, cube_center):
        """
        利用sympy生成势能的符号表达式
        
        参数:
            relevant_atoms: 相关原子列表
            cube_center: 立方体中心坐标
        
        返回:
            V_total_expr: 总势能符号表达式
            symbols_dict: 符号字典
            atom_info_list: 原子信息列表
        """
        # 定义符号变量
        x, y, z = symbols('x y z', real=True)
        
        # 构建势能表达式
        V_total_expr = 0
        atom_info_list = []
        
        for i, rel_atom_info in enumerate(relevant_atoms):
            atom = rel_atom_info['atom']
            atom_pos = np.array([atom['x'], atom['y'], atom['z']])
            atom_type = atom['type']
            atom_index = atom['index']
            
            # 获取原子参数
            p = self.atom_params[atom_type]
            method = p.get('method', 'gaussian')
            
            # 计算到该原子的距离符号表达式
            dx = x - atom_pos[0]
            dy = y - atom_pos[1]
            dz = z - atom_pos[2]
            r_squared = dx**2 + dy**2 + dz**2
            
            # 构建该原子的势能贡献
            if method == 'gaussian':
                amplitudes = p['amplitudes']
                exponents = p['exponents']
                
                V_atom = 0
                for A, a in zip(amplitudes, exponents):
                    V_atom += A * exp(-a * r_squared)
                
                V_total_expr += V_atom
                
                # 保存原子信息
                atom_info_list.append({
                    'index': i,
                    'global_index': atom_index,
                    'type': atom_type,
                    'position': tuple(atom_pos),
                    'amplitudes': amplitudes,
                    'exponents': exponents,
                    'distance_to_center': rel_atom_info['distance_to_center'],
                    'in_cube': rel_atom_info['in_cube'],
                    'r_cut': self.r_cut
                })
        
        return V_total_expr, {'x': x, 'y': y, 'z': z}, atom_info_list
    
    def process_all_cubes(self, output_base_dir, verbose=True):
        """
        处理所有立方体，生成势能表达式文件
        
        参数:
            output_base_dir: 输出基础目录
            verbose: 是否显示详细信息
        """
        print(f"\n{'='*70}")
        print("PROCESSING ALL CUBES")
        print(f"{'='*70}\n")
        
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 统计信息
        total_cubes = len(self.cube_centers)
        cubes_with_atoms = 0
        total_atoms_processed = 0
        
        # 处理每个立方体
        for cube_info in self.cube_centers:
            cube_id = cube_info['id']
            cube_idx = cube_info['index']
            cube_center = cube_info['center']
            
            # 找出相关原子
            relevant_atoms = self.find_relevant_atoms_for_cube(cube_center)
            
            if len(relevant_atoms) == 0:
                if verbose:
                    print(f"Cube {cube_id:2d} at {cube_idx}: No atoms - SKIPPED")
                continue
            
            cubes_with_atoms += 1
            total_atoms_processed += len(relevant_atoms)
            
            # 统计原子类型
            atom_type_count = Counter([a['atom']['type'] for a in relevant_atoms])
            in_cube_count = sum(1 for a in relevant_atoms if a['in_cube'])
            
            # 生成文件名
            cx, cy, cz = cube_center
            # 格式: cube_center_X_Y_Z_atoms_TYPE1_N1_TYPE2_N2_...
            atom_stat_str = "_".join([f"{atype}_{count}" 
                                     for atype, count in sorted(atom_type_count.items())])
            
            filename_base = (f"cube_center_{cx:.3f}_{cy:.3f}_{cz:.3f}_"
                           f"atoms_{atom_stat_str}_"
                           f"total_{len(relevant_atoms)}_inside_{in_cube_count}")
            
            # 清理文件名中的负号
            filename_base = filename_base.replace("-", "neg")
            
            if verbose:
                print(f"\nCube {cube_id:2d} at index {cube_idx}:")
                print(f"  Center: ({cx:7.3f}, {cy:7.3f}, {cz:7.3f})")
                print(f"  Relevant atoms: {len(relevant_atoms)} (inside: {in_cube_count})")
                print(f"  Atom types: {dict(atom_type_count)}")
            
            # 生成符号表达式
            V_expr, symbols_dict, atom_info_list = self.generate_symbolic_potential(
                relevant_atoms, cube_center
            )
            
            # 保存表达式
            self._save_cube_expression(
                V_expr, symbols_dict, atom_info_list,
                output_base_dir, filename_base,
                cube_center, cube_idx, relevant_atoms
            )
            
            if verbose:
                print(f"  ✓ Saved: {filename_base}")
        
        # 输出统计信息
        print(f"\n{'='*70}")
        print("PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total cubes: {total_cubes}")
        print(f"Cubes with atoms: {cubes_with_atoms}")
        print(f"Empty cubes: {total_cubes - cubes_with_atoms}")
        print(f"Total atoms processed: {total_atoms_processed}")
        print(f"Average atoms per non-empty cube: {total_atoms_processed/max(cubes_with_atoms,1):.1f}")
        
        return cubes_with_atoms
    
    def _save_cube_expression(self, V_expr, symbols_dict, atom_info_list, 
                             output_dir, filename_base, cube_center, cube_idx, relevant_atoms):
        """保存单个立方体的表达式"""
        
        # 1. 保存为pickle
        pickle_file = os.path.join(output_dir, f'{filename_base}.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'expr': V_expr,
                'symbols': symbols_dict,
                'atom_info': atom_info_list,
                'cube_center': tuple(cube_center),
                'cube_index': cube_idx,
                'cube_size': self.l0,
                'r_cut': self.r_cut,
                'n_atoms': len(relevant_atoms)
            }, f)
        
        # 2. 保存为文本格式的摘要
        txt_file = os.path.join(output_dir, f'{filename_base}_info.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Cube Information\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Cube Center: ({cube_center[0]:.6f}, {cube_center[1]:.6f}, {cube_center[2]:.6f}) Bohr\n")
            f.write(f"Cube Index: {cube_idx}\n")
            f.write(f"Cube Size: {self.l0:.6f} Bohr\n")
            f.write(f"Cutoff Radius: {self.r_cut:.6f} Bohr\n\n")
            
            f.write(f"Atom Statistics\n")
            f.write(f"{'-'*70}\n")
            f.write(f"Total relevant atoms: {len(relevant_atoms)}\n")
            
            atom_type_count = Counter([a['atom']['type'] for a in relevant_atoms])
            in_cube_count = sum(1 for a in relevant_atoms if a['in_cube'])
            
            f.write(f"Atoms inside cube: {in_cube_count}\n")
            f.write(f"Atoms outside cube (but within r_cut): {len(relevant_atoms) - in_cube_count}\n\n")
            
            f.write(f"Atom type distribution:\n")
            for atype in sorted(atom_type_count.keys()):
                count = atom_type_count[atype]
                f.write(f"  {atype}: {count} atoms\n")
            
            f.write(f"\nDetailed Atom List\n")
            f.write(f"{'-'*70}\n")
            for i, info in enumerate(atom_info_list):
                pos = info['position']
                f.write(f"{i+1:3d}. {info['type']:3s} at ({pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}), "
                       f"dist={info['distance_to_center']:.4f}, "
                       f"in_cube={'Yes' if info['in_cube'] else 'No'}\n")


# -------------------------
# 读取数据函数
# -------------------------
def read_cube_atoms(cube_file):
    """读取cube文件中的原子信息"""
    atomic_number_to_type = {
        1: 'P1',
        15: 'P2',
        33: 'As',
        49: 'In'
    }
    atoms = []
    with open(cube_file, 'r') as f:
        f.readline()
        f.readline()
        line = f.readline().split()
        n_atoms = int(line[0])
        
        for _ in range(3):
            f.readline()
        
        for i in range(n_atoms):
            parts = f.readline().split()
            atomic_number = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            
            if atomic_number in atomic_number_to_type:
                atoms.append({
                    'type': atomic_number_to_type[atomic_number],
                    'x': x, 'y': y, 'z': z,
                    'index': i
                })
    
    return atoms


def load_cube_expression(pickle_file):
    """
    从pickle文件读取立方体的势能表达式
    
    参数:
        pickle_file: pickle文件路径
    
    返回:
        包含所有信息的字典
    """
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded cube expression:")
    print(f"  Center: {data['cube_center']}")
    print(f"  Index: {data['cube_index']}")
    print(f"  Number of atoms: {data['n_atoms']}")
    print(f"  Cube size: {data['cube_size']:.3f} Bohr")
    
    return data


# -------------------------
# 主函数
# -------------------------
def main():
    """主函数"""
    print("="*70)
    print("CUBIC SPACE PARTITION - POTENTIAL EXPRESSION GENERATION")
    print("="*70)

    r_cut = 5.0  # 截断半径

    # ===== 配置参数 =====
    params_file = r'C:\FD\gaussian_fitting\fitting_params.json'
    cube_file = r'C:\FD\test_localPot.cube'
    output_dir = f'C:\FD\\test_cubic_space_partition_rcut={r_cut}_whole'
    os.makedirs(output_dir, exist_ok=True)
    
    L_s = 20.0  # 总空间半径
    n_divisions = 10  # 每个维度分割成10份
    
    # Step 1: 读取原子
    print("\n" + "-"*70)
    print("Step 1: Reading atomic configuration")
    print("-"*70)
    atoms = read_cube_atoms(cube_file)
    print(f"✓ Read {len(atoms)} atoms from {cube_file}")
    
    # 统计原子类型
    atom_types = Counter([a['type'] for a in atoms])
    print(f"\nAtom type distribution:")
    for atype in sorted(atom_types.keys()):
        print(f"  {atype}: {atom_types[atype]} atoms")
    
    # Step 2: 创建空间分割分析器
    print("\n" + "-"*70)
    print("Step 2: Creating space partition analyzer")
    print("-"*70)
    partition = CubicSpacePartition(params_file, atoms, L_s=L_s, 
                                   n_divisions=n_divisions, r_cut=r_cut)
    
    # Step 3: 处理所有立方体
    print("\n" + "-"*70)
    print("Step 3: Processing all cubes")
    print("-"*70)
    cubes_processed = partition.process_all_cubes(output_dir, verbose=True)
    
    # Step 4: 总结
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\nGenerated files in: {output_dir}/")
    print(f"Successfully processed: {cubes_processed} cubes")
    print(f"\nEach cube has two files:")
    print(f"  1. *_info.txt - Human-readable summary")
    print(f"  2. *.pkl - Binary format with full symbolic expression")
    
    print(f"\nExample: How to load a cube expression:")
    print("""
import pickle
from sympy import lambdify

# Load cube data
with open('cube_center_X_Y_Z_....pkl', 'rb') as f:
    data = pickle.load(f)

V_expr = data['expr']
symbols_dict = data['symbols']
cube_center = data['cube_center']

# Convert to numerical function
x, y, z = symbols_dict['x'], symbols_dict['y'], symbols_dict['z']
V_func = lambdify((x, y, z), V_expr, modules='numpy')

# Evaluate
V_value = V_func(0.5, 0.5, 0.5)
""")


if __name__ == "__main__":
    main()