"""
gaussian_potential_builder.py
使用高斯函数重构势能，支持不同网格分辨率
基于r_cut=7的截断半径
"""

import numpy as np
import json
from typing import List, Dict, Tuple

class GaussianPotentialBuilder:
    """高斯势能构建器"""
    
    def __init__(self, cube_file: str, params_file: str, r_cut: float = 7.0):
        """
        初始化
        
        Parameters:
        -----------
        cube_file : str
            cube文件路径（原子位置）
        params_file : str
            高斯参数JSON文件路径
        r_cut : float
            截断半径（单位：Å）
        """
        self.r_cut = r_cut
        
        # 读取高斯参数
        with open(params_file, 'r') as f:
            self.params = json.load(f)
        
        # 读取原子位置
        self.atoms, self.origin, self.grid_info = self._read_cube_file(cube_file)
        
        # 从grid_info提取网格信息
        self.Nx = self.grid_info[0][0]
        self.Ny = self.grid_info[1][0]
        self.Nz = self.grid_info[2][0]
        
        # 计算空间范围（从origin和grid vectors）
        dx_vec = self.grid_info[0][1]
        dy_vec = self.grid_info[1][1]
        dz_vec = self.grid_info[2][1]
        
        # 构建原始网格坐标
        self.x = self.origin[0] + np.arange(self.Nx) * dx_vec[0]
        self.y = self.origin[1] + np.arange(self.Ny) * dy_vec[1]
        self.z = self.origin[2] + np.arange(self.Nz) * dz_vec[2]
        
        self.x_min = self.x[0]
        self.x_max = self.x[-1]
        self.y_min = self.y[0]
        self.y_max = self.y[-1]
        self.z_min = self.z[0]
        self.z_max = self.z[-1]
        
        # 确定原子类型
        self.atom_types = self._determine_atom_types()
        
        print(f"Gaussian Potential Builder initialized:")
        print(f"  Cutoff radius: {self.r_cut} Å")
        print(f"  Number of atoms: {len(self.atoms)}")
        print(f"  Atom types: {set(self.atom_types)}")
    
    def _read_cube_file(self, filename: str) -> Tuple[List[Dict], np.ndarray, List]:
        """读取cube文件"""
        atoms = []
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        idx = 2
        
        # 原子数和原点
        parts = lines[idx].split()
        n_atoms = int(parts[0])
        origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        idx += 1
        
        # 网格向量
        grid_info = []
        for i in range(3):
            parts = lines[idx].split()
            n_points = int(parts[0])
            vector = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            grid_info.append((n_points, vector))
            idx += 1
        
        # 原子位置
        for i in range(n_atoms):
            parts = lines[idx].split()
            atomic_number = int(parts[0])
            charge = float(parts[1])
            position = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            atoms.append({
                'atomic_number': atomic_number,
                'charge': charge,
                'position': position
            })
            idx += 1
        
        return atoms, origin, grid_info
    
    def _determine_atom_types(self) -> List[str]:
        """确定原子类型"""
        atom_types = []
        for atom in self.atoms:
            if atom['atomic_number'] == 33:  # As
                atom_types.append('As')
            elif atom['atomic_number'] == 49:  # In
                atom_types.append('In')
            elif atom['atomic_number'] == 1:  # P
                if atom['charge'] > 0:
                    atom_types.append('P1')
                else:
                    atom_types.append('P2')
            else:
                atom_types.append('Unknown')
        return atom_types
    
    def build_potential(self, N: int, box_size: float = 40.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建势能网格
        
        Parameters:
        -----------
        N : int
            每个维度的网格点数
        box_size : float
            总空间大小（默认40Å，对应-20到+20）
            
        Returns:
        --------
        x, y, z : 1D arrays
            网格坐标
        V : 3D array
            势能
        """
        # 创建网格 - 使用cube文件的实际空间范围
        x = np.linspace(self.x_min, self.x_max, N)
        y = np.linspace(self.y_min, self.y_max, N)
        z = np.linspace(self.z_min, self.z_max, N)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 初始化势能
        V = np.zeros_like(X)
        
        # 累加每个原子的贡献
        for atom, atom_type in zip(self.atoms, self.atom_types):
            if atom_type == 'Unknown':
                continue
            
            # 获取高斯参数
            atom_params = self.params['atoms'][atom_type]
            amplitudes = atom_params['amplitudes']
            exponents = atom_params['exponents']
            
            # 原子位置
            pos = atom['position']
            
            # 计算距离
            r = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
            
            # 计算该原子的势能贡献（带截断）
            V_atom = np.zeros_like(r)
            mask = r <= self.r_cut
            
            for A, alpha in zip(amplitudes, exponents):
                V_atom[mask] += A * np.exp(-alpha * r[mask]**2)
            
            V += V_atom
        
        return x, y, z, V
    
    def get_grid_spacing(self, N: int, box_size: float = 40.0) -> float:
        """获取网格步长"""
        spatial_extent = self.x_max - self.x_min
        return spatial_extent / (N - 1)
    
    def print_grid_info(self, N: int, box_size: float = 40.0):
        """打印网格信息"""
        d = self.get_grid_spacing(N, box_size)
        spatial_extent = self.x_max - self.x_min
        print(f"\nGrid N={N}:")
        print(f"  Total points: {N**3:,}")
        print(f"  Grid spacing: {d:.4f} Å")
        print(f"  Box size: [{self.x_min:.3f}, {self.x_max:.3f}] Å ({spatial_extent:.3f} total)")


class PotentialGrid:
    """势能网格数据结构（与grid_reader兼容）"""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                 potential: np.ndarray, source: str = ""):
        self.x = x
        self.y = y
        self.z = z
        self.potential = potential
        self.source = source
        
        self.Nx = len(x)
        self.Ny = len(y)
        self.Nz = len(z)
        
        self.x_range = (x[0], x[-1])
        self.y_range = (y[0], y[-1])
        self.z_range = (z[0], z[-1])
        
        self.dx = x[1] - x[0] if len(x) > 1 else 0
        self.dy = y[1] - y[0] if len(y) > 1 else 0
        self.dz = z[1] - z[0] if len(z) > 1 else 0
    
    def __repr__(self):
        return (f"PotentialGrid(Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, "
                f"x=[{self.x_range[0]:.2f}, {self.x_range[1]:.2f}], "
                f"y=[{self.y_range[0]:.2f}, {self.y_range[1]:.2f}], "
                f"z=[{self.z_range[0]:.2f}, {self.z_range[1]:.2f}], "
                f"V=[{self.potential.min():.6e}, {self.potential.max():.6e}])")
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'shape': (self.Nx, self.Ny, self.Nz),
            'grid_spacing': (self.dx, self.dy, self.dz),
            'V_min': self.potential.min(),
            'V_max': self.potential.max(),
            'V_mean': self.potential.mean(),
            'V_std': self.potential.std(),
        }


if __name__ == "__main__":
    # 测试代码
    print("="*70)
    print("Testing Gaussian Potential Builder")
    print("="*70)
    
    # 初始化
    builder = GaussianPotentialBuilder(
        cube_file='localPot.cube',
        params_file='gaussian_fit_params.json',
        r_cut=7.0
    )
    
    print(f"\nCube file original grid:")
    print(f"  Grid size: {builder.Nx}×{builder.Ny}×{builder.Nz}")
    print(f"  Spatial range: [{builder.x_min:.3f}, {builder.x_max:.3f}] Bohr")
    print(f"  Original grid spacing: {builder.x[1] - builder.x[0]:.4f} Bohr")
    
    # 测试不同分辨率
    N_list = [15, 20, 25, 30, 40, 64]
    
    print(f"\n{'='*70}")
    print("Building potentials for different grid resolutions")
    print(f"{'='*70}")
    
    for N in N_list:
        builder.print_grid_info(N)
        
        x, y, z, V = builder.build_potential(N)
        
        print(f"  Potential range: [{V.min():.6f}, {V.max():.6f}] a.u.")
        print(f"  Memory: ~{V.nbytes / 1024**2:.2f} MB")
    
    print(f"\n{'='*70}")
    print("Test completed!")
    print(f"{'='*70}")