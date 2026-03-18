"""
target_demo.py
使用ho3d_solvers_v2求解最接近-0.25的特征值
支持多网格方法：使用粗网格解作为细网格初始猜测
注意：使用cube文件的真实空间范围，不使用L参数
"""

import numpy as np
import ho3d_solvers_v2 as solver
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid
import matplotlib.pyplot as plt
import json
import time
import os
from datetime import datetime

def convert_to_json_serializable(obj):
    """将numpy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(convert_to_json_serializable(list(obj)))
    else:
        return obj

def main():
    print("="*70)
    print("Target Eigenvalue Solver Demo with Multigrid Support")
    print("Seeking eigenvalue near E = -0.25 a.u.")
    print("="*70)
    
    # ==========================================
    # 初始化
    # ==========================================
    print("\nInitializing Gaussian Potential Builder...")
    
    builder = GaussianPotentialBuilder(
        cube_file='localPot.cube',
        params_file='gaussian_fit_params.json',
        r_cut=7.0
    )
    
    print(f"\nCube file information:")
    print(f"  Original grid: {builder.Nx}³")
    print(f"  Spatial range: [{builder.x_min:.3f}, {builder.x_max:.3f}] Bohr")
    print(f"  Box size: {builder.x_max - builder.x_min:.3f} Bohr")
    
    # ==========================================
    # 参数设置
    # ==========================================
    
    target_energy = -0.25
    n_levels = 1  # 求1个最接近的特征值
    
    # ⭐ 多网格设置
    use_multigrid = True  # 是否使用多网格方法
    
    if use_multigrid:
        # 使用更细的网格范围
        N_list = np.arange(18, 32, 3)
    else:
        # 原始设置
        N_list = np.arange(18, 32, 3)
    
    N_max = N_list[-1]
    
    methods = [
        "sinc_dvr:PRIMME_JDQMR",   # 使用JDQMR
    ]
    
    box_size = 40.0  # 用于构建势能
    L_unused = 5.0   # ⚠️ 此参数对custom potential无效
    tol = 1e-4
    maxiter = 200000
    
    print(f"\n{'='*70}")
    print("Configuration")
    print(f"{'='*70}")
    print(f"\nTarget energy: {target_energy} a.u.")
    print(f"Methods: {methods}")
    print(f"Grid resolutions: {list(N_list)}")
    print(f"Number of eigenvalues: {n_levels}")
    print(f"Tolerance: {tol}")
    print(f"Max iterations: {maxiter}")
    print(f"Use multigrid: {use_multigrid}")
    print(f"\n⚠️  Using CUBE file spatial range!")
    
    # ==========================================
    # 运行计算
    # ==========================================
    print(f"\n{'='*70}")
    print("Running Calculations")
    print(f"{'='*70}")
    
    results = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'target_energy': target_energy,
            'n_levels': n_levels,
            'tolerance': tol,
            'maxiter': maxiter,
            'use_multigrid': use_multigrid,
            'cube_spatial_range': [float(builder.x_min), float(builder.x_max)],
        },
        'methods': {},
    }
    
    for method in methods:
        method_name = method.split(":")[1]
        print(f"\n{'='*70}")
        print(f"Method: {method_name}")
        if use_multigrid:
            print("Mode: Multigrid (coarse-to-fine)")
        else:
            print("Mode: Standard")
        print(f"{'='*70}")
        
        results['methods'][method_name] = []
        
        # 多网格变量
        prev_evecs = None
        prev_N = None
        prev_potential_grid = None
        
        # 追踪最近的成功解（用于失败时回退）
        last_successful_evecs = None
        last_successful_N = None
        last_successful_potential_grid = None
        
        for N in N_list:
            print(f"\nN = {N}³ ({N**3:,} points)")
            
            try:
                # 构建势能
                print(f"  Building potential...", end=" ", flush=True)
                t_pot_start = time.time()
                x, y, z, V = builder.build_potential(N, box_size)
                t_pot = time.time() - t_pot_start
                
                grid_spacing = x[1] - x[0]
                print(f"Done ({t_pot:.3f}s)")
                print(f"    Grid spacing: {grid_spacing:.4f} Bohr")
                print(f"    Spatial range: [{x[0]:.3f}, {x[-1]:.3f}] Bohr")
                
                potential_grid = PotentialGrid(x, y, z, V, 
                    source=f"Gaussian (r_cut={builder.r_cut})")
                
                # ⭐ 多网格：准备初始猜测
                v0 = None
                if use_multigrid and prev_evecs is not None and prev_N is not None:
                    print(f"  Interpolating from N={prev_N} to N={N}...", end=" ", flush=True)
                    try:
                        v0 = solver.interpolate_eigenvector(
                            prev_evecs, 
                            N_coarse=prev_N, 
                            N_fine=N, 
                            potential_grid=prev_potential_grid,
                            L=L_unused  # 对custom potential无效
                        )
                        print(f"Done (v0 shape: {v0.shape})")
                    except Exception as e:
                        print(f"Failed: {e}")
                        v0 = None
                
                # 求解（使用target模式，可能带有v0）
                print(f"  Solving...", end=" ", flush=True)
                res = solver.solve_ho3d(
                    method,
                    N=N,
                    L=L_unused,  # 对custom potential无效
                    n_levels=n_levels,
                    potential_grid=potential_grid,
                    target=target_energy,
                    v0=v0,
                    tol=tol,
                    maxiter=maxiter,
                    maxBlockSize=2
                )
                print(f"Done ({res.meta['t_eig']:.3f}s)")
                
                # 验证空间范围
                actual_range = res.meta['spatial_range']
                print(f"    Spatial range used: [{actual_range[0]:.3f}, {actual_range[1]:.3f}]")
                
                # 找到最接近目标的特征值
                closest_idx = np.argmin(np.abs(res.evals - target_energy))
                closest_energy = res.evals[closest_idx]
                
                # 保存结果
                result_entry = {
                    'N': int(N),
                    'total_points': int(N**3),
                    'grid_spacing': float(grid_spacing),
                    'spatial_range': [float(actual_range[0]), float(actual_range[1])],
                    't_build': float(res.meta['t_build']),
                    't_solve': float(res.meta['t_eig']),
                    't_total': float(t_pot + res.meta['t_build'] + res.meta['t_eig']),
                    'all_eigenvalues': [float(e) for e in res.evals],
                    'closest_energy': float(closest_energy),
                    'closest_idx': int(closest_idx),
                    'distance_to_target': float(abs(closest_energy - target_energy)),
                    'used_v0': str(res.meta['used_v0']),
                    'success': True,
                }
                
                results['methods'][method_name].append(result_entry)
                
                print(f"  Closest eigenvalue: {closest_energy:.8f} a.u.")
                print(f"  Distance from target: {abs(closest_energy - target_energy):.2e}")
                print(f"  Used initial guess: {res.meta['used_v0']}")
                
                # ⭐ 更新多网格变量（保存当前解用于下一个网格）
                prev_evecs = res.evecs
                prev_N = N
                prev_potential_grid = potential_grid
                
                # 更新最近的成功解
                last_successful_evecs = res.evecs
                last_successful_N = N
                last_successful_potential_grid = potential_grid
                
            except Exception as e:
                error_msg = str(e)
                print(f"  Failed: {error_msg}")
                
                # 只在调试时打印完整traceback
                if "PRIMME error -3" not in error_msg:
                    import traceback
                    traceback.print_exc()
                
                result_entry = {
                    'N': int(N),
                    'success': False,
                    'error': error_msg,
                }
                results['methods'][method_name].append(result_entry)
                
                # ⭐ 失败时：保持使用最近的成功解，不更新prev_*
                if last_successful_N is not None:
                    print(f"  Will use N={last_successful_N} for next interpolation")
                    prev_evecs = last_successful_evecs
                    prev_N = last_successful_N
                    prev_potential_grid = last_successful_potential_grid
    
    # ==========================================
    # 找到参考解（最大N的成功解）
    # ==========================================
    print(f"\n{'='*70}")
    print(f"Error Analysis (Reference: N={N_max})")
    print(f"{'='*70}")
    
    E_ref = {}
    for method in methods:
        method_name = method.split(":")[1]
        res_list = results['methods'][method_name]
        
        # 找到最大N的成功解
        for res in reversed(res_list):
            if res['success']:
                E_ref[method_name] = res['closest_energy']
                print(f"\n{method_name} reference (N={res['N']}): {E_ref[method_name]:.8f} a.u.")
                break
    
    # 计算误差
    for method in methods:
        method_name = method.split(":")[1]
        if method_name not in E_ref:
            continue
        
        res_list = results['methods'][method_name]
        for res in res_list:
            if res['success']:
                error = abs(res['closest_energy'] - E_ref[method_name])
                res['energy_error'] = float(error)
    
    # ==========================================
    # 保存结果
    # ==========================================
    if use_multigrid:
        output_dir = "results_target_demo_multigrid"
    else:
        output_dir = "results_target_demo"
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_file = os.path.join(output_dir, 'results.json')
    
    # 确保所有数据都是JSON可序列化的
    results_serializable = convert_to_json_serializable(results)
    
    with open(json_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {json_file}")
    
    # ==========================================
    # 性能表格
    # ==========================================
    print(f"\n{'='*70}")
    print("Performance Summary")
    print(f"{'='*70}")
    
    print(f"\n{'N':>4s}  {'Points':>10s}  {'Time (s)':>12s}  {'Used v0':>10s}  {'Energy':>15s}  {'Error':>12s}")
    print("-" * 80)
    
    for method in methods:
        method_name = method.split(":")[1]
        res_list = results['methods'][method_name]
        
        for res in res_list:
            if res['success']:
                N = res['N']
                points = res['total_points']
                t = res['t_solve']
                used_v0 = res.get('used_v0', 'No')
                energy = res['closest_energy']
                error = res.get('energy_error', 0.0)
                
                print(f"{N:>4d}  {points:>10,}  {t:>12.3f}  {used_v0:>10s}  {energy:>15.8f}  {error:>12.2e}")
            else:
                N = res['N']
                error_short = res.get('error', 'Unknown')[:30]
                print(f"{N:>4d}  {'Failed':>10s}  {'---':>12s}  {'---':>10s}  {error_short:>15s}")
    
    # ==========================================
    # 生成图表
    # ==========================================
    print(f"\n{'='*70}")
    print("Generating Plots")
    print(f"{'='*70}")
    
    plot_data = {}
    for method in methods:
        method_name = method.split(":")[1]
        res_list = results['methods'][method_name]
        
        N_vals, times, energies, errors, used_v0_list = [], [], [], [], []
        
        for res in res_list:
            if res['success']:
                N_vals.append(res['N'])
                times.append(res['t_solve'])
                energies.append(res['closest_energy'])
                errors.append(res.get('energy_error', np.nan))
                used_v0_list.append(res.get('used_v0', 'No'))
        
        plot_data[method_name] = {
            'N': np.array(N_vals),
            'times': np.array(times),
            'energies': np.array(energies),
            'errors': np.array(errors),
            'used_v0': used_v0_list,
        }
    
    # 如果没有成功的数据，跳过绘图
    if not any(len(data['N']) > 0 for data in plot_data.values()):
        print("No successful results to plot!")
        return
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    
    # 图1: 时间 vs N
    plt.figure(figsize=(10, 6))
    
    for idx, method in enumerate(methods):
        method_name = method.split(":")[1]
        data = plot_data[method_name]
        
        if len(data['N']) == 0:
            continue
        
        # 分别标记使用和未使用v0的点
        if use_multigrid:
            mask_no_v0 = np.array([v == 'No' for v in data['used_v0']])
            mask_yes_v0 = ~mask_no_v0
            
            if np.sum(mask_no_v0) > 0:
                plt.semilogy(data['N'][mask_no_v0], data['times'][mask_no_v0],
                           marker=markers[idx], color=colors[idx],
                           linestyle='none', markersize=10,
                           label=f'{method_name} (no v0)', fillstyle='none')
            
            if np.sum(mask_yes_v0) > 0:
                plt.semilogy(data['N'][mask_yes_v0], data['times'][mask_yes_v0],
                           marker=markers[idx], color=colors[idx],
                           linestyle='-', markersize=10, linewidth=2,
                           label=f'{method_name} (with v0)')
        else:
            plt.semilogy(data['N'], data['times'],
                       marker=markers[idx], color=colors[idx],
                       linestyle='-', markersize=8, linewidth=2,
                       label=method_name)
    
    plt.xlabel('Grid Resolution N', fontsize=12)
    plt.ylabel('Solver Time (seconds)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'time_vs_N.png')
    plt.savefig(filename, dpi=300)
    print(f"\n✓ Saved: {filename}")
    plt.close()
    
    # 图2: 能量收敛
    plt.figure(figsize=(10, 6))
    
    for idx, method in enumerate(methods):
        method_name = method.split(":")[1]
        data = plot_data[method_name]
        
        if len(data['N']) == 0:
            continue
        
        plt.plot(data['N'], data['energies'],
                marker=markers[idx], color=colors[idx],
                linewidth=2, markersize=8, label=method_name)
    
    plt.axhline(y=target_energy, color='red', linestyle='--', 
                alpha=0.5, linewidth=1.5, label=f'Target ({target_energy})')
    plt.xlabel('Grid Resolution N', fontsize=12)
    plt.ylabel('Eigenvalue (a.u.)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'energy_convergence.png')
    plt.savefig(filename, dpi=300)
    print(f"✓ Saved: {filename}")
    plt.close()
    
    print(f"\nAll plots saved to: {output_dir}/")
    
    # ==========================================
    # 总结
    # ==========================================
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    print(f"\nTarget energy: {target_energy} a.u.")
    print(f"Use multigrid: {use_multigrid}")
    print(f"Cube spatial range: {results['metadata']['cube_spatial_range']}")
    
    for method in methods:
        method_name = method.split(":")[1]
        if method_name in E_ref:
            print(f"\n{method_name}:")
            ref_N = None
            for res in reversed(results['methods'][method_name]):
                if res['success']:
                    ref_N = res['N']
                    break
            print(f"  Best N={ref_N} energy: {E_ref[method_name]:.8f} a.u.")
            print(f"  Distance from target: {abs(E_ref[method_name] - target_energy):.2e}")
    
    # 统计成功/失败
    for method in methods:
        method_name = method.split(":")[1]
        res_list = results['methods'][method_name]
        n_success = sum(1 for r in res_list if r['success'])
        n_total = len(res_list)
        print(f"\n{method_name}: {n_success}/{n_total} successful")
    
    print(f"\n{'='*70}")
    print("Demo completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
