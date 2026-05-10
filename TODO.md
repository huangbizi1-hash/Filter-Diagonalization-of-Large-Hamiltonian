# TODO：将 rbf_core.py 模块化为 rbf_code/ 包

## 进度

- [x] Step 0 — 写入本 TODO.md
- [ ] Step 1 — 创建 `rbf_code/config.py` （数据类 + 常量 + 工具函数）
- [ ] Step 2 — 创建 `rbf_code/periodic.py` （周期性 / 对称性工具）
- [ ] Step 3 — 创建 `rbf_code/nodes.py` （所有节点生成函数）
- [ ] Step 4 — 创建 `rbf_code/laplacian.py` （拉普拉斯算子 + 权重矩阵 + 节点质量）
- [ ] Step 5 — 创建 `rbf_code/io_qd.py` （cube I/O + 量子点问题构建）
- [ ] Step 6 — 创建 `rbf_code/eigensolve.py` （特征値求解 + 扫描 + 迭代）
- [ ] Step 7 — 创建 `rbf_code/__init__.py` （重导出公开 API）
- [ ] Step 8 — 将 `rbf_core.py` 改为 thin wrapper （从 rbf_code 重导出）

---

## 目标结构

```
rbf_code/
  __init__.py      — 重导出全部公开 API，保持向后兼容
  config.py        — RBFConfig, RBFProblem, IterationRecord, 常量, _raise_on_nonfinite
  periodic.py      — _wrap_frac, _periodic_diff/dist, _unique_*, _periodic_delta_frac
  nodes.py         — generate_nodes, generate_sphere_nodes, generate_atom_augmented_nodes,
                     generate_conv_cell_nodes, _make_icosphere, _filter_close_points,
                     _make_unit_cube_surface, _poisson_like_periodic, 等节点工具
  laplacian.py     — build_hamiltonian_matrix, weight_matrix_conv_cell_reuse,
                     weight_matrix_ball_fingerprint, relative_laplacian_error,
                     compute_node_quality
  io_qd.py         — read_cube_file, read_cube_atoms, build_qd_problem
  eigensolve.py    — solve_lowest_eigenvalues, sweep_rbf_kernels,
                     sweep_stencil_eps, iterate_hamiltonian, build_problem
rbf_core.py        — thin wrapper: from rbf_code import *
```

## 模块依赖关系

```
config      ← 无内部依赖
  ↑
periodic    ← config
  ↑
nodes       ← config, periodic
  ↑
laplacian   ← config
  ↑
io_qd       ← config, nodes, laplacian
  ↑
eigensolve  ← config, laplacian
```

## 公开 API 列表（故 rbf_core.py 的全部公开名称）

```python
# config
RBFConfig, RBFProblem, IterationRecord
KERNELS_ALL, KERNEL_GROUPS

# nodes
generate_nodes, make_grid_points
generate_sphere_nodes, generate_atom_augmented_nodes
generate_conv_cell_nodes
compute_node_quality

# laplacian
build_hamiltonian_matrix, relative_laplacian_error
weight_matrix_conv_cell_reuse, weight_matrix_ball_fingerprint

# io_qd
read_cube_file, read_cube_atoms, build_qd_problem

# eigensolve
solve_lowest_eigenvalues, build_problem
sweep_rbf_kernels, sweep_stencil_eps, iterate_hamiltonian
```

## 不变约束

- `rbf_core.py` 的公开名称全部保留（其他脚本不需改动）
- `run_rbf_filter.py`、`compare_fft_rbf_filter_qd.py` 等调用方无需修改

## 运行方式（不变）

```bash
# 旧方式（继续有效）
from rbf_core import build_qd_problem, RBFConfig

# 新方式（全量导入子模块）
from rbf_code.io_qd import build_qd_problem
from rbf_code.config import RBFConfig
```
