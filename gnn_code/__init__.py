"""
gnn_code — Hamiltonian GNN 模块包

用图神经网络学习哈密顿算符 H = -1/2 ∇² + V(r) 在稀疏网格上的作用。

子模块
------
physics  : 物理参数、网格坐标、势能场、FFT 精确哈密顿量
data     : 波函数生成（高斯波包 / 正弦波）
graph    : 稀疏网格图结构构建（PBC 3×3×3 模板）
model    : HamiltonianGNN（含 MLP 修正）、FiniteDiffHamiltonian（基准）
train    : 训练循环，保存检查点到 gnn_models/<timestamp>/
test     : 测试 H_eff^n psi 精度并画图

入口
----
从仓库根目录运行：
    python run_gnn.py --mode all --epochs 5000
"""

# 无 torch 依赖的模块：始终可导入
from .physics import (
    L, d_fine, d_sparse, N_fine, N_sparse,
    X_f, Y_f, Z_f, X_s, Y_s, Z_s,
    V_fine, V_sparse, K2_fine,
    fft_hamiltonian, fft_energy,
)
from .data import (
    gen_gaussian_wavefunction,
    gen_sine_wavefunction,
    generate_wavefunction_and_target,
)
from .dataset import generate_k_grid_dataset, WavefunctionDataset, try_load_dataset

# torch 依赖模块：仅在 torch 可用时导入
try:
    from .graph import build_graph, build_star_graph
    from .model import (HamiltonianGNN, FiniteDiffHamiltonian,
                        HamiltonianGNN_Cross, FiniteDiffHamiltonian_Cross)
    from .train import train
    from .test  import test_baseline, test_gnn_from_run, test_gnn_ho
except ImportError:
    pass   # 无 PyTorch 时跳过；test_fd 模式只需 fd_baseline.py
