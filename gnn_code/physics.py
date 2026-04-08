"""
physics.py — 物理参数、网格坐标、势能场、FFT 精确哈密顿量

所有网格和势能在导入时即计算完毕（模块级常量），
供 data.py、graph.py 和 train/test 直接引用。
"""

import numpy as np

# ──────────────────────────────────────────────
# 物理参数（可在此集中修改）
# ──────────────────────────────────────────────
L        = 5.0    # 周期性盒子边长（Bohr）
d_fine   = 0.25   # FFT 精细网格步长
d_sparse = 0.5    # GNN 稀疏网格步长

N_fine   = int(L / d_fine)    # 精细网格每轴点数
N_sparse = int(L / d_sparse)  # 稀疏网格每轴点数

# 高斯陷势参数
A_pot     = 10.0  # 深度（Hartree）
sigma_pot = 1.0   # 宽度（Bohr）


# ──────────────────────────────────────────────
# 网格坐标
# ──────────────────────────────────────────────
def _make_grid(N: int):
    """生成 [-L/2, L/2) 的均匀周期性网格坐标，返回 (X, Y, Z)。"""
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    return np.meshgrid(x, x, x, indexing='ij')


X_f, Y_f, Z_f = _make_grid(N_fine)    # 精细网格
X_s, Y_s, Z_s = _make_grid(N_sparse)  # 稀疏网格


# ──────────────────────────────────────────────
# 势能场
# ──────────────────────────────────────────────
V_fine   = -A_pot * np.exp(
    -(X_f**2 + Y_f**2 + Z_f**2) / (2 * sigma_pot**2))
V_sparse = V_fine[::2, ::2, ::2]   # 直接下采样到稀疏网格


# ──────────────────────────────────────────────
# k 空间动能算符（精细网格）
# ──────────────────────────────────────────────
_kx = np.fft.fftfreq(N_fine, d=d_fine) * 2 * np.pi
_Kx, _Ky, _Kz = np.meshgrid(_kx, _kx, _kx, indexing='ij')
K2_fine = _Kx**2 + _Ky**2 + _Kz**2


# ──────────────────────────────────────────────
# FFT 精确哈密顿量
# ──────────────────────────────────────────────
def fft_hamiltonian(psi_fine: np.ndarray,
                    kinetic_cutoff: float = 30.0) -> np.ndarray:
    """H|ψ⟩ = -1/2 ∇²ψ + Vψ，通过 FFT 精确计算（fine grid）。

    kinetic_cutoff : float
        Cap on T(k) = |k|²/2 in k-space (atomic units).  Mirrors the
        build_k_diagonal cutoff in fft_code/grid.py.  Default 30.0.
        Set to np.inf to disable.
    """
    psi_k = np.fft.fftn(psi_fine)
    T_k   = np.minimum(0.5 * K2_fine, kinetic_cutoff)
    T_psi = np.fft.ifftn(T_k * psi_k).real
    return T_psi + V_fine * psi_fine


def fft_energy(psi_fine: np.ndarray,
               kinetic_cutoff: float = 30.0) -> float:
    """期望能量 ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩（fine grid）。"""
    H_psi = fft_hamiltonian(psi_fine, kinetic_cutoff=kinetic_cutoff)
    norm2  = np.sum(psi_fine**2) * d_fine**3
    energy = np.sum(psi_fine * H_psi) * d_fine**3
    return energy / norm2
