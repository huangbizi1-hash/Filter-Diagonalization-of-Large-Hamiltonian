"""
physics.py — 物理参数、网格坐标、势能场、FFT 精确哈密顿量

所有网格和势能在导入时即计算完毕（模块级常量），
供 data.py、graph.py 和 train/test 直接引用。

势能：标准量子谐振子 V = 0.5 * omega^2 * (x^2 + y^2 + z^2)
网格：单一网格，步长 d=0.5 Bohr，无下采样。
"""

import numpy as np

# ──────────────────────────────────────────────
# 物理参数（可在此集中修改）
# ──────────────────────────────────────────────
L     = 5.0    # 周期性盒子边长（Bohr）
d     = 0.5    # 网格步长（Bohr）
N     = int(L / d)   # 每轴网格点数（= 10）

# 谐振子参数
omega = 1.0    # 谐振子频率（原子单位）


# ──────────────────────────────────────────────
# 网格坐标
# ──────────────────────────────────────────────
def _make_grid(N_: int, d_: float):
    """生成 [-L/2, L/2) 的均匀周期性网格坐标，返回 (X, Y, Z)。"""
    x = np.linspace(-L / 2, L / 2, N_, endpoint=False)
    return np.meshgrid(x, x, x, indexing='ij')


X, Y, Z = _make_grid(N, d)


# ──────────────────────────────────────────────
# 势能场（谐振子）
# ──────────────────────────────────────────────
V = 0.5 * omega**2 * (X**2 + Y**2 + Z**2)


# ──────────────────────────────────────────────
# k 空间动能算符
# ──────────────────────────────────────────────
_kx = np.fft.fftfreq(N, d=d) * 2 * np.pi
_Kx, _Ky, _Kz = np.meshgrid(_kx, _kx, _kx, indexing='ij')
K2 = _Kx**2 + _Ky**2 + _Kz**2


# ──────────────────────────────────────────────
# FFT 精确哈密顿量
# ──────────────────────────────────────────────
def fft_hamiltonian(psi: np.ndarray,
                    kinetic_cutoff: float = 30.0) -> np.ndarray:
    """H|ψ⟩ = -1/2 ∇²ψ + Vψ，通过 FFT 精确计算。

    kinetic_cutoff : float
        Cap on T(k) = |k|²/2 in k-space (atomic units).  Default 30.0.
        Set to np.inf to disable.
    """
    psi_k = np.fft.fftn(psi)
    T_k   = np.minimum(0.5 * K2, kinetic_cutoff)
    T_psi = np.fft.ifftn(T_k * psi_k).real
    return T_psi + V * psi


def fft_energy(psi: np.ndarray,
               kinetic_cutoff: float = 30.0) -> float:
    """期望能量 ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩。"""
    H_psi = fft_hamiltonian(psi, kinetic_cutoff=kinetic_cutoff)
    norm2  = np.sum(psi**2) * d**3
    energy = np.sum(psi * H_psi) * d**3
    return energy / norm2


# ── 兼容旧代码的别名（逐步废弃）────────────────────────────────────────────────
# 旧代码曾使用 d_sparse / N_sparse / V_sparse / d_fine / X_f 等名称。
# 保留别名以减少其他文件改动量。
d_sparse = d
N_sparse = N
N_fine   = N
V_sparse = V
V_fine   = V
d_fine   = d
X_f = X;  Y_f = Y;  Z_f = Z
X_s = X;  Y_s = Y;  Z_s = Z
K2_fine = K2
