"""
fd_baseline.py — 纯 numpy/scipy 有限差分基准测试

不依赖 PyTorch 或 torch_geometric，可独立运行。

将 3×3×3 各向同性差分系数组装成 scipy 稀疏矩阵，
再与势能对角矩阵叠加，得到 H_FD = T_FD + V。

测试内容
--------
对若干随机波函数重复作用 H_FD^n psi (n=0..n_steps)，
将每步的 Ritz 能量与 FFT 精确值对比，画相对误差图。
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from .physics import (
    L, d_sparse, N_sparse, V_sparse,
    fft_hamiltonian, d_fine,
)
from .physics import X_f, Y_f, Z_f, L as _L


# ──────────────────────────────────────────────
# 构建差分哈密顿量稀疏矩阵
# ──────────────────────────────────────────────

# 各向同性 4 阶拉普拉斯系数（3×3×3 模板）
_S_off = np.array([[3, -4, 3], [-4, 16, -4], [3, -4, 3]])
_S_mid = np.array([[-4, 16, -4], [16, -72, 16], [-4, 16, -4]])
_S     = np.stack([_S_off, _S_mid, _S_off], axis=0)   # (3,3,3)
_kin_pref = -1.0 / (2.0 * d_sparse**2)


def build_fd_hamiltonian_sparse() -> csr_matrix:
    """
    将 H_FD 组装为 scipy 稀疏矩阵（形状 N³ × N³）。

    H_FD[u, v] = w_uv         (u ≠ v，近邻)
    H_FD[u, u] = Σ(-w_uv) + V[u]
    """
    n_nodes = N_sparse ** 3
    H = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

    def idx(i, j, k):
        return (i % N_sparse) * N_sparse**2 + (j % N_sparse) * N_sparse + (k % N_sparse)

    V_flat = V_sparse.flatten()

    for i in range(N_sparse):
        for j in range(N_sparse):
            for k in range(N_sparse):
                u = idx(i, j, k)
                diag_sum = 0.0

                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            v   = idx(i + di, j + dj, k + dk)
                            w   = float(_S[di+1, dj+1, dk+1]) * _kin_pref
                            H[u, v] += w
                            diag_sum -= w   # 对角项累积 -w_uv

                H[u, u] = diag_sum + V_flat[u]

    return H.tocsr()


# ──────────────────────────────────────────────
# 能量序列计算
# ──────────────────────────────────────────────

def _fd_energy_series(psi_sparse: np.ndarray,
                      H_fd: csr_matrix,
                      n_steps: int) -> list:
    """重复作用 H_FD psi，返回各步 Ritz 能量。"""
    u = psi_sparse.flatten().astype(np.float64)
    energies = []
    for _ in range(n_steps + 1):
        Hu     = H_fd @ u
        norm2  = np.dot(u, u) * d_sparse**3
        energy = np.dot(u, Hu) * d_sparse**3
        energies.append(energy / norm2)
        u = Hu
    return energies


def _fft_energy_series(psi_fine: np.ndarray, n_steps: int) -> list:
    """用 FFT 精确哈密顿量重复作用，返回各步 Ritz 能量（ground truth）。"""
    psi = psi_fine.copy()
    energies = []
    for _ in range(n_steps + 1):
        Hpsi   = fft_hamiltonian(psi)
        norm2  = np.sum(psi**2) * d_fine**3
        energy = np.sum(psi * Hpsi) * d_fine**3
        energies.append(energy / norm2)
        psi = Hpsi
    return energies


# ──────────────────────────────────────────────
# 内部：生成 fine grid 波函数
# ──────────────────────────────────────────────

def _gen_fine_wavefunction(wf_type: str, k_max: int) -> np.ndarray:
    """
    直接在 fine grid 上生成波函数（返回 psi_fine，不做下采样）。
    与 data.py 的逻辑一致，但保留完整 fine grid 以供 FFT 对比。
    """
    if wf_type == 'gaussian':
        psi = np.zeros_like(X_f, dtype=np.float64)
        for _ in range(3):
            cx, cy, cz = (np.random.rand(3) - 0.5) * L * 0.5
            w = 0.5 + np.random.rand() * 0.5
            psi += np.exp(-((X_f - cx)**2 + (Y_f - cy)**2 + (Z_f - cz)**2)
                          / (2 * w**2))
    elif wf_type == 'sine':
        psi = np.zeros_like(X_f, dtype=np.float64)
        dk  = 2 * np.pi / L
        for _ in range(np.random.randint(3, 7)):
            kx = np.random.randint(-k_max, k_max + 1) * dk
            ky = np.random.randint(-k_max, k_max + 1) * dk
            kz = np.random.randint(-k_max, k_max + 1) * dk
            b  = np.random.rand() * 2 * np.pi
            psi += np.sin(kx * X_f + ky * Y_f + kz * Z_f + b)
    else:
        raise ValueError(f"Unknown wf_type: {wf_type!r}")
    return psi


# ──────────────────────────────────────────────
# 公共接口
# ──────────────────────────────────────────────

def test_fd_baseline(
    n_steps:     int = 8,
    n_test:      int = 5,
    wf_type:     str = 'gaussian',
    k_max:       int = 2,
    output_root: str = ".",
) -> tuple:
    """
    纯差分基准测试（无 PyTorch）。

    对 n_test 个随机波函数，比较 H_FD^n psi 与 H_FFT^n psi 的 Ritz 能量，
    画相对误差图并保存到 <output_root>/gnn_models/fd_baseline_numpy.png。

    返回 (fd_energies, fft_energies) 各形状 (n_test, n_steps+1)。
    """
    print("构建有限差分稀疏矩阵 H_FD ...", flush=True)
    H_fd = build_fd_hamiltonian_sparse()
    print(f"  H_FD 形状: {H_fd.shape}  非零元: {H_fd.nnz:,}")

    fft_e_list = []
    fd_e_list  = []

    for i in range(n_test):
        print(f"  测试波函数 {i+1}/{n_test} ...", flush=True)
        psi_fine   = _gen_fine_wavefunction(wf_type, k_max)
        psi_sparse = psi_fine[::2, ::2, ::2]

        fft_e_list.append(_fft_energy_series(psi_fine,   n_steps))
        fd_e_list.append( _fd_energy_series( psi_sparse, H_fd, n_steps))

    fft_e   = np.array(fft_e_list)   # (n_test, n_steps+1)
    fd_e    = np.array(fd_e_list)
    rel_err = np.abs(fd_e - fft_e) / (np.abs(fft_e) + 1e-10)
    steps   = np.arange(n_steps + 1)

    # ── 画图 ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for i in range(n_test):
        ax.plot(steps, fd_e[i],  'b-o',  alpha=0.6, ms=4, lw=1.3,
                label='FD'      if i == 0 else None)
        ax.plot(steps, fft_e[i], 'r--s', alpha=0.6, ms=4, lw=1.3,
                label='FFT ref' if i == 0 else None)
    ax.set_xlabel("n  （H 作用次数）")
    ax.set_ylabel("Ritz 能量  ⟨H⟩")
    ax.set_title(f"FD 差分基准：能量 vs H 作用次数\n"
                 f"(d_sparse={d_sparse}, N_sparse={N_sparse})")
    ax.legend(); ax.grid(True, ls='--', alpha=0.5)

    ax = axes[1]
    mean_err = rel_err.mean(axis=0)
    std_err  = rel_err.std(axis=0)
    ax.semilogy(steps, mean_err, 'b-o', lw=1.8, label='均值相对误差')
    ax.fill_between(steps,
                    np.maximum(mean_err - std_err, 1e-12),
                    mean_err + std_err,
                    alpha=0.25, label='±1σ')
    # 标出初始误差和最终误差
    ax.annotate(f"{mean_err[0]:.2e}", xy=(0, mean_err[0]),
                xytext=(0.5, mean_err[0]*2), fontsize=9, color='navy')
    ax.annotate(f"{mean_err[-1]:.2e}", xy=(n_steps, mean_err[-1]),
                xytext=(n_steps-1.5, mean_err[-1]*2), fontsize=9, color='navy')
    ax.set_xlabel("n  （H 作用次数）")
    ax.set_ylabel("相对能量误差  |E_FD - E_FFT| / |E_FFT|")
    ax.set_title("FD 差分基准：相对误差 vs n")
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    out_dir = os.path.join(output_root, "gnn_models")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "fd_baseline_numpy.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n图像已保存 → {save_path}")

    # ── 打印汇总 ──
    print(f"\n{'n':>4}  {'mean rel err':>14}  {'std':>10}")
    print("-" * 32)
    for n, me, se in zip(steps, mean_err, std_err):
        print(f"{n:>4}  {me:>14.4e}  {se:>10.4e}")

    return fd_e, fft_e
