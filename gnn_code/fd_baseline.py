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
    fft_hamiltonian, d_fine, K2_fine,
    X_f, Y_f, Z_f,
    X_s, Y_s, Z_s,
)


# ──────────────────────────────────────────────
# 构建差分哈密顿量稀疏矩阵
# ──────────────────────────────────────────────

# 各向同性 4 阶拉普拉斯系数（3×3×3 模板）
_S_off = np.array([[3, -4, 3], [-4, 16, -4], [3, -4, 3]])
_S_mid = np.array([[-4, 16, -4], [16, -72, 16], [-4, 16, -4]])
_S     = np.stack([_S_off, _S_mid, _S_off], axis=0)   # (3,3,3)
# The isotropic stencil satisfies: sum_neighbors S[di,dj,dk]*psi[v] ≈ 12*d²*∇²psi
# So the kinetic energy T = -1/2*∇² requires prefactor -1/(2*12*d²) = -1/(24*d²)
_kin_pref = -1.0 / (24.0 * d_sparse**2)


def _build_fd_hamiltonian(V_flat: np.ndarray, N: int, d: float) -> csr_matrix:
    """
    Assemble the FD Hamiltonian as a scipy sparse matrix (N^3 x N^3).
    Uses the 4th-order isotropic 3x3x3 stencil with PBC.

    Parameters
    ----------
    V_flat : 1-D array of length N^3, potential values on the grid
    N      : number of grid points per axis
    d      : grid spacing
    """
    kin_pref = -1.0 / (24.0 * d**2)  # stencil encodes 12*d²*∇²; T=-½∇² → -1/(24d²)
    n_nodes  = N ** 3
    H = lil_matrix((n_nodes, n_nodes), dtype=np.float64)

    def idx(i, j, k):
        return (i % N) * N**2 + (j % N) * N + (k % N)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                u = idx(i, j, k)
                diag_sum = 0.0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            v = idx(i + di, j + dj, k + dk)
                            w = float(_S[di+1, dj+1, dk+1]) * kin_pref
                            H[u, v] += w
                            diag_sum -= w
                H[u, u] = diag_sum + V_flat[u]

    return H.tocsr()


def build_fd_hamiltonian_sparse() -> csr_matrix:
    """Build FD Hamiltonian on the default sparse grid with V_sparse."""
    return _build_fd_hamiltonian(V_sparse.flatten(), N_sparse, d_sparse)


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
    print("Building sparse FD Hamiltonian H_FD ...", flush=True)
    H_fd = build_fd_hamiltonian_sparse()
    print(f"  H_FD shape: {H_fd.shape}  nnz: {H_fd.nnz:,}")

    fft_e_list = []
    fd_e_list  = []

    for i in range(n_test):
        print(f"  Test wavefunction {i+1}/{n_test} ...", flush=True)
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
    ax.set_xlabel("n  (H applications)")
    ax.set_ylabel("Ritz energy  <H>")
    ax.set_title(f"FD Baseline: Energy vs H applications\n"
                 f"(d_sparse={d_sparse}, N_sparse={N_sparse})")
    ax.legend(); ax.grid(True, ls='--', alpha=0.5)

    ax = axes[1]
    mean_err = rel_err.mean(axis=0)
    std_err  = rel_err.std(axis=0)
    ax.semilogy(steps, mean_err, 'b-o', lw=1.8, label='mean relative error')
    ax.fill_between(steps,
                    np.maximum(mean_err - std_err, 1e-12),
                    mean_err + std_err,
                    alpha=0.25, label='+/-1sigma')
    ax.annotate(f"{mean_err[0]:.2e}", xy=(0, mean_err[0]),
                xytext=(0.5, mean_err[0]*2), fontsize=9, color='navy')
    ax.annotate(f"{mean_err[-1]:.2e}", xy=(n_steps, mean_err[-1]),
                xytext=(n_steps-1.5, mean_err[-1]*2), fontsize=9, color='navy')
    ax.set_xlabel("n  (H applications)")
    ax.set_ylabel("Relative energy error  |E_FD - E_FFT| / |E_FFT|")
    ax.set_title("FD Baseline: Relative error vs n")
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    out_dir = os.path.join(output_root, "gnn_models")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "fd_baseline_numpy.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved -> {save_path}")

    # ── 打印汇总 ──
    print(f"\n{'n':>4}  {'mean rel err':>14}  {'std':>10}")
    print("-" * 32)
    for n, me, se in zip(steps, mean_err, std_err):
        print(f"{n:>4}  {me:>14.4e}  {se:>10.4e}")

    return fd_e, fft_e


# ──────────────────────────────────────────────
# HO ground state correctness test
# ──────────────────────────────────────────────

def test_ho_groundstate(
    omega:       float = 1.0,
    output_root: str   = ".",
) -> dict:
    """
    Correctness test for the FD and FFT Hamiltonians on the 3D harmonic oscillator.

    H = -1/2 nabla^2 + 0.5 * omega^2 * r^2
    Exact infinite-space ground state energy: E_0 = 3/2 * omega

    NOTE: The analytic psi_0 = exp(-omega/2 r^2) is NOT a perfect eigenstate of
    H_PBC (periodic-box Hamiltonian), so repeated power iteration (H^n psi_0)
    diverges toward the highest eigenvalue — that is expected, not a bug.

    This test instead measures correctness by:
      1. Variational energy: E_ritz = <psi_0|H|psi_0>/<psi_0|psi_0> — should be
         close to E_0_exact (variational upper bound).
      2. Residual: ||H psi_0 - E_ritz psi_0|| / ||psi_0|| — how close psi_0 is
         to an eigenstate of H (depends on box size / PBC contamination).
      3. Numerical eigenvalue: the lowest eigenvalue of H_FD_HO obtained via
         scipy.sparse.linalg.eigsh — should converge to E_0 as d -> 0.
      4. Grid-convergence: compare E_ritz on sparse grid (d=0.5) vs fine grid (d=0.25).
    """
    from scipy.sparse.linalg import eigsh

    print(f"\n=== HO Ground State Correctness Test (omega={omega}) ===")
    E0_exact = 1.5 * omega
    print(f"Exact E_0 = {E0_exact:.6f} Hartree  (3/2 * omega, infinite-space)")

    # ── HO potential on both grids ──
    r2_fine   = X_f**2 + Y_f**2 + Z_f**2
    r2_sparse = X_s**2 + Y_s**2 + Z_s**2
    V_ho_fine   = 0.5 * omega**2 * r2_fine
    V_ho_sparse = 0.5 * omega**2 * r2_sparse

    # ── Analytic ground state (Gaussian, normalized) ──
    psi0_fine   = np.exp(-omega / 2.0 * r2_fine)
    psi0_sparse = np.exp(-omega / 2.0 * r2_sparse)
    psi0_fine   /= np.sqrt(np.sum(psi0_fine**2)   * d_fine**3)
    psi0_sparse /= np.sqrt(np.sum(psi0_sparse**2) * d_sparse**3)

    # ── FFT Hamiltonian with HO potential ──
    def fft_ho(psi: np.ndarray) -> np.ndarray:
        psi_k = np.fft.fftn(psi)
        T_psi = np.fft.ifftn(0.5 * K2_fine * psi_k).real
        return T_psi + V_ho_fine * psi

    # ── Variational energy and residual (FFT, fine grid) ──
    Hpsi0_fft = fft_ho(psi0_fine)
    norm2_f    = np.sum(psi0_fine**2) * d_fine**3
    E_ritz_fft = np.sum(psi0_fine * Hpsi0_fft) * d_fine**3 / norm2_f
    res_fft    = np.sqrt(np.sum((Hpsi0_fft - E_ritz_fft * psi0_fine)**2)
                         * d_fine**3 / norm2_f)

    # ── Build FD HO Hamiltonians (sparse and fine grids) ──
    print("Building FD Hamiltonians...", flush=True)
    N_fine_loc = int(L / d_fine)
    # We also build a FD HO on the FINE grid to compare discretization error fairly
    def _build_fd_fine_ho():
        kin_pref = -1.0 / (24.0 * d_fine**2)  # stencil encodes 12*d²*∇²
        S_off = np.array([[3, -4, 3], [-4, 16, -4], [3, -4, 3]])
        S_mid = np.array([[-4, 16, -4], [16, -72, 16], [-4, 16, -4]])
        S_loc = np.stack([S_off, S_mid, S_off], axis=0)
        V_flat = V_ho_fine.flatten()
        N = N_fine_loc
        H = lil_matrix((N**3, N**3), dtype=np.float64)
        def idx(i, j, k): return (i%N)*N**2 + (j%N)*N + (k%N)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    u = idx(i, j, k); ds = 0.0
                    for di in (-1,0,1):
                        for dj in (-1,0,1):
                            for dk in (-1,0,1):
                                if di==0 and dj==0 and dk==0: continue
                                v = idx(i+di, j+dj, k+dk)
                                w = float(S_loc[di+1,dj+1,dk+1]) * kin_pref
                                H[u, v] += w; ds -= w
                    H[u, u] = ds + V_flat[u]
        return H.tocsr()

    H_fd_sparse = _build_fd_hamiltonian(V_ho_sparse.flatten(), N_sparse, d_sparse)
    H_fd_fine   = _build_fd_fine_ho()
    print(f"  Sparse FD: {H_fd_sparse.shape}  nnz={H_fd_sparse.nnz:,}  (d={d_sparse})")
    print(f"  Fine   FD: {H_fd_fine.shape}  nnz={H_fd_fine.nnz:,}  (d={d_fine})")

    # ── Variational energy and residual (FD, sparse) ──
    u0_s    = psi0_sparse.flatten()
    Hu0_s   = H_fd_sparse @ u0_s
    norm2_s = np.dot(u0_s, u0_s) * d_sparse**3
    E_ritz_fd_s = np.dot(u0_s, Hu0_s) * d_sparse**3 / norm2_s
    res_fd_s    = np.sqrt(np.sum((Hu0_s - E_ritz_fd_s * u0_s)**2)
                          * d_sparse**3 / norm2_s)

    # ── Variational energy and residual (FD, fine) ──
    u0_f    = psi0_fine.flatten()
    Hu0_f   = H_fd_fine @ u0_f
    norm2_ffd = np.dot(u0_f, u0_f) * d_fine**3
    E_ritz_fd_f = np.dot(u0_f, Hu0_f) * d_fine**3 / norm2_ffd
    res_fd_f    = np.sqrt(np.sum((Hu0_f - E_ritz_fd_f * u0_f)**2)
                          * d_fine**3 / norm2_ffd)

    # ── Lowest numerical eigenvalue via scipy eigsh ──
    print("Computing lowest eigenvalue via scipy eigsh...", flush=True)
    v0_s = psi0_sparse.flatten().astype(np.float64)
    v0_f = psi0_fine.flatten().astype(np.float64)
    E_eig_s = eigsh(H_fd_sparse, k=1, which='SM', v0=v0_s, tol=1e-10,
                    maxiter=5000, return_eigenvectors=False)[0]
    E_eig_f = eigsh(H_fd_fine,   k=1, which='SM', v0=v0_f, tol=1e-10,
                    maxiter=5000, return_eigenvectors=False)[0]

    # ── Summary table ──
    print(f"\n  Method               E_ritz       Residual   |E-E0_exact|")
    print(f"  {'─'*58}")
    print(f"  FFT   (d={d_fine})       {E_ritz_fft:10.6f}   {res_fft:9.2e}   "
          f"{abs(E_ritz_fft-E0_exact):.2e}")
    print(f"  FD    (d={d_fine})       {E_ritz_fd_f:10.6f}   {res_fd_f:9.2e}   "
          f"{abs(E_ritz_fd_f-E0_exact):.2e}")
    print(f"  FD    (d={d_sparse})       {E_ritz_fd_s:10.6f}   {res_fd_s:9.2e}   "
          f"{abs(E_ritz_fd_s-E0_exact):.2e}")
    print(f"  ─── numerical eigenvalues (eigsh) ───")
    print(f"  FD eig (d={d_fine})     {E_eig_f:10.6f}               "
          f"{abs(E_eig_f-E0_exact):.2e}")
    print(f"  FD eig (d={d_sparse})     {E_eig_s:10.6f}               "
          f"{abs(E_eig_s-E0_exact):.2e}")
    print(f"  Exact (infinite box)  {E0_exact:10.6f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = [f'FFT  d={d_fine}', f'FD   d={d_fine}', f'FD   d={d_sparse}']
    e_ritz = [E_ritz_fft, E_ritz_fd_f, E_ritz_fd_s]
    e_eig  = [None,        E_eig_f,     E_eig_s]
    resids = [res_fft,     res_fd_f,    res_fd_s]
    colors = ['tomato', 'steelblue', 'darkorange']

    ax = axes[0]
    x = np.arange(len(labels))
    bars = ax.bar(x, e_ritz, color=colors, alpha=0.8, label='Ritz energy (analytic psi_0)')
    # Overlay eigsh eigenvalue as marker
    for xi, ee in zip(x[1:], e_eig[1:]):
        ax.plot(xi, ee, 'k^', ms=9, zorder=5,
                label='eigsh eigenvalue' if xi == x[1] else None)
    ax.axhline(E0_exact, color='k', ls='--', lw=1.5,
               label=f'Exact E_0={E0_exact:.4f}')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(f"HO ground state energy (omega={omega})\n"
                 f"Ritz: variational energy with analytic psi_0")
    ax.legend(fontsize=9); ax.grid(True, axis='y', ls='--', alpha=0.5)

    ax = axes[1]
    ax.bar(x, resids, color=colors, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("||H psi_0 - E_ritz psi_0|| / ||psi_0||")
    ax.set_title("Eigenvalue residual of analytic psi_0\n"
                 "(small = psi_0 is close to eigenstate of H_PBC)")
    ax.set_yscale('log'); ax.grid(True, axis='y', which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    out_dir   = os.path.join(output_root, "gnn_models")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "ho_groundstate_test.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved -> {save_path}")

    return {
        "omega": omega, "E0_exact": E0_exact,
        "E_ritz_fft": E_ritz_fft,  "res_fft": res_fft,
        "E_ritz_fd_fine": E_ritz_fd_f,   "res_fd_fine": res_fd_f,
        "E_ritz_fd_sparse": E_ritz_fd_s, "res_fd_sparse": res_fd_s,
        "E_eig_fd_fine": E_eig_f,
        "E_eig_fd_sparse": E_eig_s,
    }
