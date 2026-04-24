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
    n_steps:        int   = 10,
    omega:          float = 1.0,
    kinetic_cutoff: float = 30.0,
    output_root:    str   = ".",
) -> dict:
    """
    H^n power-iteration test on the 3D harmonic oscillator ground state.

    H = -1/2 nabla^2 + 0.5 * omega^2 * r^2,  E_0 = 3/2 * omega

    Starting from the analytic psi_0, applies H repeatedly (with renormalization
    after every step) and tracks the Ritz energy E_n = <psi_n|H|psi_n>/<psi_n|psi_n>.

    For a true eigenstate, E_n would stay constant at E_0.  In a finite periodic
    box psi_0 is not exact, so E_n drifts — the drift rate reveals the operator
    quality.  FFT (fine grid, d=0.25, with kinetic cutoff) is the near-exact
    reference; FD (sparse grid, d=0.5, 4th-order isotropic stencil) is the
    "GNN without neural network" baseline.

    Parameters
    ----------
    n_steps        : number of H applications to iterate (x-axis range 0..n_steps)
    omega          : HO frequency (Hartree atomic units)
    kinetic_cutoff : cap on T(k)=|k|^2/2 in the FFT operator (same as fft_code)
    output_root    : directory under which gnn_models/ is created
    """
    print(f"\n=== HO Power-Iteration Test  omega={omega}  n_steps={n_steps} ===")
    E0_exact = 1.5 * omega
    print(f"Exact E_0 = {E0_exact:.6f} Ha  (3/2*omega, infinite space)")
    print(f"kinetic_cutoff = {kinetic_cutoff}")

    # ── HO potential on both grids ──
    r2_fine   = X_f**2 + Y_f**2 + Z_f**2
    r2_sparse = X_s**2 + Y_s**2 + Z_s**2
    V_ho_fine   = 0.5 * omega**2 * r2_fine
    V_ho_sparse = 0.5 * omega**2 * r2_sparse

    # ── Analytic ground state (Gaussian) normalized on each grid ──
    psi0_fine   = np.exp(-omega / 2.0 * r2_fine)
    psi0_sparse = np.exp(-omega / 2.0 * r2_sparse)
    psi0_fine   /= np.sqrt(np.sum(psi0_fine**2)   * d_fine**3)
    psi0_sparse /= np.sqrt(np.sum(psi0_sparse**2) * d_sparse**3)

    # ── FFT HO Hamiltonian (fine grid, with kinetic cutoff) ──
    def fft_ho(psi: np.ndarray) -> np.ndarray:
        psi_k = np.fft.fftn(psi)
        T_k   = np.minimum(0.5 * K2_fine, kinetic_cutoff)
        T_psi = np.fft.ifftn(T_k * psi_k).real
        return T_psi + V_ho_fine * psi

    # ── FD HO Hamiltonian (sparse grid, d=d_sparse) ──
    print(f"Building FD Hamiltonian (d={d_sparse})...", flush=True)
    H_fd = _build_fd_hamiltonian(V_ho_sparse.flatten(), N_sparse, d_sparse)
    print(f"  shape={H_fd.shape}  nnz={H_fd.nnz:,}")

    # ── Power-iteration helper: compute Ritz energy series ──
    def _ritz_series_fft(psi0: np.ndarray) -> np.ndarray:
        """Apply H_FFT^n (n=0..n_steps) with renormalization; return energies."""
        psi = psi0.copy()
        energies = []
        for _ in range(n_steps + 1):
            Hpsi  = fft_ho(psi)
            norm2 = np.sum(psi**2) * d_fine**3
            E     = np.sum(psi * Hpsi) * d_fine**3 / norm2
            energies.append(E)
            nrm = np.linalg.norm(Hpsi)
            if nrm < 1e-30 or not np.isfinite(nrm):
                energies.extend([float('nan')] * (n_steps + 1 - len(energies)))
                break
            psi = Hpsi / nrm   # renormalize
        return np.array(energies)

    def _ritz_series_fd(u0: np.ndarray) -> np.ndarray:
        """Apply H_FD^n (n=0..n_steps) with renormalization; return energies."""
        u = u0.flatten().astype(np.float64)
        u /= np.linalg.norm(u)
        energies = []
        for _ in range(n_steps + 1):
            Hu    = H_fd @ u
            norm2 = np.dot(u, u) * d_sparse**3
            E     = np.dot(u, Hu) * d_sparse**3 / norm2
            energies.append(E)
            nrm = np.linalg.norm(Hu)
            if nrm < 1e-30 or not np.isfinite(nrm):
                energies.extend([float('nan')] * (n_steps + 1 - len(energies)))
                break
            u = Hu / nrm   # renormalize
        return np.array(energies)

    print("Running FFT power iteration...", flush=True)
    e_fft = _ritz_series_fft(psi0_fine)
    print("Running FD  power iteration...", flush=True)
    e_fd  = _ritz_series_fd(psi0_sparse)

    steps = np.arange(n_steps + 1)

    # ── Print table ──
    print(f"\n  {'n':>3}  {'E_FFT':>10}  {'E_FD':>10}  "
          f"{'|E_FFT-E0|':>12}  {'|E_FD-E0|':>12}")
    print("  " + "-" * 56)
    for n, ef, efd in zip(steps, e_fft, e_fd):
        sf = f"{abs(ef  - E0_exact):.3e}" if np.isfinite(ef)  else "  nan"
        sd = f"{abs(efd - E0_exact):.3e}" if np.isfinite(efd) else "  nan"
        print(f"  {n:3d}  {ef:10.6f}  {efd:10.6f}  {sf:>12}  {sd:>12}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: absolute Ritz energy vs n
    ax = axes[0]
    ax.plot(steps, e_fft, 'r-s', lw=2, ms=6,
            label=f'FFT (d={d_fine}, cutoff={kinetic_cutoff})')
    ax.plot(steps, e_fd,  'b-o', lw=2, ms=6,
            label=f'FD  (d={d_sparse}, 4th-order isotropic)')
    ax.axhline(E0_exact, color='k', ls='--', lw=1.5,
               label=f'Exact E_0 = {E0_exact:.4f} Ha')
    ax.set_xlabel("n  (H applications)")
    ax.set_ylabel("Ritz energy  <H>_n  (Ha)")
    ax.set_title(f"HO ground state power iteration  (omega={omega})\n"
                 f"Flat = eigenstate preserved; drift = PBC / discretisation error")
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)

    # Right: |E_n - E0_exact| on semilogy
    ax = axes[1]
    ax.semilogy(steps, np.abs(e_fft - E0_exact) + 1e-16,
                'r-s', lw=2, ms=6,
                label=f'FFT  |E_n - E0|')
    ax.semilogy(steps, np.abs(e_fd  - E0_exact) + 1e-16,
                'b-o', lw=2, ms=6,
                label=f'FD   |E_n - E0|')
    ax.set_xlabel("n  (H applications)")
    ax.set_ylabel("|E_n - E0_exact|  (Ha)")
    ax.set_title("Deviation from exact E_0 after n H-applications\n"
                 "(lower = operator closer to exact H)")
    ax.legend(fontsize=9); ax.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    out_dir   = os.path.join(output_root, "gnn_models")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "ho_groundstate_test.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved -> {save_path}")

    return {
        "omega": omega, "E0_exact": E0_exact,
        "n_steps": n_steps, "kinetic_cutoff": kinetic_cutoff,
        "steps":  steps.tolist(),
        "e_fft":  [x if np.isfinite(x) else None for x in e_fft.tolist()],
        "e_fd":   [x if np.isfinite(x) else None for x in e_fd.tolist()],
    }
