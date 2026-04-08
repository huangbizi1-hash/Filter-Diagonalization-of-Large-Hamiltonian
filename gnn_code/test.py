"""
test.py — 测试 H_eff^n psi 精度并画图

test_baseline(...)
    测试纯差分算子的 H^n psi 能量偏差（无神经网络）。
    输出：<output_root>/gnn_models/baseline_test.png

test_gnn_from_run(run_dir, ...)
    加载 run_dir 下的多个检查点，对每个 checkpoint 测试精度，
    画多条彩色误差曲线 + FD baseline 红虚线。
    输出：run_dir/gnn_test_energy.png
"""

import os
import json

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .physics import (
    d_fine, d_sparse, V_sparse, fft_hamiltonian,
)
from .data  import gen_fine_wavefunction
from .graph import build_graph
from .model import HamiltonianGNN, FiniteDiffHamiltonian


# ──────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────

def _gnn_energy_series(psi_sparse: np.ndarray,
                       apply_H,
                       n_steps:  int,
                       device:   torch.device) -> list:
    """
    对稀疏网格波函数 psi_sparse，重复作用 apply_H 共 n_steps 次，
    记录每次作用后的 Ritz 能量 E_k = ⟨ψ_k|H|ψ_k⟩ / ⟨ψ_k|ψ_k⟩。
    返回长度 n_steps+1 的列表（k=0 为初始态）。
    """
    u = torch.tensor(
        psi_sparse.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
    energies = []
    for _ in range(n_steps + 1):
        H_u    = apply_H(u)
        norm2  = torch.sum(u**2).item() * d_sparse**3
        energy = torch.sum(u * H_u).item() * d_sparse**3
        energies.append(energy / norm2)
        u = H_u   # ψ_{k+1} = H ψ_k（不归一化）
    return energies


def _fft_energy_series(psi_fine: np.ndarray, n_steps: int) -> list:
    """
    在 fine grid 上用 FFT 哈密顿量重复作用 n_steps 次，
    返回各步的精确 Ritz 能量（ground truth）。
    """
    psi = psi_fine.copy()
    energies = []
    for _ in range(n_steps + 1):
        H_psi  = fft_hamiltonian(psi)
        norm2  = np.sum(psi**2) * d_fine**3
        energy = np.sum(psi * H_psi) * d_fine**3
        energies.append(energy / norm2)
        psi = H_psi
    return energies


def _build_shared_graph(device):
    """构建图并移动到 device，返回 (edge_index, edge_attr, V_tensor)。"""
    edge_index, edge_attr = build_graph()
    V_tensor = torch.tensor(
        V_sparse.flatten(), dtype=torch.float32).unsqueeze(-1)
    return (edge_index.to(device),
            edge_attr.to(device),
            V_tensor.to(device))


# ──────────────────────────────────────────────
# 公共接口
# ──────────────────────────────────────────────

def test_baseline(
    n_steps:     int = 8,
    n_test:      int = 5,
    wf_type:     str = 'gaussian',
    k_max:       int = 2,
    output_root: str = ".",
):
    """
    测试纯差分（FiniteDiffHamiltonian）的 H^n psi 能量偏差。

    Parameters
    ----------
    output_root : str
        图像保存到 <output_root>/gnn_models/baseline_test.png。
    """
    print("\n=== Baseline Test (Finite Difference, no GNN) ===")
    device = torch.device('cpu')
    edge_index, edge_attr, V_tensor = _build_shared_graph(device)
    fd_ham = FiniteDiffHamiltonian(edge_index, edge_attr, V_tensor, device)

    fft_energies_list = []
    fd_energies_list  = []

    for _ in range(n_test):
        psi_fine   = gen_fine_wavefunction(wf_type, k_max)  # (N_fine,)^3
        psi_sparse = psi_fine[::2, ::2, ::2]                # (N_sparse,)^3

        fft_energies_list.append(_fft_energy_series(psi_fine,   n_steps))
        fd_energies_list.append(_gnn_energy_series( psi_sparse, fd_ham, n_steps, device))

    fft_e = np.array(fft_energies_list)   # (n_test, n_steps+1)
    fd_e  = np.array(fd_energies_list)
    rel_err = np.abs(fd_e - fft_e) / (np.abs(fft_e) + 1e-10)
    steps   = np.arange(n_steps + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for i in range(n_test):
        ax.plot(steps, fd_e[i],  'b-o',  alpha=0.5, ms=4, lw=1.2,
                label='FD'      if i == 0 else None)
        ax.plot(steps, fft_e[i], 'r--s', alpha=0.5, ms=4, lw=1.2,
                label='FFT ref' if i == 0 else None)
    ax.set_xlabel("n (H applications)")
    ax.set_ylabel("Energy ⟨H⟩")
    ax.legend(); ax.grid(True, ls='--', alpha=0.5)
    ax.set_title("Baseline: Energy vs H applications")

    ax = axes[1]
    mean_err = rel_err.mean(axis=0)
    std_err  = rel_err.std(axis=0)
    ax.semilogy(steps, mean_err, 'b-o', lw=1.5, label='mean relative error')
    ax.fill_between(steps,
                    np.maximum(mean_err - std_err, 1e-12),
                    mean_err + std_err, alpha=0.2)
    ax.set_xlabel("n (H applications)")
    ax.set_ylabel("Relative Energy Error")
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.set_title("Baseline: Relative Error vs n")

    plt.tight_layout()
    out_dir = os.path.join(output_root, "gnn_models")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "baseline_test.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Baseline test plot → {save_path}")
    return fd_e, fft_e


def test_gnn_from_run(
    run_dir:     str,
    n_steps:     int = 8,
    n_test:      int = 5,
    d_test:      int = 1,
    wf_type:     str = 'gaussian',
    k_max:       int = 2,
):
    """
    加载 run_dir 中所有检查点（每 d_test 个取一个），
    对每个 checkpoint 测试 H_GNN^n psi 的能量精度。

    左图：相对误差 vs n（多条彩色线 + FD baseline 红虚线）。
    右图：n=1, n//2, n_steps 处的误差 vs 训练 epoch。

    输出：run_dir/gnn_test_energy.png
    """
    print(f"\n=== GNN Test: {run_dir} ===")

    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    hidden_dim = config.get('hidden_dim', 64)

    ckpt_files = sorted([
        fn for fn in os.listdir(run_dir)
        if fn.startswith("epoch_") and fn.endswith(".pt")
    ])
    if not ckpt_files:
        print("No checkpoints found.")
        return None

    selected = ckpt_files[::d_test]
    print(f"Checkpoints: {len(ckpt_files)} total, {len(selected)} selected (d_test={d_test})")

    device = torch.device('cpu')
    edge_index, edge_attr, V_tensor = _build_shared_graph(device)
    fd_ham = FiniteDiffHamiltonian(edge_index, edge_attr, V_tensor, device)

    # ── 生成固定测试集 ──
    np.random.seed(42)
    test_sparse_list = []
    test_fine_list   = []
    for _ in range(n_test):
        psi_fine = gen_fine_wavefunction(wf_type, k_max)  # (N_fine,)^3
        test_sparse_list.append(psi_fine[::2, ::2, ::2])  # (N_sparse,)^3
        test_fine_list.append(psi_fine)

    # ── FFT 参考（与 checkpoint 无关）──
    fft_e_all = np.array([
        _fft_energy_series(pf, n_steps) for pf in test_fine_list
    ])  # (n_test, n_steps+1)

    # ── FD baseline ──
    fd_e_all = np.array([
        _gnn_energy_series(ps, fd_ham, n_steps, device)
        for ps in test_sparse_list
    ])
    fd_rel = np.abs(fd_e_all - fft_e_all) / (np.abs(fft_e_all) + 1e-10)
    fd_mean_err = fd_rel.mean(axis=0)

    steps = np.arange(n_steps + 1)

    # ── 遍历 checkpoint ──
    results = []
    for ckpt_fn in selected:
        ckpt      = torch.load(os.path.join(run_dir, ckpt_fn), map_location=device)
        epoch_num = ckpt['epoch']
        model = HamiltonianGNN(hidden_dim=hidden_dim).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        def apply_H(u, _model=model):
            with torch.no_grad():
                return _model(u, edge_index, edge_attr, V_tensor)

        gnn_e_all = np.array([
            _gnn_energy_series(ps, apply_H, n_steps, device)
            for ps in test_sparse_list
        ])
        rel_err  = np.abs(gnn_e_all - fft_e_all) / (np.abs(fft_e_all) + 1e-10)
        mean_err = rel_err.mean(axis=0)

        results.append({'epoch': epoch_num, 'mean_rel_err': mean_err})
        print(f"  epoch {epoch_num:5d} | err n=0: {mean_err[0]:.3e}"
              f" | n={n_steps}: {mean_err[-1]:.3e}")

    # ── 画图 ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：误差 vs n
    ax    = axes[0]
    cmap  = plt.cm.viridis
    cols  = cmap(np.linspace(0, 1, len(results)))
    for r, c in zip(results, cols):
        ax.semilogy(steps, r['mean_rel_err'], '-o', ms=4, lw=1.5,
                    color=c, label=f"ep {r['epoch']}")
    ax.semilogy(steps, fd_mean_err, 'r--^', ms=5, lw=2, label='FD baseline')
    ax.set_xlabel("n (H applications)")
    ax.set_ylabel("Mean Relative Energy Error")
    ax.set_title("GNN Error vs H Applications")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # 右图：误差 vs epoch
    ax     = axes[1]
    epochs = [r['epoch'] for r in results]
    for step_n in [1, max(1, n_steps // 2), n_steps]:
        ax.semilogy(epochs, [r['mean_rel_err'][step_n] for r in results],
                    '-o', ms=4, lw=1.5, label=f"n={step_n}")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Mean Relative Energy Error")
    ax.set_title("Error vs Training Epoch")
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(run_dir, "gnn_test_energy.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"GNN test plot → {save_path}")
    return results
