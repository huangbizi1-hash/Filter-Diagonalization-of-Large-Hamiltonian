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
from .graph import build_graph, build_star_graph
from .model import (HamiltonianGNN, FiniteDiffHamiltonian,
                    HamiltonianGNN_Cross, FiniteDiffHamiltonian_Cross)


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
    # Normalize initial vector to avoid overflow during repeated H applications
    u = u / (torch.norm(u) + 1e-30)
    energies = []
    for _ in range(n_steps + 1):
        H_u    = apply_H(u)
        norm2  = torch.sum(u**2).item() * d_sparse**3
        energy = torch.sum(u * H_u).item() * d_sparse**3
        energies.append(energy / norm2)
        # Normalize before next application to prevent numerical overflow
        nrm = torch.norm(H_u)
        if nrm < 1e-30 or not torch.isfinite(nrm):
            energies.extend([float('nan')] * (n_steps - len(energies) + 1))
            break
        u = H_u / nrm
    return energies


def _fft_energy_series(psi_fine: np.ndarray, n_steps: int) -> list:
    """
    在 fine grid 上用 FFT 哈密顿量重复作用 n_steps 次，
    返回各步的精确 Ritz 能量（ground truth）。
    """
    psi = psi_fine.copy()
    nrm = np.linalg.norm(psi)
    if nrm > 0:
        psi /= nrm
    energies = []
    for _ in range(n_steps + 1):
        H_psi  = fft_hamiltonian(psi)
        norm2  = np.sum(psi**2) * d_fine**3
        energy = np.sum(psi * H_psi) * d_fine**3
        energies.append(energy / norm2)
        nrm = np.linalg.norm(H_psi)
        if nrm < 1e-30 or not np.isfinite(nrm):
            energies.extend([float('nan')] * (n_steps - len(energies) + 1))
            break
        psi = H_psi / nrm
    return energies


def _build_shared_graph(device, graph_type='cube', fd_order=4, n_co=3):
    """
    构建图并移动到 device。

    Returns
    -------
    cube : (edge_index, edge_attr, V_tensor)
    cross: (fd_edge_index, fd_edge_attr, co_edge_index, co_edge_attr, V_tensor)
    """
    V_tensor = torch.tensor(
        V_sparse.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
    if graph_type == 'cross':
        fd_ei, fd_ea, co_ei, co_ea = build_star_graph(fd_order, n_co)
        return (fd_ei.to(device), fd_ea.to(device),
                co_ei.to(device), co_ea.to(device), V_tensor)
    else:
        edge_index, edge_attr = build_graph()
        return (edge_index.to(device), edge_attr.to(device), V_tensor)


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
    abs_err = np.abs(fd_e - fft_e)
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
    mean_err = abs_err.mean(axis=0)
    std_err  = abs_err.std(axis=0)
    ax.semilogy(steps, mean_err, 'b-o', lw=1.5, label='mean |ΔE|')
    ax.fill_between(steps,
                    np.maximum(mean_err - std_err, 1e-12),
                    mean_err + std_err, alpha=0.2)
    ax.set_xlabel("n (H applications)")
    ax.set_ylabel("Absolute Energy Error |ΔE|")
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.set_title("Baseline: Absolute Error vs n")

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
    hidden_dim  = config.get('hidden_dim',  64)
    graph_type  = config.get('graph_type',  'cube')
    fd_order    = config.get('fd_order',    4)
    n_co        = config.get('n_co',        3)

    ckpt_files = sorted(
        [fn for fn in os.listdir(run_dir)
         if fn.startswith("epoch_") and fn.endswith(".pt")],
        key=lambda fn: int(fn[len("epoch_"):-len(".pt")])   # numeric sort by epoch
    )
    if not ckpt_files:
        print("No checkpoints found.")
        return None

    selected = ckpt_files[::d_test]
    print(f"Checkpoints: {len(ckpt_files)} total, {len(selected)} selected (d_test={d_test})")
    print(f"Graph type: {graph_type}" +
          (f" (fd_order={fd_order}, n_co={n_co})" if graph_type == 'cross' else ""))

    device    = torch.device('cpu')
    graph_tup = _build_shared_graph(device, graph_type, fd_order, n_co)

    if graph_type == 'cross':
        fd_ei, fd_ea, co_ei, co_ea, V_tensor = graph_tup
        fd_ham = FiniteDiffHamiltonian_Cross(fd_ei, fd_ea, V_tensor, device)
    else:
        edge_index, edge_attr, V_tensor = graph_tup
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
    fd_mean_err = np.abs(fd_e_all - fft_e_all).mean(axis=0)

    steps = np.arange(n_steps + 1)

    # ── 遍历 checkpoint ──
    results = []
    for ckpt_fn in selected:
        ckpt      = torch.load(os.path.join(run_dir, ckpt_fn), map_location=device)
        epoch_num = ckpt['epoch']
        if graph_type == 'cross':
            model = HamiltonianGNN_Cross(hidden_dim=hidden_dim).to(device)
        else:
            model = HamiltonianGNN(hidden_dim=hidden_dim).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        if graph_type == 'cross':
            def apply_H(u, _m=model):
                with torch.no_grad():
                    return _m(u, fd_ei, fd_ea, co_ei, co_ea, V_tensor)
        else:
            def apply_H(u, _m=model):
                with torch.no_grad():
                    return _m(u, edge_index, edge_attr, V_tensor)

        gnn_e_all = np.array([
            _gnn_energy_series(ps, apply_H, n_steps, device)
            for ps in test_sparse_list
        ])
        mean_err = np.abs(gnn_e_all - fft_e_all).mean(axis=0)

        results.append({'epoch': epoch_num, 'mean_abs_err': mean_err})
        print(f"  epoch {epoch_num:5d} | |ΔE| n=0: {mean_err[0]:.3e}"
              f" | n={n_steps}: {mean_err[-1]:.3e}")

    # ── 画图 ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左图：|ΔE| vs n
    ax    = axes[0]
    cmap  = plt.cm.viridis
    cols  = cmap(np.linspace(0, 1, len(results)))
    for r, c in zip(results, cols):
        ax.semilogy(steps, r['mean_abs_err'], '-o', ms=4, lw=1.5,
                    color=c, label=f"ep {r['epoch']}")
    ax.semilogy(steps, fd_mean_err, 'r--^', ms=5, lw=2, label='FD baseline')
    ax.set_xlabel("n (H applications)")
    ax.set_ylabel("Mean Absolute Energy Error |ΔE|")
    ax.set_title("GNN |ΔE| vs H Applications")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which='both', ls='--', alpha=0.4)

    # 右图：|ΔE| vs epoch（去重，避免 n_steps 小时出现重复线）
    ax       = axes[1]
    ep_list  = [r['epoch'] for r in results]
    step_ns  = sorted({1, max(1, n_steps // 2), n_steps})   # deduplicated
    for sn in step_ns:
        ax.semilogy(ep_list, [r['mean_abs_err'][sn] for r in results],
                    '-o', ms=4, lw=1.5, label=f"n={sn}")
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Mean Absolute Energy Error |ΔE|")
    ax.set_title("|ΔE| vs Training Epoch")
    ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(run_dir, "gnn_test_energy.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"GNN test plot → {save_path}")

    # Save numerical results so plots can be reproduced without re-running
    json_path = os.path.join(run_dir, "gnn_test_energy.json")
    save_data = {
        "n_steps":      n_steps,
        "n_test":       n_test,
        "wf_type":      wf_type,
        "steps":        steps.tolist(),
        "fft_mean_err": [0.0] * (n_steps + 1),   # FFT is ground truth, error = 0
        "fd_mean_err":  fd_mean_err.tolist(),
        "checkpoints":  [
            {"epoch": r["epoch"],
             "mean_abs_err": [x if np.isfinite(x) else None
                              for x in r["mean_abs_err"].tolist()]}
            for r in results
        ],
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"GNN test data  → {json_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# HO ground state test: FFT vs FD vs GNN, energy + similarity sequences
# ──────────────────────────────────────────────────────────────────────────────

def test_gnn_ho(
    run_dir:        str,
    n_steps:        int   = 8,
    omega:          float = 1.0,
    kinetic_cutoff: float = 30.0,
):
    """
    在谐振子势能上用解析基态 ψ₀ 作为初态，比较：
      - FFT  (fine grid, d=d_fine,   参考)
      - FD   (sparse grid, d=d_sparse, 无GNN基准)
      - GNN  (sparse grid, 从 run_dir 最新 checkpoint 加载)

    输出两张子图，横轴均为 H 作用次数 n：
      左图：Ritz 能量 E_n 序列（FFT/FD/GNN）
      右图：波函数与初态 ψ₀ 的相似度 |⟨ψ_n|ψ₀⟩|（越接近1说明算符越精确）

    保存到 run_dir/ho_test.png。
    """
    from .physics import (d_fine, d_sparse, N_fine, N_sparse,
                          X_f, Y_f, Z_f, X_s, Y_s, Z_s, K2_fine)

    E0_exact = 1.5 * omega
    print(f"\n=== GNN HO Test  omega={omega}  n_steps={n_steps} ===")
    print(f"  Exact E_0 = {E0_exact:.6f} Ha  (3/2*omega)")

    # ── HO potential on both grids ──
    V_ho_fine   = 0.5 * omega**2 * (X_f**2 + Y_f**2 + Z_f**2)
    V_ho_sparse = 0.5 * omega**2 * (X_s**2 + Y_s**2 + Z_s**2)

    # ── Analytic HO ground state, normalized on each grid ──
    r2_fine   = X_f**2 + Y_f**2 + Z_f**2
    r2_sparse = X_s**2 + Y_s**2 + Z_s**2
    psi0_fine   = np.exp(-omega / 2.0 * r2_fine)
    psi0_sparse = np.exp(-omega / 2.0 * r2_sparse)
    psi0_fine   /= np.sqrt(np.sum(psi0_fine**2)   * d_fine**3)
    psi0_sparse /= np.sqrt(np.sum(psi0_sparse**2) * d_sparse**3)

    # ── Helper: FFT HO series — returns (energies, similarities) ──
    def _series_fft():
        psi = psi0_fine.copy()
        T_k = np.minimum(0.5 * K2_fine, kinetic_cutoff)
        Es, sims = [], []
        for _ in range(n_steps + 1):
            Hpsi  = np.fft.ifftn(T_k * np.fft.fftn(psi)).real + V_ho_fine * psi
            norm2 = np.sum(psi**2) * d_fine**3
            Es.append(np.sum(psi * Hpsi) * d_fine**3 / norm2)
            sims.append(abs(np.sum(psi * psi0_fine) * d_fine**3))
            nrm = np.linalg.norm(Hpsi)
            if nrm < 1e-30 or not np.isfinite(nrm):
                pad = [float('nan')] * (n_steps + 1 - len(Es))
                Es.extend(pad); sims.extend(pad); break
            psi = Hpsi / nrm
        return np.array(Es), np.array(sims)

    # ── Helper: torch-based series (FD or GNN with HO V) ──
    def _series_torch(apply_H_fn):
        device = torch.device('cpu')
        u  = torch.tensor(psi0_sparse.flatten(), dtype=torch.float32).unsqueeze(-1)
        u  = u / (torch.norm(u) + 1e-30)
        p0 = torch.tensor(psi0_sparse.flatten(), dtype=torch.float32).unsqueeze(-1)
        p0 = p0 / (torch.norm(p0) + 1e-30)
        Es, sims = [], []
        for _ in range(n_steps + 1):
            Hu    = apply_H_fn(u)
            norm2 = torch.sum(u**2).item() * d_sparse**3
            Es.append(torch.sum(u * Hu).item() * d_sparse**3 / norm2)
            sims.append(abs(torch.sum(u * p0).item()))  # cosine sim; both vector-normalized → =1 at n=0
            nrm = torch.norm(Hu)
            if nrm < 1e-30 or not torch.isfinite(nrm):
                pad = [float('nan')] * (n_steps + 1 - len(Es))
                Es.extend(pad); sims.extend(pad); break
            u = Hu / nrm
        return np.array(Es), np.array(sims)

    # ── Build graph with HO potential ──
    device = torch.device('cpu')
    edge_index, edge_attr = build_graph()
    edge_index = edge_index.to(device)
    edge_attr  = edge_attr.to(device)
    V_ho_t = torch.tensor(
        V_ho_sparse.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)

    fd_ham = FiniteDiffHamiltonian(edge_index, edge_attr, V_ho_t, device)

    print("Running FFT series...", flush=True)
    e_fft, sim_fft = _series_fft()
    print("Running FD  series...", flush=True)
    e_fd,  sim_fd  = _series_torch(fd_ham)

    # ── Load GNN (latest checkpoint in run_dir) ──
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    hidden_dim = config.get('hidden_dim', 64)
    ckpt_files = sorted(
        [fn for fn in os.listdir(run_dir)
         if fn.startswith("epoch_") and fn.endswith(".pt")],
        key=lambda fn: int(fn[len("epoch_"):-len(".pt")])
    )
    if not ckpt_files:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    ckpt_path = os.path.join(run_dir, ckpt_files[-1])
    ckpt      = torch.load(ckpt_path, map_location=device)
    epoch_num = ckpt['epoch']
    model = HamiltonianGNN(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Running GNN  series (epoch={epoch_num})...", flush=True)

    def apply_gnn_ho(u):
        with torch.no_grad():
            return model(u, edge_index, edge_attr, V_ho_t)

    e_gnn, sim_gnn = _series_torch(apply_gnn_ho)

    steps = np.arange(n_steps + 1)

    # ── Print table ──
    print(f"\n  {'n':>3}  {'E_FFT':>10}  {'E_FD':>10}  {'E_GNN':>10}  "
          f"{'sim_FFT':>9}  {'sim_FD':>9}  {'sim_GNN':>9}")
    print("  " + "─" * 72)
    for n in steps:
        ef  = f"{e_fft[n]:10.5f}"  if np.isfinite(e_fft[n])  else "       nan"
        efd = f"{e_fd[n]:10.5f}"   if np.isfinite(e_fd[n])   else "       nan"
        eg  = f"{e_gnn[n]:10.5f}"  if np.isfinite(e_gnn[n])  else "       nan"
        sf  = f"{sim_fft[n]:9.5f}" if np.isfinite(sim_fft[n]) else "      nan"
        sfd = f"{sim_fd[n]:9.5f}"  if np.isfinite(sim_fd[n])  else "      nan"
        sg  = f"{sim_gnn[n]:9.5f}" if np.isfinite(sim_gnn[n]) else "      nan"
        print(f"  {n:>3}  {ef}  {efd}  {eg}  {sf}  {sfd}  {sg}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: energy series
    ax = axes[0]
    ax.plot(steps, e_fft, 'r-s',  lw=2, ms=6, label=f'FFT (d={d_fine})')
    ax.plot(steps, e_fd,  'b-o',  lw=2, ms=6, label=f'FD  (d={d_sparse})')
    ax.plot(steps, e_gnn, 'g-^',  lw=2, ms=6, label=f'GNN (epoch={epoch_num})')
    ax.axhline(E0_exact, color='k', ls='--', lw=1.4,
               label=f'Exact E₀ = {E0_exact:.4f} Ha')
    ax.set_xlabel("n  (H applications)")
    ax.set_ylabel("Ritz energy  ⟨H⟩ₙ  (Ha)")
    ax.set_title(f"HO ground state  (ω={omega})\nEnergy vs H applications")
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)

    # Right: wavefunction similarity |<ψ_n|ψ₀>|
    ax = axes[1]
    ax.plot(steps, sim_fft, 'r-s', lw=2, ms=6, label='FFT')
    ax.plot(steps, sim_fd,  'b-o', lw=2, ms=6, label='FD')
    ax.plot(steps, sim_gnn, 'g-^', lw=2, ms=6, label=f'GNN (epoch={epoch_num})')
    ax.axhline(1.0, color='k', ls='--', lw=1.0, label='ideal = 1')
    ax.set_xlabel("n  (H applications)")
    ax.set_ylabel("|⟨ψₙ|ψ₀⟩|  (overlap with initial ground state)")
    ax.set_title("Wavefunction similarity\n(1 = perfect eigenstate preservation)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(True, ls='--', alpha=0.4)

    fig.tight_layout()
    save_path = os.path.join(run_dir, "ho_test.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {save_path}")

    return {
        "omega": omega, "E0_exact": E0_exact,
        "epoch": epoch_num, "n_steps": n_steps,
        "steps":   steps.tolist(),
        "e_fft":   e_fft.tolist(),  "sim_fft":  sim_fft.tolist(),
        "e_fd":    e_fd.tolist(),   "sim_fd":   sim_fd.tolist(),
        "e_gnn":   e_gnn.tolist(),  "sim_gnn":  sim_gnn.tolist(),
    }
