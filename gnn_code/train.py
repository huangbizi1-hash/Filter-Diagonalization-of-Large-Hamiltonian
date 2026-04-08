"""
train.py — 训练循环

训练 HamiltonianGNN，将检查点和训练记录保存到
<output_root>/gnn_models/<timestamp>/（默认 output_root 为仓库根目录）。

输出文件（run_dir = <output_root>/gnn_models/<timestamp>/）
  config.json        — 所有超参数
  epoch_NNNNN.pt     — 检查点（每 save_every epoch 保存一次）
  loss_history.json  — 每 epoch 的 loss 列表
  loss_curve.png     — loss 曲线（对数坐标）
"""

import os
import json
import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .physics import (
    L, d_fine, d_sparse, N_fine, N_sparse, A_pot, sigma_pot, V_sparse,
)
from .data  import generate_wavefunction_and_target, generate_chain, gen_fine_wavefunction
from .graph import build_graph
from .model import HamiltonianGNN


def train(
    wf_type:         str   = 'gaussian',
    k_max:           int   = 2,
    hidden_dim:      int   = 64,
    epochs:          int   = 5000,
    batch_per_epoch: int   = 10,
    save_every:      int   = 500,
    lr:              float = 1e-3,
    chain_len:       int   = 1,
    kinetic_cutoff:  float = 30.0,
    output_root:     str   = ".",
):
    """
    训练 HamiltonianGNN，返回 (model, run_dir, loss_history)。

    Parameters
    ----------
    output_root : str
        gnn_models/ 将创建在此目录下（默认"."即仓库根目录）。
    """
    # ── 建立输出目录 ──
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(output_root, "gnn_models", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # ── 保存超参数 ──
    config = dict(
        wf_type=wf_type, k_max=k_max,
        hidden_dim=hidden_dim, epochs=epochs,
        batch_per_epoch=batch_per_epoch,
        save_every=save_every, lr=lr,
        chain_len=chain_len, kinetic_cutoff=kinetic_cutoff,
        L=L, d_fine=d_fine, d_sparse=d_sparse,
        N_fine=N_fine, N_sparse=N_sparse,
        A_pot=A_pot, sigma_pot=sigma_pot,
        timestamp=timestamp,
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved → {run_dir}/config.json")

    # ── 构建图 ──
    print("Building graph...")
    edge_index, edge_attr = build_graph()
    V_tensor = torch.tensor(
        V_sparse.flatten(), dtype=torch.float32).unsqueeze(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    edge_index = edge_index.to(device)
    edge_attr  = edge_attr.to(device)
    V_tensor   = V_tensor.to(device)

    # ── 模型与优化器 ──
    model     = HamiltonianGNN(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history      = []
    step_loss_history = []   # list of lists: [epoch][step]
    epoch_recorded    = []

    # ── 训练循环 ──
    chain_info = (f"chain_len={chain_len}, kinetic_cutoff={kinetic_cutoff}"
                  if chain_len > 1 else "single-step")
    print(f"Training for {epochs} epochs ({batch_per_epoch} batches/epoch, {chain_info})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss  = 0.0
        step_totals = [0.0] * chain_len   # sum of per-step loss across batches
        step_counts = [0]   * chain_len   # number of batches that reached step k

        for _ in range(batch_per_epoch):
            optimizer.zero_grad()

            if chain_len == 1:
                # ── Original single-step: one (psi, H·psi) pair per sample ──
                psi_s, H_target = generate_wavefunction_and_target(
                    wf_type=wf_type, k_max=k_max)
                u_in   = torch.tensor(
                    psi_s.flatten(),    dtype=torch.float32).unsqueeze(-1).to(device)
                target = torch.tensor(
                    H_target.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
                pred = model(u_in, edge_index, edge_attr, V_tensor)
                loss = criterion(pred, target)
                step_totals[0] += loss.item()
                step_counts[0] += 1
            else:
                # ── Chain: generate chain_len (psi_k, H·psi_k) pairs from one psi_0 ──
                # Each psi_k is L2-normalised; H·psi_k is the unnormalised target.
                psi_0_fine = gen_fine_wavefunction(wf_type, k_max)
                pairs = generate_chain(psi_0_fine, chain_len, kinetic_cutoff)
                chain_loss  = torch.zeros(1, device=device)
                step_losses = []
                for psi_s, H_target in pairs:
                    u_in   = torch.tensor(
                        psi_s.flatten(),    dtype=torch.float32).unsqueeze(-1).to(device)
                    target = torch.tensor(
                        H_target.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
                    pred = model(u_in, edge_index, edge_attr, V_tensor)
                    sl = criterion(pred, target)
                    step_losses.append(sl)
                    chain_loss = chain_loss + sl
                loss = chain_loss / len(pairs)   # average over chain steps
                for k, sl in enumerate(step_losses):
                    step_totals[k] += sl.item()
                    step_counts[k] += 1

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / batch_per_epoch
        step_avgs = [
            step_totals[k] / step_counts[k] if step_counts[k] > 0 else float('nan')
            for k in range(chain_len)
        ]
        loss_history.append(avg_loss)
        step_loss_history.append(step_avgs)
        epoch_recorded.append(epoch)

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:5d} | Loss: {avg_loss:.6e}")

        # ── 定期保存检查点 ──
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(run_dir, f"epoch_{epoch:05d}.pt")
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                 avg_loss,
                'config':               config,
            }, ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")

    # ── 保存 loss ──
    with open(os.path.join(run_dir, "loss_history.json"), "w") as f:
        json.dump({
            "epoch":       epoch_recorded,
            "loss":        loss_history,
            "step_losses": step_loss_history,   # list[epoch][step]
        }, f)

    # ── loss 曲线 ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epoch_recorded, loss_history,
                lw=2.0, color="steelblue", label="avg", zorder=10)
    if chain_len > 1:
        colors = plt.cm.Reds(np.linspace(0.35, 0.90, chain_len))
        for k in range(chain_len):
            vals = [step_loss_history[i][k] for i in range(len(step_loss_history))]
            ax.semilogy(epoch_recorded, vals,
                        lw=1.0, color=colors[k], alpha=0.75, label=f"step {k}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss ({wf_type}, hidden={hidden_dim}, chain={chain_len})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Loss curve → {run_dir}/loss_curve.png")
    print("Training done.")

    return model, run_dir, loss_history
