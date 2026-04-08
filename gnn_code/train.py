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
from .data  import generate_wavefunction_and_target
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
    output_root:     str   = ".",       # 仓库根目录（run_gnn.py 从此处运行）
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

    loss_history   = []
    epoch_recorded = []

    # ── 训练循环 ──
    print(f"Training for {epochs} epochs ({batch_per_epoch} batches/epoch)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(batch_per_epoch):
            psi_s, H_target = generate_wavefunction_and_target(
                wf_type=wf_type, k_max=k_max)

            u_in   = torch.tensor(
                psi_s.flatten(),    dtype=torch.float32).unsqueeze(-1).to(device)
            target = torch.tensor(
                H_target.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)

            optimizer.zero_grad()
            pred = model(u_in, edge_index, edge_attr, V_tensor)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / batch_per_epoch
        loss_history.append(avg_loss)
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
        json.dump({"epoch": epoch_recorded, "loss": loss_history}, f)

    # ── loss 曲线 ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epoch_recorded, loss_history, lw=1.5, color="steelblue")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss ({wf_type}, hidden={hidden_dim})")
    ax.grid(True, which='both', ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Loss curve → {run_dir}/loss_curve.png")
    print("Training done.")

    return model, run_dir, loss_history
