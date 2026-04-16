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
    batch_size:      int   = 1,
    save_every:      int   = 500,
    lr:              float = 1e-3,
    chain_len:       int   = 1,
    chain_mode:      str   = 'teacher',  # 'teacher' | 'auto'
    kinetic_cutoff:  float = 30.0,
    output_root:     str   = ".",
    device:          str   = "auto",   # "auto" | "cpu" | "cuda"
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
        batch_per_epoch=batch_per_epoch, batch_size=batch_size,
        save_every=save_every, lr=lr,
        chain_len=chain_len, chain_mode=chain_mode, kinetic_cutoff=kinetic_cutoff,
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

    if device == "auto":
        _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _dev = torch.device(device)
    # Try the requested device; if CUDA OOM at graph loading, fall back to CPU
    try:
        edge_index = edge_index.to(_dev)
        edge_attr  = edge_attr.to(_dev)
        V_tensor   = V_tensor.to(_dev)
        device     = _dev
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() and _dev.type == 'cuda':
            print(f"WARNING: CUDA OOM when loading graph ({e}). Falling back to CPU.")
            torch.cuda.empty_cache()
            device     = torch.device('cpu')
            edge_index = edge_index.to(device)
            edge_attr  = edge_attr.to(device)
            V_tensor   = V_tensor.to(device)
        else:
            raise
    print(f"Device: {device}")

    # ── 模型与优化器 ──
    model     = HamiltonianGNN(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history      = []
    step_loss_history = []   # list of lists: [epoch][step]
    epoch_recorded    = []

    # ── 训练循环 ──
    chain_info = (f"chain_len={chain_len}, mode={chain_mode}, kinetic_cutoff={kinetic_cutoff}"
                  if chain_len > 1 else "single-step")
    print(f"Training for {epochs} epochs "
          f"({batch_per_epoch} steps/epoch, batch_size={batch_size}, {chain_info})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss  = 0.0
        step_totals = [0.0] * chain_len   # sum of per-step loss across optimizer steps
        step_counts = [0]   * chain_len

        for _ in range(batch_per_epoch):
            optimizer.zero_grad()

            # Accumulate over batch_size independent psi's; share one backward pass.
            # loss scale is independent of batch_size (average, not sum).
            batch_loss = torch.zeros(1, device=device)
            # Step-level accumulators for logging (detached)
            step_sum = [torch.zeros(1, device=device) for _ in range(chain_len)]
            step_n   = [0] * chain_len

            for _ in range(batch_size):
                if chain_len == 1:
                    # Single-step: one (psi, H·psi) pair per psi
                    psi_s, H_target = generate_wavefunction_and_target(
                        wf_type=wf_type, k_max=k_max)
                    u_in   = torch.tensor(
                        psi_s.flatten(),    dtype=torch.float32).unsqueeze(-1).to(device)
                    target = torch.tensor(
                        H_target.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
                    pred = model(u_in, edge_index, edge_attr, V_tensor)
                    sl = criterion(pred, target)
                    batch_loss     = batch_loss + sl
                    step_sum[0]    = step_sum[0] + sl.detach()
                    step_n[0]     += 1
                else:
                    # Chain: chain_len steps per psi_0
                    psi_0_fine = gen_fine_wavefunction(wf_type, k_max)
                    pairs = generate_chain(psi_0_fine, chain_len, kinetic_cutoff)
                    psi_chain_loss = torch.zeros(1, device=device)
                    prev_pred = None   # used only in 'auto' mode
                    for k, (psi_s, H_target) in enumerate(pairs):
                        if chain_mode == 'auto' and k > 0 and prev_pred is not None:
                            # Autoregressive: feed normalised GNN output as next input.
                            # Detach so gradients don't flow across steps.
                            nrm  = torch.norm(prev_pred.detach()) + 1e-30
                            u_in = prev_pred.detach() / nrm
                        else:
                            # Teacher forcing (default): use FFT reference chain input
                            u_in = torch.tensor(
                                psi_s.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
                        target = torch.tensor(
                            H_target.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
                        pred = model(u_in, edge_index, edge_attr, V_tensor)
                        prev_pred = pred
                        sl = criterion(pred, target)
                        psi_chain_loss = psi_chain_loss + sl
                        step_sum[k]    = step_sum[k] + sl.detach()
                        step_n[k]     += 1
                    # Contribute mean-over-steps to batch (same scale as chain_len=1)
                    batch_loss = batch_loss + psi_chain_loss / len(pairs)

            loss = batch_loss / batch_size   # mean over psi's in batch
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Per-step averages over the batch (for logging only)
            for k in range(chain_len):
                if step_n[k] > 0:
                    step_totals[k] += (step_sum[k] / step_n[k]).item()
                    step_counts[k] += 1

        avg_loss = total_loss / batch_per_epoch
        step_avgs = [
            step_totals[k] / step_counts[k] if step_counts[k] > 0 else float('nan')
            for k in range(chain_len)
        ]
        loss_history.append(avg_loss)
        step_loss_history.append(step_avgs)
        epoch_recorded.append(epoch)

        if epoch % 100 == 0:
            step_str = "  ".join(f"s{k}:{step_avgs[k]:.3e}" for k in range(chain_len))
            print(f"  Epoch {epoch:5d} | Loss: {avg_loss:.6e}  [{step_str}]")

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
