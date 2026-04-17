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
from .data    import generate_wavefunction_and_target, generate_chain, gen_fine_wavefunction
from .graph   import build_graph, build_star_graph
from .model   import HamiltonianGNN, HamiltonianGNN_Cross
from .dataset import try_load_dataset


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
    chain_bptt:      bool  = False,      # True: no detach in auto → full BPTT across steps
    kinetic_cutoff:  float = 30.0,
    output_root:     str   = ".",
    device:          str   = "auto",   # "auto" | "cpu" | "cuda"
    dataset_dir:     str   = None,     # pregenerated dataset directory; None = on-the-fly
    graph_type:      str   = 'cube',   # 'cube' = 3×3×3 Mehrstellen; 'cross' = star+correction
    fd_order:        int   = 4,        # FD order for graph_type='cross'
    n_co:            int   = 3,        # correction cube size for graph_type='cross'
):
    """
    训练 HamiltonianGNN / HamiltonianGNN_Cross，返回 (model, run_dir, loss_history)。

    Parameters
    ----------
    output_root  : gnn_models/ 将创建在此目录下（默认"."即仓库根目录）。
    dataset_dir  : 预生成数据集目录（由 gen_dataset 模式产生）。
                   提供时从磁盘/内存加载固定样本，忽略 wf_type/k_max 的随机生成。
    graph_type   : 'cube' 使用 3×3×3 Mehrstellen 图（原有），
                   'cross' 使用高阶十字星+correction 两套边图。
    fd_order     : graph_type='cross' 时，FD 差分阶数（偶数，默认 4）。
    n_co         : graph_type='cross' 时，correction 立方体边长（奇数，默认 3）。
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
        chain_len=chain_len, chain_mode=chain_mode, chain_bptt=chain_bptt,
        kinetic_cutoff=kinetic_cutoff,
        dataset_dir=dataset_dir,
        graph_type=graph_type, fd_order=fd_order, n_co=n_co,
        L=L, d_fine=d_fine, d_sparse=d_sparse,
        N_fine=N_fine, N_sparse=N_sparse,
        A_pot=A_pot, sigma_pot=sigma_pot,
        timestamp=timestamp,
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved → {run_dir}/config.json")

    # ── 构建图 ──
    print(f"Building graph (graph_type={graph_type}" +
          (f", fd_order={fd_order}, n_co={n_co}" if graph_type == 'cross' else "") + ")...")
    V_tensor = torch.tensor(V_sparse.flatten(), dtype=torch.float32).unsqueeze(-1)

    if graph_type == 'cross':
        fd_edge_index, fd_edge_attr, co_edge_index, co_edge_attr = build_star_graph(fd_order, n_co)
        edge_index = edge_attr = None          # not used in cross mode
    else:
        edge_index, edge_attr = build_graph()
        fd_edge_index = fd_edge_attr = co_edge_index = co_edge_attr = None

    if device == "auto":
        _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        _dev = torch.device(device)

    # Try the requested device; if CUDA OOM at graph loading, fall back to CPU
    def _to_dev(t, d):
        return t.to(d) if t is not None else None

    try:
        if graph_type == 'cross':
            fd_edge_index = fd_edge_index.to(_dev)
            fd_edge_attr  = fd_edge_attr.to(_dev)
            co_edge_index = co_edge_index.to(_dev)
            co_edge_attr  = co_edge_attr.to(_dev)
        else:
            edge_index = edge_index.to(_dev)
            edge_attr  = edge_attr.to(_dev)
        V_tensor = V_tensor.to(_dev)
        device   = _dev
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() and _dev.type == 'cuda':
            print(f"WARNING: CUDA OOM when loading graph ({e}). Falling back to CPU.")
            torch.cuda.empty_cache()
            device = torch.device('cpu')
            if graph_type == 'cross':
                fd_edge_index = fd_edge_index.to(device)
                fd_edge_attr  = fd_edge_attr.to(device)
                co_edge_index = co_edge_index.to(device)
                co_edge_attr  = co_edge_attr.to(device)
            else:
                edge_index = edge_index.to(device)
                edge_attr  = edge_attr.to(device)
            V_tensor = V_tensor.to(device)
        else:
            raise
    print(f"Device: {device}")

    # ── 模型与优化器 ──
    if graph_type == 'cross':
        model = HamiltonianGNN_Cross(hidden_dim=hidden_dim).to(device)
    else:
        model = HamiltonianGNN(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history      = []
    step_loss_history = []   # list of lists: [epoch][step]
    epoch_recorded    = []

    # ── 封装 forward（屏蔽两种图类型的接口差异）──
    if graph_type == 'cross':
        def apply_model(u):
            return model(u, fd_edge_index, fd_edge_attr, co_edge_index, co_edge_attr, V_tensor)
    else:
        def apply_model(u):
            return model(u, edge_index, edge_attr, V_tensor)

    # ── 加载数据集（如有）──
    dataset = None
    if dataset_dir is not None:
        dataset = try_load_dataset(dataset_dir)
        n_ds    = len(dataset)
        print(f"Using fixed dataset ({n_ds} samples) from {dataset_dir}")

    # ── 训练循环 ──
    bptt_tag   = "+bptt" if (chain_mode == 'auto' and chain_bptt) else ""
    chain_info = (f"chain_len={chain_len}, mode={chain_mode}{bptt_tag}, kinetic_cutoff={kinetic_cutoff}"
                  if chain_len > 1 else "single-step")
    data_src = f"dataset({len(dataset)})" if dataset is not None else "on-the-fly"
    print(f"Training for {epochs} epochs "
          f"({batch_per_epoch} steps/epoch, batch_size={batch_size}, "
          f"{chain_info}, data={data_src})...")
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
                # ── sample one wavefunction (from dataset or on-the-fly) ──
                if dataset is not None:
                    pairs_raw = dataset[int(np.random.randint(0, len(dataset)))]
                    # pairs_raw: list of (psi_tensor, target_tensor), already float32 [N,1]
                    # truncate/use as many steps as chain_len allows
                    pairs_raw = pairs_raw[:chain_len]
                else:
                    pairs_raw = None   # sentinel: generate below

                if chain_len == 1:
                    if pairs_raw is not None:
                        u_in, target = pairs_raw[0]
                        u_in   = u_in.to(device)
                        target = target.to(device)
                    else:
                        # Single-step: one (psi, H·psi) pair per psi
                        psi_s, H_target = generate_wavefunction_and_target(
                            wf_type=wf_type, k_max=k_max)
                        u_in   = torch.tensor(
                            psi_s.flatten(),    dtype=torch.float32).unsqueeze(-1).to(device)
                        target = torch.tensor(
                            H_target.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
                    pred = apply_model(u_in)
                    sl = criterion(pred, target)
                    batch_loss     = batch_loss + sl
                    step_sum[0]    = step_sum[0] + sl.detach()
                    step_n[0]     += 1
                else:
                    # Chain: chain_len steps per psi_0
                    if pairs_raw is None:
                        psi_0_fine = gen_fine_wavefunction(wf_type, k_max)
                        pairs_raw_np = generate_chain(psi_0_fine, chain_len, kinetic_cutoff)
                        # Convert to tensors on-the-fly (lazy, stay as numpy for now)
                        pairs_iter = pairs_raw_np
                        use_np = True
                    else:
                        pairs_iter = pairs_raw
                        use_np = False

                    psi_chain_loss = torch.zeros(1, device=device)
                    prev_pred = None   # used only in 'auto' mode
                    for k, step_data in enumerate(pairs_iter):
                        if use_np:
                            psi_s, H_target = step_data
                        else:
                            psi_s, H_target = step_data   # tensors [N,1]

                        if chain_mode == 'auto' and k > 0 and prev_pred is not None:
                            # Autoregressive: feed normalised GNN output as next input.
                            # chain_bptt=False: detach → independent per-step gradients
                            # chain_bptt=True : no detach → full BPTT across chain
                            src  = prev_pred if chain_bptt else prev_pred.detach()
                            nrm  = torch.norm(src) + 1e-30
                            u_in = src / nrm
                        else:
                            # Teacher forcing (default): use FFT reference chain input
                            if use_np:
                                u_in = torch.tensor(
                                    psi_s.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
                            else:
                                u_in = psi_s.to(device)

                        if use_np:
                            target = torch.tensor(
                                H_target.flatten(), dtype=torch.float32).unsqueeze(-1).to(device)
                        else:
                            target = H_target.to(device)

                        pred = apply_model(u_in)
                        prev_pred = pred
                        sl = criterion(pred, target)
                        psi_chain_loss = psi_chain_loss + sl
                        step_sum[k]    = step_sum[k] + sl.detach()
                        step_n[k]     += 1
                    # Contribute mean-over-steps to batch (same scale as chain_len=1)
                    batch_loss = batch_loss + psi_chain_loss / len(pairs_iter)

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
