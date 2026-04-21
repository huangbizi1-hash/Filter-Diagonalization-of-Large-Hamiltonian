"""
gnn_code/data_scaling.py — Experiment 1: training data quantity vs test loss

固定 hidden_dim=16, epochs=5000, teacher forcing, chain=1, fd_order=6, n_co=3。
分别用 N_k=4,5,6,8,10 的 k-grid 数据集训练，统一测试集（N_k'=10，k 偏移 > 2*k_max，
避免与任意训练集重叠），记录并绘制 test_loss vs N_k³。

用法（通过 run_gnn.py）：
    python run_gnn.py --mode data_scaling \\
        --scaling_n_k_list 4 5 6 8 10 \\
        --scaling_k_max 5 \\
        --scaling_epochs 5000 \\
        --scaling_hidden_dim 16
"""

import datetime
import json
import os
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCI_STYLE = {
    'font.size': 16, 'axes.labelsize': 16, 'axes.titlesize': 18,
    'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
    'legend.fontsize': 14, 'legend.loc': 'best', 'font.weight': 'normal',
    'mathtext.fontset': 'cm', 'mathtext.default': 'regular',
    'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3,
}


# ── test loss helper ──────────────────────────────────────────────────────────

def _eval_test_loss(run_dir: str, test_dir: str, device_str: str = 'cpu') -> float:
    """
    Load the latest checkpoint from run_dir, evaluate MSE on test_dir dataset.
    Loss formula matches training: pred = model(ψ/‖ψ‖)·‖ψ‖, loss = MSE(pred, target).
    Returns mean MSE over all test samples.
    """
    import torch
    import torch.nn as nn

    from .graph   import build_star_graph
    from .model   import HamiltonianGNN_Cross
    from .dataset import WavefunctionDataset
    from .physics import V_sparse, N_sparse, d_sparse

    dev = torch.device(device_str)

    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)

    hidden_dim = cfg.get('hidden_dim', 64)
    fd_order   = cfg.get('fd_order',   6)
    n_co       = cfg.get('n_co',       3)
    d_cfg      = float(cfg.get('d_sparse', d_sparse))
    grid_L     = N_sparse * d_cfg

    # Build graph on training grid (N_sparse)
    fd_ei, fd_ea, co_ei, co_ea = build_star_graph(
        fd_order, n_co, N=N_sparse, d=d_cfg, grid_L=grid_L)
    fd_ei = fd_ei.to(dev); fd_ea = fd_ea.to(dev)
    co_ei = co_ei.to(dev); co_ea = co_ea.to(dev)
    V_t   = torch.tensor(V_sparse.flatten(), dtype=torch.float32).unsqueeze(-1).to(dev)

    model = HamiltonianGNN_Cross(hidden_dim=hidden_dim).to(dev)

    ckpts = sorted(
        [fn for fn in os.listdir(run_dir)
         if fn.startswith('epoch_') and fn.endswith('.pt')],
        key=lambda fn: int(fn[len('epoch_'):-len('.pt')]),
    )
    if not ckpts:
        raise RuntimeError(f"No checkpoints in {run_dir}")
    ckpt = torch.load(os.path.join(run_dir, ckpts[-1]), map_location=dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    criterion = nn.MSELoss()
    ds        = WavefunctionDataset(test_dir, in_memory=True)
    total     = 0.0

    with torch.no_grad():
        for sample in ds:
            psi_t, tgt_t = sample[0]   # chain_len=1
            psi_t = psi_t.to(dev); tgt_t = tgt_t.to(dev)
            nrm   = torch.norm(psi_t) + 1e-30
            pred  = model(psi_t / nrm, fd_ei, fd_ea, co_ei, co_ea, V_t) * nrm
            total += criterion(pred, tgt_t).item()

    return total / len(ds)


# ── main pipeline ─────────────────────────────────────────────────────────────

def data_scaling_experiment(
    n_k_list:     list  = None,
    k_max:        int   = 5,
    test_n_k:     int   = 10,
    fd_order:     int   = 6,
    n_co:         int   = 3,
    hidden_dim:   int   = 16,
    epochs:       int   = 5000,
    save_every:   int   = 500,
    lr:           float = 1e-3,
    batch_per_epoch: int = 10,
    device:       str   = 'cpu',
    output_root:  str   = '.',
    description:  str   = '',
):
    """
    训练 N_k³ 个样本（N_k ∈ n_k_list）的 GNN，统一在 test 集上评估 MSE 损失，
    绘制 test_loss vs N_k³ 图，保存 JSON。

    测试集使用 k_shift = 2*k_max + 0.5，保证与任意训练集（k 在 [-k_max, k_max] 内）无重叠。
    """
    if n_k_list is None:
        n_k_list = [4, 5, 6, 8, 10]

    from .train   import train
    from .dataset import generate_k_grid_dataset

    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_root = out_dir / "gnn_datasets" / f"scaling_kmax{k_max}"
    ds_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Data scaling: n_k_list={n_k_list}  k_max={k_max}")
    print(f"  fd_order={fd_order}  n_co={n_co}  hidden_dim={hidden_dim}  epochs={epochs}")
    print(f"{'='*64}")

    # ── 生成测试集（一次性，固定）──────────────────────────────────────────────
    k_shift_test = float(2 * k_max + 0.5)   # test k 在 [k_max+0.5, 3*k_max+0.5]*dk
    test_dir = str(ds_root / f"test_nk{test_n_k}_shift{k_shift_test}")
    if not os.path.exists(os.path.join(test_dir, "metadata.json")):
        print(f"\n  Generating test dataset (N_k'={test_n_k}, k_shift={k_shift_test})...")
        generate_k_grid_dataset(
            k_max=k_max, n_k=test_n_k, chain_len=1,
            k_shift=k_shift_test, out_dir=test_dir)
    else:
        print(f"\n  Test dataset already exists: {test_dir}")

    # ── 逐 N_k 训练 + 评估 ────────────────────────────────────────────────────
    results = []
    for n_k in n_k_list:
        n_samples = n_k ** 3
        train_dir = str(ds_root / f"train_nk{n_k}")
        print(f"\n  ── N_k={n_k} ({n_samples} samples) ──")

        # 生成训练集
        if not os.path.exists(os.path.join(train_dir, "metadata.json")):
            print(f"  Generating training dataset (N_k={n_k})...")
            generate_k_grid_dataset(
                k_max=k_max, n_k=n_k, chain_len=1,
                k_shift=0.0, out_dir=train_dir)
        else:
            print(f"  Training dataset exists: {train_dir}")

        # 训练
        run_name = f"scaling_kmax{k_max}_nk{n_k}_hd{hidden_dim}_ep{epochs}"
        run_path = os.path.join(output_root, "gnn_models", run_name)
        if os.path.exists(run_path) and any(
                f.startswith("epoch_") for f in os.listdir(run_path)):
            print(f"  Model already trained: {run_path} — skipping training")
        else:
            print(f"  Training (run_name={run_name})...")
            _, run_path, _ = train(
                dataset_dir      = train_dir,
                graph_type       = 'cross',
                fd_order         = fd_order,
                n_co             = n_co,
                model_type       = 'gnn',
                hidden_dim       = hidden_dim,
                epochs           = epochs,
                save_every       = save_every,
                lr               = lr,
                batch_per_epoch  = batch_per_epoch,
                chain_len        = 1,
                chain_mode       = 'teacher',
                device           = device,
                output_root      = output_root,
                run_name         = run_name,
            )

        # 评估 test loss
        print(f"  Evaluating test loss...")
        test_loss = _eval_test_loss(run_path, test_dir, device)
        print(f"  N_k={n_k}  n_samples={n_samples}  test_loss={test_loss:.4e}")

        # 读取最终 train loss
        lh_path = os.path.join(run_path, "loss_history.json")
        with open(lh_path) as f:
            lh = json.load(f)
        train_loss_final = float(lh["loss"][-1]) if lh["loss"] else float("nan")

        results.append({
            "n_k":              n_k,
            "n_samples":        n_samples,
            "train_loss_final": train_loss_final,
            "test_loss":        test_loss,
            "run_dir":          run_path,
        })

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    x    = [r["n_samples"]        for r in results]
    ytrain = [r["train_loss_final"] for r in results]
    ytest  = [r["test_loss"]        for r in results]

    with plt.rc_context(_SCI_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, ytrain, "o-", label="Train loss (final epoch)", linewidth=1.5)
        ax.plot(x, ytest,  "s--", label="Test loss (out-of-distribution)", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Training set size $N_k^3$")
        ax.set_ylabel(r"MSE loss")
        ax.set_title(f"Data scaling: fd_order={fd_order}, n_co={n_co}, "
                     f"hidden_dim={hidden_dim}, epochs={epochs}")
        # label each point with its N_k value
        for r in results:
            ax.annotate(f"$N_k$={r['n_k']}",
                        xy=(r["n_samples"], r["test_loss"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=11)
        ax.legend()
        fig.tight_layout()
        png_path = out_dir / f"data_scaling_{ts}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    print(f"\n  Plot → {png_path}")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/data_scaling.py",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "n_k_list":     n_k_list,
            "k_max":        k_max,
            "test_n_k":     test_n_k,
            "k_shift_test": k_shift_test,
            "fd_order":     fd_order,
            "n_co":         n_co,
            "hidden_dim":   hidden_dim,
            "epochs":       epochs,
            "device":       device,
        },
        "results": results,
    }
    json_path = out_dir / f"data_scaling_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  Data → {json_path}")
    return output
