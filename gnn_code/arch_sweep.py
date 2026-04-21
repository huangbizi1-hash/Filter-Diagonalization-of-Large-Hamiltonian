"""
gnn_code/arch_sweep.py — Experiment 2: n_co / model_type sweep, k=0 fidelity

固定 N_k=10, k_max=5, fd_order=6, hidden_dim=16, epochs=5000, chain=1, teacher。
分别用 n_co=3,4,5（cross+gnn）及 so3（n_co=3,4,5）进行训练，
在训练势（V_sparse）上测试 k=0 态保真度，以 FFT（动能截断=30 Ha）为基准。

用法（通过 run_gnn.py）：
    python run_gnn.py --mode arch_sweep \\
        --arch_n_co_list 3 4 5 \\
        --arch_use_so3 \\
        --arch_fd_order 6 \\
        --arch_n_k 10 \\
        --arch_epochs 5000 \\
        --arch_hidden_dim 16
"""

import datetime
import json
import os
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator

_SCI_STYLE = {
    'font.size': 16, 'axes.labelsize': 16, 'axes.titlesize': 18,
    'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
    'legend.fontsize': 14, 'legend.loc': 'best', 'font.weight': 'normal',
    'mathtext.fontset': 'cm', 'mathtext.default': 'regular',
    'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.3,
}


# ── FFT reference on training grid ───────────────────────────────────────────

def _build_fft_train_op(kinetic_cutoff: float = 30.0) -> LinearOperator:
    """FFT Hamiltonian on training grid (V_sparse, N_sparse, d_sparse) with cutoff."""
    from .physics import V_sparse, N_sparse, d_sparse

    N   = N_sparse
    d   = d_sparse
    V_f = V_sparse.ravel().astype(np.float64)

    kx  = np.fft.fftfreq(N, d=d) * 2 * np.pi
    Kx, Ky, Kz = np.meshgrid(kx, kx, kx, indexing='ij')
    T_k = np.minimum((Kx**2 + Ky**2 + Kz**2) / 2.0, kinetic_cutoff)

    def _matvec(psi):
        psi_3d = psi.reshape(N, N, N).astype(np.float64)
        T_psi  = np.fft.ifftn(T_k * np.fft.fftn(psi_3d)).real
        V_psi  = V_f.reshape(N, N, N) * psi_3d
        return (T_psi + V_psi).ravel()

    n_grid = N ** 3
    op = LinearOperator((n_grid, n_grid), matvec=_matvec, dtype=np.float64)
    op.label = "FFT"
    return op


# ── k=0 fidelity on training grid ────────────────────────────────────────────

def _k0_fidelity(run_dir: str, fft_phi_n: np.ndarray,
                 device_str: str = 'cpu') -> dict:
    """
    Apply GNN once to k=0 state, compare to pre-computed fft_phi_n.
    Returns dict with energy_k0, phi_norm, fidelity_vs_fft.
    """
    from .gnn_operator import build_gnn_operator
    from .physics import N_sparse

    n_grid = N_sparse ** 3
    psi_k0 = np.ones(n_grid, dtype=np.float64) / np.sqrt(n_grid)

    gnn_op = build_gnn_operator(run_dir, use_fd=False, device=device_str)
    phi    = gnn_op.matvec(psi_k0)
    energy = float(np.dot(psi_k0, phi))
    nrm    = float(np.linalg.norm(phi))
    phi_n  = phi / nrm if nrm > 1e-15 else phi.copy()
    fid    = float(abs(np.dot(phi_n, fft_phi_n))) if fft_phi_n is not None else float('nan')

    return {"energy_k0": energy, "phi_norm": nrm, "fidelity_vs_fft": fid}


# ── main pipeline ─────────────────────────────────────────────────────────────

def arch_sweep_experiment(
    n_co_list:      list  = None,
    use_so3:        bool  = True,
    fd_order:       int   = 6,
    n_k:            int   = 10,
    k_max:          int   = 5,
    hidden_dim:     int   = 16,
    radial_hidden_dim: int = 32,
    epochs:         int   = 5000,
    save_every:     int   = 500,
    lr:             float = 1e-3,
    batch_per_epoch: int  = 10,
    kinetic_cutoff: float = 30.0,
    device:         str   = 'cpu',
    output_root:    str   = '.',
    description:    str   = '',
):
    """
    训练 cross-GNN（n_co=3,4,5）及可选 SO3（相同 n_co 列表），
    在训练势上测试 k=0 保真度，绘图保存。
    """
    if n_co_list is None:
        n_co_list = [3, 4, 5]

    from .train   import train
    from .dataset import generate_k_grid_dataset

    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_root = out_dir / "gnn_datasets" / f"arch_sweep_nk{n_k}_kmax{k_max}"
    ds_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Arch sweep: n_co_list={n_co_list}  use_so3={use_so3}")
    print(f"  fd_order={fd_order}  n_k={n_k}  hidden_dim={hidden_dim}  epochs={epochs}")
    print(f"{'='*64}")

    # 共用训练集
    train_dir = str(ds_root / f"train_nk{n_k}")
    if not os.path.exists(os.path.join(train_dir, "metadata.json")):
        print(f"\n  Generating training dataset (N_k={n_k})...")
        generate_k_grid_dataset(k_max=k_max, n_k=n_k, chain_len=1,
                                k_shift=0.0, out_dir=train_dir)
    else:
        print(f"\n  Training dataset exists: {train_dir}")

    # 列出所有配置：(n_co, model_type)
    configs = []
    for n_co in n_co_list:
        configs.append({"n_co": n_co, "model_type": "gnn"})
    if use_so3:
        for n_co in n_co_list:
            configs.append({"n_co": n_co, "model_type": "so3"})

    # ── 训练所有配置 ───────────────────────────────────────────────────────────
    run_dirs_map = {}   # label → run_dir
    for cfg in configs:
        n_co       = cfg["n_co"]
        model_type = cfg["model_type"]
        label      = f"{model_type}_nco{n_co}"
        run_name   = (f"arch_fd{fd_order}_nco{n_co}_{model_type}"
                      f"_hd{hidden_dim}_ep{epochs}")
        run_path   = os.path.join(output_root, "gnn_models", run_name)
        print(f"\n  ── {label} ──")

        if os.path.exists(run_path) and any(
                f.startswith("epoch_") for f in os.listdir(run_path)):
            print(f"  Already trained: {run_path} — skipping")
        else:
            print(f"  Training {label}...")
            _, run_path, _ = train(
                dataset_dir       = train_dir,
                graph_type        = 'cross',
                fd_order          = fd_order,
                n_co              = n_co,
                model_type        = model_type,
                hidden_dim        = hidden_dim,
                radial_hidden_dim = radial_hidden_dim,
                epochs            = epochs,
                save_every        = save_every,
                lr                = lr,
                batch_per_epoch   = batch_per_epoch,
                chain_len         = 1,
                chain_mode        = 'teacher',
                kinetic_cutoff    = kinetic_cutoff,
                device            = device,
                output_root       = output_root,
                run_name          = run_name,
            )
        run_dirs_map[label] = run_path
        cfg["run_dir"] = run_path

    # ── FFT 参考（k=0 保真度）──────────────────────────────────────────────────
    print("\n  Computing FFT k=0 reference...")
    from .physics import N_sparse
    n_grid    = N_sparse ** 3
    psi_k0    = np.ones(n_grid, dtype=np.float64) / np.sqrt(n_grid)
    fft_op    = _build_fft_train_op(kinetic_cutoff)
    phi_fft   = fft_op.matvec(psi_k0)
    nrm_fft   = float(np.linalg.norm(phi_fft))
    phi_fft_n = phi_fft / nrm_fft if nrm_fft > 1e-15 else phi_fft.copy()
    E_fft     = float(np.dot(psi_k0, phi_fft))
    print(f"  FFT: E_k0={E_fft:.5f} Ha")

    # ── k=0 保真度评估 ─────────────────────────────────────────────────────────
    print("\n  Evaluating k=0 fidelity for each model...")
    for cfg in configs:
        label      = f"{cfg['model_type']}_nco{cfg['n_co']}"
        run_path   = cfg["run_dir"]
        try:
            res = _k0_fidelity(run_path, phi_fft_n, device)
            cfg["fidelity_vs_fft"]  = res["fidelity_vs_fft"]
            cfg["energy_k0"]        = res["energy_k0"]
            cfg["energy_err"]       = res["energy_k0"] - E_fft
            print(f"  {label:<20}  fid={res['fidelity_vs_fft']:.5f}"
                  f"  E_k0={res['energy_k0']:.5f}  dE={cfg['energy_err']:+.4e}")
        except Exception as exc:
            print(f"  {label}: FAILED ({exc})")
            cfg["fidelity_vs_fft"] = float('nan')
            cfg["energy_k0"]       = float('nan')
            cfg["energy_err"]      = float('nan')

    # ── 绘图：保真度 & 能量偏差 ────────────────────────────────────────────────
    gnn_cfgs = [c for c in configs if c["model_type"] == "gnn"]
    so3_cfgs = [c for c in configs if c["model_type"] == "so3"]

    with plt.rc_context(_SCI_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax_idx, (ax, metric, ylabel) in enumerate([
            (axes[0], "fidelity_vs_fft",  r"Fidelity vs FFT (k=0)"),
            (axes[1], "energy_err",        r"Energy error $\Delta E$ (Ha)"),
        ]):
            if gnn_cfgs:
                x_gnn = [c["n_co"] for c in gnn_cfgs]
                y_gnn = [c[metric]  for c in gnn_cfgs]
                ax.plot(x_gnn, y_gnn, "o-", label="GNN-cross", linewidth=1.5)
            if so3_cfgs:
                x_so3 = [c["n_co"] for c in so3_cfgs]
                y_so3 = [c[metric]  for c in so3_cfgs]
                ax.plot(x_so3, y_so3, "s--", label="SO3-cross", linewidth=1.5)

            if ax_idx == 0:
                ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':',
                           label='FFT=1')
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(0.0, color='gray', linewidth=0.8, linestyle=':')

            ax.set_xlabel(r"$n_{co}$ (correction cube size)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(n_co_list)
            ax.legend()

        fig.suptitle(f"Arch sweep: fd_order={fd_order}, N_k={n_k}, "
                     f"hidden_dim={hidden_dim}, epochs={epochs}", fontsize=14)
        fig.tight_layout()
        png_path = out_dir / f"arch_sweep_{ts}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    print(f"\n  Plot → {png_path}")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/arch_sweep.py",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "n_co_list":     n_co_list, "use_so3": use_so3,
            "fd_order":      fd_order,  "n_k": n_k, "k_max": k_max,
            "hidden_dim":    hidden_dim, "epochs": epochs,
            "kinetic_cutoff": kinetic_cutoff,
            "fft_E_k0":      E_fft,
        },
        "results": configs,
    }
    json_path = out_dir / f"arch_sweep_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  Data → {json_path}")
    return output
