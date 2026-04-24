"""
gnn_code/arch_sweep.py — Experiment 2: n_co / model_type sweep, k=0 and k=0.1 fidelity

固定 N_k=10, k_max=5, fd_order=6, hidden_dim=16, epochs=5000, chain=1, teacher。
分别用 n_co=3,4,5（cross+gnn）及 so3（n_co=3,4,5）进行训练，
测试：
  · k=0 uniform 态   → H|ψ⟩ = V·ψ（动能为零，所有方法严格相同，仅作 sanity check）
  · k=0.1 sine 态    → psi = sin(0.1*2π/L · x)，动能非零，模型之间有差异

以 FFT（动能截断=30 Ha）为基准，并同时计算纯 FD 基准值作对比。

用法（通过 run_gnn.py）：
    python run_gnn.py --mode arch_sweep \\
        --arch_n_co_list 3 4 5 \\
        --arch_use_so3 \\
        --arch_fd_order 6 \\
        --arch_n_k 10 \\
        --arch_epochs 5000 \\
        --arch_hidden_dim 16

注意：k=0 uniform 态的 SO3/GNN/FD 保真度严格等于 1（H|uniform⟩ = V·uniform，
动能项为零，三种算子完全相同），因此该测试只用于 sanity check，k=0.1 态才是真正
区分各方法精度的测试。
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


# ── test state builders ───────────────────────────────────────────────────────

def _make_k0_state(N: int) -> np.ndarray:
    """Uniform k=0 state: psi = 1/sqrt(N³)."""
    n_grid = N ** 3
    return np.ones(n_grid, dtype=np.float64) / np.sqrt(n_grid)


def _make_sine_state(k_frac: float, N: int, d: float, L: float) -> np.ndarray:
    """Sine-wave state: psi = sin(k_frac * 2π/L · x), normalised.

    k_frac is in units of the fundamental wave vector dk = 2π/L.
    E.g. k_frac=0.1 gives kx = 0.1*(2π/L), T = ½*(0.1*2π/L)².
    """
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    kx = k_frac * 2 * np.pi / L
    psi = np.sin(kx * X).ravel().astype(np.float64)
    nrm = np.linalg.norm(psi)
    return psi / nrm if nrm > 1e-15 else psi


# ── single-operator fidelity evaluation ──────────────────────────────────────

def _eval_fidelity(op: LinearOperator,
                   psi_test: np.ndarray,
                   phi_ref_n: np.ndarray) -> dict:
    """
    Apply op to psi_test, compare normalised result to pre-computed phi_ref_n.
    Returns energy (Rayleigh quotient), phi_norm, fidelity_vs_ref.
    """
    phi    = op.matvec(psi_test)
    energy = float(np.dot(psi_test, phi))
    nrm    = float(np.linalg.norm(phi))
    phi_n  = phi / nrm if nrm > 1e-15 else phi.copy()
    fid    = float(abs(np.dot(phi_n, phi_ref_n))) if phi_ref_n is not None else float('nan')
    return {"energy": energy, "phi_norm": nrm, "fidelity_vs_fft": fid}


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
    在训练势上测试 k=0/k=0.1 保真度，绘图保存。

    k=0 (uniform) 测试：所有方法严格等于 FFT（动能为零），仅用作 sanity check。
    k=0.1 (sine) 测试：kx = 0.1*2π/L，动能非零，区分各方法精度。
    """
    if n_co_list is None:
        n_co_list = [3, 4, 5]

    from .train   import train
    from .dataset import generate_k_grid_dataset
    from .gnn_operator import build_gnn_operator
    from .physics import N_sparse, d_sparse, L

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
    print(f"  NOTE: k=0 (uniform) test is trivial — all methods agree by design.")
    print(f"        k=0.1 (sine) test is the meaningful discriminator.")
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

    # ── 构建测试态 & FFT 参考 ──────────────────────────────────────────────────
    print("\n  Building test states and FFT references...")
    psi_k0 = _make_k0_state(N_sparse)
    psi_k1 = _make_sine_state(0.1, N_sparse, d_sparse, L)

    fft_op   = _build_fft_train_op(kinetic_cutoff)
    phi_fft_k0   = fft_op.matvec(psi_k0)
    nrm_fft_k0   = float(np.linalg.norm(phi_fft_k0))
    phi_fft_k0_n = phi_fft_k0 / nrm_fft_k0 if nrm_fft_k0 > 1e-15 else phi_fft_k0.copy()
    E_fft_k0     = float(np.dot(psi_k0, phi_fft_k0))

    phi_fft_k1   = fft_op.matvec(psi_k1)
    nrm_fft_k1   = float(np.linalg.norm(phi_fft_k1))
    phi_fft_k1_n = phi_fft_k1 / nrm_fft_k1 if nrm_fft_k1 > 1e-15 else phi_fft_k1.copy()
    E_fft_k1     = float(np.dot(psi_k1, phi_fft_k1))
    print(f"  FFT: E_k0={E_fft_k0:.5f} Ha  E_k0.1={E_fft_k1:.5f} Ha")

    # ── 纯 FD 基准（不依赖训练，直接用第一个模型的图，use_fd=True）─────────────
    print("\n  Computing FD baseline (use_fd=True)...")
    first_run = configs[0]["run_dir"]
    fd_op = build_gnn_operator(first_run, use_fd=True, device=device)
    fd_res_k0 = _eval_fidelity(fd_op, psi_k0, phi_fft_k0_n)
    fd_res_k1 = _eval_fidelity(fd_op, psi_k1, phi_fft_k1_n)
    print(f"  FD k=0: fid={fd_res_k0['fidelity_vs_fft']:.5f}  E={fd_res_k0['energy']:.5f}")
    print(f"  FD k=0.1: fid={fd_res_k1['fidelity_vs_fft']:.5f}  E={fd_res_k1['energy']:.5f}"
          f"  dE={fd_res_k1['energy'] - E_fft_k1:+.4e}")

    # ── GNN / SO3 评估（k=0 & k=0.1）─────────────────────────────────────────
    print("\n  Evaluating GNN/SO3 fidelity (k=0 and k=0.1)...")
    for cfg in configs:
        label    = f"{cfg['model_type']}_nco{cfg['n_co']}"
        run_path = cfg["run_dir"]
        try:
            gnn_op = build_gnn_operator(run_path, use_fd=False, device=device)

            r0 = _eval_fidelity(gnn_op, psi_k0, phi_fft_k0_n)
            cfg["k0_fidelity"]    = r0["fidelity_vs_fft"]
            cfg["k0_energy"]      = r0["energy"]
            cfg["k0_energy_err"]  = r0["energy"] - E_fft_k0

            r1 = _eval_fidelity(gnn_op, psi_k1, phi_fft_k1_n)
            cfg["k1_fidelity"]    = r1["fidelity_vs_fft"]
            cfg["k1_energy"]      = r1["energy"]
            cfg["k1_energy_err"]  = r1["energy"] - E_fft_k1

            print(f"  {label:<20}"
                  f"  k0: fid={r0['fidelity_vs_fft']:.5f}  E={r0['energy']:.5f}"
                  f"  |  k0.1: fid={r1['fidelity_vs_fft']:.5f}  E={r1['energy']:.5f}"
                  f"  dE={cfg['k1_energy_err']:+.4e}")
        except Exception as exc:
            print(f"  {label}: FAILED ({exc})")
            for key in ("k0_fidelity","k0_energy","k0_energy_err",
                        "k1_fidelity","k1_energy","k1_energy_err"):
                cfg[key] = float('nan')

    # ── 绘图：2 行（k=0, k=0.1）× 2 列（保真度, 能量偏差）───────────────────────
    gnn_cfgs = [c for c in configs if c["model_type"] == "gnn"]
    so3_cfgs = [c for c in configs if c["model_type"] == "so3"]

    with plt.rc_context(_SCI_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        row_specs = [
            # (row_idx, state_label, fid_key, energy_key, E_fft, fd_fid, fd_E)
            (0, "k=0 (uniform, sanity check)",
             "k0_fidelity", "k0_energy_err", E_fft_k0,
             fd_res_k0["fidelity_vs_fft"], fd_res_k0["energy"] - E_fft_k0),
            (1, r"$k_x=0.1\times 2\pi/L$ (sine, kinetic test)",
             "k1_fidelity", "k1_energy_err", E_fft_k1,
             fd_res_k1["fidelity_vs_fft"], fd_res_k1["energy"] - E_fft_k1),
        ]

        for row_idx, state_label, fid_key, err_key, E_fft_val, fd_fid, fd_err in row_specs:
            ax_fid = axes[row_idx, 0]
            ax_err = axes[row_idx, 1]

            if gnn_cfgs:
                x_gnn = [c["n_co"] for c in gnn_cfgs]
                ax_fid.plot(x_gnn, [c[fid_key] for c in gnn_cfgs],
                            "o-", label="GNN-cross", linewidth=1.5)
                ax_err.plot(x_gnn, [c[err_key] for c in gnn_cfgs],
                            "o-", label="GNN-cross", linewidth=1.5)
            if so3_cfgs:
                x_so3 = [c["n_co"] for c in so3_cfgs]
                ax_fid.plot(x_so3, [c[fid_key] for c in so3_cfgs],
                            "s--", label="SO3-cross", linewidth=1.5)
                ax_err.plot(x_so3, [c[err_key] for c in so3_cfgs],
                            "s--", label="SO3-cross", linewidth=1.5)

            # FD baseline
            ax_fid.axhline(fd_fid, color='C2', linewidth=1.2,
                           linestyle=':', label=f"FD (order={fd_order})")
            ax_err.axhline(fd_err, color='C2', linewidth=1.2,
                           linestyle=':', label=f"FD (order={fd_order})")

            ax_fid.axhline(1.0, color='gray', linewidth=0.8, linestyle='--',
                           label='FFT=1')
            ax_err.axhline(0.0, color='gray', linewidth=0.8, linestyle='--')

            ax_fid.set_title(f"{state_label}  — Fidelity vs FFT")
            ax_err.set_title(f"{state_label}  — Energy error (Ha)")
            ax_fid.set_ylim(bottom=max(0, min(
                [c[fid_key] for c in configs if not np.isnan(c[fid_key])] + [fd_fid, 1.0]
            ) - 0.05))
            ax_fid.set_ylabel(r"Fidelity $|\langle\phi|\phi_\mathrm{FFT}\rangle|$")
            ax_err.set_ylabel(r"$\Delta E$ (Ha)")
            for ax in (ax_fid, ax_err):
                ax.set_xlabel(r"$n_{co}$ (correction cube size)")
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
            "fft_E_k0":      E_fft_k0,
            "fft_E_k1":      E_fft_k1,
        },
        "fd_baseline": {
            "k0_fidelity":   fd_res_k0["fidelity_vs_fft"],
            "k0_energy":     fd_res_k0["energy"],
            "k0_energy_err": fd_res_k0["energy"] - E_fft_k0,
            "k1_fidelity":   fd_res_k1["fidelity_vs_fft"],
            "k1_energy":     fd_res_k1["energy"],
            "k1_energy_err": fd_res_k1["energy"] - E_fft_k1,
        },
        "results": configs,
    }
    json_path = out_dir / f"arch_sweep_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  Data → {json_path}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# arch_loss_compare — 训练损失 vs 验证损失对比
# ─────────────────────────────────────────────────────────────────────────────

def arch_loss_compare(
    n_co_list:     list  = None,
    use_so3:       bool  = True,
    fd_order:      int   = 6,
    hidden_dim:    int   = 16,
    epochs:        int   = 5000,
    last_n_epochs: int   = 100,
    val_dir:       str   = None,
    k_max:         int   = 5,
    test_n_k:      int   = 10,
    device:        str   = 'cpu',
    output_root:   str   = '.',
    description:   str   = '',
):
    """
    对 arch_sweep 训练的 6 个模型（GNN/SO3 × n_co=3,4,5）比较：
      · 训练损失：loss_history.json 中最后 last_n_epochs 个 epoch 的均值
      · 验证损失：在 val_dir 数据集上评估 MSE（与训练集不重叠的 k-grid 测试集）

    val_dir 默认为 output_root/gnn_datasets/scaling_kmax{k_max}/test_nk{test_n_k}_halfshift
    （即 data_scaling 实验生成的测试集）。

    输出：1×2 子图（训练损失 / 验证损失 vs n_co）+ JSON。
    """
    if n_co_list is None:
        n_co_list = [3, 4, 5]

    from .gnn_operator import build_gnn_operator
    from .dataset     import WavefunctionDataset

    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 验证集路径 ─────────────────────────────────────────────────────────────
    if val_dir is None:
        val_dir = str(out_dir / "gnn_datasets"
                      / f"scaling_kmax{k_max}"
                      / f"test_nk{test_n_k}_halfshift")

    if not os.path.exists(os.path.join(val_dir, "metadata.json")):
        raise FileNotFoundError(
            f"验证集不存在: {val_dir}\n"
            "请先运行 --mode data_scaling 生成验证集，或用 --arch_val_dir 指定路径。")

    print(f"\n{'='*64}")
    print(f"  Arch loss compare: n_co_list={n_co_list}  use_so3={use_so3}")
    print(f"  fd_order={fd_order}  hidden_dim={hidden_dim}  epochs={epochs}")
    print(f"  last_n_epochs={last_n_epochs}  device={device}")
    print(f"  val_dir={val_dir}")
    print(f"{'='*64}")

    # ── 列出所有配置 ───────────────────────────────────────────────────────────
    configs = []
    for n_co in n_co_list:
        configs.append({"n_co": n_co, "model_type": "gnn"})
    if use_so3:
        for n_co in n_co_list:
            configs.append({"n_co": n_co, "model_type": "so3"})

    # ── 加载验证集（一次性，共用）────────────────────────────────────────────────
    print("\n  Loading validation dataset...")
    val_ds = WavefunctionDataset(val_dir, in_memory=True)
    print(f"  {len(val_ds)} validation samples")

    # ── 逐模型读取损失 & 评估验证损失 ─────────────────────────────────────────
    for cfg in configs:
        n_co       = cfg["n_co"]
        model_type = cfg["model_type"]
        label      = f"{model_type.upper()} n_co={n_co}"
        run_name   = (f"arch_fd{fd_order}_nco{n_co}_{model_type}"
                      f"_hd{hidden_dim}_ep{epochs}")
        run_path   = os.path.join(output_root, "gnn_models", run_name)
        cfg["run_dir"] = run_path
        cfg["label"]   = label
        print(f"\n  ── {label} ──")

        # ── 训练损失（最后 last_n_epochs 个 epoch 均值）─────────────────────────
        lh_path = os.path.join(run_path, "loss_history.json")
        if not os.path.exists(lh_path):
            print(f"  SKIP — loss_history.json not found: {lh_path}")
            cfg["train_loss"] = float("nan")
            cfg["val_loss"]   = float("nan")
            continue
        with open(lh_path) as f:
            lh = json.load(f)
        loss_arr = lh.get("loss", [])
        tail = loss_arr[-last_n_epochs:] if len(loss_arr) >= last_n_epochs else loss_arr
        train_loss = float(np.mean(tail)) if tail else float("nan")
        cfg["train_loss"] = train_loss
        print(f"  train_loss (last {len(tail)} ep avg) = {train_loss:.4e}")

        # ── 验证损失（MSE on val_ds）─────────────────────────────────────────────
        try:
            gnn_op = build_gnn_operator(run_path, use_fd=False, device=device)
            total_mse = 0.0
            for sample in val_ds:
                psi_t, tgt_t = sample[0]          # chain_len=1
                psi_np  = psi_t.numpy().flatten().astype(np.float64)
                tgt_np  = tgt_t.numpy().flatten().astype(np.float64)
                pred_np = gnn_op.matvec(psi_np)
                total_mse += float(np.mean((pred_np - tgt_np) ** 2))
            val_loss = total_mse / len(val_ds)
            cfg["val_loss"] = val_loss
            print(f"  val_loss (MSE, {len(val_ds)} samples) = {val_loss:.4e}")
        except Exception as exc:
            print(f"  val_loss FAILED: {exc}")
            cfg["val_loss"] = float("nan")

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    gnn_cfgs = [c for c in configs if c["model_type"] == "gnn"]
    so3_cfgs = [c for c in configs if c["model_type"] == "so3"]

    with plt.rc_context(_SCI_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for ax, metric, ylabel in [
            (axes[0], "train_loss", f"Train loss (last {last_n_epochs} ep avg)"),
            (axes[1], "val_loss",   "Val loss (MSE, out-of-distribution k-grid)"),
        ]:
            if gnn_cfgs:
                ax.plot([c["n_co"] for c in gnn_cfgs],
                        [c[metric] for c in gnn_cfgs],
                        "o-", label="GNN-cross", linewidth=1.5)
            if so3_cfgs:
                ax.plot([c["n_co"] for c in so3_cfgs],
                        [c[metric] for c in so3_cfgs],
                        "s--", label="SO3-cross", linewidth=1.5)

            ax.set_yscale("log")
            ax.set_xlabel(r"$n_{co}$")
            ax.set_ylabel(ylabel)
            ax.set_xticks(n_co_list)
            ax.legend()

        fig.suptitle(
            f"fd_order={fd_order}, hidden_dim={hidden_dim}, epochs={epochs}  "
            f"| val: test_nk{test_n_k}_halfshift",
            fontsize=14)
        fig.tight_layout()
        png_path = out_dir / f"arch_loss_{ts}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    print(f"\n  Plot → {png_path}")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/arch_sweep.py::arch_loss_compare",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "n_co_list":     n_co_list,
            "use_so3":       use_so3,
            "fd_order":      fd_order,
            "hidden_dim":    hidden_dim,
            "epochs":        epochs,
            "last_n_epochs": last_n_epochs,
            "val_dir":       val_dir,
            "n_val_samples": len(val_ds),
            "device":        device,
        },
        "results": configs,
    }
    json_path = out_dir / f"arch_loss_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  Data → {json_path}")
    return output
