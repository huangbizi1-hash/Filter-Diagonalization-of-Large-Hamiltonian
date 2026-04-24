"""
gnn_code/model_compare.py — Compare trained models: loss curves + k=0 fidelity

Loads multiple GNN run directories and:
  1. 读取每个模型的 loss_history.json，画训练损失曲线
  2. 在粗网格 QD 上对 k=0 态作用一次 H，计算瑞利商与保真度（以 FFT 为基准）
  3. 将所有绘图数据写入 JSON，图片保存为 PNG

用法（通过 run_gnn.py）：
    python run_gnn.py --mode compare \\
        --run_dirs gnn_models/c1_teacher_pm1000 gnn_models/c1_teacher_sin1000 \\
        --timing_description "pm1 vs sin wavefunction type"
"""

import datetime
import json
import os
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .physics import d_sparse
from .test_filter import _DEFAULT_CUBE, _DEFAULT_PARAMS, _load_qd_potential
from .gnn_operator import build_gnn_operator
from .benchmark import _h1_k0_test
from .test_filter import _build_fft_qd_operator


def compare_models(
    run_dirs:    list,
    device:      str  = "cpu",
    cube_file:   str  = _DEFAULT_CUBE,
    params_file: str  = _DEFAULT_PARAMS,
    output_root: str  = ".",
    description: str  = "",
):
    """
    Parameters
    ----------
    run_dirs    : list of trained GNN run directory paths
    device      : 'cpu' / 'cuda'
    output_root : directory for output JSON + PNG
    description : written to JSON
    """
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model compare: {len(run_dirs)} model(s)")
    print(f"{'='*60}")

    # ── 读取 loss 历史 ────────────────────────────────────────────────────────
    loss_data = {}   # label → {epochs, loss, final_loss, config}
    d_sparse_val = None

    for run_dir in run_dirs:
        label = os.path.basename(run_dir.rstrip("/\\"))
        cfg_path  = os.path.join(run_dir, "config.json")
        loss_path = os.path.join(run_dir, "loss_history.json")

        with open(cfg_path) as f:
            cfg = json.load(f)

        if d_sparse_val is None:
            d_sparse_val = float(cfg.get("d_sparse", d_sparse))

        loss_hist = {"epochs": [], "loss": []}
        if os.path.exists(loss_path):
            with open(loss_path) as f:
                lh = json.load(f)
            losses = lh.get("loss", [])
            loss_hist["loss"]   = losses
            loss_hist["epochs"] = list(range(1, len(losses) + 1))
        else:
            print(f"  WARNING: {label} has no loss_history.json")

        loss_data[label] = {
            "epochs":      loss_hist["epochs"],
            "loss":        loss_hist["loss"],
            "final_loss":  loss_hist["loss"][-1] if loss_hist["loss"] else None,
            "config": {
                "wf_type":    cfg.get("wf_type",    "?"),
                "chain_len":  cfg.get("chain_len",  1),
                "chain_mode": cfg.get("chain_mode", "?"),
                "hidden_dim": cfg.get("hidden_dim",  64),
                "epochs":     cfg.get("epochs",      0),
            },
        }
        n_ep = len(loss_hist["loss"])
        fl   = loss_hist["loss"][-1] if loss_hist["loss"] else float("nan")
        print(f"  [{label}]  epochs={n_ep}  final_loss={fl:.4e}"
              f"  wf={cfg.get('wf_type','?')}")

    # ── 加载 QD 势能（粗网格） ────────────────────────────────────────────────
    print(f"\n  Loading QD potential (d={d_sparse_val:.4f} Bohr)...")
    pot, N_qd, d_actual = _load_qd_potential(
        d_sparse_val, cube_file=cube_file, params_file=params_file)
    n_grid    = N_qd ** 3
    V_qd_flat = pot.potential.ravel().astype(np.float32)
    V_qd_3d   = pot.potential.reshape(N_qd, N_qd, N_qd)

    # ── 构建算符列表：FFT 排首位 ──────────────────────────────────────────────
    operators = []

    print("\n  Building FFT operator (coarse grid)...")
    try:
        fft_op = _build_fft_qd_operator(V_qd_3d, N_qd, d_actual)
        fft_op.label = "FFT"
        operators.append(("FFT", fft_op))
        print("  FFT operator ready.")
    except Exception as exc:
        print(f"  WARNING: FFT operator failed ({exc}), fidelity will be nan.")

    for run_dir in run_dirs:
        label = os.path.basename(run_dir.rstrip("/\\"))
        print(f"\n  [{label}] loading GNN operator...")
        try:
            gnn_op = build_gnn_operator(run_dir, use_fd=False, device=device,
                                        V_ext=V_qd_flat, N_grid=N_qd)
            operators.append((label, gnn_op))
        except Exception as exc:
            print(f"  WARNING: [{label}] failed ({exc}) → skipped in k=0 test.")

    # ── k=0 保真度测试 ────────────────────────────────────────────────────────
    print("\n  Running k=0 H-apply test...")
    k0_results, _ = _h1_k0_test(operators, n_grid)

    print(f"\n  {'Label':<32} {'E_k0 (Ha)':>10}  {'Fidelity':>9}")
    print(f"  {'─'*32} {'─'*10}  {'─'*9}")
    for label, res in k0_results.items():
        print(f"  {label:<32} {res['energy_k0']:>10.5f}  {res['fidelity_vs_fft']:>9.5f}")

    # ── 绘图 1：训练损失曲线 ──────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for label, ld in loss_data.items():
        if ld["loss"]:
            ax1.semilogy(ld["epochs"], ld["loss"], label=label, linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    fig1.tight_layout()
    loss_png = out_dir / f"compare_loss_{ts}.png"
    fig1.savefig(loss_png, dpi=150)
    plt.close(fig1)
    print(f"\n  Loss curve → {loss_png}")

    # ── 绘图 2：k=0 保真度条形图 ──────────────────────────────────────────────
    gnn_labels = [l for l in k0_results if l != "FFT"]
    fids  = [k0_results[l]["fidelity_vs_fft"] for l in gnn_labels]
    engs  = [k0_results[l]["energy_k0"]       for l in gnn_labels]

    if gnn_labels:
        fig2, axes = plt.subplots(1, 2, figsize=(11, 4))

        # fidelity
        axes[0].bar(range(len(gnn_labels)), fids, color="steelblue")
        axes[0].set_xticks(range(len(gnn_labels)))
        axes[0].set_xticklabels(gnn_labels, rotation=20, ha="right", fontsize=8)
        axes[0].set_ylim(0, 1.05)
        axes[0].axhline(1.0, color="red", linestyle="--", linewidth=1, label="FFT=1")
        axes[0].set_ylabel("Fidelity vs FFT")
        axes[0].set_title("k=0 Fidelity (1 H-apply)")
        axes[0].legend()

        # energy
        fft_e = k0_results.get("FFT", {}).get("energy_k0", None)
        axes[1].bar(range(len(gnn_labels)), engs, color="coral")
        if fft_e is not None:
            axes[1].axhline(fft_e, color="red", linestyle="--", linewidth=1,
                            label=f"FFT={fft_e:.4f}")
        axes[1].set_xticks(range(len(gnn_labels)))
        axes[1].set_xticklabels(gnn_labels, rotation=20, ha="right", fontsize=8)
        axes[1].set_ylabel("Energy ⟨k=0|H|k=0⟩ (Ha)")
        axes[1].set_title("k=0 Rayleigh Quotient")
        axes[1].legend()

        fig2.tight_layout()
        k0_png = out_dir / f"compare_k0_{ts}.png"
        fig2.savefig(k0_png, dpi=150)
        plt.close(fig2)
        print(f"  k=0 fidelity → {k0_png}")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/model_compare.py",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "run_dirs":  [str(r) for r in run_dirs],
            "device":    device,
            "N_qd":      N_qd,
            "d_bohr":    float(d_actual),
            "n_grid":    n_grid,
        },
        "loss_curves":   loss_data,
        "k0_test": {
            label: {
                "energy_k0":       res["energy_k0"],
                "phi_norm":        res["phi_norm"],
                "fidelity_vs_fft": res["fidelity_vs_fft"],
            }
            for label, res in k0_results.items()
        },
    }

    json_path = out_dir / f"compare_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  Data → {json_path}")
    return output
