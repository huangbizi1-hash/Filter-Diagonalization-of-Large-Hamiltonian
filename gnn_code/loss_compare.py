"""
gnn_code/loss_compare.py — 多模型训练损失曲线对比（平均 chain loss）

平均 chain loss = mean(step_losses[epoch]) over chain steps，
使不同 chain_len 的模型在同一量纲下可比。
若 loss_history.json 中没有 step_losses，则退化为总 loss。

用法（通过 run_gnn.py）：
    python run_gnn.py --mode loss_compare \\
        --run_dirs gnn_models/chain4_auto_bptt_true  \\
                   gnn_models/chain4_auto_bptt_false \\
                   gnn_models/chain50_auto_bptt_false \\
        --timing_description "chain length & BPTT comparison"
"""

import datetime
import json
import os
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── SCI 绘图样式（照用户提供的设置）──────────────────────────────────────────
_SCI_STYLE = {
    'font.size':           16,
    'axes.labelsize':      16,
    'axes.titlesize':      18,
    'axes.titleweight':    'normal',
    'axes.labelweight':    'normal',
    'legend.fontsize':     14,
    'legend.loc':          'best',
    'font.weight':         'normal',
    'mathtext.fontset':    'cm',
    'mathtext.default':    'regular',
    'axes.grid':           True,
    'grid.linestyle':      '--',
    'grid.alpha':          0.3,
}


def _load_loss_curve(run_dir: str):
    """
    读取 run_dir/loss_history.json，返回
      epochs      : np.ndarray  epoch 编号
      avg_chain   : np.ndarray  平均 chain loss（mean over step_losses；
                                fallback 到 loss 若 step_losses 不存在）
      config      : dict
    """
    loss_path = os.path.join(run_dir, "loss_history.json")
    cfg_path  = os.path.join(run_dir, "config.json")

    with open(loss_path) as f:
        lh = json.load(f)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # epoch 序列
    if "epoch" in lh and lh["epoch"]:
        epochs = np.array(lh["epoch"], dtype=float)
    else:
        epochs = np.arange(1, len(lh["loss"]) + 1, dtype=float)

    # 平均 chain loss
    step_losses = lh.get("step_losses", [])
    if step_losses and len(step_losses) == len(epochs):
        avg_chain = np.array([float(np.mean(sl)) for sl in step_losses])
    else:
        avg_chain = np.array(lh["loss"], dtype=float)

    return epochs, avg_chain, cfg


def _make_label(run_dir: str, cfg: dict) -> str:
    """从目录名 + config 构建图例标签。"""
    base = os.path.basename(run_dir.rstrip("/\\"))
    return base


def compare_loss_curves(
    run_dirs:    list,
    output_root: str  = ".",
    description: str  = "",
):
    """
    对比多个 GNN run 的平均 chain loss 曲线，保存 PNG + JSON。

    Parameters
    ----------
    run_dirs    : GNN run 目录列表
    output_root : PNG 和 JSON 的输出目录
    description : 写入 JSON description 字段
    """
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Loss compare: {len(run_dirs)} model(s)")
    print(f"{'='*60}")

    curves = {}
    for run_dir in run_dirs:
        label = _make_label(run_dir, {})
        try:
            epochs, avg_chain, cfg = _load_loss_curve(run_dir)
            label = _make_label(run_dir, cfg)
            curves[label] = {
                "epochs":     epochs.tolist(),
                "avg_chain_loss": avg_chain.tolist(),
                "final_loss": float(avg_chain[-1]) if len(avg_chain) else None,
                "config": {
                    "chain_len":  cfg.get("chain_len",  1),
                    "chain_mode": cfg.get("chain_mode", "?"),
                    "chain_bptt": cfg.get("chain_bptt", False),
                    "wf_type":    cfg.get("wf_type",    "?"),
                    "hidden_dim": cfg.get("hidden_dim",  64),
                    "epochs":     cfg.get("epochs",      0),
                },
            }
            print(f"  [{label}]  n_epochs={len(epochs)}"
                  f"  final_avg_chain_loss={avg_chain[-1]:.4e}"
                  f"  chain={cfg.get('chain_len',1)}"
                  f"  mode={cfg.get('chain_mode','?')}"
                  f"  bptt={cfg.get('chain_bptt', False)}")
        except Exception as exc:
            print(f"  WARNING: [{label}] failed ({exc}) → skipped")

    if not curves:
        raise RuntimeError("没有可用模型，退出。")

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    with plt.rc_context(_SCI_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        for label, c in curves.items():
            cfg  = c["config"]
            # 标签：显示 chain/mode/bptt 关键参数
            lbl  = (f"{label}"
                    f"  (L={cfg['chain_len']}, {cfg['chain_mode']}"
                    f", bptt={cfg['chain_bptt']})")
            ax.plot(
                c["epochs"],
                c["avg_chain_loss"],
                label=lbl,
                linestyle='-',
                linewidth=1.5,
            )

        ax.set_yscale("log")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Avg chain loss")
        ax.set_title("Training loss comparison (mean over chain steps)")
        ax.legend()
        fig.tight_layout()

        png_path = out_dir / f"loss_compare_{ts}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    print(f"\n  Plot → {png_path}")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/loss_compare.py",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "run_dirs": [str(r) for r in run_dirs],
        },
        "curves": curves,
    }
    json_path = out_dir / f"loss_compare_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  Data → {json_path}")
    return output
