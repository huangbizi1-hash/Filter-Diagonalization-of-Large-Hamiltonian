"""
gnn_code/stability_test.py — GNN 重复 H-apply 稳定性测试

从 k=0（常数归一化态）出发，重复执行：
    psi_{k+1} = H_GNN(psi_k / ||psi_k||)     (含归一化)

每步记录输出的范数与最大绝对值，检测 NaN/Inf 首次出现步数。

用法（通过 run_gnn.py）：
    python run_gnn.py --mode stability \\
        --run_dirs gnn_models/chain4_auto_bptt_false \\
                   gnn_models/chain50_auto_bptt_false \\
                   gnn_models/chain100_auto_bptt_false \\
        --stability_max_steps 200 \\
        --device cpu
"""

import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch

from .physics import N_sparse, d_sparse
from .gnn_operator import build_gnn_operator
from .test_filter import (
    _DEFAULT_CUBE, _DEFAULT_PARAMS,
    _load_qd_potential,
)


def nan_stability_test(
    run_dirs:    list,
    max_steps:   int  = 200,
    device:      str  = "cpu",
    cube_file:   str  = _DEFAULT_CUBE,
    params_file: str  = _DEFAULT_PARAMS,
    output_root: str  = ".",
    description: str  = "",
):
    """
    对每个模型，从 k=0 态出发重复作用 H（每步归一化），
    最多 max_steps 步，记录何时出现 NaN/Inf。

    每步执行：
        out  = H_GNN(psi_norm)          # psi_norm = psi / ||psi||
        psi  = out                       # 未归一化输出作为下一步输入
                                         # （归一化在下一步开始时做）
    这样既测试归一化输入下的 GNN 稳定性，也测试输出值域。

    Parameters
    ----------
    run_dirs    : GNN run 目录列表
    max_steps   : 最多作用次数（默认 200）
    device      : 'cpu' / 'cuda'
    output_root : 输出 JSON 目录
    description : 写入 JSON 的说明
    """
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev     = torch.device(device if device != "auto" else
                           ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── 读第一个 config 确定网格 ──────────────────────────────────────────
    import json as _json
    with open(os.path.join(run_dirs[0], "config.json")) as f:
        cfg0 = _json.load(f)
    d = float(cfg0.get("d_sparse", d_sparse))

    print(f"\n{'='*60}")
    print(f"  Stability test: {len(run_dirs)} model(s), max_steps={max_steps}")
    print(f"  d_sparse={d:.4f} Bohr  device={dev}")
    print(f"{'='*60}")

    # ── QD 势能 ───────────────────────────────────────────────────────────
    print("\n  Loading QD potential...")
    pot_grid, N_qd, d_actual = _load_qd_potential(
        d, cube_file=cube_file, params_file=params_file)
    n_grid   = N_qd ** 3
    V_qd_flat = pot_grid.potential.ravel().astype(np.float32)

    # ── k=0 初始态 ────────────────────────────────────────────────────────
    psi_k0 = np.ones(n_grid, dtype=np.float32) / np.sqrt(n_grid)

    results = []

    for run_dir in run_dirs:
        short = os.path.basename(run_dir.rstrip("/\\"))
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = _json.load(f)

        # 找最后一个 checkpoint
        ckpts = sorted(
            [fn for fn in os.listdir(run_dir)
             if fn.startswith("epoch_") and fn.endswith(".pt")],
            key=lambda fn: int(fn[len("epoch_"):-len(".pt")]),
        )
        last_ckpt = ckpts[-1] if ckpts else "?"

        model_params = {
            "chain_len":  cfg.get("chain_len",  "?"),
            "chain_mode": cfg.get("chain_mode", "?"),
            "chain_bptt": cfg.get("chain_bptt", False),
            "wf_type":    cfg.get("wf_type",    "?"),
            "hidden_dim": cfg.get("hidden_dim",  64),
            "epochs":     cfg.get("epochs",      0),
            "checkpoint": last_ckpt,
        }

        print(f"\n  [{short}] loading {last_ckpt}...")
        try:
            H_op = build_gnn_operator(run_dir, use_fd=False, device=str(dev),
                                      V_ext=V_qd_flat, N_grid=N_qd)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({
                "label":        short,
                "model_params": model_params,
                "error":        str(exc),
                "nan_step":     None,
                "steps":        [],
            })
            continue

        # ── 重复作用 H ───────────────────────────────────────────────────
        psi = psi_k0.copy()
        step_log = []   # list of {step, norm, max_abs, nan, inf}
        nan_step = None

        print(f"  Applying H up to {max_steps} times...")
        for step in range(1, max_steps + 1):
            # 归一化输入
            nrm_in = float(np.linalg.norm(psi))
            if nrm_in < 1e-30:
                print(f"    step {step}: norm collapsed to 0, stopping.")
                break
            psi_norm = psi / nrm_in

            # H-apply（gnn_operator 内部已含归一化 trick）
            out = H_op.matvec(psi_norm)

            has_nan = bool(np.any(np.isnan(out)))
            has_inf = bool(np.any(np.isinf(out)))
            nrm_out = float(np.linalg.norm(out)) if not (has_nan or has_inf) else float("nan")
            max_abs = float(np.max(np.abs(out[np.isfinite(out)]))) if np.any(np.isfinite(out)) else float("nan")

            step_log.append({
                "step":    step,
                "norm":    nrm_out,
                "max_abs": max_abs,
                "nan":     has_nan,
                "inf":     has_inf,
            })

            if step <= 10 or step % 20 == 0:
                status = "NaN!" if has_nan else ("Inf!" if has_inf else "OK")
                print(f"    step {step:4d}: norm={nrm_out:.4e}  max_abs={max_abs:.4e}  {status}")

            if has_nan or has_inf:
                nan_step = step
                print(f"  *** {'NaN' if has_nan else 'Inf'} first at step {step} ***")
                break

            psi = out  # 未归一化输出作为下一步输入

        if nan_step is None:
            print(f"  Stable for all {max_steps} steps.")

        results.append({
            "label":        short,
            "run_dir":      str(run_dir),
            "model_params": model_params,
            "nan_step":     nan_step,        # None = stable, int = first NaN step
            "stable":       nan_step is None,
            "n_steps_run":  len(step_log),
            "steps":        step_log,        # per-step log
        })

    # ── 汇总打印 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Model':<36} {'NaN at step':>12}  {'Stable?':>8}")
    print(f"  {'─'*36} {'─'*12}  {'─'*8}")
    for r in results:
        ns = r.get("nan_step")
        ns_str = str(ns) if ns is not None else f">{max_steps}"
        stable = "Yes" if r.get("stable") else "No"
        print(f"  {r['label']:<36} {ns_str:>12}  {stable:>8}")
    print(f"{'─'*60}")

    # ── 保存 JSON ─────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":   "gnn_code/stability_test.py",
        "datetime": datetime.datetime.now().isoformat(),
        "config": {
            "max_steps":  max_steps,
            "device":     str(dev),
            "d_sparse":   d,
            "N_qd":       N_qd,
            "n_grid":     n_grid,
            "initial_state": "k=0 (constant normalized)",
        },
        "results": results,
    }

    json_path = out_dir / f"stability_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Results → {json_path}")
    return output
