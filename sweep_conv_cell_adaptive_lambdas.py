#!/usr/bin/env python3
"""Sweep conv-cell adaptive lambda_grad/lambda_lap for compare_fft_rbf_filter_qd.py.

默认会按照用户给定的大命令参数运行，并将：
1) 扫描参数网格（5x5，logspace 0.1 -> 10.0）；
2) 每次运行的 stdout/stderr 与结果 JSON 路径汇总到 manifest；
3) 首次运行保存节点，后续复用同一节点文件（--load-nodes）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import math

SAVE_NODES_PATTERN = re.compile(r"节点数据文件已保存:\s*(.+)")
RESULT_JSON_PATTERN = re.compile(r"结果已保存:\s*(.+)")


def run_one(cmd: list[str], dry_run: bool) -> tuple[str | None, str | None]:
    print("\n>>>", " ".join(cmd), flush=True)
    if dry_run:
        return None, None

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")

    nodes_match = SAVE_NODES_PATTERN.search(proc.stdout)
    result_match = RESULT_JSON_PATTERN.search(proc.stdout)
    nodes_path = nodes_match.group(1).strip() if nodes_match else None
    result_path = result_match.group(1).strip() if result_match else None
    return nodes_path, result_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep --conv-cell-adaptive-lambda-grad and "
            "--conv-cell-adaptive-lambda-lap on log scale"
        )
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--compare-script", type=str, default="compare_fft_rbf_filter_qd.py")
    p.add_argument("--manifest-dir", type=str, default="filter_compare_results")
    p.add_argument("--save-nodes", type=str, default="rbf_nodes/")
    p.add_argument("--load-nodes", type=str, default="", help="若给定则直接复用节点文件")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--lambda-min", type=float, default=0.1)
    p.add_argument("--lambda-max", type=float, default=10.0)
    p.add_argument("--lambda-n", type=int, default=5)

    # 与用户命令保持一致的默认参数
    p.add_argument("--system", type=str, default="qd")
    p.add_argument("--qd-radius", type=int, default=17)
    p.add_argument("--el", type=float, default=-0.18)
    p.add_argument("--nc", type=int, default=500)
    p.add_argument("--n-random", type=int, default=1)
    p.add_argument("--dE", type=float, default=300.0)
    p.add_argument("--V-min", type=float, dest="V_min", default=-5.0)
    p.add_argument("--fft-kinetic-cut", type=float, default=20.0)
    p.add_argument("--rbf-node-method", type=str, default="conv_cell")
    p.add_argument("--conv-cell-template-mode", type=str, default="fcc_refined")
    p.add_argument("--conv-cell-domain-shape", type=str, default="cube")
    p.add_argument("--rbf-stencil-radius", type=float, default=1.60)
    p.add_argument("--conv-cell-fcc-scale-factor", type=int, default=8)
    p.add_argument("--conv-cell-fcc-origin-frac", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--conv-cell-fcc-atom-refine-factor", type=int, default=16)
    p.add_argument("--conv-cell-fcc-atom-radius-frac", type=float, default=0.00)
    p.add_argument("--conv-cell-a", type=float, default=11.4523)
    p.add_argument("--conv-cell-d-min-frac", type=float, default=0.02)
    p.add_argument("--conv-cell-n-random", type=int, default=300)
    p.add_argument("--conv-cell-seed", type=int, default=42)
    p.add_argument("--conv-cell-adaptive-random", action="store_true", default=True)
    p.add_argument("--conv-cell-adaptive-grid-n", type=int, default=48)
    p.add_argument("--conv-cell-adaptive-candidate-multiplier", type=float, default=12)
    p.add_argument("--rbf-phi", type=str, default="ga")
    p.add_argument("--rbf-eps", type=float, default=0.6)
    p.add_argument("--rbf-order", type=int, default=0)
    p.add_argument("--rbf-v-source", type=str, default="gaussian_direct")
    p.add_argument("--quality-probe-method", type=str, default="uniform")
    p.add_argument("--quality-probe-n", type=int, default=200000)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    compare_script = Path(args.compare_script)
    if not compare_script.exists() and not args.dry_run:
        raise FileNotFoundError(f"compare script not found: {compare_script}")

    if args.lambda_n < 2:
        lam_values = [float(args.lambda_min)]
    else:
        lo = math.log10(args.lambda_min)
        hi = math.log10(args.lambda_max)
        lam_values = [10 ** (lo + i * (hi - lo) / (args.lambda_n - 1)) for i in range(args.lambda_n)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"sweep_conv_cell_adaptive_lambdas_{ts}.json"

    base_cmd = [
        args.python,
        str(compare_script),
        "--system", args.system,
        "--qd-radius", str(args.qd_radius),
        "--el", str(args.el),
        "--nc", str(args.nc),
        "--n-random", str(args.n_random),
        "--dE", str(args.dE),
        "--V_min", str(args.V_min),
        "--fft-kinetic-cut", str(args.fft_kinetic_cut),
        "--rbf-node-method", args.rbf_node_method,
        "--conv-cell-template-mode", args.conv_cell_template_mode,
        "--conv-cell-domain-shape", args.conv_cell_domain_shape,
        "--rbf-stencil-radius", str(args.rbf_stencil_radius),
        "--conv-cell-fcc-scale-factor", str(args.conv_cell_fcc_scale_factor),
        "--conv-cell-fcc-origin-frac", *(str(v) for v in args.conv_cell_fcc_origin_frac),
        "--conv-cell-fcc-atom-refine-factor", str(args.conv_cell_fcc_atom_refine_factor),
        "--conv-cell-fcc-atom-radius-frac", str(args.conv_cell_fcc_atom_radius_frac),
        "--conv-cell-a", str(args.conv_cell_a),
        "--conv-cell-d-min-frac", str(args.conv_cell_d_min_frac),
        "--conv-cell-n-random", str(args.conv_cell_n_random),
        "--conv-cell-seed", str(args.conv_cell_seed),
        "--conv-cell-adaptive-grid-n", str(args.conv_cell_adaptive_grid_n),
        "--conv-cell-adaptive-candidate-multiplier", str(args.conv_cell_adaptive_candidate_multiplier),
        "--rbf-phi", args.rbf_phi,
        "--rbf-eps", str(args.rbf_eps),
        "--rbf-order", str(args.rbf_order),
        "--rbf-v-source", args.rbf_v_source,
        "--quality-probe-method", args.quality_probe_method,
        "--quality-probe-n", str(args.quality_probe_n),
    ]
    if args.conv_cell_adaptive_random:
        base_cmd.append("--conv-cell-adaptive-random")

    shared_nodes_path = args.load_nodes.strip() or None
    runs: list[dict[str, object]] = []

    for lam_grad in lam_values:
        for lam_lap in lam_values:
            cmd = base_cmd + [
                "--conv-cell-adaptive-lambda-grad", f"{lam_grad:.8g}",
                "--conv-cell-adaptive-lambda-lap", f"{lam_lap:.8g}",
            ]
            if shared_nodes_path:
                cmd += ["--load-nodes", shared_nodes_path]
            else:
                cmd += ["--save-nodes", args.save_nodes]

            saved_nodes, result_json = run_one(cmd, dry_run=args.dry_run)

            if not shared_nodes_path and saved_nodes:
                shared_nodes_path = saved_nodes
                print(f"[reuse] 后续组合将复用节点: {shared_nodes_path}")

            runs.append(
                {
                    "conv_cell_adaptive_lambda_grad": lam_grad,
                    "conv_cell_adaptive_lambda_lap": lam_lap,
                    "nodes_file": shared_nodes_path,
                    "result_json": result_json,
                }
            )

    manifest = {
        "script": "sweep_conv_cell_adaptive_lambdas.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "compare_script": str(compare_script),
        "lambda_values": lam_values,
        "shared_nodes_file": shared_nodes_path,
        "runs": runs,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n扫描完成，manifest 已保存: {manifest_path}")


if __name__ == "__main__":
    main()
