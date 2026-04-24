#!/usr/bin/env python3
"""Sweep rbf_eps × rbf_stencil_size for compare_fft_rbf_filter_qd.py.

特点：
1) 首次运行使用 --save-nodes 生成并保存节点；
2) 后续组合全部使用 --load-nodes 复用同一节点，避免重复节点生成；
3) 将每次运行的结果 JSON 路径汇总到一个 sweep manifest。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SAVE_NODES_PATTERN = re.compile(r"节点数据文件已保存:\s*(.+)")
RESULT_JSON_PATTERN = re.compile(r"结果已保存:\s*(.+)")


def frange(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    i = 0
    while True:
        v = round(start + i * step, 10)
        if v > end + 1e-12:
            break
        values.append(round(v, 4))
        i += 1
    return values


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
        description="Sweep --rbf-eps and --rbf-stencil-size for compare_fft_rbf_filter_qd.py",
    )

    # 扫描参数
    p.add_argument("--eps-start", type=float, default=0.2)
    p.add_argument("--eps-end", type=float, default=1.0)
    p.add_argument("--eps-step", type=float, default=0.1)
    p.add_argument("--stencil-sizes", type=int, nargs="+", default=[12, 18, 42])

    # 运行控制
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--compare-script", type=str, default="compare_fft_rbf_filter_qd.py")
    p.add_argument("--save-nodes", type=str, default="rbf_nodes/")
    p.add_argument("--load-nodes", type=str, default="", help="若给定则直接复用该节点文件，跳过首次建点")
    p.add_argument("--manifest-dir", type=str, default="filter_compare_results")
    p.add_argument("--dry-run", action="store_true")

    # 用户示例命令的默认参数（可覆盖）
    p.add_argument("--system", type=str, default="qd")
    p.add_argument("--qd-radius", type=int, default=13)
    p.add_argument("--el", type=float, default=-0.18)
    p.add_argument("--nc", type=int, default=500)
    p.add_argument("--n-random", type=int, default=64)
    p.add_argument("--dE", type=float, default=40.0)
    p.add_argument("--fft-kinetic-cut", type=float, default=20.0)
    p.add_argument("--rbf-node-method", type=str, default="conv_cell")
    p.add_argument("--conv-cell-template-mode", type=str, default="fcc_refined")
    p.add_argument("--conv-cell-domain-shape", type=str, default="cube")
    p.add_argument("--conv-cell-fcc-scale-factor", type=int, default=8)
    p.add_argument("--conv-cell-fcc-origin-frac", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--conv-cell-a", type=float, default=11.4523)
    p.add_argument("--conv-cell-d-min-frac", type=float, default=0.02)
    p.add_argument("--conv-cell-n-random", type=int, default=30)
    p.add_argument("--conv-cell-seed", type=int, default=42)
    p.add_argument("--rbf-phi", type=str, default="ga")
    p.add_argument("--rbf-order", type=int, default=0)
    p.add_argument("--rbf-v-source", type=str, default="gaussian_direct")
    p.add_argument("--quality-probe-method", type=str, default="uniform")
    p.add_argument("--quality-probe-n", type=int, default=200000)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    eps_values = frange(args.eps_start, args.eps_end, args.eps_step)

    compare_script = Path(args.compare_script)
    if not compare_script.exists() and not args.dry_run:
        raise FileNotFoundError(f"compare script not found: {compare_script}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"sweep_fft_rbf_qd_{ts}.json"

    base_cmd = [
        args.python,
        str(compare_script),
        "--system", args.system,
        "--qd-radius", str(args.qd_radius),
        "--el", str(args.el),
        "--nc", str(args.nc),
        "--n-random", str(args.n_random),
        "--dE", str(args.dE),
        "--fft-kinetic-cut", str(args.fft_kinetic_cut),
        "--rbf-node-method", args.rbf_node_method,
        "--conv-cell-template-mode", args.conv_cell_template_mode,
        "--conv-cell-domain-shape", args.conv_cell_domain_shape,
        "--conv-cell-fcc-scale-factor", str(args.conv_cell_fcc_scale_factor),
        "--conv-cell-fcc-origin-frac", *(str(v) for v in args.conv_cell_fcc_origin_frac),
        "--conv-cell-a", str(args.conv_cell_a),
        "--conv-cell-d-min-frac", str(args.conv_cell_d_min_frac),
        "--conv-cell-n-random", str(args.conv_cell_n_random),
        "--conv-cell-seed", str(args.conv_cell_seed),
        "--rbf-phi", args.rbf_phi,
        "--rbf-order", str(args.rbf_order),
        "--rbf-v-source", args.rbf_v_source,
        "--quality-probe-method", args.quality_probe_method,
        "--quality-probe-n", str(args.quality_probe_n),
    ]

    shared_nodes_path = args.load_nodes.strip() or None
    runs: list[dict[str, object]] = []

    for stencil in args.stencil_sizes:
        for eps in eps_values:
            cmd = base_cmd + [
                "--rbf-stencil-size", str(stencil),
                "--rbf-eps", f"{eps:.4f}",
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
                    "stencil_size": stencil,
                    "rbf_eps": eps,
                    "nodes_file": shared_nodes_path,
                    "result_json": result_json,
                }
            )

    manifest = {
        "script": "sweep_compare_fft_rbf_filter_qd.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "compare_script": str(compare_script),
        "eps_values": eps_values,
        "stencil_sizes": args.stencil_sizes,
        "shared_nodes_file": shared_nodes_path,
        "runs": runs,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n扫描完成，manifest 已保存: {manifest_path}")


if __name__ == "__main__":
    main()
