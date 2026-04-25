#!/usr/bin/env python3
"""扫描 conv-cell-fcc-atom-radius-frac，并比较两种 stencil 配置。

扫描参数：
1) --conv-cell-fcc-atom-radius-frac: 0.04 -> 0.20, 步长 0.02
2) stencil 模式：
   - sphere: --rbf-stencil-radius 1.60
   - fixed18: 不使用球半径，仅 --rbf-stencil-size 18
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def frange(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    i = 0
    while True:
        v = round(start + i * step, 10)
        if v > end + 1e-12:
            break
        values.append(round(v, 2))
        i += 1
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep conv-cell-fcc-atom-radius-frac with 2 stencil modes",
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--compare-script", type=str, default="compare_fft_rbf_filter_qd.py")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--radius-start", type=float, default=0.04)
    p.add_argument("--radius-end", type=float, default=0.20)
    p.add_argument("--radius-step", type=float, default=0.02)
    p.add_argument("--save-nodes", type=str, default="rbf_nodes/")

    return p.parse_args()


def run_one(cmd: list[str], dry_run: bool) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    compare_script = Path(args.compare_script)
    if not compare_script.exists() and not args.dry_run:
        raise FileNotFoundError(f"compare script not found: {compare_script}")

    atom_radius_fracs = frange(args.radius_start, args.radius_end, args.radius_step)

    base_cmd = [
        args.python,
        str(compare_script),
        "--system", "qd",
        "--qd-radius", "17",
        "--el", "-0.18",
        "--nc", "500",
        "--n-random", "128",
        "--dE", "35.0",
        "--V_min", "-10.0",
        "--fft-kinetic-cut", "20.0",
        "--rbf-node-method", "conv_cell",
        "--conv-cell-template-mode", "fcc_refined",
        "--conv-cell-domain-shape", "cube",
        "--conv-cell-fcc-scale-factor", "8",
        "--conv-cell-fcc-origin-frac", "0.0", "0.0", "0.0",
        "--conv-cell-fcc-atom-refine-factor", "16",
        "--conv-cell-a", "11.4523",
        "--conv-cell-d-min-frac", "0.02",
        "--conv-cell-n-random", "300",
        "--conv-cell-seed", "42",
        "--rbf-phi", "ga",
        "--rbf-eps", "0.6",
        "--rbf-order", "0",
        "--rbf-v-source", "gaussian_direct",
        "--quality-probe-method", "uniform",
        "--quality-probe-n", "200000",
        "--save-nodes", args.save_nodes,
    ]

    modes: list[tuple[str, list[str]]] = [
        ("sphere_r1.60", ["--rbf-stencil-radius", "1.60"]),
        ("fixed_size18", ["--rbf-stencil-size", "18"]),
    ]

    total = 0
    for atom_radius_frac in atom_radius_fracs:
        for mode_name, mode_args in modes:
            cmd = base_cmd + [
                "--conv-cell-fcc-atom-radius-frac", f"{atom_radius_frac:.2f}",
            ] + mode_args
            run_one(cmd, args.dry_run)
            total += 1
            print(f"[done] mode={mode_name}, conv-cell-fcc-atom-radius-frac={atom_radius_frac:.2f}")

    print(f"\nSweep finished. Total runs = {total}")


if __name__ == "__main__":
    main()
