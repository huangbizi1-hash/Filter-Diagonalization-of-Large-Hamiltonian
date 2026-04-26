#!/usr/bin/env python3
"""Sweep conv-cell-fcc-scale-factor for compare_fft_rbf_filter_qd.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_FACTORS = [4, 6, 8, 10, 12, 14, 16, 18]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep --conv-cell-fcc-scale-factor for compare_fft_rbf_filter_qd.py",
    )
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--compare-script", type=str, default="compare_fft_rbf_filter_qd.py")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--scale-factors", type=int, nargs="+", default=DEFAULT_FACTORS)
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

    base_cmd = [
        args.python,
        str(compare_script),
        "--system", "qd",
        "--qd-radius", "13",
        "--el", "-0.18",
        "--nc", "500",
        "--n-random", "64",
        "--dE", "40.0",
        "--fft-kinetic-cut", "20.0",
        "--rbf-node-method", "conv_cell",
        "--conv-cell-template-mode", "fcc_refined",
        "--conv-cell-domain-shape", "cube",
        "--conv-cell-fcc-origin-frac", "0.0", "0.0", "0.0",
        "--conv-cell-a", "11.4523",
        "--conv-cell-d-min-frac", "0.02",
        "--conv-cell-n-random", "300",
        "--conv-cell-seed", "42",
        "--rbf-stencil-size", "18",
        "--rbf-phi", "ga",
        "--rbf-eps", "0.6",
        "--rbf-order", "0",
        "--rbf-v-source", "gaussian_direct",
        "--quality-probe-method", "uniform",
        "--quality-probe-n", "200000",
        "--save-nodes", args.save_nodes,
    ]

    total = 0
    for factor in args.scale_factors:
        cmd = base_cmd + ["--conv-cell-fcc-scale-factor", str(factor)]
        run_one(cmd, args.dry_run)
        total += 1
        print(f"[done] conv-cell-fcc-scale-factor={factor}")

    print(f"\nSweep finished. Total runs = {total}")


if __name__ == "__main__":
    main()
