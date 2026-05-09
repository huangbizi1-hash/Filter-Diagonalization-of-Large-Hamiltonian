#!/usr/bin/env python3
"""统计目录中 cube 的 atoms 数分布。

用法:
    python count_cube_distribution.py /path/to/QD_R17_Vexpr_partition_uniform
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

TOTAL_RE = re.compile(r"_total_(\d+)_")


def count_distribution(folder: Path) -> Counter[int]:
    counter: Counter[int] = Counter()
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = TOTAL_RE.search(p.name)
        if m:
            counter[int(m.group(1))] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 cube 中 atoms 总数分布")
    parser.add_argument("folder", type=Path, help="包含 cube_* 文件的目录")
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"目录不存在或不是目录: {folder}")

    dist = count_distribution(folder)
    if not dist:
        print("未找到包含 '_total_<num>_' 的文件。")
        return

    total_cubes = sum(dist.values())
    print(f"目录: {folder}")
    print(f"总 cube 数: {total_cubes}")
    print("\natoms数目 -> cube个数")
    for n_atoms in sorted(dist):
        print(f"{n_atoms:>4} -> {dist[n_atoms]}")


if __name__ == "__main__":
    main()
