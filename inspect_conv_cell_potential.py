#!/usr/bin/env python3
"""构建并导出单个惯用晶胞内的势能采样，并生成切片图。

示例:
  python inspect_conv_cell_potential.py \
      --cube-file localPot.cube \
      --params-file gaussian_fit_params.json \
      --n-grid 64 \
      --a 11.4523
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaussian_potential_builder import GaussianPotentialBuilder


def _build_frac_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    frac = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return t, gx, gy, gz, frac


def main() -> None:
    p = argparse.ArgumentParser(description="采样并检查晶胞内部势能")
    p.add_argument("--cube-file", type=str, default="localPot.cube", help="cube 文件路径")
    p.add_argument("--params-file", type=str, default="gaussian_fit_params.json", help="高斯拟合参数 JSON")
    p.add_argument("--a", type=float, default=11.4523, help="晶格常数 a（Bohr）")
    p.add_argument("--n-grid", type=int, default=64, help="每个方向采样点数")
    p.add_argument(
        "--cell-origin",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("OX", "OY", "OZ"),
        help="晶胞原点（Bohr）",
    )
    p.add_argument("--r-cut", type=float, default=7.0, help="Gaussian 势能截断半径（与现有构建器一致）")
    p.add_argument("--out-npz", type=str, default="conv_cell_potential.npz", help="输出 npz 文件")
    p.add_argument("--out-fig", type=str, default="conv_cell_potential_slices.png", help="输出图像文件")
    args = p.parse_args()

    builder = GaussianPotentialBuilder(
        cube_file=args.cube_file,
        params_file=args.params_file,
        r_cut=args.r_cut,
    )

    n = int(args.n_grid)
    a = float(args.a)
    origin = np.asarray(args.cell_origin, dtype=np.float64).reshape(1, 3)

    t_frac, _, _, _, frac = _build_frac_grid(n)
    points_cart = origin + frac * a

    v_flat = np.asarray(builder.evaluate_at_points(points_cart), dtype=np.float64)
    v3 = v_flat.reshape(n, n, n)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        potential=v3,
        potential_flat=v_flat,
        points_cart=points_cart,
        points_frac=frac,
        frac_axis=t_frac,
        a=np.float64(a),
        cell_origin_cart=origin.reshape(3),
    )

    mid = n // 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(v3[:, :, mid], origin="lower", cmap="coolwarm")
    axes[0].set_title(f"V(x,y,z_mid), z={t_frac[mid]:.3f}a")
    axes[0].set_xlabel("x-index")
    axes[0].set_ylabel("y-index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(v3[:, mid, :], origin="lower", cmap="coolwarm")
    axes[1].set_title(f"V(x,y_mid,z), y={t_frac[mid]:.3f}a")
    axes[1].set_xlabel("x-index")
    axes[1].set_ylabel("z-index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(v3[mid, :, :], origin="lower", cmap="coolwarm")
    axes[2].set_title(f"V(x_mid,y,z), x={t_frac[mid]:.3f}a")
    axes[2].set_xlabel("y-index")
    axes[2].set_ylabel("z-index")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    print(f"[OK] saved npz: {out_npz.resolve()}")
    print(f"[OK] saved fig: {out_fig.resolve()}")
    print(
        "[stats] "
        f"Vmin={v3.min():.6e}, Vmax={v3.max():.6e}, "
        f"Vmean={v3.mean():.6e}, Vstd={v3.std():.6e}"
    )


if __name__ == "__main__":
    main()
