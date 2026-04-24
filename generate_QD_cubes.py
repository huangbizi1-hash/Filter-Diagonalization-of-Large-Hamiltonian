"""
generate_QD_cubes.py
生成不同尺寸的 InAs 量子点（Quantum Dot）Cube 文件。

固定格点步长 d=0.625 Bohr，通过改变 QD 半径得到不同格点数 N。
每个 QD 用 H 原子钝化表面悬挂键。
网格电位值填充为零（后续势能由高斯拟合参数另行加载）。

输出目录：QD_Outputs/
  QD_R{r}.cube          — 含几何结构的 Cube 文件
  QD_R{r}_structure.png — 单个 QD 结构图
  All_QDs_Grid_Summary.png — 所有 QD 汇总图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ──────────────────────────────────────────────
# 参数
# ──────────────────────────────────────────────
a          = 11.4523    # InAs 晶格常数（Bohr）
in_h_bond  = 2.7291     # In-H 键长（Bohr）
as_h_bond  = 1.2652     # As-H 键长（Bohr）
d          = 0.625      # 固定格点步长（Bohr）

output_dir = "QD_Outputs"
os.makedirs(output_dir, exist_ok=True)

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def get_ideal_neighbors():
    """闪锌矿结构四面体方向（Bohr）。"""
    v = a / 4.0
    return np.array([[v, v, v], [v, -v, -v], [-v, v, -v], [-v, -v, v]])


def build_qd_with_grid(radius_bohr, filename):
    """构建 QD 并写入 Cube 文件，返回原子坐标和网格参数。"""
    n_cells = int(np.ceil(radius_bohr / a)) + 1

    base_in = np.array([[0, 0, 0], [0.5, 0.5, 0],
                         [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    base_as = np.array([[0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
                         [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]) * a

    in_pos, as_pos = [], []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                offset = np.array([i, j, k]) * a
                for p in base_in:
                    pos = p + offset
                    if np.linalg.norm(pos) <= radius_bohr:
                        in_pos.append(pos)
                for p in base_as:
                    pos = p + offset
                    if np.linalg.norm(pos) <= radius_bohr:
                        as_pos.append(pos)

    in_pos = np.array(in_pos) if in_pos else np.empty((0, 3))
    as_pos = np.array(as_pos) if as_pos else np.empty((0, 3))
    all_pos = np.vstack([in_pos, as_pos]) if len(in_pos) and len(as_pos) else np.empty((0, 3))

    # H 钝化
    h_pos = []
    if len(all_pos):
        tree = KDTree(all_pos)
        ideal_dirs_in = get_ideal_neighbors()
        ideal_dirs_as = -get_ideal_neighbors()

        for pos in in_pos:
            for dv in ideal_dirs_in:
                if tree.query(pos + dv)[0] > 0.1 * a:
                    h_pos.append(pos + dv / np.linalg.norm(dv) * in_h_bond)
        for pos in as_pos:
            for dv in ideal_dirs_as:
                if tree.query(pos + dv)[0] > 0.1 * a:
                    h_pos.append(pos + dv / np.linalg.norm(dv) * as_h_bond)

    h_pos = np.array(h_pos) if h_pos else np.empty((0, 3))

    # 网格参数
    box_half = radius_bohr + 5.0
    origin   = -box_half
    N        = int(np.ceil(2 * box_half / d))
    natoms   = len(in_pos) + len(as_pos) + len(h_pos)

    # 写入 Cube 文件
    with open(filename, 'w') as f:
        f.write(f"Generated InAs Quantum Dot (Radius={radius_bohr})\n")
        f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        f.write(f"{natoms:5d} {origin:12.6f} {origin:12.6f} {origin:12.6f}\n")
        f.write(f"{N:5d} {d:12.6f}     0.000000     0.000000\n")
        f.write(f"{N:5d}     0.000000 {d:12.6f}     0.000000\n")
        f.write(f"{N:5d}     0.000000     0.000000 {d:12.6f}\n")

        for pos in in_pos:
            f.write(f"   49     0.000000 {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        for pos in as_pos:
            f.write(f"   33     0.000000 {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        for pos in h_pos:
            f.write(f"    1     0.000000 {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")

        total_points = N ** 3
        zero_str  = "  0.00000E+00"
        full_lines = total_points // 6
        remainder  = total_points % 6
        for _ in range(full_lines):
            f.write(zero_str * 6 + "\n")
        if remainder:
            f.write(zero_str * remainder + "\n")

    return in_pos, as_pos, h_pos, N, origin, natoms


# ──────────────────────────────────────────────
# 批量生成
# ──────────────────────────────────────────────
radii = list(range(11, 28, 2))   # [11, 13, 15, 17, 19, 21, 23, 25, 27] Bohr
print(f"生成半径列表：{radii} Bohr")

fig_grid = plt.figure(figsize=(15, 15))

for idx, r in enumerate(radii):
    cube_file = os.path.join(output_dir, f"QD_R{r}.cube")
    in_pos, as_pos, h_pos, N, orig, natoms = build_qd_with_grid(r, cube_file)
    print(f"  已保存：{cube_file}  |  网格 {N}×{N}×{N} = {N**3:,}  |  原子数：{natoms}")

    # 单独结构图
    fig_s = plt.figure(figsize=(8, 8))
    ax_s  = fig_s.add_subplot(111, projection='3d')
    if len(in_pos): ax_s.scatter(in_pos[:, 0], in_pos[:, 1], in_pos[:, 2],
                                  c='purple', label='In (49)', s=30)
    if len(as_pos): ax_s.scatter(as_pos[:, 0], as_pos[:, 1], as_pos[:, 2],
                                  c='green',  label='As (33)', s=30)
    if len(h_pos):  ax_s.scatter(h_pos[:, 0],  h_pos[:, 1],  h_pos[:, 2],
                                  c='pink',   label='H (1)',   s=10)
    ax_s.set_title(f"InAs QD  R={r} Bohr  N={N}  Atoms={natoms}")
    ax_s.set_axis_off(); ax_s.legend()
    plt.tight_layout()
    fig_s.savefig(os.path.join(output_dir, f"QD_R{r}_structure.png"), dpi=300)
    plt.close(fig_s)

    # 汇总图格子
    ax_g = fig_grid.add_subplot(3, 3, idx + 1, projection='3d')
    if len(in_pos): ax_g.scatter(in_pos[:, 0], in_pos[:, 1], in_pos[:, 2],
                                  c='purple', s=15)
    if len(as_pos): ax_g.scatter(as_pos[:, 0], as_pos[:, 1], as_pos[:, 2],
                                  c='green',  s=15)
    if len(h_pos):  ax_g.scatter(h_pos[:, 0],  h_pos[:, 1],  h_pos[:, 2],
                                  c='pink',   s=3)
    ax_g.set_title(f"R={r}  N={N}  Atoms={natoms}", fontsize=8)
    ax_g.set_axis_off()

plt.tight_layout()
fig_grid.savefig(os.path.join(output_dir, "All_QDs_Grid_Summary.png"), dpi=300)
plt.close(fig_grid)

print(f"\n全部文件已保存至：{os.path.abspath(output_dir)}")
