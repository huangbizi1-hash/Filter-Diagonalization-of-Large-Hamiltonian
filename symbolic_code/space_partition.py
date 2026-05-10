"""空间分割势能符号表达式生成。

将 [-L_s, L_s]^3 空间分割为 n_divisions^3 个小立方体，为每个立方体
生成基于高斯拟合原子势的符号表达式 V(x,y,z)，并保存为 pickle 文件，
供后续 h_powers 模块的符号 H^n 计算使用。
"""

import json
import os
import pickle
from collections import Counter
from typing import Optional

import numpy as np
import sympy as sp
from sympy import symbols, exp


class CubicSpacePartition:
    """将 3-D 空间分割为等大小的立方体，为每个非空立方体构建势能符号表达式。"""

    def __init__(self, params_file, atoms, L_s=8.0, n_divisions=4, r_cut=5.0, cube_size=None, anchor_corner=None):
        """
        Parameters
        ----------
        params_file : str
            高斯拟合参数 JSON 文件路径（需含 ``atoms`` 字段）。
        atoms : list of dict
            原子列表，每个元素含 ``'type'``, ``'x'``, ``'y'``, ``'z'`` 键。
        L_s : float
            空间半径 (Bohr)；实际区域为 ``[-L_s, L_s]^3``。
        n_divisions : int
            每维度分割数；总立方体数 = n_divisions^3。
        r_cut : float
            截断半径 (Bohr)，超出此范围的原子对该立方体无贡献。
        """
        with open(params_file, 'r', encoding='utf-8') as f:
            self.params_data = json.load(f)
        self.atom_params = self.params_data['atoms']

        self.atoms = list(atoms)
        self.L_s = L_s
        self.n_divisions = n_divisions
        self.r_cut = r_cut
        self.anchor_corner = np.array(anchor_corner, dtype=float) if anchor_corner is not None else None
        self.l0 = float(cube_size) if cube_size is not None else 2 * L_s / n_divisions  # side length of one cube

        print(f"\n{'='*70}")
        print("CUBIC SPACE PARTITION SETUP")
        print(f"{'='*70}")
        print(f"  Atom types loaded : {list(self.atom_params.keys())}")
        print(f"  Space             : [{-L_s:.1f}, {L_s:.1f}]^3 Bohr")
        if cube_size is None:
            print(f"  Divisions/dim     : {n_divisions}  →  {n_divisions**3} cubes")
        else:
            print(f"  Partition mode    : crystal-cell cubes (auto-count)")
            if self.anchor_corner is not None:
                print(f"  Anchor corner     : ({self.anchor_corner[0]:.3f}, {self.anchor_corner[1]:.3f}, {self.anchor_corner[2]:.3f})")
        print(f"  Cube side length  : {self.l0:.3f} Bohr")
        print(f"  Cutoff radius     : {r_cut:.2f} Bohr")
        print(f"  Total atoms       : {len(atoms)}")

        for i, atom in enumerate(self.atoms):
            atom['index'] = i

        self.cube_centers = self._generate_cube_centers()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _generate_cube_centers(self):
        centers = []
        half = self.l0 / 2.0
        if self.anchor_corner is None and self.l0 != (2 * self.L_s / self.n_divisions):
            lo, hi = -self.L_s, self.L_s
            n = int(np.ceil((hi - lo) / self.l0))
            start = lo + half
            coords = start + np.arange(n) * self.l0
            coords = coords[coords <= (hi - half + 1e-12)]
        elif self.anchor_corner is None:
            coords = np.linspace(-self.L_s + half, self.L_s - half, self.n_divisions)
        else:
            lo, hi = -self.L_s, self.L_s
            nmin = np.floor((lo - self.anchor_corner) / self.l0).astype(int)
            nmax = np.ceil((hi - self.anchor_corner) / self.l0).astype(int) - 1
            coords_xyz = [self.anchor_corner[d] + (np.arange(nmin[d], nmax[d] + 1) + 0.5) * self.l0 for d in range(3)]
            coords = None
        if coords is not None:
            iter_x = list(enumerate(coords)); iter_y = list(enumerate(coords)); iter_z = list(enumerate(coords))
        else:
            iter_x = [(int(ii), cx) for ii, cx in zip(range(len(coords_xyz[0])), coords_xyz[0])]
            iter_y = [(int(jj), cy) for jj, cy in zip(range(len(coords_xyz[1])), coords_xyz[1])]
            iter_z = [(int(kk), cz) for kk, cz in zip(range(len(coords_xyz[2])), coords_xyz[2])]
        for i, cx in iter_x:
            for j, cy in iter_y:
                for k, cz in iter_z:
                    centers.append({
                        'index': (i, j, k),
                        'center': np.array([cx, cy, cz]),
                        'id': len(centers),
                    })
        print(f"\n  Generated {len(centers)} cube centers.")
        return centers

    # ------------------------------------------------------------------
    # Atom–cube geometry
    # ------------------------------------------------------------------

    def find_relevant_atoms_for_cube(self, cube_center):
        """Return atoms within r_cut of *cube_center*, sorted by cube distance."""
        half = self.l0 / 2.0
        relevant = []
        for atom in self.atoms:
            pos = np.array([atom['x'], atom['y'], atom['z']])
            d_cube = self._point_to_cube_distance(pos, cube_center, half)
            if d_cube > self.r_cut:
                continue
            relevant.append({
                'atom': atom,
                'distance_to_center': float(np.linalg.norm(pos - cube_center)),
                'distance_to_cube': d_cube,
                'in_cube': self._point_in_cube(pos, cube_center, half),
            })
        relevant.sort(key=lambda x: x['distance_to_cube'])
        return relevant

    def _point_in_cube(self, point, cube_center, half):
        return bool(np.all(np.abs(point - cube_center) <= half))

    def _point_to_cube_distance(self, point, cube_center, half):
        d = np.abs(point - cube_center) - half
        return float(np.sqrt(np.sum(np.maximum(d, 0) ** 2)))

    # ------------------------------------------------------------------
    # Symbolic potential
    # ------------------------------------------------------------------

    def generate_symbolic_potential(self, relevant_atoms, cube_center):
        """Build V(x,y,z) = sum of Gaussian contributions from nearby atoms.

        Returns
        -------
        V_expr : sympy expression
        symbols_dict : {'x': x_sym, 'y': y_sym, 'z': z_sym}
        atom_info_list : list of metadata dicts
        """
        x, y, z = symbols('x y z', real=True)
        V_total = sp.Integer(0)
        atom_info_list = []

        for i, rel in enumerate(relevant_atoms):
            atom = rel['atom']
            pos = np.array([atom['x'], atom['y'], atom['z']])
            atype = atom['type']
            p = self.atom_params[atype]

            dx = x - float(pos[0])
            dy = y - float(pos[1])
            dz = z - float(pos[2])
            r2 = dx**2 + dy**2 + dz**2

            if p.get('method', 'gaussian') == 'gaussian':
                V_atom = sum(
                    float(A) * exp(-float(a) * r2)
                    for A, a in zip(p['amplitudes'], p['exponents'])
                )
                V_total += V_atom
                atom_info_list.append({
                    'index': i,
                    'global_index': atom['index'],
                    'type': atype,
                    'position': tuple(pos),
                    'amplitudes': p['amplitudes'],
                    'exponents': p['exponents'],
                    'distance_to_center': rel['distance_to_center'],
                    'in_cube': rel['in_cube'],
                    'r_cut': self.r_cut,
                })

        return V_total, {'x': x, 'y': y, 'z': z}, atom_info_list

    # ------------------------------------------------------------------
    # Bulk processing
    # ------------------------------------------------------------------

    def process_all_cubes(self, output_base_dir, verbose=True, stats_filename: Optional[str] = 'cube_atom_count_histogram.json'):
        """Process every cube and save per-cube potential pickle + info text."""
        os.makedirs(output_base_dir, exist_ok=True)
        total = len(self.cube_centers)
        with_atoms = 0
        atom_count_hist = Counter()

        for cube_info in self.cube_centers:
            cube_center = cube_info['center']
            cube_idx = cube_info['index']
            cube_id = cube_info['id']

            relevant = self.find_relevant_atoms_for_cube(cube_center)
            if not relevant:
                if verbose:
                    print(f"  Cube {cube_id:3d} {cube_idx}: no atoms – skipped")
                continue

            with_atoms += 1
            atom_count_hist[len(relevant)] += 1
            type_count = Counter(r['atom']['type'] for r in relevant)
            in_n = sum(1 for r in relevant if r['in_cube'])
            cx, cy, cz = cube_center

            atom_stat = "_".join(
                f"{t}_{c}" for t, c in sorted(type_count.items())
            )
            fname = (
                f"cube_center_{cx:.3f}_{cy:.3f}_{cz:.3f}_"
                f"atoms_{atom_stat}_total_{len(relevant)}_inside_{in_n}"
            ).replace("-", "neg")

            if verbose:
                print(
                    f"  Cube {cube_id:3d} {cube_idx}: "
                    f"center=({cx:.2f},{cy:.2f},{cz:.2f}), "
                    f"{len(relevant)} atoms ({in_n} inside)"
                )

            V_expr, sym_dict, atom_info = self.generate_symbolic_potential(
                relevant, cube_center
            )
            self._save_cube_expression(
                V_expr, sym_dict, atom_info,
                output_base_dir, fname,
                cube_center, cube_idx, relevant,
            )


        stats = {
            'total_cubes': total,
            'cubes_with_atoms': with_atoms,
            'empty_cubes': total - with_atoms,
            'cube_size': self.l0,
            'r_cut': self.r_cut,
            'atom_count_histogram': {str(k): v for k, v in sorted(atom_count_hist.items())},
        }
        if stats_filename:
            stats_path = os.path.join(output_base_dir, stats_filename)
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"Saved cube atom-count histogram to: {stats_path}")

        print(f"\nDone. Cubes total={total}, with_atoms={with_atoms}, "
              f"empty={total - with_atoms}")
        return with_atoms

    def _save_cube_expression(
        self, V_expr, symbols_dict, atom_info_list,
        output_dir, filename_base,
        cube_center, cube_idx, relevant_atoms,
    ):
        # --- pickle ---
        pkl_path = os.path.join(output_dir, f'{filename_base}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'expr': V_expr,
                'symbols': symbols_dict,
                'atom_info': atom_info_list,
                'cube_center': tuple(cube_center),
                'cube_index': cube_idx,
                'cube_size': self.l0,
                'r_cut': self.r_cut,
                'n_atoms': len(relevant_atoms),
            }, f)

        # --- human-readable summary ---
        txt_path = os.path.join(output_dir, f'{filename_base}_info.txt')
        type_count = Counter(r['atom']['type'] for r in relevant_atoms)
        in_n = sum(1 for r in relevant_atoms if r['in_cube'])
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Cube center : {tuple(cube_center)} Bohr\n")
            f.write(f"Cube index  : {cube_idx}\n")
            f.write(f"Cube size   : {self.l0:.6f} Bohr\n")
            f.write(f"r_cut       : {self.r_cut:.6f} Bohr\n\n")
            f.write(f"Relevant atoms: {len(relevant_atoms)}  "
                    f"(inside={in_n}, outside={len(relevant_atoms)-in_n})\n")
            for t, c in sorted(type_count.items()):
                f.write(f"  {t}: {c}\n")
            f.write("\nAtom list:\n")
            for info in atom_info_list:
                pos = info['position']
                f.write(
                    f"  {info['type']:3s} "
                    f"({pos[0]:8.4f},{pos[1]:8.4f},{pos[2]:8.4f}) "
                    f"dist={info['distance_to_center']:.4f} "
                    f"in={'Y' if info['in_cube'] else 'N'}\n"
                )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def read_cube_atoms(cube_file):
    """Parse atomic positions from a Gaussian cube file.

    Atomic-number → type mapping: {1:'P1', 15:'P2', 33:'As', 49:'In'}.
    Returns a list of dicts with keys 'type', 'x', 'y', 'z', 'index'.
    """
    an_to_type = {1: 'P1', 15: 'P2', 33: 'As', 49: 'In'}
    atoms = []
    with open(cube_file, 'r') as f:
        f.readline()  # comment 1
        f.readline()  # comment 2
        n_atoms = int(f.readline().split()[0])
        for _ in range(3):  # voxel vectors
            f.readline()
        for i in range(n_atoms):
            parts = f.readline().split()
            an = int(parts[0])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            if an in an_to_type:
                atoms.append({
                    'type': an_to_type[an],
                    'x': x, 'y': y, 'z': z,
                    'index': i,
                })
    return atoms


def load_cube_expression(pickle_file):
    """Load a per-cube potential expression from a pickle file."""
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded: center={data['cube_center']}, n_atoms={data['n_atoms']}")
    return data
