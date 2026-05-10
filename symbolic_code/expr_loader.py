"""CubicExpressionManager – load per-cube symbolic potential expressions.

This module provides the ``CubicExpressionManager`` class that was previously
imported as ``load_partitioned_Vexpr.CubicExpressionManager`` in the original
scripts.  It scans a directory of ``.pkl`` files produced by
:mod:`symbolic_code.space_partition` and offers indexed access by cube index.
"""

import pickle
from pathlib import Path


class CubicExpressionManager:
    """Indexed store of per-cube sympy potential expressions.

    Attributes
    ----------
    expressions : dict
        Maps ``(i, j, k)`` tuple → sympy ``V_expr``.
    cube_info : dict
        Maps ``(i, j, k)`` → metadata dict with keys
        ``center``, ``n_atoms``, ``cube_size``, ``r_cut``.
    i_range, j_range, k_range : tuple of (min, max)
        Extent of loaded cube indices along each axis.
    """

    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : str or Path
            Directory containing ``.pkl`` files from
            :meth:`CubicSpacePartition.process_all_cubes`.
        """
        self.directory = Path(directory)
        self.expressions = {}
        self.cube_info = {}
        self.i_range = self.j_range = self.k_range = (0, 0)
        self._load_all()

    def _load_all(self):
        pkl_files = sorted(
            p for p in self.directory.glob('*.pkl')
            if not p.name.endswith('_info.pkl')
        )
        if not pkl_files:
            print(f"[CubicExpressionManager] No .pkl files in {self.directory}")
            return

        for path in pkl_files:
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
            except Exception as exc:
                print(f"  Warning: could not load {path.name}: {exc}")
                continue

            idx = tuple(data['cube_index'])
            self.expressions[idx] = data['expr']
            self.cube_info[idx] = {
                'center':    data['cube_center'],
                'n_atoms':   data['n_atoms'],
                'cube_size': data['cube_size'],
                'r_cut':     data['r_cut'],
                'symbols':   data.get('symbols'),
                'atom_info': data.get('atom_info'),
            }

        if self.expressions:
            all_i = [k[0] for k in self.expressions]
            all_j = [k[1] for k in self.expressions]
            all_k = [k[2] for k in self.expressions]
            self.i_range = (min(all_i), max(all_i))
            self.j_range = (min(all_j), max(all_j))
            self.k_range = (min(all_k), max(all_k))

        print(
            f"[CubicExpressionManager] Loaded {len(self.expressions)} cubes "
            f"from {self.directory}"
        )

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_expression(self, i, j, k):
        """Return the sympy V_expr for cube (i, j, k), or None if absent."""
        return self.expressions.get((i, j, k))

    def __len__(self):
        return len(self.expressions)

    def __contains__(self, idx):
        return tuple(idx) in self.expressions

    def __repr__(self):
        return (
            f"CubicExpressionManager({len(self)} cubes, "
            f"i={self.i_range}, j={self.j_range}, k={self.k_range})"
        )
