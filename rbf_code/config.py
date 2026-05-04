"""rbf_code.config — shared data classes, kernel constants, and scalar helpers.

Every other rbf_code sub-module imports from here; this module has NO imports
from the rest of rbf_code so there are no circular dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from rbf.pde.fd import weight_matrix
from rbf.pde.nodes import poisson_disc_nodes


Array = np.ndarray


def _raise_on_nonfinite(name: str, arr: Array, max_examples: int = 5) -> None:
    """Raise ValueError with compact diagnostics when arr contains NaN/Inf."""
    a = np.asarray(arr)
    mask = ~np.isfinite(a)
    if not np.any(mask):
        return
    bad_idx = np.argwhere(mask)
    snippets = []
    for idx in bad_idx[:max_examples]:
        idx_t = tuple(int(i) for i in idx)
        snippets.append(f"{idx_t}: {a[idx_t]!r}")
    raise ValueError(
        f"{name} contains non-finite values: "
        f"{int(mask.sum())}/{int(a.size)} entries are NaN/Inf. "
        f"Examples -> {'; '.join(snippets)}"
    )


def _periodic_delta_frac(x: Array, y: Array) -> Array:
    """Minimum-image signed displacement in fractional coordinates."""
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return d - np.round(d)

# ─── All supported RBF kernels ───────────────────────────────────────────────

KERNELS_ALL: list[str] = [
    # A. Polyharmonic splines
    "phs1", "phs2", "phs3", "phs4", "phs5", "phs6", "phs7", "phs8",
    # B. Classic global kernels
    "mq", "imq", "iq", "ga",
    # C. Length-scale kernels
    "exp", "se", "mat32", "mat52",
    # D. Compactly supported Wendland kernels
    "wen10", "wen11", "wen12", "wen30", "wen31", "wen32",
]

KERNEL_GROUPS: dict[str, list[str]] = {
    "PHS":         ["phs1", "phs2", "phs3", "phs4", "phs5", "phs6", "phs7", "phs8"],
    "Global":      ["mq", "imq", "iq", "ga"],
    "LengthScale": ["exp", "se", "mat32", "mat52"],
    "Wendland":    ["wen10", "wen11", "wen12", "wen30", "wen31", "wen32"],
}


@dataclass
class RBFConfig:
    spacing: float = 0.5
    L: float = 5.0
    stencil_size: int = 80
    phi: str = "ga"
    eps: float = 0.5
    order: int = 2
    grid_N: int = 60


@dataclass
class RBFProblem:
    config: RBFConfig
    nodes: Array
    groups: Dict[str, Array]
    interior_idx: Array
    laplacian_matrix: object
    psi_interp_matrix: Optional[object]
    lap_interp_matrix: Optional[object]
    grid_points: Optional[Array]
    grid_shape: Optional[Tuple[int, int, int]]
    # Custom potential on interior nodes (overrides harmonic V=0.5*r² when set)
    V_nodes: Optional[Array] = None
    # Cached sparse H on interior nodes (built lazily by apply_H_flat)
    _H_sparse: Optional[object] = None

    def ground_state(self, x: Optional[Array] = None) -> Array:
        pts = self.nodes if x is None else x
        r2 = np.sum(pts**2, axis=1)
        return np.pi ** (-0.75) * np.exp(-0.5 * r2)

    def exact_laplacian(self, x: Optional[Array] = None) -> Array:
        pts = self.nodes if x is None else x
        psi = self.ground_state(pts)
        r2 = np.sum(pts**2, axis=1)
        return (r2 - 3.0) * psi

    def potential(self, x: Optional[Array] = None) -> Array:
        if self.V_nodes is not None and x is None:
            return self.V_nodes
        pts = self.nodes[self.interior_idx] if x is None else x
        return 0.5 * np.sum(pts**2, axis=1)

    def apply_laplacian(self, psi_nodes: Array) -> Array:
        psi_nodes = np.asarray(psi_nodes, dtype=float)
        if psi_nodes.shape[0] != self.nodes.shape[0]:
            raise ValueError("psi_nodes length must match number of nodes")

        lap_full = np.zeros_like(psi_nodes)
        lap_full[self.interior_idx] = self.laplacian_matrix.dot(psi_nodes)
        return lap_full

    def apply_hamiltonian(self, psi_nodes: Array) -> Tuple[Array, Array, Array]:
        lap_full = self.apply_laplacian(psi_nodes)
        T_full = np.zeros_like(psi_nodes)
        V_full = np.zeros_like(psi_nodes)

        T_full[self.interior_idx] = -0.5 * lap_full[self.interior_idx]
        V_full[self.interior_idx] = self.potential() * psi_nodes[self.interior_idx]
        H_full = T_full + V_full
        return H_full, T_full, V_full

    def apply_H_flat(self, psi_interior: Array) -> Array:
        """
        Flat-vector matvec H(ψ) on interior nodes only: (n_interior,) → (n_interior,).

        Builds and caches a sparse H = -0.5·L_int[:, interior] + diag(V) on first call.
        Suitable for the filter-diagonalisation loop (expects such a callable).
        """
        if self._H_sparse is None:
            # lazy import avoids circular dependency with laplacian module
            from rbf_code.laplacian import build_hamiltonian_matrix
            self._H_sparse = build_hamiltonian_matrix(self, symmetrize=False)
        return self._H_sparse.dot(np.asarray(psi_interior, dtype=np.float64))

    def interpolate_to_grid(self, values_on_interior: Array, kind: str = "psi") -> Array:
        values_on_interior = np.asarray(values_on_interior, dtype=float)
        if self.psi_interp_matrix is None or self.grid_shape is None:
            raise ValueError("This problem was built without interpolation matrices")
        if values_on_interior.shape[0] != self.interior_idx.shape[0]:
            raise ValueError("values_on_interior length must match number of interior nodes")

        if kind == "psi":
            mat = self.psi_interp_matrix
        elif kind == "lap":
            mat = self.lap_interp_matrix
        else:
            raise ValueError("kind must be 'psi' or 'lap'")

        return mat.dot(values_on_interior).reshape(self.grid_shape)


@dataclass
class IterationRecord:
    n: int
    E_H: float
    E_T: float
    E_V: float
    exact: float
    rel_err: float
    scale_factor: float
