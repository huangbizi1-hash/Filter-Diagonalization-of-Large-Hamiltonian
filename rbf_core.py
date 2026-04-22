from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from rbf.pde.fd import weight_matrix
from rbf.pde.nodes import poisson_disc_nodes


Array = np.ndarray


@dataclass
class RBFConfig:
    spacing: float = 0.5
    L: float = 5.0
    stencil_size: int = 80
    phi: str = "phs3"
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


def generate_nodes(spacing: float = 0.5, L: float = 5.0):
    vert = np.array(
        [
            [-L, -L, -L],
            [L, -L, -L],
            [L, L, -L],
            [-L, L, -L],
            [-L, -L, L],
            [L, -L, L],
            [L, L, L],
            [-L, L, L],
        ]
    )
    smp = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ]
    )
    nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp))
    return nodes, groups



def make_grid_points(L: float, N: int) -> Tuple[Array, Tuple[int, int, int]]:
    X, Y, Z = np.meshgrid(
        np.linspace(-L, L, N),
        np.linspace(-L, L, N),
        np.linspace(-L, L, N),
        indexing="ij",
    )
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return pts, X.shape



def build_problem(config: Optional[RBFConfig] = None, build_interpolation: bool = True) -> RBFProblem:
    config = RBFConfig() if config is None else config
    nodes, groups = generate_nodes(spacing=config.spacing, L=config.L)
    interior_idx = groups["interior"]

    laplacian_matrix = weight_matrix(
        x=nodes[interior_idx],
        p=nodes,
        n=config.stencil_size,
        diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        phi=config.phi,
        eps=config.eps,
        order=config.order,
    )

    psi_interp_matrix = None
    lap_interp_matrix = None
    grid_points = None
    grid_shape = None

    if build_interpolation:
        grid_points, grid_shape = make_grid_points(config.L, config.grid_N)
        psi_interp_matrix = weight_matrix(
            x=grid_points,
            p=nodes[interior_idx],
            n=config.stencil_size,
            diffs=[0, 0, 0],
            phi=config.phi,
            eps=config.eps,
            order=config.order,
        )
        lap_interp_matrix = psi_interp_matrix

    return RBFProblem(
        config=config,
        nodes=nodes,
        groups=groups,
        interior_idx=interior_idx,
        laplacian_matrix=laplacian_matrix,
        psi_interp_matrix=psi_interp_matrix,
        lap_interp_matrix=lap_interp_matrix,
        grid_points=grid_points,
        grid_shape=grid_shape,
    )



def relative_laplacian_error(problem: RBFProblem, psi_nodes: Optional[Array] = None) -> Dict[str, float]:
    if psi_nodes is None:
        psi_nodes = problem.ground_state()
    exact = problem.exact_laplacian()[problem.interior_idx]
    approx = problem.laplacian_matrix.dot(psi_nodes)
    abs_err = np.abs(approx - exact)
    denom = np.maximum(np.abs(exact), 1e-14)
    rel_err = abs_err / denom
    return {
        "mean_abs_err": float(np.mean(abs_err)),
        "max_abs_err": float(np.max(abs_err)),
        "mean_rel_err": float(np.mean(rel_err)),
        "max_rel_err": float(np.max(rel_err)),
    }



def iterate_hamiltonian(problem: RBFProblem, n_max: int = 20, normalize_each_step: bool = True):
    psi_nodes = problem.ground_state()
    records = []
    cumulative_scale = 1.0

    for n in range(1, n_max + 1):
        Hpsi, Tpsi, Vpsi = problem.apply_hamiltonian(psi_nodes)

        psi_interior = psi_nodes[problem.interior_idx]
        H_interior = Hpsi[problem.interior_idx]
        T_interior = Tpsi[problem.interior_idx]
        V_interior = Vpsi[problem.interior_idx]

        den = float(np.dot(psi_interior, psi_interior))
        E_H = float(np.dot(psi_interior, H_interior) * cumulative_scale / den)
        E_T = float(np.dot(psi_interior, T_interior) / den)
        E_V = float(np.dot(psi_interior, V_interior) / den)
        exact = float(1.5**n)
        rel_err = float(abs(E_H - exact) / abs(exact))

        records.append(
            IterationRecord(
                n=n,
                E_H=E_H,
                E_T=E_T,
                E_V=E_V,
                exact=exact,
                rel_err=rel_err,
                scale_factor=cumulative_scale,
            )
        )

        psi_nodes = Hpsi.copy()
        if normalize_each_step:
            max_val = float(np.max(np.abs(psi_nodes)))
            if max_val > 0.0:
                cumulative_scale *= max_val
                psi_nodes /= max_val

    return records
