from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rbf.pde.fd import weight_matrix
from rbf.pde.nodes import poisson_disc_nodes


Array = np.ndarray


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
            # lazy-build the sparse H restricted to interior → interior
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



def build_hamiltonian_matrix(
    problem: RBFProblem,
    symmetrize: bool = False,
) -> sp.csr_matrix:
    """
    Assemble sparse H = -0.5 * L_int + diag(V) on interior nodes.

    symmetrize=False (default)
        Return H as-is.  RBF-FD Laplacians are generally non-symmetric;
        the result must be solved with a non-symmetric eigensolver.
    symmetrize=True
        Apply (H + Hᵀ)/2 before returning, enabling eigsh (real, symmetric).
    """
    L_int = sp.csr_matrix(problem.laplacian_matrix)[:, problem.interior_idx]
    V_diag = sp.diags(problem.potential(), format="csr")
    H = -0.5 * sp.csr_matrix(L_int) + V_diag
    if symmetrize:
        H = 0.5 * (H + H.T)
    return H.tocsr()


def solve_lowest_eigenvalues(
    problem: RBFProblem,
    n_eigs: int = 6,
    device: str = "cpu",
    symmetrize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (eigenvalues, eigenvectors) for the n_eigs lowest eigenvalues of H.

    symmetrize=False (default) — non-symmetric solver, complex output
    ─────────────────────────────────────────────────────────────────
    CPU  : scipy.sparse.linalg.eigs  (ARPACK non-symmetric, sparse)
           shift-invert at sigma=0 → finds smallest-magnitude eigenvalues.
           Eigenvalues are complex; imaginary parts reflect discretisation
           asymmetry and should be small for well-conditioned stencils.
           Sort by real part ascending.
    CUDA : torch.linalg.eig on dense CUDA tensor → complex eigenvalues.

    symmetrize=True — symmetric solver, real output
    ─────────────────────────────────────────────────────────────────
    CPU  : scipy.sparse.linalg.eigsh (ARPACK symmetric, sparse) → real.
    CUDA : torch.linalg.eigh on dense CUDA tensor → real.

    eigenvalues : shape (n_eigs,)  — complex128 or float64 depending on symmetrize
    eigenvectors: shape (n_interior, n_eigs) — complex128 or float64
    """
    H = build_hamiltonian_matrix(problem, symmetrize=symmetrize)

    if device == "cpu":
        if symmetrize:
            vals, vecs = spla.eigsh(H, k=n_eigs, which="SM")
            order = np.argsort(vals)
            return vals[order].astype(np.float64), vecs[:, order]
        else:
            # shift-invert at 0: for positive-spectrum H, finds smallest eigenvalues
            vals, vecs = spla.eigs(H, k=n_eigs, sigma=0.0, which="LM")
            order = np.argsort(vals.real)
            return vals[order].astype(np.complex128), vecs[:, order].astype(np.complex128)

    elif device == "cuda":
        import torch
        H_dense = torch.tensor(H.toarray(), dtype=torch.float64, device="cuda")
        if symmetrize:
            vals_t, vecs_t = torch.linalg.eigh(H_dense)
            vals = vals_t[:n_eigs].cpu().numpy().astype(np.float64)
            vecs = vecs_t[:, :n_eigs].cpu().numpy()
        else:
            vals_t, vecs_t = torch.linalg.eig(H_dense)
            order = torch.argsort(vals_t.real)[:n_eigs]
            vals = vals_t[order].cpu().numpy().astype(np.complex128)
            vecs = vecs_t[:, order].cpu().numpy().astype(np.complex128)
        return vals, vecs

    else:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")


# ─── Cube file reader ────────────────────────────────────────────────────────

def read_cube_file(path: str) -> Tuple[Array, Array, Array, Array]:
    """
    Read a Gaussian cube file (orthogonal grid assumed).

    Returns
    -------
    x, y, z  : 1-D coordinate arrays (Bohr)
    potential : shape (nx, ny, nz) in Hartree
    """
    with open(path, "r") as f:
        lines = f.readlines()
    n_atoms = abs(int(lines[2].split()[0]))
    ox, oy, oz = (float(v) for v in lines[2].split()[1:4])
    nx, dx = int(lines[3].split()[0]), float(lines[3].split()[1])
    ny, dy = int(lines[4].split()[0]), float(lines[4].split()[2])
    nz, dz = int(lines[5].split()[0]), float(lines[5].split()[3])
    data_start = 6 + n_atoms
    vals: list[float] = []
    for line in lines[data_start:]:
        vals.extend(map(float, line.split()))
    potential = np.array(vals[: nx * ny * nz], dtype=np.float64).reshape(nx, ny, nz)
    x = ox + np.arange(nx) * dx
    y = oy + np.arange(ny) * dy
    z = oz + np.arange(nz) * dz
    return x, y, z, potential


def read_cube_atoms(path: str) -> Tuple[Array, Array]:
    """
    Read atom positions (and atomic numbers) from a Gaussian cube file.

    Returns
    -------
    positions      : (n_atoms, 3) in Bohr (same units as the cube grid)
    atomic_numbers : (n_atoms,) int
    """
    with open(path, "r") as f:
        lines = f.readlines()
    n_atoms = abs(int(lines[2].split()[0]))
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    numbers = np.zeros(n_atoms, dtype=np.int64)
    for i in range(n_atoms):
        parts = lines[6 + i].split()
        numbers[i] = int(parts[0])
        positions[i] = [float(parts[2]), float(parts[3]), float(parts[4])]
    return positions, numbers


# ─── Sphere boundary (icosphere) ─────────────────────────────────────────────

def _make_icosphere(R: float, n_subdivide: int = 3) -> Tuple[Array, Array]:
    """Triangulated sphere surface of radius R for use with poisson_disc_nodes."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw = np.array(
        [[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
         [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
         [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]],
        dtype=float,
    )
    verts: list[Array] = list(raw / np.linalg.norm(raw[0]) * R)
    faces: list[list[int]] = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]

    for _ in range(n_subdivide):
        midpoints: Dict[Tuple[int, int], int] = {}

        def _mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                m = (np.asarray(verts[a]) + np.asarray(verts[b])) / 2.0
                m = m / np.linalg.norm(m) * R
                midpoints[key] = len(verts)
                verts.append(m)
            return midpoints[key]

        new_faces: list[list[int]] = []
        for f in faces:
            a, b, c = f
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

    return np.array(verts), np.array(faces, dtype=int)


def generate_sphere_nodes(
    spacing: float, R: float = 20.0, n_subdivide: int = 3
):
    """Poisson disc nodes inside a sphere of radius R."""
    vert, smp = _make_icosphere(R, n_subdivide)
    nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp))
    return nodes, groups


# ─── Atom-augmented node placement (for real QD) ─────────────────────────────

def _filter_close_points(candidates: Array, pinned: Array, min_dist: float) -> Array:
    """Drop rows of `candidates` that lie within `min_dist` of any row of `pinned`.

    Uses KDTree for O(N log N) scaling.  Returns the kept subset.
    """
    if len(pinned) == 0 or min_dist <= 0.0:
        return candidates
    from scipy.spatial import cKDTree
    tree = cKDTree(pinned)
    d, _ = tree.query(candidates, k=1)
    return candidates[d > min_dist]


def generate_atom_augmented_nodes(
    atom_positions: Array,
    domain: str = "sphere",
    R: float = 20.0,
    spacing: float = 0.8,
    augment: str = "poisson_disc",
    exclude_radius: float = 0.0,
    sphere_subdivide: int = 3,
    cube_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[Array, Dict[str, Array]]:
    """
    Build RBF-FD nodes that are pinned at atom positions, optionally augmented
    with Poisson-disc fill of the surrounding domain.

    Strategy
    --------
    1. Pin all atom positions as interior nodes (they are never discarded).
    2. If augment == 'poisson_disc':
         generate Poisson-disc nodes inside the `domain` (sphere or box),
         drop any that lie within `exclude_radius` of an atom,
         concatenate to the atom list.
       If augment == 'none':
         only atoms form the interior; the user must provide a separate
         boundary (sphere surface nodes are added here too so groups are
         well-defined).

    Parameters
    ----------
    atom_positions   : (n_atoms, 3) array in Bohr
    domain           : 'sphere' — augment inside a sphere of radius R,
                       'box'    — augment inside the box cube_bounds × 3
    R                : sphere radius (Bohr), used if domain=='sphere'
    spacing          : Poisson-disc spacing for the augmentation fill
    augment          : 'poisson_disc' | 'none'
    exclude_radius   : drop augmentation candidates within this distance
                       (Bohr) of any atom; 0.0 keeps everything
    sphere_subdivide : icosphere subdivision for the sphere boundary
    cube_bounds      : (lo, hi) — box domain [lo, hi]^3 when domain=='box'

    Returns
    -------
    nodes  : (n_total, 3) concatenated array.  Atoms come first.
    groups : dict with keys 'interior', 'boundary', 'atoms'.
             'atoms'   : indices of the pinned atom nodes (subset of interior)
             'interior': atom indices + augmentation interior indices
             'boundary': boundary indices produced by the Poisson disc fill
                         (Dirichlet nodes); empty when augment=='none'
    """
    atom_positions = np.asarray(atom_positions, dtype=np.float64).reshape(-1, 3)
    n_atoms = len(atom_positions)

    if augment not in ("poisson_disc", "none"):
        raise ValueError(f"augment must be 'poisson_disc' or 'none', got {augment!r}")

    if augment == "none":
        # Atoms only — no boundary.  User must ensure the stencil has enough
        # neighbours; this mode is typically too sparse for a usable Laplacian,
        # but we expose it for experimentation.
        nodes = atom_positions.copy()
        groups = {
            "interior": np.arange(n_atoms, dtype=np.int64),
            "boundary": np.empty(0, dtype=np.int64),
            "atoms":    np.arange(n_atoms, dtype=np.int64),
        }
        return nodes, groups

    # ── augment == 'poisson_disc' ────────────────────────────────────────────
    if domain == "sphere":
        vert, smp = _make_icosphere(R, sphere_subdivide)
    elif domain == "box":
        if cube_bounds is None:
            raise ValueError("domain='box' requires cube_bounds=(lo, hi)")
        lo, hi = cube_bounds
        vert = np.array([
            [lo, lo, lo], [hi, lo, lo], [hi, hi, lo], [lo, hi, lo],
            [lo, lo, hi], [hi, lo, hi], [hi, hi, hi], [lo, hi, hi],
        ], dtype=np.float64)
        smp = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ], dtype=int)
    else:
        raise ValueError(f"domain must be 'sphere' or 'box', got {domain!r}")

    # Try pinning atoms directly via rbf's poisson_disc_nodes (recent API).
    # Fall back to unpinned fill + KDTree de-dup if the version doesn't accept it.
    try:
        aug_nodes, aug_groups, _ = poisson_disc_nodes(
            spacing, (vert, smp), pinned_nodes=atom_positions,
        )
    except TypeError:
        aug_nodes, aug_groups, _ = poisson_disc_nodes(spacing, (vert, smp))
        # Remove anything too close to an atom, then prepend atoms
        keep_interior = _filter_close_points(
            aug_nodes[aug_groups["interior"]], atom_positions,
            min_dist=max(exclude_radius, 0.5 * spacing),
        )
        keep_boundary = aug_nodes[aug_groups["boundary"]]  # keep all boundary
        nodes = np.vstack([atom_positions, keep_interior, keep_boundary])
        n_int_aug = len(keep_interior)
        n_bd = len(keep_boundary)
        groups = {
            "atoms":    np.arange(n_atoms, dtype=np.int64),
            "interior": np.arange(n_atoms + n_int_aug, dtype=np.int64),
            "boundary": np.arange(n_atoms + n_int_aug,
                                   n_atoms + n_int_aug + n_bd, dtype=np.int64),
        }
        return nodes, groups

    # With pinned_nodes, atoms are at aug_groups['interior'][:n_atoms]
    # (rbf places pinned nodes first in the interior group).
    interior = aug_groups["interior"]
    boundary = aug_groups.get("boundary", np.empty(0, dtype=np.int64))
    # Enforce exclusion zone manually on NON-atom interior nodes
    if exclude_radius > 0.0 and n_atoms > 0:
        non_atom_mask = np.ones(len(interior), dtype=bool)
        non_atom_mask[:n_atoms] = False  # never drop atoms
        non_atom_interior = aug_nodes[interior[non_atom_mask]]
        kept = _filter_close_points(non_atom_interior, atom_positions, exclude_radius)
        # Rebuild node set
        atom_interior = aug_nodes[interior[:n_atoms]]
        nodes = np.vstack([atom_interior, kept, aug_nodes[boundary]])
        n_int_aug = len(kept)
        groups = {
            "atoms":    np.arange(n_atoms, dtype=np.int64),
            "interior": np.arange(n_atoms + n_int_aug, dtype=np.int64),
            "boundary": np.arange(n_atoms + n_int_aug,
                                   n_atoms + n_int_aug + len(boundary),
                                   dtype=np.int64),
        }
        return nodes, groups

    groups_out = {
        "atoms":    np.arange(n_atoms, dtype=np.int64),
        "interior": np.asarray(interior, dtype=np.int64),
        "boundary": np.asarray(boundary, dtype=np.int64),
    }
    return aug_nodes, groups_out


# ─── Conventional-cell hybrid node generator (zincblende, e.g. InAs) ──────────

# Default zincblende atomic basis (fractional, wrt conventional cubic cell)
_CONV_CELL_IN_FRAC_DEFAULT: Array = np.array(
    [[0.0, 0.0, 0.0],
     [0.5, 0.5, 0.0],
     [0.5, 0.0, 0.5],
     [0.0, 0.5, 0.5]],
    dtype=np.float64,
)
_CONV_CELL_AS_FRAC_DEFAULT: Array = np.array(
    [[0.25, 0.25, 0.25],
     [0.75, 0.75, 0.25],
     [0.75, 0.25, 0.75],
     [0.25, 0.75, 0.75]],
    dtype=np.float64,
)

# The 4 FCC lattice points inside a conventional cubic cell (used to expand any
# high-symmetry base point into an FCC-symmetric set of cosets).
_FCC_OFFSETS_FRAC: Array = np.array(
    [[0.0, 0.0, 0.0],
     [0.5, 0.5, 0.0],
     [0.5, 0.0, 0.5],
     [0.0, 0.5, 0.5]],
    dtype=np.float64,
)

# level-3 "FCC high-symmetry" base points: the 8 corners of the sub-cube
# spanned by fractional coordinates in {1/3, 2/3}.  Each base point is expanded
# by the 4 FCC cosets → ≤32 cosets per cell (duplicates removed mod 1).
_LEVEL3_BASE_FRAC_DEFAULT: Array = np.array(
    [[1/3, 1/3, 1/3],
     [2/3, 2/3, 2/3],
     [1/3, 1/3, 2/3],
     [1/3, 2/3, 1/3],
     [2/3, 1/3, 1/3],
     [2/3, 2/3, 1/3],
     [2/3, 1/3, 2/3],
     [1/3, 2/3, 2/3]],
    dtype=np.float64,
)


def _wrap_frac(x: Array) -> Array:
    return np.asarray(x, dtype=np.float64) % 1.0


def _unique_rows_mod1(arr: Array, tol: float = 1e-10) -> Array:
    arr = _wrap_frac(arr)
    arr_q = np.round(arr / tol).astype(np.int64)
    _, idx = np.unique(arr_q, axis=0, return_index=True)
    return arr[np.sort(idx)]


def _unique_cart_rows(arr: Array, tol: float = 1e-8) -> Array:
    arr_q = np.round(arr / tol).astype(np.int64)
    _, idx = np.unique(arr_q, axis=0, return_index=True)
    return arr[np.sort(idx)]


def _periodic_diff(x: Array, y: Array) -> Array:
    return np.abs(_periodic_delta_frac(x, y))


def _periodic_dist(x: Array, y: Array) -> float:
    return float(np.linalg.norm(_periodic_diff(x, y)))


def _build_uniform_frac_grid(n_per_axis: int) -> Array:
    if n_per_axis < 2:
        raise ValueError(f"n_per_axis must be >= 2, got {n_per_axis}")
    t = np.linspace(0.0, 1.0, int(n_per_axis), endpoint=False, dtype=np.float64)
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _estimate_grad_laplacian_uniform(
    V: Array,
    n_per_axis: int,
    a: float,
) -> tuple[Array, Array]:
    """Estimate |∇V| and |ΔV| on a uniform [0,1)^3 grid of one conventional cell."""
    V3 = np.asarray(V, dtype=np.float64).reshape(n_per_axis, n_per_axis, n_per_axis)
    h = float(a) / float(n_per_axis)
    # periodic finite differences
    dVdx = (np.roll(V3, -1, axis=0) - np.roll(V3, 1, axis=0)) / (2.0 * h)
    dVdy = (np.roll(V3, -1, axis=1) - np.roll(V3, 1, axis=1)) / (2.0 * h)
    dVdz = (np.roll(V3, -1, axis=2) - np.roll(V3, 1, axis=2)) / (2.0 * h)
    grad_norm = np.sqrt(dVdx * dVdx + dVdy * dVdy + dVdz * dVdz)
    lap = (
        np.roll(V3, -1, axis=0) + np.roll(V3, 1, axis=0) +
        np.roll(V3, -1, axis=1) + np.roll(V3, 1, axis=1) +
        np.roll(V3, -1, axis=2) + np.roll(V3, 1, axis=2) -
        6.0 * V3
    ) / (h * h)
    return grad_norm.ravel(), np.abs(lap.ravel())


def _adaptive_accept_and_filter_periodic(
    candidates_frac: Array,
    weights_h: Array,
    d_min_frac: float,
    n_target: int,
    rng: np.random.Generator,
    pinned_frac: Optional[Array] = None,
) -> Array:
    """
    Weighted accept-reject + periodic KDTree minimum-distance filter.
    """
    from scipy.spatial import cKDTree

    cand = _wrap_frac(np.asarray(candidates_frac, dtype=np.float64).reshape(-1, 3))
    if cand.size == 0 or n_target <= 0:
        return np.empty((0, 3), dtype=np.float64)

    w = np.asarray(weights_h, dtype=np.float64).reshape(-1)
    if w.shape[0] != cand.shape[0]:
        raise ValueError("weights_h size mismatch with candidates_frac")
    h_max = float(np.max(w)) if w.size else 0.0
    if h_max <= 0.0:
        return np.empty((0, 3), dtype=np.float64)

    p_accept = np.clip(w / h_max, 0.0, 1.0)
    accepted_mask = rng.random(cand.shape[0]) < p_accept
    accepted = cand[accepted_mask]
    if accepted.shape[0] == 0:
        accepted = cand[np.argsort(-w)[: min(8, cand.shape[0])]]

    # process high-weight points first, improves quality for fixed budget
    acc_w = w[accepted_mask] if np.any(accepted_mask) else np.full(len(accepted), h_max)
    accepted = accepted[np.argsort(-acc_w)]

    shifts = np.array(
        [[i, j, k] for i in (-1.0, 0.0, 1.0)
         for j in (-1.0, 0.0, 1.0)
         for k in (-1.0, 0.0, 1.0)],
        dtype=np.float64,
    )

    kept: list[Array] = []
    pinned = (_wrap_frac(np.asarray(pinned_frac, dtype=np.float64).reshape(-1, 3))
              if pinned_frac is not None and len(pinned_frac) else
              np.empty((0, 3), dtype=np.float64))

    tree_points = pinned.copy()
    tree_aug = np.vstack([tree_points + s for s in shifts]) if len(tree_points) else np.empty((0, 3))
    tree = cKDTree(tree_aug) if len(tree_aug) else None
    r = float(d_min_frac)

    for x in accepted:
        if len(kept) >= n_target:
            break
        if tree is not None and tree.query_ball_point(x, r):
            continue
        kept.append(x)
        px = x[None, :]
        if tree_points.size:
            tree_points = np.vstack([tree_points, px])
        else:
            tree_points = px.copy()
        tree_aug = np.vstack([tree_points + s for s in shifts])
        tree = cKDTree(tree_aug)

    if not kept:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(kept, axis=0)


def _greedy_filter_by_dmin_periodic(points_frac: Array, d_min: float) -> Array:
    """Greedy d_min filter using periodic (minimum-image) distance in fractional
    coordinates.  Preserves the input order of `points_frac`."""
    kept: list[Array] = []
    for x in points_frac:
        if kept:
            dmin = min(_periodic_dist(x, y) for y in kept)
            if dmin < d_min:
                continue
        kept.append(x)
    if not kept:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(kept, axis=0)


def _make_unit_cube_surface(a: float) -> Tuple[Array, Array]:
    """8-vertex / 12-triangle surface of the cube [0,a]^3 for rbf's poisson_disc_nodes."""
    vert = np.array([
        [0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0],
        [0, 0, a], [a, 0, a], [a, a, a], [0, a, a],
    ], dtype=np.float64)
    smp = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [3, 2, 6], [3, 6, 7],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ], dtype=int)
    return vert, smp


def generate_conv_cell_nodes(
    bbox_min: Array,
    bbox_max: Array,
    a: float,
    d_min_frac: float = 0.06,
    n_random_target: int = 120,
    seed: int = 42,
    include_parity: bool = True,
    atom_frac: Optional[Array] = None,
    level3_base_frac: Optional[Array] = None,
    use_rbf_poisson: bool = True,
    boundary_margin_frac: float = 0.06,
    domain_shape: str = "cube",
    sphere_radius: Optional[float] = None,
    sphere_subdivide: int = 3,
    # adaptive random-node sampler options
    adaptive_random: bool = False,
    adaptive_grid_n: int = 36,
    adaptive_lambda_grad: float = 0.0,
    adaptive_lambda_lap: float = 0.0,
    adaptive_candidate_multiplier: float = 8.0,
    adaptive_gaussian_builder: Optional[Any] = None,
    verbose: bool = True,
) -> Tuple[Array, Dict[str, Array], Dict[str, Any]]:
    """
    Hybrid node generator on a conventional cubic cell (zincblende by default),
    tiled to fill an axis-aligned box.

    The per-cell node template is the union of:

        (1) atomic sites — 4 In + 4 As at the zincblende positions
            (override via `atom_frac`, shape (k, 3) in fractional coords)

        (2) level-3 FCC high-symmetry points — 8 base points at
            (i/3, j/3, k/3), i,j,k∈{1,2}, each expanded by the 4 FCC cosets
            (override bases via `level3_base_frac`, shape (m, 3))

        (3) Poisson-like random points — generated inside the unit cell with
            `rbf.pde.nodes.poisson_disc_nodes` (`use_rbf_poisson=True`, default)
            using the skeleton (1)+(2) as `pinned_nodes` and radius = d_min_frac*a.
            Falls back to the user's original periodic rejection sampler if
            `use_rbf_poisson=False`.

        (4) parity counterparts — for each random point r, add `1 − r` (inversion
            about the cell centre).  Disable via `include_parity=False`.

    After the union, a greedy d_min filter is applied in fractional coordinates
    with periodic (minimum-image) distance, preserving the order atoms → level3
    → random → parity.  The surviving fractional template is scaled by `a` and
    tiled by integer shifts `(nx, ny, nz) · a` to cover [bbox_min, bbox_max).

    Interior / boundary assignment on the tiled set:
        domain_shape='cube' (default):
            - a node within `boundary_margin_frac * a` of any face of the bbox
              is flagged as boundary (Dirichlet)
            - everything else is interior
        domain_shape='sphere':
            - keep only tiled template points inside a sphere
            - add explicit spherical-surface boundary nodes from an icosphere
              Poisson-disc boundary set
            - all kept template points are interior, sphere-surface nodes are
              boundary

    Returns
    -------
    nodes   : (N, 3) Cartesian, Bohr
    groups  : dict with int64 arrays under keys
              'interior', 'boundary', 'atoms', 'level3', 'random', 'parity'
              ('atoms' etc. are *template-role* groups relative to the one-cell
              template; after tiling they index every tiled copy)
    stats   : counts {cell_skeleton, cell_random, cell_accepted_parity,
                      cell_after_filter, tiled_total} plus one-cell template
              arrays for saving/inspection:
                  cell_template_nodes_frac : (N_cell, 3), [0,1)^3
                  cell_template_nodes_cart : (N_cell, 3), Bohr
                  cell_template_roles      : (N_cell,), 0/1/2/3
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)

    if atom_frac is None:
        atom_frac = np.vstack([_CONV_CELL_IN_FRAC_DEFAULT,
                                _CONV_CELL_AS_FRAC_DEFAULT])
    else:
        atom_frac = np.asarray(atom_frac, dtype=np.float64).reshape(-1, 3)

    if level3_base_frac is None:
        level3_base_frac = _LEVEL3_BASE_FRAC_DEFAULT
    else:
        level3_base_frac = np.asarray(level3_base_frac, dtype=np.float64).reshape(-1, 3)

    # level-3: expand each base by the 4 FCC cosets and reduce mod 1
    level3_frac = np.vstack([
        _wrap_frac(r0 + _FCC_OFFSETS_FRAC) for r0 in level3_base_frac
    ])
    level3_frac = _unique_rows_mod1(level3_frac)

    # skeleton = atoms + level-3 (these are all pinned)
    skeleton_frac = _unique_rows_mod1(np.vstack([atom_frac, level3_frac]))

    # ── Poisson-like / adaptive random points in [0,1)^3 ─────────────────────
    rng = np.random.default_rng(seed)
    if adaptive_random:
        if adaptive_gaussian_builder is None:
            raise ValueError("adaptive_random=True requires adaptive_gaussian_builder")
        n_grid = int(adaptive_grid_n)
        cand_frac = _build_uniform_frac_grid(n_grid)
        cand_cart = cand_frac * a
        V_cand = np.asarray(adaptive_gaussian_builder.evaluate_at_points(cand_cart),
                            dtype=np.float64)
        gnorm, lap_abs = _estimate_grad_laplacian_uniform(V_cand, n_grid, a)
        denom = 1.0 + float(adaptive_lambda_grad) * gnorm + float(adaptive_lambda_lap) * lap_abs
        h_w = 1.0 / np.maximum(denom, 1e-12)
        n_candidates = int(max(1, round(float(adaptive_candidate_multiplier) * n_random_target)))
        # draw top-weight-biased subset first to reduce KDTree insert loops
        take = np.argsort(-h_w)[: min(n_candidates, len(h_w))]
        random_frac = _adaptive_accept_and_filter_periodic(
            candidates_frac=cand_frac[take],
            weights_h=h_w[take],
            d_min_frac=d_min_frac,
            n_target=n_random_target,
            rng=rng,
            pinned_frac=skeleton_frac,
        )
    elif use_rbf_poisson:
        # Use the repo's Poisson-disc sampler on the unit cube [0,a]^3 with
        # the skeleton pinned.  radius = d_min_frac*a (Cartesian).
        vert, smp = _make_unit_cube_surface(a)
        pinned_cart = skeleton_frac * a
        try:
            rbf_nodes_cart, _rbf_groups, _ = poisson_disc_nodes(
                d_min_frac * a, (vert, smp), pinned_nodes=pinned_cart,
            )
            # Drop the pinned copies — we already have them in skeleton_frac
            # (rbf places pinned first in the interior group)
            n_pin = len(pinned_cart)
            extra_cart = rbf_nodes_cart[n_pin:]
            # Drop boundary nodes from the Poisson output — the cube faces'
            # boundary nodes aren't the user's "random" points
            # (filter anything within 1e-6*a of a face)
            face_tol = 1e-6 * a
            in_interior = np.all(
                (extra_cart > face_tol) & (extra_cart < a - face_tol), axis=1)
            extra_cart = extra_cart[in_interior]
            random_frac = extra_cart / a

            # Optionally truncate / seed-permute to match n_random_target
            if len(random_frac) > n_random_target:
                perm = rng.permutation(len(random_frac))[:n_random_target]
                random_frac = random_frac[perm]
        except TypeError:
            # Old rbf without pinned_nodes support — fall back to rejection
            use_rbf_poisson = False

    if not use_rbf_poisson:
        # Periodic rejection sampler (user's original algorithm)
        random_frac, _trials = _poisson_like_periodic(
            skeleton_frac, n_random_target, d_min_frac,
            max_trials=max(200_000, 2000 * n_random_target), rng=rng,
        )

    # ── parity counterparts ──────────────────────────────────────────────────
    if include_parity and len(random_frac) > 0:
        parity_frac = _wrap_frac(1.0 - random_frac)
    else:
        parity_frac = np.empty((0, 3), dtype=np.float64)

    # ── assemble template with role labels, then greedy d_min filter ─────────
    parts = [
        ("atoms",  atom_frac),
        ("level3", level3_frac),
        ("random", random_frac),
        ("parity", parity_frac),
    ]
    ordered_frac = np.vstack([p[1] for p in parts if len(p[1])])
    ordered_frac = _unique_rows_mod1(ordered_frac)
    # Track roles (by fractional coordinate lookup, using rounding)
    role_map: Dict[str, Array] = {}
    for name, arr in parts:
        role_map[name] = _wrap_frac(arr) if len(arr) else np.empty((0, 3), dtype=np.float64)

    cell_nodes_frac = _greedy_filter_by_dmin_periodic(ordered_frac, d_min_frac)
    cell_nodes_frac = _unique_rows_mod1(cell_nodes_frac)

    # For each surviving cell node, record which role set it came from
    # (first match in priority order atoms > level3 > random > parity)
    cell_role_idx: list[int] = []
    role_priority = ["atoms", "level3", "random", "parity"]
    for node in cell_nodes_frac:
        assigned = 3  # fallback = parity
        for ri, rname in enumerate(role_priority):
            src = role_map[rname]
            if len(src) == 0:
                continue
            if np.any(np.all(np.abs(_periodic_diff(src, node)) < 1e-9, axis=1)):
                assigned = ri
                break
        cell_role_idx.append(assigned)
    cell_role_idx = np.asarray(cell_role_idx, dtype=np.int64)

    cell_stats: Dict[str, Any] = {
        "cell_skeleton":         int(len(skeleton_frac)),
        "cell_random":           int(len(random_frac)),
        "cell_accepted_parity":  int(len(parity_frac)),
        "cell_after_filter":     int(len(cell_nodes_frac)),
        "cell_template_nodes_frac": cell_nodes_frac.copy(),
        "cell_template_nodes_cart": (cell_nodes_frac * a).copy(),
        "cell_template_roles":      cell_role_idx.copy(),
    }

    # ── tile the one-cell template to cover [bbox_min, bbox_max) ─────────────
    cell_nodes_cart = cell_nodes_frac * a
    if len(cell_nodes_cart) == 0:
        return (np.empty((0, 3), dtype=np.float64),
                {k: np.empty(0, dtype=np.int64)
                 for k in ("interior", "boundary", "atoms", "level3", "random", "parity")},
                {**cell_stats, "tiled_total": 0})

    nmin = np.floor((bbox_min - cell_nodes_cart.max(axis=0)) / a).astype(int) - 1
    nmax = np.ceil((bbox_max - cell_nodes_cart.min(axis=0)) / a).astype(int) + 1

    tiled_nodes: list[Array] = []
    tiled_roles: list[Array] = []
    for nx in range(nmin[0], nmax[0] + 1):
        for ny in range(nmin[1], nmax[1] + 1):
            for nz in range(nmin[2], nmax[2] + 1):
                shift = np.array([nx, ny, nz], dtype=np.float64) * a
                pts = cell_nodes_cart + shift
                m = np.all(pts >= bbox_min, axis=1) & np.all(pts < bbox_max, axis=1)
                if np.any(m):
                    tiled_nodes.append(pts[m])
                    tiled_roles.append(cell_role_idx[m])

    if not tiled_nodes:
        return (np.empty((0, 3), dtype=np.float64),
                {k: np.empty(0, dtype=np.int64)
                 for k in ("interior", "boundary", "atoms", "level3", "random", "parity")},
                {**cell_stats, "tiled_total": 0})

    nodes = np.vstack(tiled_nodes)
    roles = np.concatenate(tiled_roles)
    # final Cartesian dedup
    nodes_u, idx_u = np.unique(
        np.round(nodes / 1e-8).astype(np.int64), axis=0, return_index=True)
    idx_u = np.sort(idx_u)
    nodes = nodes[idx_u]
    roles = roles[idx_u]

    if domain_shape not in ("cube", "sphere"):
        raise ValueError(
            f"domain_shape must be 'cube' or 'sphere', got {domain_shape!r}")

    if domain_shape == "cube":
        # Interior / boundary: mark nodes within `boundary_margin_frac * a` of
        # any bbox face as boundary.
        margin = boundary_margin_frac * a
        near_face = (
            (nodes[:, 0] < bbox_min[0] + margin) | (nodes[:, 0] > bbox_max[0] - margin) |
            (nodes[:, 1] < bbox_min[1] + margin) | (nodes[:, 1] > bbox_max[1] - margin) |
            (nodes[:, 2] < bbox_min[2] + margin) | (nodes[:, 2] > bbox_max[2] - margin)
        )
        interior_idx = np.where(~near_face)[0].astype(np.int64)
        boundary_idx = np.where(near_face)[0].astype(np.int64)
        sphere_center = None
        sphere_radius_eff = None
    else:
        # Sphere mode: keep tiled template points inside the sphere and add
        # explicit boundary nodes on the sphere surface.
        sphere_center = 0.5 * (bbox_min + bbox_max)
        if sphere_radius is None:
            sphere_radius_eff = 0.5 * float(np.min(bbox_max - bbox_min))
        else:
            sphere_radius_eff = float(sphere_radius)
        if sphere_radius_eff <= 0.0:
            raise ValueError(
                f"sphere_radius must be > 0, got {sphere_radius_eff}")

        r = np.linalg.norm(nodes - sphere_center[None, :], axis=1)
        inside = r < sphere_radius_eff

        nodes_in = nodes[inside]
        roles_in = roles[inside]

        vert_s, smp_s = _make_icosphere(sphere_radius_eff, sphere_subdivide)
        vert_s = vert_s + sphere_center[None, :]
        spacing_surf = max(d_min_frac * a, 1e-6)
        surf_nodes_all, surf_groups, _ = poisson_disc_nodes(spacing_surf, (vert_s, smp_s))
        surf_boundary = surf_nodes_all[surf_groups["boundary"]]

        # Keep only boundary points to avoid adding volume fill from the sphere
        # Poisson solve; boundary points are on the triangulated surface.
        if len(nodes_in):
            nodes = np.vstack([nodes_in, surf_boundary])
        else:
            nodes = surf_boundary.copy()
        roles = np.concatenate([roles_in, -np.ones(len(surf_boundary), dtype=np.int64)])

        nodes_q = np.round(nodes / 1e-8).astype(np.int64)
        _, idx_u2 = np.unique(nodes_q, axis=0, return_index=True)
        idx_u2 = np.sort(idx_u2)
        nodes = nodes[idx_u2]
        roles = roles[idx_u2]

        is_surf = roles < 0
        boundary_idx = np.where(is_surf)[0].astype(np.int64)
        interior_idx = np.where(~is_surf)[0].astype(np.int64)
        margin = 0.0

    groups: Dict[str, Array] = {
        "interior": interior_idx,
        "boundary": boundary_idx,
        "atoms":    np.where(roles == 0)[0].astype(np.int64),
        "level3":   np.where(roles == 1)[0].astype(np.int64),
        "random":   np.where(roles == 2)[0].astype(np.int64),
        "parity":   np.where(roles == 3)[0].astype(np.int64),
    }
    stats = {
        **cell_stats,
        "tiled_total": int(len(nodes)),
        "domain_shape": domain_shape,
        "sphere_radius": (float(sphere_radius_eff) if sphere_radius_eff is not None else None),
        "adaptive_random": bool(adaptive_random),
    }

    if verbose:
        tile_range = (nmax - nmin + 1).tolist()
        a_bbox_min = nodes.min(axis=0)
        a_bbox_max = nodes.max(axis=0)
        i_bbox_min = (nodes[interior_idx].min(axis=0)
                      if len(interior_idx) else np.full(3, np.nan))
        i_bbox_max = (nodes[interior_idx].max(axis=0)
                      if len(interior_idx) else np.full(3, np.nan))
        print(f"[conv_cell] a={a:.4f}  bbox={bbox_min.tolist()} → {bbox_max.tolist()}")
        print(f"[conv_cell]   cell template: {cell_stats['cell_after_filter']} nodes "
              f"(skeleton={cell_stats['cell_skeleton']}, random={cell_stats['cell_random']}, "
              f"parity={cell_stats['cell_accepted_parity']})")
        print(f"[conv_cell]   tile shifts: {tile_range[0]}×{tile_range[1]}×{tile_range[2]} "
              f"→ {len(nodes)} tiled nodes")
        print(f"[conv_cell]   all-node bbox     : {np.round(a_bbox_min, 3).tolist()} → "
              f"{np.round(a_bbox_max, 3).tolist()}")
        if domain_shape == "cube":
            print(f"[conv_cell]   margin={margin:.3f} Bohr  (frac={boundary_margin_frac})")
        else:
            print(f"[conv_cell]   sphere center      : {np.round(sphere_center, 3).tolist()}")
            print(f"[conv_cell]   sphere radius      : {sphere_radius_eff:.3f} Bohr")
            print(f"[conv_cell]   sphere subdivide   : {sphere_subdivide}")
        print(f"[conv_cell]   interior / boundary = "
              f"{len(interior_idx)} / {len(boundary_idx)}")
        if len(interior_idx):
            print(f"[conv_cell]   interior bbox     : {np.round(i_bbox_min, 3).tolist()} → "
                  f"{np.round(i_bbox_max, 3).tolist()}")
    return nodes, groups, stats


def _poisson_like_periodic(
    skeleton_frac: Array,
    n_target: int,
    d_min: float,
    max_trials: int,
    rng: np.random.Generator,
) -> Tuple[Array, int]:
    """Fallback sampler (user's original algorithm): periodic rejection.

    Kept for environments where `rbf.pde.nodes.poisson_disc_nodes` doesn't
    accept `pinned_nodes`.  Uses periodic minimum-image distance in [0,1)^3.
    """
    accepted: list[Array] = []
    trials = 0
    while len(accepted) < n_target and trials < max_trials:
        trials += 1
        x = rng.random(3)
        ok = True
        for y in skeleton_frac:
            if _periodic_dist(x, y) < d_min:
                ok = False
                break
        if ok:
            for y in accepted:
                if _periodic_dist(x, y) < d_min:
                    ok = False
                    break
        if ok:
            accepted.append(x)
    if not accepted:
        return np.empty((0, 3), dtype=np.float64), trials
    return np.stack(accepted, axis=0), trials


# ─── Node quality metrics ────────────────────────────────────────────────────

def compute_node_quality(
    nodes: Array,
    bbox_min: Optional[Array] = None,
    bbox_max: Optional[Array] = None,
    probe_method: str = "uniform",
    n_probe: Optional[int] = None,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    Compute geometric quality metrics for a scattered node set X ⊂ Ω.

        q = ½ · min_{i≠j} ‖xᵢ − xⱼ‖        (node separation)
        h = sup_{x∈Ω} min_i ‖x − xᵢ‖       (fill radius / mesh norm)
        ρ = h / q                         (mesh ratio; → 1 is uniform)

    Parameters
    ----------
    nodes        : (N, 3) Cartesian
    bbox_min/max : axis-aligned probe domain Ω.  If None, taken as the nodes'
                   own bbox (+ no margin).
    probe_method : 'uniform' — dense uniform grid in Ω
                   'random'  — uniform random points in Ω
    n_probe      : number of probe points (only used for 'random';
                   the 'uniform' mode derives a cubic grid count from n_probe^(1/3)).
                   If None, auto-chosen ~ 32× interior nodes (cheap upper bound).

    Returns
    -------
    dict with q, h, rho, n_nodes, n_probe, probe_method,
    min_pair_dist (= 2q), domain_bbox_min/max
    """
    from scipy.spatial import cKDTree

    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[0] < 2:
        return {
            "q": float("nan"), "h": float("nan"), "rho": float("nan"),
            "n_nodes": int(nodes.shape[0] if nodes.size else 0),
            "n_probe": 0, "probe_method": probe_method,
            "min_pair_dist": float("nan"),
        }

    if bbox_min is None:
        bbox_min = nodes.min(axis=0)
    if bbox_max is None:
        bbox_max = nodes.max(axis=0)
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)

    # q: half the minimum pairwise distance (2nd-nearest neighbour via KDTree)
    tree = cKDTree(nodes)
    dists2, _ = tree.query(nodes, k=2)   # (N, 2): self + nearest neighbour
    min_pair = float(np.min(dists2[:, 1]))
    q = 0.5 * min_pair

    # h: approximate fill radius via probe points
    if n_probe is None:
        n_probe_target = max(4096, 32 * nodes.shape[0])
    else:
        n_probe_target = int(n_probe)

    if probe_method == "uniform":
        m = max(4, int(round(n_probe_target ** (1.0 / 3.0))))
        xs = np.linspace(bbox_min[0], bbox_max[0], m)
        ys = np.linspace(bbox_min[1], bbox_max[1], m)
        zs = np.linspace(bbox_min[2], bbox_max[2], m)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        probes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    elif probe_method == "random":
        rng = np.random.default_rng(random_seed)
        probes = rng.uniform(
            low=bbox_min, high=bbox_max, size=(n_probe_target, 3))
    else:
        raise ValueError(f"probe_method must be 'uniform' or 'random', got {probe_method!r}")

    probe_d, _ = tree.query(probes, k=1)
    h = float(np.max(probe_d))

    return {
        "q": q, "h": h, "rho": (h / q if q > 0 else float("inf")),
        "n_nodes": int(nodes.shape[0]),
        "n_probe": int(probes.shape[0]),
        "probe_method": probe_method,
        "min_pair_dist": min_pair,
        "domain_bbox_min": bbox_min.tolist(),
        "domain_bbox_max": bbox_max.tolist(),
    }


# ─── QD problem builder ──────────────────────────────────────────────────────

def build_qd_problem(
    cube_file: str,
    domain: str = "cube",
    spacing: float = 0.5,
    R: float = 20.0,
    stencil_size: int = 80,
    phi: str = "phs3",
    eps: float = 0.5,
    order: int = 2,
    sphere_subdivide: int = 3,
    augment: str = "poisson_disc",
    exclude_radius: float = 0.0,
    v_clip_percentile: float = 99.9,
    # conv_cell-only knobs
    conv_cell_a: float = 11.4523,
    conv_cell_d_min_frac: float = 0.06,
    conv_cell_n_random: int = 120,
    conv_cell_seed: int = 42,
    conv_cell_parity: bool = True,
    conv_cell_boundary_margin_frac: float = 0.06,
    conv_cell_use_rbf_poisson: bool = True,
    conv_cell_domain_shape: str = "cube",
    conv_cell_sphere_radius: Optional[float] = None,
    conv_cell_sphere_subdivide: int = 3,
    conv_cell_adaptive_random: bool = False,
    conv_cell_adaptive_grid_n: int = 36,
    conv_cell_adaptive_lambda_grad: float = 0.0,
    conv_cell_adaptive_lambda_lap: float = 0.0,
    conv_cell_adaptive_candidate_multiplier: float = 8.0,
    # ── V_nodes source ──
    # "grid_interp"     : linear-interpolate V from the cube-file grid to the
    #                     node positions (legacy, has interp error)
    # "gaussian_direct" : evaluate the Gaussian-fit analytic potential
    #                     Σ_atom Σ_g A·exp(-α|r-r_atom|²) directly at nodes,
    #                     using the same formula as the cube-grid builder.
    #                     No V-interpolation error.  Requires gaussian_params_file.
    v_source: str = "grid_interp",
    gaussian_params_file: Optional[str] = None,
    r_cut: float = 7.0,
) -> RBFProblem:
    """
    Build RBFProblem with QD potential from a Gaussian cube file.

    domain='cube'
        Use the exact regular grid from the cube file as nodes.
        Interior = all non-face points; boundary (Dirichlet=0) = face points.
        `spacing` and `R` are ignored.

    domain='sphere'
        Poisson disc nodes inside a sphere of radius R (default 20.0 Bohr).
        QD potential is interpolated to node positions via linear
        RegularGridInterpolator.  Points outside the cube extent get V=0.

    domain='atoms'
        Place nodes at every atom position read from the cube file, then
        (optionally) augment with Poisson-disc fill inside a sphere of radius R.
            augment='poisson_disc' : add Poisson-disc nodes (default)
            augment='none'         : use atom positions only (experimental)
        `exclude_radius` drops augmentation candidates within that distance
        of any atom (Bohr).

    domain='conv_cell'
        Conventional-cubic-cell hybrid template (zincblende atoms + level-3 FCC
        high-symmetry points + Poisson-disc random + parity counterparts),
        tiled to cover the cube-file box.  See `generate_conv_cell_nodes`.

    Parameters
    ----------
    cube_file        : path to Gaussian .cube file
    domain           : 'cube' | 'sphere' | 'atoms' | 'conv_cell'
    spacing          : Poisson-disc spacing (sphere / atoms domains)
    R                : sphere radius (Bohr) for sphere / atoms domains
    stencil_size     : RBF-FD stencil size
    phi              : RBF kernel name
    eps              : RBF shape parameter
    order            : polynomial augmentation order
    sphere_subdivide : icosphere subdivision count
    augment          : 'poisson_disc' | 'none' (atoms domain only)
    exclude_radius   : Bohr; Poisson candidates closer to any atom are dropped
    v_clip_percentile: clip V at this percentile to tame cube tail artefacts
    v_source         : 'grid_interp'     — linear interp of cube V to nodes (legacy)
                       'gaussian_direct' — direct analytic sum at nodes (no interp)
    gaussian_params_file : path to Gaussian-fit JSON; required for gaussian_direct
    r_cut            : Gaussian cutoff radius (Bohr) for gaussian_direct
    conv_cell_a              : lattice constant (Bohr, conv_cell only, default InAs 11.4523)
    conv_cell_d_min_frac     : d_min in fractional coords for the greedy filter (conv_cell)
    conv_cell_n_random       : target # of Poisson-like random points per cell
    conv_cell_seed           : RNG seed (conv_cell)
    conv_cell_parity         : include 1-r inversion counterparts (conv_cell)
    conv_cell_boundary_margin_frac : nodes within this fraction of `a` of a
                                     bbox face are flagged boundary (conv_cell)
    conv_cell_use_rbf_poisson: use rbf's poisson_disc_nodes (default) vs.
                               periodic rejection sampler
    conv_cell_domain_shape   : 'cube' (default) | 'sphere' (conv_cell domain)
    conv_cell_sphere_radius  : sphere radius (Bohr) when conv_cell_domain_shape='sphere';
                               None means 0.5*min(bbox side lengths)
    conv_cell_sphere_subdivide : icosphere subdivision for spherical boundary
    conv_cell_adaptive_random : if True, random template points are sampled by
                                dense-grid potential-adaptive accept/reject
    conv_cell_adaptive_grid_n : one-cell dense uniform grid resolution per axis
    conv_cell_adaptive_lambda_grad : λ1 in h = h_max/(1+λ1|∇V|+λ2|ΔV|)
    conv_cell_adaptive_lambda_lap  : λ2 in h = h_max/(1+λ1|∇V|+λ2|ΔV|)
    conv_cell_adaptive_candidate_multiplier : preselection budget multiplier
                                              before accept+KDTree filtering
    """
    from scipy.interpolate import RegularGridInterpolator

    x_grid, y_grid, z_grid, pot_3d = read_cube_file(cube_file)
    interp_fn = RegularGridInterpolator(
        (x_grid, y_grid, z_grid), pot_3d,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    if domain == "cube":
        nx, ny, nz = len(x_grid), len(y_grid), len(z_grid)
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        i_idx, j_idx, k_idx = np.unravel_index(np.arange(len(nodes)), (nx, ny, nz))
        on_boundary = (
            (i_idx == 0) | (i_idx == nx - 1) |
            (j_idx == 0) | (j_idx == ny - 1) |
            (k_idx == 0) | (k_idx == nz - 1)
        )
        interior_idx = np.where(~on_boundary)[0]
        groups = {
            "interior": interior_idx,
            "boundary": np.where(on_boundary)[0],
        }
        cfg_L = float(np.max(np.abs(nodes)))

    elif domain == "sphere":
        nodes, groups = generate_sphere_nodes(spacing, R, sphere_subdivide)
        interior_idx = groups["interior"]
        cfg_L = R

    elif domain == "atoms":
        atom_pos, _ = read_cube_atoms(cube_file)
        nodes, groups = generate_atom_augmented_nodes(
            atom_positions=atom_pos,
            domain="sphere",
            R=R,
            spacing=spacing,
            augment=augment,
            exclude_radius=exclude_radius,
            sphere_subdivide=sphere_subdivide,
        )
        interior_idx = groups["interior"]
        cfg_L = R

    elif domain == "conv_cell":
        # bbox from cube-file grid extents (orthorhombic assumed)
        bbox_min = np.array([x_grid[0], y_grid[0], z_grid[0]], dtype=np.float64)
        bbox_max = np.array([x_grid[-1] + (x_grid[1] - x_grid[0]),
                              y_grid[-1] + (y_grid[1] - y_grid[0]),
                              z_grid[-1] + (z_grid[1] - z_grid[0])],
                             dtype=np.float64)
        adaptive_builder = None
        if conv_cell_adaptive_random:
            if gaussian_params_file is None:
                raise ValueError(
                    "conv_cell_adaptive_random=True requires gaussian_params_file "
                    "(to evaluate one-cell dense-grid potential).")
            from gaussian_potential_builder import GaussianPotentialBuilder
            adaptive_builder = GaussianPotentialBuilder(
                cube_file=cube_file,
                params_file=gaussian_params_file,
                r_cut=r_cut,
            )
        nodes, groups, _cell_stats = generate_conv_cell_nodes(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            a=conv_cell_a,
            d_min_frac=conv_cell_d_min_frac,
            n_random_target=conv_cell_n_random,
            seed=conv_cell_seed,
            include_parity=conv_cell_parity,
            boundary_margin_frac=conv_cell_boundary_margin_frac,
            use_rbf_poisson=conv_cell_use_rbf_poisson,
            domain_shape=conv_cell_domain_shape,
            sphere_radius=conv_cell_sphere_radius,
            sphere_subdivide=conv_cell_sphere_subdivide,
            adaptive_random=conv_cell_adaptive_random,
            adaptive_grid_n=conv_cell_adaptive_grid_n,
            adaptive_lambda_grad=conv_cell_adaptive_lambda_grad,
            adaptive_lambda_lap=conv_cell_adaptive_lambda_lap,
            adaptive_candidate_multiplier=conv_cell_adaptive_candidate_multiplier,
            adaptive_gaussian_builder=adaptive_builder,
        )
        groups["conv_cell_template_nodes_frac"] = _cell_stats["cell_template_nodes_frac"]
        groups["conv_cell_template_nodes_cart"] = _cell_stats["cell_template_nodes_cart"]
        groups["conv_cell_template_roles"] = _cell_stats["cell_template_roles"]
        interior_idx = groups["interior"]
        cfg_L = float(np.max(np.abs(nodes))) if len(nodes) else 0.0

    else:
        raise ValueError(
            "domain must be 'cube', 'sphere', 'atoms', or 'conv_cell', "
            f"got {domain!r}")

    # ── QD potential on interior nodes ──────────────────────────────────────
    if v_source == "gaussian_direct":
        if gaussian_params_file is None:
            raise ValueError(
                "v_source='gaussian_direct' requires gaussian_params_file "
                "(path to the Gaussian-fit JSON, e.g. gaussian_fit_params.json)")
        from gaussian_potential_builder import GaussianPotentialBuilder
        builder = GaussianPotentialBuilder(
            cube_file=cube_file,
            params_file=gaussian_params_file,
            r_cut=r_cut,
        )
        V_nodes = builder.evaluate_at_points(nodes[interior_idx])
    elif v_source == "grid_interp":
        V_nodes = interp_fn(nodes[interior_idx]).astype(np.float64)
    else:
        raise ValueError(
            "v_source must be 'grid_interp' or 'gaussian_direct', "
            f"got {v_source!r}")

    # Clamp tail artefacts (mostly relevant for grid_interp; harmless for
    # gaussian_direct since the Gaussian fit is analytic and smooth)
    v_cap = float(np.percentile(V_nodes, v_clip_percentile))
    V_nodes = np.clip(V_nodes, None, v_cap)

    laplacian_matrix = weight_matrix(
        x=nodes[interior_idx],
        p=nodes,
        n=stencil_size,
        diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        phi=phi,
        eps=eps,
        order=order,
    )

    cfg = RBFConfig(
        spacing=spacing, L=cfg_L,
        stencil_size=stencil_size, phi=phi, eps=eps, order=order,
    )
    return RBFProblem(
        config=cfg,
        nodes=nodes,
        groups=groups,
        interior_idx=interior_idx,
        laplacian_matrix=laplacian_matrix,
        psi_interp_matrix=None,
        lap_interp_matrix=None,
        grid_points=None,
        grid_shape=None,
        V_nodes=V_nodes,
    )


# ─── Kernel sweep experiment ─────────────────────────────────────────────────

def _ho_exact_levels(n: int) -> np.ndarray:
    """First n exact 3D HO eigenvalues in ascending order (including degeneracy)."""
    levels: list[float] = []
    for s in range(100):
        e = s + 1.5
        deg = (s + 1) * (s + 2) // 2
        levels.extend([e] * deg)
        if len(levels) >= n:
            break
    return np.array(levels[:n], dtype=float)


def sweep_rbf_kernels(
    nodes: Array,
    groups: Dict[str, Array],
    stencil_size: int = 16,
    eps: float = 0.7,
    order: int = 0,
    n_eigs: int = 10,
    device: str = "cpu",
    symmetrize: bool = False,
    kernels: Optional[list] = None,
) -> list:
    """
    For each RBF kernel in `kernels`, rebuild only the Laplacian (nodes fixed),
    assemble H, solve n_eigs lowest eigenvalues, and compute mean relative
    spectral error against exact 3D HO levels.

    When symmetrize=False eigenvalues are complex; rel_err uses np.abs(val).
    Returns a list of result dicts sorted ascending by mean_rel_err.
    Failed kernels appear at the end with status='failed'.
    """
    if kernels is None:
        kernels = KERNELS_ALL

    interior_idx = groups["interior"]
    exact = _ho_exact_levels(n_eigs)
    results = []

    for phi in kernels:
        print(f"  {phi:<10}", end="  ", flush=True)
        try:
            lap = weight_matrix(
                x=nodes[interior_idx],
                p=nodes,
                n=stencil_size,
                diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                phi=phi,
                eps=eps,
                order=order,
            )
            prob = RBFProblem(
                config=RBFConfig(stencil_size=stencil_size, phi=phi,
                                 eps=eps, order=order),
                nodes=nodes,
                groups=groups,
                interior_idx=interior_idx,
                laplacian_matrix=lap,
                psi_interp_matrix=None,
                lap_interp_matrix=None,
                grid_points=None,
                grid_shape=None,
            )
            vals, _ = solve_lowest_eigenvalues(prob, n_eigs=n_eigs,
                                               device=device, symmetrize=symmetrize)
            # use |val| for complex case so rel_err is always real
            rel_errs = np.abs(np.abs(vals) - exact) / np.abs(exact)
            mean_rel_err = float(np.mean(rel_errs))
            max_imag = float(np.max(np.abs(np.imag(vals)))) if np.iscomplexobj(vals) else 0.0
            print(f"mean_rel_err={mean_rel_err:.4e}  max|Im|={max_imag:.2e}")
            results.append({
                "phi":          phi,
                "status":       "ok",
                "eigenvalues_re": [float(v.real) for v in vals],
                "eigenvalues_im": [float(v.imag) for v in vals] if np.iscomplexobj(vals) else [0.0]*len(vals),
                "exact":        [float(v) for v in exact],
                "rel_errs":     [float(v) for v in rel_errs],
                "mean_rel_err": mean_rel_err,
                "max_imag":     max_imag,
            })
        except Exception as exc:
            print(f"FAILED: {exc}")
            results.append({
                "phi":    phi,
                "status": "failed",
                "error":  str(exc),
            })

    return sorted(results, key=lambda r: r.get("mean_rel_err", float("inf")))


def sweep_stencil_eps(
    nodes: Array,
    groups: Dict[str, Array],
    phi: str = "ga",
    stencil_sizes: Optional[list] = None,
    eps_values: Optional[list] = None,
    order: int = 0,
    n_eigs: int = 10,
    device: str = "cpu",
    symmetrize: bool = False,
) -> list:
    """
    2D sweep over (stencil_size, eps) for a fixed RBF kernel (phi).
    Nodes are generated once and fixed; only the Laplacian is rebuilt per pair.

    Returns a flat list of result dicts (all pairs including failed ones),
    sorted ascending by mean_rel_err.  Each dict contains:
        stencil_size, eps, status, mean_rel_err, max_imag,
        eigenvalues_re, eigenvalues_im
    """
    if stencil_sizes is None:
        stencil_sizes = [8, 16, 32, 64, 128, 256]
    if eps_values is None:
        eps_values = [round(0.1 * i, 10) for i in range(1, 11)]

    interior_idx = groups["interior"]
    exact = _ho_exact_levels(n_eigs)
    results = []
    n_total = len(stencil_sizes) * len(eps_values)
    done = 0

    for n_st in stencil_sizes:
        for eps in eps_values:
            done += 1
            print(f"  [{done:3d}/{n_total}]  stencil={n_st:4d}  eps={eps:.2f}",
                  end="  ", flush=True)
            try:
                lap = weight_matrix(
                    x=nodes[interior_idx],
                    p=nodes,
                    n=n_st,
                    diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                    phi=phi,
                    eps=eps,
                    order=order,
                )
                prob = RBFProblem(
                    config=RBFConfig(stencil_size=n_st, phi=phi,
                                     eps=eps, order=order),
                    nodes=nodes,
                    groups=groups,
                    interior_idx=interior_idx,
                    laplacian_matrix=lap,
                    psi_interp_matrix=None,
                    lap_interp_matrix=None,
                    grid_points=None,
                    grid_shape=None,
                )
                vals, _ = solve_lowest_eigenvalues(
                    prob, n_eigs=n_eigs, device=device, symmetrize=symmetrize)
                rel_errs = np.abs(np.abs(vals) - exact) / np.abs(exact)
                mean_rel_err = float(np.mean(rel_errs))
                max_imag = (float(np.max(np.abs(np.imag(vals))))
                            if np.iscomplexobj(vals) else 0.0)
                print(f"mean_rel_err={mean_rel_err:.4e}  max|Im|={max_imag:.2e}")
                results.append({
                    "stencil_size":   n_st,
                    "eps":            float(eps),
                    "status":         "ok",
                    "mean_rel_err":   mean_rel_err,
                    "max_imag":       max_imag,
                    "eigenvalues_re": [float(v.real) for v in vals],
                    "eigenvalues_im": ([float(v.imag) for v in vals]
                                       if np.iscomplexobj(vals)
                                       else [0.0] * len(vals)),
                })
            except Exception as exc:
                print(f"FAILED: {exc}")
                results.append({
                    "stencil_size": n_st,
                    "eps":          float(eps),
                    "status":       "failed",
                    "error":        str(exc),
                    "mean_rel_err": float("inf"),
                })

    return sorted(results, key=lambda r: r.get("mean_rel_err", float("inf")))


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
