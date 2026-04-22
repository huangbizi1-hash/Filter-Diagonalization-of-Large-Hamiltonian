from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rbf.pde.fd import weight_matrix
from rbf.pde.nodes import poisson_disc_nodes


Array = np.ndarray

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
) -> RBFProblem:
    """
    Build RBFProblem with QD potential from a Gaussian cube file.

    domain='cube'
        Use the exact regular grid from the cube file as nodes.
        Interior = all non-face points; boundary (Dirichlet=0) = face points.
        `spacing` and `R` are ignored.

    domain='sphere'
        Poisson disc nodes inside a sphere of radius R (default 20.0 Bohr).
        QD potential is interpolated to node positions via linear RegularGridInterpolator.
        Points outside the cube file extent get V=0.

    Parameters
    ----------
    cube_file      : path to Gaussian .cube file
    domain         : 'cube' or 'sphere'
    spacing        : Poisson disc node spacing (sphere domain only)
    R              : sphere radius in Bohr (sphere domain only, default 20.0)
    stencil_size   : RBF-FD stencil size
    phi            : RBF kernel name
    eps            : RBF shape parameter
    order          : polynomial augmentation order
    sphere_subdivide: icosphere subdivision count (higher = smoother sphere)
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

    else:
        raise ValueError(f"domain must be 'cube' or 'sphere', got {domain!r}")

    # QD potential on interior nodes (clamp tail artefacts near nuclei)
    V_nodes = interp_fn(nodes[interior_idx]).astype(np.float64)
    v_cap = float(np.percentile(V_nodes, 99.9))
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
