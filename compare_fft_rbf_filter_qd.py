"""
compare_fft_rbf_filter_qd.py

在同一批随机初态上比较 FFT 与 RBF 稀疏矩阵（RBF-FD）滤波结果。
默认参数与用户请求一致：EL=-0.18, NC=500。

输出：
  filter_compare_results/fft_rbf_qd_R{R}_{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from filter_core import (
    PhysParams,
    apply_filter_H_all_op,
    build_filter_coefficients,
    make_filter_func,
    svd_rayleigh_ritz_op,
)
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid
from ho3d_solvers_v2 import build_3d_fft_operator
from rbf_core import RBFConfig, build_hamiltonian_matrix, build_problem, build_qd_problem


@dataclass
class CompareConfig:
    system: str = "qd"  # qd | ho
    qd_radius: int = 11
    el: float = -0.18
    nc: int = 500
    dE: float = 50.0
    Vmin: float = -5.0
    n_random: int = 16
    svd_tol: float = 1e-3
    max_energies: int = 30
    seed: int = 42

    rbf_stencil_size: int = 80
    rbf_phi: str = "phs3"
    rbf_eps: float = 0.5
    rbf_order: int = 2
    rbf_v_clip_percentile: float = 99.9
    rbf_domain: str = "cube"  # cube | sphere | atoms
    rbf_spacing: float = 0.8
    rbf_R: float = 20.0
    rbf_sphere_subdivide: int = 3
    rbf_augment: str = "poisson_disc"  # atoms domain only: poisson_disc | none
    rbf_exclude_radius: float = 0.0

    potential_cube_file: str = "localPot.cube"
    potential_params_file: str = "gaussian_fit_params.json"
    potential_r_cut: float = 7.0

    out_dir: str = "filter_compare_results"
    ho_N: int = 24
    ho_L: float = 8.0
    fft_kinetic_cut: float = 30.0
    power_iters: int = 30


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(np.real(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _read_cube_header(cube_path: Path):
    with cube_path.open("r", encoding="utf-8") as f:
        f.readline(); f.readline()
        parts3 = f.readline().split()
        origin = float(parts3[1])
        parts4 = f.readline().split()
        N = int(parts4[0])
        d = float(parts4[1])
    return N, d, origin


def _rayleigh(H_apply, psi: np.ndarray) -> float:
    hpsi = H_apply(psi)
    den = float(np.vdot(psi, psi).real)
    if not np.isfinite(den) or den <= 1e-30:
        return float("nan")
    val = float(np.vdot(psi, hpsi).real / den)
    return val if np.isfinite(val) else float("nan")


def _normalize_if_valid(v: np.ndarray, eps: float = 1e-30) -> tuple[np.ndarray, float, bool]:
    nrm = float(np.linalg.norm(v))
    if np.isfinite(nrm) and nrm > eps:
        vv = v / nrm
        if np.all(np.isfinite(vv)):
            return vv, nrm, True
    return v, nrm, False


def _power_method_energy(
    H_apply,
    n: int,
    rng: np.random.Generator,
    n_iter: int = 30,
) -> dict[str, Any]:
    """Estimate largest-eigenvalue energy via repeated H-apply + normalization."""
    v = rng.standard_normal(n)
    v, nrm0, ok0 = _normalize_if_valid(v)
    if not ok0:
        return {"ok": False, "reason": "invalid_initial_vector", "n_iter": 0}

    it_done = 0
    for _ in range(n_iter):
        w = H_apply(v)
        v, _, ok = _normalize_if_valid(w)
        if not ok:
            return {"ok": False, "reason": "invalid_iterate", "n_iter": it_done}
        it_done += 1

    E = _rayleigh(H_apply, v)
    if not np.isfinite(E):
        return {"ok": False, "reason": "non_finite_energy", "n_iter": it_done}
    return {"ok": True, "n_iter": it_done, "energy_estimate": float(E)}


def _make_qd_potential(cube_path: Path, cfg: CompareConfig) -> PotentialGrid:
    N, d_qd, origin_qd = _read_cube_header(cube_path)

    builder = GaussianPotentialBuilder(
        cfg.potential_cube_file,
        cfg.potential_params_file,
        cfg.potential_r_cut,
    )
    x0, y0, z0, V0 = builder.build_potential(64)

    interp = RegularGridInterpolator(
        (x0, y0, z0),
        V0.astype(float),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    x_new = origin_qd + np.arange(N) * d_qd
    X, Y, Z = np.meshgrid(x_new, x_new, x_new, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    V_new = interp(pts).reshape(N, N, N)

    return PotentialGrid(x_new, x_new, x_new, V_new, source=f"QD_R{cfg.qd_radius}")


def run(cfg: CompareConfig) -> Path:
    timings: dict[str, float] = {}
    qd_cube = None

    if cfg.system == "qd":
        qd_cube = Path(f"QD_Outputs/QD_R{cfg.qd_radius}.cube")
        if not qd_cube.exists():
            raise FileNotFoundError(
                f"未找到 {qd_cube}。请先运行 generate_QD_cubes.py 或指定已存在的 QD 半径。"
            )
        t0 = time.perf_counter()
        pot = _make_qd_potential(qd_cube, cfg)
        timings["build_qd_potential"] = time.perf_counter() - t0
        N = pot.Nx
        n_grid = N ** 3
    else:
        pot = None
        N = int(cfg.ho_N)
        n_grid = N ** 3

    dt = (cfg.nc / (cfg.dE * 2.5)) ** 2
    phys = PhysParams(dE=cfg.dE, Vmin=cfg.Vmin, dt=dt)

    t1 = time.perf_counter()
    filt_func = make_filter_func("gaussian", dt=dt)
    an, samp = build_filter_coefficients(
        np.array([cfg.el]),
        phys,
        cfg.nc,
        filter_func=filt_func,
        samp_method="ashkenazy",
        interpolation_tolerance=1e-6,
        enhance_step=10,
        max_enhance_iters=30,
    )
    timings["build_filter_coeff"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    H_fft, _, _ = build_3d_fft_operator(
        N=N,
        potential_grid=pot,
        L=cfg.ho_L,
        kinetic_cut=cfg.fft_kinetic_cut,
    )
    timings["build_fft_operator"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    if cfg.system == "qd":
        problem = build_qd_problem(
            cube_file=str(qd_cube),
            domain=cfg.rbf_domain,
            spacing=cfg.rbf_spacing,
            R=cfg.rbf_R,
            stencil_size=cfg.rbf_stencil_size,
            phi=cfg.rbf_phi,
            eps=cfg.rbf_eps,
            order=cfg.rbf_order,
            sphere_subdivide=cfg.rbf_sphere_subdivide,
            augment=cfg.rbf_augment,
            exclude_radius=cfg.rbf_exclude_radius,
            v_clip_percentile=cfg.rbf_v_clip_percentile,
        )
    else:
        problem = build_problem(
            RBFConfig(
                spacing=cfg.rbf_spacing,
                L=cfg.ho_L,
                stencil_size=cfg.rbf_stencil_size,
                phi=cfg.rbf_phi,
                eps=cfg.rbf_eps,
                order=cfg.rbf_order,
                grid_N=cfg.ho_N,
            ),
            build_interpolation=False,
        )
    H_rbf = build_hamiltonian_matrix(problem, symmetrize=True)
    H_rbf_op = spla.aslinearoperator(H_rbf)
    interior_idx = problem.interior_idx
    timings["build_rbf_operator"] = time.perf_counter() - t3

    rng = np.random.default_rng(cfg.seed)

    # --- 先做幂法：估计两种 H 的最大特征值对应能量 ---
    tpm = time.perf_counter()
    power_fft = _power_method_energy(H_fft.matvec, n_grid, rng, n_iter=cfg.power_iters)
    power_rbf = _power_method_energy(H_rbf_op.matvec, len(interior_idx), rng, n_iter=cfg.power_iters)
    timings["power_method"] = time.perf_counter() - tpm

    per_state = []
    fft_basis = []
    rbf_basis = []

    t4 = time.perf_counter()
    same_state_mode = (cfg.system == "qd" and cfg.rbf_domain == "cube")
    for i in range(cfg.n_random):
        psi_full = rng.standard_normal(n_grid)
        psi_full /= np.linalg.norm(psi_full)
        if same_state_mode:
            psi_int = psi_full[interior_idx]
            psi_int /= np.linalg.norm(psi_int)
        else:
            # 非 cube 节点域（sphere/atoms）与 FFT 网格自由度不同，
            # 无法做一一映射，因此用同一 RNG 下独立抽样保证可复现。
            psi_int = rng.standard_normal(len(interior_idx))
            psi_int /= np.linalg.norm(psi_int)

        fft_filt = apply_filter_H_all_op(H_fft.matvec, psi_full, samp, an, phys)[0]
        rbf_filt = apply_filter_H_all_op(H_rbf_op.matvec, psi_int, samp, an, phys)[0]

        fft_filt, norm_fft, valid_fft = _normalize_if_valid(fft_filt)
        rbf_filt, norm_rbf, valid_rbf = _normalize_if_valid(rbf_filt)

        E_fft = _rayleigh(H_fft.matvec, fft_filt)
        E_rbf = _rayleigh(H_rbf_op.matvec, rbf_filt)

        diff = (E_rbf - E_fft) if (np.isfinite(E_fft) and np.isfinite(E_rbf)) else float("nan")
        per_state.append(
            {
                "state_index": i,
                "filter_norm_fft": norm_fft,
                "filter_norm_rbf": norm_rbf,
                "energy_fft": E_fft,
                "energy_rbf": E_rbf,
                "abs_diff": abs(diff) if np.isfinite(diff) else float("nan"),
                "signed_diff": diff,
                "same_state_mode": same_state_mode,
                "valid_fft": bool(valid_fft and np.isfinite(E_fft)),
                "valid_rbf": bool(valid_rbf and np.isfinite(E_rbf)),
            }
        )
        if valid_fft and np.isfinite(E_fft):
            fft_basis.append(fft_filt)
        if valid_rbf and np.isfinite(E_rbf):
            rbf_basis.append(rbf_filt)

    timings["filter_states"] = time.perf_counter() - t4

    if fft_basis:
        fft_basis_mat = np.column_stack(fft_basis)
        t5 = time.perf_counter()
        evals_fft, _, rank_fft = svd_rayleigh_ritz_op(
            fft_basis_mat,
            H_fft.matvec,
            svd_tol=cfg.svd_tol,
            max_energies=cfg.max_energies,
            hermitian=True,
        )
        timings["rr_fft"] = time.perf_counter() - t5
    else:
        evals_fft = np.array([], dtype=float)
        rank_fft = 0
        timings["rr_fft"] = 0.0

    if rbf_basis:
        rbf_basis_mat = np.column_stack(rbf_basis)
        t6 = time.perf_counter()
        evals_rbf, _, rank_rbf = svd_rayleigh_ritz_op(
            rbf_basis_mat,
            H_rbf_op.matvec,
            svd_tol=cfg.svd_tol,
            max_energies=cfg.max_energies,
            hermitian=True,
        )
        timings["rr_rbf"] = time.perf_counter() - t6
    else:
        evals_rbf = np.array([], dtype=float)
        rank_rbf = 0
        timings["rr_rbf"] = 0.0

    k = min(len(evals_fft), len(evals_rbf))
    paired = []
    if k > 0:
        for i in range(k):
            paired.append(
                {
                    "level_index": i,
                    "fft": float(evals_fft[i]),
                    "rbf": float(evals_rbf[i]),
                    "abs_diff": float(abs(evals_fft[i] - evals_rbf[i])),
                    "signed_diff": float(evals_rbf[i] - evals_fft[i]),
                }
            )

    valid_state_diffs = [x["abs_diff"] for x in per_state if np.isfinite(x["abs_diff"])]
    avg_state_err = float(np.mean(valid_state_diffs)) if valid_state_diffs else None
    avg_eval_err = float(np.mean([x["abs_diff"] for x in paired])) if paired else None

    out = {
        "script": "compare_fft_rbf_filter_qd.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "grid": {
            "N": N,
            "N_grid": n_grid,
            "n_interior_rbf": int(len(interior_idx)),
            "qd_cube": str(qd_cube) if qd_cube is not None else None,
            "same_state_mode": same_state_mode,
            "system": cfg.system,
            "n_valid_fft_basis": len(fft_basis),
            "n_valid_rbf_basis": len(rbf_basis),
        },
        "filter": {
            "EL": cfg.el,
            "NC_input": cfg.nc,
            "NC_true": int(len(samp)),
            "dE": cfg.dE,
            "Vmin": cfg.Vmin,
            "dt": dt,
            "sigma": float(1.0 / np.sqrt(2.0 * dt)),
        },
        "power_method_max_energy_estimate": {
            "n_iter": int(cfg.power_iters),
            "fft": power_fft,
            "rbf": power_rbf,
        },
        "rr": {
            "rank_fft": int(rank_fft),
            "rank_rbf": int(rank_rbf),
            "evals_fft": [float(x) for x in evals_fft],
            "evals_rbf": [float(x) for x in evals_rbf],
            "paired_level_diffs": paired,
            "avg_abs_error_eigenvalues": avg_eval_err,
        },
        "per_state_filtered_energy": per_state,
        "summary": {
            "avg_abs_error_per_state_filtered_energy": avg_state_err,
            "avg_abs_error_eigenvalues": avg_eval_err,
            "max_abs_error_per_state_filtered_energy": float(np.max([x["abs_diff"] for x in per_state])) if per_state else None,
            "max_abs_error_eigenvalues": float(np.max([x["abs_diff"] for x in paired])) if paired else None,
        },
        "timings_sec": timings,
    }

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.system == "qd":
        out_name = f"fft_rbf_qd_R{cfg.qd_radius}_{ts}.json"
    else:
        out_name = f"fft_rbf_ho_N{cfg.ho_N}_{ts}.json"
    out_path = out_dir / out_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(out), f, ensure_ascii=False, indent=2)

    return out_path


def parse_args() -> CompareConfig:
    p = argparse.ArgumentParser(description="Compare FFT and RBF filter on QD/HO")
    p.add_argument("--system", type=str, default="qd", choices=["qd", "ho"])
    p.add_argument("--qd-radius", type=int, default=11)
    p.add_argument("--el", type=float, default=-0.18)
    p.add_argument("--nc", type=int, default=500)
    p.add_argument("--dE", type=float, default=50.0)
    p.add_argument("--Vmin", type=float, default=-5.0)
    p.add_argument("--n-random", type=int, default=16)
    p.add_argument("--svd-tol", type=float, default=1e-3)
    p.add_argument("--max-energies", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rbf-stencil-size", type=int, default=80)
    p.add_argument("--rbf-phi", type=str, default="phs3")
    p.add_argument("--rbf-eps", type=float, default=0.5)
    p.add_argument("--rbf-order", type=int, default=2)
    p.add_argument("--rbf-v-clip-percentile", type=float, default=99.9)
    p.add_argument("--rbf-domain", type=str, default="cube",
                   choices=["cube", "sphere", "atoms"])
    p.add_argument("--rbf-spacing", type=float, default=0.8)
    p.add_argument("--rbf-R", type=float, default=20.0)
    p.add_argument("--rbf-sphere-subdivide", type=int, default=3)
    p.add_argument("--rbf-augment", type=str, default="poisson_disc",
                   choices=["poisson_disc", "none"])
    p.add_argument("--rbf-exclude-radius", type=float, default=0.0)
    p.add_argument("--out-dir", type=str, default="filter_compare_results")
    p.add_argument("--ho-N", type=int, default=24,
                   help="HO mode only: FFT grid size per axis")
    p.add_argument("--ho-L", type=float, default=8.0,
                   help="HO mode only: box half-length")
    p.add_argument("--L", dest="ho_L", type=float, default=8.0,
                   help="Alias of --ho-L")
    p.add_argument("--fft-kinetic-cut", type=float, default=30.0,
                   help="FFT kinetic energy cutoff")
    p.add_argument("--power-iters", type=int, default=30,
                   help="Power-method iterations for max-eigenvalue estimate")

    a = p.parse_args()
    return CompareConfig(
        system=a.system,
        qd_radius=a.qd_radius,
        el=a.el,
        nc=a.nc,
        dE=a.dE,
        Vmin=a.Vmin,
        n_random=a.n_random,
        svd_tol=a.svd_tol,
        max_energies=a.max_energies,
        seed=a.seed,
        rbf_stencil_size=a.rbf_stencil_size,
        rbf_phi=a.rbf_phi,
        rbf_eps=a.rbf_eps,
        rbf_order=a.rbf_order,
        rbf_v_clip_percentile=a.rbf_v_clip_percentile,
        rbf_domain=a.rbf_domain,
        rbf_spacing=a.rbf_spacing,
        rbf_R=a.rbf_R,
        rbf_sphere_subdivide=a.rbf_sphere_subdivide,
        rbf_augment=a.rbf_augment,
        rbf_exclude_radius=a.rbf_exclude_radius,
        out_dir=a.out_dir,
        ho_N=a.ho_N,
        ho_L=a.ho_L,
        fft_kinetic_cut=a.fft_kinetic_cut,
        power_iters=a.power_iters,
    )


if __name__ == "__main__":
    cfg = parse_args()
    out_json = run(cfg)
    print(f"结果已保存: {out_json}")
