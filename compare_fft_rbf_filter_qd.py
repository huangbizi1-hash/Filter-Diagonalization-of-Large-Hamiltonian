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
from rbf_core import build_hamiltonian_matrix, build_qd_problem
from rbf_core import RBFConfig, build_problem


@dataclass
class CompareConfig:
    system: str = "qd"  # qd | ho
    qd_radius: int = 11
    ho_N: int = 16
    ho_L: float = 5.0
    el: float = -0.18
    nc: int = 500
    dE: float = 50.0
    Vmin: float = -5.0
    n_random: int = 16
    svd_tol: float = 1e-3
    max_energies: int = 30
    seed: int = 42

    rbf_spacing: float = 0.5
    rbf_stencil_size: int = 80
    rbf_phi: str = "phs3"
    rbf_eps: float = 0.5
    rbf_order: int = 2
    rbf_v_clip_percentile: float = 99.9

    potential_cube_file: str = "localPot.cube"
    potential_params_file: str = "gaussian_fit_params.json"
    potential_r_cut: float = 7.0

    out_dir: str = "filter_compare_results"
    power_steps: int = 30
    fft_kinetic_cut: float = 30.0


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
    return float(np.vdot(psi, hpsi).real / np.vdot(psi, psi).real)


def _power_method_energy(H_apply, psi0: np.ndarray, n_steps: int = 30) -> float:
    psi = np.asarray(psi0, dtype=float).copy()
    nrm = np.linalg.norm(psi)
    if nrm == 0:
        raise ValueError("power method 初始向量范数为 0")
    psi /= nrm

    for _ in range(n_steps):
        psi = H_apply(psi)
        nrm = np.linalg.norm(psi)
        if nrm == 0:
            raise RuntimeError("power method 迭代中向量范数变为 0")
        psi /= nrm
    return _rayleigh(H_apply, psi)


def _build_node_storage(problem) -> dict[str, Any]:
    nodes = np.asarray(problem.nodes, dtype=float)
    interior_idx = np.asarray(problem.interior_idx, dtype=int)
    interior_nodes = nodes[interior_idx]

    group_counts: dict[str, int] = {}
    for name, idx in problem.groups.items():
        group_counts[str(name)] = int(len(idx))

    return {
        "total_nodes": int(nodes.shape[0]),
        "interior_nodes": int(interior_nodes.shape[0]),
        "dimension": int(nodes.shape[1]) if nodes.ndim == 2 else None,
        "bbox_min": nodes.min(axis=0).tolist() if nodes.size else [],
        "bbox_max": nodes.max(axis=0).tolist() if nodes.size else [],
        "groups": group_counts,
        "metadata": {
            "stencil_size": int(problem.config.stencil_size),
            "phi": str(problem.config.phi),
            "eps": float(problem.config.eps),
            "order": int(problem.config.order),
        },
        "coordinates_all": nodes.tolist(),
        "coordinates_interior": interior_nodes.tolist(),
    }


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
    qd_cube: Path | None = None
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
    elif cfg.system == "ho":
        pot = None
        N = cfg.ho_N
    else:
        raise ValueError(f"--system 仅支持 qd 或 ho，收到: {cfg.system!r}")

    n_grid = N**3

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
        N=N, potential_grid=pot, L=cfg.ho_L, kinetic_cut=cfg.fft_kinetic_cut
    )
    timings["build_fft_operator"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    if cfg.system == "qd":
        assert qd_cube is not None
        problem = build_qd_problem(
            cube_file=str(qd_cube),
            domain="cube",
            stencil_size=cfg.rbf_stencil_size,
            phi=cfg.rbf_phi,
            eps=cfg.rbf_eps,
            order=cfg.rbf_order,
            v_clip_percentile=cfg.rbf_v_clip_percentile,
        )
    else:
        rbf_cfg = RBFConfig(
            spacing=cfg.rbf_spacing,
            L=cfg.ho_L,
            stencil_size=cfg.rbf_stencil_size,
            phi=cfg.rbf_phi,
            eps=cfg.rbf_eps,
            order=cfg.rbf_order,
        )
        problem = build_problem(config=rbf_cfg, build_interpolation=False)
    H_rbf = build_hamiltonian_matrix(problem, symmetrize=True)
    H_rbf_op = spla.aslinearoperator(H_rbf)
    interior_idx = problem.interior_idx
    node_storage = _build_node_storage(problem)
    timings["build_rbf_operator"] = time.perf_counter() - t3

    # FFT 和 RBF 生活在完全不同的向量空间：
    #   FFT: 均匀网格，维度 n_grid = N³
    #   RBF: 散点节点，维度 n_interior（与 n_grid 无关）
    # 必须为两者各自独立生成随机向量，不能用 psi_full[interior_idx] 互相索引。
    n_interior = int(len(interior_idx))
    rng_fft = np.random.default_rng(cfg.seed)
    rng_rbf = np.random.default_rng(cfg.seed + 1)

    t_power = time.perf_counter()
    psi_power_fft = rng_fft.standard_normal(n_grid)
    psi_power_rbf = rng_rbf.standard_normal(n_interior)
    Emax_fft = _power_method_energy(H_fft.matvec, psi_power_fft, n_steps=cfg.power_steps)
    Emax_rbf = _power_method_energy(H_rbf_op.matvec, psi_power_rbf, n_steps=cfg.power_steps)
    timings["power_method"] = time.perf_counter() - t_power

    per_state = []
    fft_basis = []
    rbf_basis = []

    t4 = time.perf_counter()
    for i in range(cfg.n_random):
        psi_full = rng_fft.standard_normal(n_grid)
        psi_full /= np.linalg.norm(psi_full)
        psi_int = rng_rbf.standard_normal(n_interior)
        psi_int /= np.linalg.norm(psi_int)

        fft_filt = apply_filter_H_all_op(H_fft.matvec, psi_full, samp, an, phys)[0]
        rbf_filt = apply_filter_H_all_op(H_rbf_op.matvec, psi_int, samp, an, phys)[0]

        norm_fft = float(np.linalg.norm(fft_filt))
        norm_rbf = float(np.linalg.norm(rbf_filt))
        if norm_fft > 0:
            fft_filt = fft_filt / norm_fft
        if norm_rbf > 0:
            rbf_filt = rbf_filt / norm_rbf

        E_fft = _rayleigh(H_fft.matvec, fft_filt)
        E_rbf = _rayleigh(H_rbf_op.matvec, rbf_filt)

        per_state.append(
            {
                "state_index": i,
                "filter_norm_fft": norm_fft,
                "filter_norm_rbf": norm_rbf,
                "energy_fft": E_fft,
                "energy_rbf": E_rbf,
                "abs_diff": abs(E_fft - E_rbf),
                "signed_diff": E_rbf - E_fft,
            }
        )

        fft_basis.append(fft_filt)
        rbf_basis.append(rbf_filt)

    timings["filter_states"] = time.perf_counter() - t4

    fft_basis_mat = np.column_stack(fft_basis)
    rbf_basis_mat = np.column_stack(rbf_basis)

    t5 = time.perf_counter()
    evals_fft, _, rank_fft = svd_rayleigh_ritz_op(
        fft_basis_mat,
        H_fft.matvec,
        svd_tol=cfg.svd_tol,
        max_energies=cfg.max_energies,
        hermitian=True,
    )
    timings["rr_fft"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    evals_rbf, _, rank_rbf = svd_rayleigh_ritz_op(
        rbf_basis_mat,
        H_rbf_op.matvec,
        svd_tol=cfg.svd_tol,
        max_energies=cfg.max_energies,
        hermitian=True,
    )
    timings["rr_rbf"] = time.perf_counter() - t6

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

    avg_state_err = float(np.mean([x["abs_diff"] for x in per_state])) if per_state else None
    avg_eval_err = float(np.mean([x["abs_diff"] for x in paired])) if paired else None

    out = {
        "script": "compare_fft_rbf_filter_qd.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "grid": {
            "N_fft": N,
            "N_grid_fft": n_grid,
            "n_interior_rbf": n_interior,
            "qd_cube": str(qd_cube) if qd_cube is not None else None,
        },
        "rbf_nodes": node_storage,
        "filter": {
            "EL": cfg.el,
            "NC_input": cfg.nc,
            "NC_true": int(len(samp)),
            "dE": cfg.dE,
            "Vmin": cfg.Vmin,
            "dt": dt,
            "sigma": float(1.0 / np.sqrt(2.0 * dt)),
        },
        "power_method": {
            "steps": int(cfg.power_steps),
            "max_energy_fft": float(Emax_fft),
            "max_energy_rbf": float(Emax_rbf),
            "abs_diff": float(abs(Emax_fft - Emax_rbf)),
            "signed_diff": float(Emax_rbf - Emax_fft),
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
    out_path = out_dir / f"fft_rbf_qd_R{cfg.qd_radius}_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(out), f, ensure_ascii=False, indent=2)

    return out_path


def parse_args() -> CompareConfig:
    p = argparse.ArgumentParser(description="Compare FFT and RBF filter on QD")
    p.add_argument("--system", type=str, choices=["qd", "ho"], default="qd")
    p.add_argument("--qd-radius", type=int, default=11)
    p.add_argument("--ho-N", type=int, default=16)
    p.add_argument("--ho-L", type=float, default=5.0)
    p.add_argument("--el", type=float, default=-0.18)
    p.add_argument("--nc", type=int, default=500)
    p.add_argument("--dE", type=float, default=50.0)
    p.add_argument("--Vmin", type=float, default=-5.0)
    p.add_argument("--n-random", type=int, default=16)
    p.add_argument("--svd-tol", type=float, default=1e-3)
    p.add_argument("--max-energies", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rbf-spacing", type=float, default=0.5)
    p.add_argument("--rbf-stencil-size", type=int, default=80)
    p.add_argument("--rbf-phi", type=str, default="phs3")
    p.add_argument("--rbf-eps", type=float, default=0.5)
    p.add_argument("--rbf-order", type=int, default=2)
    p.add_argument("--rbf-v-clip-percentile", type=float, default=99.9)
    p.add_argument("--power-steps", type=int, default=30)
    p.add_argument("--fft-kinetic-cut", type=float, default=30.0)
    p.add_argument("--out-dir", type=str, default="filter_compare_results")

    a = p.parse_args()
    return CompareConfig(
        system=a.system,
        qd_radius=a.qd_radius,
        ho_N=a.ho_N,
        ho_L=a.ho_L,
        el=a.el,
        nc=a.nc,
        dE=a.dE,
        Vmin=a.Vmin,
        n_random=a.n_random,
        svd_tol=a.svd_tol,
        max_energies=a.max_energies,
        seed=a.seed,
        rbf_spacing=a.rbf_spacing,
        rbf_stencil_size=a.rbf_stencil_size,
        rbf_phi=a.rbf_phi,
        rbf_eps=a.rbf_eps,
        rbf_order=a.rbf_order,
        rbf_v_clip_percentile=a.rbf_v_clip_percentile,
        power_steps=a.power_steps,
        fft_kinetic_cut=a.fft_kinetic_cut,
        out_dir=a.out_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    out_json = run(cfg)
    print(f"结果已保存: {out_json}")
