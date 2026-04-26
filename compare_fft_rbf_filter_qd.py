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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator

from filter_core import (
    PhysParams,
    build_filter_coefficients,
    make_filter_func,
    svd_rayleigh_ritz_op,
)
from gaussian_potential_builder import GaussianPotentialBuilder, PotentialGrid
from ho3d_solvers_v2 import build_3d_fft_operator
from rbf_core import (
    RBFConfig,
    RBFProblem,
    build_hamiltonian_matrix,
    build_problem,
    build_qd_problem,
    compute_node_quality,
    read_cube_file,
)


@dataclass
class CompareConfig:
    system: str = "qd"  # qd | ho
    qd_radius: int = 11
    ho_N: int = 16
    ho_L: float = 5.0
    el: float = -0.18
    el_list: list = field(default_factory=list)   # 多滤波中心；非空时覆盖 el
    nc: int = 500
    dE: float = 50.0
    Vmin: float = -5.0
    n_random: int = 16
    svd_tol: float = 1e-3
    max_energies: int = 30
    seed: int = 42
    psi_init_type: str = "sine"      # sine | gaussian
    psi_kmax: float = 3.0            # |k| 上界（仅 sine 模式）

    rbf_spacing: float = 0.5
    rbf_stencil_size: int = 80
    rbf_stencil_radius: float = 0.0
    rbf_stencil_fingerprint_tol: float = 1e-4
    rbf_stencil_inner_radius: float = 0.0
    rbf_stencil_max_neighbors: int = 0
    rbf_stencil_select_near_first: bool = True
    rbf_phi: str = "phs3"
    rbf_eps: float = 0.5
    rbf_order: int = 2
    rbf_v_clip_percentile: float = 99.9

    # ── QD 节点放置方式（QD 模式下生效） ───────────────────────────────────
    # cube       : cube 文件自带的规则网格（默认，节点均匀分布）
    # sphere     : 球内 Poisson-disc 节点（非均匀，推荐与 RBF-FD 搭配）
    # atoms      : 原子位置做节点 + 可选 Poisson-disc 加密
    # conv_cell  : 惯用晶胞混合模板 (atoms+level-3 FCC+Poisson+parity) × 平铺
    rbf_node_method: str = "cube"
    rbf_R: float = 20.0                    # sphere / atoms 的球半径（Bohr）
    rbf_sphere_subdivide: int = 3           # 球面 icosphere 细分级数
    rbf_augment: str = "poisson_disc"       # atoms 模式：poisson_disc | none
    rbf_exclude_radius: float = 0.0         # atoms 模式：原子排斥半径（Bohr）

    # conv_cell 专用参数（仅 --rbf-node-method conv_cell 生效）
    conv_cell_a: float = 11.4523            # 晶格常数（Bohr），InAs zincblende 默认
    conv_cell_d_min_frac: float = 0.06      # fractional 坐标下的 greedy d_min
    conv_cell_n_random: int = 120           # 每个晶胞 Poisson-like 目标点数
    conv_cell_seed: int = 42
    conv_cell_parity: bool = True           # 是否加 1-r 反演对偶点
    conv_cell_template_mode: str = "hybrid" # hybrid | fcc_refined
    conv_cell_fcc_scale_factor: int = 8
    conv_cell_fcc_origin_frac: tuple[float, float, float] = (0.0, 0.0, 0.0)
    conv_cell_fcc_atom_refine_factor: int = 0
    conv_cell_fcc_atom_radius_frac: float = 0.0
    # 面附近多近算 boundary：薄壳（≈1·d_min）才合理；以前 0.5 太大，会让中心
    # 立方只剩 ~36% 体积，视觉上像没铺满。默认 = d_min_frac。
    conv_cell_boundary_margin_frac: float = 0.06
    conv_cell_use_rbf_poisson: bool = True  # True=rbf.poisson_disc_nodes, False=周期拒绝采样
    conv_cell_domain_shape: str = "cube"    # conv_cell 体域形状：cube | sphere
    conv_cell_sphere_radius: float = 0.0    # >0 时使用该球半径(Bohr)；<=0 自动取 bbox 内切球
    conv_cell_sphere_subdivide: int = 3     # sphere 边界 icosphere 细分级数
    conv_cell_adaptive_random: bool = False
    conv_cell_adaptive_grid_n: int = 36
    conv_cell_adaptive_lambda_grad: float = 0.0
    conv_cell_adaptive_lambda_lap: float = 0.0
    conv_cell_adaptive_candidate_multiplier: float = 8.0
    conv_cell_include_skeleton: bool = True
    include_interior: bool = True
    include_boundary: bool = True
    node_min_dist: float = 0.0

    # ── 节点质量度量 ───────────────────────────────────────────────────────
    quality_probe_method: str = "uniform"   # 'uniform' 或 'random'
    quality_probe_n: int = 0                # 0 = 自动（~32× n_nodes）

    # ── 节点持久化 ─────────────────────────────────────────────────────────
    save_nodes: str = ""   # 非空则把节点单独写入 NumPy 数据文件（.npz）
    load_nodes: str = ""   # 非空则从已存节点文件（.npz/.json）加载节点，跳过节点生成；节点上的
                           #   laplacian / V_nodes 仍然会按 CLI 的 --rbf-phi / eps /
                           #   order / stencil-size / v-clip 等参数重新计算

    # ── 可选：带插值的对照 Ritz（用于量化"插值到均匀格点"带来的误差） ──────
    # 当前默认的 RBF Ritz 是直接在非均匀 interior 节点空间里做的，不含任何
    # 波函数插值。启用此开关时，会额外：
    #   (a) 用 RBF 权重矩阵构造 n_grid × n_interior 的插值算子 P，
    #   (b) 把 RBF 滤波基 Φ_rbf 插到 FFT 均匀格点：Φ_interp = P @ Φ_rbf，
    #   (c) 用 FFT 的 H 在插值后基上做 Ritz，得到 evals_rbf_interp。
    # evals_rbf（原有，无插值）与 evals_rbf_interp（新增，含插值）之差即可
    # 用来定量评估插值的影响。启用会增加一次 weight_matrix 构造 + 一次 Ritz。
    rbf_interp_ritz: bool = False

    potential_cube_file: str = "localPot.cube"
    potential_params_file: str = "gaussian_fit_params.json"
    potential_r_cut: float = 7.0
    # RBF 节点上 V 的来源：
    #   'gaussian_direct' — 直接对节点调用 GaussianPotentialBuilder.evaluate_at_points
    #                       （与 cube 网格 V 共用同一套高斯展开公式，零插值误差）
    #   'grid_interp'     — 从 cube 网格线性插值到节点（老路径，有插值误差）
    rbf_v_source: str = "gaussian_direct"

    out_dir: str = "filter_compare_results"
    power_steps: int = 30
    fft_kinetic_cut: float = 30.0
    fft_only: bool = False
    filter_norm_blowup_threshold: float = 1e200


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
    denom = float(np.vdot(psi, psi).real)
    if (not np.isfinite(denom)) or denom <= 0.0:
        return float("nan")
    num = float(np.vdot(psi, hpsi).real)
    return float(num / denom)


def _finite_stats(name: str, arr: np.ndarray) -> dict[str, Any]:
    """Return finite/non-finite diagnostics for a vector or matrix."""
    a = np.asarray(arr)
    finite_mask = np.isfinite(a)
    n_total = int(a.size)
    n_finite = int(np.count_nonzero(finite_mask))
    stats: dict[str, Any] = {
        "name": name,
        "shape": list(a.shape),
        "n_total": n_total,
        "n_finite": n_finite,
        "n_nonfinite": n_total - n_finite,
        "all_finite": bool(np.all(finite_mask)),
    }
    if n_finite > 0:
        a_f = a[finite_mask]
        stats.update({
            "min": float(np.min(a_f)),
            "max": float(np.max(a_f)),
            "mean_abs": float(np.mean(np.abs(a_f))),
        })
    return stats


def _first_nonfinite_detail(arr: np.ndarray) -> dict[str, Any] | None:
    a = np.asarray(arr)
    bad = np.argwhere(~np.isfinite(a))
    if bad.size == 0:
        return None
    idx = tuple(int(i) for i in bad[0])
    return {"index": list(idx), "value": float(np.real(a[idx]))}


def _trace_matvec_nonfinite(
    H_apply,
    psi0: np.ndarray,
    n_steps: int,
    stage_name: str,
) -> dict[str, Any]:
    """
    Probe repeated H-apply stability to localize where NaN/Inf first appears.
    """
    psi = np.asarray(psi0, dtype=float).copy()
    trace: dict[str, Any] = {
        "stage": stage_name,
        "input_stats": _finite_stats(f"{stage_name}_psi0", psi),
        "first_nonfinite_step": None,
        "first_nonfinite_component": None,
    }
    if not trace["input_stats"]["all_finite"]:
        trace["first_nonfinite_step"] = 0
        trace["first_nonfinite_component"] = _first_nonfinite_detail(psi)
        return trace

    for k in range(1, max(1, int(n_steps)) + 1):
        psi = H_apply(psi)
        if not np.all(np.isfinite(psi)):
            trace["first_nonfinite_step"] = k
            trace["first_nonfinite_component"] = _first_nonfinite_detail(psi)
            trace["step_stats"] = _finite_stats(f"{stage_name}_after_{k}_matvec", psi)
            break
        nrm = np.linalg.norm(psi)
        if (not np.isfinite(nrm)) or nrm == 0.0:
            trace["first_nonfinite_step"] = k
            trace["first_nonfinite_component"] = {"norm": float(nrm)}
            break
        psi /= nrm

    return trace


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


def _apply_filter_with_blowup_guard(
    H_apply,
    psi: np.ndarray,
    nodes: np.ndarray,
    an: np.ndarray,
    par: PhysParams,
    norm_blowup_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply filter with per-step norm diagnostics.

    Returns (results, diag), where diag["blowup_step"] is the H-apply count
    when norms first become non-finite / zero / exceed threshold.
    """
    ms, nc = an.shape
    idx = (slice(None),) + (None,) * psi.ndim
    results = an[:, 0][idx] * psi[None, ...]

    psi_prev = psi.copy()
    diag: dict[str, Any] = {
        "blowup_step": None,
        "reason": None,
        "psi_curr_norm": None,
        "result_norm": None,
    }

    for j in range(1, nc):
        H_psi = H_apply(psi_prev)
        psi_curr = ((4.0 / par.dE) * (H_psi - par.Vmin * psi_prev)
                    - 2.0 * psi_prev
                    - nodes[j - 1] * psi_prev)
        results = results + an[:, j][idx] * psi_curr[None, ...]

        psi_curr_norm = float(np.linalg.norm(psi_curr))
        result_norm = float(np.linalg.norm(results[0]))
        if (not np.isfinite(psi_curr_norm)) or (not np.isfinite(result_norm)):
            diag.update({
                "blowup_step": j,
                "reason": "nonfinite_norm",
                "psi_curr_norm": psi_curr_norm,
                "result_norm": result_norm,
            })
            break
        if psi_curr_norm == 0.0:
            diag.update({
                "blowup_step": j,
                "reason": "zero_norm",
                "psi_curr_norm": psi_curr_norm,
                "result_norm": result_norm,
            })
            break
        if (psi_curr_norm >= norm_blowup_threshold) or (result_norm >= norm_blowup_threshold):
            diag.update({
                "blowup_step": j,
                "reason": "norm_threshold_exceeded",
                "psi_curr_norm": psi_curr_norm,
                "result_norm": result_norm,
            })
            break

        psi_prev = psi_curr

    return results, diag


def _build_node_storage(
    problem,
    provenance: dict | None = None,
    quality_probe_method: str = "uniform",
    quality_probe_n: int = 0,
) -> dict[str, Any]:
    """Serialise a RBFProblem's node layout to a JSON-compatible dict.

    provenance: optional dict recording how the nodes were generated
                (domain / spacing / R / augment / exclude_radius / cube_file …).
                Included as 'source' so the JSON can later be reloaded without
                re-running the generator.

    Node quality metrics (q, h, rho) are computed on the interior node set
    over the bbox of the whole node set; see rbf_core.compute_node_quality.
    """
    nodes = np.asarray(problem.nodes, dtype=float)
    interior_idx = np.asarray(problem.interior_idx, dtype=int)
    interior_nodes = nodes[interior_idx]

    groups_full: dict[str, list[int]] = {}
    group_counts: dict[str, int] = {}
    aux_groups: dict[str, Any] = {}
    for name, idx in problem.groups.items():
        arr = np.asarray(idx)
        name_s = str(name)
        # 仅把“索引分组”放进 groups（兼容 --load-nodes）；其他辅助数组单独存 aux
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
            arr_i = arr.astype(int, copy=False)
            groups_full[name_s] = arr_i.tolist()
            group_counts[name_s] = int(len(arr_i))
        else:
            aux_groups[name_s] = arr

    # q, h, ρ on interior nodes (what the Hamiltonian actually sees)
    quality: dict[str, Any] = {}
    if interior_nodes.shape[0] >= 2:
        quality = compute_node_quality(
            interior_nodes,
            bbox_min=nodes.min(axis=0) if nodes.size else None,
            bbox_max=nodes.max(axis=0) if nodes.size else None,
            probe_method=quality_probe_method,
            n_probe=(None if quality_probe_n <= 0 else quality_probe_n),
        )

    return {
        "total_nodes": int(nodes.shape[0]),
        "interior_nodes": int(interior_nodes.shape[0]),
        "dimension": int(nodes.shape[1]) if nodes.ndim == 2 else None,
        "bbox_min": nodes.min(axis=0).tolist() if nodes.size else [],
        "bbox_max": nodes.max(axis=0).tolist() if nodes.size else [],
        "group_counts": group_counts,
        "metadata": {
            "stencil_size": int(problem.config.stencil_size),
            "phi": str(problem.config.phi),
            "eps": float(problem.config.eps),
            "order": int(problem.config.order),
        },
        "quality": quality,     # {q, h, rho, ...}
        "source": provenance or {},
        # 完整坐标 + 分组索引 → 足以重建 RBFProblem 节点布局
        "coordinates_all": nodes.tolist(),
        "coordinates_interior": interior_nodes.tolist(),
        "groups": groups_full,
        "aux_groups": _to_jsonable(aux_groups),
    }


def _save_conv_cell_template_npz(problem: RBFProblem, nodes_data_path: Path) -> Path | None:
    nodes_frac = problem.groups.get("conv_cell_template_nodes_frac")
    nodes_cart = problem.groups.get("conv_cell_template_nodes_cart")
    roles = problem.groups.get("conv_cell_template_roles")
    if nodes_frac is None or nodes_cart is None or roles is None:
        return None

    nodes_frac_arr = np.asarray(nodes_frac, dtype=np.float64)
    nodes_cart_arr = np.asarray(nodes_cart, dtype=np.float64)
    roles_arr = np.asarray(roles, dtype=np.int64)
    atoms_cart_arr = nodes_cart_arr[roles_arr == 0]
    out = nodes_data_path.with_name(nodes_data_path.stem + "_conv_cell_template.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        atom_coordinates_cart=atoms_cart_arr,
        node_coordinates_cart=nodes_cart_arr,
        node_coordinates_frac=nodes_frac_arr,
        node_roles=roles_arr,
    )
    return out


def _save_nodes_npz(node_storage: dict[str, Any], path: Path) -> Path:
    coords_all = np.asarray(node_storage["coordinates_all"], dtype=np.float64)
    coords_interior = np.asarray(node_storage["coordinates_interior"], dtype=np.float64)
    groups = node_storage.get("groups", {})
    interior_idx = np.asarray(groups.get("interior", []), dtype=np.int64)
    boundary_idx = np.asarray(groups.get("boundary", []), dtype=np.int64)
    coords_boundary = (
        coords_all[boundary_idx] if boundary_idx.size > 0 else np.empty((0, 3), dtype=np.float64)
    )
    metadata = {
        "total_nodes": int(node_storage.get("total_nodes", coords_all.shape[0])),
        "interior_nodes": int(node_storage.get("interior_nodes", coords_interior.shape[0])),
        "group_counts": node_storage.get("group_counts", {}),
        "metadata": node_storage.get("metadata", {}),
        "quality": node_storage.get("quality", {}),
        "source": node_storage.get("source", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coordinates_all=coords_all,
        coordinates_interior=coords_interior,
        coordinates_boundary=coords_boundary,
        interior_idx=interior_idx,
        boundary_idx=boundary_idx,
        metadata_json=np.array(json.dumps(_to_jsonable(metadata), ensure_ascii=False)),
    )
    return path


def _node_summary_for_result(node_storage: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_nodes": int(node_storage.get("total_nodes", 0)),
        "interior_nodes": int(node_storage.get("interior_nodes", 0)),
        "dimension": node_storage.get("dimension"),
        "bbox_min": node_storage.get("bbox_min", []),
        "bbox_max": node_storage.get("bbox_max", []),
        "group_counts": node_storage.get("group_counts", {}),
        "quality": node_storage.get("quality", {}),
        "source": node_storage.get("source", {}),
    }


def _fft_grid_points(pot: PotentialGrid | None, N: int, ho_L: float) -> np.ndarray:
    """返回 FFT 所用均匀格点的 3D 坐标，shape (N³, 3)。

    QD 模式 (pot 非 None)：直接用 pot.x / pot.y / pot.z
    HO 模式 (pot is None)：按 build_3d_fft_operator 的约定
                           d = 2L/N，x1d = (arange(N) - N/2) * d
    """
    if pot is not None:
        x1d, y1d, z1d = pot.x, pot.y, pot.z
    else:
        d = 2.0 * ho_L / N
        x1d = (np.arange(N) - N / 2) * d
        y1d = x1d.copy()
        z1d = x1d.copy()
    X, Y, Z = np.meshgrid(x1d, y1d, z1d, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def _sine_psi_on_pts(coords: np.ndarray, K_max: float, rng: np.random.Generator) -> np.ndarray:
    """Normalized random sine wave on an arbitrary 3-D point cloud.

    Draws (kx, ky, kz) ~ Uniform[-K_max, K_max]^3 and phase b ~ Uniform[-π, π],
    then evaluates psi[i] = sin(kx*x[i] + ky*y[i] + kz*z[i] + b) and normalises.
    Returns a real flat array of shape (N,).
    """
    kx, ky, kz = rng.uniform(-K_max, K_max, 3)
    b = rng.uniform(-np.pi, np.pi)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    psi = np.sin(kx * x + ky * y + kz * z + b)
    nrm = np.linalg.norm(psi)
    return psi / nrm if nrm > 0 else psi


def _build_problem_from_nodes_file(path: Path, cfg: "CompareConfig",
                                    cube_path: Path) -> tuple[RBFProblem, dict[str, Any]]:
    """Reconstruct a RBFProblem from a saved nodes file (.npz/.json).

    Node positions and group assignments are loaded from the file as-is.
    The RBF-FD Laplacian and V_nodes are recomputed using the **current** CLI
    parameters (stencil_size / phi / eps / order / v_clip_percentile / v_source),
    so you can reuse the same node layout while sweeping RBF kernels.
    """
    from rbf.pde.fd import weight_matrix

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            nodes = np.asarray(data["coordinates_all"], dtype=np.float64)
            interior_idx = np.asarray(data["interior_idx"], dtype=np.int64)
            boundary_idx = np.asarray(data["boundary_idx"], dtype=np.int64)
            groups = {
                "interior": interior_idx,
                "boundary": boundary_idx,
            }
            saved = {
                "coordinates_all": nodes.tolist(),
                "groups": {
                    "interior": interior_idx.tolist(),
                    "boundary": boundary_idx.tolist(),
                },
            }
    else:
        with path.open("r", encoding="utf-8") as f:
            saved = json.load(f)

        nodes = np.asarray(saved["coordinates_all"], dtype=float)
        groups = {str(k): np.asarray(v, dtype=int) for k, v in saved["groups"].items()}
        if "interior" not in groups:
            raise ValueError(
                f"saved nodes file {path} missing 'interior' group")
        interior_idx = groups["interior"]

    # V_nodes: honour cfg.rbf_v_source (same semantics as build_qd_problem)
    if cfg.rbf_v_source == "gaussian_direct":
        from gaussian_potential_builder import GaussianPotentialBuilder
        builder = GaussianPotentialBuilder(
            cube_file=str(cube_path),
            params_file=cfg.potential_params_file,
            r_cut=cfg.potential_r_cut,
        )
        V_nodes = builder.evaluate_at_points(nodes[interior_idx])
    else:
        from scipy.interpolate import RegularGridInterpolator
        x_cube, y_cube, z_cube, pot_3d = read_cube_file(str(cube_path))
        interp = RegularGridInterpolator(
            (x_cube, y_cube, z_cube), pot_3d,
            method="linear", bounds_error=False, fill_value=0.0,
        )
        V_nodes = interp(nodes[interior_idx]).astype(np.float64)
    v_cap = float(np.percentile(V_nodes, cfg.rbf_v_clip_percentile))
    V_nodes = np.clip(V_nodes, None, v_cap)

    # Recompute the RBF-FD Laplacian from the saved geometry
    laplacian_matrix = weight_matrix(
        x=nodes[interior_idx],
        p=nodes,
        n=cfg.rbf_stencil_size,
        diffs=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        phi=cfg.rbf_phi,
        eps=cfg.rbf_eps,
        order=cfg.rbf_order,
    )
    rbf_cfg = RBFConfig(
        spacing=cfg.rbf_spacing,
        L=float(np.max(np.abs(nodes))) if nodes.size else 0.0,
        stencil_size=cfg.rbf_stencil_size,
        phi=cfg.rbf_phi,
        eps=cfg.rbf_eps,
        order=cfg.rbf_order,
    )
    problem = RBFProblem(
        config=rbf_cfg,
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
    return problem, saved


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

    El_array = np.array(cfg.el_list, dtype=float) if cfg.el_list else np.array([cfg.el], dtype=float)
    ms = len(El_array)
    if ms > 1:
        print(f"[filter] El_list ({ms} centers): {El_array.tolist()}")

    t1 = time.perf_counter()
    filt_func = make_filter_func("gaussian", dt=dt)
    an, samp = build_filter_coefficients(
        El_array,
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

    provenance: dict[str, Any] = {}
    problem: RBFProblem | None = None
    node_storage: dict[str, Any] = {}
    nodes_data_path: Path | None = None
    n_interior = 0
    H_rbf_op = None
    h_rbf_matrix_stats: dict[str, Any] | None = None
    rbf_min_eig_info: dict[str, Any] | None = None
    interior_idx = np.array([], dtype=np.int64)
    if not cfg.fft_only:
        t3 = time.perf_counter()
        provenance = {
            "system":          cfg.system,
            "qd_radius":       cfg.qd_radius if cfg.system == "qd" else None,
            "qd_cube":         str(qd_cube) if qd_cube is not None else None,
            "node_method":     cfg.rbf_node_method,
            "rbf_spacing":     cfg.rbf_spacing,
            "rbf_R":           cfg.rbf_R,
            "rbf_sphere_subdivide": cfg.rbf_sphere_subdivide,
            "rbf_augment":     cfg.rbf_augment,
            "rbf_exclude_radius": cfg.rbf_exclude_radius,
            "conv_cell_a":     cfg.conv_cell_a,
            "conv_cell_d_min_frac": cfg.conv_cell_d_min_frac,
            "conv_cell_n_random":   cfg.conv_cell_n_random,
            "conv_cell_seed":       cfg.conv_cell_seed,
            "conv_cell_parity":     cfg.conv_cell_parity,
            "conv_cell_template_mode": cfg.conv_cell_template_mode,
            "conv_cell_fcc_scale_factor": cfg.conv_cell_fcc_scale_factor,
            "conv_cell_fcc_origin_frac": list(cfg.conv_cell_fcc_origin_frac),
            "conv_cell_fcc_atom_refine_factor": cfg.conv_cell_fcc_atom_refine_factor,
            "conv_cell_fcc_atom_radius_frac": cfg.conv_cell_fcc_atom_radius_frac,
            "conv_cell_boundary_margin_frac": cfg.conv_cell_boundary_margin_frac,
            "conv_cell_use_rbf_poisson":      cfg.conv_cell_use_rbf_poisson,
            "conv_cell_domain_shape":         cfg.conv_cell_domain_shape,
            "conv_cell_sphere_radius":        cfg.conv_cell_sphere_radius,
            "conv_cell_sphere_subdivide":     cfg.conv_cell_sphere_subdivide,
            "conv_cell_adaptive_random":      cfg.conv_cell_adaptive_random,
            "conv_cell_adaptive_grid_n":      cfg.conv_cell_adaptive_grid_n,
            "conv_cell_adaptive_lambda_grad": cfg.conv_cell_adaptive_lambda_grad,
            "conv_cell_adaptive_lambda_lap":  cfg.conv_cell_adaptive_lambda_lap,
            "conv_cell_adaptive_candidate_multiplier": cfg.conv_cell_adaptive_candidate_multiplier,
            "conv_cell_include_skeleton": cfg.conv_cell_include_skeleton,
            "rbf_stencil_radius": cfg.rbf_stencil_radius,
            "rbf_stencil_fingerprint_tol": cfg.rbf_stencil_fingerprint_tol,
            "rbf_stencil_inner_radius": cfg.rbf_stencil_inner_radius,
            "rbf_stencil_max_neighbors": cfg.rbf_stencil_max_neighbors,
            "rbf_stencil_select_near_first": cfg.rbf_stencil_select_near_first,
            "include_interior": cfg.include_interior,
            "include_boundary": cfg.include_boundary,
            "node_min_dist": cfg.node_min_dist,
            "ho_L":            cfg.ho_L,
            "loaded_from":     cfg.load_nodes or None,
        }

        if cfg.load_nodes:
            # 从已存节点文件重建 RBFProblem（Laplacian + V 按当前 CLI 重新算）
            if cfg.system != "qd" or qd_cube is None:
                raise ValueError("--load-nodes 目前只支持 --system qd（需要 cube 文件做 V 插值）")
            load_path = Path(cfg.load_nodes)
            if not load_path.exists():
                raise FileNotFoundError(f"--load-nodes 指定的文件不存在: {load_path}")
            problem, _saved_meta = _build_problem_from_nodes_file(load_path, cfg, qd_cube)
            provenance["loaded_from"] = str(load_path)
        elif cfg.system == "qd":
            assert qd_cube is not None
            if cfg.rbf_node_method not in ("cube", "sphere", "atoms", "conv_cell"):
                raise ValueError(
                    "--rbf-node-method 仅支持 cube | sphere | atoms | conv_cell，"
                    f"收到 {cfg.rbf_node_method!r}"
                )
            problem = build_qd_problem(
                cube_file=str(qd_cube),
                domain=cfg.rbf_node_method,
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
                conv_cell_a=cfg.conv_cell_a,
                conv_cell_d_min_frac=cfg.conv_cell_d_min_frac,
                conv_cell_n_random=cfg.conv_cell_n_random,
                conv_cell_seed=cfg.conv_cell_seed,
                conv_cell_parity=cfg.conv_cell_parity,
                conv_cell_template_mode=cfg.conv_cell_template_mode,
                conv_cell_fcc_scale_factor=cfg.conv_cell_fcc_scale_factor,
                conv_cell_fcc_origin_frac=np.asarray(cfg.conv_cell_fcc_origin_frac, dtype=np.float64),
                conv_cell_fcc_atom_refine_factor=cfg.conv_cell_fcc_atom_refine_factor,
                conv_cell_fcc_atom_radius_frac=cfg.conv_cell_fcc_atom_radius_frac,
                conv_cell_boundary_margin_frac=cfg.conv_cell_boundary_margin_frac,
                conv_cell_use_rbf_poisson=cfg.conv_cell_use_rbf_poisson,
                conv_cell_domain_shape=cfg.conv_cell_domain_shape,
                conv_cell_sphere_radius=(
                    cfg.conv_cell_sphere_radius
                    if cfg.conv_cell_sphere_radius > 0.0 else None
                ),
                conv_cell_sphere_subdivide=cfg.conv_cell_sphere_subdivide,
                conv_cell_adaptive_random=cfg.conv_cell_adaptive_random,
                conv_cell_adaptive_grid_n=cfg.conv_cell_adaptive_grid_n,
                conv_cell_adaptive_lambda_grad=cfg.conv_cell_adaptive_lambda_grad,
                conv_cell_adaptive_lambda_lap=cfg.conv_cell_adaptive_lambda_lap,
                conv_cell_adaptive_candidate_multiplier=cfg.conv_cell_adaptive_candidate_multiplier,
                conv_cell_include_skeleton=cfg.conv_cell_include_skeleton,
                stencil_radius=cfg.rbf_stencil_radius,
                stencil_fingerprint_tol=cfg.rbf_stencil_fingerprint_tol,
                stencil_inner_radius=cfg.rbf_stencil_inner_radius,
                stencil_max_neighbors=cfg.rbf_stencil_max_neighbors,
                stencil_select_near_first=cfg.rbf_stencil_select_near_first,
                include_interior=cfg.include_interior,
                include_boundary=cfg.include_boundary,
                node_min_dist=cfg.node_min_dist,
                v_source=cfg.rbf_v_source,
                gaussian_params_file=(cfg.potential_params_file
                                      if cfg.rbf_v_source == "gaussian_direct"
                                      else None),
                r_cut=cfg.potential_r_cut,
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
        h_rbf_matrix_stats = _finite_stats("H_rbf.data", H_rbf.data)
        if not h_rbf_matrix_stats["all_finite"]:
            bad_detail = _first_nonfinite_detail(H_rbf.data)
            raise RuntimeError(
                "RBF Hamiltonian contains NaN/Inf values. "
                f"first_bad={bad_detail}. "
                "Try adjusting node/stencil parameters (e.g. --rbf-stencil-size, "
                "--rbf-eps, --conv-cell-d-min-frac) to improve conditioning."
            )
        t_min_eval = time.perf_counter()
        try:
            eig_min_arr = spla.eigs(
                H_rbf,
                k=1,
                which="SR",
                return_eigenvectors=False,
                tol=1e-8,
                maxiter=max(2000, 5 * H_rbf.shape[0]),
            )
            eig_min = complex(eig_min_arr[0])
            rbf_min_eig_info = {
                "success": True,
                "which": "SR",
                "value": {
                    "real": float(np.real(eig_min)),
                    "imag": float(np.imag(eig_min)),
                },
                "abs": float(np.abs(eig_min)),
            }
            print("[rbf] 稀疏矩阵最小特征值(复数, which='SR') = "
                  f"{eig_min.real:+.10e} {eig_min.imag:+.10e}j")
        except Exception as exc:
            rbf_min_eig_info = {
                "success": False,
                "which": "SR",
                "error": str(exc),
            }
            print(f"[rbf] 警告：最小特征值计算失败（which='SR'）：{exc}")
        timings["rbf_min_eigenvalue"] = time.perf_counter() - t_min_eval
        H_rbf_op = spla.aslinearoperator(H_rbf)
        interior_idx = problem.interior_idx
        n_interior = int(len(interior_idx))
        node_storage = _build_node_storage(
            problem,
            provenance=provenance,
            quality_probe_method=cfg.quality_probe_method,
            quality_probe_n=cfg.quality_probe_n,
        )
        q_info = node_storage.get("quality", {})
        if q_info:
            print(f"[node quality] q={q_info.get('q', float('nan')):.4e}  "
                  f"h={q_info.get('h', float('nan')):.4e}  "
                  f"ρ={q_info.get('rho', float('nan')):.3f}  "
                  f"(n_probe={q_info.get('n_probe', 0)}, "
                  f"method={q_info.get('probe_method', '')})")
        timings["build_rbf_operator"] = time.perf_counter() - t3

    # 按需单独落盘节点数据（便于后续 --load-nodes 复用，不依赖结果 JSON）
    ts_nodes = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.save_nodes and (not cfg.fft_only):
        # 用户给了路径——如果是目录则自动命名，否则当作完整文件路径
        p_user = Path(cfg.save_nodes)
        if p_user.suffix == "" or p_user.is_dir():
            p_user.mkdir(parents=True, exist_ok=True)
            tag = (f"qd_R{cfg.qd_radius}" if cfg.system == "qd"
                   else f"ho_N{cfg.ho_N}")
            nodes_data_path = p_user / f"nodes_{tag}_{cfg.rbf_node_method}_{ts_nodes}.npz"
        else:
            nodes_data_path = p_user if p_user.suffix.lower() == ".npz" else p_user.with_suffix(".npz")
        _save_nodes_npz(node_storage, nodes_data_path)
        print(f"节点数据文件已保存: {nodes_data_path}")
        if cfg.rbf_node_method == "conv_cell":
            template_path = _save_conv_cell_template_npz(problem, nodes_data_path)
            if template_path is not None:
                print(f"原胞模板节点已保存: {template_path}")

    # FFT 和 RBF 生活在完全不同的向量空间：
    #   FFT: 均匀网格，维度 n_grid = N³
    #   RBF: 散点节点，维度 n_interior（与 n_grid 无关）
    # 必须为两者各自独立生成随机向量，不能用 psi_full[interior_idx] 互相索引。
    rng_fft = np.random.default_rng(cfg.seed)
    rng_rbf = np.random.default_rng(cfg.seed + 1) if not cfg.fft_only else None

    # 正弦初态所需的 3D 坐标（sine 模式下提前计算，避免循环内重复）
    _use_sine = (cfg.psi_init_type == "sine")
    fft_pts_sine: np.ndarray | None = None
    rbf_pts_sine: np.ndarray | None = None
    if _use_sine:
        fft_pts_sine = _fft_grid_points(pot, N, cfg.ho_L)   # (N³, 3)
        if not cfg.fft_only and H_rbf_op is not None and problem is not None:
            rbf_pts_sine = problem.nodes[interior_idx]       # (n_interior, 3)
        print(f"[psi init] type=sine  K_max={cfg.psi_kmax}")
    else:
        print(f"[psi init] type=gaussian")

    t_power = time.perf_counter()
    psi_power_fft = rng_fft.standard_normal(n_grid)
    Emax_fft = _power_method_energy(H_fft.matvec, psi_power_fft, n_steps=cfg.power_steps)
    Emax_rbf = None
    if not cfg.fft_only and rng_rbf is not None and H_rbf_op is not None:
        psi_power_rbf = rng_rbf.standard_normal(n_interior)
        Emax_rbf = _power_method_energy(H_rbf_op.matvec, psi_power_rbf, n_steps=cfg.power_steps)
    timings["power_method"] = time.perf_counter() - t_power

    per_state = []
    fft_basis = []
    rbf_basis = []
    rbf_invalid_states: list[dict[str, Any]] = []

    t4 = time.perf_counter()
    for i in range(cfg.n_random):
        if _use_sine:
            psi_full = _sine_psi_on_pts(fft_pts_sine, cfg.psi_kmax, rng_fft)
        else:
            psi_full = rng_fft.standard_normal(n_grid)
            psi_full /= np.linalg.norm(psi_full)
        fft_filt_all, fft_filter_diag = _apply_filter_with_blowup_guard(
            H_apply=H_fft.matvec,
            psi=psi_full,
            nodes=samp,
            an=an,
            par=phys,
            norm_blowup_threshold=cfg.filter_norm_blowup_threshold,
        )
        # fft_filt_all: (ms, n_grid)

        psi_int = None
        rbf_filt_all = None
        rbf_filter_diag = None
        if not cfg.fft_only and rng_rbf is not None and H_rbf_op is not None:
            if _use_sine and rbf_pts_sine is not None:
                psi_int = _sine_psi_on_pts(rbf_pts_sine, cfg.psi_kmax, rng_rbf)
            else:
                psi_int = rng_rbf.standard_normal(n_interior)
                psi_int /= np.linalg.norm(psi_int)
            rbf_filt_all, rbf_filter_diag = _apply_filter_with_blowup_guard(
                H_apply=H_rbf_op.matvec,
                psi=psi_int,
                nodes=samp,
                an=an,
                par=phys,
                norm_blowup_threshold=cfg.filter_norm_blowup_threshold,
            )
            # rbf_filt_all: (ms, n_interior)
            if rbf_filter_diag["blowup_step"] is not None:
                print(
                    "[RBF filter blowup] "
                    f"state={i}, H_apply_count={rbf_filter_diag['blowup_step']}, "
                    f"reason={rbf_filter_diag['reason']}, "
                    f"psi_curr_norm={rbf_filter_diag['psi_curr_norm']}, "
                    f"result_norm={rbf_filter_diag['result_norm']}"
                )

        for ie in range(ms):
            el_ctr = float(El_array[ie])

            fft_filt = fft_filt_all[ie].copy()
            norm_fft = float(np.linalg.norm(fft_filt))
            if norm_fft > 0:
                fft_filt = fft_filt / norm_fft
            E_fft = _rayleigh(H_fft.matvec, fft_filt)
            fft_basis.append(fft_filt)

            norm_rbf = None
            E_rbf = None
            if rbf_filt_all is not None:
                rbf_filt = rbf_filt_all[ie].copy()
                norm_rbf = float(np.linalg.norm(rbf_filt))
                is_valid_rbf = np.isfinite(norm_rbf) and (norm_rbf > 0.0) and np.all(np.isfinite(rbf_filt))
                if is_valid_rbf:
                    rbf_filt = rbf_filt / norm_rbf
                    E_rbf = _rayleigh(H_rbf_op.matvec, rbf_filt)
                    if np.isfinite(E_rbf):
                        rbf_basis.append(rbf_filt)
                    else:
                        rbf_invalid_states.append({
                            "state_index": i,
                            "el_idx": ie,
                            "el_center": el_ctr,
                            "reason": "rayleigh_nonfinite",
                            "filter_norm_rbf": norm_rbf,
                            "filter_stats": _finite_stats("rbf_filt", rbf_filt),
                            "nonfinite_detail": _first_nonfinite_detail(rbf_filt),
                            "filter_diag": rbf_filter_diag,
                            "matvec_trace": _trace_matvec_nonfinite(
                                H_rbf_op.matvec, psi_int, n_steps=min(8, cfg.nc), stage_name="rbf_filter_input"
                            ),
                        })
                        E_rbf = None
                else:
                    rbf_invalid_states.append({
                        "state_index": i,
                        "el_idx": ie,
                        "el_center": el_ctr,
                        "reason": "filter_output_nonfinite_or_zero_norm",
                        "filter_norm_rbf": norm_rbf,
                        "filter_stats": _finite_stats("rbf_filt", rbf_filt),
                        "nonfinite_detail": _first_nonfinite_detail(rbf_filt),
                        "filter_diag": rbf_filter_diag,
                        "matvec_trace": _trace_matvec_nonfinite(
                            H_rbf_op.matvec, psi_int, n_steps=min(8, cfg.nc), stage_name="rbf_filter_input"
                        ),
                    })
                    E_rbf = None

            per_state.append({
                "state_index": i,
                "el_idx": ie,
                "el_center": el_ctr,
                "filter_norm_fft": norm_fft,
                "filter_norm_rbf": norm_rbf,
                "filter_diag_fft": fft_filter_diag,
                "filter_diag_rbf": rbf_filter_diag,
                "energy_fft": E_fft,
                "energy_rbf": E_rbf,
                "abs_diff": abs(E_fft - E_rbf) if E_rbf is not None else None,
                "signed_diff": (E_rbf - E_fft) if E_rbf is not None else None,
            })

    timings["filter_states"] = time.perf_counter() - t4

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

    evals_rbf: np.ndarray = np.array([], dtype=np.float64)
    rank_rbf = 0
    if not cfg.fft_only and H_rbf_op is not None and len(rbf_basis) > 0:
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
    elif not cfg.fft_only and H_rbf_op is not None:
        timings["rr_rbf"] = 0.0

    # ── 可选：带插值的对照 Ritz ─────────────────────────────────────────────
    # 把 RBF 滤波基从 n_interior 非均匀节点插到 FFT 均匀格点 (n_grid)，
    # 再用 FFT 的 H 在插值后基上做 Ritz。供对比"插值 vs 不插值"。
    evals_rbf_interp: np.ndarray = np.array([], dtype=np.float64)
    rank_rbf_interp = 0
    t_interp_build = 0.0
    t_rr_rbf_interp = 0.0
    per_state_interp: list[dict[str, Any]] = []

    if cfg.rbf_interp_ritz and (not cfg.fft_only):
        t_ib0 = time.perf_counter()
        from rbf.pde.fd import weight_matrix as _weight_matrix_for_interp

        fft_pts = _fft_grid_points(pot, N, cfg.ho_L)          # (N³, 3)
        interior_pts = problem.nodes[problem.interior_idx]    # (n_interior, 3)
        # RBF 插值算子：diffs=[0,0,0] 即在点上直接求值
        P_interp = _weight_matrix_for_interp(
            x=fft_pts,
            p=interior_pts,
            n=cfg.rbf_stencil_size,
            diffs=[0, 0, 0],
            phi=cfg.rbf_phi,
            eps=cfg.rbf_eps,
            order=cfg.rbf_order,
        )
        t_interp_build = time.perf_counter() - t_ib0
        timings["build_interp_matrix"] = t_interp_build

        # 插值整个基：(n_grid, n_interior) @ (n_interior, n_random) → (n_grid, n_random)
        rbf_basis_interp = np.asarray(P_interp @ rbf_basis_mat)

        # 每个插值后态在 H_fft 下的 Rayleigh 能量（便于定量对比）
        for i in range(rbf_basis_interp.shape[1]):
            psi_i = rbf_basis_interp[:, i]
            nrm_i = float(np.linalg.norm(psi_i))
            E_interp = (_rayleigh(H_fft.matvec, psi_i / nrm_i)
                        if nrm_i > 0 else float("nan"))
            per_state_interp.append({
                "state_index": i,
                "norm_after_interp": nrm_i,
                "energy_rbf_interp": E_interp,
                "energy_fft":        per_state[i]["energy_fft"],
                "energy_rbf_nodes":  per_state[i]["energy_rbf"],
                "abs_diff_vs_fft":   abs(E_interp - per_state[i]["energy_fft"]),
                "abs_diff_vs_rbf_nodes": abs(E_interp - per_state[i]["energy_rbf"]),
            })

        t_rr0 = time.perf_counter()
        evals_rbf_interp, _, rank_rbf_interp = svd_rayleigh_ritz_op(
            rbf_basis_interp,
            H_fft.matvec,
            svd_tol=cfg.svd_tol,
            max_energies=cfg.max_energies,
            hermitian=True,
        )
        t_rr_rbf_interp = time.perf_counter() - t_rr0
        timings["rr_rbf_interp"] = t_rr_rbf_interp

    # ── 配对差 ──────────────────────────────────────────────────────────────
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

    # rbf_nodes (无插值) vs rbf_interp (插到均匀格点后 Ritz)：
    # 两者都是"同一份滤波态"的 Ritz 结果，差异 = 插值的净影响
    paired_rbf_interp_vs_nodes: list[dict[str, Any]] = []
    paired_interp_vs_fft:       list[dict[str, Any]] = []
    if len(evals_rbf_interp) > 0:
        k2 = min(len(evals_rbf_interp), len(evals_rbf))
        for i in range(k2):
            paired_rbf_interp_vs_nodes.append({
                "level_index": i,
                "rbf_nodes":  float(evals_rbf[i]),
                "rbf_interp": float(evals_rbf_interp[i]),
                "abs_diff":   float(abs(evals_rbf_interp[i] - evals_rbf[i])),
                "signed_diff": float(evals_rbf_interp[i] - evals_rbf[i]),
            })
        k3 = min(len(evals_rbf_interp), len(evals_fft))
        for i in range(k3):
            paired_interp_vs_fft.append({
                "level_index": i,
                "fft":        float(evals_fft[i]),
                "rbf_interp": float(evals_rbf_interp[i]),
                "abs_diff":   float(abs(evals_rbf_interp[i] - evals_fft[i])),
                "signed_diff": float(evals_rbf_interp[i] - evals_fft[i]),
            })

    state_abs_diffs = [x["abs_diff"] for x in per_state if x["abs_diff"] is not None]
    avg_state_err = float(np.mean(state_abs_diffs)) if state_abs_diffs else None
    avg_eval_err = float(np.mean([x["abs_diff"] for x in paired])) if paired else None
    avg_interp_eff = (
        float(np.mean([x["abs_diff"] for x in paired_rbf_interp_vs_nodes]))
        if paired_rbf_interp_vs_nodes else None)

    out = {
        "script": "compare_fft_rbf_filter_qd.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "grid": {
            "N_fft": N,
            "N_grid_fft": n_grid,
            "n_interior_rbf": n_interior if not cfg.fft_only else None,
            "qd_cube": str(qd_cube) if qd_cube is not None else None,
        },
        # 结果 JSON 仅保留节点元数据；完整坐标在独立 .npz 文件中（如有）
        "rbf_nodes": _node_summary_for_result(node_storage),
        "rbf_nodes_file": str(nodes_data_path) if nodes_data_path else None,
        "filter": {
            "EL": cfg.el,
            "El_list": El_array.tolist(),
            "n_el_centers": ms,
            "NC_input": cfg.nc,
            "NC_true": int(len(samp)),
            "dE": cfg.dE,
            "Vmin": cfg.Vmin,
            "dt": dt,
            "sigma": float(1.0 / np.sqrt(2.0 * dt)),
            "psi_init_type": cfg.psi_init_type,
            "psi_kmax": cfg.psi_kmax,
        },
        "power_method": {
            "steps": int(cfg.power_steps),
            "max_energy_fft": float(Emax_fft),
            "max_energy_rbf": float(Emax_rbf) if Emax_rbf is not None else None,
            "abs_diff": float(abs(Emax_fft - Emax_rbf)) if Emax_rbf is not None else None,
            "signed_diff": float(Emax_rbf - Emax_fft) if Emax_rbf is not None else None,
        },
        "rr": {
            # evals_rbf 始终是"直接在非均匀节点上 Ritz"的结果（无波函数插值）
            # evals_rbf_interp 仅在 --rbf-interp-ritz 时存在：把 RBF 基插到 FFT
            # 均匀格点后用 FFT 的 H 做 Ritz，用来量化插值本身的影响
            "rank_fft": int(rank_fft),
            "rank_rbf_nodes": int(rank_rbf),
            "rank_rbf_interp": int(rank_rbf_interp),
            "evals_fft": [float(x) for x in evals_fft],
            "evals_rbf": [float(x) for x in evals_rbf],                     # no interp
            "evals_rbf_nodes": [float(x) for x in evals_rbf],               # alias
            "evals_rbf_interp": [float(x) for x in evals_rbf_interp],       # with interp
            "paired_level_diffs": paired,                                   # fft vs rbf_nodes
            "paired_level_diffs_rbf_interp_vs_nodes": paired_rbf_interp_vs_nodes,
            "paired_level_diffs_rbf_interp_vs_fft":   paired_interp_vs_fft,
            "avg_abs_error_eigenvalues": avg_eval_err,
            "avg_abs_error_interp_effect": avg_interp_eff,   # ← 插值的净影响
            "n_valid_rbf_basis": int(len(rbf_basis)),
            "n_invalid_rbf_basis": int(len(rbf_invalid_states)),
        },
        "per_state_filtered_energy": per_state,
        "rbf_invalid_states": rbf_invalid_states,
        "per_state_filtered_energy_interp": per_state_interp,
        "summary": {
            "avg_abs_error_per_state_filtered_energy": avg_state_err,
            "avg_abs_error_eigenvalues": avg_eval_err,
            "max_abs_error_per_state_filtered_energy": float(np.max(state_abs_diffs)) if state_abs_diffs else None,
            "max_abs_error_eigenvalues": float(np.max([x["abs_diff"] for x in paired])) if paired else None,
            "avg_abs_error_interp_effect_eigenvalues": avg_interp_eff,
            "max_abs_error_interp_effect_eigenvalues": (
                float(np.max([x["abs_diff"] for x in paired_rbf_interp_vs_nodes]))
                if paired_rbf_interp_vs_nodes else None),
        },
        "timings_sec": timings,
    }
    if h_rbf_matrix_stats is not None:
        out["rbf_operator"] = {"matrix_data_stats": h_rbf_matrix_stats}
        if rbf_min_eig_info is not None:
            out["rbf_operator"]["min_eigenvalue_complex"] = rbf_min_eig_info

    if cfg.rbf_node_method == "conv_cell" and not cfg.fft_only and not cfg.load_nodes:
        grp = problem.groups
        enh = {
            "fcc_base_nodes":            int(grp.get("_enh1_fcc_base", 0)),
            "enh1_atom_refine_added":    int(grp.get("_enh1_added", 0)),
            "enh2_adaptive_random_added": int(grp.get("_enh2_added", 0)),
            "fcc_local_tiled":           int(len(grp.get("fcc_local", []))),
        }
        out["conv_cell_enhancement"] = enh
        if enh["enh1_atom_refine_added"] > 0 or enh["enh2_adaptive_random_added"] > 0:
            print(
                f"[conv_cell enhancement] "
                f"fcc_base={enh['fcc_base_nodes']}  "
                f"enh1_fine={enh['enh1_atom_refine_added']}  "
                f"enh2_rand={enh['enh2_adaptive_random_added']}  "
                f"fcc_local_tiled={enh['fcc_local_tiled']}"
            )

    if cfg.rbf_stencil_radius > 0.0 and not cfg.fft_only and not cfg.load_nodes:
        _ball_stats = problem.groups.get("_ball_stencil_stats")
        if _ball_stats is not None:
            out["stencil_neighbor_stats"] = _ball_stats
            # Pretty console summary
            bc = _ball_stats["neighbor_bins"]
            bp = _ball_stats["neighbor_bins_pct"]
            print(
                f"[stencil] outer={_ball_stats['outer_radius_bohr']:.3f} Bohr"
                + (f"  inner={_ball_stats['inner_radius_bohr']:.3f}" if _ball_stats["inner_radius_bohr"] > 0 else "")
                + (f"  max_cap={_ball_stats['max_neighbors_cap']}({'near' if _ball_stats['select_near_first'] else 'far'})"
                   if _ball_stats["max_neighbors_cap"] > 0 else "")
            )
            print(
                f"  min/mean/max neighbours: "
                f"{_ball_stats['min_neighbors']} / "
                f"{_ball_stats['mean_neighbors']:.1f} / "
                f"{_ball_stats['max_neighbors_actual']}"
            )
            print("  distribution (count / %):")
            for lbl in ["<8", "8-15", "16-31", "32-63", "64-127", "128-255", ">=256"]:
                cnt = bc.get(lbl, 0)
                pct = bp.get(lbl, 0.0)
                if cnt > 0:
                    print(f"    {lbl:>8} neighbours: {cnt:>8}  ({pct:.1f}%)")

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
    p.add_argument("--el-list", type=float, nargs="+", default=[],
                   metavar="E",
                   help="多个滤波中心能量（Hartree），非空时覆盖 --el；"
                        "例：--el-list -0.20 -0.19 -0.18 -0.17 -0.16")
    p.add_argument("--nc", type=int, default=500)
    p.add_argument("--dE", type=float, default=50.0)
    p.add_argument(
        "--emin", "--V-min", "--Vmin", "--V_min",
        dest="emin",
        type=float,
        default=-5.0,
        help="filter 能量窗口下界 E_min（兼容旧参数 --Vmin/--V_min）",
    )
    p.add_argument("--n-random", type=int, default=16)
    p.add_argument("--svd-tol", type=float, default=1e-3)
    p.add_argument("--max-energies", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--psi-init-type", type=str, choices=["sine", "gaussian"],
                   default="sine",
                   help="初始波函数类型：sine（随机正弦叠加，默认）或 gaussian（高斯白噪声）")
    p.add_argument("--psi-kmax", type=float, default=3.0,
                   help="正弦初态波矢上界 |k_{x,y,z}| ≤ K_max（仅 sine 模式，默认 3.0）")
    p.add_argument("--rbf-spacing", type=float, default=0.5)
    p.add_argument("--rbf-stencil-size", type=int, default=80)
    p.add_argument("--rbf-stencil-radius", type=float, default=0.0,
                   help="ball-stencil 搜索半径（Bohr）；>0 时用 ball 半径搜索邻居+指纹加速，"
                        "0=禁用（使用 k-nearest stencil_size）")
    p.add_argument("--rbf-stencil-fingerprint-tol", type=float, default=1e-4,
                   help="ball-stencil 指纹分组舍入容差（Bohr，默认 1e-4）")
    p.add_argument("--rbf-stencil-inner-radius", type=float, default=0.0,
                   help="ball-stencil 内半径（Bohr）；在 [inner, outer] 球壳内选邻居，"
                        "中心节点自身（距离0）始终保留；0=禁用")
    p.add_argument("--rbf-stencil-max-neighbors", type=int, default=0,
                   help="ball-stencil 最大邻居数上限（0=不限）；超出时按"
                        " --rbf-stencil-select-near/far 策略裁剪")
    p.add_argument("--rbf-stencil-select-far-first", action="store_true",
                   help="超过最大邻居数时优先保留远处节点（默认保留近处节点）")
    p.add_argument("--rbf-phi", type=str, default="phs3")
    p.add_argument("--rbf-eps", type=float, default=0.5)
    p.add_argument("--rbf-order", type=int, default=2)
    p.add_argument("--rbf-v-clip-percentile", type=float, default=99.9)
    p.add_argument("--rbf-v-source", type=str,
                   choices=["gaussian_direct", "grid_interp"],
                   default="gaussian_direct",
                   help="RBF 节点上 V 的来源：gaussian_direct=直接用高斯展开求值（默认，"
                        "零插值误差），grid_interp=从 cube 网格线性插值（老行为）")

    # QD 节点放置方式
    p.add_argument(
        "--rbf-node-method", type=str,
        choices=["cube", "sphere", "atoms", "conv_cell"], default="cube",
        help="QD 节点来源：cube=规则网格(默认)，sphere=球内 Poisson-disc，"
             "atoms=原子位置 + 可选 Poisson-disc 加密，"
             "conv_cell=惯用晶胞混合模板（atoms+level-3 FCC+Poisson+parity）× 平铺")
    p.add_argument("--rbf-R", type=float, default=20.0,
                   help="sphere / atoms 球半径（Bohr）")
    p.add_argument("--rbf-sphere-subdivide", type=int, default=3,
                   help="球面 icosphere 细分级数（越大边界越圆滑）")
    p.add_argument("--rbf-augment", type=str,
                   choices=["poisson_disc", "none"], default="poisson_disc",
                   help="atoms 模式加密方式")
    p.add_argument("--rbf-exclude-radius", type=float, default=0.0,
                   help="atoms 模式下 Poisson 候选点离原子更近则丢弃（Bohr）")

    # conv_cell 专用
    p.add_argument("--conv-cell-a", type=float, default=11.4523,
                   help="惯用晶胞晶格常数 a（Bohr），默认 InAs 11.4523")
    p.add_argument("--conv-cell-d-min-frac", type=float, default=0.06,
                   help="conv_cell fractional d_min（greedy 过滤阈值）")
    p.add_argument("--conv-cell-n-random", type=int, default=120,
                   help="conv_cell 每个晶胞 Poisson-like 目标点数")
    p.add_argument("--conv-cell-seed", type=int, default=42,
                   help="conv_cell 随机种子")
    p.add_argument("--conv-cell-no-parity", action="store_true",
                   help="不加 1-r 反演对偶点")
    p.add_argument("--conv-cell-template-mode", type=str,
                   choices=["hybrid", "fcc_refined"], default="hybrid",
                   help="conv_cell 模板模式：hybrid(默认) 或 fcc_refined(确定性FCC)")
    p.add_argument("--conv-cell-fcc-scale-factor", type=int, default=8,
                   help="fcc_refined 模式每轴细分 scale_factor（总点数约 4*sf^3/胞）")
    p.add_argument("--conv-cell-fcc-origin-frac", type=float, nargs=3,
                   default=[0.0, 0.0, 0.0], metavar=("FX", "FY", "FZ"),
                   help="fcc_refined 模式 FCC 原点分数坐标偏移")
    p.add_argument("--conv-cell-fcc-atom-refine-factor", type=int, default=0,
                   help="增强方法一：原子附近 local FCC 的更细密 scale_factor（0=关闭）")
    p.add_argument("--conv-cell-fcc-atom-radius-frac", type=float, default=0.0,
                   help="增强方法一：仅保留与任意原子周期距离 < radius_frac 的 local FCC 点（0=关闭）")
    p.add_argument("--conv-cell-boundary-margin-frac", type=float, default=0.06,
                   help="距 bbox 面小于 margin*a 的节点判为 boundary；"
                        "默认 0.06 = d_min_frac，薄壳；设 0 关闭 boundary 划分")
    p.add_argument("--conv-cell-use-legacy-poisson", action="store_true",
                   help="用用户原版周期拒绝采样而不是 rbf.poisson_disc_nodes")
    p.add_argument("--conv-cell-domain-shape", type=str,
                   choices=["cube", "sphere"], default="cube",
                   help="conv_cell 的体域形状：cube=平铺立方体(默认)，sphere=平铺球形体域")
    p.add_argument("--conv-cell-sphere-radius", type=float, default=0.0,
                   help="conv_cell sphere 模式球半径(Bohr)；<=0 自动取 bbox 内切球半径")
    p.add_argument("--conv-cell-sphere-subdivide", type=int, default=3,
                   help="conv_cell sphere 边界的 icosphere 细分级数")
    p.add_argument("--conv-cell-adaptive-random", action="store_true",
                   help="conv_cell: 启用势能自适应随机点采样（细密网格+接受概率+KDTree 过滤）")
    p.add_argument("--conv-cell-adaptive-grid-n", type=int, default=36,
                   help="conv_cell 自适应采样时，一个原胞细密均匀网格每轴点数")
    p.add_argument("--conv-cell-adaptive-lambda-grad", type=float, default=0.0,
                   help="conv_cell 自适应采样 λ1（h = h_max/(1+λ1|∇V|+λ2|ΔV|)）")
    p.add_argument("--conv-cell-adaptive-lambda-lap", type=float, default=0.0,
                   help="conv_cell 自适应采样 λ2（h = h_max/(1+λ1|∇V|+λ2|ΔV|)）")
    p.add_argument("--conv-cell-adaptive-candidate-multiplier", type=float, default=8.0,
                   help="conv_cell 自适应采样候选预算倍数（相对 n_random）")
    p.add_argument("--conv-cell-no-skeleton", action="store_true",
                   help="conv_cell hybrid 模式：去掉原子+level3骨架，仅保留自适应泊松采样节点")
    p.add_argument("--exclude-boundary", action="store_true",
                   help="仅保留 interior 节点（去掉 boundary）")
    p.add_argument("--exclude-interior", action="store_true",
                   help="仅保留 boundary 节点（去掉 interior）")
    p.add_argument("--node-min-dist", type=float, default=0.0,
                   help="节点最小距离过滤阈值（Bohr，0=关闭）")

    # 节点质量度量
    p.add_argument("--quality-probe-method", type=str,
                   choices=["uniform", "random"], default="uniform",
                   help="计算 fill radius h 时的探测点分布方式")
    p.add_argument("--quality-probe-n", type=int, default=0,
                   help="探测点总数（0=自动，约 32× n_nodes）")

    # 可选对照 Ritz：量化"插值到均匀格点"的影响
    p.add_argument(
        "--rbf-interp-ritz", action="store_true",
        help="额外跑一次'把 RBF 滤波基插到 FFT 均匀格点后再用 FFT 的 H 做 Ritz'；"
             "结果写入 evals_rbf_interp，与 evals_rbf（直接在节点上 Ritz）对比即为"
             "插值本身带来的误差")

    # 节点持久化
    p.add_argument(
        "--save-nodes", type=str, default="",
        help="把节点单独存成 NumPy 数据文件 .npz（可以是目录或完整文件路径）；"
             "目录时会自动命名为 nodes_<tag>_<method>_<ts>.npz")
    p.add_argument(
        "--load-nodes", type=str, default="",
        help="从已存节点文件加载（.npz/.json；跳过节点生成，Laplacian/V 仍按当前 CLI 重新算）")

    p.add_argument("--power-steps", type=int, default=30)
    p.add_argument("--fft-kinetic-cut", type=float, default=30.0)
    p.add_argument("--fft-only", action="store_true",
                   help="只运行 FFT filter + Ritz，跳过 RBF 建点/哈密顿量/对比项")
    p.add_argument("--out-dir", type=str, default="filter_compare_results")

    a = p.parse_args()
    if a.exclude_interior and a.exclude_boundary:
        p.error("--exclude-interior 和 --exclude-boundary 不能同时设置")
    return CompareConfig(
        system=a.system,
        qd_radius=a.qd_radius,
        ho_N=a.ho_N,
        ho_L=a.ho_L,
        el=a.el,
        el_list=a.el_list,
        nc=a.nc,
        dE=a.dE,
        Vmin=a.emin,
        n_random=a.n_random,
        svd_tol=a.svd_tol,
        max_energies=a.max_energies,
        seed=a.seed,
        psi_init_type=a.psi_init_type,
        psi_kmax=a.psi_kmax,
        rbf_spacing=a.rbf_spacing,
        rbf_stencil_size=a.rbf_stencil_size,
        rbf_stencil_radius=a.rbf_stencil_radius,
        rbf_stencil_fingerprint_tol=a.rbf_stencil_fingerprint_tol,
        rbf_stencil_inner_radius=a.rbf_stencil_inner_radius,
        rbf_stencil_max_neighbors=a.rbf_stencil_max_neighbors,
        rbf_stencil_select_near_first=(not a.rbf_stencil_select_far_first),
        rbf_phi=a.rbf_phi,
        rbf_eps=a.rbf_eps,
        rbf_order=a.rbf_order,
        rbf_v_clip_percentile=a.rbf_v_clip_percentile,
        rbf_v_source=a.rbf_v_source,
        rbf_node_method=a.rbf_node_method,
        rbf_R=a.rbf_R,
        rbf_sphere_subdivide=a.rbf_sphere_subdivide,
        rbf_augment=a.rbf_augment,
        rbf_exclude_radius=a.rbf_exclude_radius,
        save_nodes=a.save_nodes,
        load_nodes=a.load_nodes,
        rbf_interp_ritz=a.rbf_interp_ritz,
        conv_cell_a=a.conv_cell_a,
        conv_cell_d_min_frac=a.conv_cell_d_min_frac,
        conv_cell_n_random=a.conv_cell_n_random,
        conv_cell_seed=a.conv_cell_seed,
        conv_cell_parity=(not a.conv_cell_no_parity),
        conv_cell_template_mode=a.conv_cell_template_mode,
        conv_cell_fcc_scale_factor=a.conv_cell_fcc_scale_factor,
        conv_cell_fcc_origin_frac=tuple(float(v) for v in a.conv_cell_fcc_origin_frac),
        conv_cell_fcc_atom_refine_factor=a.conv_cell_fcc_atom_refine_factor,
        conv_cell_fcc_atom_radius_frac=a.conv_cell_fcc_atom_radius_frac,
        conv_cell_boundary_margin_frac=a.conv_cell_boundary_margin_frac,
        conv_cell_use_rbf_poisson=(not a.conv_cell_use_legacy_poisson),
        conv_cell_domain_shape=a.conv_cell_domain_shape,
        conv_cell_sphere_radius=a.conv_cell_sphere_radius,
        conv_cell_sphere_subdivide=a.conv_cell_sphere_subdivide,
        conv_cell_adaptive_random=a.conv_cell_adaptive_random,
        conv_cell_adaptive_grid_n=a.conv_cell_adaptive_grid_n,
        conv_cell_adaptive_lambda_grad=a.conv_cell_adaptive_lambda_grad,
        conv_cell_adaptive_lambda_lap=a.conv_cell_adaptive_lambda_lap,
        conv_cell_adaptive_candidate_multiplier=a.conv_cell_adaptive_candidate_multiplier,
        conv_cell_include_skeleton=(not a.conv_cell_no_skeleton),
        include_interior=(not a.exclude_interior),
        include_boundary=(not a.exclude_boundary),
        node_min_dist=a.node_min_dist,
        quality_probe_method=a.quality_probe_method,
        quality_probe_n=a.quality_probe_n,
        power_steps=a.power_steps,
        fft_kinetic_cut=a.fft_kinetic_cut,
        fft_only=a.fft_only,
        out_dir=a.out_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    out_json = run(cfg)
    print(f"结果已保存: {out_json}")
