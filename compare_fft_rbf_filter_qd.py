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
    # 面附近多近算 boundary：薄壳（≈1·d_min）才合理；以前 0.5 太大，会让中心
    # 立方只剩 ~36% 体积，视觉上像没铺满。默认 = d_min_frac。
    conv_cell_boundary_margin_frac: float = 0.06
    conv_cell_use_rbf_poisson: bool = True  # True=rbf.poisson_disc_nodes, False=周期拒绝采样

    # ── 节点质量度量 ───────────────────────────────────────────────────────
    quality_probe_method: str = "uniform"   # 'uniform' 或 'random'
    quality_probe_n: int = 0                # 0 = 自动（~32× n_nodes）

    # ── 节点持久化 ─────────────────────────────────────────────────────────
    save_nodes: str = ""   # 非空则把节点单独写入此 JSON（无需后缀，未填则自动命名）
    load_nodes: str = ""   # 非空则从此 JSON 加载节点，跳过节点生成；节点上的
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


def _save_conv_cell_template_json(problem: RBFProblem, nodes_json_path: Path) -> Path | None:
    nodes_frac = problem.groups.get("conv_cell_template_nodes_frac")
    nodes_cart = problem.groups.get("conv_cell_template_nodes_cart")
    roles = problem.groups.get("conv_cell_template_roles")
    if nodes_frac is None or nodes_cart is None or roles is None:
        return None

    out = nodes_json_path.with_name(nodes_json_path.stem + "_conv_cell_template.json")
    payload = {
        "nodes_frac": np.asarray(nodes_frac, dtype=float).tolist(),
        "nodes_cart": np.asarray(nodes_cart, dtype=float).tolist(),
        "roles": np.asarray(roles, dtype=int).tolist(),
        "role_legend": {
            "0": "atoms",
            "1": "level3",
            "2": "random",
            "3": "parity",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out


def _save_nodes_json(node_storage: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(node_storage), f, ensure_ascii=False, indent=2)
    return path


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


def _build_problem_from_nodes_json(path: Path, cfg: "CompareConfig",
                                    cube_path: Path) -> tuple[RBFProblem, dict[str, Any]]:
    """Reconstruct a RBFProblem from a saved nodes JSON.

    Node positions and group assignments are loaded from the file as-is.
    The RBF-FD Laplacian and V_nodes are recomputed using the **current** CLI
    parameters (stencil_size / phi / eps / order / v_clip_percentile / v_source),
    so you can reuse the same node layout while sweeping RBF kernels.
    """
    from rbf.pde.fd import weight_matrix

    with path.open("r", encoding="utf-8") as f:
        saved = json.load(f)

    nodes = np.asarray(saved["coordinates_all"], dtype=float)
    groups = {str(k): np.asarray(v, dtype=int) for k, v in saved["groups"].items()}
    if "interior" not in groups:
        raise ValueError(
            f"saved nodes JSON {path} missing 'interior' group")
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
    provenance: dict[str, Any] = {
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
        "conv_cell_boundary_margin_frac": cfg.conv_cell_boundary_margin_frac,
        "conv_cell_use_rbf_poisson":      cfg.conv_cell_use_rbf_poisson,
        "ho_L":            cfg.ho_L,
        "loaded_from":     cfg.load_nodes or None,
    }

    if cfg.load_nodes:
        # 从已存节点 JSON 重建 RBFProblem（Laplacian + V 按当前 CLI 重新算）
        if cfg.system != "qd" or qd_cube is None:
            raise ValueError("--load-nodes 目前只支持 --system qd（需要 cube 文件做 V 插值）")
        load_path = Path(cfg.load_nodes)
        if not load_path.exists():
            raise FileNotFoundError(f"--load-nodes 指定的文件不存在: {load_path}")
        problem, _saved_meta = _build_problem_from_nodes_json(load_path, cfg, qd_cube)
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
            conv_cell_boundary_margin_frac=cfg.conv_cell_boundary_margin_frac,
            conv_cell_use_rbf_poisson=cfg.conv_cell_use_rbf_poisson,
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
    H_rbf_op = spla.aslinearoperator(H_rbf)
    interior_idx = problem.interior_idx
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

    # 按需单独落盘节点 JSON（便于后续 --load-nodes 复用，不依赖结果 JSON）
    nodes_json_path: Path | None = None
    ts_nodes = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.save_nodes:
        # 用户给了路径——如果是目录则自动命名，否则当作完整文件路径
        p_user = Path(cfg.save_nodes)
        if p_user.suffix == "" or p_user.is_dir():
            p_user.mkdir(parents=True, exist_ok=True)
            tag = (f"qd_R{cfg.qd_radius}" if cfg.system == "qd"
                   else f"ho_N{cfg.ho_N}")
            nodes_json_path = p_user / f"nodes_{tag}_{cfg.rbf_node_method}_{ts_nodes}.json"
        else:
            nodes_json_path = p_user
        _save_nodes_json(node_storage, nodes_json_path)
        print(f"节点 JSON 已保存: {nodes_json_path}")
        if cfg.rbf_node_method == "conv_cell":
            template_path = _save_conv_cell_template_json(problem, nodes_json_path)
            if template_path is not None:
                print(f"原胞模板节点已保存: {template_path}")

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

    # ── 可选：带插值的对照 Ritz ─────────────────────────────────────────────
    # 把 RBF 滤波基从 n_interior 非均匀节点插到 FFT 均匀格点 (n_grid)，
    # 再用 FFT 的 H 在插值后基上做 Ritz。供对比"插值 vs 不插值"。
    evals_rbf_interp: np.ndarray = np.array([], dtype=np.float64)
    rank_rbf_interp = 0
    t_interp_build = 0.0
    t_rr_rbf_interp = 0.0
    per_state_interp: list[dict[str, Any]] = []

    if cfg.rbf_interp_ritz:
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

    avg_state_err = float(np.mean([x["abs_diff"] for x in per_state])) if per_state else None
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
            "n_interior_rbf": n_interior,
            "qd_cube": str(qd_cube) if qd_cube is not None else None,
        },
        # 结果 JSON 中保留完整 node_storage；nodes_file 指向独立落盘的副本（如有）
        "rbf_nodes": node_storage,
        "rbf_nodes_file": str(nodes_json_path) if nodes_json_path else None,
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
        },
        "per_state_filtered_energy": per_state,
        "per_state_filtered_energy_interp": per_state_interp,
        "summary": {
            "avg_abs_error_per_state_filtered_energy": avg_state_err,
            "avg_abs_error_eigenvalues": avg_eval_err,
            "max_abs_error_per_state_filtered_energy": float(np.max([x["abs_diff"] for x in per_state])) if per_state else None,
            "max_abs_error_eigenvalues": float(np.max([x["abs_diff"] for x in paired])) if paired else None,
            "avg_abs_error_interp_effect_eigenvalues": avg_interp_eff,
            "max_abs_error_interp_effect_eigenvalues": (
                float(np.max([x["abs_diff"] for x in paired_rbf_interp_vs_nodes]))
                if paired_rbf_interp_vs_nodes else None),
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
    p.add_argument("--conv-cell-boundary-margin-frac", type=float, default=0.06,
                   help="距 bbox 面小于 margin*a 的节点判为 boundary；"
                        "默认 0.06 = d_min_frac，薄壳；设 0 关闭 boundary 划分")
    p.add_argument("--conv-cell-use-legacy-poisson", action="store_true",
                   help="用用户原版周期拒绝采样而不是 rbf.poisson_disc_nodes")

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
        help="把节点单独存成 JSON（可以是目录或完整文件路径）；"
             "目录时会自动命名为 nodes_<tag>_<method>_<ts>.json")
    p.add_argument(
        "--load-nodes", type=str, default="",
        help="从已存的节点 JSON 加载（跳过节点生成，Laplacian/V 仍按当前 CLI 重新算）")

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
        conv_cell_boundary_margin_frac=a.conv_cell_boundary_margin_frac,
        conv_cell_use_rbf_poisson=(not a.conv_cell_use_legacy_poisson),
        quality_probe_method=a.quality_probe_method,
        quality_probe_n=a.quality_probe_n,
        power_steps=a.power_steps,
        fft_kinetic_cut=a.fft_kinetic_cut,
        out_dir=a.out_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    out_json = run(cfg)
    print(f"结果已保存: {out_json}")
