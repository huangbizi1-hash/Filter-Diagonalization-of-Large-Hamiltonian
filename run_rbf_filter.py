"""
run_rbf_filter.py
=================
RBF-FD 版本的滤波对角化（Filter Diagonalization）入口脚本。

算法与 main.py（FFT 版）完全一致——只把「H 如何作用到 ψ」从 FFT 动能换成
RBF-FD 稀疏矩阵。滤波系数、Newton 递推、Rayleigh-Ritz 三个阶段都复用
filter_core.py 中与算符无关的通用函数。

流程概览
--------
  1. 从 cube 文件构建 RBFProblem（按 --domain 选择节点放置方式）：
       cube    : cube 文件原有的规则网格（和 FFT 一致）
       sphere  : 球内 Poisson disc
       atoms   : 先在所有原子位置取节点，再可选 Poisson disc 加密

  2. 组装稀疏 H = -½·L_rbf + diag(V)（interior-only）

  3. 构建 Newton 多项式滤波系数（filter_core.build_filter_coefficients）

  4. 对 n_random 个随机态做滤波：filter_core.apply_filter_H_all_op
     共享 Newton 基底，H-apply 次数 = nc

  5. SVD + Rayleigh-Ritz：filter_core.svd_rayleigh_ritz_op

  6. 保存结果 / 绘图到 rbf_results/<timestamp>_<tag>/

示例命令
--------
  # 1) cube 网格 + bandpass 滤波（与 FFT 版最接近）
  python run_rbf_filter.py --set potential.cube_file=localPot.cube \\
                           --set domain=cube \\
                           --set filter_type=bandpass --set beta=45 --set E1=0.1

  # 2) 球形域 + Poisson disc 节点
  python run_rbf_filter.py --set domain=sphere --set R=20 --set spacing=0.625

  # 3) atoms：所有原子位置取节点 + Poisson disc 加密
  python run_rbf_filter.py --set domain=atoms \\
                           --set augment=poisson_disc \\
                           --set spacing=0.8 --set exclude_radius=0.4 \\
                           --set R=20

  # 4) atoms：只用原子位置（实验性，stencil 需要大）
  python run_rbf_filter.py --set domain=atoms --set augment=none \\
                           --set stencil_size=40

  # 5) 用外部 JSON 覆盖全部配置
  python run_rbf_filter.py --cfg my_rbf_cfg.json

  # 6) SCAN 多个配置
  python run_rbf_filter.py --scan_json '[{"phi":"phs3"},{"phi":"ga"}]'
"""
from __future__ import annotations

# ============================================================
# 标准库 / 第三方
# ============================================================
import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# filter_core 通用滤波 / Rayleigh-Ritz
# ============================================================
from filter_core import (
    PhysParams,
    IstParams,
    build_filter_coefficients,
    make_filter_func,
    compute_newton_an,
    apply_filter_H_all_op,
    apply_filter_H_op,
    svd_rayleigh_ritz_op,
)


# ============================================================
# ✅  CONFIG  ― 仅修改此块以切换运行配置
# ============================================================
CONFIG: Dict[str, Any] = {
    # ---------- 输出 ----------
    "out_root": "rbf_results",
    "tag": "rbf_filter_run",

    # ---------- 势能来源（当前只支持 cube 文件） ----------
    "potential": {
        "cube_file":         "localPot.cube",
        "v_clip_percentile": 99.9,
    },

    # ---------- 节点放置 ----------
    # domain:
    #   "cube"      : 使用 cube 文件原有的规则网格（interior = 非面边界）
    #   "sphere"    : 球内 Poisson disc
    #   "atoms"     : 从 cube 文件读原子坐标作为节点，再按 augment 加密
    #   "conv_cell" : 惯用晶胞模板 (atoms+level-3 FCC+Poisson+parity) × 平铺
    "domain":           "atoms",
    "spacing":          0.8,         # Poisson-disc 间距（sphere/atoms）
    "R":                20.0,        # 球半径（Bohr，sphere/atoms）
    "sphere_subdivide": 3,
    # atoms 专用：
    "augment":          "poisson_disc",   # "poisson_disc" | "none"
    "exclude_radius":   0.4,              # Bohr，Poisson 候选点离原子更近则丢弃
    # conv_cell 专用：
    "conv_cell_a":                    11.4523,
    "conv_cell_d_min_frac":           0.06,
    "conv_cell_n_random":             120,
    "conv_cell_seed":                 42,
    "conv_cell_parity":               True,
    "conv_cell_boundary_margin_frac": 0.5,
    "conv_cell_use_rbf_poisson":      True,
    # 节点质量检测（运行时计算 q, h, ρ 并打印 / 写 JSON）：
    "quality_probe_method":           "uniform",
    "quality_probe_n":                0,

    # ---------- RBF 设置 ----------
    "stencil_size": 80,
    "phi":          "phs3",
    "eps":          0.5,
    "order":        2,

    # ---------- 滤波器 ----------
    # ⚠️  与 FFT 版一致：先用小参数跑一次看 H 谱范围，再调 dE/Vmin
    "nc":       500,
    "dE":       50.0,
    "Vmin":     -5.0,
    "El_list":  list(np.arange(-0.2, -0.1, 0.02).tolist()),

    # ---------- 窗函数类型 ----------
    "filter_type": "bandpass",   # "gaussian" | "gabor" | "bandpass" | "split_bandpass"
    "alpha_f":     45.0,
    "k_f":         20.0,
    "n0":          4,
    "beta":        45.0,
    "E1":          0.1,

    # ---------- Newton 节点选取方式 ----------
    "samp_method":   "ashkenazy",
    "deriv_bg_frac": 0.2,
    "density_lo":    -0.3,
    "density_hi":     0.0,
    "density_alpha":  0.0,

    # ---------- 随机态 ----------
    "n_random": 1,
    "seed":     42,

    # ---------- SVD / Rayleigh-Ritz ----------
    "svd_tol":      1e-3,
    "max_energies": 200,
    # RBF-FD 的 L 一般非对称，默认用 eig（取实部）；symmetrize=True 则用 eigh
    "hermitian_RR": True,

    # ---------- 杂项 ----------
    "print_every_filter": 1,

    # ---------- Newton 节点自适应增强 ----------
    "interval_samp_enhance":   None,          # 设成 [lo, hi] 开启
    "interpolation_tolerance": 1e-3,
    "enhance_step":            10,
    "max_enhance_iters":       0,
    "enhance_density_factor":  16,
}
CONFIG["dt"] = (CONFIG["nc"] / (CONFIG["dE"] * 2.5)) ** 2


# ============================================================
# ✅  SCAN  ― 参数扫描列表（留空则只跑一次 CONFIG）
# ============================================================
SCAN: list = [
    # {"domain": "cube"},
    # {"domain": "atoms", "augment": "poisson_disc", "spacing": 0.6},
]


# ============================================================
# JSON 序列化辅助
# ============================================================
def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.complexfloating)):
        return float(obj.real)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, ensure_ascii=False, indent=2)


# ============================================================
# RBFProblem 的随机态工厂
# ============================================================
def _random_psi(n: int, rng: np.random.Generator,
                kind: str = "normal") -> np.ndarray:
    if kind == "normal":
        v = rng.standard_normal(n)
    elif kind == "pm1":
        v = rng.choice([-1.0, 1.0], size=n).astype(np.float64)
    else:
        raise ValueError(f"Unknown random kind={kind!r}")
    return v / np.linalg.norm(v)


# ============================================================
# 小工具：画节点散点（前 2 个坐标轴投影）
# ============================================================
def _plot_nodes(problem, out_dir: Path) -> None:
    nodes = problem.nodes
    interior = problem.interior_idx
    atom_idx = problem.groups.get("atoms", np.array([], dtype=int))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (ii, jj, lab) in zip(
        axes, [(0, 1, "XY"), (0, 2, "XZ")]):
        ax.scatter(nodes[interior, ii], nodes[interior, jj],
                   s=2, color="steelblue", label="interior", alpha=0.5)
        bd = problem.groups.get("boundary", np.array([], dtype=int))
        if len(bd):
            ax.scatter(nodes[bd, ii], nodes[bd, jj],
                       s=2, color="gray", label="boundary", alpha=0.3)
        if len(atom_idx):
            ax.scatter(nodes[atom_idx, ii], nodes[atom_idx, jj],
                       s=15, color="red", label="atoms", zorder=5)
        ax.set_aspect("equal")
        ax.set_title(f"{lab} projection  n_total={len(nodes)}  "
                     f"n_interior={len(interior)}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "nodes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 主运行函数
# ============================================================
def run(cfg: Dict[str, Any]) -> None:
    # ---- 输出目录 ----
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = cfg.get("tag", "")
    out_dir = Path(cfg["out_root"]) / (stamp + (f"_{tag}" if tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n")

    # ---- 随机种子 ----
    seed = cfg.get("seed", 0)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    timings: Dict[str, float] = {}
    t_total_start = time.perf_counter()

    # ================================================================
    # 1. 构建 RBFProblem（节点放置 + Laplacian + V）
    # ================================================================
    print("=" * 60)
    print("1. Building RBFProblem ...")
    t0 = time.perf_counter()

    # rbf_core 依赖 rbf-python；延迟导入，方便 --help 等只查看帮助
    from rbf_core import build_qd_problem, compute_node_quality

    pot = cfg["potential"]
    problem = build_qd_problem(
        cube_file        = pot["cube_file"],
        domain           = cfg["domain"],
        spacing          = cfg["spacing"],
        R                = cfg["R"],
        stencil_size     = cfg["stencil_size"],
        phi              = cfg["phi"],
        eps              = cfg["eps"],
        order            = cfg["order"],
        sphere_subdivide = cfg["sphere_subdivide"],
        augment          = cfg.get("augment", "poisson_disc"),
        exclude_radius   = cfg.get("exclude_radius", 0.0),
        v_clip_percentile= pot.get("v_clip_percentile", 99.9),
        conv_cell_a                    = cfg.get("conv_cell_a", 11.4523),
        conv_cell_d_min_frac           = cfg.get("conv_cell_d_min_frac", 0.06),
        conv_cell_n_random             = cfg.get("conv_cell_n_random", 120),
        conv_cell_seed                 = cfg.get("conv_cell_seed", 42),
        conv_cell_parity               = cfg.get("conv_cell_parity", True),
        conv_cell_boundary_margin_frac = cfg.get("conv_cell_boundary_margin_frac", 0.5),
        conv_cell_use_rbf_poisson      = cfg.get("conv_cell_use_rbf_poisson", True),
    )
    n_interior = len(problem.interior_idx)
    n_total    = len(problem.nodes)
    atom_idx   = problem.groups.get("atoms", np.array([], dtype=int))

    timings["build_problem"] = time.perf_counter() - t0
    print(f"   domain           : {cfg['domain']}")
    if cfg["domain"] == "atoms":
        print(f"   augment          : {cfg.get('augment')}")
        print(f"   exclude_radius   : {cfg.get('exclude_radius')} Bohr")
    elif cfg["domain"] == "conv_cell":
        print(f"   a (lattice)      : {cfg.get('conv_cell_a')} Bohr")
        print(f"   d_min_frac       : {cfg.get('conv_cell_d_min_frac')}")
        print(f"   n_random/cell    : {cfg.get('conv_cell_n_random')}")
        print(f"   parity           : {cfg.get('conv_cell_parity')}")
    print(f"   n_nodes total    : {n_total}")
    print(f"   n_interior       : {n_interior}")

    # 节点质量度量
    nodes_np = np.asarray(problem.nodes, dtype=float)
    quality = compute_node_quality(
        nodes_np[problem.interior_idx],
        bbox_min=nodes_np.min(axis=0) if nodes_np.size else None,
        bbox_max=nodes_np.max(axis=0) if nodes_np.size else None,
        probe_method=cfg.get("quality_probe_method", "uniform"),
        n_probe=(cfg.get("quality_probe_n", 0) or None),
    )
    print(f"   quality q        : {quality.get('q', float('nan')):.4e}")
    print(f"   quality h        : {quality.get('h', float('nan')):.4e}")
    print(f"   quality ρ = h/q  : {quality.get('rho', float('nan')):.3f}")
    if len(atom_idx):
        print(f"   n_atoms (pinned) : {len(atom_idx)}")
    print(f"   V range          : [{problem.V_nodes.min():.4f}, "
          f"{problem.V_nodes.max():.4f}] Ha")
    print(f"   Time: {timings['build_problem']:.3f} s")

    _plot_nodes(problem, out_dir)

    # 频谱覆盖检查（H_max 用 Gershgorin 近似上界）
    H_max_est = float(problem.V_nodes.max()) + 50.0  # 保守；用户可覆盖 dE
    H_min_est = float(problem.V_nodes.min())
    win_lo    = cfg["Vmin"]
    win_hi    = cfg["Vmin"] + cfg["dE"]
    if win_lo > H_min_est or win_hi < H_max_est:
        print(f"\n  ⚠️  WARNING: filter window [{win_lo}, {win_hi}] may not cover "
              f"spectrum [≈{H_min_est:.2f}, ≈{H_max_est:.2f}]!")
    else:
        print(f"   Spectrum check OK: [{H_min_est:.2f}, {H_max_est:.2f}] "
              f"⊆ window [{win_lo}, {win_hi}]")

    # ================================================================
    # 2. 稀疏 H（interior-only）— 一次组装并缓存在 problem._H_sparse
    # ================================================================
    _ = problem.apply_H_flat(np.zeros(n_interior))
    # wrap 成 filter_core 需要的 callable
    H_apply = problem.apply_H_flat

    # ================================================================
    # 3. 构建滤波系数
    # ================================================================
    print("\n2. Building filter coefficients ...")
    t0 = time.perf_counter()

    El_list     = np.array(cfg["El_list"], dtype=float)
    nc          = cfg["nc"]
    dt          = cfg["dt"]
    filter_type = cfg.get("filter_type", "gaussian")
    alpha_f     = cfg.get("alpha_f", 0.5)
    k_f         = cfg.get("k_f", 1.0)
    n0          = cfg.get("n0", 4)
    beta        = cfg.get("beta", 10.0)
    E1          = cfg.get("E1", 0.05)
    samp_method = cfg.get("samp_method", "ashkenazy")
    par         = PhysParams(dE=cfg["dE"], Vmin=cfg["Vmin"], dt=dt)

    filter_func = make_filter_func(
        filter_type, dt=dt, alpha_f=alpha_f, k_f=k_f, n0=n0, beta=beta, E1=E1)

    print(f"   nc (initial)={nc},  dE={par.dE},  Vmin={par.Vmin},  dt={dt:.4f}")
    print(f"   Filter type : {filter_type}")
    print(f"   Sampling    : {samp_method}")
    print(f"   n_centres   : {len(El_list)}")

    samp_kw: Dict[str, Any] = {}
    if samp_method == "derivative_adapted":
        samp_kw["E1"]      = E1
        samp_kw["beta"]    = beta
        samp_kw["bg_frac"] = cfg.get("deriv_bg_frac", 0.2)
    if samp_method == "density_mapped":
        samp_kw["density_lo"]    = cfg.get("density_lo",    -0.22)
        samp_kw["density_hi"]    = cfg.get("density_hi",    -0.13)
        samp_kw["density_alpha"] = cfg.get("density_alpha", 10.0)
    samp_kw["enhance_density_factor"] = cfg.get("enhance_density_factor", 16)

    enhance_interval = cfg.get("interval_samp_enhance", None)
    if enhance_interval is not None:
        enhance_interval = tuple(enhance_interval)

    an_hi = an_lo = None
    if filter_type == "split_bandpass":
        filter_func_hi = make_filter_func("highpass", beta=beta, E1=E1)
        filter_func_lo = make_filter_func("lowpass",  beta=beta, E1=E1)
        _, samp = build_filter_coefficients(
            El_list, par, nc,
            filter_func=filter_func_hi,
            samp_method=samp_method,
            interval_samp_enhance=enhance_interval,
            interpolation_tolerance=cfg.get("interpolation_tolerance", 1e-3),
            enhance_step=cfg.get("enhance_step", 10),
            max_enhance_iters=cfg.get("max_enhance_iters", 30),
            **samp_kw,
        )
        an_hi = compute_newton_an(filter_func_hi, El_list, samp, par)
        an_lo = compute_newton_an(filter_func_lo, El_list, samp, par)
        an    = compute_newton_an(filter_func,   El_list, samp, par)
    else:
        an, samp = build_filter_coefficients(
            El_list, par, nc,
            filter_func=filter_func,
            samp_method=samp_method,
            interval_samp_enhance=enhance_interval,
            interpolation_tolerance=cfg.get("interpolation_tolerance", 1e-3),
            enhance_step=cfg.get("enhance_step", 10),
            max_enhance_iters=cfg.get("max_enhance_iters", 30),
            **samp_kw,
        )

    nc_true = len(samp)
    ist     = IstParams(nc=nc_true, ms=len(El_list))
    timings["build_filter"] = time.perf_counter() - t0
    print(f"   nc_true = {nc_true}"
          + (f"  (+{nc_true - nc} enhanced)" if nc_true != nc else ""))
    print(f"   Time: {timings['build_filter']:.3f} s")

    # ================================================================
    # 4. 滤波随机态
    # ================================================================
    print("\n3. Filtering random states ...")
    t0 = time.perf_counter()

    n_random = cfg["n_random"]
    filtered_basis = np.zeros((n_interior, ist.ms * n_random), dtype=np.float64)
    E_temp_all     = [[] for _ in range(ist.ms)]
    print_every    = cfg.get("print_every_filter", 1)

    for i in range(n_random):
        psi_rand = _random_psi(n_interior, rng, kind="normal")

        if filter_type == "split_bandpass":
            psi_hi_all = apply_filter_H_all_op(
                H_apply, psi_rand, samp, an_hi, par)   # (ms, n_interior)
            psi_filt_all = np.zeros_like(psi_hi_all)
            for ie in range(ist.ms):
                psi_filt_all[ie] = apply_filter_H_op(
                    H_apply, psi_hi_all[ie], samp, an_lo[ie], par)
        else:
            psi_filt_all = apply_filter_H_all_op(
                H_apply, psi_rand, samp, an, par)       # (ms, n_interior)

        for ie in range(ist.ms):
            psi_filt = psi_filt_all[ie]
            norm = np.linalg.norm(psi_filt)
            if norm > 0:
                psi_filt = psi_filt / norm
                H_psi = H_apply(psi_filt)
                E_exp = float(np.dot(psi_filt, H_psi))
                E_temp_all[ie].append(E_exp)
                filtered_basis[:, ie * n_random + i] = psi_filt

    E_mean = [float(np.mean(e)) if e else float("nan") for e in E_temp_all]
    E_std  = [float(np.std(e))  if e else float("nan") for e in E_temp_all]
    for ie in range(ist.ms):
        if ie % print_every == 0:
            print(f"   El={El_list[ie]:.4f}  mean={E_mean[ie]:.4f}  "
                  f"std={E_std[ie]:.4e}  n_good={len(E_temp_all[ie])}")

    timings["filter_states"] = time.perf_counter() - t0
    error_mean = float(np.nanmean(np.abs(np.array(E_mean) - El_list)))
    print(f"   Mean error vs El: {error_mean:.6f}")
    print(f"   Time: {timings['filter_states']:.3f} s")

    # ================================================================
    # 5. Rayleigh-Ritz
    # ================================================================
    print("\n4. Rayleigh-Ritz diagonalisation ...")
    t0 = time.perf_counter()

    energies, Ur, rank = svd_rayleigh_ritz_op(
        filtered_basis, H_apply,
        svd_tol       = cfg.get("svd_tol", 1e-3),
        max_energies  = cfg.get("max_energies", 200),
        hermitian     = cfg.get("hermitian_RR", True),
    )

    timings["rayleigh_ritz"] = time.perf_counter() - t0
    timings["total"]         = time.perf_counter() - t_total_start

    print(f"   Rank r = {rank}")
    print(f"   First 10 energies: {np.round(energies[:10], 6).tolist()}")
    print(f"   Time: {timings['rayleigh_ritz']:.3f} s")
    print(f"\n   Total wall time: {timings['total']:.3f} s")

    # ================================================================
    # 6. 简单能级图
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(energies, bins=40, color="steelblue",
            edgecolor="white", alpha=0.85)
    for el in El_list:
        ax.axvline(el, color="gray", lw=0.6, alpha=0.5)
    ax.set_xlabel("Eigenvalue (Hartree)")
    ax.set_ylabel("Count")
    ax.set_title(f"RBF-FD Filter Diag — domain={cfg['domain']}, "
                 f"n_interior={n_interior}, nc={nc_true}, n_random={n_random}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "energies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================================================================
    # 7. 保存 JSON
    # ================================================================
    results = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "config":          cfg,
        "timings_seconds": timings,
        "problem": {
            "domain":       cfg["domain"],
            "n_total":      n_total,
            "n_interior":   n_interior,
            "n_atoms":      int(len(atom_idx)),
            "V_min":        float(problem.V_nodes.min()),
            "V_max":        float(problem.V_nodes.max()),
            "V_mean":       float(problem.V_nodes.mean()),
            "quality":      quality,     # q, h, rho on interior nodes
        },
        "filter": {
            "filter_type":      filter_type,
            "nc":               nc,
            "nc_true":          nc_true,
            "dt":               dt,
            "El_list":          El_list.tolist(),
            "E_mean":           E_mean,
            "E_std":            E_std,
            "error_mean_vs_El": error_mean,
        },
        "rayleigh_ritz": {
            "rank":       rank,
            "svd_tol":    cfg.get("svd_tol", 1e-3),
            "hermitian":  cfg.get("hermitian_RR", True),
            "n_energies": len(energies),
            "energies":   energies.tolist(),
        },
    }
    save_json(results, out_dir / "res.json")
    print(f"   Saved: {out_dir / 'res.json'}")
    print(f"\nAll outputs in: {out_dir}")


# ============================================================
# 辅助：将 override dict 深度合并到 base dict（原地修改 base）
# ============================================================
def _merge_override(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge_override(base[k], v)
        else:
            base[k] = v
    if "nc" in override or "dE" in override:
        base["dt"] = (base["nc"] / (base["dE"] * 2.5)) ** 2


def _set_nested(cfg: Dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split(".", 1)
    if len(parts) == 1:
        cfg[key_path] = value
    else:
        parent, rest = parts
        if not isinstance(cfg.get(parent), dict):
            cfg[parent] = {}
        _set_nested(cfg[parent], rest, value)


def _parse_val(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


# ============================================================
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="RBF-FD filter-diagonalisation solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（与 main.py 完全一致的覆盖/扫描风格）：
  python run_rbf_filter.py --set domain=atoms --set augment=poisson_disc
  python run_rbf_filter.py --cfg my_cfg.json --set nc=1000
  python run_rbf_filter.py --scan_json '[{"phi":"phs3"},{"phi":"ga"}]'
""",
    )
    parser.add_argument("--cfg",       type=str, default=None,
                        help="JSON 配置文件，深度合并到内置 CONFIG")
    parser.add_argument("--scan",      type=str, default=None,
                        help="JSON 文件路径，内含 SCAN 列表")
    parser.add_argument("--scan_json", type=str, default=None,
                        help="内联 JSON 字符串形式的 SCAN 列表")
    parser.add_argument("--set",       action="append", default=[],
                        metavar="KEY=VALUE",
                        help="覆盖单个 CONFIG 键，可重复；支持点号嵌套")
    args = parser.parse_args()

    base_cfg = copy.deepcopy(CONFIG)
    if args.cfg:
        with open(args.cfg, "r") as f:
            _merge_override(base_cfg, json.load(f))

    for kv in args.set:
        if "=" not in kv:
            parser.error(f"--set 需要 KEY=VALUE 格式，收到：{kv!r}")
        key, val_str = kv.split("=", 1)
        _set_nested(base_cfg, key, _parse_val(val_str))
    if args.set:
        base_cfg["dt"] = (base_cfg["nc"] / (base_cfg["dE"] * 2.5)) ** 2

    if args.scan_json:
        scan_list = json.loads(args.scan_json)
    elif args.scan:
        with open(args.scan, "r") as f:
            scan_list = json.load(f)
    else:
        scan_list = SCAN

    if not scan_list:
        run(base_cfg)
        return

    n = len(scan_list)
    print(f"\n{'='*60}\n  SCAN 模式：共 {n} 组配置\n{'='*60}")
    for i, override in enumerate(scan_list, start=1):
        cfg = copy.deepcopy(base_cfg)
        _merge_override(cfg, override)
        base_tag = cfg.get("tag", "")
        cfg["tag"] = f"{base_tag}_scan{i}" if base_tag else f"scan{i}"
        changed = ", ".join(f"{k}={v}" for k, v in override.items() if k != "tag")
        print(f"\n{'─'*60}\n  运行 {i}/{n}：{changed}\n{'─'*60}")
        run(cfg)
    print(f"\n{'='*60}\n  SCAN 完成\n{'='*60}\n")


if __name__ == "__main__":
    main()
