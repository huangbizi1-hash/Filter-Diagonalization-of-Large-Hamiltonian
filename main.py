"""
main.py
=======
FFT 滤波对角化（Filter Diagonalization Method）入口脚本。

算法概述
--------
在三维均匀实空间网格上，通过以下流程求解定态薛定谔方程 H|ψ⟩ = E|ψ⟩ 的本征能级：

  1. 构建势能场 V(r)
     支持四种类型：
       - ho3d          : 三维谐振子  V = ½ω²r²
       - harmonic_well : 截断谐振势阱（可多中心叠加）
       - gaussian_blob : 单高斯凹坑  V = h·exp(-μr²)
       - gaussian_files: 从 ab-initio 计算输出的 .cube 文件和
                         高斯拟合参数 .json 文件重建原子势
                         （调用 gaussian_potential_builder.GaussianPotentialBuilder）

  2. 构建动能算符对角元 T(k) = |k|²/2，截断于 kinetic_cut

  3. 计算 Newton 多项式滤波器系数（fft_code.filter_coeff）

  4. 滤波随机态生成子空间基（fft_code.hamiltonian）

  5. Rayleigh-Ritz 对角化（fft_code.rayleigh_ritz）

  6. 保存结果与图像至 results/<timestamp>_<tag>/

模块结构
--------
  fft_code/
    params.py        — IstParams / PhysParams 数据类
    grid.py          — 网格构建 / k 空间动能对角元
    wavefunction.py  — 波函数归一化 / 随机初态
    hamiltonian.py   — FFT 动能 / 哈密顿量 / 滤波器作用
    filter_coeff.py  — Newton 插值滤波系数
    rayleigh_ritz.py — SVD + Rayleigh-Ritz 对角化
    potentials.py    — 测试势能 + build_potential_from_config 分派器
    plotting.py      — 所有绘图函数

关键参数说明
-----------
  N             : 每轴网格点数（总格点数 = N³）
  nc            : Newton 插值节点数，越大越精确
  dE            : 滤波窗口宽度，须满足 Vmin + dE ≥ V_max + kinetic_cut
  Vmin          : 滤波窗口下界，须满足 Vmin ≤ V_min(grid)
  dt            : 高斯滤波宽度，由 dt = (nc / (dE×2.5))² 自动推导
  El_list       : 滤波中心能量列表
  n_random      : 每个滤波中心的随机初态数量
  svd_tol       : SVD 秩截断阈值
  kinetic_cut   : k 空间动能截断上限（原子单位）

使用方法
--------
  python main.py                  # 使用文件内 CONFIG 运行
  python main.py --cfg my.json    # 从外部 JSON 文件覆盖 CONFIG 参数

注意事项
--------
  ⚠️  更换势能类型时必须同步调整 Vmin 和 dE，使滤波窗口完整覆盖
      哈密顿量谱范围。程序启动后会打印 "Spectrum check" 可据此调参。
"""

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

# ============================================================
# fft_code 子包
# ============================================================
from fft_code.params       import IstParams, PhysParams
from fft_code.grid         import build_k_diagonal
from fft_code.wavefunction import random_sine_psi, normalize_psi
from fft_code.hamiltonian  import apply_H, apply_filter_H_all
from fft_code.filter_coeff import build_filter_coefficients, make_filter_func
from fft_code.rayleigh_ritz import svd_rayleigh_ritz
from fft_code.potentials   import build_potential_from_config
from fft_code.plotting     import (
    plot_filter_interpolation,
    plot_window_comparison,
    plot_filtered_energies,
    plot_energy_levels,
    plot_energy_errors,
    plot_potential_slice,
)


# ============================================================
# ✅  CONFIG  ― 仅修改此块以切换运行配置
# ============================================================
CONFIG: Dict[str, Any] = {
    # ---------- 输出 ----------
    "out_root": "results",
    "tag": "gaussian_run",

    # ---------- 势能 ----------
    "potential": {
        "type": "gaussian_files",
        "d": 0.5,                               # 重采样网格步长
        "cube_file": "localPot.cube",
        "params_file": "gaussian_fit_params.json",
        "r_cut": 7.0,
    },

    # ---------- 网格 ----------
    "N": 64,

    # ---------- 滤波器 ----------
    # ⚠️  先用小参数跑一次，看启动时打印的 Spectrum check 里的 H_max，再调这两个值
    "nc": 1000,
    "dE": 50.0,             # 根据实际 H_max 调整
    "Vmin": -5.0,           # 根据实际 V_min 调整
    "El_list": list(np.arange(-0.2, -0.1, 0.1).tolist()),

    # ---------- 窗函数类型 ----------
    # "gaussian" : 经典高斯，宽度由 dt=(nc/(dE×2.5))² 决定（窄 → 高 nc）
    # "gabor"    : 超高斯包络 × cos 调制，关于 El 对称，宽度由 alpha_f/n0 控制
    #              f(x) = exp(-alpha_f*|x-El|**n0) · cos(k_f*(x-El))
    #              n0=2 → 普通高斯包络；n0>2 → 超高斯（平顶更宽、边沿更陡）
    # "bandpass" : 平滑带通窗（双 tanh 差分），关于 El 对称
    #              w(x) = 0.5·[tanh(β(x-EL)) - tanh(β(x-ER))]
    #              EL = El - E1，ER = El + E1
    #              beta 越大截止越锐利；E1 为带通半宽（Hartree）
    "filter_type": "gabor",   # "gaussian" | "gabor" | "bandpass"
    "alpha_f": 45.0,             # Gabor 包络衰减系数
    "k_f": 20.0,                 # Gabor 余弦调制频率（Hartree⁻¹）
    "n0": 4,                     # Gabor 包络指数（n0=2 普通高斯，n0=4 超高斯）
    "beta": 10.0,                # 带通窗边沿陡峭系数（仅 bandpass 使用）
    "E1": 0.05,                  # 带通窗半宽（Hartree，仅 bandpass 使用）

    # ---------- Newton 插值节点选取方式 ----------
    # "ashkenazy"          : 贪心最大化 Vandermonde 行列式（默认），近似 Chebyshev 分布
    # "chebyshev"          : 第一类 Chebyshev 节点，两端密、中间稀
    # "derivative_adapted" : 基于 |dw/dx| 的反 CDF 自适应（仅推荐与 bandpass 配合使用）
    #                        在 EL=El-E1 和 ER=El+E1 两个 tanh 过渡区密集放点，
    #                        通带顶部（≈1）和远端衰减区（≈0）少放点
    "samp_method": "ashkenazy",  # "ashkenazy" | "chebyshev" | "derivative_adapted"
    # 以下仅在 samp_method="derivative_adapted" 时生效：
    "deriv_bg_frac": 0.2,        # 均匀背景占比（0=纯自适应；0.2 = 20% 均匀，保证远端衰减区有节点）

    # ---------- 窗函数对比绘图 ----------
    # 若 plot_window_bands 非空，在 window_comparison.png 中标记目标频带和 gap
    "plot_window_bands": {
        "target": [-0.22, -0.13],   # 感兴趣的本征值区间（绿色阴影）
        "gap":    [-0.20, -0.15],   # 无本征值的 gap（红色阴影）
    },

    # ---------- 随机态 ----------
    "n_random": 1,
    "seed": 42,

    # ---------- SVD / Rayleigh-Ritz ----------
    "svd_tol": 1e-3,
    "max_energies": 200,

    # ---------- 动能截断 ----------
    "kinetic_cut": 30.0,

    # ---------- 杂项 ----------
    "print_every_filter": 1,

    # ---------- Newton 节点自适应增强 ----------
    # 在 samp_method 生成的基础节点之上，若插值 MAE 超过阈值，
    # 则在 interval_samp_enhance（物理坐标，Hartree）内逐步追加节点。
    # nc 作为初始估计，实际用到的 nc_true = len(samp) 会在运行时确定并记录。
    # 设为 null（Python None）可完全禁用自适应增强。
    "interval_samp_enhance":    [-0.22, -0.13],  # 需要加密的能量区间 [lo, hi]（Hartree）
    "interpolation_tolerance":  1e-3,            # 插值 MAE 阈值；超过则继续追加节点
    "enhance_step":             10,              # 每轮追加节点数
    "max_enhance_iters":        30,              # 最大增强轮数（防止不收敛）

    # ---------- 画图 ----------
    "plot_interval": [-0.4, 0.1],   # 滤波函数绘图能量区间 [E_lo, E_hi]
}
CONFIG["dt"] = (CONFIG["nc"] / (CONFIG["dE"] * 2.5)) ** 2


# ============================================================
# ✅  SCAN  ― 参数扫描列表（留空则只跑一次 CONFIG）
#
# 用法：在列表中每加一个 dict，就多跑一次。
# dict 里只写想要覆盖的键，其余键保持 CONFIG 默认值。
# 支持嵌套 dict（如 "potential"）：会递归合并而非整体替换。
#
# 示例：
#   SCAN = [
#       {"nc": 100},                          # 第1次：nc=100
#       {"nc": 200, "n_random": 10},          # 第2次：nc=200, n_random=10
#       {"nc": 500, "dE": 60.0, "Vmin": -6}, # 第3次：同时改三个参数
#   ]
#
# 若 SCAN = []，则退化为只跑一次 CONFIG，与原行为完全一致。
# ============================================================
SCAN = [
    # {"nc": 100},
    # {"nc": 200, "n_random": 10},
]


# ============================================================
# JSON 辅助
# ============================================================
def _to_jsonable(obj: Any) -> Any:
    """递归将 numpy 类型转为 Python 原生类型以便 JSON 序列化。"""
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
    # 1. 构建势能
    # ================================================================
    print("=" * 60)
    print("1. Building potential ...")
    t0 = time.perf_counter()

    N = cfg["N"]
    x, y, z, X, Y, Z, x_grid, V = build_potential_from_config(cfg, N)
    Nx = Ny = Nz = N

    timings["build_potential"] = time.perf_counter() - t0
    print(f"   V shape: {V.shape},  min={V.min():.4f},  max={V.max():.4f}")
    print(f"   Grid spacing: {x[1]-x[0]:.4f} a.u.")
    print(f"   Time: {timings['build_potential']:.3f} s")

    # ---- 频谱覆盖检查 ----
    H_max_est = float(V.max()) + cfg.get("kinetic_cut", 30.0)
    H_min_est = float(V.min())
    win_lo    = cfg["Vmin"]
    win_hi    = cfg["Vmin"] + cfg["dE"]
    if win_lo > H_min_est or win_hi < H_max_est:
        print(f"\n  ⚠️  WARNING: filter window [{win_lo}, {win_hi}] does NOT cover "
              f"estimated spectrum [{H_min_est:.2f}, {H_max_est:.2f}]!")
        print(f"     Suggested fix: Vmin <= {H_min_est:.2f}, "
              f"dE >= {H_max_est - H_min_est:.2f}\n")
    else:
        print(f"   Spectrum check OK: [{H_min_est:.2f}, {H_max_est:.2f}] "
              f"⊆ window [{win_lo}, {win_hi}]")

    plot_potential_slice(V, x, z, out_dir)

    # ================================================================
    # 2. 构建动能对角元
    # ================================================================
    kinetic_cut  = cfg.get("kinetic_cut", 30.0)
    T_k_diagonal = build_k_diagonal(x_grid, kinetic_cut=kinetic_cut)

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

    # 构造窗函数
    filter_func = make_filter_func(
        filter_type, dt=dt, alpha_f=alpha_f, k_f=k_f, n0=n0, beta=beta, E1=E1)

    print(f"   nc (initial)={nc},  dE={par.dE},  Vmin={par.Vmin},  dt={dt:.4f}")
    print(f"   Filter type : {filter_type}")
    if filter_type == "gaussian":
        print(f"   sigma (Gaussian) = {1/np.sqrt(2*dt):.6f} Hartree")
        filter_label = f"Gaussian (sigma={1/np.sqrt(2*dt):.4f})"
    elif filter_type == "gabor":
        print(f"   alpha_f = {alpha_f},  k_f = {k_f},  n0 = {n0}")
        filter_label = f"Gabor (alpha_f={alpha_f}, k_f={k_f}, n0={n0})"
    elif filter_type == "bandpass":
        print(f"   beta = {beta},  E1 = {E1}  →  半带宽 ±{E1} Hartree")
        filter_label = f"Bandpass (beta={beta}, E1={E1})"
    else:
        filter_label = filter_type
    print(f"   Sampling method : {samp_method}")
    print(f"   Number of filter centres: {len(El_list)}")

    # 构建 Newton 插值节点和系数
    samp_kw = {}
    if samp_method == "derivative_adapted":
        samp_kw["E1"]      = E1
        samp_kw["beta"]    = beta
        samp_kw["bg_frac"] = cfg.get("deriv_bg_frac", 0.2)

    enhance_interval = cfg.get("interval_samp_enhance", None)
    if enhance_interval is not None:
        enhance_interval = tuple(enhance_interval)
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
    ist     = IstParams(nc=nc_true, ms=len(El_list))   # 用实际节点数覆盖初始估计

    timings["build_filter"] = time.perf_counter() - t0
    print(f"   nc (initial) = {nc},  nc_true = {nc_true}"
          + (f"  (+{nc_true - nc} enhanced)" if nc_true != nc else ""))
    print(f"   Time: {timings['build_filter']:.3f} s")

    interval = tuple(cfg["plot_interval"])

    # 窗函数 vs Newton 插值对比图
    plot_filter_interpolation(El_list, an, samp, par, interval, out_dir,
                              filter_func=filter_func, filter_label=filter_label)

    # 多种窗函数形状对比图（帮助直观比较 gaussian / gabor 宽度差异）
    bands     = cfg.get("plot_window_bands", {})
    target    = bands.get("target", None)
    gap       = bands.get("gap",    None)
    # 构造对比所用的两条曲线：当前使用的窗 + 另一种窗（仅当两者不同时才添加对比）
    cmp_funcs: Dict[str, Any] = {filter_label: filter_func}
    if filter_type == "gaussian":
        from fft_code.filter_coeff import _filt_func_gabor
        cmp_funcs["Gabor (alpha_f=0.5, k_f=1.0, n0=4) [参考]"] = (
            lambda x, El: _filt_func_gabor(x, El, 0.5, 1.0, 4)
        )
    elif filter_type == "gabor":
        from fft_code.filter_coeff import _filt_func_gaussian
        cmp_funcs[f"Gaussian (sigma={1/np.sqrt(2*dt):.4f}) [参考]"] = (
            lambda x, El: _filt_func_gaussian(x, El, dt)
        )
    elif filter_type == "bandpass":
        from fft_code.filter_coeff import _filt_func_gaussian
        cmp_funcs[f"Gaussian (sigma={1/np.sqrt(2*dt):.4f}) [参考]"] = (
            lambda x, El: _filt_func_gaussian(x, El, dt)
        )
    plot_window_comparison(El_list, par, interval, out_dir,
                           window_funcs=cmp_funcs,
                           highlight_band=target,
                           gap_band=gap)

    # ================================================================
    # 4. 滤波随机态
    # ================================================================
    # 使用 apply_filter_H_all：所有 El 共享 Newton 基底向量，
    # H 作用次数从 ms*nc 降至 nc。
    print("\n3. Filtering random states ...")
    t0 = time.perf_counter()

    n_random            = cfg["n_random"]
    filtered_psi_matrix = np.zeros(
        (ist.ms * n_random, Nx, Ny, Nz), dtype='complex128')
    E_temp_all  = [[] for _ in range(ist.ms)]
    print_every = cfg.get("print_every_filter", 1)

    for i in range(n_random):
        psi_rand     = random_sine_psi(X, Y, Z, rng=rng)
        psi_filt_all = apply_filter_H_all(
            psi_rand, V, samp, an, par, T_k_diagonal)  # (ms, Nx, Ny, Nz)
        for ie in range(ist.ms):
            psi_filt = normalize_psi(psi_filt_all[ie])
            H_psi    = apply_H(psi_filt, V, T_k_diagonal)
            E_exp    = float(np.sum(psi_filt.conj() * H_psi).real)
            E_temp_all[ie].append(E_exp)
            filtered_psi_matrix[ie * n_random + i] = psi_filt

    E_mean = []
    E_std  = []
    for ie in range(ist.ms):
        mean_e = float(np.mean(E_temp_all[ie]))
        std_e  = float(np.std(E_temp_all[ie]))
        E_mean.append(mean_e)
        E_std.append(std_e)
        if ie % print_every == 0:
            print(f"   El={El_list[ie]:.2f}  mean={mean_e:.4f}  std={std_e:.4f}")

    timings["filter_states"] = time.perf_counter() - t0
    error_mean = float(np.mean(np.abs(np.array(E_mean) - El_list)))
    print(f"   Mean error vs El: {error_mean:.6f}")
    print(f"   Time: {timings['filter_states']:.3f} s")

    plot_filtered_energies(El_list, [E_mean], [E_std], [N], [error_mean],
                            n_random, out_dir)

    # ================================================================
    # 5. Rayleigh-Ritz 对角化
    # ================================================================
    print("\n4. Rayleigh-Ritz diagonalisation ...")
    t0 = time.perf_counter()

    energies, Ur, rank = svd_rayleigh_ritz(
        filtered_psi_matrix, x_grid, V, Nx, Ny, Nz,
        T_k_diagonal=T_k_diagonal,
        svd_tol=cfg.get("svd_tol", 1e-3),
        max_energies=cfg.get("max_energies", 200),
    )

    timings["rayleigh_ritz"] = time.perf_counter() - t0
    timings["total"]         = time.perf_counter() - t_total_start

    print(f"   Rank r = {rank}")
    print(f"   First 10 energies: {np.round(energies[:10], 4).tolist()}")
    print(f"   Time: {timings['rayleigh_ritz']:.3f} s")
    print(f"\n   Total wall time: {timings['total']:.3f} s")

    # ================================================================
    # 6. 能级图（参考精确能级仅对 ho3d 有意义）
    # ================================================================
    exact_energies = np.arange(1.5, 18.5, 1.0)
    plot_energy_levels(energies, exact_energies, out_dir)
    plot_energy_errors({f"N={N}": energies}, exact_energies, out_dir)

    # ================================================================
    # 7. 保存 JSON 结果
    # ================================================================
    print("\n5. Saving results ...")
    results = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "config":          cfg,
        "timings_seconds": timings,
        "grid": {
            "N": N, "Nx": Nx, "Ny": Ny, "Nz": Nz,
            "x_min": float(x[0]), "x_max": float(x[-1]),
            "dx":    float(x[1] - x[0]),
        },
        "potential": {
            "V_min":  float(V.min()),
            "V_max":  float(V.max()),
            "V_mean": float(V.mean()),
        },
        "filter": {
            "filter_type": filter_type,
            "nc":      nc,
            "nc_true": nc_true,
            "dt":     dt,
            "sigma_gaussian": float(1 / np.sqrt(2 * dt)),
            **({"alpha_f": alpha_f, "k_f": k_f, "n0": n0}
               if filter_type == "gabor" else {}),
            **({"beta": beta, "E1": E1}
               if filter_type == "bandpass" else {}),
            "El_list": El_list.tolist(),
            "E_mean": E_mean,
            "E_std":  E_std,
            "error_mean_vs_El": error_mean,
        },
        "rayleigh_ritz": {
            "rank":       rank,
            "svd_tol":    cfg.get("svd_tol", 1e-3),
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
    # 若 nc 或 dE 被覆盖，重新推导 dt
    if "nc" in override or "dE" in override:
        base["dt"] = (base["nc"] / (base["dE"] * 2.5)) ** 2


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="FFT filter-diagonalise solver")
    parser.add_argument("--cfg",  type=str, default=None,
                        help="Path to a JSON file that overrides CONFIG")
    parser.add_argument("--scan", type=str, default=None,
                        help="Path to a JSON file containing a SCAN list "
                             "(overrides the in-file SCAN)")
    args = parser.parse_args()

    # ---- 加载基础 cfg ----
    base_cfg = copy.deepcopy(CONFIG)
    if args.cfg:
        with open(args.cfg, "r") as f:
            _merge_override(base_cfg, json.load(f))

    # ---- 确定 scan 列表 ----
    if args.scan:
        with open(args.scan, "r") as f:
            scan_list = json.load(f)
    else:
        scan_list = SCAN  # 使用文件内定义的 SCAN

    # ---- 无扫描：单次运行 ----
    if not scan_list:
        run(base_cfg)
        return

    # ---- 有扫描：遍历每个 override ----
    n = len(scan_list)
    print(f"\n{'='*60}")
    print(f"  SCAN 模式：共 {n} 组配置")
    print(f"{'='*60}")

    for i, override in enumerate(scan_list, start=1):
        cfg = copy.deepcopy(base_cfg)
        _merge_override(cfg, override)

        # tag 只保留序号，参数详情打印到终端
        changed = ", ".join(
            f"{k}={v}" for k, v in override.items() if k != "tag"
        )
        base_tag = cfg.get("tag", "")
        cfg["tag"] = f"{base_tag}_scan{i}" if base_tag else f"scan{i}"

        print(f"\n{'─'*60}")
        print(f"  运行 {i}/{n}：{changed}")
        print(f"{'─'*60}")
        run(cfg)

    print(f"\n{'='*60}")
    print(f"  SCAN 完成：共 {n} 组配置均已运行")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
