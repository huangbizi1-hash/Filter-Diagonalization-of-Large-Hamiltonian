"""
gnn_code/timing_benchmark.py — H-apply 计时基准

比较不同 H 实现在 QD 上作用一次 H 的耗时（不需要训练好的模型）：

  · FFT（细网格，d_fine = d_coarse/2）：精确参照，结果输出为水平虚线
  · FD-cube   ：FiniteDiffHamiltonian，3×3×3 Mehrstellen，0 参数
  · FD-cross  ：FiniteDiffHamiltonian_Cross，高阶十字星，0 参数
  · GNN-cube  ：HamiltonianGNN，hidden_dim 可变
  · GNN-cross ：HamiltonianGNN_Cross，hidden_dim 可变
  · SO3-cross ：SO3HamiltonianNet，radial_hidden_dim 可变

模型使用随机初始化权重（eval 模式），不加载 checkpoint。
每种配置先热身 n_warmup 次，再计时 n_reps 次，取均值与标准差。

用法（通过 run_gnn.py）：
    python run_gnn.py --mode timing \\
        --timing_hidden_dims 8 16 32 64 128 256 \\
        --timing_n_reps 100 --timing_n_warmup 10 \\
        --device cpu \\
        --timing_description "cube vs cross vs so3 on QD"

输出：
    <output_root>/timing_<timestamp>.json
"""

import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch

from .physics import d_sparse
from .graph import build_graph, build_star_graph
from .model import (
    HamiltonianGNN, FiniteDiffHamiltonian,
    HamiltonianGNN_Cross, FiniteDiffHamiltonian_Cross,
    SO3HamiltonianNet,
)
from .test_filter import (
    _DEFAULT_CUBE, _DEFAULT_PARAMS,
    _load_qd_potential, _build_fft_qd_operator,
)


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def _count_params(model) -> int:
    if model is None:
        return 0
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters())
    return 0


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_fn(fn, n_reps: int, n_warmup: int, device: torch.device):
    """
    Call fn() n_warmup times (discard), then n_reps times (record wall time).
    Returns (mean_s, std_s, list_of_times_s).
    """
    for _ in range(n_warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(n_reps):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)

    return float(np.mean(times)), float(np.std(times)), [float(t) for t in times]


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def time_h_apply(
    n_reps:              int   = 100,
    n_warmup:            int   = 10,
    hidden_dims:         list  = None,
    radial_hidden_dims:  list  = None,
    fd_order:            int   = 4,
    n_co:                int   = 3,
    device:              str   = "cpu",
    cube_file:           str   = _DEFAULT_CUBE,
    params_file:         str   = _DEFAULT_PARAMS,
    d_coarse:            float = None,
    output_root:         str   = ".",
    description:         str   = "",
):
    """
    Measure single H-apply wall time for all model architectures.

    Parameters
    ----------
    n_reps             : timing repetitions (after warmup)
    n_warmup           : warmup calls before timing
    hidden_dims        : list of hidden_dim values for GNN-cube / GNN-cross
    radial_hidden_dims : list of radial_hidden_dim for SO3; defaults to hidden_dims
    fd_order           : FD accuracy order for cross/SO3 graphs
    n_co               : correction cube side length for cross/SO3 graphs
    device             : 'cpu' or 'cuda'
    d_coarse           : grid spacing for GNN/FD; None → use d_sparse from physics.py
    output_root        : directory for output JSON
    description        : free-text description stored in JSON
    """
    if hidden_dims is None:
        hidden_dims = [8, 16, 32, 64, 128, 256]
    if radial_hidden_dims is None:
        radial_hidden_dims = hidden_dims

    d_coarse = d_coarse if d_coarse is not None else d_sparse
    d_fine   = d_coarse / 2.0

    dev = (torch.device("cuda") if device == "cuda" and torch.cuda.is_available()
           else torch.device("cpu"))
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  H-apply timing benchmark")
    print(f"  d_coarse={d_coarse}  d_fine={d_fine}  device={dev}")
    print(f"  n_reps={n_reps}  n_warmup={n_warmup}")
    print(f"{'='*64}")

    # ── QD 势能（粗/细两套网格） ───────────────────────────────────────────
    print("\n  Loading QD potential (coarse grid)...")
    pot_coarse, N_qd, d_actual_c = _load_qd_potential(
        d_coarse, cube_file=cube_file, params_file=params_file)
    n_grid_c = N_qd ** 3
    V_flat_c = pot_coarse.potential.ravel().astype(np.float32)
    V_3d_c   = pot_coarse.potential.reshape(N_qd, N_qd, N_qd)

    print("\n  Loading QD potential (fine grid)...")
    pot_fine, N_qd_f, d_actual_f = _load_qd_potential(
        d_fine, cube_file=cube_file, params_file=params_file)
    n_grid_f = N_qd_f ** 3
    V_3d_f   = pot_fine.potential.reshape(N_qd_f, N_qd_f, N_qd_f)

    print(f"\n  Coarse grid: N={N_qd}  n_grid={n_grid_c:,}  d_actual={d_actual_c:.4f}")
    print(f"  Fine   grid: N={N_qd_f}  n_grid={n_grid_f:,}  d_actual={d_actual_f:.4f}")

    # ── 预先构建图结构 ─────────────────────────────────────────────────────
    grid_L_c = N_qd   * d_actual_c
    grid_L_f = N_qd_f * d_actual_f

    print("\n  Building cube graph (coarse)...")
    ei_cube, ea_cube = build_graph(N=N_qd, d=d_actual_c, grid_L=grid_L_c)
    ei_cube = ei_cube.to(dev); ea_cube = ea_cube.to(dev)

    print("  Building star graph (coarse)...")
    fd_ei, fd_ea, co_ei, co_ea = build_star_graph(
        fd_order, n_co, N=N_qd, d=d_actual_c, grid_L=grid_L_c)
    fd_ei = fd_ei.to(dev); fd_ea = fd_ea.to(dev)
    co_ei = co_ei.to(dev); co_ea = co_ea.to(dev)

    V_t = torch.tensor(V_flat_c, dtype=torch.float32).unsqueeze(-1).to(dev)

    # ── 随机测试向量 ───────────────────────────────────────────────────────
    rng     = np.random.default_rng(42)
    psi_np  = rng.standard_normal(n_grid_c).astype(np.float32)
    psi_np /= np.linalg.norm(psi_np)
    psi_t   = torch.tensor(psi_np, dtype=torch.float32).unsqueeze(-1).to(dev)

    psi_f_np  = rng.standard_normal(n_grid_f).astype(np.float32)
    psi_f_np /= np.linalg.norm(psi_f_np)

    results = []  # list of dicts

    # ── FFT 细网格（参照） ─────────────────────────────────────────────────
    print("\n  [FFT-fine] building & timing...")
    fft_op = _build_fft_qd_operator(V_3d_f, N_qd_f, d_actual_f)
    t_mean, t_std, t_all = _time_fn(
        lambda: fft_op.matvec(psi_f_np), n_reps, n_warmup, dev)
    results.append({
        "label":       "FFT-fine",
        "type":        "FFT",
        "grid":        "fine",
        "n_grid":      n_grid_f,
        "n_params":    0,
        "hidden_dim":  None,
        "t_mean_ms":   t_mean * 1e3,
        "t_std_ms":    t_std  * 1e3,
        "t_all_ms":    [t * 1e3 for t in t_all],
    })
    print(f"    {t_mean*1e3:.2f} ± {t_std*1e3:.2f} ms")

    # ── FD-cube（无参数） ──────────────────────────────────────────────────
    print("\n  [FD-cube] timing...")
    fd_cube = FiniteDiffHamiltonian(ei_cube, ea_cube, V_t, dev)
    t_mean, t_std, t_all = _time_fn(
        lambda: fd_cube(psi_t), n_reps, n_warmup, dev)
    results.append({
        "label":      "FD-cube",
        "type":       "FD",
        "grid":       "coarse",
        "n_grid":     n_grid_c,
        "n_params":   0,
        "hidden_dim": None,
        "t_mean_ms":  t_mean * 1e3,
        "t_std_ms":   t_std  * 1e3,
        "t_all_ms":   [t * 1e3 for t in t_all],
    })
    print(f"    {t_mean*1e3:.2f} ± {t_std*1e3:.2f} ms")

    # ── FD-cross（无参数） ─────────────────────────────────────────────────
    print("\n  [FD-cross] timing...")
    fd_cross = FiniteDiffHamiltonian_Cross(fd_ei, fd_ea, V_t, dev)
    t_mean, t_std, t_all = _time_fn(
        lambda: fd_cross(psi_t), n_reps, n_warmup, dev)
    results.append({
        "label":      "FD-cross",
        "type":       "FD",
        "grid":       "coarse",
        "n_grid":     n_grid_c,
        "n_params":   0,
        "hidden_dim": None,
        "t_mean_ms":  t_mean * 1e3,
        "t_std_ms":   t_std  * 1e3,
        "t_all_ms":   [t * 1e3 for t in t_all],
    })
    print(f"    {t_mean*1e3:.2f} ± {t_std*1e3:.2f} ms")

    # ── GNN-cube（随机初始化，hidden_dim 可变） ────────────────────────────
    for h in hidden_dims:
        label = f"GNN-cube-h{h}"
        print(f"\n  [{label}] building & timing...")
        model = HamiltonianGNN(hidden_dim=h).to(dev)
        model.eval()
        n_p = _count_params(model)

        def _fn_cube(m=model):
            with torch.no_grad():
                nrm = torch.norm(psi_t) + 1e-30
                return m(psi_t / nrm, ei_cube, ea_cube, V_t) * nrm

        t_mean, t_std, t_all = _time_fn(_fn_cube, n_reps, n_warmup, dev)
        results.append({
            "label":      label,
            "type":       "GNN-cube",
            "grid":       "coarse",
            "n_grid":     n_grid_c,
            "n_params":   n_p,
            "hidden_dim": h,
            "t_mean_ms":  t_mean * 1e3,
            "t_std_ms":   t_std  * 1e3,
            "t_all_ms":   [t * 1e3 for t in t_all],
        })
        print(f"    n_params={n_p}  {t_mean*1e3:.2f} ± {t_std*1e3:.2f} ms")

    # ── GNN-cross（随机初始化，hidden_dim 可变） ───────────────────────────
    for h in hidden_dims:
        label = f"GNN-cross-h{h}"
        print(f"\n  [{label}] building & timing...")
        model = HamiltonianGNN_Cross(hidden_dim=h).to(dev)
        model.eval()
        n_p = _count_params(model)

        def _fn_cross(m=model):
            with torch.no_grad():
                nrm = torch.norm(psi_t) + 1e-30
                return m(psi_t / nrm, fd_ei, fd_ea, co_ei, co_ea, V_t) * nrm

        t_mean, t_std, t_all = _time_fn(_fn_cross, n_reps, n_warmup, dev)
        results.append({
            "label":      label,
            "type":       "GNN-cross",
            "grid":       "coarse",
            "n_grid":     n_grid_c,
            "n_params":   n_p,
            "hidden_dim": h,
            "t_mean_ms":  t_mean * 1e3,
            "t_std_ms":   t_std  * 1e3,
            "t_all_ms":   [t * 1e3 for t in t_all],
        })
        print(f"    n_params={n_p}  {t_mean*1e3:.2f} ± {t_std*1e3:.2f} ms")

    # ── SO3-cross（随机初始化，radial_hidden_dim 可变） ────────────────────
    for rh in radial_hidden_dims:
        label = f"SO3-cross-rh{rh}"
        print(f"\n  [{label}] building & timing...")
        model = SO3HamiltonianNet(radial_hidden_dim=rh).to(dev)
        model.eval()
        n_p = _count_params(model)

        def _fn_so3(m=model):
            with torch.no_grad():
                nrm = torch.norm(psi_t) + 1e-30
                return m(psi_t / nrm, fd_ei, fd_ea, co_ei, co_ea, V_t) * nrm

        t_mean, t_std, t_all = _time_fn(_fn_so3, n_reps, n_warmup, dev)
        results.append({
            "label":          label,
            "type":           "SO3-cross",
            "grid":           "coarse",
            "n_grid":         n_grid_c,
            "n_params":       n_p,
            "radial_hidden_dim": rh,
            "hidden_dim":     None,
            "t_mean_ms":      t_mean * 1e3,
            "t_std_ms":       t_std  * 1e3,
            "t_all_ms":       [t * 1e3 for t in t_all],
        })
        print(f"    n_params={n_p}  {t_mean*1e3:.2f} ± {t_std*1e3:.2f} ms")

    # ── 汇总打印 ───────────────────────────────────────────────────────────
    fft_t = next((r["t_mean_ms"] for r in results if r["label"] == "FFT-fine"), float("nan"))
    print(f"\n{'─'*72}")
    print(f"  {'Label':<26} {'n_params':>9} {'t_mean(ms)':>11} {'t/FFT':>7}")
    print(f"  {'─'*26} {'─'*9} {'─'*11} {'─'*7}")
    for r in results:
        ratio = r["t_mean_ms"] / fft_t if fft_t > 0 else float("nan")
        print(f"  {r['label']:<26} {r['n_params']:>9} "
              f"{r['t_mean_ms']:>11.3f} {ratio:>7.3f}")
    print(f"{'─'*72}")

    # ── 保存 JSON ──────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":   "gnn_code/timing_benchmark.py",
        "datetime": datetime.datetime.now().isoformat(),
        "config": {
            "d_coarse":          d_coarse,
            "d_fine":            d_fine,
            "d_actual_coarse":   float(d_actual_c),
            "d_actual_fine":     float(d_actual_f),
            "N_qd_coarse":       N_qd,
            "N_qd_fine":         N_qd_f,
            "n_grid_coarse":     n_grid_c,
            "n_grid_fine":       n_grid_f,
            "fd_order":          fd_order,
            "n_co":              n_co,
            "n_reps":            n_reps,
            "n_warmup":          n_warmup,
            "device":            str(dev),
            "hidden_dims":       hidden_dims,
            "radial_hidden_dims": radial_hidden_dims,
            "fft_fine_t_mean_ms": fft_t,
        },
        "fft_fine_reference": next(
            (r for r in results if r["label"] == "FFT-fine"), {}),
        "methods": [r for r in results if r["label"] != "FFT-fine"],
    }

    json_path = out_dir / f"timing_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Results → {json_path}")
    return output
