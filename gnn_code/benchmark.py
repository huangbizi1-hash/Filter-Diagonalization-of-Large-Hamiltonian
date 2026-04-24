"""
gnn_code/benchmark.py — 多模型精度-效率基准测试

统一将 FFT / FD / 各 GNN 模型包装为 LinearOperator.matvec，
在相同测试流程下比较精度（误差）与效率（单次 H-apply 时间）。

测试流程
--------
1. 单次 H-apply 计时：对随机向量重复 n_timing_reps 次，取中位数
2. k=0 单次 H-apply：phi = H|k=0>，返回
     · energy_k0  = <k=0|H|k=0>  （Rayleigh 商，恰好 1 次 H-apply）
     · fidelity   = |<phi_FFT/‖·‖ | phi_method/‖·‖>|  （以 FFT 为基准）
3. Newton 滤波 + Rayleigh-Ritz：
     · nc H-apply/vec，n_random 随机初态，Ritz 特征值列表
     · max_err / mean_err：与 FFT Ritz 值的偏差

用法（通过 run_gnn.py）：
    python run_gnn.py --mode benchmark \\
        --run_dirs gnn_models/chain1_teacher gnn_models/chain2_teacher ... \\
        --filter_nc 100 --filter_el_list -0.8 --filter_n_random 10 \\
        --device cpu

输出：
    <output_root>/benchmark_<timestamp>.json
"""

import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigh

from .physics import N_sparse, d_sparse
from .gnn_operator import build_gnn_operator
from .test_filter import (
    _DEFAULT_CUBE, _DEFAULT_PARAMS, _QD_VMIN, _QD_DE,
    _load_qd_potential, _build_fft_qd_operator,
    _apply_filter_all, _rayleigh_ritz,
)

try:
    from fft_code.params import PhysParams
    from fft_code.filter_coeff import build_filter_coefficients, _filt_func_gaussian
    _HAS_FILTER = True
except ImportError:
    _HAS_FILTER = False

try:
    from ho3d_solvers_v2 import build_3d_fd_operator
    _HAS_HO3D = True
except ImportError:
    _HAS_HO3D = False


# ─────────────────────────────────────────────────────────────────────────────
# 计时
# ─────────────────────────────────────────────────────────────────────────────

def _time_matvec(H_op: LinearOperator, n_grid: int,
                 n_reps: int = 20, seed: int = 123):
    """
    对随机单位向量重复调用 H_op.matvec，取中位数作为单次 H-apply 时间。

    Returns
    -------
    t_median : float   中位数（秒）
    t_all    : list    每次计时结果（秒）
    """
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(n_grid)
    psi /= np.linalg.norm(psi)
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        H_op.matvec(psi)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), [float(t) for t in times]


# ─────────────────────────────────────────────────────────────────────────────
# k=0 单次 H-apply 测试
# ─────────────────────────────────────────────────────────────────────────────

def _h1_k0_test(operators: list, n_grid: int):
    """
    对 k=0（常数归一化）态作用一次 H，统计：
      energy_k0  = <k=0|H|k=0>       （Rayleigh 商，1 次 H-apply）
      phi_norm   = ‖H|k=0>‖
      fidelity   = |<phi_n_FFT | phi_n>|  （FFT 为基准，FFT 自身=1）

    operators 中 FFT 须排在首位；若无 FFT 则 fidelity=nan。

    Returns
    -------
    results     : dict  label → {energy_k0, phi_norm, fidelity_vs_fft}
    fft_phi_n   : ndarray or None   H_FFT|k=0>/‖·‖，供外部复用
    """
    psi_k0  = np.ones(n_grid, dtype=np.float64) / np.sqrt(n_grid)
    results = {}
    ref_phi_n = None

    for label, H_op in operators:
        phi      = H_op.matvec(psi_k0)
        energy   = float(np.dot(psi_k0, phi))   # <k=0|H|k=0>
        phi_norm = float(np.linalg.norm(phi))
        phi_n    = phi / phi_norm if phi_norm > 1e-15 else phi.copy()

        if label == "FFT":
            ref_phi_n = phi_n
            fidelity  = 1.0
        elif ref_phi_n is not None:
            fidelity  = float(abs(np.dot(phi_n, ref_phi_n)))
        else:
            fidelity  = float("nan")

        results[label] = {
            "energy_k0":       energy,
            "phi_norm":        phi_norm,
            "fidelity_vs_fft": fidelity,
        }

    return results, ref_phi_n


# ─────────────────────────────────────────────────────────────────────────────
# Newton 滤波 + Rayleigh-Ritz
# ─────────────────────────────────────────────────────────────────────────────

def _filter_ritz_test(operators: list, nodes: np.ndarray, an: np.ndarray,
                      par, n_grid: int, n_random: int,
                      svd_tol: float = 1e-3, n_max: int = 20):
    """
    对所有算符运行 Newton 滤波 + Rayleigh-Ritz，返回：
      energies, rank, t_filter, t_rr, n_H（仅计 f(H) 中的 H-apply 次数）
    """
    nc_true = an.shape[1]
    ms      = an.shape[0]
    results = {}

    for label, H_op in operators:
        rng      = np.random.default_rng(42)   # 所有算符使用相同种子
        filtered = np.zeros((ms * n_random, n_grid))

        t0 = time.perf_counter()
        for i in range(n_random):
            psi_flat  = rng.standard_normal(n_grid)
            psi_flat /= np.linalg.norm(psi_flat)
            out = _apply_filter_all(H_op, psi_flat, nodes, an, par)
            for ie in range(ms):
                v   = out[ie]
                nrm = np.linalg.norm(v)
                if nrm > 1e-15:
                    filtered[ie * n_random + i] = v / nrm
        t_filter = time.perf_counter() - t0

        t0 = time.perf_counter()
        energies, rank = _rayleigh_ritz(filtered.T, H_op, svd_tol, n_max)
        t_rr = time.perf_counter() - t0

        results[label] = {
            "energies": energies.tolist() if hasattr(energies, "tolist") else list(energies),
            "rank":     int(rank),
            "t_filter": float(t_filter),
            "t_rr":     float(t_rr),
            "n_H":      int((nc_true - 1) * n_random),
        }

    # 以 FFT Ritz 值为基准计算误差
    fft_ev = np.array(results.get("FFT", {}).get("energies", []))
    for label, res in results.items():
        ev       = np.array(res["energies"])
        n_common = min(len(ev), len(fft_ev))
        if n_common > 0 and len(fft_ev) > 0:
            diffs = np.abs(ev[:n_common] - fft_ev[:n_common])
            res["max_err_vs_fft"]  = float(np.max(diffs))
            res["mean_err_vs_fft"] = float(np.mean(diffs))
        else:
            res["max_err_vs_fft"]  = float("nan")
            res["mean_err_vs_fft"] = float("nan")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_models(
    run_dirs:      list,
    nc:            int   = 100,
    el:            float = -0.8,
    n_random:      int   = 10,
    svd_tol:       float = 1e-3,
    n_timing_reps: int   = 20,
    device:        str   = "cpu",
    cube_file:     str   = _DEFAULT_CUBE,
    params_file:   str   = _DEFAULT_PARAMS,
    vmin:          float = None,
    d_e:           float = None,
    output_root:   str   = ".",
):
    """
    对 run_dirs 中的每个 GNN 模型（以及 FFT / FD 基准）运行基准测试。

    Parameters
    ----------
    run_dirs      : GNN run 目录列表（含 config.json + epoch_*.pt）
    nc            : Newton 滤波阶数
    el            : 目标能量（Ha）
    n_random      : 随机初态数量
    svd_tol       : Rayleigh-Ritz SVD 截断阈值
    n_timing_reps : 单次 H-apply 计时重复次数
    device        : GNN 推断设备（'cpu' / 'cuda'）
    output_root   : benchmark_<ts>.json 保存目录
    """
    if not _HAS_FILTER:
        raise ImportError("fft_code 未找到，请从仓库根目录运行。")
    if not _HAS_HO3D:
        raise ImportError("ho3d_solvers_v2 未找到，请从仓库根目录运行。")
    if not run_dirs:
        raise ValueError("run_dirs 不能为空。")

    vmin    = vmin if vmin is not None else _QD_VMIN
    d_e     = d_e  if d_e  is not None else _QD_DE
    El_list = np.array([el])
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 从第一个 run_dir 读取网格参数 ──────────────────────────────────────
    with open(os.path.join(run_dirs[0], "config.json")) as f:
        cfg0 = json.load(f)
    d        = float(cfg0.get("d_sparse", d_sparse))
    fd_order = int(cfg0.get("fd_order", 4))

    print(f"\n{'='*68}")
    print(f"  Benchmark: FFT + FD-{fd_order} + {len(run_dirs)} GNN model(s)")
    print(f"  nc={nc}  El={el} Ha  n_random={n_random}  d_sparse={d:.4f} Bohr")
    print(f"{'='*68}")

    # ── QD 势能 ────────────────────────────────────────────────────────────
    print("\n  Loading QD potential...")
    pot_grid, N_qd, d_actual = _load_qd_potential(
        d, cube_file=cube_file, params_file=params_file)
    n_grid   = N_qd ** 3
    V_qd_3d  = pot_grid.potential.reshape(N_qd, N_qd, N_qd)
    V_qd_flat = pot_grid.potential.ravel().astype(np.float32)

    # ── Newton 滤波系数 ────────────────────────────────────────────────────
    dt           = (nc / (d_e * 2.5)) ** 2
    par          = PhysParams(dE=d_e, Vmin=vmin, dt=dt)
    filter_func  = lambda x, el_: _filt_func_gaussian(x, el_, dt)
    print("  Building Newton filter coefficients...")
    an, samp = build_filter_coefficients(
        El_list, par, nc,
        filter_func=filter_func,
        samp_method="ashkenazy",
        interpolation_tolerance=1e-6,
        enhance_step=10, max_enhance_iters=30,
    )
    nc_true = len(samp)
    nodes   = samp
    print(f"  nc_true={nc_true}")

    # ── 构建所有算符 ────────────────────────────────────────────────────────
    operators:     list = []   # [(label, H_op), ...]
    operator_meta: dict = {}   # label → meta dict

    # FFT（精确参照，排首位）
    print("\n  [FFT] building...")
    try:
        fft_op = _build_fft_qd_operator(V_qd_3d, N_qd, d_actual)
        operators.append(("FFT", fft_op))
        operator_meta["FFT"] = {"type": "fft", "model_params": {}}
        print("  [FFT] ready.")
    except Exception as exc:
        print(f"  WARNING: FFT failed ({exc}) → skipped.")

    # FD（有限差分，无 NN）
    lbl_fd = f"FD-{fd_order}"
    print(f"\n  [{lbl_fd}] building...")
    try:
        fd_op, _, _ = build_3d_fd_operator(N_qd, pot_grid, fd_order=fd_order)
        fd_op.label = lbl_fd
        operators.append((lbl_fd, fd_op))
        operator_meta[lbl_fd] = {"type": "fd", "model_params": {"fd_order": fd_order}}
        print(f"  [{lbl_fd}] ready.")
    except Exception as exc:
        print(f"  WARNING: FD failed ({exc}) → skipped.")

    # GNN 模型
    for run_dir in run_dirs:
        short = os.path.basename(run_dir.rstrip("/\\"))
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)

        ckpts = sorted(
            [fn for fn in os.listdir(run_dir)
             if fn.startswith("epoch_") and fn.endswith(".pt")],
            key=lambda fn: int(fn[len("epoch_"):-len(".pt")]),
        )
        last_ckpt = ckpts[-1] if ckpts else "?"

        model_params = {
            "chain_len":  cfg.get("chain_len",  1),
            "chain_mode": cfg.get("chain_mode", "teacher"),
            "chain_bptt": cfg.get("chain_bptt", False),
            "wf_type":    cfg.get("wf_type",    "?"),
            "hidden_dim": cfg.get("hidden_dim",  64),
            "epochs":     cfg.get("epochs",      0),
            "graph_type": cfg.get("graph_type",  "cube"),
            "model_type": cfg.get("model_type",  "gnn"),
            "checkpoint": last_ckpt,
        }

        print(f"\n  [{short}] building GNN operator "
              f"(chain={model_params['chain_len']}, mode={model_params['chain_mode']})...")
        try:
            gnn_op = build_gnn_operator(
                run_dir, use_fd=False, device=device,
                V_ext=V_qd_flat, N_grid=N_qd)
            operators.append((short, gnn_op))
            operator_meta[short] = {
                "type":         "gnn",
                "run_dir":      str(run_dir),
                "model_params": model_params,
            }
            print(f"  [{short}] loaded {last_ckpt}.")
        except Exception as exc:
            print(f"  WARNING: [{short}] failed ({exc}) → skipped.")

    if not operators:
        raise RuntimeError("没有可用算符，退出。")

    # ── 单次 H-apply 计时 ──────────────────────────────────────────────────
    print(f"\n  Timing single H-apply ({n_timing_reps} reps)...")
    timing_results: dict = {}
    for label, H_op in operators:
        t_med, t_all = _time_matvec(H_op, n_grid, n_reps=n_timing_reps)
        timing_results[label] = {"t_median": t_med, "t_all": t_all}
        print(f"    {label:<24}  {t_med*1000:7.2f} ms/apply")

    # ── k=0 单次 H-apply 测试 ──────────────────────────────────────────────
    print("\n  k=0 single H-apply test...")
    k0_results, _ = _h1_k0_test(operators, n_grid)
    for label, res in k0_results.items():
        print(f"    {label:<24}  E_k0={res['energy_k0']:+.4f} Ha  "
              f"fid={res['fidelity_vs_fft']:.4f}")

    # ── Newton 滤波 + Rayleigh-Ritz ────────────────────────────────────────
    print(f"\n  Newton filter + Rayleigh-Ritz "
          f"(nc_true={nc_true}, El={El_list.tolist()}, n_random={n_random})...")
    ritz_results = _filter_ritz_test(
        operators, nodes, an, par, n_grid, n_random, svd_tol)
    for label, res in ritz_results.items():
        ev = res["energies"]
        e0 = ev[0] if ev else float("nan")
        print(f"    {label:<24}  E[0]={e0:+.4f}  rank={res['rank']}  "
              f"max_err={res['max_err_vs_fft']:.3e}  "
              f"mean_err={res['mean_err_vs_fft']:.3e}")

    # ── 组装 JSON ──────────────────────────────────────────────────────────
    methods_out = []
    for label, _ in operators:
        meta   = operator_meta.get(label, {})
        timing = timing_results.get(label, {})
        k0     = k0_results.get(label, {})
        ritz   = ritz_results.get(label, {})
        methods_out.append({
            "label":            label,
            "type":             meta.get("type", "?"),
            "model_params":     meta.get("model_params", {}),
            "run_dir":          meta.get("run_dir"),
            "t_h_apply_ms":     timing.get("t_median", float("nan")) * 1000,
            "t_h_apply_all_ms": [t * 1000 for t in timing.get("t_all", [])],
            "h1_k0":            k0,
            "ritz":             ritz,
        })

    output = {
        "script":   "gnn_code/benchmark.py",
        "datetime": datetime.datetime.now().isoformat(),
        "config": {
            "nc":           nc,
            "nc_true":      nc_true,
            "el":           el,
            "El_list":      El_list.tolist(),
            "n_random":     n_random,
            "vmin":         vmin,
            "d_e":          d_e,
            "dt":           float(dt),
            "svd_tol":      svd_tol,
            "N_qd":         N_qd,
            "d_sparse":     d,
            "d_actual":     float(d_actual),
            "n_timing_reps": n_timing_reps,
            "device":       device,
        },
        "methods": methods_out,
    }

    json_path = out_dir / f"benchmark_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Results → {json_path}")

    # ── 汇总表 ────────────────────────────────────────────────────────────
    fft_t = next((m["t_h_apply_ms"] for m in methods_out if m["label"] == "FFT"),
                 float("nan"))
    print(f"\n{'─'*84}")
    print(f"  {'Method':<24} {'t_H(ms)':>8} {'t/FFT':>6} "
          f"{'E_k0(Ha)':>10} {'k0_fid':>7} {'E[0]_Ritz':>11} {'max_err':>9}")
    print(f"  {'─'*24} {'─'*8} {'─'*6} {'─'*10} {'─'*7} {'─'*11} {'─'*9}")
    for m in methods_out:
        t_h  = m["t_h_apply_ms"]
        ratio = t_h / fft_t if fft_t > 0 else float("nan")
        ek0  = m["h1_k0"].get("energy_k0", float("nan"))
        fid  = m["h1_k0"].get("fidelity_vs_fft", float("nan"))
        ev   = m["ritz"].get("energies", [])
        e0   = ev[0] if ev else float("nan")
        merr = m["ritz"].get("max_err_vs_fft", float("nan"))
        print(f"  {m['label']:<24} {t_h:>8.2f} {ratio:>6.2f} "
              f"{ek0:>10.4f} {fid:>7.4f} {e0:>11.5f} {merr:>9.3e}")
    print(f"{'─'*84}")

    return output
