"""
gnn_code/timing_benchmark.py — Architecture comparison: H-apply wall time

比较不同架构 GNN/FD/FFT 在 QD 上单次 H-apply 的平均耗时。

固定 hidden_dim 与 radial_hidden_dim，随机初始权重（无需训练）。
对以下架构各自计时（coarse grid，d = d_sparse）：
  · gnn-cube             : HamiltonianGNN    (3×3×3 Mehrstellen, 26 近邻)
  · fd-cube              : FiniteDiffHamiltonian    (同图，0 参数)
  · gnn-cross(pP,cC)     : HamiltonianGNN_Cross     (fd_order=P, n_co=C)
  · fd-cross(pP,cC)      : FiniteDiffHamiltonian_Cross (同图，0 参数)
  · so3-cross(pP,cC)     : SO3HamiltonianNet         (同图，l=0+l=1 修正)
  · fft-fine             : FFT (d = d_sparse/2)，作为水平参考虚线

用法（通过 run_gnn.py）：
    python run_gnn.py --mode timing \\
        --timing_hidden_dim 64 \\
        --timing_radial_hidden_dim 32 \\
        --timing_fd_orders 2 4 6 8 \\
        --timing_n_cos 1 3 3 5 \\
        --timing_n_reps 100 \\
        --timing_n_warmup 10 \\
        --timing_description "architecture comparison on QD"
"""

import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch

from .physics import d_sparse
from .graph   import build_graph, build_star_graph
from .model   import (HamiltonianGNN,       FiniteDiffHamiltonian,
                      HamiltonianGNN_Cross, FiniteDiffHamiltonian_Cross,
                      SO3HamiltonianNet)
from .test_filter import _DEFAULT_CUBE, _DEFAULT_PARAMS, _load_qd_potential


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _time_fn(fn, psi: torch.Tensor, n_reps: int, n_warmup: int,
             device: torch.device) -> tuple:
    """Run fn(psi) n_warmup+n_reps times; return (mean_ms, std_ms)."""
    is_cuda = device.type == 'cuda'

    for _ in range(n_warmup):
        with torch.no_grad():
            fn(psi)
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_reps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fn(psi)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times))


def time_h_apply(
    n_reps:            int   = 100,
    n_warmup:          int   = 10,
    hidden_dim:        int   = 64,
    radial_hidden_dim: int   = 32,
    fd_orders:         list  = None,
    n_cos:             list  = None,
    cube_only:         bool  = False,
    device:            str   = 'cpu',
    cube_file:         str   = _DEFAULT_CUBE,
    params_file:       str   = _DEFAULT_PARAMS,
    output_root:       str   = '.',
    description:       str   = '',
):
    """
    Benchmark single H-apply time across architectures with fixed hidden_dim.

    Parameters
    ----------
    fd_orders : fd_order values for cross-type configs (default [2, 4, 6, 8])
    n_cos     : paired n_co values, same length as fd_orders (default [1,3,3,5])
    """
    if cube_only:
        fd_orders, n_cos = [], []
    else:
        if fd_orders is None:
            fd_orders = [2, 4, 6, 8]
        if n_cos is None:
            n_cos = [1, 3, 3, 5]
        if len(fd_orders) != len(n_cos):
            raise ValueError("fd_orders and n_cos must have the same length")

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev     = torch.device(device if device != 'auto' else
                           ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"\n{'='*62}")
    print(f"  Timing benchmark — architecture comparison")
    print(f"  hidden_dim={hidden_dim}  radial_hidden_dim={radial_hidden_dim}")
    print(f"  n_reps={n_reps}  n_warmup={n_warmup}  device={dev}")
    print(f"{'='*62}")

    # ── Coarse QD grid ────────────────────────────────────────────────────────
    print("\n  Loading QD potential (coarse grid)...")
    pot_c, N_c, d_c = _load_qd_potential(
        d_sparse, cube_file=cube_file, params_file=params_file)
    n_c     = N_c ** 3
    V_flat_c = pot_c.potential.ravel().astype(np.float32)
    grid_L_c = N_c * d_c

    # ── Fine QD grid (FFT reference) ──────────────────────────────────────────
    d_fine = d_c / 2.0
    print(f"\n  Loading QD potential (fine grid, d={d_fine:.4f} Bohr)...")
    pot_f, N_f, d_f = _load_qd_potential(
        d_fine, cube_file=cube_file, params_file=params_file)
    n_f = N_f ** 3

    # ── Random test vectors ───────────────────────────────────────────────────
    rng   = np.random.default_rng(42)
    psi_c = torch.tensor(rng.standard_normal(n_c).astype(np.float32),
                         dtype=torch.float32, device=dev).unsqueeze(-1)
    psi_f = torch.tensor(rng.standard_normal(n_f).astype(np.float32),
                         dtype=torch.float32, device=dev).unsqueeze(-1)

    V_t_c = torch.tensor(V_flat_c, dtype=torch.float32).unsqueeze(-1).to(dev)

    methods  = []
    fft_ref  = None

    # ── FFT fine grid (reference) ─────────────────────────────────────────────
    try:
        from fft_code.hamiltonian import apply_H as _fft_apply_H
        V_3d_f = pot_f.potential.astype(np.float32)
        kx = np.fft.fftfreq(N_f, d=d_f) * 2 * np.pi
        ky = np.fft.fftfreq(N_f, d=d_f) * 2 * np.pi
        kz = np.fft.fftfreq(N_f, d=d_f) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        T_k = (KX**2 + KY**2 + KZ**2) / 2.0

        def _fft_fn(psi_t):
            psi_np = psi_t.cpu().numpy().reshape(N_f, N_f, N_f)
            out_np = _fft_apply_H(psi_np, T_k, V_3d_f)
            return torch.tensor(out_np.ravel(), dtype=torch.float32,
                                device=dev).unsqueeze(-1)

        t_mean, t_std = _time_fn(_fft_fn, psi_f, n_reps, n_warmup, dev)
        fft_ref = {
            "label":     "fft-fine",
            "N_grid":    N_f,
            "n_grid":    n_f,
            "d_bohr":    float(d_f),
            "t_mean_ms": t_mean,
            "t_std_ms":  t_std,
        }
        print(f"\n  [fft-fine]  N={N_f}  n_grid={n_f:,}"
              f"  t={t_mean:.3f}±{t_std:.3f} ms")
    except Exception as exc:
        fft_ref = {"label": "fft-fine", "error": str(exc)}
        print(f"\n  [fft-fine] SKIP: {exc}")

    # ── Cube architectures ────────────────────────────────────────────────────
    print(f"\n  Building cube graph (N={N_c})...")
    ei_c, ea_c = build_graph(N=N_c, d=d_c, grid_L=grid_L_c)
    ei_c = ei_c.to(dev); ea_c = ea_c.to(dev)
    n_cube_edges = ei_c.shape[1]

    # gnn-cube
    model_gcube = HamiltonianGNN(hidden_dim=hidden_dim).to(dev)
    model_gcube.eval()
    n_p_gcube = _count_params(model_gcube)

    def _fn_gnn_cube(psi):
        nrm = torch.norm(psi) + 1e-30
        return model_gcube(psi / nrm, ei_c, ea_c, V_t_c) * nrm

    t_mean, t_std = _time_fn(_fn_gnn_cube, psi_c, n_reps, n_warmup, dev)
    r = {
        "label":       "gnn-cube",
        "model_type":  "gnn",
        "graph_type":  "cube",
        "fd_order":    None,
        "n_co":        None,
        "hidden_dim":  hidden_dim,
        "n_params":    n_p_gcube,
        "n_fd_edges":  n_cube_edges,
        "n_co_edges":  0,
        "t_mean_ms":   t_mean,
        "t_std_ms":    t_std,
    }
    methods.append(r)
    print(f"  [gnn-cube]  n_params={n_p_gcube}  edges={n_cube_edges:,}"
          f"  t={t_mean:.3f}±{t_std:.3f} ms")

    # fd-cube
    fd_cube = FiniteDiffHamiltonian(ei_c, ea_c, V_t_c, dev)

    def _fn_fd_cube(psi):
        return fd_cube(psi)

    t_mean, t_std = _time_fn(_fn_fd_cube, psi_c, n_reps, n_warmup, dev)
    r = {
        "label":       "fd-cube",
        "model_type":  "fd",
        "graph_type":  "cube",
        "fd_order":    None,
        "n_co":        None,
        "hidden_dim":  None,
        "n_params":    0,
        "n_fd_edges":  n_cube_edges,
        "n_co_edges":  0,
        "t_mean_ms":   t_mean,
        "t_std_ms":    t_std,
    }
    methods.append(r)
    print(f"  [fd-cube]   n_params=0         edges={n_cube_edges:,}"
          f"  t={t_mean:.3f}±{t_std:.3f} ms")

    # ── Cross architectures ───────────────────────────────────────────────────
    for fd_order, n_co in zip(fd_orders, n_cos):
        print(f"\n  Building cross graph (N={N_c}, fd_order={fd_order}, n_co={n_co})...")
        try:
            fd_ei, fd_ea, co_ei, co_ea = build_star_graph(
                fd_order, n_co, N=N_c, d=d_c, grid_L=grid_L_c)
            fd_ei = fd_ei.to(dev); fd_ea = fd_ea.to(dev)
            co_ei = co_ei.to(dev); co_ea = co_ea.to(dev)
            n_fd_e = fd_ei.shape[1]
            n_co_e = co_ei.shape[1]
            print(f"  fd_edges={n_fd_e:,}  co_edges={n_co_e:,}")
        except Exception as exc:
            print(f"  ERROR building graph: {exc}")
            for arch in ("gnn-cross", "fd-cross", "so3-cross"):
                methods.append({
                    "label":      f"{arch}(p{fd_order},c{n_co})",
                    "model_type": arch,
                    "graph_type": "cross",
                    "fd_order":   fd_order,
                    "n_co":       n_co,
                    "error":      str(exc),
                })
            continue

        # gnn-cross
        model_gx = HamiltonianGNN_Cross(hidden_dim=hidden_dim).to(dev)
        model_gx.eval()
        n_p_gx = _count_params(model_gx)

        def _fn_gnn_cross(psi, _m=model_gx):
            nrm = torch.norm(psi) + 1e-30
            return _m(psi / nrm, fd_ei, fd_ea, co_ei, co_ea, V_t_c) * nrm

        t_mean, t_std = _time_fn(_fn_gnn_cross, psi_c, n_reps, n_warmup, dev)
        r = {
            "label":       f"gnn-cross(p{fd_order},c{n_co})",
            "model_type":  "gnn-cross",
            "graph_type":  "cross",
            "fd_order":    fd_order,
            "n_co":        n_co,
            "hidden_dim":  hidden_dim,
            "n_params":    n_p_gx,
            "n_fd_edges":  n_fd_e,
            "n_co_edges":  n_co_e,
            "t_mean_ms":   t_mean,
            "t_std_ms":    t_std,
        }
        methods.append(r)
        print(f"  [gnn-cross(p{fd_order},c{n_co})]  n_params={n_p_gx}"
              f"  t={t_mean:.3f}±{t_std:.3f} ms")

        # fd-cross
        fd_cross = FiniteDiffHamiltonian_Cross(fd_ei, fd_ea, V_t_c, dev)

        def _fn_fd_cross(psi, _fd=fd_cross):
            return _fd(psi)

        t_mean, t_std = _time_fn(_fn_fd_cross, psi_c, n_reps, n_warmup, dev)
        r = {
            "label":       f"fd-cross(p{fd_order},c{n_co})",
            "model_type":  "fd-cross",
            "graph_type":  "cross",
            "fd_order":    fd_order,
            "n_co":        n_co,
            "hidden_dim":  None,
            "n_params":    0,
            "n_fd_edges":  n_fd_e,
            "n_co_edges":  0,
            "t_mean_ms":   t_mean,
            "t_std_ms":    t_std,
        }
        methods.append(r)
        print(f"  [fd-cross(p{fd_order},c{n_co})]   n_params=0"
              f"  t={t_mean:.3f}±{t_std:.3f} ms")

        # so3-cross
        model_so3 = SO3HamiltonianNet(radial_hidden_dim=radial_hidden_dim).to(dev)
        model_so3.eval()
        n_p_so3 = _count_params(model_so3)

        def _fn_so3(psi, _m=model_so3):
            nrm = torch.norm(psi) + 1e-30
            return _m(psi / nrm, fd_ei, fd_ea, co_ei, co_ea, V_t_c) * nrm

        t_mean, t_std = _time_fn(_fn_so3, psi_c, n_reps, n_warmup, dev)
        r = {
            "label":            f"so3-cross(p{fd_order},c{n_co})",
            "model_type":       "so3",
            "graph_type":       "cross",
            "fd_order":         fd_order,
            "n_co":             n_co,
            "radial_hidden_dim": radial_hidden_dim,
            "n_params":         n_p_so3,
            "n_fd_edges":       n_fd_e,
            "n_co_edges":       n_co_e,
            "t_mean_ms":        t_mean,
            "t_std_ms":         t_std,
        }
        methods.append(r)
        print(f"  [so3-cross(p{fd_order},c{n_co})]  n_params={n_p_so3}"
              f"  t={t_mean:.3f}±{t_std:.3f} ms")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'Label':<34} {'n_params':>10}  {'t_mean_ms':>10}  {'t_std_ms':>9}")
    print(f"  {'─'*34} {'─'*10}  {'─'*10}  {'─'*9}")
    if fft_ref and 't_mean_ms' in fft_ref:
        print(f"  {'fft-fine (reference)':<34} {'—':>10}  "
              f"{fft_ref['t_mean_ms']:>10.3f}  {fft_ref['t_std_ms']:>9.3f}")
    for m in methods:
        if 'error' in m:
            print(f"  {m['label']:<34}  ERROR: {m['error']}")
        else:
            np_str = str(m.get('n_params', 0))
            print(f"  {m['label']:<34} {np_str:>10}  "
                  f"{m['t_mean_ms']:>10.3f}  {m['t_std_ms']:>9.3f}")
    print(f"{'─'*70}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/timing_benchmark.py",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "n_reps":            n_reps,
            "n_warmup":          n_warmup,
            "hidden_dim":        hidden_dim,
            "radial_hidden_dim": radial_hidden_dim,
            "fd_orders_tested":  fd_orders,
            "n_cos_tested":      n_cos,
            "device":            str(dev),
            "N_qd_coarse":       N_c,
            "d_coarse_bohr":     float(d_c),
            "n_grid_coarse":     n_c,
            "N_qd_fine":         N_f,
            "d_fine_bohr":       float(d_f),
            "n_grid_fine":       n_f,
        },
        "fft_fine_reference": fft_ref,
        "methods":             methods,
    }

    json_path = out_dir / f"timing_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Results → {json_path}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# time_random_cross — 随机权重 GNN/SO3 cross 模型计时（无需训练）
# ─────────────────────────────────────────────────────────────────────────────

def time_random_cross(
    n_co_list:         list  = None,
    model_types:       list  = None,
    fd_order:          int   = 6,
    hidden_dim:        int   = 16,
    radial_hidden_dim: int   = 32,
    n_reps:            int   = 10,
    n_warmup:          int   = 3,
    device:            str   = 'cpu',
    output_root:       str   = '.',
    description:       str   = '',
):
    """
    对随机初始化（未训练）的 GNN-cross / SO3-cross 模型在 V_sparse 上计时。

    用途：测试不同 n_co 带来的图规模（co 边数量）对 H-apply 耗时的影响，
    与训练权重无关。

    Parameters
    ----------
    n_co_list   : n_co 值列表（默认 [3, 4, 5]）
    model_types : 'gnn' 和/或 'so3'（默认 ['gnn', 'so3']）
    fd_order    : FD 精度阶数
    """
    import json as _json

    if n_co_list   is None: n_co_list   = [3, 4, 5]
    if model_types is None: model_types = ['gnn', 'so3']

    from .physics import N_sparse, V_sparse, d_sparse
    from .graph   import build_star_graph
    from .model   import (HamiltonianGNN_Cross, FiniteDiffHamiltonian_Cross,
                          SO3HamiltonianNet)

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_str = (device if device != 'auto' else
               ('cuda' if torch.cuda.is_available() else 'cpu'))
    dev     = torch.device(dev_str)

    n_grid  = N_sparse ** 3
    grid_L  = N_sparse * d_sparse
    V_t     = torch.tensor(V_sparse.flatten(), dtype=torch.float32).unsqueeze(-1).to(dev)

    rng     = np.random.default_rng(42)
    psi_np  = rng.standard_normal(n_grid).astype(np.float64)
    psi_np /= np.linalg.norm(psi_np)
    psi_t   = torch.tensor(psi_np.astype(np.float32),
                            dtype=torch.float32).unsqueeze(-1).to(dev)

    print(f"\n{'='*64}")
    print(f"  Random-weight cross timing  (N={N_sparse}, n={n_grid})")
    print(f"  fd_order={fd_order}  n_co_list={n_co_list}  model_types={model_types}")
    print(f"  n_reps={n_reps}  n_warmup={n_warmup}  device={dev_str}")
    print(f"{'='*64}")

    results = []

    for n_co in n_co_list:
        print(f"\n  Building graph  n_co={n_co}  (expected {n_co**3-1} co-neighbors/node)...")
        fd_ei, fd_ea, co_ei, co_ea = build_star_graph(
            fd_order, n_co, N=N_sparse, d=d_sparse, grid_L=grid_L)
        fd_ei = fd_ei.to(dev); fd_ea = fd_ea.to(dev)
        co_ei = co_ei.to(dev); co_ea = co_ea.to(dev)
        n_co_edges = co_ei.shape[1]
        print(f"    co_edges total = {n_co_edges:,}  "
              f"({n_co_edges // n_grid} per node, expected {n_co**3-1})")

        # FD 基准（只建一次 per n_co，不依赖模型类型）
        _fd = FiniteDiffHamiltonian_Cross(fd_ei, fd_ea, V_t, dev)

        def _fd_fn(psi, _f=_fd): return _f(psi)

        for _ in range(n_warmup):
            with torch.no_grad(): _fd_fn(psi_t)
        fd_times = []
        for _ in range(n_reps):
            if dev.type == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(): _fd_fn(psi_t)
            if dev.type == 'cuda': torch.cuda.synchronize()
            fd_times.append((time.perf_counter() - t0) * 1000.0)
        fd_mean, fd_std = float(np.mean(fd_times)), float(np.std(fd_times))
        print(f"    FD:   {fd_mean:.3f}±{fd_std:.3f} ms")
        results.append({
            "model_type":   "fd",
            "n_co":         n_co,
            "n_co_edges":   n_co_edges,
            "t_mean_ms":    fd_mean,
            "t_std_ms":     fd_std,
        })

        # GNN / SO3
        for mt in model_types:
            if mt == 'gnn':
                model = HamiltonianGNN_Cross(hidden_dim=hidden_dim).to(dev)
            else:
                model = SO3HamiltonianNet(radial_hidden_dim=radial_hidden_dim).to(dev)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())

            def _fn(psi, _m=model):
                nrm = torch.norm(psi) + 1e-30
                return _m(psi / nrm, fd_ei, fd_ea, co_ei, co_ea, V_t) * nrm

            for _ in range(n_warmup):
                with torch.no_grad(): _fn(psi_t)
            times = []
            for _ in range(n_reps):
                if dev.type == 'cuda': torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad(): _fn(psi_t)
                if dev.type == 'cuda': torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            t_mean, t_std = float(np.mean(times)), float(np.std(times))
            print(f"    {mt.upper():<4} (params={n_params}):  "
                  f"{t_mean:.3f}±{t_std:.3f} ms")
            results.append({
                "model_type":   mt,
                "n_co":         n_co,
                "n_co_edges":   n_co_edges,
                "hidden_dim":   hidden_dim if mt == 'gnn' else None,
                "radial_hidden_dim": radial_hidden_dim if mt == 'so3' else None,
                "n_params":     n_params,
                "t_mean_ms":    t_mean,
                "t_std_ms":     t_std,
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'model':<8}  {'n_co':>5}  {'co_edges':>10}  "
          f"{'t_mean_ms':>10}  {'t_std_ms':>9}")
    print(f"  {'─'*8}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*9}")
    for r in results:
        print(f"  {r['model_type']:<8}  {r['n_co']:>5}  "
              f"{r['n_co_edges']:>10,}  "
              f"{r['t_mean_ms']:>10.3f}  {r['t_std_ms']:>9.3f}")
    print(f"{'─'*60}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/timing_benchmark.py::time_random_cross",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "n_co_list":         n_co_list,
            "model_types":       model_types,
            "fd_order":          fd_order,
            "hidden_dim":        hidden_dim,
            "radial_hidden_dim": radial_hidden_dim,
            "n_reps":            n_reps,
            "n_warmup":          n_warmup,
            "device":            dev_str,
            "N_sparse":          N_sparse,
            "n_grid":            n_grid,
            "potential":         "V_sparse",
        },
        "results": results,
    }
    json_path = out_dir / f"arch_timing_rand_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Results → {json_path}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# time_trained_gnn — 对已训练 arch_sweep 模型计时
# ─────────────────────────────────────────────────────────────────────────────

def time_trained_gnn(
    n_co_list:   list  = None,
    model_types: list  = None,
    fd_order:    int   = 6,
    hidden_dim:  int   = 16,
    epochs:      int   = 5000,
    n_reps:      int   = 10,
    n_warmup:    int   = 3,
    use_qd:      bool  = False,
    cube_file:   str   = None,
    params_file: str   = None,
    device:      str   = 'cpu',
    output_root: str   = '.',
    description: str   = '',
):
    """
    对 arch_sweep 训练好的 GNN / SO3 模型计时。

    模型路径约定（与 arch_sweep_experiment 一致）：
        output_root/gnn_models/arch_fd{fd_order}_nco{n_co}_{model_type}_hd{hidden_dim}_ep{epochs}/

    Parameters
    ----------
    use_qd      : False → 在 V_sparse 训练势（N_sparse³）上计时；
                  True  → 在真实 QD 势（d_sparse 重采样，N_qd³）上计时
    cube_file   : QD cube 文件路径（use_qd=True 时有效，默认 localPot.cube）
    params_file : Gaussian fit params JSON 路径（use_qd=True 时有效）
    n_reps      : 计时重复次数（取均值，默认 10）
    n_warmup    : 热身次数（不计入统计，默认 3）
    """
    import os
    import json as _json

    if n_co_list is None:
        n_co_list = [3, 4, 5]
    if model_types is None:
        model_types = ['gnn', 'so3']

    from .gnn_operator import build_gnn_operator
    from .physics import N_sparse, V_sparse, d_sparse

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_str = (device if device != 'auto' else
               ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── 确定势能和网格 ────────────────────────────────────────────────────────
    if use_qd:
        from .test_filter import _load_qd_potential, _DEFAULT_CUBE, _DEFAULT_PARAMS
        _cube   = cube_file   or _DEFAULT_CUBE
        _params = params_file or _DEFAULT_PARAMS
        print(f"\n  Loading real QD potential (d={d_sparse} Bohr)...")
        pot_grid, N_grid, d_actual = _load_qd_potential(
            d_sparse, cube_file=_cube, params_file=_params)
        V_ext  = pot_grid.potential.ravel().astype(np.float64)
        n_grid = N_grid ** 3
        pot_label = f"QD (N={N_grid}, n={n_grid:,}, d={d_actual:.4f} Bohr)"
        gnn_op_kw = {"V_ext": V_ext, "N_grid": N_grid}
    else:
        N_grid    = N_sparse
        V_ext     = None
        n_grid    = N_sparse ** 3
        pot_label = f"V_sparse (N={N_sparse}, n={n_grid})"
        gnn_op_kw = {}

    rng    = np.random.default_rng(42)
    psi_np = rng.standard_normal(n_grid).astype(np.float64)
    psi_np /= np.linalg.norm(psi_np)

    print(f"\n{'='*64}")
    print(f"  GNN trained-model timing on {pot_label}")
    print(f"  fd_order={fd_order}  hidden_dim={hidden_dim}  epochs={epochs}")
    print(f"  n_co_list={n_co_list}  model_types={model_types}")
    print(f"  n_reps={n_reps}  n_warmup={n_warmup}  device={dev_str}")
    print(f"{'='*64}")

    results = []

    # ── 内部计时工具 ──────────────────────────────────────────────────────────
    def _run_timing(op):
        for _ in range(n_warmup):
            op.matvec(psi_np)
        ts_list = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            op.matvec(psi_np)
            ts_list.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(ts_list)), float(np.std(ts_list))

    # ── FD 基准（use_fd=True，用第一个可用模型的图）─────────────────────────────
    fd_row = None
    for n_co in n_co_list:
        for mt in model_types:
            run_name = f"arch_fd{fd_order}_nco{n_co}_{mt}_hd{hidden_dim}_ep{epochs}"
            run_path = os.path.join(output_root, "gnn_models", run_name)
            if os.path.isdir(run_path):
                try:
                    fd_op = build_gnn_operator(
                        run_path, use_fd=True, device=dev_str, **gnn_op_kw)
                    t_fd_mean, t_fd_std = _run_timing(fd_op)
                    fd_row = {
                        "label":      f"FD (order={fd_order})",
                        "model_type": "fd",
                        "n_co":       n_co,
                        "fd_order":   fd_order,
                        "t_mean_ms":  t_fd_mean,
                        "t_std_ms":   t_fd_std,
                    }
                    print(f"\n  [FD order={fd_order}]"
                          f"  t={t_fd_mean:.3f}±{t_fd_std:.3f} ms")
                except Exception as exc:
                    print(f"\n  [FD] FAILED: {exc}")
                break
        if fd_row is not None:
            break

    # ── GNN / SO3 逐 (n_co, model_type) 计时 ─────────────────────────────────
    for mt in model_types:
        for n_co in n_co_list:
            run_name = f"arch_fd{fd_order}_nco{n_co}_{mt}_hd{hidden_dim}_ep{epochs}"
            run_path = os.path.join(output_root, "gnn_models", run_name)
            label    = f"{mt.upper()} n_co={n_co}"

            if not os.path.isdir(run_path):
                print(f"\n  [{label}]  SKIP — not found: {run_path}")
                results.append({
                    "label": label, "model_type": mt, "n_co": n_co,
                    "fd_order": fd_order, "error": "run_dir not found",
                })
                continue

            try:
                gnn_op = build_gnn_operator(
                    run_path, use_fd=False, device=dev_str, **gnn_op_kw)
                t_mean, t_std = _run_timing(gnn_op)
            except Exception as exc:
                print(f"\n  [{label}]  FAILED: {exc}")
                results.append({
                    "label": label, "model_type": mt, "n_co": n_co,
                    "fd_order": fd_order, "error": str(exc),
                })
                continue

            print(f"  [{label}]  t={t_mean:.3f}±{t_std:.3f} ms")
            results.append({
                "label":      label,
                "model_type": mt,
                "n_co":       n_co,
                "fd_order":   fd_order,
                "run_dir":    run_path,
                "t_mean_ms":  t_mean,
                "t_std_ms":   t_std,
            })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*54}")
    print(f"  {'Label':<22}  {'t_mean_ms':>10}  {'t_std_ms':>9}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*9}")
    if fd_row:
        print(f"  {fd_row['label']:<22}  {fd_row['t_mean_ms']:>10.3f}"
              f"  {fd_row['t_std_ms']:>9.3f}")
    for r in results:
        if 'error' in r:
            print(f"  {r['label']:<22}  SKIP/ERROR")
        else:
            print(f"  {r['label']:<22}  {r['t_mean_ms']:>10.3f}"
                  f"  {r['t_std_ms']:>9.3f}")
    print(f"{'─'*54}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/timing_benchmark.py::time_trained_gnn",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "n_co_list":   n_co_list,
            "model_types": model_types,
            "fd_order":    fd_order,
            "hidden_dim":  hidden_dim,
            "epochs":      epochs,
            "n_reps":      n_reps,
            "n_warmup":    n_warmup,
            "device":      dev_str,
            "potential":   "qd" if use_qd else "V_sparse",
            "N_grid":      N_grid,
            "n_grid":      n_grid,
        },
        "fd_baseline": fd_row,
        "results":     results,
    }
    json_path = out_dir / f"arch_timing_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Results → {json_path}")
    return output
