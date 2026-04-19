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
