"""
gnn_code/fd_benchmark.py — FD orders vs FFT: timing + accuracy on QD

比较不同阶数的 FD Hamiltonian 与 FFT 的时间和精度：
  · fd-cross(p)  : FiniteDiffHamiltonian_Cross，fd_order=p（p=2,4,6,8,10）
  · fd-cube      : FiniteDiffHamiltonian（3×3×3 Mehrstellen，26 近邻）
  · fft-coarse   : FFT 在粗网格上的结果（精度参考，不计入 timing）
  · fft-fine     : FFT 在细网格上的结果（timing 参考，不计入精度）

精度（accuracy）
-----------
在同一粗网格上，使用若干光滑余弦测试态（低频模式）：
  ψ_{nx,ny,nz}(r) = cos(2π·nx·i/N) · cos(2π·ny·j/N) · cos(2π·nz·k/N)
比较各方法的瑞利商 E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩。
以 FFT-coarse（同网格，无动能截断）为参考值。
误差 = |E_FD - E_FFT_coarse| / |E_FFT_coarse|

计时（timing）
-------
FD 各阶在粗网格上计时（n_reps 次均值）。
FFT 在细网格上计时（timing 参考基准线）。

用法（通过 run_gnn.py）：
    python run_gnn.py --mode fd_compare \\
        --fd_compare_orders 2 4 6 8 10 \\
        --timing_n_reps 100 \\
        --timing_description "FD order vs FFT comparison"
"""

import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch

from .physics import d_sparse
from .graph   import build_graph, build_star_graph
from .model   import FiniteDiffHamiltonian, FiniteDiffHamiltonian_Cross
from .test_filter import _DEFAULT_CUBE, _DEFAULT_PARAMS, _load_qd_potential


# ── Test wavefunction modes (smooth, low-frequency) ──────────────────────────
_TEST_MODES = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1),
    (1, 1, 1),
    (2, 0, 0), (2, 1, 0), (2, 1, 1),
]


def _make_cosine_state(nx: int, ny: int, nz: int, N: int) -> np.ndarray:
    """Normalized cosine product mode on N³ grid (PBC, row-major)."""
    idx = np.arange(N, dtype=np.float64) / N   # [0, 1)
    x, y, z = np.meshgrid(idx, idx, idx, indexing='ij')
    psi = (np.cos(2 * np.pi * nx * x) *
           np.cos(2 * np.pi * ny * y) *
           np.cos(2 * np.pi * nz * z)).ravel().astype(np.float32)
    nrm = np.linalg.norm(psi)
    return psi / nrm if nrm > 1e-30 else psi


def _fft_apply_coarse(psi_flat: np.ndarray, N: int, d: float,
                      V_flat: np.ndarray) -> np.ndarray:
    """FFT Hamiltonian on coarse grid (no kinetic cutoff) — accuracy reference."""
    psi_3d = psi_flat.reshape(N, N, N).astype(np.float64)
    kx = np.fft.fftfreq(N, d=d) * 2 * np.pi
    Kx, Ky, Kz = np.meshgrid(kx, kx, kx, indexing='ij')
    T_k   = (Kx**2 + Ky**2 + Kz**2) / 2.0
    T_psi = np.fft.ifftn(T_k * np.fft.fftn(psi_3d)).real
    V_psi = V_flat.reshape(N, N, N).astype(np.float64) * psi_3d
    return (T_psi + V_psi).ravel().astype(np.float32)


def _rayleigh(psi: np.ndarray, Hpsi: np.ndarray) -> float:
    return float(np.dot(psi.astype(np.float64), Hpsi.astype(np.float64)) /
                 np.dot(psi.astype(np.float64), psi.astype(np.float64)))


def _time_fn(fn, psi_t: torch.Tensor, n_reps: int, n_warmup: int,
             device: torch.device) -> tuple:
    is_cuda = device.type == 'cuda'
    for _ in range(n_warmup):
        with torch.no_grad():
            fn(psi_t)
    if is_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fn(psi_t)
        if is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def fd_accuracy_timing(
    fd_orders:   list = None,
    n_reps:      int  = 100,
    n_warmup:    int  = 10,
    device:      str  = 'cpu',
    cube_file:   str  = _DEFAULT_CUBE,
    params_file: str  = _DEFAULT_PARAMS,
    output_root: str  = '.',
    description: str  = '',
):
    """
    Compare FD of various orders against FFT in time and accuracy.

    Parameters
    ----------
    fd_orders : fd_order values to test (default [2,4,6,8,10])
    """
    if fd_orders is None:
        fd_orders = [2, 4, 6, 8, 10]

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev     = torch.device(device if device != 'auto' else
                           ('cuda' if torch.cuda.is_available() else 'cpu'))

    print(f"\n{'='*64}")
    print(f"  FD benchmark: orders {fd_orders}")
    print(f"  n_reps={n_reps}  n_warmup={n_warmup}  device={dev}")
    print(f"{'='*64}")

    # ── Coarse QD grid ────────────────────────────────────────────────────────
    print("\n  Loading QD potential (coarse grid)...")
    pot_c, N_c, d_c = _load_qd_potential(
        d_sparse, cube_file=cube_file, params_file=params_file)
    n_c      = N_c ** 3
    V_flat_c = pot_c.potential.ravel().astype(np.float32)
    grid_L_c = N_c * d_c
    V_t_c    = torch.tensor(V_flat_c, dtype=torch.float32).unsqueeze(-1).to(dev)

    # ── Fine QD grid (FFT timing reference) ───────────────────────────────────
    d_fine = d_c / 2.0
    print(f"\n  Loading QD potential (fine grid, d={d_fine:.4f} Bohr)...")
    pot_f, N_f, d_f = _load_qd_potential(
        d_fine, cube_file=cube_file, params_file=params_file)
    n_f = N_f ** 3

    # ── Test states (cosine modes on coarse grid) ─────────────────────────────
    test_states = [(nx, ny, nz,
                    _make_cosine_state(nx, ny, nz, N_c))
                   for nx, ny, nz in _TEST_MODES]
    print(f"\n  Test modes: {len(test_states)}")

    # ── FFT coarse (accuracy reference) ───────────────────────────────────────
    print("  Computing FFT-coarse reference energies...")
    E_fft_c = {}
    for nx, ny, nz, psi in test_states:
        Hpsi = _fft_apply_coarse(psi, N_c, d_c, V_flat_c)
        E_fft_c[(nx, ny, nz)] = _rayleigh(psi, Hpsi)

    # ── Random vector for timing ──────────────────────────────────────────────
    rng   = np.random.default_rng(42)
    psi_t_c = torch.tensor(rng.standard_normal(n_c).astype(np.float32),
                            dtype=torch.float32, device=dev).unsqueeze(-1)
    psi_t_f = torch.tensor(rng.standard_normal(n_f).astype(np.float32),
                            dtype=torch.float32, device=dev).unsqueeze(-1)

    methods  = []
    fft_ref  = None

    # ── FFT fine (timing reference) ───────────────────────────────────────────
    try:
        from fft_code.hamiltonian import apply_H as _fft_apply_H
        V_3d_f = pot_f.potential.astype(np.float32)
        kx_f   = np.fft.fftfreq(N_f, d=d_f) * 2 * np.pi
        Kx_f, Ky_f, Kz_f = np.meshgrid(kx_f, kx_f, kx_f, indexing='ij')
        T_k_f  = (Kx_f**2 + Ky_f**2 + Kz_f**2) / 2.0

        def _fft_fine_fn(psi_t):
            psi_np = psi_t.cpu().numpy().reshape(N_f, N_f, N_f)
            out    = _fft_apply_H(psi_np, T_k_f, V_3d_f)
            return torch.tensor(np.real(out).ravel(), dtype=torch.float32,
                                device=dev).unsqueeze(-1)

        t_mean, t_std = _time_fn(_fft_fine_fn, psi_t_f, n_reps, n_warmup, dev)
        fft_ref = {"label": "fft-fine", "N_grid": N_f, "d_bohr": float(d_f),
                   "t_mean_ms": t_mean, "t_std_ms": t_std}
        print(f"  [fft-fine]  N={N_f}  t={t_mean:.3f}±{t_std:.3f} ms  (timing ref)")
    except Exception as exc:
        fft_ref = {"label": "fft-fine", "error": str(exc)}
        print(f"  [fft-fine] SKIP: {exc}")

    # ── fd-cube (Mehrstellen 26-point) ────────────────────────────────────────
    print(f"\n  Building cube graph (N={N_c})...")
    ei_c, ea_c = build_graph(N=N_c, d=d_c, grid_L=grid_L_c)
    ei_c = ei_c.to(dev); ea_c = ea_c.to(dev)
    n_cube_edges = ei_c.shape[1]
    fd_cube = FiniteDiffHamiltonian(ei_c, ea_c, V_t_c, dev)

    # accuracy
    E_cube = {}
    for nx, ny, nz, psi in test_states:
        psi_t = torch.tensor(psi, dtype=torch.float32, device=dev).unsqueeze(-1)
        with torch.no_grad():
            Hpsi_t = fd_cube(psi_t)
        E_cube[(nx, ny, nz)] = _rayleigh(psi, Hpsi_t.cpu().numpy().ravel())

    errs_cube = [abs(E_cube[k] - E_fft_c[k]) / (abs(E_fft_c[k]) + 1e-30)
                 for k in E_fft_c]
    E_mean_cube = float(np.mean([E_cube[k] for k in E_fft_c]))

    # timing
    def _fn_cube(psi_t_):
        return fd_cube(psi_t_)

    t_mean, t_std = _time_fn(_fn_cube, psi_t_c, n_reps, n_warmup, dev)
    r = {
        "label":        "fd-cube",
        "model_type":   "fd-cube",
        "fd_order":     None,
        "n_co":         None,
        "n_fd_edges":   n_cube_edges,
        "E_mean":       E_mean_cube,
        "E_err_mean":   float(np.mean(errs_cube)),
        "E_err_max":    float(np.max(errs_cube)),
        "E_per_mode":   {f"{k}": float(E_cube[k]) for k in E_cube},
        "t_mean_ms":    t_mean,
        "t_std_ms":     t_std,
    }
    methods.append(r)
    print(f"  [fd-cube]  edges={n_cube_edges:,}"
          f"  E_err_mean={r['E_err_mean']:.3e}  max={r['E_err_max']:.3e}"
          f"  t={t_mean:.3f}±{t_std:.3f} ms")

    # ── fd-cross(p) for each order ────────────────────────────────────────────
    for fd_order in fd_orders:
        print(f"\n  Building cross graph (N={N_c}, fd_order={fd_order})...")
        try:
            fd_ei, fd_ea, co_ei, co_ea = build_star_graph(
                fd_order, 1, N=N_c, d=d_c, grid_L=grid_L_c)
            fd_ei = fd_ei.to(dev); fd_ea = fd_ea.to(dev)
            n_fd_edges = fd_ei.shape[1]
            fd_cross = FiniteDiffHamiltonian_Cross(fd_ei, fd_ea, V_t_c, dev)
            print(f"  n_fd_edges={n_fd_edges:,}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            methods.append({"label": f"fd-cross(p{fd_order})", "error": str(exc),
                             "fd_order": fd_order})
            continue

        # accuracy
        E_cross = {}
        for nx, ny, nz, psi in test_states:
            psi_t = torch.tensor(psi, dtype=torch.float32,
                                 device=dev).unsqueeze(-1)
            with torch.no_grad():
                Hpsi_t = fd_cross(psi_t)
            E_cross[(nx, ny, nz)] = _rayleigh(psi, Hpsi_t.cpu().numpy().ravel())

        errs = [abs(E_cross[k] - E_fft_c[k]) / (abs(E_fft_c[k]) + 1e-30)
                for k in E_fft_c]
        E_mean_cross = float(np.mean([E_cross[k] for k in E_fft_c]))

        # timing
        def _fn_cross(psi_t_, _fd=fd_cross):
            return _fd(psi_t_)

        t_mean, t_std = _time_fn(_fn_cross, psi_t_c, n_reps, n_warmup, dev)
        r = {
            "label":        f"fd-cross(p{fd_order})",
            "model_type":   "fd-cross",
            "fd_order":     fd_order,
            "n_co":         1,
            "n_fd_edges":   n_fd_edges,
            "E_mean":       E_mean_cross,
            "E_err_mean":   float(np.mean(errs)),
            "E_err_max":    float(np.max(errs)),
            "E_per_mode":   {f"{k}": float(E_cross[k]) for k in E_cross},
            "t_mean_ms":    t_mean,
            "t_std_ms":     t_std,
        }
        methods.append(r)
        print(f"  [fd-cross(p{fd_order})]  edges={n_fd_edges:,}"
              f"  E_err_mean={r['E_err_mean']:.3e}  max={r['E_err_max']:.3e}"
              f"  t={t_mean:.3f}±{t_std:.3f} ms")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  {'Label':<22} {'n_fd_edges':>12}  {'t_ms':>8}  "
          f"{'E_err_mean':>12}  {'E_err_max':>12}")
    print(f"  {'─'*22} {'─'*12}  {'─'*8}  {'─'*12}  {'─'*12}")
    if fft_ref and 't_mean_ms' in fft_ref:
        print(f"  {'fft-fine (ref)':<22} {'—':>12}  "
              f"{fft_ref['t_mean_ms']:>8.3f}  {'—':>12}  {'—':>12}")
    for m in methods:
        if 'error' in m:
            print(f"  {m['label']:<22}  ERROR: {m['error']}")
        else:
            print(f"  {m['label']:<22} {m['n_fd_edges']:>12,}  "
                  f"{m['t_mean_ms']:>8.3f}  "
                  f"{m['E_err_mean']:>12.3e}  {m['E_err_max']:>12.3e}")
    print(f"{'─'*78}")
    print(f"  Accuracy reference: fft-coarse (same {N_c}³ grid, no kinetic cutoff)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/fd_benchmark.py",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "fd_orders":        fd_orders,
            "n_reps":           n_reps,
            "n_warmup":         n_warmup,
            "device":           str(dev),
            "N_qd_coarse":      N_c,
            "d_coarse_bohr":    float(d_c),
            "n_grid_coarse":    n_c,
            "N_qd_fine":        N_f,
            "d_fine_bohr":      float(d_f),
            "n_grid_fine":      n_f,
            "test_modes":       list(_TEST_MODES),
            "accuracy_reference": "fft-coarse (same grid, no kinetic cutoff)",
        },
        "fft_coarse_reference_energies": {
            f"{k}": float(v) for k, v in E_fft_c.items()
        },
        "fft_fine_reference": fft_ref,
        "methods":            methods,
    }

    json_path = out_dir / f"fd_benchmark_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Results → {json_path}")
    return output
