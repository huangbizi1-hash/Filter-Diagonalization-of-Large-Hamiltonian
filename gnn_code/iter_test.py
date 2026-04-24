"""
gnn_code/iter_test.py — Repeated H-apply fidelity & energy test (FD vs FFT)

对 k=0（均匀态）和 k=1（(1,0,0) 余弦平面波）初态施加 1/10/50/100/200/500 次 H，
比较各阶 FD 与两套 FFT 标准值的保真度和能量偏差：
  · fft-coarse : 粗网格 FFT（与 FD 同网格，动能截断 kinetic_cutoff）
  · fft-fine   : 细网格（d/2）FFT（同截断），作用完后 Fourier 截断下采样到粗网格

用法（通过 run_gnn.py）：
    python run_gnn.py --mode iter_test \\
        --iter_fd_orders 2 4 6 8 10 \\
        --iter_n_steps 1 10 50 100 200 500 \\
        --iter_kinetic_cutoff 30.0
"""

import datetime
import json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .physics    import d_sparse
from .test_filter import _DEFAULT_CUBE, _DEFAULT_PARAMS, _load_qd_potential
from .graph      import build_star_graph
from .model      import FiniteDiffHamiltonian_Cross

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── 初态构建 ──────────────────────────────────────────────────────────────────

def _make_k0_state(N: int) -> np.ndarray:
    """k=0 均匀归一化态。"""
    return np.ones(N ** 3, dtype=np.float64) / np.sqrt(N ** 3)


def _make_k1_state(N: int) -> np.ndarray:
    """k=1 余弦平面波 cos(2π·ix/N)，仅 x 方向，归一化。"""
    ix = np.arange(N, dtype=np.float64)
    psi = (np.cos(2 * np.pi * ix / N)[:, None, None]
           * np.ones((N, N, N), dtype=np.float64)).ravel()
    nrm = np.linalg.norm(psi)
    return psi / nrm if nrm > 1e-30 else psi


# ── FFT Hamiltonian（预计算 T_k）────────────────────────────────────────────

def _precompute_Tk(N: int, d: float, kinetic_cutoff: float) -> np.ndarray:
    """预计算截断后的 N³ 动能格点 T(k)（形状 (N,N,N)，已截断）。"""
    kx = np.fft.fftfreq(N, d=d) * 2 * np.pi
    Kx, Ky, Kz = np.meshgrid(kx, kx, kx, indexing='ij')
    T_k = (Kx ** 2 + Ky ** 2 + Kz ** 2) / 2.0
    return np.minimum(T_k, kinetic_cutoff)


def _fft_apply(psi_flat: np.ndarray, N: int, T_k: np.ndarray,
               V_flat: np.ndarray) -> np.ndarray:
    """一次 FFT H-apply（动能已截断）。"""
    psi_3d = psi_flat.reshape(N, N, N)
    T_psi  = np.fft.ifftn(T_k * np.fft.fftn(psi_3d)).real
    V_psi  = V_flat.reshape(N, N, N) * psi_3d
    return (T_psi + V_psi).ravel()


# ── Fourier 截断下采样：细网格 → 粗网格 ──────────────────────────────────────

def _downsample_fourier(psi_fine: np.ndarray, N_c: int, N_f: int) -> np.ndarray:
    """
    Fourier 截断法将细网格（N_f³）波函数下采样到粗网格（N_c³）。
    取频域中心 N_c³ 块，等价于低通滤波后在粗网格采样。
    返回未归一化的粗网格向量（调用方负责归一化）。
    """
    psi_k       = np.fft.fftn(psi_fine.reshape(N_f, N_f, N_f))
    psi_k_shift = np.fft.fftshift(psi_k)   # DC 移至中心
    start       = (N_f - N_c) // 2
    end         = start + N_c
    crop        = psi_k_shift[start:end, start:end, start:end].copy()
    psi_k_out   = np.fft.ifftshift(crop)
    return np.fft.ifftn(psi_k_out).real.ravel()


# ── 辅助：归一化 + 瑞利商 ────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v.copy()


def _rayleigh(psi_n: np.ndarray, N: int, T_k: np.ndarray,
              V_flat: np.ndarray) -> float:
    """<ψ_n|H|ψ_n>，ψ_n 应已归一化。"""
    Hpsi = _fft_apply(psi_n, N, T_k, V_flat)
    return float(np.dot(psi_n, Hpsi))


# ── 构建 FD 算符（torch）────────────────────────────────────────────────────

def _build_fd_ops(fd_orders, N_c, d_c, V_flat_c, device):
    """
    返回 dict: fd_order → callable (psi_np_float64 → psi_np_float64)。
    """
    import torch
    grid_L = N_c * d_c
    V_t    = torch.tensor(V_flat_c.astype(np.float32),
                          dtype=torch.float32, device=device).unsqueeze(-1)

    fd_ops = {}
    for order in fd_orders:
        print(f"  Building FD-cross(p{order}) (N={N_c})...")
        try:
            fd_ei, fd_ea, co_ei, co_ea = build_star_graph(
                order, 1, N=N_c, d=d_c, grid_L=grid_L)
            fd_ei = fd_ei.to(device)
            fd_ea = fd_ea.to(device)
            model  = FiniteDiffHamiltonian_Cross(fd_ei, fd_ea, V_t, device)

            def _apply(psi_np, _m=model, _dev=device):
                psi_t = torch.tensor(
                    psi_np.astype(np.float32),
                    dtype=torch.float32, device=_dev).unsqueeze(-1)
                with torch.no_grad():
                    out = _m(psi_t)
                return out.cpu().numpy().ravel().astype(np.float64)

            fd_ops[order] = _apply
            print(f"  [FD-cross(p{order})] ready. n_fd_edges={fd_ei.shape[1]:,}")
        except Exception as exc:
            print(f"  [FD-cross(p{order})] FAILED: {exc}")
    return fd_ops


# ── 主函数 ────────────────────────────────────────────────────────────────────

def iter_fidelity_test(
    fd_orders:       list  = None,
    n_steps_list:    list  = None,
    kinetic_cutoff:  float = 30.0,
    cube_file:       str   = _DEFAULT_CUBE,
    params_file:     str   = _DEFAULT_PARAMS,
    output_root:     str   = ".",
    description:     str   = "",
):
    """
    对每种初态（k=0, k=1）及每阶 FD，重复施加 H，在各 checkpoint 处记录：
      · fidelity_vs_fft_coarse : |<ψ_FD | ψ_fft_coarse>|
      · fidelity_vs_fft_fine   : |<ψ_FD | ψ_fft_fine_downsampled>|
      · energy_FD              : <ψ_FD|H_fft_coarse|ψ_FD>  (瑞利商)
      · energy_fft_coarse      : <ψ_fft_coarse|H|ψ_fft_coarse>
      · energy_fft_fine_ds     : <ψ_fft_fine_ds|H|ψ_fft_fine_ds>
      · fidelity_fine_vs_coarse: |<ψ_fft_fine_ds | ψ_fft_coarse>|  (网格误差)
    """
    if fd_orders    is None: fd_orders    = [2, 4, 6, 8, 10]
    if n_steps_list is None: n_steps_list = [1, 10, 50, 100, 200, 500]

    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for iter_fidelity_test.")

    import torch
    device = torch.device("cpu")

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(set(n_steps_list))
    n_max       = max(checkpoints)

    print(f"\n{'='*64}")
    print(f"  Iter-fidelity test: fd_orders={fd_orders}")
    print(f"  checkpoints={checkpoints}  kinetic_cutoff={kinetic_cutoff}")
    print(f"{'='*64}")

    # ── 粗网格 ────────────────────────────────────────────────────────────────
    print("\n  Loading QD potential (coarse grid, d=d_sparse)...")
    pot_c, N_c, d_c = _load_qd_potential(
        d_sparse, cube_file=cube_file, params_file=params_file)
    V_flat_c = pot_c.potential.ravel().astype(np.float64)
    T_k_c    = _precompute_Tk(N_c, d_c, kinetic_cutoff)
    print(f"  Coarse: N={N_c}, d={d_c:.4f} Bohr, n_grid={N_c**3:,}")

    # ── 细网格 ────────────────────────────────────────────────────────────────
    d_fine = d_c / 2.0
    print(f"\n  Loading QD potential (fine grid, d={d_fine:.4f} Bohr)...")
    pot_f, N_f, d_f = _load_qd_potential(
        d_fine, cube_file=cube_file, params_file=params_file)
    V_flat_f = pot_f.potential.ravel().astype(np.float64)
    T_k_f    = _precompute_Tk(N_f, d_f, kinetic_cutoff)
    print(f"  Fine:   N={N_f}, d={d_f:.4f} Bohr, n_grid={N_f**3:,}")

    # ── FD 算符 ───────────────────────────────────────────────────────────────
    print()
    fd_ops = _build_fd_ops(fd_orders, N_c, d_c, V_flat_c.astype(np.float32), device)
    active_orders = [o for o in fd_orders if o in fd_ops]

    # ── 初态 ──────────────────────────────────────────────────────────────────
    init_states = {
        "k=0": {
            "coarse": _make_k0_state(N_c),
            "fine":   _make_k0_state(N_f),
        },
        "k=1": {
            "coarse": _make_k1_state(N_c),
            "fine":   _make_k1_state(N_f),
        },
    }

    # ── 主迭代循环 ────────────────────────────────────────────────────────────
    all_results = {}

    for state_label, states in init_states.items():
        print(f"\n  {'─'*60}")
        print(f"  Initial state: {state_label}")
        print(f"  {'─'*60}")

        # 每个方法的当前向量（每步归一化，避免 |λ_max|^n 溢出；等价于 power iteration）
        psi_fd    = {o: states["coarse"].copy() for o in active_orders}
        psi_fft_c = states["coarse"].copy()
        psi_fft_f = states["fine"].copy()

        step_results = {str(n): {} for n in checkpoints}

        for step in range(1, n_max + 1):
            # 推进一步并立即归一化（H^n 模随 |λ_max|^n 指数增长，不归一化会溢出）
            for o in active_orders:
                v = fd_ops[o](psi_fd[o])
                psi_fd[o] = _normalize(v)
            psi_fft_c = _normalize(_fft_apply(psi_fft_c, N_c, T_k_c, V_flat_c))
            psi_fft_f = _normalize(_fft_apply(psi_fft_f, N_f, T_k_f, V_flat_f))

            if step not in checkpoints:
                continue

            print(f"  step={step}")

            # 参考态已归一化
            ref_c    = psi_fft_c
            ref_f_ds = _normalize(_downsample_fourier(psi_fft_f, N_c, N_f))

            # FFT-fine vs FFT-coarse（网格差异）
            fid_fine_vs_coarse = float(abs(np.dot(ref_f_ds, ref_c)))
            E_fft_c  = _rayleigh(ref_c,    N_c, T_k_c, V_flat_c)
            E_fft_f  = _rayleigh(ref_f_ds, N_c, T_k_c, V_flat_c)

            entry = {
                "fft_coarse": {
                    "energy": E_fft_c,
                },
                "fft_fine_downsampled": {
                    "energy":               E_fft_f,
                    "fidelity_vs_fft_coarse": fid_fine_vs_coarse,
                },
                "fd": {},
            }

            for o in active_orders:
                psi_n   = _normalize(psi_fd[o].copy())
                fid_c   = float(abs(np.dot(psi_n, ref_c)))
                fid_f   = float(abs(np.dot(psi_n, ref_f_ds)))
                E_fd    = _rayleigh(psi_n, N_c, T_k_c, V_flat_c)
                entry["fd"][str(o)] = {
                    "fidelity_vs_fft_coarse": fid_c,
                    "fidelity_vs_fft_fine":   fid_f,
                    "energy":                 E_fd,
                    "energy_err_vs_fft_coarse": E_fd - E_fft_c,
                    "energy_err_vs_fft_fine":   E_fd - E_fft_f,
                }
                print(f"    FD-cross(p{o:2d})  "
                      f"fid_c={fid_c:.4f}  fid_f={fid_f:.4f}  "
                      f"dE_c={E_fd - E_fft_c:+.4e}")

            step_results[str(step)] = entry

        all_results[state_label] = step_results

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    state_labels = list(init_states.keys())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(active_orders)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='col')
    col_titles = ["Fidelity vs FFT-coarse",
                  "Fidelity vs FFT-fine (downsampled)",
                  "Energy error vs FFT-coarse (Ha)"]

    for row, sl in enumerate(state_labels):
        sr  = all_results[sl]
        xs  = checkpoints
        ax0, ax1, ax2 = axes[row, 0], axes[row, 1], axes[row, 2]

        # fft-fine vs fft-coarse 参考线
        fid_fnvc = [sr[str(n)]["fft_fine_downsampled"]["fidelity_vs_fft_coarse"]
                    for n in xs]
        ax0.plot(xs, fid_fnvc, "k--", linewidth=1.0,
                 label="fft-fine↓ vs coarse", zorder=3)

        for ci, o in enumerate(active_orders):
            fid_c = [sr[str(n)]["fd"][str(o)]["fidelity_vs_fft_coarse"]     for n in xs]
            fid_f = [sr[str(n)]["fd"][str(o)]["fidelity_vs_fft_fine"]        for n in xs]
            dE_c  = [sr[str(n)]["fd"][str(o)]["energy_err_vs_fft_coarse"]    for n in xs]
            lbl   = f"FD-cross(p{o})"
            c     = colors[ci]
            ax0.plot(xs, fid_c, "o-", color=c, linewidth=1.5, markersize=4, label=lbl)
            ax1.plot(xs, fid_f, "o-", color=c, linewidth=1.5, markersize=4, label=lbl)
            ax2.plot(xs, dE_c,  "o-", color=c, linewidth=1.5, markersize=4, label=lbl)

        for ax in (ax0, ax1):
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylim(-0.05, 1.1)
        ax2.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")

        for ax in (ax0, ax1, ax2):
            ax.set_xscale("log")
            ax.set_xlabel("# H-applies")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.set_title(f"{col_titles[[ax0,ax1,ax2].index(ax)]}  [{sl}]",
                         fontsize=9)

        ax0.set_ylabel("Fidelity")
        ax2.set_ylabel("Energy error (Ha)")
        ax0.legend(fontsize=7, loc="lower left")

    fig.suptitle(f"FD repeated H-apply: fidelity & energy vs FFT "
                 f"(cutoff={kinetic_cutoff} Ha)", fontsize=11)
    fig.tight_layout()
    png_path = out_dir / f"iter_test_{ts}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"\n  Plot → {png_path}")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    output = {
        "description": description,
        "script":      "gnn_code/iter_test.py",
        "datetime":    datetime.datetime.now().isoformat(),
        "config": {
            "fd_orders":       fd_orders,
            "active_orders":   active_orders,
            "n_steps_list":    n_steps_list,
            "kinetic_cutoff":  kinetic_cutoff,
            "N_coarse":        N_c,
            "d_coarse_bohr":   float(d_c),
            "N_fine":          N_f,
            "d_fine_bohr":     float(d_f),
        },
        "results": all_results,
    }

    json_path = out_dir / f"iter_test_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=float)
    print(f"  Data → {json_path}")
    return output
