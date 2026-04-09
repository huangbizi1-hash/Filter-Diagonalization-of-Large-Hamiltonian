"""
explosion_qd.py
多阶切比雪夫爆炸滤波器 —— 真实 QD（localPot.cube）。

算法
----
第 k 阶（k = 0, …, n_stages-1）：
  输入集合 = 初始随机正弦态（k=0）或上一阶归一化/Ritz输出（k>0）
  → T_m(H_scaled) 作用，H_scaled 将 [E_lower[k], E_upper] 映射到 [-1, 1]
  → E < E_lower[k] 的分量被放大，其余被抑制
每阶输出均并入最终子空间 → SVD + Rayleigh-Ritz 提取 Ritz 能量。

--ritz_filter 模式（可选）
--------------------------
每阶滤波后先做一次 SVD+Ritz：
  - 保留 E < E_lower[k+1] 的 Ritz 向量
  - 以这些 Ritz 向量（而非简单的滤波归一化态）作为下一阶的输入
  - 同时将 Ritz 向量也并入最终子空间

默认参数
--------
  两阶：E_lower = [-0.5, -0.4]，E_upper = 50.0，m = 40
  n_states = 64，k_max = 3.0，N = 64

用法
----
  python explosion_qd.py
  python explosion_qd.py --ritz_filter
  python explosion_qd.py --n_stages 3 --E_lower -0.5 -0.4 -0.3 --ritz_filter
  python explosion_qd.py --n_states 128 --m 60 --k_max 2.0
  python explosion_qd.py --E_upper 33.0 --svd_tol 5e-4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev as _Cheb
from scipy.linalg import eigh

from fft_code.grid         import build_k_diagonal, grid_to_vec, vec_to_grid
from fft_code.hamiltonian  import apply_H, apply_chebyshev_explosion
from fft_code.rayleigh_ritz import svd_rayleigh_ritz
from gaussian_potential_builder import GaussianPotentialBuilder

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Multi-stage Chebyshev explosion filter on QD from localPot.cube")
parser.add_argument("--cube_file",   type=str,   default="localPot.cube")
parser.add_argument("--params_file", type=str,   default="gaussian_fit_params.json")
parser.add_argument("--r_cut",       type=float, default=7.0,
                    help="Gaussian cutoff radius")
parser.add_argument("--N",           type=int,   default=64,
                    help="Grid points per axis")
parser.add_argument("--n_stages",    type=int,   default=2,
                    help="Number of explosion stages")
parser.add_argument("--E_lower",     type=float, nargs="+", default=[-0.5, -0.4],
                    help="E_lower for each stage (list, length = n_stages)")
parser.add_argument("--E_upper",     type=float, default=50.0,
                    help="Spectral upper bound for Chebyshev mapping")
parser.add_argument("--m",           type=int,   default=40,
                    help="Chebyshev polynomial order (same for all stages)")
parser.add_argument("--n_states",    type=int,   default=64,
                    help="Number of random initial states")
parser.add_argument("--k_max",       type=float, default=3.0,
                    help="Wavevector cutoff for random sine states (Bohr^-1)")
parser.add_argument("--svd_tol",     type=float, default=1e-3,
                    help="SVD truncation threshold")
parser.add_argument("--ritz_filter", action="store_true",
                    help="After each intermediate stage: SVD+Ritz, keep only Ritz vectors "
                         "with E < E_lower[next], use them as input to next stage")
parser.add_argument("--project",     action="store_true",
                    help="After each intermediate stage: SVD+Ritz on filtered states, "
                         "project subsequent filtered states onto the orthogonal complement "
                         "before the next SVD+Ritz. Keeps each stage's Ritz vectors "
                         "separately and suppresses duplicates.")
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--out_dir",     type=str,   default="figs_explosion_qd")
args = parser.parse_args()

# Validate E_lower length
if len(args.E_lower) != args.n_stages:
    parser.error(f"--E_lower must have exactly --n_stages={args.n_stages} values, "
                 f"got {len(args.E_lower)}")

out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)
rng = np.random.default_rng(args.seed)

# ──────────────────────────────────────────────────────────────────────────────
# 加载 QD 势能
# ──────────────────────────────────────────────────────────────────────────────
print(f"Loading QD potential from {args.cube_file} ...")
builder = GaussianPotentialBuilder(args.cube_file, args.params_file, args.r_cut)
x, y, z, V = builder.build_potential(args.N)
N = args.N
d = x[1] - x[0]   # grid spacing (Bohr)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

x_grid = (x, y, z)
T_k    = build_k_diagonal(x_grid)   # kinetic energy diagonal, capped at 30 Ha

print(f"  Grid: N={N}^3, d={d:.4f} Bohr, box [{x[0]:.2f}, {x[-1]:.2f}] Bohr")
print(f"  V range: [{V.min():.4f}, {V.max():.4f}]  (should cover E_lower values)")
print(f"  T_k max: {T_k.max():.4f} Ha")
print(f"  E_upper = {args.E_upper}  (should exceed T_k_max + max V)")

# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────
def ritz_energy(psi):
    """⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ on real-space grid."""
    H_psi  = apply_H(psi, V, T_k)
    norm2  = np.sum(psi**2)     * d**3
    expval = np.sum(psi * H_psi) * d**3
    return float(expval / norm2) if norm2 > 0 else float('nan')

def normalize(psi):
    nrm = np.sqrt(np.sum(psi**2) * d**3)
    return psi / nrm if nrm > 1e-30 else psi

def rand_sine(k_max):
    """Random sine wave with |k| components uniformly in [-k_max, k_max]."""
    kx, ky, kz = rng.uniform(-k_max, k_max, 3)
    b = rng.uniform(0, 2 * np.pi)
    psi = np.sin(kx * X + ky * Y + kz * Z + b)
    return normalize(psi)

def intermediate_svd_ritz(states, svd_tol, label=""):
    """
    SVD + Rayleigh-Ritz on a list of (N,N,N) states.
    Returns (energies, ritz_vecs) where ritz_vecs is a list of (N,N,N) real arrays,
    one per Ritz vector, sorted by energy.
    """
    mat  = np.stack(states, axis=0)              # (n, N, N, N)
    C_f  = grid_to_vec(mat)                      # (N^3, n)
    nrms = np.linalg.norm(C_f, axis=0)
    C_f  = C_f[:, nrms > 0] / nrms[nrms > 0]   # normalise, drop zeros
    C_f  = C_f[:, ~np.any(np.isinf(C_f), axis=0)]

    Q, R = np.linalg.qr(C_f, mode='reduced')
    U1, sigma, _ = np.linalg.svd(R, full_matrices=False)
    U  = Q @ U1
    r  = int(np.sum(sigma > svd_tol))
    if r == 0:
        print(f"  {label} WARNING: rank=0 after SVD, lowering tol to 1e-8")
        r = int(np.sum(sigma > 1e-8))
    Ur = U[:, :r]                                # (N^3, r)

    # Build projected Hamiltonian
    Ur_grid  = vec_to_grid(Ur, N, N, N)          # (r, N, N, N)
    HUr_grid = np.stack([apply_H(Ur_grid[i], V, T_k) for i in range(r)])
    H_tilde  = Ur.T.conj() @ grid_to_vec(HUr_grid)   # (r, r)

    energies, evecs = eigh(H_tilde)              # evecs: (r, r) columns
    energies = energies.real

    # Ritz vectors in original space: Ur @ evecs  → (N^3, r)
    ritz_flat = (Ur @ evecs).real
    ritz_grid = vec_to_grid(ritz_flat, N, N, N)  # (r, N, N, N)

    ritz_vecs = [normalize(ritz_grid[i].real) for i in range(r)]
    print(f"  {label}SVD rank={r}, "
          f"E range [{energies[0]:.4f}, {energies[-1]:.4f}]")
    return energies, ritz_vecs


def project_out(states, basis_vecs):
    """
    Remove the component of each state in `states` that lies in span(basis_vecs).

    Uses the formula:  v_perp = v - A (A^T A)^{-1} A^T v
    where A is the matrix whose columns are the flattened basis vectors.

    Parameters
    ----------
    states     : list of (N,N,N) arrays — vectors to be projected
    basis_vecs : list of (N,N,N) arrays — orthogonal basis (Ritz vectors)

    Returns
    -------
    list of (N,N,N) arrays — v_perp for each input state (normalised)
    """
    if not basis_vecs or not states:
        return states

    # Build A: (n_grid, n_basis)
    A = np.column_stack([v.ravel() for v in basis_vecs])  # (N^3, n_basis)

    # Precompute (A^T A)^{-1} A^T  — least-squares projector
    # For well-conditioned A, use lstsq; for orthonormal A, A^T A ≈ I.
    ATA = A.T @ A
    try:
        ATA_inv = np.linalg.inv(ATA)
    except np.linalg.LinAlgError:
        ATA_inv = np.linalg.pinv(ATA)
    proj_mat = A @ ATA_inv @ A.T   # (N^3, N^3) — too large to form explicitly

    # Instead compute projection vector-by-vector:
    #   v_par = A @ (ATA_inv @ (A^T @ v))
    ATA_inv_AT = ATA_inv @ A.T    # (n_basis, N^3)

    perp_states = []
    for psi in states:
        v      = psi.ravel().astype(float)
        coords = ATA_inv_AT @ v        # (n_basis,)  — coordinates in A
        v_par  = A @ coords            # (N^3,)
        v_perp = (v - v_par).reshape(psi.shape)
        perp_states.append(normalize(v_perp))
    return perp_states

# ──────────────────────────────────────────────────────────────────────────────
# 绘制各阶滤波窗形状
# ──────────────────────────────────────────────────────────────────────────────
E_arr = np.linspace(V.min() - 1.0, args.E_upper, 3000)
CLIP  = 100.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_stage = plt.cm.tab10(np.linspace(0, 0.9, args.n_stages))

for k in range(args.n_stages):
    E_lo = args.E_lower[k]
    a    = 2.0 / (args.E_upper - E_lo)
    b_   = -(args.E_upper + E_lo) / (args.E_upper - E_lo)
    coef = np.zeros(args.m + 1); coef[args.m] = 1.0
    Tm_vals = _Cheb(coef)(a * E_arr + b_)
    lbl = f"stage {k}  E_lower={E_lo}"
    axes[0].plot(E_arr, np.clip(np.abs(Tm_vals), 0, CLIP),
                 color=colors_stage[k], lw=1.4, label=lbl)
    axes[1].plot(E_arr, np.clip(np.abs(Tm_vals), 0, CLIP),
                 color=colors_stage[k], lw=1.4, label=lbl)
    axes[0].axvline(E_lo, color=colors_stage[k], ls=':', lw=0.8)
    axes[1].axvline(E_lo, color=colors_stage[k], ls=':', lw=0.8)

axes[0].axhline(1, color='gray', ls='--', lw=0.7)
axes[0].set_ylim(0, CLIP); axes[0].set_xlabel("Energy"); axes[0].set_ylabel("|T_m(E)|")
axes[0].set_title(f"Filter windows (m={args.m}, E_upper={args.E_upper})")
axes[0].legend(fontsize=8); axes[0].grid(True)

# zoom near E_lower region
zoom_lo = min(args.E_lower) - 1.0
zoom_hi = max(args.E_lower) + 1.0
mask = (E_arr >= zoom_lo) & (E_arr <= zoom_hi)
axes[1].set_xlim(zoom_lo, zoom_hi); axes[1].set_ylim(0, CLIP)
axes[1].axhline(1, color='gray', ls='--', lw=0.7)
axes[1].set_xlabel("Energy"); axes[1].set_title("Zoom near E_lower thresholds")
axes[1].legend(fontsize=8); axes[1].grid(True)

fig.tight_layout()
fig.savefig(out_dir / "filter_windows.png", dpi=150)
plt.close(fig)
print(f"Filter window plot → {out_dir}/filter_windows.png")

# ──────────────────────────────────────────────────────────────────────────────
# 生成初始随机正弦态
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nGenerating {args.n_states} random sine states (k_max={args.k_max})...")
initial_psi = [rand_sine(args.k_max) for _ in range(args.n_states)]
print(f"  Initial Ritz energy range: "
      f"[{min(ritz_energy(p) for p in initial_psi):.4f}, "
      f"{max(ritz_energy(p) for p in initial_psi):.4f}]")

# ──────────────────────────────────────────────────────────────────────────────
# 多阶爆炸滤波
# ──────────────────────────────────────────────────────────────────────────────
sets               = []          # sets[k] = list of filtered psi arrays for stage k
current_set        = initial_psi # input to first stage
t_total_filt       = 0.0
accumulated_basis  = []          # Ritz vecs accumulated across stages (--project)
stage_ritz_results = []          # per-stage Ritz info (--project)
t_ritz_proj        = 0.0         # cumulative intermediate SVD+Ritz time (--project)

for k in range(args.n_stages):
    E_lo = args.E_lower[k]
    print(f"\n── Stage {k}  T_{args.m}(H_s)  E_lower={E_lo}  E_upper={args.E_upper} ──")
    print(f"   Acting on {len(current_set)} states...")

    t0 = time.perf_counter()
    stage_out   = []
    ritz_before = []
    ritz_after  = []
    norms_after = []

    for i, psi_in in enumerate(current_set):
        ritz_before.append(ritz_energy(psi_in))
        psi_f = apply_chebyshev_explosion(psi_in, V, T_k, args.m, E_lo, args.E_upper)
        nrm   = np.sqrt(np.sum(psi_f**2) * d**3)
        norms_after.append(nrm)
        psi_f_norm = psi_f / nrm if nrm > 1e-30 else psi_f
        ritz_after.append(ritz_energy(psi_f_norm))
        stage_out.append(psi_f)      # keep unnormalized for SVD amplitude info
        if (i + 1) % 16 == 0:
            print(f"   {i+1}/{len(current_set)}  {time.perf_counter()-t0:.1f}s")

    dt = time.perf_counter() - t0
    t_total_filt += dt
    print(f"   Stage {k} done, {dt:.2f}s")

    # ── per-state table ──
    print(f"\n   {'#':>3}  {'E_before':>10}  {'E_after':>10}  {'norm_after':>12}")
    print(f"   {'─'*48}")
    for i in range(len(current_set)):
        marker = " ←" if ritz_after[i] < E_lo else ""
        print(f"   {i:>3}  {ritz_before[i]:>10.4f}  {ritz_after[i]:>10.4f}"
              f"  {norms_after[i]:>12.3e}{marker}")
    print(f"   {'─'*48}")
    n_hit = sum(1 for e in ritz_after if e < E_lo)
    print(f"   States with E_after < {E_lo}: {n_hit}/{len(current_set)}")

    sets.append(stage_out)

    # ── [--project] per-stage SVD+Ritz with orthogonal projection ──
    if args.project:
        t0p = time.perf_counter()
        if accumulated_basis:
            print(f"\n   [project] Projecting out {len(accumulated_basis)} basis "
                  f"vectors from stage {k} output...")
            states_for_ritz = project_out(stage_out, accumulated_basis)
        else:
            states_for_ritz = stage_out
        e_stage, rv_stage = intermediate_svd_ritz(
            states_for_ritz, args.svd_tol, label=f"stage {k} [proj]")
        t_ritz_proj += time.perf_counter() - t0p

        print(f"\n   [project] Stage {k} Ritz values ({len(e_stage)} total):")
        print(f"   {'#':>3}  {'E_ritz':>12}  kept?")
        print(f"   {'─'*38}")
        for i, E in enumerate(e_stage):
            marker = f"  ← E < {E_lo}" if E < E_lo else ""
            print(f"   {i:>3}  {E:>12.6f}{marker}")
        print(f"   {'─'*38}")

        new_basis = [rv_stage[i] for i in range(len(e_stage)) if e_stage[i] < E_lo]
        accumulated_basis.extend(new_basis)
        stage_ritz_results.append({
            "stage":    k,
            "E_lower":  E_lo,
            "energies": [float(e) for e in e_stage],
            "n_below":  len(new_basis),
        })
        print(f"   Added {len(new_basis)}/{len(e_stage)} Ritz vecs (E<{E_lo}) to "
              f"accumulated_basis → total = {len(accumulated_basis)}")

    # ── prepare input for next stage ──
    if k < args.n_stages - 1:
        threshold_next = args.E_lower[k + 1]

        if args.ritz_filter:
            # Intermediate SVD+Ritz: extract and filter Ritz vectors
            print(f"\n   [ritz_filter] Intermediate SVD+Ritz after stage {k}...")
            e_int, rv_int = intermediate_svd_ritz(
                stage_out, args.svd_tol, label=f"stage {k} → ")

            # Print all intermediate Ritz values with keep/discard label
            print(f"   {'#':>3}  {'E_ritz':>10}  decision")
            print(f"   {'─'*32}")
            for i, E in enumerate(e_int):
                decision = f"keep  (< {threshold_next})" if E < threshold_next else "discard"
                print(f"   {i:>3}  {E:>10.4f}  {decision}")
            print(f"   {'─'*32}")

            kept = [(e_int[i], rv_int[i]) for i in range(len(e_int))
                    if e_int[i] < threshold_next]
            if not kept:
                print(f"   WARNING: no Ritz vectors below {threshold_next}, "
                      f"keeping all {len(e_int)}")
                kept = list(zip(e_int, rv_int))
            print(f"   Kept {len(kept)}/{len(e_int)} Ritz vectors "
                  f"with E < {threshold_next}")

            # Also add the kept Ritz vectors to the final combined set
            sets.append([psi for _, psi in kept])

            current_set = [psi for _, psi in kept]

        else:
            # Original behaviour: normalize filtered states
            current_set = [psi_f / np.sqrt(np.sum(psi_f**2) * d**3)
                           if np.sqrt(np.sum(psi_f**2) * d**3) > 1e-30 else psi_f
                           for psi_f in stage_out]

# ──────────────────────────────────────────────────────────────────────────────
# 合并结果 → 输出 Ritz 能量
# ──────────────────────────────────────────────────────────────────────────────
if args.project:
    # ── Collect per-stage Ritz energies (each stage contributes E < E_lower[k]) ──
    per_stage_e = []
    for sr in stage_ritz_results:
        below = [e for e in sr["energies"] if e < sr["E_lower"]]
        per_stage_e.append(np.array(below))

    all_e    = np.concatenate(per_stage_e) if per_stage_e else np.array([])
    energies = np.sort(all_e)
    rank     = len(energies)
    t_ritz   = t_ritz_proj

    # ── Per-stage breakdown ──
    print(f"\n{'═'*60}")
    print(f"  [--project] Ritz energies ({rank} total from {args.n_stages} stages)")
    print(f"{'═'*60}")
    for sr in stage_ritz_results:
        below = [e for e in sr["energies"] if e < sr["E_lower"]]
        print(f"\n  Stage {sr['stage']}  E_lower={sr['E_lower']}  "
              f"({len(below)} energies below threshold):")
        for i, E in enumerate(sorted(below)):
            print(f"    {i:>3}  {E:>14.6f}")

    # ── Duplicate / near-duplicate check across stages ──
    if len(energies) > 1:
        diffs   = np.diff(energies)
        tol_dup = 1e-4
        dup_idx = np.where(diffs < tol_dup)[0]
        if len(dup_idx) > 0:
            print(f"\n  WARNING: {len(dup_idx)} near-duplicate pair(s) "
                  f"(|ΔE| < {tol_dup}):")
            for idx in dup_idx:
                print(f"    E[{idx}]={energies[idx]:.6f}  "
                      f"E[{idx+1}]={energies[idx+1]:.6f}  "
                      f"ΔE={diffs[idx]:.2e}")
        else:
            print(f"\n  No near-duplicates found (tol={tol_dup}).")

    print(f"\n{'═'*60}")
    print(f"  All collected energies ({len(energies)} total):")
    print(f"{'═'*60}")
    for i, E in enumerate(energies):
        print(f"  {i:>3}  {E:>14.6f}")
    print(f"{'═'*60}")

    # Per-threshold summary
    print()
    for k in range(args.n_stages):
        n = int(np.sum(energies < args.E_lower[k]))
        print(f"  E < {args.E_lower[k]}: {n} Ritz values")

else:
    # ── Standard: combine all stages → SVD + Rayleigh-Ritz ──
    all_states = [psi for stage in sets for psi in stage]
    sizes = [len(s) for s in sets]
    print(f"\nCombining {len(all_states)} vectors from {len(sets)} sets: {sizes}")
    print(f"SVD + Rayleigh-Ritz (svd_tol={args.svd_tol})...")

    filtered_matrix = np.stack(all_states, axis=0)   # (n_stages*n_states, N, N, N)
    t0 = time.perf_counter()
    energies, Ur, rank = svd_rayleigh_ritz(
        filtered_matrix, x_grid, V, N, N, N, T_k,
        svd_tol=args.svd_tol, max_energies=len(all_states),
    )
    t_ritz = time.perf_counter() - t0
    print(f"  Rank r={rank}, time {t_ritz:.2f}s")

    print(f"\n{'═'*60}")
    print(f"  All Ritz energies ({len(energies)} total, rank={rank})")
    print(f"{'═'*60}")
    for i, E in enumerate(energies):
        stage_label = ""
        for k in range(args.n_stages - 1, -1, -1):
            if E < args.E_lower[k]:
                stage_label = f"  ← E < {args.E_lower[k]}"
                break
        print(f"  {i:>3}  {E:>14.6f}{stage_label}")
    print(f"{'═'*60}")

    # Per-threshold summary
    print()
    for k in range(args.n_stages):
        n = int(np.sum(energies < args.E_lower[k]))
        print(f"  E < {args.E_lower[k]}: {n} Ritz values")

print(f"\nTotal filter time: {t_total_filt:.2f}s   Ritz time: {t_ritz:.2f}s")

# ──────────────────────────────────────────────────────────────────────────────
# 绘制 Ritz 能量散点图
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

# Color code by threshold
cmap_r = plt.cm.tab10(np.linspace(0, 0.9, args.n_stages + 1))
prev_lo = -np.inf
for k in range(args.n_stages):
    E_lo  = args.E_lower[k]
    mask  = (energies >= prev_lo) & (energies < E_lo)
    if mask.any():
        ax.scatter(energies[mask], np.zeros(mask.sum()),
                   marker='|', s=400, linewidths=2, color=cmap_r[k],
                   label=f"E ∈ [{prev_lo:.2g}, {E_lo}): {mask.sum()}")
    prev_lo = E_lo

# Remaining (above last threshold)
mask = energies >= args.E_lower[-1]
if mask.any():
    ax.scatter(energies[mask], np.zeros(mask.sum()),
               marker='|', s=400, linewidths=1.5, color=cmap_r[args.n_stages],
               alpha=0.5, label=f"E ≥ {args.E_lower[-1]}: {mask.sum()}")

for k in range(args.n_stages):
    ax.axvline(args.E_lower[k], color=colors_stage[k], ls='--', lw=1.2,
               label=f"E_lower[{k}]={args.E_lower[k]}")

ax.set_yticks([])
ax.set_xlabel("Energy")
ax.set_title(f"QD Ritz energies  (N={N}, m={args.m}, {args.n_stages} stages, "
             f"rank={rank})")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, axis='x')
fig.tight_layout()
fig.savefig(out_dir / "ritz_energies.png", dpi=150)
plt.close(fig)
print(f"Ritz energy plot → {out_dir}/ritz_energies.png")

# ──────────────────────────────────────────────────────────────────────────────
# 保存结果到 JSON
# ──────────────────────────────────────────────────────────────────────────────
counts = {str(args.E_lower[k]): int(np.sum(energies < args.E_lower[k]))
          for k in range(args.n_stages)}
results_dict = {
    "params": {
        "cube_file":   args.cube_file,
        "N":           N,
        "n_stages":    args.n_stages,
        "E_lower":     args.E_lower,
        "E_upper":     args.E_upper,
        "m":           args.m,
        "n_states":    args.n_states,
        "k_max":       args.k_max,
        "svd_tol":     args.svd_tol,
        "ritz_filter": args.ritz_filter,
        "project":     args.project,
        "seed":        args.seed,
    },
    "rank":              int(rank),
    "n_ritz":            len(energies),
    "ritz_energies":     [float(e) for e in energies],
    "counts_below":      counts,        # {"E_lower": n_ritz_below}
    "stage_ritz":        stage_ritz_results if args.project else None,
    "timing": {
        "filter_s":   round(t_total_filt, 3),
        "ritz_s":     round(t_ritz, 3),
        "total_s":    round(t_total_filt + t_ritz, 3),
    },
}
json_path = out_dir / "results.json"
with open(json_path, "w") as f:
    json.dump(results_dict, f, indent=2)
print(f"Results JSON  → {json_path}")
