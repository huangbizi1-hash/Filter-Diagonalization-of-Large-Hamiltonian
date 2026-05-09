#!/usr/bin/env python3
"""benchmark_qd_julia.py
Measure actual Julia evaluation times for QD filter expressions.

Task 1 – sample mode
    Pick --sample cubes (n_atoms>0), time the Julia eval on their interior
    grid points, and report n_atoms / N_op / n_pts / ms-per-wave.
    Scatter plots show what drives per-cube cost.

Task 2 – full mode
    Run ALL cubes with --n_waves waves, accumulate per-cube timings, and
    produce a "total QD time vs n_waves" prediction curve (from a single run:
    predicted_ms(N) = warmup_total + N × ms_per_wave_total).

Grid assignment (same logic as run_qd_r11.py Stage 5)
    Builds the full uniform grid (spacing d_grid) and assigns point ix to
    cube ci via:   cube_ax[ix] = min(ix * N_DIVISIONS // Ng, N_DIVISIONS-1)
    N_DIVISIONS is inferred from cube_size stored in the vexpr pkl files.

Usage
-----
    python benchmark_qd_julia.py \\
        --vexpr_dir QD_R17_Vexpr_partition_uniform \\
        --expr_dir  QD_R17_Julia_exp/partition_uniform_no_expansion \\
        --m 8 --n_waves 20 --sample 20 --L_s 22.0

    # just sample, skip full QD run:
    python benchmark_qd_julia.py ... --mode sample

    # full QD only, limit to first 30 cubes for a quick test:
    python benchmark_qd_julia.py ... --mode full --max_cubes 30
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from symbolic_code.expr_loader import CubicExpressionManager


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cube_dirname(cx, cy, cz):
    """Build the cube subdirectory name (must match h_powers.py convention)."""
    def fmt(v):
        return f"{v:.3f}".replace("-", "neg")
    return f"cube_{fmt(cx)}_{fmt(cy)}_{fmt(cz)}"


def find_jl_file(expr_dir, cube_dir_name, m):
    """Return the first eval_filter_m{m}_*.jl found in cube_dir, or None."""
    cube_path = Path(expr_dir) / cube_dir_name
    if not cube_path.exists():
        return None
    matches = sorted(cube_path.glob(f'eval_filter_m{m}_*.jl'))
    return matches[0] if matches else None


def count_julia_ops(jl_path):
    """Count arithmetic operators per grid point from a Julia batch script.

    Handles both code structures:
    - Horner  (build_julia_batch_script): poly in @inline _poly_cos/sin_i return lines
    - CSE     (selective_cse_build):      temps + accum lines inside @inbounds for loop

    Returns (n_ops, n_cos_groups, n_sin_groups).
    """
    text = Path(jl_path).read_text(encoding='utf-8')
    n_ops = 0

    # --- Horner-style: ops live in `return ...` lines of @inline poly functions ---
    poly_blocks = re.findall(
        r'@inline function _poly_(?:cos|sin)_\d+.*?\nend', text, re.DOTALL)
    for blk in poly_blocks:
        for ret in re.findall(r'^\s+return\s+(.+)$', blk, re.MULTILINE):
            n_ops += len(re.findall(r'[+\-*/^]', ret))

    # --- CSE-style: ops live inside the @inbounds for loop ---
    m_inbounds = re.search(
        r'@inbounds for i in 1:N\n(.*?)\n    end\nend', text, re.DOTALL)
    if m_inbounds:
        loop_body = m_inbounds.group(1)
        # CSE temp assignment lines: "        _s123 = ..."
        for line in re.findall(r'(?:_[sh]\d+|_[sh][a-z]\d*)\s*=\s*(.+)', loop_body):
            n_ops += len(re.findall(r'[+\-*/^]', line))
        # accumulation lines: "_sum_cos += ..."
        for line in re.findall(r'_sum_(?:cos|sin)\s*\+=\s*(.+)', loop_body):
            n_ops += len(re.findall(r'[+\-*/^]', line))

    # If neither found, fall back to counting ALL ops in the file
    if n_ops == 0:
        n_ops = len(re.findall(r'[+\-*/^]', text))

    n_cos = len(re.findall(r'@inline function _exp_cos_\d+', text))
    n_sin = len(re.findall(r'@inline function _exp_sin_\d+', text))
    return n_ops, n_cos, n_sin


def infer_grid_params(manager, L_s, d_grid):
    """Infer N_DIVISIONS and Ng from cube_info and grid spacing."""
    cube_size = next(iter(manager.cube_info.values()))['cube_size']
    N_DIVISIONS = max(1, round(2 * L_s / cube_size))
    Ng = int(np.ceil(2 * L_s / d_grid))
    return Ng, N_DIVISIONS, cube_size


def build_full_grid(L_s, d_grid):
    """Build the full uniform grid; return (x1, X, Y, Z)."""
    Ng = int(np.ceil(2 * L_s / d_grid))
    x1 = np.linspace(-L_s, L_s, Ng, endpoint=False)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
    return x1, X, Y, Z


def cube_interior_points(cube_idx, Ng, N_DIVISIONS, X, Y, Z):
    """Return (Xf, Yf, Zf, flat_idx) for the grid points owned by cube_idx.

    Uses the same assignment rule as run_filter_diag:
        cube_ax[ix] = min(ix * N_DIVISIONS // Ng, N_DIVISIONS - 1)
    """
    ci, cj, ck = cube_idx
    ix_all  = np.arange(Ng, dtype=np.int32)
    cube_ax = np.minimum((ix_all * N_DIVISIONS) // Ng, N_DIVISIONS - 1)

    ix_c = np.where(cube_ax == ci)[0]
    iy_c = np.where(cube_ax == cj)[0]
    iz_c = np.where(cube_ax == ck)[0]

    IXc, IYc, IZc = np.meshgrid(ix_c, iy_c, iz_c, indexing='ij')
    flat_idx = (IXc * Ng * Ng + IYc * Ng + IZc).ravel()

    Xf = X.ravel()[flat_idx]
    Yf = Y.ravel()[flat_idx]
    Zf = Z.ravel()[flat_idx]
    return Xf, Yf, Zf, flat_idx


def run_julia_on_cube(jl_path, Xf, Yf, Zf, k_vals, b_vals, work_dir,
                      julia_exe='julia'):
    """Run a pre-built Julia batch script on interior points of one cube.

    Returns timing dict: {warmup_s, eval_s, n_eval_waves, total_wall_s}.
    Raises RuntimeError on Julia failure.
    """
    n_waves = len(k_vals)
    N_pts   = len(Xf)

    grid_bin  = Path(work_dir) / '_grid.bin'
    kvals_bin = Path(work_dir) / '_kvals.bin'
    out_bin   = Path(work_dir) / '_out.bin'

    with open(grid_bin, 'wb') as f:
        Xf.astype('<f8').tofile(f)
        Yf.astype('<f8').tofile(f)
        Zf.astype('<f8').tofile(f)
    kb = np.column_stack([k_vals, b_vals]).astype('<f8')
    kb.ravel().tofile(str(kvals_bin))

    cmd = [julia_exe, str(jl_path),
           str(grid_bin), str(kvals_bin), str(out_bin),
           str(N_pts), str(n_waves)]

    t0   = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    for p in (grid_bin, kvals_bin, out_bin):
        p.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"exit {proc.returncode}\n"
            f"stderr: {proc.stderr[:600]}\nstdout: {proc.stdout[:200]}")

    timing = {'warmup_s': None, 'eval_s': None,
              'n_eval_waves': None, 'total_wall_s': wall}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                timing.update(json.loads(line))
            except json.JSONDecodeError:
                pass
            break
    return timing


# ---------------------------------------------------------------------------
# Collect valid cube records (shared by both modes)
# ---------------------------------------------------------------------------

def collect_cube_records(manager, expr_dir, m, Ng, N_DIVISIONS, X, Y, Z,
                         require_atoms=False):
    """Return list of dicts describing valid cubes (have .jl + interior pts)."""
    records = []
    for idx in sorted(manager.cube_info.keys()):
        info   = manager.cube_info[idx]
        cx, cy, cz = info['center']
        n_atoms    = info.get('n_atoms', 0)

        if require_atoms and n_atoms == 0:
            continue

        dirname  = _cube_dirname(cx, cy, cz)
        jl_path  = find_jl_file(expr_dir, dirname, m)
        if jl_path is None:
            continue

        Xf, Yf, Zf, flat_idx = cube_interior_points(idx, Ng, N_DIVISIONS, X, Y, Z)
        if len(Xf) == 0:
            continue

        records.append({
            'idx'     : idx,
            'cx'      : cx, 'cy': cy, 'cz': cz,
            'n_atoms' : n_atoms,
            'n_pts'   : len(Xf),
            'jl_path' : jl_path,
            'Xf'      : Xf, 'Yf': Yf, 'Zf': Zf,
        })
    return records


# ---------------------------------------------------------------------------
# Task 1 – sample mode
# ---------------------------------------------------------------------------

def run_sample_mode(manager, expr_dir, m, n_waves, k_max, L_s, d_grid,
                    sample, seed, julia_exe, outdir):
    rng = np.random.default_rng(seed)
    _, X, Y, Z = build_full_grid(L_s, d_grid)
    Ng, N_DIVISIONS, cube_size = infer_grid_params(manager, L_s, d_grid)
    print(f"  Grid: Ng={Ng}  N_DIVISIONS={N_DIVISIONS}  "
          f"cube_size={cube_size:.3f} Bohr  d_grid={d_grid}")

    all_records = collect_cube_records(
        manager, expr_dir, m, Ng, N_DIVISIONS, X, Y, Z, require_atoms=True)
    print(f"  Valid cubes (n_atoms>0, have jl): {len(all_records)}")

    chosen_idx = rng.choice(len(all_records),
                            size=min(sample, len(all_records)),
                            replace=False)
    chosen = [all_records[i] for i in sorted(chosen_idx)]
    print(f"  Sampled: {len(chosen)} cubes  (seed={seed})\n")

    k_vals = rng.uniform(-k_max, k_max, (n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, n_waves)

    rows = []
    with tempfile.TemporaryDirectory(prefix='qdjl_samp_') as work_dir:
        for i, rec in enumerate(chosen):
            idx     = rec['idx']
            n_atoms = rec['n_atoms']
            n_pts   = rec['n_pts']
            jl_path = rec['jl_path']

            N_op, n_cos, n_sin = count_julia_ops(jl_path)
            print(f"  [{i+1:3d}/{len(chosen)}] {idx}  "
                  f"center=({rec['cx']:.1f},{rec['cy']:.1f},{rec['cz']:.1f})  "
                  f"n_atoms={n_atoms:3d}  n_pts={n_pts:5d}  "
                  f"N_op={N_op:6d}  n_cos={n_cos}", end='  ', flush=True)

            try:
                timing = run_julia_on_cube(
                    jl_path, rec['Xf'], rec['Yf'], rec['Zf'],
                    k_vals, b_vals, work_dir, julia_exe)
            except RuntimeError as exc:
                print(f"FAILED: {str(exc)[:80]}")
                continue

            eval_s    = timing.get('eval_s') or 0.0
            n_eval    = timing.get('n_eval_waves') or max(n_waves - 1, 1)
            warmup_ms = (timing.get('warmup_s') or 0.0) * 1000
            ms_wave   = eval_s / n_eval * 1000
            ns_pt     = ms_wave / n_pts * 1e6 if n_pts else 0.0

            print(f"warmup={warmup_ms:.0f}ms  "
                  f"eval={ms_wave:.3f}ms/wave  "
                  f"{ns_pt:.2f}ns/pt/wave")

            rows.append({
                'cube_idx'     : str(idx),
                'cx'           : rec['cx'],
                'cy'           : rec['cy'],
                'cz'           : rec['cz'],
                'n_atoms'      : n_atoms,
                'n_pts'        : n_pts,
                'N_op'         : N_op,
                'n_cos_groups' : n_cos,
                'n_sin_groups' : n_sin,
                'warmup_ms'    : round(warmup_ms, 3),
                'ms_per_wave'  : round(ms_wave, 4),
                'ns_per_pt_per_wave': round(ns_pt, 4),
                'n_waves'      : n_waves,
            })

    if not rows:
        print("  No valid timing data collected.")
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f'sample_cubes_m{m}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  CSV  → {csv_path}")

    _plot_sample(rows, outdir, m)


def _plot_sample(rows, outdir, m):
    na  = np.array([r['n_atoms']            for r in rows])
    np_ = np.array([r['n_pts']              for r in rows])
    op  = np.array([r['N_op']               for r in rows])
    ms  = np.array([r['ms_per_wave']        for r in rows])
    ns  = np.array([r['ns_per_pt_per_wave'] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Sample cube timing  m={m}  ({len(rows)} cubes)', fontsize=12)

    # --- n_atoms vs ms/wave ---
    ax = axes[0]
    sc = ax.scatter(na, ms, c=np_, cmap='viridis', s=40, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='n_pts')
    ax.set_xlabel('n_atoms')
    ax.set_ylabel('ms / wave')
    ax.set_title('Eval time vs n_atoms')
    ax.grid(True, alpha=0.3)

    # --- N_op vs ms/wave ---
    ax = axes[1]
    sc = ax.scatter(op, ms, c=na, cmap='plasma', s=40, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='n_atoms')
    ax.set_xlabel('N_op (arithmetic ops in .jl)')
    ax.set_ylabel('ms / wave')
    ax.set_title('Eval time vs N_op')
    ax.grid(True, alpha=0.3)

    # --- n_pts vs ms/wave (should be linear) ---
    ax = axes[2]
    sc = ax.scatter(np_, ms, c=na, cmap='cool', s=40, alpha=0.8)
    if len(np_) >= 3:
        coef = np.polyfit(np_, ms, 1)
        xs = np.linspace(np_.min(), np_.max(), 200)
        ax.plot(xs, np.polyval(coef, xs), 'r--', alpha=0.7,
                label=f'fit: {coef[0]*1e6:.1f} ns/pt/wave')
        ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label='n_atoms')
    ax.set_xlabel('n_pts (interior grid points)')
    ax.set_ylabel('ms / wave')
    ax.set_title('Eval time vs n_pts  (linear expected)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = Path(outdir) / f'sample_cubes_m{m}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {fig_path}")


# ---------------------------------------------------------------------------
# Task 2 – full QD mode
# ---------------------------------------------------------------------------

def run_full_mode(manager, expr_dir, m, n_waves, k_max, L_s, d_grid,
                  julia_exe, outdir, max_cubes=None):
    rng = np.random.default_rng(42)
    _, X, Y, Z = build_full_grid(L_s, d_grid)
    Ng, N_DIVISIONS, cube_size = infer_grid_params(manager, L_s, d_grid)
    print(f"  Grid: Ng={Ng}^3={Ng**3:,} pts  "
          f"N_DIVISIONS={N_DIVISIONS}  cube_size={cube_size:.3f} Bohr")

    all_records = collect_cube_records(
        manager, expr_dir, m, Ng, N_DIVISIONS, X, Y, Z, require_atoms=False)
    if max_cubes:
        all_records = all_records[:max_cubes]
    print(f"  Cubes to run: {len(all_records)}")

    k_vals = rng.uniform(-k_max, k_max, (n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, n_waves)

    per_cube = []   # per-cube timing rows
    total_warmup_s = 0.0
    total_eval_s   = 0.0
    total_pts      = 0
    n_done = n_fail = 0

    t_wall_all = time.time()
    with tempfile.TemporaryDirectory(prefix='qdjl_full_') as work_dir:
        for i, rec in enumerate(all_records):
            try:
                timing = run_julia_on_cube(
                    rec['jl_path'],
                    rec['Xf'], rec['Yf'], rec['Zf'],
                    k_vals, b_vals, work_dir, julia_exe)
            except RuntimeError as exc:
                print(f"  [{i+1}] Cube {rec['idx']}: FAILED – {str(exc)[:60]}")
                n_fail += 1
                continue

            eval_s    = timing.get('eval_s') or 0.0
            warmup_s  = timing.get('warmup_s') or 0.0
            n_eval    = timing.get('n_eval_waves') or max(n_waves - 1, 1)
            ms_wave   = eval_s / n_eval * 1000

            total_warmup_s += warmup_s
            total_eval_s   += eval_s
            total_pts      += rec['n_pts']
            n_done         += 1

            per_cube.append({
                'cube_idx'     : str(rec['idx']),
                'n_atoms'      : rec['n_atoms'],
                'n_pts'        : rec['n_pts'],
                'warmup_ms'    : round(warmup_s * 1000, 2),
                'ms_per_wave'  : round(ms_wave, 4),
            })

            if (i + 1) % 20 == 0 or i == len(all_records) - 1:
                elapsed = time.time() - t_wall_all
                print(f"  [{i+1:4d}/{len(all_records)}]  "
                      f"done={n_done}  fail={n_fail}  "
                      f"total_eval_s={total_eval_s:.1f}  "
                      f"elapsed={elapsed:.0f}s")

    wall_total = time.time() - t_wall_all
    n_eval_waves = max(n_waves - 1, 1)
    total_ms_per_wave = total_eval_s / n_eval_waves * 1000
    total_warmup_ms   = total_warmup_s * 1000
    ns_per_pt = total_ms_per_wave / total_pts * 1e6 if total_pts else 0.0

    print(f"\n  === Full QD summary (m={m}, n_waves={n_waves}) ===")
    print(f"  Cubes: done={n_done}  fail={n_fail}")
    print(f"  Total grid points covered: {total_pts:,}  / {Ng**3:,}")
    print(f"  Total warmup (JIT):   {total_warmup_ms:.0f} ms")
    print(f"  Total eval/wave:      {total_ms_per_wave:.1f} ms/wave")
    print(f"  Per-point:            {ns_per_pt:.2f} ns/pt/wave")
    print(f"  Wall time (all cubes): {wall_total:.1f} s")
    print(f"  Predicted total for N waves: "
          f"warmup + N×{total_ms_per_wave:.1f} ms")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Per-cube CSV
    if per_cube:
        csv_path = outdir / f'full_qd_per_cube_m{m}.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(per_cube[0].keys()))
            w.writeheader(); w.writerows(per_cube)
        print(f"\n  Per-cube CSV → {csv_path}")

    # Summary CSV (one row per run)
    summary = {
        'm'                   : m,
        'n_waves'             : n_waves,
        'n_cubes_done'        : n_done,
        'total_pts'           : total_pts,
        'total_warmup_ms'     : round(total_warmup_ms, 1),
        'total_ms_per_wave'   : round(total_ms_per_wave, 3),
        'ns_per_pt_per_wave'  : round(ns_per_pt, 3),
        'wall_s'              : round(wall_total, 1),
    }
    sum_path = outdir / f'full_qd_summary_m{m}.csv'
    with open(sum_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader(); w.writerow(summary)
    print(f"  Summary CSV → {sum_path}")

    _plot_full(per_cube, total_warmup_ms, total_ms_per_wave, outdir, m, n_waves)


def _plot_full(per_cube, total_warmup_ms, total_ms_per_wave, outdir, m, n_waves):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Full QD evaluation  m={m}  n_waves={n_waves}', fontsize=12)

    # --- Predicted total time vs n_waves ---
    ax = axes[0]
    n_arr = np.arange(1, max(500, n_waves * 5) + 1)
    t_total = total_warmup_ms + n_arr * total_ms_per_wave
    t_eval  = n_arr * total_ms_per_wave
    ax.plot(n_arr, t_total / 1000, 'b-',  label='warmup + eval (s)')
    ax.plot(n_arr, t_eval  / 1000, 'r--', label='eval only (s)')
    ax.axvline(n_waves, color='gray', linestyle=':', alpha=0.7,
               label=f'measured n_waves={n_waves}')
    ax.set_xlabel('n_waves (number of filter vectors)')
    ax.set_ylabel('total time (s)')
    ax.set_title('Predicted QD time vs n_waves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Per-cube ms/wave distribution ---
    ax = axes[1]
    if per_cube:
        ms_arr = np.array([r['ms_per_wave'] for r in per_cube])
        ax.hist(ms_arr, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(ms_arr.mean(), color='r', linestyle='--',
                   label=f'mean={ms_arr.mean():.3f} ms')
        ax.set_xlabel('ms / wave (per cube)')
        ax.set_ylabel('count')
        ax.set_title('Per-cube eval time distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- ms/wave vs n_pts and n_atoms ---
    ax = axes[2]
    if per_cube:
        na_arr  = np.array([r['n_atoms']    for r in per_cube])
        npt_arr = np.array([r['n_pts']      for r in per_cube])
        ms_arr  = np.array([r['ms_per_wave'] for r in per_cube])
        sc = ax.scatter(npt_arr, ms_arr, c=na_arr, cmap='plasma',
                        s=20, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='n_atoms')
        if len(npt_arr) >= 3:
            coef = np.polyfit(npt_arr, ms_arr, 1)
            xs = np.linspace(npt_arr.min(), npt_arr.max(), 200)
            ax.plot(xs, np.polyval(coef, xs), 'r--', alpha=0.7,
                    label=f'{coef[0]*1e6:.1f} ns/pt/wave')
            ax.legend(fontsize=8)
        ax.set_xlabel('n_pts')
        ax.set_ylabel('ms / wave')
        ax.set_title('Per-cube: time vs n_pts (all cubes)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = Path(outdir) / f'full_qd_m{m}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Julia eval time for QD filter expressions.'
    )
    parser.add_argument('--vexpr_dir', required=True,
                        help='V_expr pkl directory (from space partition)')
    parser.add_argument('--expr_dir', required=True,
                        help='Julia script directory (from Stage 4)')
    parser.add_argument('--m', type=int, default=8,
                        help='Chebyshev order to look for (default 8)')
    parser.add_argument('--L_s', type=float, default=22.0,
                        help='Half-box size in Bohr (default 22.0 for QD R=17)')
    parser.add_argument('--d_grid', type=float, default=0.625,
                        help='Grid spacing in Bohr (default 0.625)')
    parser.add_argument('--n_waves', type=int, default=20,
                        help='Plane waves for timing (default 20; use >=10)')
    parser.add_argument('--k_max', type=float, default=1.0,
                        help='Max k-vector magnitude (default 1.0)')
    parser.add_argument('--sample', type=int, default=40,
                        help='Cubes to sample in sample mode (default 40)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default 0)')
    parser.add_argument('--mode', choices=['sample', 'full', 'both'],
                        default='both',
                        help='Which analysis to run (default both)')
    parser.add_argument('--max_cubes', type=int, default=None,
                        help='Limit cubes in full mode (for testing)')
    parser.add_argument('--julia_exe', default='julia',
                        help='Julia executable (default julia)')
    parser.add_argument('--outdir', default='figs_qd_timing',
                        help='Output directory (default figs_qd_timing/)')
    args = parser.parse_args()

    if args.n_waves < 2:
        print('WARNING: n_waves < 2 → Julia timed section runs 0 iterations '
              '(eval_s=0). Use --n_waves 20 or more.\n')

    print(f"Loading cube manager from {args.vexpr_dir} ...")
    manager = CubicExpressionManager(args.vexpr_dir)
    if not manager.cube_info:
        print("ERROR: No cubes found in vexpr_dir. Check the path.")
        sys.exit(1)
    print(f"  {len(manager.cube_info)} cubes loaded")

    # Quick sanity-check: infer grid params and show them
    try:
        Ng, N_DIV, csz = infer_grid_params(manager, args.L_s, args.d_grid)
        print(f"  Inferred: Ng={Ng}  N_DIVISIONS={N_DIV}  cube_size={csz:.3f} Bohr\n")
    except Exception as exc:
        print(f"  Warning: could not infer grid params: {exc}")

    if args.mode in ('sample', 'both'):
        print(f"{'='*60}")
        print(f"Task 1 – sample  "
              f"(m={args.m}, sample={args.sample}, n_waves={args.n_waves})")
        print(f"{'='*60}")
        run_sample_mode(
            manager, args.expr_dir, args.m,
            args.n_waves, args.k_max,
            args.L_s, args.d_grid,
            args.sample, args.seed,
            args.julia_exe, args.outdir)

    if args.mode in ('full', 'both'):
        print(f"\n{'='*60}")
        print(f"Task 2 – full QD  "
              f"(m={args.m}, n_waves={args.n_waves})")
        print(f"{'='*60}")
        run_full_mode(
            manager, args.expr_dir, args.m,
            args.n_waves, args.k_max,
            args.L_s, args.d_grid,
            args.julia_exe, args.outdir,
            max_cubes=args.max_cubes)

    print('\nDone.')


if __name__ == '__main__':
    main()
