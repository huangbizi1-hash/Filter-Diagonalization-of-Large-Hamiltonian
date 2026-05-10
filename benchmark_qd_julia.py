#!/usr/bin/env python3
"""benchmark_qd_julia.py
Sweep Chebyshev order n=1..m_max for sampled QD cubes and measure
N_op and Julia eval time for each (cube, n) pair.

Output CSV columns (compatible with the old cube_complexity.csv plotting script):
    cube, atoms, n, part, plus, mul, total,
    n_pts, ms_per_wave, ns_per_pt_per_wave

  part : "Pc" = cos-envelope polynomial (_poly_cos_* in the .jl)
         "Ps" = sin-envelope polynomial (_poly_sin_* in the .jl)
  plus : +/- operator count in that part's return expressions
  mul  : * / ^ operator count in that part's return expressions
  total: plus + mul

Modes
-----
  sweep (default)
      For --sample cubes (n_atoms>0), generate eval_filter_m{n}_*.jl for
      n=1..m_max, count ops, and time Julia eval on interior grid points.

  full
      Run ALL cubes at fixed n=m, accumulate timings, output predicted
      total QD time vs n_waves curve.

  both
      sweep + full.

Grid assignment (identical to run_qd_r11.py Stage 5)
    cube_ax[ix] = min(ix * N_DIVISIONS // Ng, N_DIVISIONS - 1)
    N_DIVISIONS is inferred from cube_size stored in the vexpr pkl files.

Usage
-----
    python benchmark_qd_julia.py \\
        --vexpr_dir QD_R17_Vexpr_partition_uniform \\
        --expr_dir  QD_R17_Julia_exp/partition_uniform_no_expansion \\
        --m 8 --n_waves 20 --sample 20 --L_s 22.0

    # N_op only, no Julia timing:
    python benchmark_qd_julia.py ... --no_timing

    # full QD only, limit to 30 cubes for a quick test:
    python benchmark_qd_julia.py ... --mode full --max_cubes 30
"""

import argparse
import csv
import json
import pickle
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from symbolic_code.expr_loader import CubicExpressionManager
from symbolic_code.chebyshev_filter import (
    apply_f_of_H_from_raw_powers,
    extract_cos_sin_coeffs,
    group_by_exp_combined,
    apply_horner,
)
from symbolic_code.julia_codegen import build_julia_batch_script


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _cube_dirname(cx, cy, cz):
    def fmt(v):
        return f"{v:.3f}".replace("-", "neg")
    return f"cube_{fmt(cx)}_{fmt(cy)}_{fmt(cz)}"


def _jl_name_for(n, a, b):
    a_key = f"{a:.8g}".replace('.', 'p').replace('-', 'm')
    b_key = f"{b:.8g}".replace('.', 'p').replace('-', 'm')
    return f"eval_filter_m{n}_{a_key}_{b_key}.jl"


# ---------------------------------------------------------------------------
# Filter parameter inference
# ---------------------------------------------------------------------------

def _parse_ab_from_jl_name(jl_path):
    """Decode (a, b) from eval_filter_m{n}_{a_key}_{b_key}.jl filename."""
    parts = Path(jl_path).stem.split('_')
    m_idx = next(i for i, p in enumerate(parts) if re.match(r'^m\d+$', p))
    a_key = parts[m_idx + 1]
    b_key = parts[m_idx + 2]
    a = float(a_key.replace('p', '.').replace('m', '-'))
    b = float(b_key.replace('p', '.').replace('m', '-'))
    return a, b


def _find_any_jl_in_dir(expr_dir, m):
    """Return the first eval_filter_m{m}_*.jl found anywhere in expr_dir."""
    for cube_path in sorted(Path(expr_dir).iterdir()):
        if cube_path.is_dir():
            matches = sorted(cube_path.glob(f'eval_filter_m{m}_*.jl'))
            if matches:
                return matches[0]
    return None


def _estimate_E_hi(L_s, Ng, pref=0.5, V_peak=12.0):
    """Rough E_hi from Nyquist kinetic energy + potential peak."""
    dx = 2.0 * L_s / Ng
    k_max = np.pi / dx
    return pref * 3.0 * k_max**2 + V_peak


# ---------------------------------------------------------------------------
# .jl file lookup and on-demand generation
# ---------------------------------------------------------------------------

def find_jl_file(expr_dir, cube_dir_name, n):
    """Return eval_filter_m{n}_*.jl in cube_dir, or None."""
    cube_path = Path(expr_dir) / cube_dir_name
    if not cube_path.exists():
        return None
    matches = sorted(cube_path.glob(f'eval_filter_m{n}_*.jl'))
    return matches[0] if matches else None


def generate_hn_jl_for_cube(cube_dir, n, regen=False):
    """Generate eval_Hn_m{n}.jl from H_power_{n}.pkl (single power, not filter).

    Unlike generate_jl_for_cube, this does NOT call apply_f_of_H_from_raw_powers
    and does NOT combine multiple powers.  It loads H^n·ψ directly from pkl,
    groups by Gaussian factor, applies Horner, and writes Julia code.
    Fast and cannot OOM.  Returns jl_path, or None if pkl missing.
    """
    cube_dir = Path(cube_dir)
    pkl_path = cube_dir / f'H_power_{n}.pkl'
    if not pkl_path.exists():
        return None

    jl_path = cube_dir / f'eval_Hn_m{n}.jl'
    if jl_path.exists() and not regen:
        return jl_path

    try:
        with open(pkl_path, 'rb') as fh:
            data = pickle.load(fh)
        expr_cos = sp.expand(data.get('Pc', sp.Integer(0)))
        expr_sin = sp.expand(data.get('Ps', sp.Integer(0)))
        terms_cos = apply_horner(group_by_exp_combined(expr_cos))
        terms_sin = apply_horner(group_by_exp_combined(expr_sin))
        jl_src = build_julia_batch_script(terms_cos, terms_sin)
        jl_path.write_text(jl_src, encoding='utf-8')
        return jl_path
    except Exception as exc:
        print(f"\n    [Hn codegen error] {cube_dir.name} n={n}: {exc}")
        return None



    """Generate eval_filter_m{n}_*.jl from H^n pkl files.

    Uses the same codegen as run_qd_r11.py Stage 4.
    Returns jl_path on success, None if H_power_{n}.pkl is missing.
    When regen=True, overwrites any existing .jl file.
    """
    cube_dir = Path(cube_dir)
    if not (cube_dir / f'H_power_{n}.pkl').exists():
        return None

    jl_path = cube_dir / _jl_name_for(n, a, b)
    if jl_path.exists() and not regen:
        return jl_path

    try:
        if not expand:
            expr_cos_raw, expr_sin_raw = apply_f_of_H_from_raw_powers(
                str(cube_dir), n, a=a, b=b,
                file_type='pkl', return_envelopes=True)
            expr_cos = sp.expand(expr_cos_raw)
            expr_sin = sp.expand(expr_sin_raw)
        else:
            psi_fH = apply_f_of_H_from_raw_powers(
                str(cube_dir), n, a=a, b=b, file_type='pkl')
            expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)

        terms_cos = apply_horner(group_by_exp_combined(expr_cos))
        terms_sin = apply_horner(group_by_exp_combined(expr_sin))
        jl_src = build_julia_batch_script(terms_cos, terms_sin)
        jl_path.write_text(jl_src, encoding='utf-8')
        return jl_path
    except Exception as exc:
        print(f"\n    [codegen error] {cube_dir.name} n={n}: {exc}")
        return None


def find_or_generate_jl(expr_dir, cube_dir_name, n, a, b, expand=False, regen=False):
    if not regen:
        jl_path = find_jl_file(expr_dir, cube_dir_name, n)
        if jl_path is not None:
            return jl_path
    cube_dir = Path(expr_dir) / cube_dir_name
    if not cube_dir.exists():
        return None
    return generate_jl_for_cube(cube_dir, n, a, b, expand=expand, regen=regen)


# ---------------------------------------------------------------------------
# Operator counting (per part: Pc = cos, Ps = sin)
# ---------------------------------------------------------------------------

def count_ops_per_part(jl_path):
    """Count +/- and */^ operators in cos and sin polynomial blocks.

    Handles Horner-style .jl (return lines of @inline _poly_{cos|sin}_N).

    Returns dict:
        plus_cos, mul_cos, total_cos,
        plus_sin, mul_sin, total_sin,
        n_cos_groups, n_sin_groups
    """
    text = Path(jl_path).read_text(encoding='utf-8')

    counts = {'cos': [0, 0, 0], 'sin': [0, 0, 0]}  # [plus, mul, groups]

    for match in re.finditer(
        r'@inline function _poly_(cos|sin)_\d+.*?\nend', text, re.DOTALL
    ):
        part = match.group(1)
        blk  = match.group(0)
        for ret in re.findall(r'^\s+return\s+(.+)$', blk, re.MULTILINE):
            counts[part][0] += len(re.findall(r'[+\-]', ret))
            counts[part][1] += len(re.findall(r'[*/^]', ret))
            counts[part][2] += 1

    pc, ps = counts['cos'], counts['sin']
    return {
        'plus_cos'    : pc[0], 'mul_cos'  : pc[1],
        'total_cos'   : pc[0] + pc[1], 'n_cos_groups': pc[2],
        'plus_sin'    : ps[0], 'mul_sin'  : ps[1],
        'total_sin'   : ps[0] + ps[1], 'n_sin_groups': ps[2],
    }


def count_pkl_ops(cube_dir, n):
    """Count SymPy ops in H^n directly from H_power_{n}.pkl (Ps and Pc parts).

    Loads the pkl without combining with other powers, so this is fast and
    cannot OOM.  Uses count_ops(visual=True) to extract ADD and MUL counts,
    matching the approach in analyze_cube_expr_complexity.py.

    Returns the same dict shape as count_ops_per_part, or None if pkl missing.
    """
    pkl_path = Path(cube_dir) / f'H_power_{n}.pkl'
    if not pkl_path.exists():
        return None
    with open(pkl_path, 'rb') as fh:
        data = pickle.load(fh)
    out = {}
    for key, part in (('Ps', 'sin'), ('Pc', 'cos')):
        expr = data.get(key, sp.Integer(0))
        visual = sp.count_ops(expr, visual=True)
        plus = int(visual.coeff(sp.Symbol('ADD')))
        mul  = int(visual.coeff(sp.Symbol('MUL')))
        out[f'plus_{part}']  = plus
        out[f'mul_{part}']   = mul
        out[f'total_{part}'] = plus + mul
    out['n_cos_groups'] = 0
    out['n_sin_groups'] = 0
    return out


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def infer_grid_params(manager, L_s, d_grid):
    cube_size = next(iter(manager.cube_info.values()))['cube_size']
    N_DIVISIONS = max(1, round(2 * L_s / cube_size))
    Ng = int(np.ceil(2 * L_s / d_grid))
    return Ng, N_DIVISIONS, cube_size


def build_full_grid(L_s, d_grid):
    Ng = int(np.ceil(2 * L_s / d_grid))
    x1 = np.linspace(-L_s, L_s, Ng, endpoint=False)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
    return x1, X, Y, Z


def cube_interior_points(cube_idx, Ng, N_DIVISIONS, X, Y, Z):
    """Return (Xf, Yf, Zf, flat_idx) owned by cube_idx.

    Assignment: cube_ax[ix] = min(ix * N_DIVISIONS // Ng, N_DIVISIONS - 1)
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


# ---------------------------------------------------------------------------
# Julia runner
# ---------------------------------------------------------------------------

def run_julia_on_cube(jl_path, Xf, Yf, Zf, k_vals, b_vals, work_dir,
                      julia_exe='julia'):
    """Run a Julia batch script on interior points; return timing dict."""
    n_waves = len(k_vals)
    N_pts   = len(Xf)

    grid_bin  = Path(work_dir) / '_grid.bin'
    kvals_bin = Path(work_dir) / '_kvals.bin'
    out_bin   = Path(work_dir) / '_out.bin'

    with open(grid_bin, 'wb') as f:
        Xf.astype('<f8').tofile(f)
        Yf.astype('<f8').tofile(f)
        Zf.astype('<f8').tofile(f)
    np.column_stack([k_vals, b_vals]).astype('<f8').ravel().tofile(str(kvals_bin))

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


def run_sympy_on_cube(cube_dir, n, Xf, Yf, Zf, k_vals, b_vals, a, b, expand=False):
    """Evaluate SymPy expression numerically on interior points; return timing dict."""
    if not expand:
        expr_cos_raw, expr_sin_raw = apply_f_of_H_from_raw_powers(
            str(cube_dir), n, a=a, b=b, file_type='pkl', return_envelopes=True)
        expr_cos = sp.expand(expr_cos_raw)
        expr_sin = sp.expand(expr_sin_raw)
    else:
        psi_fH = apply_f_of_H_from_raw_powers(str(cube_dir), n, a=a, b=b, file_type='pkl')
        expr_cos, expr_sin = extract_cos_sin_coeffs(psi_fH)

    x, y, z, kx, ky, kz = sp.symbols('x y z kx ky kz')
    f_cos = sp.lambdify((x, y, z, kx, ky, kz), expr_cos, 'numpy')
    f_sin = sp.lambdify((x, y, z, kx, ky, kz), expr_sin, 'numpy')

    _ = f_cos(Xf, Yf, Zf, *k_vals[0])
    _ = f_sin(Xf, Yf, Zf, *k_vals[0])
    t0 = time.time()
    for iw in range(1, len(k_vals)):
        phase = k_vals[iw, 0] * Xf + k_vals[iw, 1] * Yf + k_vals[iw, 2] * Zf + b_vals[iw]
        out = f_cos(Xf, Yf, Zf, *k_vals[iw]) * np.cos(phase) + \
              f_sin(Xf, Yf, Zf, *k_vals[iw]) * np.sin(phase)
        _ = out
    eval_s = time.time() - t0
    return {'eval_s': eval_s, 'n_eval_waves': max(len(k_vals) - 1, 1)}


# ---------------------------------------------------------------------------
# Mode 1 – sweep:  n = 1..m_max  for sampled cubes
# ---------------------------------------------------------------------------


def _pick_cubes_by_atom_counts(candidates, atom_counts, rng):
    """Pick one cube for each requested atom count.

    Returns (chosen, missing_counts).
    """
    chosen = []
    missing = []
    for count in atom_counts:
        matched = [c for c in candidates if c['n_atoms'] == count]
        if not matched:
            missing.append(count)
            continue
        pick = matched[int(rng.integers(0, len(matched)))]
        chosen.append(pick)
    return chosen, missing

def run_sweep_mode(manager, expr_dir, m_max, n_waves, k_max, L_s, d_grid,
                   sample, seed, julia_exe, outdir, a, b,
                   select_atoms=None, expand=False, do_timing=True,
                   do_sympy_timing=False, nop_json_path=None, regen=False,
                   count_pkl=False, hn_julia=False):
    """For --sample cubes sweep n=1..m_max; count ops and optionally time."""
    rng = np.random.default_rng(seed)
    _, X, Y, Z = build_full_grid(L_s, d_grid)
    Ng, N_DIVISIONS, cube_size = infer_grid_params(manager, L_s, d_grid)
    print(f"  Grid: Ng={Ng}  N_DIVISIONS={N_DIVISIONS}  "
          f"cube_size={cube_size:.3f} Bohr  d_grid={d_grid}")

    # Collect cubes that have n_atoms > 0 and at least H_power_1.pkl
    candidates = []
    for idx in sorted(manager.cube_info.keys()):
        info = manager.cube_info[idx]
        if info.get('n_atoms', 0) == 0:
            continue
        cx, cy, cz = info['center']
        cube_dir = Path(expr_dir) / _cube_dirname(cx, cy, cz)
        if not (cube_dir / 'H_power_1.pkl').exists() and \
           find_jl_file(expr_dir, _cube_dirname(cx, cy, cz), 1) is None:
            continue
        Xf, Yf, Zf, _ = cube_interior_points(idx, Ng, N_DIVISIONS, X, Y, Z)
        if len(Xf) == 0:
            continue
        candidates.append({
            'idx': idx, 'cx': cx, 'cy': cy, 'cz': cz,
            'n_atoms': info['n_atoms'], 'n_pts': len(Xf),
            'Xf': Xf, 'Yf': Yf, 'Zf': Zf,
            'dirname': _cube_dirname(cx, cy, cz),
        })

    print(f"  Candidate cubes (n_atoms>0, have pkl/jl): {len(candidates)}")
    if select_atoms:
        chosen, missing = _pick_cubes_by_atom_counts(candidates, select_atoms, rng)
        if missing:
            print(f"  WARNING: no candidate cubes found for atom counts: {missing}")
        n_pick = len(chosen)
        print(f"  Selected by atoms: requested={select_atoms}  picked={n_pick} cubes  (seed={seed})")
    else:
        n_pick = min(sample, len(candidates))
        chosen_idx = rng.choice(len(candidates), size=n_pick, replace=False)
        chosen = [candidates[i] for i in sorted(chosen_idx)]
        print(f"  Sampled: {n_pick} cubes  (seed={seed})")
    print(f"  Sweep n = 1 .. {m_max}  "
          f"{'with timing (' + str(n_waves) + ' waves)' if do_timing else 'N_op only'}"
          f"{' + SymPy timing' if do_sympy_timing else ''}\n")

    k_vals = rng.uniform(-k_max, k_max, (n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, n_waves)

    rows = []   # one row per (cube, n, part)

    with tempfile.TemporaryDirectory(prefix='qdjl_sweep_') as work_dir:
        for ci, rec in enumerate(chosen):
            print(f"  [{ci+1}/{n_pick}] Cube {rec['idx']}  "
                  f"center=({rec['cx']:.1f},{rec['cy']:.1f},{rec['cz']:.1f})  "
                  f"n_atoms={rec['n_atoms']}  n_pts={rec['n_pts']}")

            for n in range(1, m_max + 1):
                if count_pkl:
                    # --- count ops directly from H^n pkl (fast, no OOM risk) ---
                    cube_dir_path = Path(expr_dir) / rec['dirname']
                    try:
                        ops = count_pkl_ops(cube_dir_path, n)
                    except Exception as exc:
                        print(f"    n={n}: pkl op-count error: {exc}")
                        continue
                    if ops is None:
                        print(f"    n={n}: H_power_{n}.pkl missing – skipped")
                        continue
                    jl_path = None
                elif hn_julia:
                    # --- generate eval_Hn_m{n}.jl from H^n pkl directly ---
                    cube_dir_path = Path(expr_dir) / rec['dirname']
                    jl_path = generate_hn_jl_for_cube(cube_dir_path, n, regen=regen)
                    if jl_path is None:
                        print(f"    n={n}: H_power_{n}.pkl missing – skipped")
                        continue
                    try:
                        ops = count_ops_per_part(jl_path)
                    except Exception as exc:
                        print(f"    n={n}: op-count error: {exc}")
                        continue
                else:
                    # --- find or generate eval_filter .jl for order n ---
                    jl_path = find_or_generate_jl(
                        expr_dir, rec['dirname'], n, a, b, expand=expand, regen=regen)
                    if jl_path is None:
                        print(f"    n={n}: no pkl/jl – skipped")
                        continue

                    # --- count ops from .jl ---
                    try:
                        ops = count_ops_per_part(jl_path)
                    except Exception as exc:
                        print(f"    n={n}: op-count error: {exc}")
                        continue

                # --- optional timing (only when .jl available) ---
                ms_wave = ns_pt = 0.0
                sympy_ms_wave = sympy_ns_pt = 0.0
                if do_timing and jl_path is not None:
                    try:
                        timing = run_julia_on_cube(
                            jl_path, rec['Xf'], rec['Yf'], rec['Zf'],
                            k_vals, b_vals, work_dir, julia_exe)
                        eval_s  = timing.get('eval_s') or 0.0
                        n_eval  = timing.get('n_eval_waves') or max(n_waves - 1, 1)
                        ms_wave = eval_s / n_eval * 1000
                        ns_pt   = ms_wave / rec['n_pts'] * 1e6 if rec['n_pts'] else 0.0
                    except RuntimeError as exc:
                        print(f"    n={n}: Julia error: {str(exc)[:60]}")
                if do_sympy_timing:
                    try:
                        sympy_timing = run_sympy_on_cube(
                            Path(expr_dir) / rec['dirname'], n,
                            rec['Xf'], rec['Yf'], rec['Zf'],
                            k_vals, b_vals, a, b, expand=expand)
                        sympy_eval_s = sympy_timing.get('eval_s') or 0.0
                        sympy_n_eval = sympy_timing.get('n_eval_waves') or max(n_waves - 1, 1)
                        sympy_ms_wave = sympy_eval_s / sympy_n_eval * 1000
                        sympy_ns_pt = sympy_ms_wave / rec['n_pts'] * 1e6 if rec['n_pts'] else 0.0
                    except Exception as exc:
                        print(f"    n={n}: SymPy error: {str(exc)[:60]}")

                print(f"    n={n}  "
                      f"Nop_cos={ops['total_cos']:5d}  "
                      f"Nop_sin={ops['total_sin']:5d}  "
                      f"Nop_tot={ops['total_cos']+ops['total_sin']:5d}"
                      + (f"  julia={ms_wave:.3f}ms/wave" if do_timing else "")
                      + (f"  sympy={sympy_ms_wave:.3f}ms/wave" if do_sympy_timing else ""))

                cube_id = rec['dirname']
                base = {
                    'cube'  : cube_id,
                    'atoms' : rec['n_atoms'],
                    'n'     : n,
                    'n_pts' : rec['n_pts'],
                    'ms_per_wave'         : float(ms_wave),
                    'ns_per_pt_per_wave'  : float(ns_pt),
                    'sympy_ms_per_wave'   : float(sympy_ms_wave),
                    'sympy_ns_per_pt_per_wave': float(sympy_ns_pt),
                }
                rows.append({**base, 'part': 'Pc',
                             'plus': ops['plus_cos'], 'mul': ops['mul_cos'],
                             'total': ops['total_cos']})
                rows.append({**base, 'part': 'Ps',
                             'plus': ops['plus_sin'], 'mul': ops['mul_sin'],
                             'total': ops['total_sin']})

    if not rows:
        print("  No data collected.")
        return

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f'cube_complexity_m{m_max}.csv'
    fieldnames = ['cube', 'atoms', 'n', 'part',
                  'plus', 'mul', 'total',
                  'n_pts', 'ms_per_wave', 'ns_per_pt_per_wave',
                  'sympy_ms_per_wave', 'sympy_ns_per_pt_per_wave']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV  → {csv_path}")

    timing_json_path = outdir / f'timing_sweep_m{m_max}.json'
    _write_timing_json(rows, timing_json_path)

    _plot_sweep(rows, outdir, m_max, do_timing)

    if nop_json_path:
        _write_nop_summary_json(rows, nop_json_path)



def _write_timing_json(rows, json_path):
    """Write high-precision timing data grouped by cube and Chebyshev order n."""
    payload = {
        'description': 'High-precision sweep timing and operation counts per cube/order',
        'records': []
    }

    for row in rows:
        if row['part'] != 'Ps':
            continue

        cube = row['cube']
        n = int(row['n'])
        cos_row = next((r for r in rows
                        if r['cube'] == cube and r['n'] == n and r['part'] == 'Pc'), None)
        if cos_row is None:
            continue

        payload['records'].append({
            'cube': cube,
            'atoms': int(row['atoms']),
            'n': n,
            'n_pts': int(row['n_pts']),
            'nop_cos': int(cos_row['total']),
            'nop_sin': int(row['total']),
            'nop_total': int(cos_row['total']) + int(row['total']),
            'ms_per_wave': float(row['ms_per_wave']),
            'ns_per_pt_per_wave': float(row['ns_per_pt_per_wave']),
            'sympy_ms_per_wave': float(row['sympy_ms_per_wave']),
            'sympy_ns_per_pt_per_wave': float(row['sympy_ns_per_pt_per_wave']),
        })

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"  JSON → {json_path}")


def _write_nop_summary_json(rows, json_path):
    """Write JSON: atoms -> n -> Nop_tot, plus cube metadata."""
    summary = {}
    for row in rows:
        if row['part'] != 'Ps':
            continue
        cube = row['cube']
        atoms = int(row['atoms'])
        n = int(row['n'])

        cos_row = next((r for r in rows
                        if r['cube'] == cube and r['n'] == n and r['part'] == 'Pc'), None)
        if cos_row is None:
            continue
        nop_tot = int(row['total']) + int(cos_row['total'])

        atoms_key = str(atoms)
        summary.setdefault(atoms_key, {'cube': cube, 'n_to_nop_tot': {}})
        summary[atoms_key]['n_to_nop_tot'][str(n)] = nop_tot

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'description': 'Nop_tot per selected atom-count cube across Chebyshev order n',
        'atoms_to_cube_and_nop_tot': summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"  JSON → {json_path}")


def _plot_sweep(rows, outdir, m_max, do_timing):
    import pandas as pd
    df = pd.DataFrame(rows)

    # Unique cubes sorted by n_atoms
    cube_info = (df.groupby('cube', as_index=False)['atoms']
                   .first()
                   .sort_values('atoms'))
    cubes = cube_info['cube'].tolist()
    n_cubes = len(cubes)

    cmap = plt.get_cmap('viridis', n_cubes)

    n_panels = 3 if do_timing else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f'Chebyshev sweep n=1..{m_max}  ({n_cubes} cubes)', fontsize=12)

    # --- N_op total vs n ---
    ax = axes[0]
    for ki, cube in enumerate(cubes):
        sub = df[(df['cube'] == cube) & (df['part'] == 'Ps')].sort_values('n')
        na  = sub['atoms'].iloc[0]
        sub_cos = df[(df['cube'] == cube) & (df['part'] == 'Pc')].sort_values('n')
        total = sub['total'].values + sub_cos['total'].values
        ax.semilogy(sub['n'].values, total,
                    marker='o', linewidth=1.6, markersize=4,
                    color=cmap(ki), label=fr'$N_{{\mathrm{{atom}}}}={na}$')
    ax.set_xlabel('$n$')
    ax.set_ylabel('$N_\\mathrm{op}$  (Ps + Pc)')
    ax.set_title('Total arithmetic ops vs order')
    ax.legend(fontsize=8, frameon=False,
              ncol=max(1, n_cubes // 8))
    ax.grid(True, alpha=0.3)

    if do_timing:
        # --- ms/wave vs n ---
        ax = axes[1]
        for ki, cube in enumerate(cubes):
            sub = df[(df['cube'] == cube) & (df['part'] == 'Ps')].sort_values('n')
            na  = sub['atoms'].iloc[0]
            ax.semilogy(sub['n'].values, sub['ms_per_wave'].values,
                        marker='o', linewidth=1.6, markersize=4,
                        color=cmap(ki), label=fr'$N_{{\mathrm{{atom}}}}={na}$')
        ax.set_xlabel('$n$')
        ax.set_ylabel('ms / wave')
        ax.set_title('Julia eval time vs order')
        ax.legend(fontsize=8, frameon=False,
                  ncol=max(1, n_cubes // 8))
        ax.grid(True, alpha=0.3)

        # --- ms/wave vs N_op scatter ---
        ax = axes[2]
        df_ps = df[df['part'] == 'Ps'].copy()
        df_pc = df[df['part'] == 'Pc']
        df_ps = df_ps.copy()
        df_ps['total_both'] = df_ps['total'].values + df_pc['total'].values
        sc = ax.scatter(df_ps['total_both'], df_ps['ms_per_wave'],
                        c=df_ps['n'], cmap='plasma', s=20, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='n (order)')
        ax.set_xlabel('$N_\\mathrm{op}$')
        ax.set_ylabel('ms / wave')
        ax.set_title('Time vs N_op  (coloured by order)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = Path(outdir) / f'sweep_m{m_max}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot → {fig_path}")


# ---------------------------------------------------------------------------
# Mode 2 – full: all cubes at fixed n=m, predict total QD time vs n_waves
# ---------------------------------------------------------------------------

def _collect_full_records(manager, expr_dir, m, Ng, N_DIVISIONS, X, Y, Z,
                           a, b, expand, max_cubes, regen=False):
    records = []
    for idx in sorted(manager.cube_info.keys()):
        info = manager.cube_info[idx]
        cx, cy, cz = info['center']
        dirname = _cube_dirname(cx, cy, cz)
        jl_path = find_or_generate_jl(expr_dir, dirname, m, a, b, expand=expand, regen=regen)
        if jl_path is None:
            continue
        Xf, Yf, Zf, _ = cube_interior_points(idx, Ng, N_DIVISIONS, X, Y, Z)
        if len(Xf) == 0:
            continue
        records.append({
            'idx': idx, 'n_atoms': info.get('n_atoms', 0),
            'n_pts': len(Xf), 'jl_path': jl_path,
            'Xf': Xf, 'Yf': Yf, 'Zf': Zf,
        })
        if max_cubes and len(records) >= max_cubes:
            break
    return records


def run_full_mode(manager, expr_dir, m, n_waves, k_max, L_s, d_grid,
                  julia_exe, outdir, max_cubes=None, a=None, b=None, expand=False,
                  regen=False):
    rng = np.random.default_rng(42)
    _, X, Y, Z = build_full_grid(L_s, d_grid)
    Ng, N_DIVISIONS, cube_size = infer_grid_params(manager, L_s, d_grid)
    print(f"  Grid: Ng={Ng}^3={Ng**3:,} pts  "
          f"N_DIVISIONS={N_DIVISIONS}  cube_size={cube_size:.3f} Bohr")

    all_records = _collect_full_records(
        manager, expr_dir, m, Ng, N_DIVISIONS, X, Y, Z, a, b, expand, max_cubes,
        regen=regen)
    print(f"  Cubes to run: {len(all_records)}")

    k_vals = rng.uniform(-k_max, k_max, (n_waves, 3))
    b_vals = rng.uniform(0, 2 * np.pi, n_waves)

    per_cube = []
    total_warmup_s = total_eval_s = 0.0
    total_pts = n_done = n_fail = 0

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

            eval_s   = timing.get('eval_s') or 0.0
            warmup_s = timing.get('warmup_s') or 0.0
            n_eval   = timing.get('n_eval_waves') or max(n_waves - 1, 1)
            ms_wave  = eval_s / n_eval * 1000

            total_warmup_s += warmup_s
            total_eval_s   += eval_s
            total_pts      += rec['n_pts']
            n_done         += 1

            per_cube.append({
                'cube_idx' : str(rec['idx']),
                'n_atoms'  : rec['n_atoms'],
                'n_pts'    : rec['n_pts'],
                'warmup_ms': round(warmup_s * 1000, 2),
                'ms_per_wave': round(ms_wave, 4),
            })

            if (i + 1) % 20 == 0 or i == len(all_records) - 1:
                elapsed = time.time() - t_wall_all
                print(f"  [{i+1:4d}/{len(all_records)}]  "
                      f"done={n_done}  fail={n_fail}  "
                      f"total_eval_s={total_eval_s:.1f}  elapsed={elapsed:.0f}s")

    wall_total = time.time() - t_wall_all
    n_eval_waves = max(n_waves - 1, 1)
    total_ms_per_wave = total_eval_s / n_eval_waves * 1000
    total_warmup_ms   = total_warmup_s * 1000
    ns_per_pt = total_ms_per_wave / total_pts * 1e6 if total_pts else 0.0

    print(f"\n  === Full QD summary (m={m}, n_waves={n_waves}) ===")
    print(f"  Cubes: done={n_done}  fail={n_fail}")
    print(f"  Total pts: {total_pts:,}  / {Ng**3:,}")
    print(f"  Total warmup (JIT):  {total_warmup_ms:.0f} ms")
    print(f"  Total eval/wave:     {total_ms_per_wave:.1f} ms/wave")
    print(f"  Per-point:           {ns_per_pt:.2f} ns/pt/wave")
    print(f"  Wall time (all):     {wall_total:.1f} s")
    print(f"  Predicted for N waves: warmup + N×{total_ms_per_wave:.1f} ms")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if per_cube:
        csv_path = outdir / f'full_qd_per_cube_m{m}.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(per_cube[0].keys()))
            w.writeheader(); w.writerows(per_cube)
        print(f"\n  Per-cube CSV → {csv_path}")

    summary = {
        'm': m, 'n_waves': n_waves, 'n_cubes_done': n_done,
        'total_pts': total_pts,
        'total_warmup_ms': round(total_warmup_ms, 1),
        'total_ms_per_wave': round(total_ms_per_wave, 3),
        'ns_per_pt_per_wave': round(ns_per_pt, 3),
        'wall_s': round(wall_total, 1),
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

    ax = axes[0]
    n_arr   = np.arange(1, max(500, n_waves * 5) + 1)
    t_total = total_warmup_ms + n_arr * total_ms_per_wave
    ax.plot(n_arr, t_total / 1000, 'b-',
            label='warmup + eval (s)')
    ax.plot(n_arr, n_arr * total_ms_per_wave / 1000, 'r--',
            label='eval only (s)')
    ax.axvline(n_waves, color='gray', linestyle=':', alpha=0.7,
               label=f'measured n_waves={n_waves}')
    ax.set_xlabel('n_waves')
    ax.set_ylabel('total time (s)')
    ax.set_title('Predicted QD time vs n_waves')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if per_cube:
        ms_arr = np.array([r['ms_per_wave'] for r in per_cube])
        ax.hist(ms_arr, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(ms_arr.mean(), color='r', linestyle='--',
                   label=f'mean={ms_arr.mean():.3f} ms')
        ax.set_xlabel('ms / wave (per cube)')
        ax.set_ylabel('count')
        ax.set_title('Per-cube eval time distribution')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    if per_cube:
        na_arr  = np.array([r['n_atoms']    for r in per_cube])
        npt_arr = np.array([r['n_pts']      for r in per_cube])
        ms_arr  = np.array([r['ms_per_wave'] for r in per_cube])
        sc = ax.scatter(npt_arr, ms_arr, c=na_arr, cmap='plasma', s=20, alpha=0.7)
        plt.colorbar(sc, ax=ax, label='n_atoms')
        if len(npt_arr) >= 3:
            coef = np.polyfit(npt_arr, ms_arr, 1)
            xs = np.linspace(npt_arr.min(), npt_arr.max(), 200)
            ax.plot(xs, np.polyval(coef, xs), 'r--', alpha=0.7,
                    label=f'{coef[0]*1e6:.1f} ns/pt/wave')
            ax.legend(fontsize=8)
        ax.set_xlabel('n_pts')
        ax.set_ylabel('ms / wave')
        ax.set_title('Per-cube: time vs n_pts')
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
        description='Sweep n=1..m per cube: N_op + Julia eval timing.'
    )
    parser.add_argument('--vexpr_dir', required=True,
                        help='V_expr pkl directory (from space partition)')
    parser.add_argument('--expr_dir', required=True,
                        help='H^n pkl / Julia script directory (Stage 3/4 output)')
    parser.add_argument('--m', type=int, default=8,
                        help='Max Chebyshev order used by full mode and default sweep upper bound (default 8).')
    parser.add_argument('--m_sweep_max', type=int, default=None,
                        help='Optional sweep-only upper bound for n. Example: --m 8 --m_sweep_max 5 sweeps n=1..5')
    parser.add_argument('--E_lo', type=float, default=0.0,
                        help='Lower energy bound for filter (default 0.0 Hartree)')
    parser.add_argument('--E_hi', type=float, default=None,
                        help='Upper energy bound. If omitted: parsed from existing '
                             '.jl filenames, or auto-estimated from grid.')
    parser.add_argument('--expand', action='store_true', default=False,
                        help='Use expanded pkl codegen path (default False)')
    parser.add_argument('--L_s', type=float, default=22.0,
                        help='Half-box size in Bohr (default 22.0 for QD R=17)')
    parser.add_argument('--d_grid', type=float, default=0.625,
                        help='Grid spacing in Bohr (default 0.625)')
    parser.add_argument('--n_waves', type=int, default=20,
                        help='Plane waves for timing (default 20; use >=10)')
    parser.add_argument('--k_max', type=float, default=1.0,
                        help='Max k-vector magnitude (default 1.0)')
    parser.add_argument('--no_timing', action='store_true', default=False,
                        help='Skip Julia timing; only count N_op (fast)')
    parser.add_argument('--sympy_timing', action='store_true', default=False,
                        help='Also benchmark direct SymPy/numpy expression evaluation time')
    parser.add_argument('--sample', type=int, default=20,
                        help='Cubes to sample in sweep mode (default 20)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default 0)')
    parser.add_argument('--select_atoms', type=str, default=None,
                        help='Comma-separated atom counts (e.g. 1,2,3,8,10,19). '
                             'If set, sweep picks one cube per requested atom count and ignores --sample.')
    parser.add_argument('--mode', choices=['sweep', 'full', 'both'],
                        default='sweep',
                        help='sweep: n-sweep for sampled cubes; '
                             'full: all cubes at fixed m; '
                             'both (default: sweep)')
    parser.add_argument('--max_cubes', type=int, default=None,
                        help='Limit cubes in full mode (for testing)')
    parser.add_argument('--julia_exe', default='julia',
                        help='Julia executable (default julia)')
    parser.add_argument('--outdir', default='figs_qd_timing',
                        help='Output directory (default figs_qd_timing/)')
    parser.add_argument('--nop_json', default=None,
                        help='Optional JSON path to save atoms->n->Nop_tot summary in sweep mode')
    parser.add_argument('--regen', action='store_true', default=False,
                        help='Force re-generation of all .jl files even if they exist on disk '
                             '(needed after julia_codegen.py is updated)')
    parser.add_argument('--count_pkl', action='store_true', default=False,
                        help='Sweep mode: count ops from H_power_n.pkl directly '
                             '(measures H^n complexity, fast, no OOM risk). '
                             'Skips .jl generation and Julia timing.')
    parser.add_argument('--hn_julia', action='store_true', default=False,
                        help='Sweep mode: generate eval_Hn_m{n}.jl from H_power_n.pkl '
                             'and time Julia eval of H^n (single power, not filter). '
                             'Fast: loads pkl directly, no Chebyshev combination.')
    args = parser.parse_args()

    if args.n_waves < 2 and not args.no_timing:
        print('WARNING: n_waves < 2 gives no timed iterations. Use --n_waves 20+.\n')

    if args.m < 1:
        print('ERROR: --m must be >= 1')
        sys.exit(1)
    if args.m_sweep_max is not None and args.m_sweep_max < 1:
        print('ERROR: --m_sweep_max must be >= 1')
        sys.exit(1)

    print(f"Loading cube manager from {args.vexpr_dir} ...")
    manager = CubicExpressionManager(args.vexpr_dir)
    if not manager.cube_info:
        print("ERROR: No cubes found in vexpr_dir. Check the path.")
        sys.exit(1)
    print(f"  {len(manager.cube_info)} cubes loaded")

    try:
        Ng, N_DIV, csz = infer_grid_params(manager, args.L_s, args.d_grid)
        print(f"  Inferred: Ng={Ng}  N_DIVISIONS={N_DIV}  cube_size={csz:.3f} Bohr")
    except Exception as exc:
        print(f"  Warning: could not infer grid params: {exc}")
        Ng = int(np.ceil(2 * args.L_s / args.d_grid))

    # Determine a, b for filter (needed to generate .jl files)
    a = b = None
    if args.E_hi is not None:
        a = 2.0 / (args.E_hi - args.E_lo)
        b = -(args.E_hi + args.E_lo) / (args.E_hi - args.E_lo)
        print(f"  Filter: E_lo={args.E_lo}  E_hi={args.E_hi:.4f} "
              f"→ a={a:.8g}  b={b:.8g}")
    else:
        # Try to read a/b from any existing .jl file (any order 1..m)
        for try_m in range(args.m, 0, -1):
            sample_jl = _find_any_jl_in_dir(args.expr_dir, try_m)
            if sample_jl is not None:
                try:
                    a, b = _parse_ab_from_jl_name(sample_jl)
                    E_hi_inf = 2.0 / a + args.E_lo
                    print(f"  Filter a/b parsed from {sample_jl.name}: "
                          f"a={a:.8g}  b={b:.8g}  (E_hi≈{E_hi_inf:.3f})")
                    break
                except Exception:
                    pass
        if a is None:
            E_hi_est = _estimate_E_hi(args.L_s, Ng)
            a = 2.0 / (E_hi_est - args.E_lo)
            b = -(E_hi_est + args.E_lo) / (E_hi_est - args.E_lo)
            print(f"  Filter: E_hi auto-estimated={E_hi_est:.3f} Hartree "
                  f"→ a={a:.8g}  b={b:.8g}")
    print()


    select_atoms = None
    if args.select_atoms:
        try:
            select_atoms = [int(x.strip()) for x in args.select_atoms.split(',') if x.strip()]
        except ValueError:
            print('ERROR: --select_atoms must be comma-separated integers, e.g. 1,2,3,8,10,19')
            sys.exit(1)
        if not select_atoms:
            print('ERROR: --select_atoms was provided but no valid integers were found.')
            sys.exit(1)

    if args.mode in ('sweep', 'both'):
        print(f"{'='*60}")
        print(f"Sweep mode  (m={args.m}, sample={args.sample})")
        print(f"{'='*60}")
        sweep_m_max = args.m if args.m_sweep_max is None else min(args.m, args.m_sweep_max)
        if args.m_sweep_max is not None and args.m_sweep_max > args.m:
            print(f"  NOTE: --m_sweep_max={args.m_sweep_max} > --m={args.m}; using n<= {sweep_m_max}.")
        run_sweep_mode(
            manager, args.expr_dir, sweep_m_max,
            args.n_waves, args.k_max,
            args.L_s, args.d_grid,
            args.sample, args.seed,
            args.julia_exe, args.outdir,
            a=a, b=b, select_atoms=select_atoms, expand=args.expand,
            do_timing=not args.no_timing,
            do_sympy_timing=args.sympy_timing,
            nop_json_path=args.nop_json,
            regen=args.regen,
            count_pkl=args.count_pkl,
            hn_julia=args.hn_julia)

    if args.mode in ('full', 'both'):
        print(f"\n{'='*60}")
        print(f"Full QD mode  (m={args.m}, n_waves={args.n_waves})")
        print(f"{'='*60}")
        run_full_mode(
            manager, args.expr_dir, args.m,
            args.n_waves, args.k_max,
            args.L_s, args.d_grid,
            args.julia_exe, args.outdir,
            max_cubes=args.max_cubes,
            a=a, b=b, expand=args.expand, regen=args.regen)

    print('\nDone.')


if __name__ == '__main__':
    main()
