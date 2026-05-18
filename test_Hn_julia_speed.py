#!/usr/bin/env python3
"""test_Hn_julia_speed.py

Test the evaluation speed of individual H^n Julia expressions on a 3-D grid.

For each n = 0, 1, ..., n_max, generates a Julia script that evaluates
H^n * sin(k·r + b) on all grid points, warms up the JIT, then measures
evaluation time for n_waves plane waves.

Usage
-----
  python test_Hn_julia_speed.py \\
      --N 12 --box_L 5.0 \\
      --n_max 10 --n_waves 100 \\
      --cache_dir ho3d_symbolic_cache \\
      --julia_exe julia \\
      --force_rebuild
"""

import argparse
import json
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).parent))


# ── helpers (standalone, no imports from comparison script) ──────────────────

def _load_H_power_n(cache_dir: Path, n: int) -> dict:
    fpath = cache_dir / f'H_power_{n}.pkl'
    if not fpath.exists():
        raise FileNotFoundError(f"H_power_{n}.pkl not found in {cache_dir}")
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def _sympy_to_julia(expr) -> str:
    """Convert a sympy scalar expression to Julia code."""
    s = sp.julia_code(expr)
    for op in ('.+', '.-', '.*', './', '.^'):
        s = s.replace(op, op[1:])
    return s


def _build_hn_julia_script(Pc_n, Ps_n) -> str:
    """Generate a Julia batch script that evaluates H^n * sin(k·r+b).

    Identical I/O convention to the combined filter script:
        julia <script>.jl  grid.bin  kvals.bin  out.bin  N  n_waves
    Stdout last line: JSON {warmup_s, eval_s, n_eval_waves}
    """
    jl_cos = _sympy_to_julia(Pc_n)
    jl_sin = _sympy_to_julia(Ps_n)

    return f"""\
# Auto-generated H^n Julia evaluation script — DO NOT EDIT
# H^n * sin(k·r+b) = Pc_n * cos(phase) + Ps_n * sin(phase)
#
# Usage:  julia <script>.jl  grid.bin  kvals.bin  out.bin  N  n_waves

function eval_Hn_wave!(
        out  :: AbstractVector{{Float64}},
        X    :: Vector{{Float64}},
        Y    :: Vector{{Float64}},
        Z    :: Vector{{Float64}},
        kx   :: Float64,
        ky   :: Float64,
        kz   :: Float64,
        b    :: Float64)

    N = length(X)
    phase = @. kx * X + ky * Y + kz * Z + b
    cos_p = cos.(phase)
    sin_p = sin.(phase)

    @inbounds for i in 1:N
        x = X[i]; y = Y[i]; z = Z[i]
        p_cos = ({jl_cos})
        p_sin = ({jl_sin})
        out[i] = cos_p[i] * p_cos + sin_p[i] * p_sin
    end
end


function main()
    if length(ARGS) != 5
        error("Usage: julia script.jl grid.bin kvals.bin out.bin N n_waves")
    end

    grid_file  = ARGS[1]
    kvals_file = ARGS[2]
    out_file   = ARGS[3]
    N          = parse(Int, ARGS[4])
    n_waves    = parse(Int, ARGS[5])

    grid_bytes = read(grid_file)
    grid_data  = reinterpret(Float64, grid_bytes)
    X = Vector{{Float64}}(grid_data[1:N])
    Y = Vector{{Float64}}(grid_data[N+1:2N])
    Z = Vector{{Float64}}(grid_data[2N+1:3N])

    kb_bytes = read(kvals_file)
    kb       = reinterpret(Float64, kb_bytes)

    out_all = Vector{{Float64}}(undef, N * n_waves)

    t_warmup = @elapsed eval_Hn_wave!(
        view(out_all, 1:N), X, Y, Z,
        Float64(kb[1]), Float64(kb[2]), Float64(kb[3]), Float64(kb[4]))

    t_eval = @elapsed begin
        for iw in 1:n_waves-1
            ofs = iw * N + 1
            kid = iw * 4 + 1
            eval_Hn_wave!(
                view(out_all, ofs:ofs+N-1), X, Y, Z,
                Float64(kb[kid]),   Float64(kb[kid+1]),
                Float64(kb[kid+2]), Float64(kb[kid+3]))
        end
    end

    open(out_file, "w") do io
        write(io, out_all)
    end

    n_eval = max(n_waves - 1, 1)
    println(\"{{\\\"warmup_s\\\": $t_warmup, \\\"eval_s\\\": $t_eval, \" *
            \"\\\"n_eval_waves\\\": $n_eval}}\")
end

main()
"""


def ensure_Hn_julia_script(cache_dir: Path, n: int,
                            force_rebuild: bool = False) -> Path:
    """Build (or load) Julia script for H^n evaluation."""
    jl_path = cache_dir / f'julia_Hn_{n}.jl'

    if force_rebuild and jl_path.exists():
        print(f"  --force_rebuild: deleting {jl_path.name}")
        jl_path.unlink()

    if jl_path.exists():
        return jl_path

    t0 = time.perf_counter()
    print(f"  Building julia_Hn_{n}.jl ...", end=' ', flush=True)

    if n == 0:
        Pc_n = sp.Integer(0)
        Ps_n = sp.Integer(1)
    else:
        entry = _load_H_power_n(cache_dir, n)
        Pc_n = entry['Pc']
        Ps_n = entry['Ps']

    src = _build_hn_julia_script(Pc_n, Ps_n)
    jl_path.write_text(src, encoding='utf-8')
    print(f"{time.perf_counter()-t0:.2f}s")
    return jl_path


def run_julia_Hn_eval(jl_path: Path, x1d: np.ndarray,
                      k_vals: np.ndarray, b_vals: np.ndarray,
                      julia_exe: str) -> dict:
    """Run Julia to evaluate H^n*sin(k·r+b) for all plane waves.

    Returns timing dict (warmup_s, eval_s, n_eval_waves, total_wall_s).
    """
    N_grid  = len(x1d)
    n_waves = len(k_vals)
    X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')

    with tempfile.TemporaryDirectory(prefix='hn_speed_') as tmp:
        tmp       = Path(tmp)
        grid_bin  = tmp / 'grid.bin'
        kvals_bin = tmp / 'kvals.bin'
        out_bin   = tmp / 'out.bin'

        np.concatenate([X.ravel(), Y.ravel(), Z.ravel()]).astype('<f8').tofile(
            str(grid_bin))
        kb = np.column_stack([k_vals, b_vals.reshape(-1, 1)]).astype('<f8')
        kb.ravel().tofile(str(kvals_bin))

        cmd = [julia_exe, str(jl_path),
               str(grid_bin), str(kvals_bin), str(out_bin),
               str(N_grid**3), str(n_waves)]

        t0   = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"Julia exited {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}")

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


def make_grid(N: int, box_L: float):
    d = 2.0 * box_L / N
    L = (N - 1) * d / 2.0
    return d, np.linspace(-L, L, N)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--N',             type=int,   default=12,
                    help='Grid size per dimension')
    ap.add_argument('--box_L',         type=float, default=5.0,
                    help='Box half-length')
    ap.add_argument('--n_max',         type=int,   default=5,
                    help='Maximum H^n order to test')
    ap.add_argument('--n_waves',       type=int,   default=50,
                    help='Number of plane waves per test (wave 0 = warmup)')
    ap.add_argument('--seed',          type=int,   default=42)
    ap.add_argument('--k_max',         type=float, default=0.0,
                    help='k cutoff (0 = pi/d)')
    ap.add_argument('--cache_dir',     default='ho3d_symbolic_cache',
                    help='Directory containing H_power_n.pkl files')
    ap.add_argument('--julia_exe',     default='julia')
    ap.add_argument('--force_rebuild', action='store_true',
                    help='Rebuild cached .jl scripts')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    N         = args.N
    d, x1d    = make_grid(N, args.box_L)
    k_max     = args.k_max if args.k_max > 0 else np.pi / d
    N3        = N ** 3

    rng    = np.random.default_rng(args.seed)
    k_vals = rng.uniform(-k_max, k_max, (args.n_waves, 3))
    b_vals = rng.uniform(0.0, 2 * np.pi, args.n_waves)

    print("=== H^n Julia evaluation speed test ===")
    print(f"  N={N}  d={d:.4f}  N3={N3}  k_max={k_max:.3f}")
    print(f"  n_max={args.n_max}  n_waves={args.n_waves} (wave 0 = warmup)")
    print(f"  cache_dir={cache_dir}")
    print()

    rows = []
    for n in range(args.n_max + 1):
        print(f"-- H^{n} --")
        try:
            jl_path = ensure_Hn_julia_script(cache_dir, n, args.force_rebuild)
            timing  = run_julia_Hn_eval(jl_path, x1d, k_vals, b_vals, args.julia_exe)

            es       = timing.get('eval_s') or 0.0
            nw       = timing.get('n_eval_waves') or 1
            ms_wave  = es / nw * 1e3
            ns_point = es / nw / N3 * 1e9

            print(f"  warmup={timing.get('warmup_s','?'):.5f}s  "
                  f"eval={es:.5f}s/{nw} waves  "
                  f"{ms_wave:.4f} ms/wave  "
                  f"{ns_point:.4f} ns/point  "
                  f"wall={timing['total_wall_s']:.2f}s")

            rows.append({
                'n': n, 'ok': True,
                'warmup_s': timing.get('warmup_s'),
                'eval_s':   es,
                'n_waves':  nw,
                'ms_wave':  ms_wave,
                'ns_point': ns_point,
                'wall_s':   timing['total_wall_s'],
            })
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append({'n': n, 'ok': False, 'error': str(exc)})

    # Summary table
    print()
    print(f"{'n':>4}  {'ms/wave':>10}  {'ns/point':>10}  {'eval_s':>8}  {'wall_s':>7}")
    print('-' * 52)
    for r in rows:
        if not r['ok']:
            print(f"{r['n']:>4}  ERROR: {r.get('error','?')}")
        else:
            print(f"{r['n']:>4}  {r['ms_wave']:>10.4f}  {r['ns_point']:>10.4f}  "
                  f"{r['eval_s']:>8.5f}  {r['wall_s']:>7.2f}")


if __name__ == '__main__':
    main()
