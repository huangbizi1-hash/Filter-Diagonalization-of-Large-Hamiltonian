import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import primme

from ho3d_solvers_v2 import build_3d_fft_operator, build_3d_fd_operator, FD_STENCILS


def parse_n_sweep(spec: str):
    """Parse N sweep spec like '8:16' (inclusive) or '8,10,12,16'."""
    spec = spec.strip()
    if not spec:
        raise ValueError("--N_sweep 不能为空")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError("--N_sweep 使用 start:end 形式时只能有一个冒号")
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            raise ValueError("--N_sweep start 不能大于 end")
        vals = list(range(start, end + 1))
    else:
        vals = [int(x.strip()) for x in spec.split(",") if x.strip()]

    # FFT/FD 算子都使用周期边界，N 至少建议 >= 6；另外剔除非正数
    vals = [n for n in vals if n > 0]
    if not vals:
        raise ValueError("--N_sweep 解析后为空")
    return vals


def run_jdqmr(H_op, n_levels, tol, max_matvecs, blocksize):
    ncv = max(80, 2 * n_levels)
    t0 = time.perf_counter()
    evals, _, stats = primme.eigsh(
        H_op,
        k=n_levels,
        which='SA',
        method='PRIMME_JDQMR',
        maxBlockSize=blocksize,
        ncv=ncv,
        tol=tol,
        maxMatvecs=max_matvecs,
        return_stats=True,
        return_history=False,
    )
    return np.sort(evals.real), time.perf_counter() - t0, int(stats.get('numMatvecs', -1))


def main():
    parser = argparse.ArgumentParser(
        description='Compare FFT/FD orders with explosion filter or JDQMR on HO3D')
    parser.add_argument('--solver', choices=['explosion', 'jdqmr'], default='jdqmr',
                        help='explosion: 爆炸滤波占位；jdqmr: 用JDQMR求HO3D本征值')
    parser.add_argument('--N', type=int, default=64, help='单个网格 N（不使用 --N_sweep 时生效）')
    parser.add_argument('--N_sweep', type=str, default=None,
                        help='网格扫描："8:16"(含端点) 或 "8,10,12,16"')
    parser.add_argument('--box_L', type=float, default=5.0,
                        help='HO3D 盒子半长 L（用于 potential_grid=None）')
    parser.add_argument('--n-levels', type=int, default=20, help='输出本征值个数（JDQMR模式）')
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--max-matvecs', type=int, default=100000)
    parser.add_argument('--blocksize', type=int, default=1)
    parser.add_argument('--fft-only', action='store_true')
    parser.add_argument('--out-dir', type=str, default='fd_results')
    args = parser.parse_args()

    if args.solver == 'explosion':
        raise NotImplementedError('当前脚本仅实现 --solver jdqmr；explosion 请继续使用现有 explosion 脚本。')

    n_values = parse_n_sweep(args.N_sweep) if args.N_sweep else [args.N]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    rows = []
    for n in n_values:
        print(f"\n=== N={n}, L={args.box_L} ===")
        H_fft, _, _ = build_3d_fft_operator(n, potential_grid=None, L=args.box_L)
        e_fft, t_fft, mv_fft = run_jdqmr(H_fft, args.n_levels, args.tol, args.max_matvecs, args.blocksize)
        rows.append({'N': n, 'method': 'FFT', 'fd_order': None,
                     'evals': e_fft.tolist(), 't_wall': t_fft, 'n_mv': mv_fft})

        if not args.fft_only:
            for order in sorted(FD_STENCILS.keys()):
                H_fd, _, _ = build_3d_fd_operator(n, potential_grid=None, fd_order=order, L=args.box_L)
                evals, t_wall, n_mv = run_jdqmr(H_fd, args.n_levels, args.tol, args.max_matvecs, args.blocksize)
                rows.append({'N': n, 'method': f'FD-{order}', 'fd_order': order,
                             'evals': evals.tolist(), 't_wall': t_wall, 'n_mv': n_mv})

    out = {
        'script': 'compare_fd_fft_explosion.py',
        'datetime': ts,
        'config': vars(args),
        'problem': 'ho3d',
        'results': rows,
    }
    out_path = out_dir / f'compare_fd_fft_explosion_{args.solver}_{ts}.json'
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f'\nSaved: {out_path}')

    print('\nN   Method   First 5 eigenvalues')
    for row in rows:
        preview = ', '.join(f'{x:.8f}' for x in row['evals'][:5])
        print(f"{row['N']:<3} {row['method']:<7} {preview}")


if __name__ == '__main__':
    main()
