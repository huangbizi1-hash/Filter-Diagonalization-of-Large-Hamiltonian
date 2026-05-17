import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import primme

from ho3d_solvers_v2 import build_3d_fft_operator, build_3d_fd_operator, FD_STENCILS


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
    parser.add_argument('--solver', choices=['explosion', 'jdqmr'], default='explosion',
                        help='explosion: 保留原爆炸滤波流程（占位）；jdqmr: 用JDQMR求HO3D前n_levels个特征值')
    parser.add_argument('--N', type=int, default=64, help='每个维度网格数')
    parser.add_argument('--n-levels', type=int, default=20, help='输出本征值个数（JDQMR模式）')
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--max-matvecs', type=int, default=100000)
    parser.add_argument('--blocksize', type=int, default=1)
    parser.add_argument('--fft-only', action='store_true')
    parser.add_argument('--out-dir', type=str, default='fd_results')
    args = parser.parse_args()

    if args.solver == 'explosion':
        raise NotImplementedError(
            'explosion 模式请继续使用你现有的爆炸滤波实现；新增 --solver jdqmr 已可用于 HO3D 前20本征值对比。')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    rows = []
    H_fft, _, _ = build_3d_fft_operator(args.N, potential_grid=None)
    e_fft, t_fft, mv_fft = run_jdqmr(H_fft, args.n_levels, args.tol, args.max_matvecs, args.blocksize)
    rows.append({'method': 'FFT', 'fd_order': None, 'evals': e_fft.tolist(), 't_wall': t_fft, 'n_mv': mv_fft})

    if not args.fft_only:
        for order in sorted(FD_STENCILS.keys()):
            H_fd, _, _ = build_3d_fd_operator(args.N, potential_grid=None, fd_order=order)
            evals, t_wall, n_mv = run_jdqmr(H_fd, args.n_levels, args.tol, args.max_matvecs, args.blocksize)
            rows.append({'method': f'FD-{order}', 'fd_order': order, 'evals': evals.tolist(), 't_wall': t_wall, 'n_mv': n_mv})

    out = {
        'script': 'compare_fd_fft_explosion.py',
        'datetime': ts,
        'config': vars(args),
        'problem': 'ho3d',
        'results': rows,
    }
    out_path = out_dir / f'compare_fd_fft_explosion_{args.solver}_{ts}.json'
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f'Saved: {out_path}')

    print('\nMethod   First 5 eigenvalues')
    for row in rows:
        preview = ', '.join(f'{x:.8f}' for x in row['evals'][:5])
        print(f"{row['method']:<7} {preview}")


if __name__ == '__main__':
    main()
