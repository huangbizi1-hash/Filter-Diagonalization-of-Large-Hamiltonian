#!/usr/bin/env python3
"""plot_fft_h_timing.py

从 fft_h_timing_60to80.json 中提取 FFT 单次 H 作用时间并绘图。

数据来源：--solver explosion 的 filter_per_state_s 字段。
Chebyshev 递推每步调用一次 H，共 cheb_m 次，因此：

    t_H_per_state = filter_per_state_s / cheb_m

注：explosion 模式只对 n_random(=4) 个随机态计时，统计量有限；
    --solver bench 模式用 h_repeat=500 次平均，结果更准确。

用法：
    python plot_fft_h_timing.py fft_h_timing_60to80.json
    python plot_fft_h_timing.py fft_h_timing_60to80.json --out timing.png --no_show
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sympy import factorint  # 用来显示 N 的质因数分解（可选）

try:
    from sympy import factorint as _fi
    def _factstr(n):
        d = _fi(n)
        return '×'.join(f'{p}^{e}' if e > 1 else str(p) for p, e in sorted(d.items()))
except ImportError:
    def _factstr(n): return ''


# ---------------------------------------------------------------------------

def load_json(path: Path) -> tuple:
    with open(path) as f:
        data = json.load(f)
    params = data['params']
    cheb_m = params['cheb_m']
    sweep  = data['sweep']

    rows = []
    for r in sweep:
        fft_entry = next(
            (m for m in r['methods'] if m['label'] == 'fft'), None)
        if fft_entry is None:
            continue
        rows.append({
            'N'              : r['N'],
            'N3'             : r['N3'],
            'd'              : r['d'],
            'filter_s'       : fft_entry['filter_time_s'],
            'per_state_s'    : fft_entry['filter_per_state_s'],
            'per_pt_us'      : fft_entry['filter_per_pt_us'],
            't_H_ms'         : fft_entry['filter_per_state_s'] / cheb_m * 1e3,
            't_H_us_per_pt'  : fft_entry['filter_per_pt_us']  / cheb_m,
        })
    return params, cheb_m, rows


def is_power2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def smooth_composite(n: int) -> int:
    """Return largest prime factor of n (smaller = more FFT-friendly)."""
    k = n
    largest = 1
    for p in range(2, k + 1):
        if p * p > k:
            break
        while k % p == 0:
            largest = p
            k //= p
    if k > 1:
        largest = k
    return largest


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json', help='Input JSON file')
    ap.add_argument('--out', default=None, help='Output PNG path')
    ap.add_argument('--no_show', action='store_true',
                    help='Do not call plt.show()')
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    json_path = Path(args.json)
    params, cheb_m, rows = load_json(json_path)

    N_arr     = np.array([r['N']            for r in rows])
    N3_arr    = np.array([r['N3']           for r in rows])
    d_arr     = np.array([r['d']            for r in rows])
    tH_ms     = np.array([r['t_H_ms']       for r in rows])
    tH_us_pt  = np.array([r['t_H_us_per_pt'] for r in rows])
    largest_p = np.array([smooth_composite(n) for n in N_arr])

    # ── print table ──────────────────────────────────────────────────────────
    hdr = (f'{"N":>4} {"N³":>8} {"d(Bohr)":>9} '
           f'{"filter/state(ms)":>18} {"t_H(ms)":>10} '
           f'{"t_H/N³(µs/pt)":>15} {"largest prime":>14}  factorisation')
    print(hdr)
    print('-' * len(hdr))
    for i, r in enumerate(rows):
        tag = ' ← 2^' + str(int(np.log2(r['N']))) if is_power2(r['N']) else ''
        print(f'{r["N"]:>4} {r["N3"]:>8} {r["d"]:>9.5f} '
              f'{r["per_state_s"]*1e3:>18.3f} {r["t_H_ms"]:>10.3f} '
              f'{r["t_H_us_per_pt"]:>15.5f} {largest_p[i]:>14}  '
              f'{_factstr(r["N"])}{tag}')

    # ── color by largest prime ────────────────────────────────────────────────
    # FFT速度主要由最大质因数决定：质因数越小越快
    uniq_primes = sorted(set(largest_p))
    cmap = matplotlib.colormaps.get_cmap('RdYlGn_r')
    norm = matplotlib.colors.Normalize(vmin=min(uniq_primes),
                                       vmax=max(uniq_primes))
    colors = [cmap(norm(p)) for p in largest_p]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    title = (f'FFT  t_H per state  (cheb_m={cheb_m}, '
             f'box_L={params["box_L"]} Bohr, ω={params["omega"]})\n'
             f't_H = filter_per_state / {cheb_m}   '
             f'[colour = largest prime factor of N]')
    fig.suptitle(title, fontsize=11)

    # ── left: t_H (ms) vs N ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(N_arr, tH_ms, '-', color='#aaaaaa', lw=1.2, zorder=1)
    sc = ax.scatter(N_arr, tH_ms, c=colors, s=70, zorder=2,
                    edgecolors='k', linewidths=0.4)

    # annotate points where N is a power of 2
    for i, n in enumerate(N_arr):
        if is_power2(n):
            ax.annotate(f'N={n}\n(2^{int(np.log2(n))})',
                        xy=(n, tH_ms[i]),
                        xytext=(n + 0.4, tH_ms[i] * 1.06),
                        fontsize=8, color='green',
                        arrowprops=dict(arrowstyle='->', color='green', lw=0.8))

    # annotate slow points (largest prime ≥ 31)
    for i, n in enumerate(N_arr):
        if largest_p[i] >= 31:
            ax.annotate(f'N={n}\n(p={largest_p[i]})',
                        xy=(n, tH_ms[i]),
                        xytext=(n - 2.5, tH_ms[i] * 1.05),
                        fontsize=7.5, color='firebrick',
                        arrowprops=dict(arrowstyle='->', color='firebrick', lw=0.7))

    ax.set_xlabel('N  (grid points per axis)', fontsize=11)
    ax.set_ylabel('t_H  (ms per state)', fontsize=11)
    ax.set_title('H-apply time vs N', fontsize=11)
    ax.grid(True, ls=':', alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    # colour bar for largest prime
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label('largest prime factor of N', fontsize=8)
    cbar.set_ticks(uniq_primes)

    # ── right: t_H / N³ (µs/pt) vs N ────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(N_arr, tH_us_pt, '-', color='#aaaaaa', lw=1.2, zorder=1)
    ax2.scatter(N_arr, tH_us_pt, c=colors, s=70, zorder=2,
                edgecolors='k', linewidths=0.4)

    # N³ log N reference (ideal FFT scaling)
    ref = N3_arr * np.log2(N_arr) / (N3_arr[0] * np.log2(N_arr[0])) * tH_us_pt[0]
    ax2.plot(N_arr, ref, '--', color='gray', lw=1, alpha=0.7,
             label='N³ log₂N  (ideal FFT scaling, normalised)')

    ax2.set_xlabel('N  (grid points per axis)', fontsize=11)
    ax2.set_ylabel('t_H / N³  (µs per grid point)', fontsize=11)
    ax2.set_title('Per-point H cost vs N', fontsize=11)
    ax2.grid(True, ls=':', alpha=0.4)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.legend(fontsize=8)

    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=ax2, pad=0.02, fraction=0.04)
    cbar2.set_label('largest prime factor of N', fontsize=8)
    cbar2.set_ticks(uniq_primes)

    plt.tight_layout()

    out_png = Path(args.out) if args.out else json_path.with_suffix('.png')
    fig.savefig(out_png, dpi=args.dpi, bbox_inches='tight')
    print(f'\nSaved → {out_png}')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
