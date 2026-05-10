#!/usr/bin/env python3
"""Analyze symbolic complexity (# '+' and '*' ops) for selected cubes and H^n orders."""
from __future__ import annotations
import argparse, pickle, random, re
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import sympy as sp
except Exception as e:  # noqa
    raise SystemExit("sympy is required. Please install with: pip install sympy") from e

ATOM_RE = re.compile(r"_total_(\d+)_inside_")
CENTER_RE = re.compile(r"cube(?:_center)?_([^_]+)_([^_]+)_([^_]+)")

def parse_atom_count(name: str) -> int | None:
    m = ATOM_RE.search(name)
    return int(m.group(1)) if m else None

def parse_center(name: str):
    m = CENTER_RE.search(name)
    if not m:
        return None
    return tuple(float(x.replace('neg', '-')) if 'neg' in x else float(x) for x in m.groups())


def build_center_to_atoms(vexpr_dir: Path):
    out = {}
    for p in vexpr_dir.glob('*.pkl'):
        c = parse_center(p.name)
        n = parse_atom_count(p.name)
        if c is not None and n is not None:
            out[c] = n
    return out


def count_plus_mul(expr):
    visual = sp.count_ops(expr, visual=True)
    plus = int(visual.coeff(sp.Symbol('ADD')))
    mul = int(visual.coeff(sp.Symbol('MUL')))
    return plus, mul, plus + mul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vexpr_dir', default='QD_R11_Vexpr_cell')
    ap.add_argument('--expr_dir', default='QD_R11_Julia_exp/no_expansion_cell')
    ap.add_argument('--n_max', type=int, default=8)
    ap.add_argument('--sample', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default='figs_cube_complexity')
    args = ap.parse_args()

    vexpr_dir = Path(args.vexpr_dir)
    expr_dir = Path(args.expr_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    center_atoms = build_center_to_atoms(vexpr_dir)

    candidates = []
    for d in sorted(expr_dir.glob('cube_*')):
        c = parse_center(d.name)
        if c is None:
            continue
        atoms = center_atoms.get(c, None)
        if atoms is None or atoms <= 0:
            continue
        if (d / 'H_power_0.pkl').exists():
            candidates.append((d, atoms))

    random.seed(args.seed)
    random.shuffle(candidates)
    selected = candidates[: args.sample]

    rows = []
    for d, atoms in selected:
        for n in range(args.n_max + 1):
            fp = d / f'H_power_{n}.pkl'
            if not fp.exists():
                continue
            with open(fp, 'rb') as f:
                data = pickle.load(f)
            for part in ('Ps', 'Pc'):
                plus, mul, total = count_plus_mul(data[part])
                rows.append((d.name, atoms, n, part, plus, mul, total))

    csv = out_dir / 'cube_complexity.csv'
    with open(csv, 'w') as f:
        f.write('cube,atoms,n,part,plus,mul,total\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')

    # Plot total complexity of (Ps+Pc)
    by = {}
    for cube, atoms, n, part, plus, mul, total in rows:
        by.setdefault((cube, atoms, n), 0)
        by[(cube, atoms, n)] += total

    plt.figure(figsize=(10, 6))
    for cube, atoms in {(r[0], r[1]) for r in rows}:
        xs, ys = [], []
        for n in range(args.n_max + 1):
            k = (cube, atoms, n)
            if k in by:
                xs.append(n); ys.append(by[k])
        if xs:
            plt.plot(xs, ys, marker='o', alpha=0.7, label=f'{cube[:24]}.. (atoms={atoms})')

    plt.xlabel('n in H^n')
    plt.ylabel('complexity = #ADD + #MUL (Ps + Pc)')
    plt.title('Selected cubes symbolic complexity')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig = out_dir / 'cube_complexity.png'
    plt.savefig(fig, dpi=160)
    print(f'Saved: {csv}')
    print(f'Saved: {fig}')

if __name__ == '__main__':
    main()
