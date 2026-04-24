"""
summarize_sweep.py
读取 results_sweep/ 下所有 results.json，汇总并绘制对比图。

用法：
  python summarize_sweep.py
  python summarize_sweep.py --root results_sweep --out summary_sweep
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--root", default="results_sweep")
p.add_argument("--out",  default="summary_sweep")
p.add_argument("--E_target", type=float, default=-0.1,
               help="Energy threshold to count Ritz values below (default -0.1)")
args = p.parse_args()

root    = Path(args.root)
out_dir = Path(args.out)
out_dir.mkdir(exist_ok=True)

# ── 读取所有 results.json ──
records = []
for json_path in sorted(root.glob("*/results.json")):
    tag = json_path.parent.name
    with open(json_path) as f:
        d = json.load(f)
    params  = d["params"]
    energies = np.array(d["ritz_energies"])
    n_below  = int(np.sum(energies < args.E_target))
    records.append({
        "tag":       tag,
        "mode":      "multistage" if params["n_stages"] > 1 else "single",
        "n_stages":  params["n_stages"],
        "n_states":  params["n_states"],
        "E_lower":   params["E_lower"],
        "rank":      d["rank"],
        "n_ritz":    d["n_ritz"],
        "n_below":   n_below,
        "energies":  energies,
        "timing":    d["timing"],
    })
    print(f"  {tag:30s}  n_states={params['n_states']:5d}  "
          f"rank={d['rank']:4d}  E<{args.E_target}: {n_below:3d}  "
          f"t={d['timing']['total_s']:.1f}s")

if not records:
    print(f"No results.json found under {root}/")
    raise SystemExit(1)

n_states_vals = sorted({r["n_states"] for r in records})
multi  = {r["n_states"]: r for r in records if r["mode"] == "multistage"}
single = {r["n_states"]: r for r in records if r["mode"] == "single"}

# ── 图1：n_ritz below E_target vs n_states ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
if multi:
    ns  = sorted(multi)
    ax.plot(ns, [multi[n]["n_below"] for n in ns],
            'b-o', lw=1.8, ms=7, label=f"multistage (5 stages)")
if single:
    ns  = sorted(single)
    ax.plot(ns, [single[n]["n_below"] for n in ns],
            'r--s', lw=1.8, ms=7, label="single (E_lower=-0.1)")
ax.set_xlabel("n_states (initial random states)")
ax.set_ylabel(f"Ritz values with E < {args.E_target}")
ax.set_title(f"Found eigenvalues E < {args.E_target} vs n_states")
ax.set_xscale("log", base=2)
ax.set_xticks(n_states_vals); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.5)

# ── 图2：total compute time vs n_states ──
ax = axes[1]
if multi:
    ns = sorted(multi)
    ax.plot(ns, [multi[n]["timing"]["total_s"] for n in ns],
            'b-o', lw=1.8, ms=7, label="multistage")
if single:
    ns = sorted(single)
    ax.plot(ns, [single[n]["timing"]["total_s"] for n in ns],
            'r--s', lw=1.8, ms=7, label="single")
ax.set_xlabel("n_states")
ax.set_ylabel("Total time (s)")
ax.set_title("Compute time vs n_states")
ax.set_xscale("log", base=2)
ax.set_xticks(n_states_vals); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.5)

fig.tight_layout()
fig.savefig(out_dir / "sweep_summary.png", dpi=150)
plt.close(fig)
print(f"\nSummary plot → {out_dir}/sweep_summary.png")

# ── 图3：Ritz energy spectrum for all runs ──
fig, axes = plt.subplots(len(n_states_vals), 1,
                         figsize=(13, 3 * len(n_states_vals)), sharex=True)
if len(n_states_vals) == 1:
    axes = [axes]

for ax, n in zip(axes, n_states_vals):
    row = 0.0
    if n in multi:
        e = multi[n]["energies"]
        e_show = e[e < args.E_target + 0.5]  # show a bit above threshold
        ax.scatter(e_show, np.full_like(e_show, row + 0.2),
                   marker='|', s=300, lw=2, color='steelblue',
                   label=f"multistage  n_found={multi[n]['n_below']}")
    if n in single:
        e = single[n]["energies"]
        e_show = e[e < args.E_target + 0.5]
        ax.scatter(e_show, np.full_like(e_show, row - 0.2),
                   marker='|', s=300, lw=2, color='tomato',
                   label=f"single      n_found={single[n]['n_below']}")
    ax.axvline(args.E_target, color='gray', ls='--', lw=1.2)
    ax.set_yticks([]); ax.set_ylabel(f"n={n}", rotation=0, labelpad=40)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, axis='x', ls='--', alpha=0.4)

axes[-1].set_xlabel("Energy")
fig.suptitle(f"Ritz energy spectra (E < {args.E_target + 0.5:.1f} shown, "
             f"dashed = E={args.E_target})", y=1.01)
fig.tight_layout()
fig.savefig(out_dir / "sweep_spectra.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Spectra plot  → {out_dir}/sweep_spectra.png")

# ── 打印汇总表 ──
print(f"\n{'─'*75}")
print(f"  {'tag':30s}  {'n_states':>8}  {'rank':>6}  "
      f"{'E<{}'.format(args.E_target):>8}  {'time(s)':>8}")
print(f"{'─'*75}")
for r in sorted(records, key=lambda x: (x["mode"], x["n_states"])):
    print(f"  {r['tag']:30s}  {r['n_states']:>8d}  {r['rank']:>6d}  "
          f"  {r['n_below']:>6d}  {r['timing']['total_s']:>8.1f}")
print(f"{'─'*75}")

# ── 保存汇总 JSON ──
summary = [
    {k: v for k, v in r.items() if k != "energies"}   # omit large arrays
    for r in records
]
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary JSON  → {out_dir}/summary.json")
