"""
collect_scaling_ngrid_QD_three_methods.py
收集三种方法（JDQMR / PRIMME_GD+Jacobi / CG+Folded）最近一次 scaling JSON，
汇总并画对比图。

默认会自动读取：
- scaling_results/scaling_ngrid_QD_None_*.json               (JDQMR)
- scaling_results/scaling_ngrid_QD_PRIMME_GD_Jacobi_*.json   (PRIMME_GD+Jacobi)
- scaling_results/scaling_ngrid_QD_CG_FOLDED_*.json          (CG+Folded)

可用 --jdqmr-json / --gd-json / --cg-json 手动指定。
"""

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("scaling_results")
OUT_DIR.mkdir(exist_ok=True)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

TARGETS = [-0.17, -0.20, -0.22, -0.25]
COLORS = {
    "JDQMR": "steelblue",
    "PRIMME_GD+Jacobi": "darkorange",
    "CG+Folded": "forestgreen",
}
MARKERS = {
    "JDQMR": "o",
    "PRIMME_GD+Jacobi": "s",
    "CG+Folded": "^",
}


def latest_file(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def load_rows(path: str, method_label: str):
    with open(path) as f:
        data = json.load(f)
    rows = data.get("results", [])

    normalized = []
    for row in rows:
        if not row.get("success", False):
            continue
        n_h = row.get("N_H", None)
        if n_h is None:
            n_h = row.get("N_H_equiv", None)
        normalized.append({
            "method": method_label,
            "target": row["target"],
            "N_grid": row["N_grid"],
            "T_wall": row.get("T_wall", None),
            "N_H": n_h,
            "N": row.get("N", None),
            "radius": row.get("radius", None),
        })
    return normalized, data


def pick_best(rows):
    """同一 (method,target,N_grid) 可能重复，保留 T_wall 最小的一条。"""
    best = {}
    for r in rows:
        key = (r["method"], r["target"], r["N_grid"])
        if key not in best:
            best[key] = r
        else:
            t_old = best[key]["T_wall"]
            t_new = r["T_wall"]
            if t_new is not None and (t_old is None or t_new < t_old):
                best[key] = r
    return list(best.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jdqmr-json", type=str, default=None)
    parser.add_argument("--gd-json", type=str, default=None)
    parser.add_argument("--cg-json", type=str, default=None)
    args = parser.parse_args()

    jdqmr_json = args.jdqmr_json or latest_file("scaling_results/scaling_ngrid_QD_None_*.json")
    gd_json    = args.gd_json    or latest_file("scaling_results/scaling_ngrid_QD_PRIMME_GD_Jacobi_*.json")
    cg_json    = args.cg_json    or latest_file("scaling_results/scaling_ngrid_QD_CG_FOLDED_*.json")

    missing = [
        ("JDQMR", jdqmr_json),
        ("PRIMME_GD+Jacobi", gd_json),
        ("CG+Folded", cg_json),
    ]
    for name, p in missing:
        if p is None:
            raise FileNotFoundError(f"未找到 {name} 的 JSON，请手动用 --{name} 指定")

    print("使用输入文件：")
    print(f"  JDQMR            : {jdqmr_json}")
    print(f"  PRIMME_GD+Jacobi : {gd_json}")
    print(f"  CG+Folded        : {cg_json}")

    rows_jdqmr, data_jdqmr = load_rows(jdqmr_json, "JDQMR")
    rows_gd, data_gd       = load_rows(gd_json, "PRIMME_GD+Jacobi")
    rows_cg, data_cg       = load_rows(cg_json, "CG+Folded")

    all_rows = pick_best(rows_jdqmr + rows_gd + rows_cg)

    summary = {
        "script": "collect_scaling_ngrid_QD_three_methods.py",
        "datetime": TS,
        "inputs": {
            "JDQMR": jdqmr_json,
            "PRIMME_GD+Jacobi": gd_json,
            "CG+Folded": cg_json,
        },
        "results": all_rows,
    }
    out_json = OUT_DIR / f"scaling_ngrid_QD_three_methods_{TS}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总 JSON: {out_json}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, ylabel in zip(
        axes,
        ["T_wall", "N_H"],
        ["Wall time T (s)", "Matvec count N_H (CG 用等效值)"]
    ):
        for target in TARGETS:
            for method in ["JDQMR", "PRIMME_GD+Jacobi", "CG+Folded"]:
                rows = [r for r in all_rows
                        if r["target"] == target and r["method"] == method and r[metric] is not None]
                if not rows:
                    continue
                rows = sorted(rows, key=lambda rr: rr["N_grid"])
                x_vals = np.array([r["N_grid"] for r in rows], float)
                y_vals = np.array([r[metric] for r in rows], float)

                label = f"{method}, target={target}"
                ax.loglog(
                    x_vals, y_vals,
                    marker=MARKERS[method], linestyle="-",
                    color=COLORS[method], alpha=0.8,
                    label=label
                )

        ax.set_xlabel("N_grid = N³", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, which="both", alpha=0.35)

    axes[0].set_title("Three methods: T_wall vs N_grid", fontsize=11)
    axes[1].set_title("Three methods: N_H vs N_grid", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    fig.legend(list(uniq.values()), list(uniq.keys()),
               loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    fig.suptitle("QD scaling comparison: JDQMR vs PRIMME_GD+Jacobi vs CG+Folded", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    out_png = OUT_DIR / f"scaling_ngrid_QD_three_methods_{TS}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图: {out_png}")


if __name__ == "__main__":
    main()
