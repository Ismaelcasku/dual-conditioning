#!/usr/bin/env python3
"""Figure 4 — trajectory-level Tanimoto and Protrude improvement distributions."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figstyle import apply_style, soft_grid, panel_tag, tidy, BRANCH_COLORS, BRANCH_LABELS, PAIR_LABELS, MUTED

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAIRS = ["x0434_x2193", "x0874_x1093"]
ARMS = [3, 4, 5, 6]
BRANCHES = ["A10", "directed"]
BRANCH_SOURCE = {"A10": "A10", "directed": "B"}


def trajectory_improvement(stages, metric):
    rows = []
    keys = ["pair", "add_n", "branch", "seed", "rep"]
    for key, g in stages.groupby(keys, dropna=False):
        vals = g.sort_values("stage")[metric].dropna()
        if vals.empty:
            continue
        rows.append({"pair": key[0], "add_n": int(key[1]), "branch": key[2],
                     "improvement": float(vals.iloc[0] - vals.iloc[-1])})
    return pd.DataFrame(rows)


def add_violin(ax, data, ylabel):
    ticks, labels = [], []
    x = 0.0
    for arm in ARMS:
        positions = []
        for branch in BRANCHES:
            src = BRANCH_SOURCE[branch]
            vals = data[(data["add_n"] == arm) & (data["branch"] == src)]["improvement"].dropna().to_numpy()
            if len(vals) == 0:
                x += 1.0
                continue
            parts = ax.violinplot([vals], positions=[x], widths=0.82,
                                  showmeans=False, showmedians=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(BRANCH_COLORS[branch]); body.set_edgecolor("none"); body.set_alpha(0.38)
            box = ax.boxplot([vals], positions=[x], widths=0.27, patch_artist=True,
                             showfliers=False,
                             medianprops={"color": "black", "linewidth": 1.15},
                             whiskerprops={"color": "black", "linewidth": 0.9},
                             capprops={"color": "black", "linewidth": 0.9},
                             boxprops={"edgecolor": "black", "linewidth": 0.9})
            box["boxes"][0].set_facecolor(BRANCH_COLORS[branch]); box["boxes"][0].set_alpha(0.88)
            positions.append(x); x += 1.0
        if positions:
            ticks.append(float(np.mean(positions))); labels.append(f"+{arm}")
        x += 0.55
    ax.axhline(0, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    soft_grid(ax, "y"); tidy(ax)


def save_all(fig, prefix):
    common = dict(bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, **common)
    fig.savefig(prefix.with_suffix(".pdf"), **common)
    fig.savefig(prefix.with_suffix(".svg"), **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=Path, default=PROJECT_ROOT / "exp2/data/all_stages_long_shapeAB.tsv")
    ap.add_argument("--out-prefix", type=Path, default=SCRIPT_DIR / "figure4_violin_distributions")
    args = ap.parse_args()
    apply_style()
    d = pd.read_csv(args.stages, sep="\t")
    for c in ["add_n", "stage", "tani_B", "prot_B"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["branch"] = d["branch"].astype(str).str.strip()
    tani = trajectory_improvement(d, "tani_B")
    prot = trajectory_improvement(d, "prot_B")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.86, wspace=0.12, hspace=0.12)
    for col, pair in enumerate(PAIRS):
        add_violin(axes[0, col], tani[tani["pair"] == pair],
                   "Shape Tanimoto\nimprovement to B" if col == 0 else "")
        add_violin(axes[1, col], prot[prot["pair"] == pair],
                   "Shape Protrude\nimprovement to B" if col == 0 else "")
        axes[0, col].set_title(PAIR_LABELS[pair])
        axes[1, col].set_xlabel("atoms added per stage")
    panel_tag(axes[0, 0], "A", dx=-0.14, dy=1.10)
    panel_tag(axes[1, 0], "B", dx=-0.14, dy=1.10)
    handles = [mpl.patches.Patch(facecolor=BRANCH_COLORS[b], alpha=0.75, label=BRANCH_LABELS[b]) for b in BRANCHES]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.965), fontsize=10.0)
    save_all(fig, args.out_prefix)
    plt.close(fig)
    print(f"Wrote {args.out_prefix}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
