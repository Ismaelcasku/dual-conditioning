#!/usr/bin/env python3
"""Figure 5 — fully connected staged-growth states in the A–B shape plane."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import pandas as pd
from figstyle import apply_style, soft_grid, panel_tag, tidy, BRANCH_LABELS, PAIR_LABELS, SIZE_CMAP, MUTED

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAIRS = ["x0434_x2193", "x0874_x1093"]
BRANCHES = ["A10", "directed"]
BRANCH_SOURCE = {"A10": "A10", "directed": "B"}


def truthy(s):
    return s.astype(str).str.strip().eq("True")


def save_all(fig, prefix):
    common = dict(bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, **common)
    fig.savefig(prefix.with_suffix(".pdf"), **common)
    fig.savefig(prefix.with_suffix(".svg"), **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=Path, default=PROJECT_ROOT / "exp2/data/all_stages_long_shapeAB.tsv")
    ap.add_argument("--topology", type=Path, default=PROJECT_ROOT / "exp2/data/all_stages_long.tsv")
    ap.add_argument("--out-prefix", type=Path, default=SCRIPT_DIR / "figure5_AB_plane")
    args = ap.parse_args()
    apply_style()

    shape = pd.read_csv(args.shape, sep="\t")
    topo = pd.read_csv(args.topology, sep="\t")
    for c in ["add_n", "stage", "anchor_heavy", "tani_A", "tani_B"]:
        shape[c] = pd.to_numeric(shape[c], errors="coerce")
    shape["branch"] = shape["branch"].astype(str).str.strip()
    topo["branch"] = topo["branch"].astype(str).str.strip()
    topo["connected"] = truthy(topo["connected"])
    keys = ["pair", "add_n", "branch", "seed", "rep", "stage"]
    d = shape.merge(topo[keys + ["connected"]], on=keys, how="left", validate="one_to_one")
    d = d[d["connected"].fillna(False)].dropna(subset=["tani_A", "tani_B", "anchor_heavy"])

    norm = mpl.colors.Normalize(vmin=float(d["anchor_heavy"].min()), vmax=float(d["anchor_heavy"].max()))
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.90, bottom=0.09, top=0.91, wspace=0.08, hspace=0.12)
    limits = (0.35, 0.95)
    scatter = None
    for row, branch in enumerate(BRANCHES):
        src = BRANCH_SOURCE[branch]
        for col, pair in enumerate(PAIRS):
            ax = axes[row, col]
            sub = d[(d["pair"] == pair) & (d["branch"] == src)]
            scatter = ax.scatter(sub["tani_A"], sub["tani_B"], c=sub["anchor_heavy"],
                                 cmap=SIZE_CMAP, norm=norm, s=26, alpha=0.78,
                                 edgecolors="white", linewidths=0.3, zorder=3)
            ax.plot(limits, limits, ls="--", color=MUTED, lw=1.0)
            ax.set_xlim(limits); ax.set_ylim(limits); ax.set_aspect("equal", adjustable="box")
            if row == 0: ax.set_title(PAIR_LABELS[pair])
            if row == 1: ax.set_xlabel("Shape Tanimoto distance to A")
            if col == 0: ax.set_ylabel(f"{BRANCH_LABELS[branch]}\n\nShape Tanimoto distance to B")
            ax.text(0.05, 0.93, "closer to A", transform=ax.transAxes, color=MUTED, fontsize=8.5)
            ax.text(0.67, 0.07, "closer to B", transform=ax.transAxes, color=MUTED, fontsize=8.5)
            n_b = int((sub["tani_B"] < sub["tani_A"]).sum())
            ax.text(0.57, 0.95, f"closer to B: {n_b}/{len(sub)}", transform=ax.transAxes,
                    va="top", fontsize=8.6, color=MUTED)
            soft_grid(ax); tidy(ax)
    panel_tag(axes[0, 0], "A", dx=-0.14, dy=1.10)
    panel_tag(axes[1, 0], "B", dx=-0.14, dy=1.10)
    cax = fig.add_axes([0.925, 0.20, 0.018, 0.60])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label("anchor heavy atoms")
    save_all(fig, args.out_prefix)
    plt.close(fig)
    print(f"Wrote {args.out_prefix}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
