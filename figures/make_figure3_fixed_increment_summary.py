#!/usr/bin/env python3
"""Figure 3 — fixed-increment staged growth: mean improvement and connectivity.

Compares the two controlled ten-sample branches:
  A10: best-of-ten selection without consulting ligand B.
  B:   B-directed selection.

Outputs PNG (600 dpi), PDF and SVG to figures_ultimate.
"""
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
from figstyle import apply_style, soft_grid, panel_tag, tidy, BRANCH_COLORS, BRANCH_LABELS, PAIR_LABELS

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAIRS = ["x0434_x2193", "x0874_x1093"]
ARMS = [3, 4, 5, 6]
BRANCHES = ["A10", "directed"]
BRANCH_SOURCE = {"A10": "A10", "directed": "B"}


def ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def save_all(fig, prefix: Path):
    common = dict(bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, **common)
    fig.savefig(prefix.with_suffix(".pdf"), **common)
    fig.savefig(prefix.with_suffix(".svg"), **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=PROJECT_ROOT / "exp2/data/trajectory_summary.tsv")
    ap.add_argument("--out-prefix", type=Path, default=SCRIPT_DIR / "figure3_fixed_increment_summary")
    args = ap.parse_args()

    apply_style()
    d = pd.read_csv(args.summary, sep="\t")
    d["add_n"] = pd.to_numeric(d["add_n"], errors="coerce")
    d["tani_B_improvement"] = pd.to_numeric(d["tani_B_improvement"], errors="coerce")
    d["branch"] = d["branch"].astype(str).str.strip()
    d["last_connected"] = d["last_connected"].astype(str).str.strip().eq("True")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.86, wspace=0.08, hspace=0.15)

    for col, pair in enumerate(PAIRS):
        ax_top, ax_bottom = axes[0, col], axes[1, col]
        for branch in BRANCHES:
            src = BRANCH_SOURCE[branch]
            sub = d[(d["pair"] == pair) & (d["branch"] == src)]
            means, errors, conn = [], [], []
            for arm in ARMS:
                vals = sub.loc[sub["add_n"] == arm, "tani_B_improvement"].dropna().to_numpy()
                means.append(np.mean(vals) if len(vals) else np.nan)
                errors.append(ci95(vals))
                conn.append(sub.loc[sub["add_n"] == arm, "last_connected"].mean())
            ax_top.errorbar(
                ARMS, means, yerr=errors, marker="o", lw=2.0, markersize=6.5,
                capsize=3, color=BRANCH_COLORS[branch], markeredgecolor="white",
                markeredgewidth=0.7, label=BRANCH_LABELS[branch], zorder=3,
            )
            ax_bottom.plot(
                ARMS, conn, "o-", lw=2.0, markersize=6.5,
                color=BRANCH_COLORS[branch], markeredgecolor="white",
                markeredgewidth=0.7, zorder=3,
            )
        ax_top.set_title(PAIR_LABELS[pair])
        ax_top.set_xticks(ARMS)
        ax_bottom.set_xticks(ARMS)
        ax_bottom.set_xlabel("atoms added per stage")
        ax_bottom.set_ylim(-0.02, 1.02)
        soft_grid(ax_top, "y"); soft_grid(ax_bottom, "y")
        tidy(ax_top); tidy(ax_bottom)

    axes[0, 0].set_ylabel("mean Shape Tanimoto\nimprovement to B (95% CI)")
    axes[1, 0].set_ylabel("final connected fraction")
    panel_tag(axes[0, 0], "A", dx=-0.14, dy=1.10)
    panel_tag(axes[1, 0], "B", dx=-0.14, dy=1.10)

    handles = [
        plt.Line2D([0], [0], color=BRANCH_COLORS[b], marker="o", lw=2.0,
                   markersize=6.2, markeredgecolor="white", markeredgewidth=0.7,
                   label=BRANCH_LABELS[b])
        for b in BRANCHES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.965), fontsize=10.0)
    save_all(fig, args.out_prefix)
    plt.close(fig)
    print(f"Wrote {args.out_prefix}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
