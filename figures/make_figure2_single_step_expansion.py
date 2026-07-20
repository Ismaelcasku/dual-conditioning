#!/usr/bin/env python3
"""Generate Figure 2: single-step expansion sweep.

Every record in this figure comes from one independent expansion of the bare
seven-heavy-atom warhead. No generated scaffold was propagated to a later stage.

Default project layout:
  ./
      exp1/data/stage1_summary_by_grid.tsv
      exp1/data/stage1_audit_per_molecule.tsv
      figures_ultimate/

Run:
  python figures/make_figure2_single_step_expansion.py

The script writes PNG (600 dpi), PDF, and SVG files to figures_ultimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#B8BCC2",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#2B2B2B",
    "xtick.color": "#2B2B2B",
    "ytick.color": "#2B2B2B",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

HARD = "#B23A48"
MODERATE = "#2E5E8C"
FRAG = "#C9CCD1"
GRID = "#E6E8EB"
INK = "#2B2B2B"
MUTED = "#8A9099"
SIZE_CMAP = "cividis"

PAIR_COLORS = {
    "x0434_x2193": HARD,
    "x0874_x1093": MODERATE,
}
PAIR_LABELS = {
    "x0434_x2193": "x0434 → x2193  (hard)",
    "x0874_x1093": "x0874 → x1093  (moderate)",
}
PAIRS = ["x0434_x2193", "x0874_x1093"]
ADD_ORDER = ["2", "4", "6", "8", "auto"]
RESAMPLINGS = [5, 10, 20]


def soft_grid(ax, axis="both"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8, zorder=0)


def panel_tag(ax, letter, dx=-0.15, dy=1.07):
    ax.text(
        dx, dy, letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def tidy(ax):
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8BCC2")


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().eq("True")


def save_all(fig: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    common = dict(bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, **common)
    fig.savefig(prefix.with_suffix(".pdf"), **common)
    fig.savefig(prefix.with_suffix(".svg"), **common)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--per-molecule", type=Path, default=None)
    parser.add_argument("--out-prefix", type=Path, default=None)
    args = parser.parse_args()

    grid_path = args.grid or (
        args.project_root / "exp1/data/stage1_summary_by_grid.tsv"
    )
    per_molecule_path = args.per_molecule or (
        args.project_root / "exp1/data/stage1_audit_per_molecule.tsv"
    )
    out_prefix = args.out_prefix or (
        args.project_root / "figures_ultimate/figure2_single_step_expansion"
    )

    grid = pd.read_csv(grid_path, sep="\t")
    grid = grid[grid["pair_id"].isin(PAIRS)].copy()
    grid["add_n_nodes"] = grid["add_n_nodes"].astype(str).str.strip()
    for col in ["lambda_global", "resamplings", "heavy_connected_rate"]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce")

    records = pd.read_csv(per_molecule_path, sep="\t")
    records = records[records["pair_id"].isin(PAIRS)].copy()
    for col in ["anchor_heavy_atoms", "anchor_tanimoto_A", "anchor_tanimoto_B"]:
        records[col] = pd.to_numeric(records[col], errors="coerce")
    records["shape_ok"] = truthy(records["anchor_shape_ok"])
    records["connected"] = truthy(records["heavy_connected"])

    fig = plt.figure(figsize=(11.4, 9.4), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2,
        left=0.085,
        right=0.93,
        bottom=0.085,
        top=0.91,
        wspace=0.28,
        hspace=0.34,
        height_ratios=[0.90, 1.08],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # A — connected fraction vs requested atoms in one independent expansion.
    for pair in PAIRS:
        sub = grid[grid["pair_id"] == pair]
        values = [
            sub.loc[sub["add_n_nodes"] == add, "heavy_connected_rate"].mean()
            for add in ADD_ORDER
        ]
        ax_a.plot(
            range(len(ADD_ORDER)),
            values,
            "o-",
            color=PAIR_COLORS[pair],
            lw=2.0,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax_a.axvspan(3.5, 4.5, color=FRAG, alpha=0.35, zorder=0)
    ax_a.set_xticks(range(len(ADD_ORDER)))
    ax_a.set_xticklabels(["+2", "+4", "+6", "+8", "auto"])
    ax_a.set_xlabel("atoms requested in the single expansion")
    ax_a.set_ylabel("connected fraction")
    ax_a.set_ylim(0.0, 0.72)
    soft_grid(ax_a, "y")
    tidy(ax_a)
    panel_tag(ax_a, "A")

    # B — connected fraction vs resampling iterations.
    for pair in PAIRS:
        sub = grid[grid["pair_id"] == pair]
        values = [
            sub.loc[sub["resamplings"] == n, "heavy_connected_rate"].mean()
            for n in RESAMPLINGS
        ]
        ax_b.plot(
            RESAMPLINGS,
            values,
            "o-",
            color=PAIR_COLORS[pair],
            lw=2.0,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax_b.set_xticks(RESAMPLINGS)
    ax_b.set_xlabel("resampling iterations")
    ax_b.set_ylabel("connected fraction")
    ax_b.set_ylim(0.0, 0.40)
    soft_grid(ax_b, "y")
    tidy(ax_b)
    panel_tag(ax_b, "B")

    # C–D — connected, shape-auditable anchors from the one-step sweep.
    shape = records.loc[records["shape_ok"] & records["connected"]].dropna(
        subset=["anchor_tanimoto_A", "anchor_tanimoto_B", "anchor_heavy_atoms"]
    )
    norm = mpl.colors.Normalize(
        vmin=float(shape["anchor_heavy_atoms"].min()),
        vmax=float(shape["anchor_heavy_atoms"].max()),
    )
    xlim = (0.30, 0.90)
    ylim = (0.30, 0.90)
    scatter = None

    for letter, ax, pair in zip(["C", "D"], [ax_c, ax_d], PAIRS):
        sub = shape[shape["pair_id"] == pair]
        scatter = ax.scatter(
            sub["anchor_tanimoto_A"],
            sub["anchor_tanimoto_B"],
            c=sub["anchor_heavy_atoms"],
            cmap=SIZE_CMAP,
            norm=norm,
            s=30,
            alpha=0.80,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        ax.plot(
            [xlim[0], xlim[1]],
            [ylim[0], ylim[1]],
            ls="--",
            color=MUTED,
            lw=1.0,
            zorder=1,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("anchor Shape Tanimoto distance to A")
        if pair == PAIRS[0]:
            ax.set_ylabel("anchor Shape Tanimoto distance to B")
        else:
            ax.tick_params(labelleft=False)

        ax.set_title(PAIR_LABELS[pair], fontsize=10.5, fontweight="bold")

        n_closer_b = int(
            (sub["anchor_tanimoto_B"] < sub["anchor_tanimoto_A"]).sum()
        )
        ax.text(
            0.57,
            0.95,
            f"closer to B: {n_closer_b}/{len(sub)}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.8,
            color=MUTED,
        )
        ax.text(
            0.05, 0.90, "closer to A",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=8.3,
        )
        ax.text(
            0.69, 0.07, "closer to B",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=8.3,
        )
        soft_grid(ax)
        tidy(ax)
        panel_tag(ax, letter)

    cbar = fig.colorbar(scatter, ax=[ax_c, ax_d], fraction=0.035, pad=0.025)
    cbar.set_label("anchor heavy atoms")

    pair_handles = [
        plt.Line2D(
            [0], [0],
            color=PAIR_COLORS[pair],
            marker="o",
            lw=2.0,
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=PAIR_LABELS[pair],
        )
        for pair in PAIRS
    ]
    fig.legend(
        handles=pair_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.978),
        fontsize=10.0,
    )

    save_all(fig, out_prefix)
    plt.close(fig)
    print(f"Wrote {out_prefix}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
