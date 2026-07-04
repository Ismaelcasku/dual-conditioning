"""Generate manuscript Figure 4 from the selected minimization panel."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/source_data/candidate_shape_anchor.tsv"
OUTPUT_DIR = ROOT / "results/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------
preferred_fonts = [
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "DejaVu Sans",
]

available_fonts = {
    font.name for font in font_manager.fontManager.ttflist
}

font_family = next(
    (
        font
        for font in preferred_fonts
        if font in available_fonts
    ),
    "DejaVu Sans",
)

plt.rcParams.update(
    {
        "font.family": font_family,
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT, sep="\t")

required_columns = {
    "pair_id",
    "delta_shape_generated_to_minimized",
    "minimized_warhead_rmsd_A",
}

missing = required_columns.difference(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


pair_styles = {
    "x0434_x1093": {
        "label": "Compatible: x0434 → x1093",
        "color": "#0072B2",
        "marker": "o",
    },
    "x0874_x1093": {
        "label": "Moderate: x0874 → x1093",
        "color": "#E69F00",
        "marker": "s",
    },
    "x0434_x2193": {
        "label": "Difficult: x0434 → x2193",
        "color": "#D55E00",
        "marker": "^",
    },
}


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(6.5, 4.4),
)

fig.subplots_adjust(
    left=0.14,
    right=0.985,
    bottom=0.19,
    top=0.82,
)


# Operational boundaries
ax.axvline(
    0.0,
    linestyle=(0, (4, 3)),
    linewidth=0.85,
    color="#666666",
    zorder=1,
)

ax.axhline(
    0.2,
    linestyle=(0, (4, 3)),
    linewidth=0.85,
    color="#666666",
    zorder=1,
)


# All candidates shown equivalently
for pair_id, style in pair_styles.items():
    subset = df.loc[
        df["pair_id"] == pair_id
    ]

    ax.scatter(
        subset["delta_shape_generated_to_minimized"],
        subset["minimized_warhead_rmsd_A"],
        s=62,
        marker=style["marker"],
        facecolor=style["color"],
        edgecolor="#1A1A1A",
        linewidth=0.7,
        alpha=0.95,
        label=style["label"],
        zorder=3,
    )


# ---------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------
ax.set_xlabel(
    r"Change in Shape Tanimoto to B",
    labelpad=8,
)

ax.set_ylabel(
    r"Warhead RMSD to A after minimization ($\AA$)",
    labelpad=8,
)


# ---------------------------------------------------------------------
# Interpretation labels
# ---------------------------------------------------------------------
ax.text(
    0.101,
    0.204,
    r"0.2 $\AA$ anchor threshold",
    fontsize=7.6,
    color="#555555",
    ha="right",
    va="bottom",
)

ax.text(
    -0.150,
    0.010,
    "Similarity to B decreases",
    fontsize=7.5,
    color="#666666",
    ha="left",
    va="bottom",
)

ax.text(
    0.004,
    0.010,
    "Similarity to B increases",
    fontsize=7.5,
    color="#666666",
    ha="left",
    va="bottom",
)

ax.text(
    0.101,
    0.384,
    "Anchor displaced",
    fontsize=7.5,
    color="#666666",
    ha="right",
    va="top",
)

ax.text(
    0.101,
    0.187,
    "Anchor retained",
    fontsize=7.5,
    color="#666666",
    ha="right",
    va="top",
)


# ---------------------------------------------------------------------
# Axes styling
# ---------------------------------------------------------------------
ax.set_xlim(-0.155, 0.105)
ax.set_ylim(0.0, 0.395)

ax.xaxis.set_major_locator(
    MultipleLocator(0.05)
)

ax.yaxis.set_major_locator(
    MultipleLocator(0.05)
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="both",
    direction="out",
    length=3.5,
    width=0.75,
    pad=4,
)


# Legend outside the plotting area
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.025),
    ncol=3,
    frameon=False,
    handletextpad=0.5,
    columnspacing=1.4,
    borderaxespad=0.0,
)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
base = (
    OUTPUT_DIR
    / "figure4_relaxation_outcome_map"
)

fig.savefig(
    base.with_suffix(".png"),
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    base.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white",
    metadata={
        "Title": (
            "Shape similarity and anchor preservation "
            "after restrained minimization"
        ),
    },
)

fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("FIGURE4_RELAXATION_OUTCOME_MAP_STATUS=OK")
print(f"Font: {font_family}")
print(base.with_suffix(".png"))
print(base.with_suffix(".pdf"))
print(base.with_suffix(".svg"))
