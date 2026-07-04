"""Generate the pair-compatibility summary figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results/source_data/pair_reference_shape.tsv"
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT, sep="\t")

required = {
    "pair_id",
    "A_B_shape_tanimoto_similarity",
    "A_to_B_shape_protrude_distance",
}

missing = required.difference(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


labels = {
    "x0434_x1093": "Compatible\nx0434 → x1093",
    "x0874_x1093": "Moderate\nx0874 → x1093",
    "x0434_x2193": "Difficult\nx0434 → x2193",
}

order = [
    "x0434_x1093",
    "x0874_x1093",
    "x0434_x2193",
]

colors = {
    "x0434_x1093": "#0072B2",
    "x0874_x1093": "#E69F00",
    "x0434_x2193": "#D55E00",
}

df = (
    df.set_index("pair_id")
    .loc[order]
    .reset_index()
)

x_positions = range(len(df))


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.1, 3.35),
)

fig.subplots_adjust(
    left=0.09,
    right=0.985,
    bottom=0.25,
    top=0.91,
    wspace=0.30,
)


# ---------------------------------------------------------------------
# Panel A: Shape Tanimoto
# ---------------------------------------------------------------------
ax = axes[0]

values = df["A_B_shape_tanimoto_similarity"]

bars = ax.bar(
    x_positions,
    values,
    width=0.58,
    color=[colors[pair] for pair in df["pair_id"]],
    edgecolor="#1A1A1A",
    linewidth=0.7,
)

ax.set_ylabel("In-frame Shape Tanimoto A–B")
ax.set_xticks(
    list(x_positions),
    [labels[pair] for pair in df["pair_id"]],
)

ax.set_ylim(0.0, 0.62)
ax.yaxis.set_major_locator(MultipleLocator(0.1))

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.018,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.text(
    -0.18,
    1.04,
    "A",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    ha="left",
    va="top",
)

ax.text(
    0.98,
    0.96,
    "Higher = more similar",
    transform=ax.transAxes,
    fontsize=7.5,
    color="#666666",
    ha="right",
    va="top",
)


# ---------------------------------------------------------------------
# Panel B: Shape Protrude
# ---------------------------------------------------------------------
ax = axes[1]

values = df["A_to_B_shape_protrude_distance"]

bars = ax.bar(
    x_positions,
    values,
    width=0.58,
    color=[colors[pair] for pair in df["pair_id"]],
    edgecolor="#1A1A1A",
    linewidth=0.7,
)

ax.set_ylabel("In-frame Shape Protrude A→B")
ax.set_xticks(
    list(x_positions),
    [labels[pair] for pair in df["pair_id"]],
)

ax.set_ylim(0.0, 0.86)
ax.yaxis.set_major_locator(MultipleLocator(0.1))

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.022,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.text(
    -0.18,
    1.04,
    "B",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    ha="left",
    va="top",
)

ax.text(
    0.98,
    0.96,
    "Lower = more compatible",
    transform=ax.transAxes,
    fontsize=7.5,
    color="#666666",
    ha="right",
    va="top",
)


# ---------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.5,
        width=0.75,
        pad=4,
    )

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linewidth=0.45,
        color="#D9D9D9",
        alpha=0.75,
    )


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
base = OUTPUT_DIR / "supp_pair_compatibility"

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
        "Title": "Pairwise geometric compatibility",
    },
)

fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("PAIR_COMPATIBILITY_FIGURE_STATUS=OK")
print(f"Font: {font_family}")
print(base.with_suffix(".png"))
print(base.with_suffix(".pdf"))
print(base.with_suffix(".svg"))
