"""Generate manuscript Figure 3 from the per-molecule fragmentation audit."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    MultipleLocator,
    PercentFormatter,
)
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

PAIR_EVALUATION_PATH = Path(
    os.environ.get(
        "DC_FIG3_PAIR_EVAL_TSV",
        ROOT
        / "results/source_data"
        / "fragment_audit_per_evaluation_directional.tsv",
    )
)

UNIQUE_STRUCTURE_PATH = Path(
    os.environ.get(
        "DC_FIG3_UNIQUE_TSV",
        ROOT
        / "results/source_data"
        / "fragment_audit_unique_structures.tsv",
    )
)

FULL_RENDER = (
    ROOT
    / "results/source_renderings"
    / "figure3_full_record_vs_B.png"
)

PARENT_RENDER = (
    ROOT
    / "results/source_renderings"
    / "figure3_parent_component_vs_B.png"
)

OUTPUT_DIR = (
    ROOT
    / "results/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
    font.name
    for font in font_manager.fontManager.ttflist
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
        "font.size": 8.0,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.8,
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ---------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------
COLOR_PARENT = "#E69F00"
COLOR_SECONDARY = "#777777"
COLOR_B = "#18B8C9"

COLOR_TEXT = "#202020"
COLOR_SECONDARY_TEXT = "#606060"
COLOR_GRID = "#E0E0E0"
COLOR_ZERO = "#777777"

pair_order = [
    "x0434_x1093",
    "x0874_x1093",
    "x0434_x2193",
]

lambda_order = [
    0.0,
    20.0,
    50.0,
    100.0,
    200.0,
]

pair_styles = {
    "x0434_x1093": {
        "label": "Compatible",
        "color": "#0072B2",
        "marker": "o",
        "offset": -0.10,
    },
    "x0874_x1093": {
        "label": "Moderate",
        "color": "#E69F00",
        "marker": "s",
        "offset": 0.00,
    },
    "x0434_x2193": {
        "label": "Difficult",
        "color": "#D55E00",
        "marker": "^",
        "offset": 0.10,
    },
}

class_order = [
    "potential_missing_bond",
    "bond_distance_valence_limited",
    "close_nonbonded",
    "geometrically_separated",
]

class_labels = {
    "potential_missing_bond": (
        "Potential missing-bond"
    ),
    "bond_distance_valence_limited": (
        "Valence-limited"
    ),
    "close_nonbonded": (
        "Close nonbonded"
    ),
    "geometrically_separated": (
        "Geometrically separated"
    ),
}

class_colors = {
    "potential_missing_bond": "#D55E00",
    "bond_distance_valence_limited": "#CC79A7",
    "close_nonbonded": "#999999",
    "geometrically_separated": "#0072B2",
}


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
# Shape-dependent analyses use pair-specific evaluations because they
# depend on the selected global reference B.
#
# Connectivity and interfragment classification use unique generated
# structures because these properties are independent of B. Shared
# unguided baselines are therefore counted only once.

if not PAIR_EVALUATION_PATH.is_file():
    raise FileNotFoundError(
        "Pair-specific audit table not found: "
        f"{PAIR_EVALUATION_PATH}\n"
        "Set DC_FIG3_PAIR_EVAL_TSV to its location."
    )

if not UNIQUE_STRUCTURE_PATH.is_file():
    raise FileNotFoundError(
        "Unique-structure audit table not found: "
        f"{UNIQUE_STRUCTURE_PATH}\n"
        "Set DC_FIG3_UNIQUE_TSV to its location."
    )


df = pd.read_csv(
    PAIR_EVALUATION_PATH,
    sep="\t",
)

unique_df = pd.read_csv(
    UNIQUE_STRUCTURE_PATH,
    sep="\t",
)


def parse_boolean(series):
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


# Harmonize the corrected pair-specific audit schema with the names
# used by the established plotting implementation.
numeric_columns = [
    "seed",
    "lambda_global",
    "n_heavy_components",
    "parent_heavy_fraction",
    "full_tanimoto_distance_to_b",
    "parent_tanimoto_distance_to_b",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


for column in [
    "read_ok",
    "sanitize_ok",
    "full_shape_ok",
    "parent_shape_ok",
]:
    df[column] = parse_boolean(
        df[column]
    )


df["audit_ok"] = (
    df["read_ok"]
    & df["sanitize_ok"]
)

df["n_heavy_fragments"] = (
    df["n_heavy_components"]
)

df["full_tanimoto_B"] = (
    df["full_tanimoto_distance_to_b"]
)

df["parent_tanimoto_B"] = (
    df["parent_tanimoto_distance_to_b"]
)

df["full_similarity_B"] = (
    1.0 - df["full_tanimoto_B"]
)

df["parent_similarity_B"] = (
    1.0 - df["parent_tanimoto_B"]
)

df["shape_inflation_B"] = (
    df["full_similarity_B"]
    - df["parent_similarity_B"]
)


# Pair-specific sanitizable evaluation records.
auditable = df.loc[
    (df["audit_ok"] == True)
    & df["n_heavy_fragments"].notna()
].copy()

fragmented = auditable.loc[
    auditable["n_heavy_fragments"] > 1
].copy()

shape_valid = fragmented.loc[
    (fragmented["full_shape_ok"] == True)
    & (fragmented["parent_shape_ok"] == True)
    & fragmented[
        "parent_heavy_fraction"
    ].notna()
    & fragmented[
        "shape_inflation_B"
    ].notna()
].copy()


# Unique generated structures for connectivity-independent analyses.
for column in [
    "seed",
    "lambda_global",
    "n_heavy_components",
    "parent_heavy_fraction",
]:
    unique_df[column] = pd.to_numeric(
        unique_df[column],
        errors="coerce",
    )


for column in [
    "read_ok",
    "sanitize_ok",
    "connected",
]:
    unique_df[column] = parse_boolean(
        unique_df[column]
    )


unique_df["n_heavy_fragments"] = (
    unique_df["n_heavy_components"]
)

unique_auditable = unique_df.loc[
    (unique_df["read_ok"] == True)
    & (unique_df["sanitize_ok"] == True)
    & unique_df["n_heavy_fragments"].notna()
].copy()

unique_fragmented = unique_auditable.loc[
    unique_auditable["n_heavy_fragments"] > 1
].copy()

unique_non_sanitizable = unique_df.loc[
    unique_df["sanitize_ok"] == False
].copy()


expected_totals = {
    "pair_specific_evaluations": 691,
    "sanitizable_pair_specific_evaluations": 690,
    "fragmented_pair_specific_evaluations": 562,
    "unique_generated_structures": 649,
    "unique_sanitizable_structures": 648,
    "unique_fragmented_sanitizable": 555,
    "unique_non_sanitizable": 1,
}

observed_totals = {
    "pair_specific_evaluations": len(df),
    "sanitizable_pair_specific_evaluations": len(
        auditable
    ),
    "fragmented_pair_specific_evaluations": len(
        fragmented
    ),
    "unique_generated_structures": len(
        unique_df
    ),
    "unique_sanitizable_structures": len(
        unique_auditable
    ),
    "unique_fragmented_sanitizable": len(
        unique_fragmented
    ),
    "unique_non_sanitizable": len(
        unique_non_sanitizable
    ),
}

if observed_totals != expected_totals:
    raise RuntimeError(
        "Unexpected corrected-audit totals.\n"
        f"Expected: {expected_totals}\n"
        f"Observed: {observed_totals}"
    )


expected_class_counts = {
    "potential_missing_bond": 5,
    "bond_distance_valence_limited": 299,
    "close_nonbonded": 176,
    "geometrically_separated": 75,
}

observed_class_counts = (
    unique_fragmented[
        "interfragment_class"
    ]
    .value_counts()
    .reindex(
        class_order,
        fill_value=0,
    )
    .astype(int)
    .to_dict()
)

if observed_class_counts != expected_class_counts:
    raise RuntimeError(
        "Unexpected fragmentation-class counts.\n"
        f"Expected: {expected_class_counts}\n"
        f"Observed: {observed_class_counts}"
    )


# Panel C remains pair-specific because it displays separate A-to-B
# trajectories.
seed_fragment_summary = (
    auditable
    .groupby(
        [
            "pair_id",
            "seed",
            "lambda_global",
        ],
        as_index=False,
    )
    .agg(
        mean_heavy_fragments=(
            "n_heavy_fragments",
            "mean",
        )
    )
)

fragment_summary = (
    seed_fragment_summary
    .groupby(
        [
            "pair_id",
            "lambda_global",
        ],
        as_index=False,
    )
    .agg(
        mean_heavy_fragments=(
            "mean_heavy_fragments",
            "mean",
        ),
        sd_heavy_fragments=(
            "mean_heavy_fragments",
            "std",
        ),
    )
)


# Panel E uses unique generated structures.
class_counts = pd.crosstab(
    unique_fragmented[
        "lambda_global"
    ],
    unique_fragmented[
        "interfragment_class"
    ],
)

class_counts = (
    class_counts
    .reindex(
        index=lambda_order,
        fill_value=0,
    )
    .reindex(
        columns=class_order,
        fill_value=0,
    )
)

class_proportions = class_counts.div(
    class_counts.sum(axis=1),
    axis=0,
)


# Panel D remains pair-specific because shape inflation depends on B.
shape_valid["parent_fraction_bin"] = pd.qcut(
    shape_valid["parent_heavy_fraction"],
    q=7,
    duplicates="drop",
)

binned_trend = (
    shape_valid
    .groupby(
        "parent_fraction_bin",
        observed=True,
    )
    .agg(
        x_median=(
            "parent_heavy_fraction",
            "median",
        ),
        y_median=(
            "shape_inflation_B",
            "median",
        ),
        y_q25=(
            "shape_inflation_B",
            lambda x: x.quantile(0.25),
        ),
        y_q75=(
            "shape_inflation_B",
            lambda x: x.quantile(0.75),
        ),
    )
    .reset_index(drop=True)
)


def spearman_without_scipy(x, y):
    valid = x.notna() & y.notna()

    return (
        x.loc[valid]
        .rank()
        .corr(
            y.loc[valid].rank()
        )
    )


rho = spearman_without_scipy(
    shape_valid["parent_heavy_fraction"],
    shape_valid["shape_inflation_B"],
)


print(
    "FIGURE3_PAIR_EVALUATIONS="
    f"{len(df)}"
)
print(
    "FIGURE3_PAIR_FRAGMENTED_SANITIZABLE="
    f"{len(fragmented)}"
)
print(
    "FIGURE3_UNIQUE_STRUCTURES="
    f"{len(unique_df)}"
)
print(
    "FIGURE3_UNIQUE_FRAGMENTED_SANITIZABLE="
    f"{len(unique_fragmented)}"
)
print(
    "FIGURE3_FRAGMENTATION_CLASSES="
    f"{observed_class_counts}"
)

# ---------------------------------------------------------------------
# Structural image preparation
# ---------------------------------------------------------------------
full_image = Image.open(
    FULL_RENDER
).convert("RGBA")

parent_image = Image.open(
    PARENT_RENDER
).convert("RGBA")


def common_crop(
    image_a,
    image_b,
    padding=80,
):
    array_a = np.asarray(image_a)
    array_b = np.asarray(image_b)

    mask = (
        (array_a[:, :, 3] > 5)
        | (array_b[:, :, 3] > 5)
    )

    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        return image_a, image_b

    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)

    x_min = max(
        0,
        x_min - padding,
    )

    y_min = max(
        0,
        y_min - padding,
    )

    x_max = min(
        image_a.width,
        x_max + padding,
    )

    y_max = min(
        image_a.height,
        y_max + padding,
    )

    box = (
        x_min,
        y_min,
        x_max,
        y_max,
    )

    return (
        image_a.crop(box),
        image_b.crop(box),
    )


full_image, parent_image = common_crop(
    full_image,
    parent_image,
)


def add_top_padding(
    image,
    fraction=0.10,
):
    top_padding = int(
        round(image.height * fraction)
    )

    canvas = Image.new(
        "RGBA",
        (
            image.width,
            image.height + top_padding,
        ),
        (
            255,
            255,
            255,
            0,
        ),
    )

    canvas.paste(
        image,
        (
            0,
            top_padding,
        ),
        image,
    )

    return canvas


full_image = add_top_padding(
    full_image,
    fraction=0.10,
)

parent_image = add_top_padding(
    parent_image,
    fraction=0.10,
)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def panel_label(
    ax,
    label,
    x=-0.055,
    y=1.055,
):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=COLOR_TEXT,
        ha="left",
        va="top",
    )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.0,
        width=0.7,
        pad=3.5,
    )

    ax.grid(
        axis="y",
        color=COLOR_GRID,
        linewidth=0.45,
        zorder=0,
    )

    ax.set_axisbelow(True)


x_lookup = {
    value: index
    for index, value in enumerate(
        lambda_order
    )
}

x_positions = np.arange(
    len(lambda_order),
    dtype=float,
)

seed_values = sorted(
    seed_fragment_summary[
        "seed"
    ].dropna().unique()
)

seed_jitter = {
    seed: jitter
    for seed, jitter in zip(
        seed_values,
        np.linspace(
            -0.025,
            0.025,
            len(seed_values),
        ),
    )
}


# ---------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------
fig = plt.figure(
    figsize=(7.45, 9.05),
)

grid = fig.add_gridspec(
    nrows=4,
    ncols=2,
    height_ratios=[1.28, 0.13, 1.10, 1.08],
    left=0.080,
    right=0.980,
    bottom=0.105,
    top=0.975,
    wspace=0.31,
    hspace=0.52,
)

ax_a = fig.add_subplot(
    grid[0, 0]
)

ax_b = fig.add_subplot(
    grid[0, 1]
)

ax_structural_legend = fig.add_subplot(
    grid[1, :]
)

ax_structural_legend.set_axis_off()

ax_c = fig.add_subplot(
    grid[2, 0]
)

# Variable ax_e corresponds to displayed panel D.
ax_e = fig.add_subplot(
    grid[2, 1]
)

# Variable ax_d corresponds to displayed panel E.
ax_d = fig.add_subplot(
    grid[3, :]
)


# ---------------------------------------------------------------------
# A. Complete fragmented record
# ---------------------------------------------------------------------
ax_a.imshow(
    full_image,
    interpolation="lanczos",
)

ax_a.set_axis_off()
panel_label(
    ax_a,
    "A",
    x=-0.085,
    y=1.075,
)

ax_a.text(
    0.075,
    0.975,
    "Full generated record",
    transform=ax_a.transAxes,
    fontsize=8.8,
    fontweight="bold",
    color=COLOR_TEXT,
    ha="left",
    va="top",
)

ax_a.text(
    0.075,
    0.875,
    "x0434 → x2193, λ = 50; four components",
    transform=ax_a.transAxes,
    fontsize=7.2,
    color=COLOR_SECONDARY_TEXT,
    ha="left",
    va="top",
)

ax_a.text(
    0.97,
    0.055,
    "Similarity to B = 0.360",
    transform=ax_a.transAxes,
    fontsize=7.2,
    fontweight="bold",
    color=COLOR_TEXT,
    ha="right",
    va="bottom",
)


# ---------------------------------------------------------------------
# B. Largest connected component
# ---------------------------------------------------------------------
ax_b.imshow(
    parent_image,
    interpolation="lanczos",
)

ax_b.set_axis_off()
panel_label(
    ax_b,
    "B",
    x=-0.085,
    y=1.075,
)

ax_b.text(
    0.075,
    0.975,
    "Largest connected component",
    transform=ax_b.transAxes,
    fontsize=8.8,
    fontweight="bold",
    color=COLOR_TEXT,
    ha="left",
    va="top",
)

ax_b.text(
    0.075,
    0.875,
    "Seven of fifteen heavy atoms retained",
    transform=ax_b.transAxes,
    fontsize=7.2,
    color=COLOR_SECONDARY_TEXT,
    ha="left",
    va="top",
)

ax_b.text(
    0.97,
    0.055,
    "Similarity to B = 0.146",
    transform=ax_b.transAxes,
    fontsize=7.2,
    fontweight="bold",
    color=COLOR_TEXT,
    ha="right",
    va="bottom",
)


# Structural legend
structural_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.4,
        markerfacecolor=COLOR_PARENT,
        markeredgecolor="#333333",
        markeredgewidth=0.4,
        label="Largest component",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.4,
        markerfacecolor=COLOR_SECONDARY,
        markeredgecolor="#333333",
        markeredgewidth=0.4,
        label="Secondary components",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.4,
        markerfacecolor=COLOR_B,
        markeredgecolor="#333333",
        markeredgewidth=0.4,
        label="Reference B",
    ),
]

ax_structural_legend.legend(
    handles=structural_handles,
    loc="center",
    bbox_to_anchor=(0.50, 0.72),
    ncol=3,
    frameon=False,
    fontsize=7.2,
    handletextpad=0.35,
    columnspacing=1.45,
    borderaxespad=0.0,
)


# ---------------------------------------------------------------------
# C. Heavy-fragment count
# ---------------------------------------------------------------------
panel_label(
    ax_c,
    "C",
    x=-0.145,
    y=1.095,
)

ax_c.set_title(
    "Heavy-fragment count",
    fontweight="bold",
    pad=11,
)

for pair_id in pair_order:
    style = pair_styles[pair_id]

    seed_data = seed_fragment_summary.loc[
        seed_fragment_summary[
            "pair_id"
        ] == pair_id
    ].copy()

    summary_data = fragment_summary.loc[
        fragment_summary[
            "pair_id"
        ] == pair_id
    ].copy()

    seed_x = np.asarray(
        [
            x_lookup[value]
            + style["offset"]
            + seed_jitter[seed]
            for value, seed in zip(
                seed_data["lambda_global"],
                seed_data["seed"],
            )
        ],
        dtype=float,
    )

    ax_c.scatter(
        seed_x,
        seed_data[
            "mean_heavy_fragments"
        ],
        s=12,
        marker=style["marker"],
        facecolor=style["color"],
        edgecolor="none",
        alpha=0.28,
        zorder=2,
    )

    mean_x = np.asarray(
        [
            x_lookup[value]
            + style["offset"]
            for value in summary_data[
                "lambda_global"
            ]
        ],
        dtype=float,
    )

    ax_c.errorbar(
        mean_x,
        summary_data[
            "mean_heavy_fragments"
        ],
        yerr=summary_data[
            "sd_heavy_fragments"
        ].fillna(0.0),
        color=style["color"],
        marker=style["marker"],
        markersize=4.5,
        markeredgecolor="#222222",
        markeredgewidth=0.4,
        linewidth=1.25,
        elinewidth=0.75,
        capsize=2.0,
        label=style["label"],
        zorder=4,
    )

ax_c.set_xticks(
    x_positions,
    ["0", "20", "50", "100", "200"],
)

ax_c.set_xlabel(
    "Guidance strength λ"
)

ax_c.set_ylabel(
    "Mean heavy fragments"
)

ax_c.set_ylim(
    0.5,
    max(
        9.5,
        fragment_summary[
            "mean_heavy_fragments"
        ].max() + 1.5,
    ),
)

ax_c.yaxis.set_major_locator(
    MultipleLocator(2)
)

style_axis(ax_c)

# Controls the physical height-to-width ratio.
ax_c.set_box_aspect(0.72)

ax_c.legend(
    loc="upper left",
    frameon=False,
    borderaxespad=0.2,
    handlelength=1.1,
    handletextpad=0.35,
)


# ---------------------------------------------------------------------
# D. Interfragment classification
# ---------------------------------------------------------------------
panel_label(
    ax_d,
    "E",
    x=-0.075,
    y=1.105,
)

ax_d.set_title(
    "Interfragment geometry",
    fontweight="bold",
    pad=12,
)

bottom = np.zeros(
    len(lambda_order),
    dtype=float,
)

for class_name in class_order:
    values = class_proportions[
        class_name
    ].to_numpy(
        dtype=float
    )

    ax_d.bar(
        x_positions,
        values,
        width=0.50,
        bottom=bottom,
        color=class_colors[
            class_name
        ],
        edgecolor="white",
        linewidth=0.45,
        label=class_labels[
            class_name
        ],
        zorder=3,
    )

    bottom += values


for x, count in zip(
    x_positions,
    class_counts.sum(axis=1),
):
    ax_d.text(
        x,
        1.025,
        f"n={int(count)}",
        fontsize=6.4,
        color=COLOR_SECONDARY_TEXT,
        ha="center",
        va="bottom",
    )


ax_d.set_xticks(
    x_positions,
    ["0", "20", "50", "100", "200"],
)

ax_d.set_xlim(
    -0.72,
    4.72,
)

ax_d.set_xlim(
    -0.65,
    4.65,
)

ax_d.set_xlabel(
    "Guidance strength λ"
)

ax_d.set_ylabel(
    "Unique fragmented structures"
)

ax_d.set_ylim(
    0.0,
    1.09,
)

ax_d.yaxis.set_major_formatter(
    PercentFormatter(
        xmax=1.0,
        decimals=0,
    )
)

style_axis(ax_d)

# Wide panel with sufficient vertical height.
ax_d.set_box_aspect(0.34)

ax_d.legend(
    loc="upper center",
    bbox_to_anchor=(0.50, -0.24),
    ncol=4,
    frameon=False,
    fontsize=6.4,
    columnspacing=1.35,
    handlelength=1.35,
    handletextpad=0.40,
)


# ---------------------------------------------------------------------
# E. Fragmentation and metric inflation
# ---------------------------------------------------------------------
panel_label(
    ax_e,
    "D",
    x=-0.145,
    y=1.095,
)

ax_e.set_title(
    "Fragmentation-driven inflation",
    fontweight="bold",
    pad=11,
)

for pair_id in pair_order:
    style = pair_styles[pair_id]

    group = shape_valid.loc[
        shape_valid["pair_id"] == pair_id
    ]

    ax_e.scatter(
        group["parent_heavy_fraction"],
        group["shape_inflation_B"],
        s=11,
        marker=style["marker"],
        facecolor=style["color"],
        edgecolor="none",
        alpha=0.19,
        rasterized=True,
        zorder=2,
    )


x_binned = binned_trend[
    "x_median"
].to_numpy(
    dtype=float
)

y_binned = binned_trend[
    "y_median"
].to_numpy(
    dtype=float
)

lower_error = (
    y_binned
    - binned_trend[
        "y_q25"
    ].to_numpy(
        dtype=float
    )
)

upper_error = (
    binned_trend[
        "y_q75"
    ].to_numpy(
        dtype=float
    )
    - y_binned
)

ax_e.errorbar(
    x_binned,
    y_binned,
    yerr=np.vstack(
        [
            lower_error,
            upper_error,
        ]
    ),
    color="#111111",
    marker="o",
    markersize=3.8,
    markerfacecolor="white",
    markeredgecolor="#111111",
    markeredgewidth=0.7,
    linewidth=1.2,
    elinewidth=0.7,
    capsize=1.8,
    zorder=5,
)

ax_e.axhline(
    0.0,
    color=COLOR_ZERO,
    linewidth=0.75,
    linestyle=(0, (4, 3)),
    zorder=1,
)

ax_e.set_xlabel(
    "Parent heavy-atom fraction"
)

ax_e.set_ylabel(
    "Shape-similarity inflation to B"
)

ax_e.set_xlim(
    0.20,
    1.02,
)

ax_e.set_ylim(
    -0.08,
    0.27,
)

ax_e.xaxis.set_major_locator(
    MultipleLocator(0.2)
)

ax_e.yaxis.set_major_locator(
    MultipleLocator(0.05)
)

style_axis(ax_e)

# Same physical proportion as panel C.
ax_e.set_box_aspect(0.72)

ax_e.text(
    0.04,
    0.95,
    (
        f"Spearman ρ = {rho:.2f}\n"
        f"n = {len(shape_valid)}"
    ),
    transform=ax_e.transAxes,
    fontsize=6.8,
    color=COLOR_TEXT,
    ha="left",
    va="top",
)


# ---------------------------------------------------------------------
# Final placement adjustments
# ---------------------------------------------------------------------

# Move the A–B structural legend closer to the molecular panels.
legend_position = (
    ax_structural_legend.get_position()
)

ax_structural_legend.set_position(
    [
        legend_position.x0,
        legend_position.y0 + 0.035,
        legend_position.width,
        legend_position.height,
    ]
)


# Move panels C and D upward, reducing the empty band around
# the structural legend without changing their dimensions.
panel_c_position = ax_c.get_position()

ax_c.set_position(
    [
        panel_c_position.x0,
        panel_c_position.y0 + 0.022,
        panel_c_position.width,
        panel_c_position.height,
    ]
)

panel_d_position = ax_e.get_position()

ax_e.set_position(
    [
        panel_d_position.x0,
        panel_d_position.y0 + 0.022,
        panel_d_position.width,
        panel_d_position.height,
    ]
)


# Move panel E down and reduce its width while keeping it centred.
panel_e_position = ax_d.get_position()

panel_e_width = (
    panel_e_position.width * 0.86
)

panel_e_x0 = (
    panel_e_position.x0
    + (
        panel_e_position.width
        - panel_e_width
    )
    / 2.0
)

ax_d.set_position(
    [
        panel_e_x0,
        panel_e_position.y0 - 0.030,
        panel_e_width,
        panel_e_position.height,
    ]
)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
base = (
    OUTPUT_DIR
    / "figure3_fragmentation"
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
            "Fragmentation mechanism and "
            "shape-metric inflation"
        ),
    },
)

fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("FIGURE3_FRAGMENTATION_STATUS=OK")
print(
    "Fragmented pair-specific evaluations: "
    f"{len(fragmented)}"
)
print(
    "Unique fragmented sanitizable structures: "
    f"{len(unique_fragmented)}"
)
print(
    "Unique fragmentation classes: "
    f"{observed_class_counts}"
)
print(
    "Shape-valid fragmented evaluations: "
    f"{len(shape_valid)}"
)
print(f"Spearman rho: {rho:.4f}")
print(base.with_suffix(".png"))
print(base.with_suffix(".pdf"))
print(base.with_suffix(".svg"))
