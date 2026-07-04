"""Generate manuscript Figure 2 from frozen campaign summary tables."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results/source_data"
OUTPUT_DIR = ROOT / "results/figures"

CONDITION_PATH = DATA_DIR / "fragment_audit_condition_summary.tsv"
ACROSS_PATH = DATA_DIR / "fragment_audit_across_seed_summary.tsv"

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
        "font.size": 8.5,
        "axes.labelsize": 9.3,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
condition = pd.read_csv(
    CONDITION_PATH,
    sep="\t",
)

across = pd.read_csv(
    ACROSS_PATH,
    sep="\t",
)

required_condition = {
    "pair_id",
    "seed",
    "lambda_global",
    "heavy_connected_rate",
    "parent_heavy_fraction_mean",
    "full_tanimoto_B_mean",
    "parent_tanimoto_B_mean",
}

required_across = {
    "pair_id",
    "lambda_global",
    "heavy_connected_rate_seedmean",
    "heavy_connected_rate_seedsd",
    "parent_heavy_fraction_mean_seedmean",
    "parent_heavy_fraction_mean_seedsd",
    "full_strict_dual_total",
    "parent_strict_dual_total",
    "anchor_strict_dual_total",
    "connected_strict_dual_total",
}

missing_condition = required_condition.difference(
    condition.columns
)

missing_across = required_across.difference(
    across.columns
)

if missing_condition:
    raise ValueError(
        "Missing condition columns: "
        f"{sorted(missing_condition)}"
    )

if missing_across:
    raise ValueError(
        "Missing across-seed columns: "
        f"{sorted(missing_across)}"
    )


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
        "label": "Compatible: x0434 → x1093",
        "color": "#0072B2",
        "marker": "o",
        "offset": -0.11,
    },
    "x0874_x1093": {
        "label": "Moderate: x0874 → x1093",
        "color": "#E69F00",
        "marker": "s",
        "offset": 0.00,
    },
    "x0434_x2193": {
        "label": "Difficult: x0434 → x2193",
        "color": "#D55E00",
        "marker": "^",
        "offset": 0.11,
    },
}

condition["pair_id"] = pd.Categorical(
    condition["pair_id"],
    categories=pair_order,
    ordered=True,
)

across["pair_id"] = pd.Categorical(
    across["pair_id"],
    categories=pair_order,
    ordered=True,
)

condition = condition.sort_values(
    ["pair_id", "lambda_global", "seed"]
)

across = across.sort_values(
    ["pair_id", "lambda_global"]
)


# The source columns contain RDKit Shape Tanimoto distances.
condition["full_similarity_B"] = (
    1.0 - condition["full_tanimoto_B_mean"]
)

condition["parent_similarity_B"] = (
    1.0 - condition["parent_tanimoto_B_mean"]
)

condition["shape_inflation_B"] = (
    condition["full_similarity_B"]
    - condition["parent_similarity_B"]
)

inflation_summary = (
    condition
    .groupby(
        ["pair_id", "lambda_global"],
        observed=True,
        as_index=False,
    )
    .agg(
        shape_inflation_mean=(
            "shape_inflation_B",
            "mean",
        ),
        shape_inflation_sd=(
            "shape_inflation_B",
            "std",
        ),
    )
)

strict_by_lambda = (
    across
    .groupby(
        "lambda_global",
        observed=True,
        as_index=False,
    )[
        [
            "full_strict_dual_total",
            "parent_strict_dual_total",
            "anchor_strict_dual_total",
            "connected_strict_dual_total",
        ]
    ]
    .sum()
    .set_index("lambda_global")
    .loc[lambda_order]
    .reset_index()
)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
x_lookup = {
    value: index
    for index, value in enumerate(lambda_order)
}

x_positions = np.arange(
    len(lambda_order),
    dtype=float,
)

seed_jitter = {
    seed: offset
    for seed, offset in zip(
        sorted(condition["seed"].unique()),
        np.linspace(-0.025, 0.025, 5),
    )
}


def style_axis(
    ax,
    panel_letter,
    ylabel,
    ylim=None,
    zero_line=False,
):
    ax.text(
        -0.15,
        1.07,
        panel_letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
    )

    ax.set_ylabel(
        ylabel,
        labelpad=6,
    )

    ax.set_xticks(
        x_positions,
        ["0", "20", "50", "100", "200"],
    )

    ax.set_xlabel(
        "Guidance strength λ",
        labelpad=6,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    if zero_line:
        ax.axhline(
            0.0,
            color="#777777",
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            zorder=1,
        )

    # Separates the unguided baseline from guided conditions.
    ax.axvline(
        0.5,
        color="#D0D0D0",
        linewidth=0.8,
        linestyle=(0, (3, 3)),
        zorder=1,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.3,
        width=0.75,
        pad=4,
    )

    ax.grid(
        axis="y",
        color="#E1E1E1",
        linewidth=0.45,
        zorder=0,
    )

    ax.set_axisbelow(True)


def plot_pair_metric(
    ax,
    condition_column,
    mean_column,
    sd_column,
):
    for pair_id in pair_order:
        style = pair_styles[pair_id]

        seed_data = condition.loc[
            condition["pair_id"] == pair_id
        ].copy()

        summary_data = across.loc[
            across["pair_id"] == pair_id
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

        ax.scatter(
            seed_x,
            seed_data[condition_column],
            s=15,
            marker=style["marker"],
            facecolor=style["color"],
            edgecolor="none",
            alpha=0.30,
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

        means = summary_data[
            mean_column
        ].to_numpy(
            dtype=float
        )

        errors = summary_data[
            sd_column
        ].fillna(
            0.0
        ).to_numpy(
            dtype=float
        )

        ax.errorbar(
            mean_x,
            means,
            yerr=errors,
            color=style["color"],
            marker=style["marker"],
            markersize=5.2,
            markeredgecolor="#1A1A1A",
            markeredgewidth=0.45,
            linewidth=1.35,
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            label=style["label"],
            zorder=4,
        )


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.4, 6.2),
)

fig.subplots_adjust(
    left=0.095,
    right=0.985,
    bottom=0.09,
    top=0.885,
    wspace=0.31,
    hspace=0.36,
)


# ---------------------------------------------------------------------
# A. Connected-molecule rate
# ---------------------------------------------------------------------
ax = axes[0, 0]

plot_pair_metric(
    ax=ax,
    condition_column="heavy_connected_rate",
    mean_column="heavy_connected_rate_seedmean",
    sd_column="heavy_connected_rate_seedsd",
)

style_axis(
    ax=ax,
    panel_letter="A",
    ylabel="Connected-molecule rate",
    ylim=(-0.04, 1.04),
)

ax.yaxis.set_major_locator(
    MultipleLocator(0.2)
)


# ---------------------------------------------------------------------
# B. Parent heavy-atom fraction
# ---------------------------------------------------------------------
ax = axes[0, 1]

plot_pair_metric(
    ax=ax,
    condition_column="parent_heavy_fraction_mean",
    mean_column="parent_heavy_fraction_mean_seedmean",
    sd_column="parent_heavy_fraction_mean_seedsd",
)

style_axis(
    ax=ax,
    panel_letter="B",
    ylabel="Parent heavy-atom fraction",
    ylim=(0.35, 1.04),
)

ax.yaxis.set_major_locator(
    MultipleLocator(0.1)
)


# ---------------------------------------------------------------------
# C. Shape-score inflation
# ---------------------------------------------------------------------
ax = axes[1, 0]

for pair_id in pair_order:
    style = pair_styles[pair_id]

    seed_data = condition.loc[
        condition["pair_id"] == pair_id
    ].copy()

    summary_data = inflation_summary.loc[
        inflation_summary["pair_id"] == pair_id
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

    ax.scatter(
        seed_x,
        seed_data["shape_inflation_B"],
        s=15,
        marker=style["marker"],
        facecolor=style["color"],
        edgecolor="none",
        alpha=0.30,
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

    ax.errorbar(
        mean_x,
        summary_data["shape_inflation_mean"],
        yerr=summary_data[
            "shape_inflation_sd"
        ].fillna(0.0),
        color=style["color"],
        marker=style["marker"],
        markersize=5.2,
        markeredgecolor="#1A1A1A",
        markeredgewidth=0.45,
        linewidth=1.35,
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        zorder=4,
    )

style_axis(
    ax=ax,
    panel_letter="C",
    ylabel="Shape-similarity inflation to B",
    ylim=(-0.035, 0.115),
    zero_line=True,
)

ax.yaxis.set_major_locator(
    MultipleLocator(0.025)
)


# ---------------------------------------------------------------------
# D. Strict-dual counts
# ---------------------------------------------------------------------
ax = axes[1, 1]

bar_definitions = [
    {
        "column": "full_strict_dual_total",
        "label": "Full record",
        "color": "#999999",
        "hatch": "",
    },
    {
        "column": "parent_strict_dual_total",
        "label": "Largest fragment",
        "color": "#56B4E9",
        "hatch": "",
    },
    {
        "column": "anchor_strict_dual_total",
        "label": "Anchor component",
        "color": "#009E73",
        "hatch": "",
    },
    {
        "column": "connected_strict_dual_total",
        "label": "Connected molecule",
        "color": "#D55E00",
        "hatch": "//",
    },
]

bar_width = 0.18

bar_offsets = np.asarray(
    [-1.5, -0.5, 0.5, 1.5]
) * bar_width

for definition, offset in zip(
    bar_definitions,
    bar_offsets,
):
    values = strict_by_lambda[
        definition["column"]
    ].to_numpy(
        dtype=float
    )

    ax.bar(
        x_positions + offset,
        values,
        width=bar_width,
        color=definition["color"],
        edgecolor="#1A1A1A",
        linewidth=0.55,
        hatch=definition["hatch"],
        label=definition["label"],
        zorder=3,
    )

style_axis(
    ax=ax,
    panel_letter="D",
    ylabel="Strict-dual candidates",
    ylim=(0.0, 40.0),
)

ax.yaxis.set_major_locator(
    MultipleLocator(10)
)

ax.legend(
    loc="upper left",
    frameon=False,
    fontsize=7.1,
    handlelength=1.3,
    handletextpad=0.45,
    borderaxespad=0.3,
)


# ---------------------------------------------------------------------
# Shared pair legend
# ---------------------------------------------------------------------
handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.54, 0.982),
    ncol=3,
    frameon=False,
    fontsize=8.0,
    handletextpad=0.5,
    columnspacing=1.5,
)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
base = (
    OUTPUT_DIR
    / "figure2_generation_outcomes"
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
            "Connectivity and shape outcomes "
            "across guidance strengths"
        ),
    },
)

fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("ALL_GENERATION_OUTCOMES_STATUS=OK")
print(f"Font: {font_family}")
print(base.with_suffix(".png"))
print(base.with_suffix(".pdf"))
print(base.with_suffix(".svg"))
