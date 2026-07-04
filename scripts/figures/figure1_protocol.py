"""Assemble manuscript Figure 1 from frozen PyMOL renderings."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

INPUT_RENDER = (
    ROOT
    / "results/source_renderings"
    / "figure1_difficult_input_AB_anchor.png"
)

GENERATED_RENDER = (
    ROOT
    / "results/source_renderings"
    / "figure1_difficult_generated_B_anchor.png"
)

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
        "font.size": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ---------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------
COLOR_A = "#999999"
COLOR_B = "#18B8C9"
COLOR_GENERATED = "#E69F00"
COLOR_ANCHOR = "#111111"
COLOR_RELAX = "#009E73"

COLOR_TEXT = "#202020"
COLOR_SECONDARY = "#626262"
COLOR_LINE = "#777777"
COLOR_LIGHT = "#C8C8C8"


# Preserve the identical PyMOL canvas in both panels.
input_image = Image.open(INPUT_RENDER).convert("RGBA")
generated_image = Image.open(GENERATED_RENDER).convert("RGBA")


def panel_label(ax, label):
    ax.text(
        -0.025,
        1.015,
        label,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLOR_TEXT,
    )


def structural_panel(
    ax,
    image,
    label,
    title,
    detail,
):
    ax.imshow(
        image,
        interpolation="lanczos",
    )

    ax.set_axis_off()
    panel_label(ax, label)

    ax.text(
        0.04,
        0.965,
        title,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        color=COLOR_TEXT,
        ha="left",
        va="top",
    )

    ax.text(
        0.04,
        0.905,
        detail,
        transform=ax.transAxes,
        fontsize=7.2,
        color=COLOR_SECONDARY,
        ha="left",
        va="top",
    )


def arrow_between(ax, x0, x1, y):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=0.8,
            color=COLOR_LINE,
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )


def protocol_node(
    ax,
    x,
    title,
    detail,
    edgecolor,
):
    y = 0.49

    ax.add_patch(
        Circle(
            (x, y),
            radius=0.026,
            facecolor="white",
            edgecolor=edgecolor,
            linewidth=1.5,
            zorder=3,
        )
    )

    ax.text(
        x,
        0.35,
        title,
        fontsize=7.4,
        fontweight="bold",
        color=COLOR_TEXT,
        ha="center",
        va="top",
    )

    ax.text(
        x,
        0.19,
        detail,
        fontsize=6.7,
        color=COLOR_SECONDARY,
        ha="center",
        va="top",
        linespacing=1.15,
    )


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
fig = plt.figure(
    figsize=(7.45, 4.95),
)

grid = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[2.05, 1.35],
    left=0.045,
    right=0.985,
    bottom=0.065,
    top=0.96,
    wspace=0.025,
    hspace=0.17,
)

ax_a = fig.add_subplot(grid[0, 0])
ax_b = fig.add_subplot(grid[0, 1])
ax_c = fig.add_subplot(grid[1, :])


# ---------------------------------------------------------------------
# Structural panels
# ---------------------------------------------------------------------
structural_panel(
    ax=ax_a,
    image=input_image,
    label="A",
    title="Conditioning references",
    detail="x0434 → x2193",
)

structural_panel(
    ax=ax_b,
    image=generated_image,
    label="B",
    title="Guided sample",
    detail="λ = 20",
)


# ---------------------------------------------------------------------
# Panel C — algorithmic protocol
# ---------------------------------------------------------------------
ax_c.set_xlim(0.0, 1.0)
ax_c.set_ylim(0.0, 1.0)
ax_c.set_axis_off()

panel_label(ax_c, "C")

# Title separated from the methodological content
ax_c.text(
    0.045,
    0.985,
    "Sampling and evaluation protocol",
    fontsize=8.8,
    fontweight="bold",
    color=COLOR_TEXT,
    ha="left",
    va="top",
)

ax_c.plot(
    [0.045, 0.975],
    [0.875, 0.875],
    color=COLOR_LIGHT,
    linewidth=0.75,
)


# -----------------------------------------------------------------
# Lane 1: operations repeated at every reverse-diffusion step
# -----------------------------------------------------------------
ax_c.text(
    0.045,
    0.795,
    "AT EACH REVERSE-DIFFUSION STEP",
    fontsize=6.6,
    fontweight="bold",
    color=COLOR_SECONDARY,
    ha="left",
    va="center",
)

sampling_x = [
    0.075,
    0.315,
    0.585,
    0.875,
]

sampling_states = [
    r"$x_t$",
    r"$\hat{x}_0$",
    r"$\tilde{x}_0$",
    r"$x_{t-1}$",
]

sampling_details = [
    "noisy coordinates",
    "denoiser estimate",
    "shape-guided free atoms",
    "fixed atoms restored",
]

sampling_operations = [
    ("Denoiser", COLOR_LINE),
    ("Shape gradient on free atoms", COLOR_B),
    ("Hard anchor overwrite", COLOR_ANCHOR),
]

state_y = 0.505
operation_y = 0.665

for index, (
    x,
    state,
    detail,
) in enumerate(
    zip(
        sampling_x,
        sampling_states,
        sampling_details,
    )
):
    ax_c.text(
        x,
        state_y,
        state,
        fontsize=10.0,
        color=COLOR_TEXT,
        ha="center",
        va="center",
    )

    ax_c.text(
        x,
        0.370,
        detail,
        fontsize=6.7,
        color=COLOR_SECONDARY,
        ha="center",
        va="top",
    )

    if index < len(sampling_x) - 1:
        next_x = sampling_x[index + 1]

        ax_c.add_patch(
            FancyArrowPatch(
                (x + 0.045, state_y),
                (next_x - 0.045, state_y),
                arrowstyle="-|>",
                mutation_scale=8.5,
                linewidth=0.85,
                color=COLOR_LINE,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

        operation, operation_color = (
            sampling_operations[index]
        )

        ax_c.text(
            (x + next_x) / 2,
            operation_y,
            operation,
            fontsize=6.9,
            fontweight="bold",
            color=operation_color,
            ha="center",
            va="center",
        )


# Separator between sampling and post-sampling evaluation
ax_c.plot(
    [0.045, 0.975],
    [0.260, 0.260],
    color=COLOR_LIGHT,
    linewidth=0.75,
)


# -----------------------------------------------------------------
# Lane 2: operations performed after generation
# -----------------------------------------------------------------
ax_c.text(
    0.045,
    0.175,
    "AFTER SAMPLING",
    fontsize=6.6,
    fontweight="bold",
    color=COLOR_SECONDARY,
    ha="left",
    va="center",
)

audit_x = [
    0.075,
    0.285,
    0.490,
    0.700,
    0.915,
]

audit_labels = [
    "Generated\ncoordinates",
    "Molecular\ngraph",
    "Connected\ncomponents",
    "Component-aware\nshape scores",
    "Restrained\nminimization",
]

audit_colors = [
    COLOR_GENERATED,
    COLOR_LINE,
    COLOR_LINE,
    COLOR_B,
    COLOR_RELAX,
]

audit_y = 0.055

for index, (
    x,
    label,
    color,
) in enumerate(
    zip(
        audit_x,
        audit_labels,
        audit_colors,
    )
):
    ax_c.text(
        x,
        audit_y,
        label,
        fontsize=6.8,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        linespacing=1.05,
    )

    if index < len(audit_x) - 1:
        next_x = audit_x[index + 1]

        ax_c.add_patch(
            FancyArrowPatch(
                (x + 0.063, audit_y),
                (next_x - 0.063, audit_y),
                arrowstyle="-|>",
                mutation_scale=8.0,
                linewidth=0.8,
                color=COLOR_LINE,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )


# ---------------------------------------------------------------------
# Compact legend
# ---------------------------------------------------------------------
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.2,
        markerfacecolor=COLOR_A,
        markeredgecolor="#333333",
        markeredgewidth=0.4,
        label="Reference A",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.2,
        markerfacecolor=COLOR_B,
        markeredgecolor="#333333",
        markeredgewidth=0.4,
        label="Reference B",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.2,
        markerfacecolor=COLOR_GENERATED,
        markeredgecolor="#333333",
        markeredgewidth=0.4,
        label="Generated",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markersize=5.2,
        markerfacecolor=COLOR_ANCHOR,
        markeredgecolor=COLOR_ANCHOR,
        label="Fixed anchor",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.50, 0.505),
    ncol=4,
    frameon=False,
    fontsize=7.2,
    handletextpad=0.35,
    columnspacing=1.35,
)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
base = (
    OUTPUT_DIR
    / "figure1_protocol"
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
            "Dual-conditioning task and "
            "evaluation protocol"
        ),
    },
)

fig.savefig(
    base.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("FIGURE1_PROTOCOL_STATUS=OK")
print(base.with_suffix(".png"))
print(base.with_suffix(".pdf"))
print(base.with_suffix(".svg"))
