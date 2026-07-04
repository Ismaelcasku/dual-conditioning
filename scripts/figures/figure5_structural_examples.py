"""Assemble manuscript Figure 5 from PyMOL structural renderings."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "results/source_renderings/relaxation"
OUTPUT_DIR = ROOT / "results/figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Inputs and panel order
# ---------------------------------------------------------------------
panels = [
    {
        "letter": "A",
        "title": "Compatible boundary case",
        "file": IMAGE_DIR / "01_stable_dual_overlay.png",
    },
    {
        "letter": "B",
        "title": "B-directed signal attenuation",
        "file": IMAGE_DIR / "02_hard_pair_relaxation_overlay.png",
    },
    {
        "letter": "C",
        "title": "Global improvement with anchor drift",
        "file": IMAGE_DIR / "04_global_shape_anchor_drift_overlay.png",
    },
    {
        "letter": "D",
        "title": "Minimization-rescued pose",
        "file": IMAGE_DIR / "03_minimization_rescued_overlay.png",
    },
]

for panel in panels:
    if not panel["file"].is_file():
        raise FileNotFoundError(
            f"Missing structural image: {panel['file']}"
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.4, 5.9),
)

fig.subplots_adjust(
    left=0.015,
    right=0.985,
    top=0.955,
    bottom=0.115,
    wspace=0.025,
    hspace=0.17,
)

for ax, panel in zip(axes.flat, panels):
    image = plt.imread(panel["file"])

    ax.imshow(
        image,
        interpolation="lanczos",
    )

    ax.set_axis_off()

    # Short descriptive panel heading
    ax.set_title(
        panel["title"],
        fontsize=9.2,
        fontweight="normal",
        pad=5,
    )

    # Panel identifier inside a small white box
    ax.text(
        0.018,
        0.975,
        panel["letter"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        bbox={
            "boxstyle": "square,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        zorder=10,
    )


# ---------------------------------------------------------------------
# Shared molecular colour legend
# ---------------------------------------------------------------------
legend_items = [
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markersize=7,
        markerfacecolor="#999999",
        markeredgecolor="#333333",
        label="Local reference A",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markersize=7,
        markerfacecolor="#00FFFF",
        markeredgecolor="#333333",
        label="Global reference B",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markersize=7,
        markerfacecolor="#FFA500",
        markeredgecolor="#333333",
        label="Generated",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        linestyle="none",
        markersize=7,
        markerfacecolor="#00CC33",
        markeredgecolor="#333333",
        label="Minimized",
    ),
]

fig.legend(
    handles=legend_items,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.025),
    ncol=4,
    frameon=False,
    fontsize=8.2,
    handletextpad=0.45,
    columnspacing=1.7,
)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
base = OUTPUT_DIR / "figure5_structural_examples"

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
        "Title": "Structural outcomes after restrained minimization",
    },
)

fig.savefig(
    base.with_suffix(".tiff"),
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    pil_kwargs={
        "compression": "tiff_lzw",
    },
)

plt.close(fig)

print("FIGURE5_STRUCTURAL_EXAMPLES_STATUS=OK")
print(f"Font: {font_family}")
print(base.with_suffix(".png"))
print(base.with_suffix(".pdf"))
print(base.with_suffix(".tiff"))
