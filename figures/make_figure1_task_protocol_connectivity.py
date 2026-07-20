#!/usr/bin/env python3
"""Create the combined Figure 1 for the staged-growth manuscript.

Panels
------
A  Conditioning references for the hard x0434 -> x2193 pair.
B  Representative guided single-shot sample at lambda = 20.
C  Sampling and post-sampling evaluation protocol.
D  Full-record connected-molecule rate versus guidance strength.
E  Largest-connected-component heavy-atom fraction versus guidance strength.

The script is intended to live in:
    figures

It reads source files from ../exp0 and writes PNG, PDF, and SVG outputs to
its own directory.
"""

from pathlib import Path
import argparse

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "data/derived/exp0"

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
parser.add_argument(
    "--out-prefix",
    type=Path,
    default=SCRIPT_DIR / "figure1_task_protocol_connectivity",
)
args = parser.parse_args()

INPUT_RENDER = args.data_root / "source_renderings/figure1_difficult_input_AB_anchor.png"
GENERATED_RENDER = args.data_root / "source_renderings/figure1_difficult_generated_B_anchor.png"
PER_MOL = args.data_root / "source_data/fragment_audit_per_molecule.tsv"
OUT_BASE = args.out_prefix

for required in (INPUT_RENDER, GENERATED_RENDER, PER_MOL):
    if not required.exists():
        raise FileNotFoundError(f"Required input not found: {required}")


# -----------------------------------------------------------------------------
# Typography and colours
# -----------------------------------------------------------------------------
preferred_fonts = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
FONT = next((f for f in preferred_fonts if f in available_fonts), "DejaVu Sans")

mpl.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 8.0,
        "axes.titlesize": 8.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.7,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B8BCC2",
        "axes.linewidth": 0.8,
        "xtick.color": "#2B2B2B",
        "ytick.color": "#2B2B2B",
        "text.color": "#202020",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)

COLOR_A = "#999999"
COLOR_B = "#18B8C9"
COLOR_GENERATED = "#E69F00"
COLOR_ANCHOR = "#111111"
COLOR_RELAX = "#009E73"

HARD = "#B23A48"
MODERATE = "#2E5E8C"
GRID = "#E6E8EB"
TEXT = "#202020"
SECONDARY = "#626262"
LINE = "#777777"
LIGHT = "#C8C8C8"

PAIR_ORDER = ["x0434_x2193", "x0874_x1093"]
PAIR_COLORS = {"x0434_x2193": HARD, "x0874_x1093": MODERATE}
PAIR_LABELS = {
    "x0434_x2193": "x0434 → x2193  (hard)",
    "x0874_x1093": "x0874 → x1093  (moderate)",
}
LAMBDAS = [0.0, 20.0, 50.0, 100.0, 200.0]


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
d = pd.read_csv(PER_MOL, sep="\t")
d = d[d["pair_id"].isin(PAIR_ORDER)].copy()
for column in ["lambda_global", "parent_heavy_fraction"]:
    d[column] = pd.to_numeric(d[column], errors="coerce")
d["connected"] = d["heavy_connected"].astype(str).str.strip().eq("True")

input_image = Image.open(INPUT_RENDER).convert("RGBA")
generated_image = Image.open(GENERATED_RENDER).convert("RGBA")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def panel_label(ax, label, x=-0.055, y=1.03):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT,
        clip_on=False,
    )


def structural_panel(ax, image, label, title, detail):
    ax.imshow(image, interpolation="lanczos")
    ax.set_axis_off()
    panel_label(ax, label, x=-0.025, y=1.015)
    ax.text(
        0.04,
        0.965,
        title,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.04,
        0.905,
        detail,
        transform=ax.transAxes,
        fontsize=7.1,
        color=SECONDARY,
        ha="left",
        va="top",
    )


def soft_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.7)


def clean_axes(ax):
    ax.spines["left"].set_color("#B8BCC2")
    ax.spines["bottom"].set_color("#B8BCC2")


# -----------------------------------------------------------------------------
# Figure layout
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.45, 8.85), facecolor="white")
grid = fig.add_gridspec(
    nrows=4,
    ncols=2,
    height_ratios=[2.05, 1.30, 0.16, 1.48],
    left=0.075,
    right=0.985,
    bottom=0.065,
    top=0.975,
    wspace=0.25,
    hspace=0.24,
)

ax_a = fig.add_subplot(grid[0, 0])
ax_b = fig.add_subplot(grid[0, 1])
ax_c = fig.add_subplot(grid[1, :])
ax_legend = fig.add_subplot(grid[2, :])
ax_d = fig.add_subplot(grid[3, 0])
ax_e = fig.add_subplot(grid[3, 1])


# -----------------------------------------------------------------------------
# Panels A and B: structural definition
# -----------------------------------------------------------------------------
structural_panel(
    ax_a,
    input_image,
    "A",
    "Conditioning references",
    "x0434 → x2193",
)
structural_panel(
    ax_b,
    generated_image,
    "B",
    "Guided single-shot sample",
    "λ = 20",
)

structural_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markersize=5.0,
           markerfacecolor=COLOR_A, markeredgecolor="#333333",
           markeredgewidth=0.4, label="Reference A"),
    Line2D([0], [0], marker="o", linestyle="none", markersize=5.0,
           markerfacecolor=COLOR_B, markeredgecolor="#333333",
           markeredgewidth=0.4, label="Reference B"),
    Line2D([0], [0], marker="o", linestyle="none", markersize=5.0,
           markerfacecolor=COLOR_GENERATED, markeredgecolor="#333333",
           markeredgewidth=0.4, label="Generated"),
    Line2D([0], [0], marker="o", linestyle="none", markersize=5.0,
           markerfacecolor=COLOR_ANCHOR, markeredgecolor=COLOR_ANCHOR,
           label="Fixed warhead"),
]
fig.legend(
    handles=structural_handles,
    loc="upper center",
    bbox_to_anchor=(0.50, 0.690),
    ncol=4,
    frameon=False,
    fontsize=7.0,
    handletextpad=0.35,
    columnspacing=1.30,
)


# -----------------------------------------------------------------------------
# Panel C: protocol
# -----------------------------------------------------------------------------
ax_c.set_xlim(0.0, 1.0)
ax_c.set_ylim(0.0, 1.0)
ax_c.set_axis_off()
panel_label(ax_c, "C", x=-0.055, y=1.02)

ax_c.text(
    0.0,
    0.985,
    "Sampling and evaluation protocol",
    fontsize=8.8,
    fontweight="bold",
    ha="left",
    va="top",
)
ax_c.plot([0.0, 1.0], [0.865, 0.865], color=LIGHT, linewidth=0.75)

ax_c.text(
    0.0,
    0.785,
    "AT EACH REVERSE-DIFFUSION STEP",
    fontsize=6.5,
    fontweight="bold",
    color=SECONDARY,
    ha="left",
    va="center",
)

sampling_x = [0.035, 0.305, 0.585, 0.885]
sampling_states = [r"$x_t$", r"$\hat{x}_0$", r"$\tilde{x}_0$", r"$x_{t-1}$"]
sampling_details = [
    "noisy coordinates",
    "denoiser estimate",
    "shape-guided free atoms",
    "fixed atoms restored",
]
sampling_operations = [
    ("Denoiser", LINE),
    ("Shape gradient on free atoms", COLOR_B),
    ("Hard warhead overwrite", COLOR_ANCHOR),
]
state_y = 0.500
operation_y = 0.650

for idx, (x, state, detail) in enumerate(
    zip(sampling_x, sampling_states, sampling_details)
):
    ax_c.text(x, state_y, state, fontsize=9.5, ha="center", va="center")
    ax_c.text(
        x,
        0.355,
        detail,
        fontsize=6.4,
        color=SECONDARY,
        ha="center",
        va="top",
    )
    if idx < len(sampling_x) - 1:
        next_x = sampling_x[idx + 1]
        ax_c.add_patch(
            FancyArrowPatch(
                (x + 0.045, state_y),
                (next_x - 0.045, state_y),
                arrowstyle="-|>",
                mutation_scale=8.0,
                linewidth=0.8,
                color=LINE,
                shrinkA=0,
                shrinkB=0,
            )
        )
        operation, operation_color = sampling_operations[idx]
        ax_c.text(
            (x + next_x) / 2,
            operation_y,
            operation,
            fontsize=6.7,
            fontweight="bold",
            color=operation_color,
            ha="center",
            va="center",
        )

ax_c.plot([0.0, 1.0], [0.245, 0.245], color=LIGHT, linewidth=0.75)
ax_c.text(
    0.0,
    0.165,
    "AFTER SAMPLING",
    fontsize=6.5,
    fontweight="bold",
    color=SECONDARY,
    ha="left",
    va="center",
)

audit_x = [0.035, 0.265, 0.485, 0.710, 0.945]
audit_labels = [
    "Generated\ncoordinates",
    "Molecular\ngraph",
    "Connected\ncomponents",
    "Component-aware\nshape scores",
    "Restrained\nminimization",
]
audit_colors = [COLOR_GENERATED, LINE, LINE, COLOR_B, COLOR_RELAX]
audit_y = 0.050

for idx, (x, label, color) in enumerate(zip(audit_x, audit_labels, audit_colors)):
    ax_c.text(
        x,
        audit_y,
        label,
        fontsize=6.5,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        linespacing=1.05,
    )
    if idx < len(audit_x) - 1:
        next_x = audit_x[idx + 1]
        ax_c.add_patch(
            FancyArrowPatch(
                (x + 0.065, audit_y),
                (next_x - 0.065, audit_y),
                arrowstyle="-|>",
                mutation_scale=7.7,
                linewidth=0.75,
                color=LINE,
                shrinkA=0,
                shrinkB=0,
            )
        )


# -----------------------------------------------------------------------------
# Panels D and E: single-shot connectivity failure
# -----------------------------------------------------------------------------
for pair in PAIR_ORDER:
    sub = d[d["pair_id"] == pair]
    rates = [sub.loc[sub["lambda_global"] == lam, "connected"].mean()
             for lam in LAMBDAS]
    fractions = [sub.loc[sub["lambda_global"] == lam, "parent_heavy_fraction"].mean()
                 for lam in LAMBDAS]

    common = dict(
        color=PAIR_COLORS[pair],
        linewidth=1.8,
        marker="o",
        markersize=4.6,
        markeredgecolor="white",
        markeredgewidth=0.65,
        zorder=3,
        label=PAIR_LABELS[pair],
    )
    ax_d.plot(range(len(LAMBDAS)), rates, **common)
    ax_e.plot(range(len(LAMBDAS)), fractions, **common)

for ax in (ax_d, ax_e):
    ax.set_xticks(range(len(LAMBDAS)))
    ax.set_xticklabels([int(lam) for lam in LAMBDAS])
    ax.set_xlabel("guidance strength  λ")
    soft_grid(ax, "y")
    clean_axes(ax)

ax_d.set_ylabel("connected-molecule rate")
ax_d.set_ylim(-0.03, 1.0)
ax_d.set_title("Connected-molecule rate", pad=8)
panel_label(ax_d, "D", x=-0.18, y=1.08)

ax_e.set_ylabel("largest-component\nheavy-atom fraction")
ax_e.set_ylim(0.4, 1.0)
ax_e.set_title("Largest connected component", pad=8)
panel_label(ax_e, "E", x=-0.20, y=1.08)

pair_handles = [
    Line2D([0], [0], color=PAIR_COLORS[p], marker="o", markersize=4.5,
           linewidth=1.8, markeredgecolor="white", markeredgewidth=0.6,
           label=PAIR_LABELS[p])
    for p in PAIR_ORDER
]
ax_legend.set_axis_off()
ax_legend.legend(
    handles=pair_handles,
    loc="center",
    ncol=2,
    frameon=False,
    fontsize=7.0,
    handlelength=1.7,
    columnspacing=1.5,
)


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
fig.savefig(
    OUT_BASE.with_suffix(".png"),
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)
fig.savefig(
    OUT_BASE.with_suffix(".pdf"),
    bbox_inches="tight",
    facecolor="white",
    metadata={"Title": "Dual-conditioning task, protocol, and connectivity failure"},
)
fig.savefig(
    OUT_BASE.with_suffix(".svg"),
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

print(f"Wrote {OUT_BASE.with_suffix('.png')}")
print(f"Wrote {OUT_BASE.with_suffix('.pdf')}")
print(f"Wrote {OUT_BASE.with_suffix('.svg')}")
