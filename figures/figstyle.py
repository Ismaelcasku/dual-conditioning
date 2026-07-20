"""Shared design system for the staged-growth manuscript figures.

Single source of truth for typography, palette, and helpers so every figure
in the paper is visually consistent. Import and call apply_style() at the top
of each figure script.
"""
import matplotlib as mpl
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------------
# Palette. Two-pair identity + branch identity + neutral greys.
# ---------------------------------------------------------------------------
# Pair identity (kept constant across the whole paper)
HARD      = "#B23A48"   # x0434 -> x2193 (hard)   — muted red
MODERATE  = "#2E5E8C"   # x0874 -> x1093 (moderate) — deep blue

# Branch identity (Experiment 2)
B_BLIND   = "#7B9BB5"   # best-of-10, B-blind    — desaturated blue
DIRECTED  = "#D98A3D"   # directed selection     — warm amber

# Semantic
POSITIVE  = "#3B7A57"   # movement toward B / good
FRAG      = "#C9CCD1"   # fragmented / background points
GRID      = "#E6E8EB"
INK       = "#2B2B2B"   # main text/ink
MUTED     = "#8A9099"   # secondary text

PAIR_COLORS = {"x0434_x2193": HARD, "x0874_x1093": MODERATE}
PAIR_LABELS = {
    "x0434_x2193": "x0434 \u2192 x2193  (hard)",
    "x0874_x1093": "x0874 \u2192 x1093  (moderate)",
}
BRANCH_COLORS = {"A10": B_BLIND, "directed": DIRECTED}
BRANCH_LABELS = {
    "A10": "best-of-10 (B-blind)",
    "directed": "directed (fixed-increment)",
}

# Sequential colormap for anchor size (perceptually uniform, on-brand)
SIZE_CMAP = "cividis"


def apply_style():
    # Prefer a clean sans; DejaVu Sans is guaranteed present and renders well.
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#B8BCC2",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 10,
        "axes.labelpad": 5,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "figure.dpi": 130,
    })


def soft_grid(ax, axis="both"):
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8, zorder=0)


def panel_tag(ax, letter, dx=-0.10, dy=1.06):
    """Bold panel letter (A, B, ...) in axis-fraction coords."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left", color=INK)


def tidy(ax):
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#B8BCC2")
