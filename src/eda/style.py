"""Shared matplotlib styling for every EDA figure, so the three reports read
as one system. Palette values are the validated default from the dataviz
skill (light mode only -- these are static PNGs embedded in markdown, so
there's no theme toggle to serve)."""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical order -- never cycled/reordered, never assigned by rank.
CATEGORICAL = {
    "blue": "#2a78d6",
    "green": "#008300",
    "magenta": "#e87ba4",
    "yellow": "#eda100",
    "aqua": "#1baf7a",
    "orange": "#eb6834",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CATEGORICAL_ORDER = list(CATEGORICAL.values())

NORMAL_COLOR = CATEGORICAL["blue"]
ATTACK_COLOR = CATEGORICAL["red"]

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]
)

DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list(
    "diverging_blue_red", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8a1f1e"]
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "lines.linewidth": 1.4,
    })


def categorical_color(i: int) -> str:
    return CATEGORICAL_ORDER[i % len(CATEGORICAL_ORDER)]
