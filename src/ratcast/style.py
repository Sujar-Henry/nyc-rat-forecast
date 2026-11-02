"""One visual language for every figure and map (dark surface, video-friendly)."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

SERIES = ["#3987e5", "#d95926", "#199e70"]  # blue, orange, aqua (dark-mode steps)

# Sequential blue. On a dark surface the anchor flips: the darkest step sits
# near the surface (low), the lightest step glows (high).
BLUE_RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
             "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
RISK_CMAP = LinearSegmentedColormap.from_list("ratcast_risk", BLUE_RAMP)

# Diverging blue <-> red with a neutral gray midpoint ("model was right").
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "ratcast_diff", ["#3987e5", "#256abf", "#383835", "#c94a4a", "#e66767"]
)

FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": PAGE,
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlecolor": INK,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 30,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "legend.fontsize": 10,
    })


def subtitle(ax, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, color=MUTED, fontsize=10.5, va="bottom")


def save(fig, path, dpi: int = 200, exact_frame: bool = False) -> None:
    """Tight-crop by default; ``exact_frame`` keeps the figure size (video frames)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if exact_frame:
        fig.savefig(path, dpi=dpi)
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"  wrote {path.relative_to(path.parents[2])}")
