import matplotlib.pyplot as plt
import numpy as np

from shotlab import metrics
from shotlab.data import RESULTS

FIGURES = RESULTS / "figures"
MODELS = ["logreg", "hgb", "tabfm", "tabicl"]
LABELS = {"logreg": "Logistic regression", "hgb": "Gradient boosting", "tabfm": "TabFM", "tabicl": "TabICL"}
COLORS = {"logreg": "#2a78d6", "hgb": "#eb6834", "tabfm": "#1baf7a", "tabicl": "#eda100"}
MARKERS = {"logreg": "o", "hgb": "s", "tabfm": "^", "tabicl": "D"}
DIRECTION_LABELS = {"S1_to_S2": "train S1 → score S2", "S2_to_S1": "train S2 → score S1", "S1+S2_to_T": "train S1 + S2 → predict T"}
INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
KIND_COLORS = ["#6250d6", "#e87ba4", "#008300"]
SEQUENTIAL = ["#9ec5f4", "#5598e7", "#256abf", "#104281"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 110,
        "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": SECONDARY, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY, "text.color": INK, "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8,
    })


def series(model):
    return {"color": COLORS[model], "marker": MARKERS[model], "label": LABELS[model], "markeredgecolor": SURFACE,
            "markeredgewidth": 0.8}


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    return fig


def reliability(ax, y, p, bins=20, **line):
    """Observed make rate against mean prediction in equal-count bins, with a 95% band."""
    table = metrics.reliability_table(np.asarray(y), np.asarray(p), bins)
    low, high = table["predicted"].min(), table["predicted"].max()
    ax.plot([low, high], [low, high], color=AXIS, linewidth=1, zorder=1)
    ax.fill_between(table["predicted"], table["observed"] - 1.96 * table["se"], table["observed"] + 1.96 * table["se"],
                    color=line.get("color", INK), alpha=0.15, linewidth=0)
    ax.plot(table["predicted"], table["observed"], **{"marker": "o", "markersize": 3.5, "linewidth": 1.4, **line})
    ax.set_xlabel("predicted make probability")
    ax.set_ylabel("observed make rate")
    return table


def interval_rows(ax, labels, centers, lows, highs, colors=None, zero=True):
    """Horizontal point-and-interval rows, first label on top."""
    positions = np.arange(len(labels))[::-1]
    colors = colors or [INK] * len(labels)
    for position, center, low, high, color in zip(positions, centers, lows, highs, colors):
        ax.plot([low, high], [position, position], color=color, linewidth=2, solid_capstyle="round")
        ax.plot([center], [position], marker="o", markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=1)
    if zero:
        ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(positions, labels)
    ax.grid(axis="y", visible=False)
