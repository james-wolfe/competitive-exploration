"""Plotting style and helpers shared by the paper figures."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

from .paths import FIGURES_DIR

# Outcome types, indexed by the codes in equilibria (both stay, both explore,
# anti-coordination, multiple, other).
OUTCOME_LABELS = ["both stay", "both explore", "anti-coordination", "multiple", "other"]
OUTCOME_CMAP = ListedColormap(["#ff7f0e", "#1f77b4", "#2ca02c", "magenta", "#b3b3b3"])
OUTCOME_NORM = BoundaryNorm(np.arange(6) - 0.5, OUTCOME_CMAP.N)

SOCIAL_OPTIMUM_COLOR = "magenta"

# PNAS figure widths, in inches: one column, one and a half columns, and two columns
# (the 512 pt \textwidth of the PNAS LaTeX class, which figure* uses).
CM = 1 / 2.54
PNAS_WIDTHS = {"single": 8.7 * CM, "one_and_half": 11.4 * CM, "double": 512 / 72}
# Side of the square panels of Figures 5 and 6, in inches: about the size of a Figure 4
# panel.
PANEL_SIZE = 1.4


def use_paper_style(final_size=False):
    """Reset matplotlib and apply the font sizes and output settings of every figure.

    With final_size, the figure is drawn and saved at exactly the size it is printed,
    and the text uses the PNAS sizes of 6-8 pt; otherwise the figure is drawn larger,
    trimmed to its contents, and scaled down in LaTeX.
    """
    text, title, legend = (7, 8, 6.5) if final_size else (10, 12, 9)
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": text,
            "axes.titlesize": title,
            "axes.labelsize": text,
            "xtick.labelsize": text,
            "ytick.labelsize": text,
            "legend.fontsize": legend,
            "grid.alpha": 0.3,
            "savefig.dpi": 500,
            "savefig.bbox": None if final_size else "tight",
        }
    )


def save_figure(fig, filename):
    """Save a figure to the figures directory."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / filename)


def place_panels(fig, axes, size, left, bottom, wspace, hspace):
    """Place a grid of axes as square panels of a fixed size.

    Sets each axis in the 2-D array axes to a square of side size, with the lower left
    panel at (left, bottom) and gaps wspace and hspace between panels, all in inches.
    Set the figure size first.
    """
    width, height = fig.get_size_inches()
    nrows = axes.shape[0]
    for row, col in np.ndindex(axes.shape):
        x = left + col * (size + wspace)
        y = bottom + (nrows - 1 - row) * (size + hspace)
        axes[row, col].set_position([x / width, y / height, size / width, size / height])


def add_row_label(ax, text, x=-0.3):
    """Label a row of panels to the left of its first axis."""
    ax.text(
        x,
        0.5,
        text,
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=plt.rcParams["axes.titlesize"],
    )


def add_colored_curve(ax, x, y, c, cmap="coolwarm_r", linewidth=2.5):
    """Draw y(x) with each segment colored by c in [0, 1]."""
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lines = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1), linewidth=linewidth)
    lines.set_array(c[:-1])
    ax.add_collection(lines)


def centers_to_edges(x):
    """Cell edges for a grid of cell centers, for pcolormesh."""
    x = np.asarray(x, dtype=float)
    mid = (x[:-1] + x[1:]) / 2
    return np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])


def plot_basins(ax, eps_grid, codes, endpoints, social_mask, scale=1.0):
    """Basins of attraction of the local best-response dynamics (Figures 4 and S8).

    Shades each starting pair by the outcome it reaches, marks the endpoints in the
    color of their outcome, and rings the socially optimal endpoints. scale shrinks the
    markers and their outlines, for a figure drawn at its printed size.
    """
    extent = [eps_grid[0], eps_grid[-1], eps_grid[0], eps_grid[-1]]
    ax.imshow(
        codes,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap=OUTCOME_CMAP,
        norm=OUTCOME_NORM,
        alpha=0.75,
    )

    i, j = np.nonzero(endpoints)
    ax.scatter(
        eps_grid[j],
        eps_grid[i],
        s=90 * scale**2,
        c=codes[i, j],
        cmap=OUTCOME_CMAP,
        norm=OUTCOME_NORM,
        edgecolors="black",
        linewidths=1.7 * scale,
        zorder=10,
        clip_on=False,
    )

    i, j = np.nonzero(social_mask)
    ax.scatter(
        eps_grid[j],
        eps_grid[i],
        s=165 * scale**2,
        facecolors="none",
        edgecolors=SOCIAL_OPTIMUM_COLOR,
        linewidths=2.5 * scale,
        zorder=20,
        clip_on=False,
    )
    ax.set_xlim(eps_grid[0], eps_grid[-1])
    ax.set_ylim(eps_grid[0], eps_grid[-1])
