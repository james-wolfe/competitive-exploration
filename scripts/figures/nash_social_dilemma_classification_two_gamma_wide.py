"""Figure 5: type of the Nash equilibria and of the socially optimal locally stable
point over reward ratio and distance, with strict social dilemmas hatched, for two
discount factors. SI Appendix shows the shortest horizon as well."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle

from competitive_exploration.equilibria import OTHER
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    OUTCOME_CMAP,
    OUTCOME_LABELS,
    OUTCOME_NORM,
    PNAS_WIDTHS,
    TWO_GAMMA_PANEL_SIZE,
    add_row_label,
    centers_to_edges,
    place_panels,
    save_figure,
    use_paper_style,
)

use_paper_style(final_size=True)
# Distances d = 2-30 (Methods). At d = 1 every strict dilemma is a shortfall of at
# most 0.022% in collective payoff between neighboring near-symmetric equilibria on the grid.
D_MIN = 2
T_SHOWN = [5000, 100000]
raw = np.load(DATA_DIR / "sweep_nash_social_dilemma_fine401.npz")
keep = raw["d_values"] >= D_MIN
t_keep = [list(raw["T_values"]).index(T) for T in T_SHOWN]
data = {key: raw[key] for key in raw.files}
data["T_values"] = raw["T_values"][t_keep]
data["d_values"] = raw["d_values"][keep]
for key in ["nash_type", "social_optimum_type", "strict_dilemma"]:
    data[key] = raw[key][t_keep][:, keep]
R_edges = centers_to_edges(data["R_values"])
d_edges = centers_to_edges(data["d_values"])
types = {key: data[key].copy() for key in ["nash_type", "social_optimum_type"]}
assert not (types["social_optimum_type"] == OTHER).any() and not (types["nash_type"] == OTHER).any()


def hatch_cells(ax, mask):
    """Hatch the cells where mask is True and outline the hatched region."""
    outline = []
    with mpl.rc_context({"hatch.linewidth": 0.4}):
        for a, r in np.argwhere(mask):
            x0, x1, y0, y1 = R_edges[r], R_edges[r + 1], d_edges[a], d_edges[a + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="none",
                    edgecolor="white",
                    hatch="//////",
                    linewidth=0,
                    zorder=10,
                )
            )
            if r > 0 and not mask[a, r - 1]:
                outline.append([(x0, y0), (x0, y1)])
            if r < mask.shape[1] - 1 and not mask[a, r + 1]:
                outline.append([(x1, y0), (x1, y1)])
            if a > 0 and not mask[a - 1, r]:
                outline.append([(x0, y0), (x1, y0)])
            if a < mask.shape[0] - 1 and not mask[a + 1, r]:
                outline.append([(x0, y1), (x1, y1)])
    ax.add_collection(
        LineCollection(outline, colors="white", linewidths=0.6, zorder=11)
    )


# Square panels the size of those in Figure 6, with the legend on the right, at the
# width of a 3/4-page figure.
panel = TWO_GAMMA_PANEL_SIZE
width, left, bottom, top, gap = PNAS_WIDTHS["one_and_half"], 0.58, 0.33, 0.2, 0.1
height = bottom + 2 * panel + gap + top
fig, axes = plt.subplots(
    2, len(data["T_values"]), figsize=(width, height), sharex=True, sharey=True
)
place_panels(fig, axes, panel, left, bottom, wspace=gap, hspace=gap)
for t, T in enumerate(data["T_values"]):
    for row, key in enumerate(["nash_type", "social_optimum_type"]):
        axes[row, t].pcolormesh(
            R_edges,
            d_edges,
            types[key][t],
            cmap=OUTCOME_CMAP,
            norm=OUTCOME_NORM,
            alpha=0.95,
            antialiased=False,
            rasterized=True,
        )
    hatch_cells(axes[1, t], data["strict_dilemma"][t])
    axes[0, t].set_title(rf"$\gamma = 1 - 1/{T}$")
    axes[1, t].set_xlabel(r"Reward ratio $R$")

for ax in axes.ravel():
    ax.set_xticks([5, 10, 15, 20])
    ax.set_yticks([5, 10, 15, 20, 25, 30])
    ax.tick_params(length=2, width=0.5, pad=2)
    for spine in ax.spines.values():
        spine.set_zorder(30)
        spine.set_linewidth(0.6)
for row, label in enumerate(["Nash equilibrium", "Social optimum"]):
    axes[row, 0].set_ylabel(r"Distance $d$")
    add_row_label(axes[row, 0], label, x=-0.37)

present = np.union1d(types["nash_type"], types["social_optimum_type"])
fig.legend(
    handles=[
        Patch(color=OUTCOME_CMAP(code), label=OUTCOME_LABELS[code]) for code in present
    ]
    + [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="//////",
            label="strict social\ndilemma",
        )
    ],
    loc="center left",
    bbox_to_anchor=(
        (left + 2 * panel + gap + 0.08) / width,
        (bottom + panel + gap / 2) / height,
    ),
    handlelength=1.0,
    handletextpad=0.5,
    borderaxespad=0,
    frameon=False,
)

save_figure(fig, "nash_social_dilemma_classification_two_gamma_wide.pdf")
