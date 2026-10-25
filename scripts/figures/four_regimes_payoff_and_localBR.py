"""Figure 4: payoffs, Nash equilibria, and local best-response basins in four regimes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from competitive_exploration.equilibria import best_response
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    OUTCOME_CMAP,
    OUTCOME_LABELS,
    CM,
    PNAS_WIDTHS,
    SOCIAL_OPTIMUM_COLOR,
    add_row_label,
    plot_basins,
    save_figure,
    use_paper_style,
)

use_paper_style(final_size=True)
SCALE = 0.55  # marker sizes and line widths, relative to the defaults of plot_basins
data = np.load(DATA_DIR / "two_agent_local_br.npz")
eps_grid = data["eps_grid"]
Z_all = data["Z"]

# Printed at 0.99 \linewidth.
fig = plt.figure(figsize=(0.99 * PNAS_WIDTHS["double"], 8.67 * CM), layout="constrained")
grid = fig.add_gridspec(
    2, 5, width_ratios=[1, 1, 1, 1, 0.035], wspace=0, hspace=0
)
# Pads chosen to make the gaps between rows and between columns equal.
fig.get_layout_engine().set(w_pad=0.022, h_pad=0.01, wspace=0, hspace=0)
axes = np.array([[fig.add_subplot(grid[row, k]) for k in range(4)] for row in range(2)])

for k, (d, R) in enumerate(zip(data["d_values"], data["R_values"])):
    top, bottom = axes[:, k]

    image = top.imshow(
        Z_all[k],
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="viridis",
        vmin=Z_all.min(),
        vmax=Z_all.max(),
    )
    top.plot(
        eps_grid,
        eps_grid[best_response(Z_all[k])],
        color="black",
        linewidth=2 * SCALE,
        zorder=10,
    )
    i, j = np.nonzero(data["nash"][k])
    top.scatter(
        eps_grid[j],
        eps_grid[i],
        s=75 * SCALE**2,
        facecolors="white",
        edgecolors="black",
        linewidths=1.7 * SCALE,
        zorder=11,
        clip_on=False,
    )
    top.set_title(rf"$d = {d}$, $R = {R:g}$")
    for ax in (top, bottom):
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
        ax.tick_params(length=2.5, width=0.6, pad=2)
    top.set_xticklabels([])

    plot_basins(
        bottom,
        eps_grid,
        data["codes"][k],
        data["endpoints"][k],
        data["social_optimum"][k],
        scale=SCALE,
    )
    bottom.set_xlabel(r"P1 exploration rate $\varepsilon_1$")

    if k == 0:
        top.set_ylabel(r"P2 exploration rate $\varepsilon_2$")
        bottom.set_ylabel(r"P2 exploration rate $\varepsilon_2$")
    else:
        top.set_yticklabels([])
        bottom.set_yticklabels([])

fig.colorbar(image, cax=fig.add_subplot(grid[0, 4])).set_label("P2 expected reward")
add_row_label(axes[0, 0], "Payoff matrix", x=-0.44)
add_row_label(axes[1, 0], "Local BR outcome", x=-0.44)


def marker(label, face, edge="black", width=1.7 * SCALE):
    return Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markersize=8.5 * SCALE,
        markerfacecolor=face,
        markeredgecolor=edge,
        markeredgewidth=width,
        label=label,
    )


LEGEND_STYLE = dict(
    loc="upper right",
    fontsize=6,
    handlelength=1.5,
    handletextpad=0.4,
    borderpad=0.3,
    labelspacing=0.25,
    borderaxespad=0.3,
)
axes[0, 0].legend(
    handles=[
        Line2D([], [], color="black", linewidth=2 * SCALE, label="P2 best response"),
        marker("Nash equilibrium", "white"),
    ],
    **LEGEND_STYLE,
)
axes[1, 0].legend(
    handles=[marker(OUTCOME_LABELS[code], OUTCOME_CMAP(code)) for code in range(3)]
    + [marker("social optimum", "none", SOCIAL_OPTIMUM_COLOR, 2.4 * SCALE)],
    **LEGEND_STYLE,
)

save_figure(fig, "four_regimes_payoff_and_localBR.pdf")
