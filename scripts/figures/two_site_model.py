"""Figure S8: local best-response basins in the two-site model and the lattice model."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    OUTCOME_CMAP,
    OUTCOME_LABELS,
    SOCIAL_OPTIMUM_COLOR,
    plot_basins,
    save_figure,
    use_paper_style,
)

use_paper_style()
two_site = np.load(DATA_DIR / "two_site_local_br.npz")
lattice = np.load(DATA_DIR / "two_agent_local_br.npz")
k = list(zip(lattice["d_values"], lattice["R_values"])).index((4, 2.1))

fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
plot_basins(
    axes[0],
    two_site["eps_grid"],
    two_site["codes"],
    two_site["endpoints"],
    two_site["social_optimum"],
)
axes[0].set_title(rf"Two-site model: $d = {two_site['d']}$, $R = {two_site['R']:g}$")
plot_basins(
    axes[1],
    lattice["eps_grid"],
    lattice["codes"][k],
    lattice["endpoints"][k],
    lattice["social_optimum"][k],
)
axes[1].set_title(r"Original model: $d = 4$, $R = 2.1$")

for ax in axes:
    ax.set_xlabel(r"P1 exploration rate $\varepsilon_1$")
    ax.set_ylabel(r"P2 exploration rate $\varepsilon_2$")

axes[0].legend(
    handles=[
        Patch(
            facecolor=OUTCOME_CMAP(code), edgecolor="black", label=OUTCOME_LABELS[code]
        )
        for code in np.unique(two_site["codes"])
    ]
    + [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=SOCIAL_OPTIMUM_COLOR,
            markeredgewidth=2.5,
            markersize=9,
            label="social optimum",
        )
    ],
    loc="upper right",
    framealpha=0.95,
)

save_figure(fig, "two_site_model.pdf")
