"""Figure S6: collective payoff reached by the local best-response dynamics, averaged
over starting pairs, relative to the global optimum."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    centers_to_edges,
    save_figure,
    use_paper_style,
)

use_paper_style()
data = np.load(DATA_DIR / "local_br_outcomes.npz")
ratio = data["mean_payoff"] / data["global_optimum_payoff"]

fig, axes = plt.subplots(
    1,
    len(data["T_values"]),
    figsize=(9.75, 3.25),
    sharex=True,
    sharey=True,
    layout="constrained",
)
for t, T in enumerate(data["T_values"]):
    image = axes[t].pcolormesh(
        centers_to_edges(data["R_values"]),
        centers_to_edges(data["d_values"]),
        ratio[t],
        cmap="viridis",
        vmin=ratio.min(),
        vmax=ratio.max(),
        rasterized=True,
    )
    axes[t].set_title(rf"$\gamma = 1 - 1/{T}$")
    axes[t].set_xlabel(r"Reward ratio $R$")
axes[0].set_ylabel(r"Distance $d$")

fig.colorbar(image, ax=axes, pad=0.02, shrink=0.9).set_label(
    "Mean collective payoff / global optimum"
)

save_figure(fig, "local_br_mean_fitness_ratio.pdf")
