"""Figure S3: optimal exploration rate when the distance is random, d ~ Poisson + 1,
and its difference from the optimum at the matching fixed distance."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    centers_to_edges,
    save_figure,
    use_paper_style,
)

use_paper_style()
poisson = np.load(DATA_DIR / "opt_eps_poisson.npz")
comparison = np.load(DATA_DIR / "opt_eps_poisson_comparison.npz")
T_values = poisson["T_values"]
difference = comparison["difference"]
limit = np.abs(difference).max()

fig, axes = plt.subplots(
    2,
    len(T_values),
    figsize=(10.15, 6.3),
    sharex="row",
    sharey="row",
    layout="constrained",
)
for t, T in enumerate(T_values):
    top = axes[0, t].pcolormesh(
        centers_to_edges(poisson["R_values"]),
        centers_to_edges(poisson["mean_distances"]),
        poisson["opt_eps"][t],
        cmap="viridis",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    bottom = axes[1, t].pcolormesh(
        centers_to_edges(comparison["R_values"]),
        centers_to_edges(comparison["d_values"]),
        difference[t],
        cmap="coolwarm_r",
        vmin=-limit,
        vmax=limit,
        rasterized=True,
    )
    axes[0, t].set_title(rf"$\gamma = 1 - 1/{T}$")
    axes[1, t].set_xlabel(r"Reward ratio $R$")
for row in axes[:, 0]:
    row.set_ylabel(r"Mean distance $E[d]$")

fig.colorbar(top, ax=axes[0], pad=0.01, shrink=0.95).set_label(
    r"Optimal exploration rate $\varepsilon^*_{\mathrm{Poisson}}$"
)
fig.colorbar(bottom, ax=axes[1], pad=0.01, shrink=0.95).set_label(
    r"$\varepsilon^*_{\mathrm{Poisson}} - \varepsilon^*_{\mathrm{fixed}}$"
)

save_figure(fig, "opt_eps_poisson.pdf")
