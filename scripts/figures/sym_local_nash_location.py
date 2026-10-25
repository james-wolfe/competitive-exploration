"""Figure S4: rate at the symmetric locally stable point, and its difference from the
rate at the positive local maximum of a lone agent's reward curve."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    centers_to_edges,
    save_figure,
    use_paper_style,
)

use_paper_style()
sweep = np.load(DATA_DIR / "sweep_nash_social_dilemma_fine.npz")
single = np.load(DATA_DIR / "local_max_eps_single.npz")
symmetric_eps = sweep["symmetric_eps"]
difference = symmetric_eps - single["local_max_eps"]
limit = np.nanmax(np.abs(difference))
R_edges = centers_to_edges(sweep["R_values"])
d_edges = centers_to_edges(sweep["d_values"])

fig, axes = plt.subplots(
    2,
    len(sweep["T_values"]),
    figsize=(10.15, 6.3),
    sharex=True,
    sharey=True,
    layout="constrained",
)
for t, T in enumerate(sweep["T_values"]):
    top = axes[0, t].pcolormesh(
        R_edges,
        d_edges,
        np.ma.masked_invalid(symmetric_eps[t]),
        cmap="viridis",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    bottom = axes[1, t].pcolormesh(
        R_edges,
        d_edges,
        np.ma.masked_invalid(difference[t]),
        cmap="coolwarm_r",
        norm=TwoSlopeNorm(0, -limit, limit),
        rasterized=True,
    )
    axes[0, t].set_title(rf"$\gamma = 1 - 1/{T}$")
    axes[1, t].set_xlabel(r"Reward ratio $R$")
for row in axes[:, 0]:
    row.set_ylabel(r"Distance $d$")

fig.colorbar(top, ax=axes[0], pad=0.02).set_label(
    r"Symmetric locally stable $\varepsilon$"
)
fig.colorbar(bottom, ax=axes[1], pad=0.02).set_label(
    r"$\varepsilon_{\mathrm{symmetric}} - \varepsilon_{\mathrm{single\,local\,max}}$"
)

save_figure(fig, "sym_local_nash_location.pdf")
