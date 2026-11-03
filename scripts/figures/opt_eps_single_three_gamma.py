"""Figure S2: the optimal exploration rate over reward ratio and distance for three
discount factors (Figure 2B with the shortest horizon as well)."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    CM,
    PNAS_WIDTHS,
    centers_to_edges,
    save_figure,
    use_paper_style,
)

use_paper_style(final_size=True)
opt_data = np.load(DATA_DIR / "opt_eps_single.npz")

fig = plt.figure(figsize=(PNAS_WIDTHS["double"], 6.4 * CM), layout="constrained")
R_edges = centers_to_edges(opt_data["R_values"])
d_edges = centers_to_edges(opt_data["d_values"])
axes = fig.subplots(1, len(opt_data["T_values"]), sharey=True)
for ax, T, opt_eps in zip(axes, opt_data["T_values"], opt_data["opt_eps"]):
    image = ax.pcolormesh(
        R_edges, d_edges, opt_eps, cmap="viridis", vmin=0, vmax=1, rasterized=True
    )
    ax.set_title(rf"$\gamma = 1 - 1/{T}$")
    ax.set_xlabel(r"Reward ratio $R$")
axes[0].set_ylabel(r"Distance $d$")
colorbar = fig.colorbar(image, ax=axes, pad=0.01, shrink=0.95)
colorbar.set_label(r"Optimal exploration rate $\varepsilon^*$")

save_figure(fig, "opt_eps_single_three_gamma.pdf")
