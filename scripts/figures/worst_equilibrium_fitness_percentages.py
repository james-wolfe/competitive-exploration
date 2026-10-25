"""Figure S5: collective payoff at the worst locally stable point and at the worst Nash
equilibrium, relative to the global optimum and to the best locally stable point."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    centers_to_edges,
    save_figure,
    use_paper_style,
)

use_paper_style()
sweep = np.load(DATA_DIR / "sweep_nash_social_dilemma_fine.npz")
small_d = np.load(DATA_DIR / "sweep_nash_social_dilemma_small_d.npz")
T_values = sweep["T_values"]

# The shortest distances come from the sweep that adds rates close to zero.
payoff = {}
replace = np.isin(sweep["d_values"], small_d["d_values"])
for key in [
    "worst_local_payoff",
    "best_local_payoff",
    "worst_nash_payoff",
    "global_optimum_payoff",
]:
    payoff[key] = sweep[key].copy()
    payoff[key][:, replace] = small_d[key]

rows = [
    (
        payoff["worst_local_payoff"] / payoff["global_optimum_payoff"],
        r"$\Pi_{\mathrm{worst\,local}}/\Pi^*$",
        r"\shortstack{Worst locally stable\\as \% of global optimum}",
    ),
    (
        payoff["worst_local_payoff"] / payoff["best_local_payoff"],
        r"$\Pi_{\mathrm{worst\,local}}/\Pi_{\mathrm{best\,local}}$",
        r"\shortstack{Worst locally stable\\as \% of best locally stable}",
    ),
    (
        payoff["worst_nash_payoff"] / payoff["global_optimum_payoff"],
        r"$\Pi_{\mathrm{worst\,Nash}}/\Pi^*$",
        r"\shortstack{Worst Nash\\as \% of global optimum}",
    ),
]

R_edges = centers_to_edges(sweep["R_values"])
d_edges = centers_to_edges(sweep["d_values"])
fig, axes = plt.subplots(
    len(rows),
    len(T_values),
    figsize=(9.75, 9),
    sharex=True,
    sharey=True,
    layout="constrained",
)
for row, (ratio, colorbar_label, row_label) in enumerate(rows):
    norm = plt.Normalize(np.nanmin(ratio), np.nanmax(ratio))
    for t, T in enumerate(T_values):
        image = axes[row, t].pcolormesh(
            R_edges,
            d_edges,
            ratio[t],
            cmap="viridis",
            norm=norm,
            antialiased=False,
            rasterized=True,
        )
    colorbar = fig.colorbar(image, ax=axes[row], fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)
    axes[row, 0].set_ylabel(r"Distance $d$")
    fig.text(
        -0.025,
        1 - (row + 0.5) / len(rows),
        row_label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=plt.rcParams["axes.titlesize"],
    )

# The bottom row's scale starts well above zero; label it in steps of 0.1.
colorbar.set_ticks(np.arange(np.ceil(norm.vmin * 10) / 10, norm.vmax + 1e-9, 0.1))
colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

for t, T in enumerate(T_values):
    axes[0, t].set_title(rf"$\gamma = 1 - 1/{T}$")
    axes[-1, t].set_xlabel(r"Reward ratio $R$")
for ax in axes.ravel():
    ax.set_xticks([5, 10, 15, 20])
    ax.set_yticks([5, 10, 15, 20, 25, 30])

save_figure(fig, "worst_equilibrium_fitness_percentages.pdf")
