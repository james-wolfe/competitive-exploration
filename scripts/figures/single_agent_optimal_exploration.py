"""Figure 2: (A) a lone agent's discounted reward against its exploration
rate, (B) the optimal rate against distance and against reward ratio, and (C) the optimal
rate over reward ratio and distance for three discount factors."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    CM,
    PNAS_WIDTHS,
    add_colored_curve,
    add_row_label,
    centers_to_edges,
    save_figure,
    use_paper_style,
)

use_paper_style(final_size=True)
rewards_data = np.load(DATA_DIR / "single_agent_rewards.npz")
opt_data = np.load(DATA_DIR / "opt_eps_single.npz")

fig = plt.figure(figsize=(PNAS_WIDTHS["double"], 14 * CM), layout="constrained")
top, panel_c = fig.subfigures(2, 1, height_ratios=[1.22, 1])
panel_a, panel_b = top.subfigures(1, 2, width_ratios=[0.54, 0.46])

for subfig, letter in [(panel_a, "A"), (panel_b, "B"), (panel_c, "C")]:
    subfig.text(0, 1, rf"\textbf{{{letter}}}", ha="left", va="top", fontsize=8)

# A
rewards, eps_grid = rewards_data["rewards"], rewards_data["eps_grid"]
d_values, R_values = rewards_data["d_values"], rewards_data["R_values"]
total = rewards.sum(axis=-1)
pad = 0.05 * (total.max() - total.min())

axes = panel_a.subplots(len(d_values), len(R_values), sharex=True, sharey=True)
for a, d in enumerate(d_values):
    for r, R in enumerate(R_values):
        ax = axes[a, r]
        y = total[a, r]
        add_colored_curve(ax, eps_grid, y, rewards[a, r, :, 1] / y, linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(total.min() - pad, total.max() + pad)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
        ax.set_yticks([5000, 10000, 15000])
        ax.grid(True)
    axes[a, 0].set_ylabel("Discounted reward")
    add_row_label(axes[a, 0], rf"$d = {d}$", x=-0.58)
for r, R in enumerate(R_values):
    axes[0, r].set_title(rf"$R = {R:g}$")
    axes[-1, r].set_xlabel(r"Exploration rate $\varepsilon$")
colorbar = panel_a.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 1), "coolwarm_r"),
    ax=axes,
    fraction=0.045,
    pad=0.03,
    ticks=[0, 0.25, 0.5, 0.75, 1],
)
colorbar.set_label(r"Fraction of discounted reward from $d$")

# B
opt_eps = opt_data["opt_eps"][list(opt_data["T_values"]).index(5000)]
d_values, R_values = opt_data["d_values"], opt_data["R_values"]
r = np.argmin(np.abs(R_values - 10))
a = list(d_values).index(10)

axes = panel_b.subplots(2, 1)
axes[0].plot(d_values, opt_eps[:, r], ".", markersize=5)
axes[0].set_xlabel(r"Distance $d$")
axes[0].set_title(rf"$R = {R_values[r]:.2f}$")
axes[1].scatter(R_values, opt_eps[a], s=2)
axes[1].set_xticks([5, 10, 15, 20])
axes[1].set_xlabel(r"Reward ratio $R$")
axes[1].set_title(rf"$d = {d_values[a]}$")
for ax in axes:
    ax.grid(True)
panel_b.supylabel(r"Optimal exploration rate $\varepsilon^*$", fontsize=7)

# C
R_edges = centers_to_edges(opt_data["R_values"])
d_edges = centers_to_edges(opt_data["d_values"])
axes = panel_c.subplots(1, len(opt_data["T_values"]), sharey=True)
for ax, T, opt_eps in zip(axes, opt_data["T_values"], opt_data["opt_eps"]):
    image = ax.pcolormesh(
        R_edges, d_edges, opt_eps, cmap="viridis", vmin=0, vmax=1, rasterized=True
    )
    ax.set_title(rf"$\gamma = 1 - 1/{T}$")
    ax.set_xlabel(r"Reward ratio $R$")
axes[0].set_ylabel(r"Distance $d$")
colorbar = panel_c.colorbar(image, ax=axes, pad=0.01, shrink=0.95)
colorbar.set_label(r"Optimal exploration rate $\varepsilon^*$")

save_figure(fig, "single_agent_optimal_exploration.pdf")
