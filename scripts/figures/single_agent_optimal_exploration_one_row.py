"""Figure 2: (A) a lone agent's discounted reward against its exploration rate, and
(B) the optimal rate over reward ratio and distance for two discount factors. SI
Appendix shows (B) with the shortest horizon as well."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    CM,
    PNAS_WIDTHS,
    add_colored_curve,
    centers_to_edges,
    save_figure,
    use_paper_style,
)

use_paper_style(final_size=True)
rewards_data = np.load(DATA_DIR / "single_agent_rewards.npz")
opt_data = np.load(DATA_DIR / "opt_eps_single.npz")
T_SHOWN = [5000, 100000]

fig = plt.figure(figsize=(PNAS_WIDTHS["double"], 6.3 * CM), layout="constrained")
panel_a, panel_b = fig.subfigures(1, 2, width_ratios=[0.47, 0.53], wspace=0.02)

for subfig, letter in [(panel_a, "A"), (panel_b, "B")]:
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
    axes[a, 0].set_ylabel(rf"$d = {d}$" "\nDiscounted reward")
for r, R in enumerate(R_values):
    axes[0, r].set_title(rf"$R = {R:g}$")
    axes[-1, r].set_xlabel(r"Exploration rate $\varepsilon$")
colorbar = panel_a.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 1), "coolwarm_r"),
    ax=axes,
    fraction=0.07,
    pad=0.04,
    ticks=[0, 0.25, 0.5, 0.75, 1],
)
colorbar.set_label(r"Fraction of discounted reward from $d$")

# B
R_edges = centers_to_edges(opt_data["R_values"])
d_edges = centers_to_edges(opt_data["d_values"])
T_values = list(opt_data["T_values"])
axes = panel_b.subplots(1, len(T_SHOWN), sharey=True)
for ax, T in zip(axes, T_SHOWN):
    image = ax.pcolormesh(
        R_edges,
        d_edges,
        opt_data["opt_eps"][T_values.index(T)],
        cmap="viridis",
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    ax.set_title(rf"$\gamma = 1 - 1/{T}$")
    ax.set_xlabel(r"Reward ratio $R$")
    ax.set_xticks([5, 10, 15, 20])
axes[0].set_ylabel(r"Distance $d$")
colorbar = panel_b.colorbar(image, ax=axes, fraction=0.07, pad=0.04)
colorbar.set_label(r"Optimal exploration rate $\varepsilon^*$")

save_figure(fig, "single_agent_optimal_exploration_one_row.pdf")
