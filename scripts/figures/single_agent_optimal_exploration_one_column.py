"""Figure 2: (A) a lone agent's discounted reward against its exploration rate, and below
it (B) the optimal rate over reward ratio and distance for two discount factors. SI
Appendix shows (B) with the shortest horizon as well."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
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

# One column wide; layout in inches, from the bottom up. A and B share their columns and
# the position and width of their colorbars; the panels of A are wider than tall, those
# of B square.
width = PNAS_WIDTHS["single"]
colorbar_space, colorbar_pad, colorbar_width = 0.62, 0.08, 0.07
left, gap = 0.6, 0.14
panel_width = (width - left - gap - colorbar_space) / 2
b_bottom, b_size = 0.33, panel_width
a_width, a_height, a_gap = panel_width, 0.82, 0.08
a_bottom = b_bottom + b_size + 0.74
height = a_bottom + 2 * a_height + a_gap + 0.24
fig = plt.figure(figsize=(width, height))


def box(x, y, w, h):
    return [x / width, y / height, w / width, h / height]


# A
rewards, eps_grid = rewards_data["rewards"], rewards_data["eps_grid"]
d_values, R_values = rewards_data["d_values"], rewards_data["R_values"]
total = rewards.sum(axis=-1)
pad = 0.05 * (total.max() - total.min())

axes = np.empty((len(d_values), len(R_values)), dtype=object)
for a in range(len(d_values)):
    for r in range(len(R_values)):
        y = a_bottom + (len(d_values) - 1 - a) * (a_height + a_gap)
        axes[a, r] = fig.add_axes(box(left + r * (a_width + gap), y, a_width, a_height))
for a, d in enumerate(d_values):
    for r, R in enumerate(R_values):
        ax = axes[a, r]
        y = total[a, r]
        add_colored_curve(ax, eps_grid, y, rewards[a, r, :, 1] / y, linewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(total.min() - pad, total.max() + pad)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
        ax.set_yticks([5000, 10000, 15000], ["5", "10", "15"])
        ax.grid(True)
        if a < len(d_values) - 1:
            ax.tick_params(labelbottom=False)
        if r > 0:
            ax.tick_params(labelleft=False)
    axes[a, 0].set_ylabel(rf"$d = {d}$")
for r, R in enumerate(R_values):
    axes[0, r].set_title(rf"$R = {R:g}$")
    axes[-1, r].set_xlabel(r"Exploration rate $\varepsilon$")
colorbar = fig.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 1), "coolwarm_r"),
    cax=fig.add_axes(box(width - colorbar_space + colorbar_pad, a_bottom, colorbar_width,
                         2 * a_height + a_gap)),
    ticks=[0, 0.5, 1],
)
colorbar.set_label("Fraction of discounted\n" r"reward from $d$")
fig.text(
    0.02 / width,
    (a_bottom + a_height + a_gap / 2) / height,
    r"Discounted reward ($\times 10^3$)",
    rotation=90,
    ha="left",
    va="center",
)

# B
R_edges = centers_to_edges(opt_data["R_values"])
d_edges = centers_to_edges(opt_data["d_values"])
T_values = list(opt_data["T_values"])
for t, T in enumerate(T_SHOWN):
    ax = fig.add_axes(box(left + t * (b_size + gap), b_bottom, b_size, b_size))
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
    if t == 0:
        ax.set_ylabel(r"Distance $d$")
    else:
        ax.tick_params(labelleft=False)
colorbar = fig.colorbar(
    image,
    cax=fig.add_axes(box(width - colorbar_space + colorbar_pad, b_bottom, colorbar_width,
                         b_size)),
    ticks=[0, 0.5, 1],
)
colorbar.set_label("Optimal exploration\n" r"rate $\varepsilon^*$")

for letter, y in [("A", height - 0.02), ("B", b_bottom + b_size + 0.26)]:
    fig.text(0.02 / width, y / height, rf"\textbf{{{letter}}}", ha="left", va="top",
             fontsize=8)

save_figure(fig, "single_agent_optimal_exploration_one_column.pdf")
