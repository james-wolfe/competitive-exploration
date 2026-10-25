"""Figure 3: (A) player 2's payoff, best response, and the Nash equilibria at
d = 4, R = 2.1, and (B) player 2's payoff against its own rate at three rates of
player 1."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.equilibria import best_response, classify_points, find_nash
from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import (
    PNAS_WIDTHS,
    add_colored_curve,
    save_figure,
    use_paper_style,
)

use_paper_style(final_size=True)
data = np.load(DATA_DIR / "two_agent_payoffs.npz")
Z_d = data["Z_d"]
Z = data["Z_0"] + Z_d
eps_grid = data["eps_grid"]

# Player 1 never explores, explores at the symmetric Nash rate, or explores at the
# anti-coordination Nash rate.
symmetric, anti_coordination, _, _ = classify_points(find_nash(Z)[0])
columns = [0, symmetric[0][1], max(anti_coordination[0])]

# Panels placed in inches, so that all four share the same top and bottom edges.
HEIGHT, BOTTOM, TOP = 1.35, 0.36, 0.19
A_LEFT, B_LEFT = 0.45, 3.0
B_WIDTH, B_GAP = 1.06, 0.14
BAR_GAP, BAR = 0.07, 0.08
W, H = PNAS_WIDTHS["double"], BOTTOM + HEIGHT + TOP
fig = plt.figure(figsize=(W, H))


def place(left, width):
    return fig.add_axes([left / W, BOTTOM / H, width / W, HEIGHT / H])


for x, letter in [(0.0, "A"), (B_LEFT - 0.5, "B")]:
    y = (BOTTOM + HEIGHT + 0.085) / H
    fig.text(x / W, y, rf"\textbf{{{letter}}}", ha="left", va="baseline", fontsize=8)

# A
ax = place(A_LEFT, HEIGHT)
image = ax.imshow(Z, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
cax = place(A_LEFT + HEIGHT + BAR_GAP, BAR)
fig.colorbar(image, cax=cax).set_label("P2 expected reward")
ax.plot(
    eps_grid,
    eps_grid[best_response(Z)],
    color="black",
    linewidth=1.2,
    clip_on=False,
    zorder=10,
    label="P2 best response",
)
i, j = np.array(find_nash(Z)[0]).T
ax.scatter(
    eps_grid[j],
    eps_grid[i],
    s=22,
    facecolors="white",
    edgecolors="black",
    linewidth=1,
    clip_on=False,
    zorder=11,
    label="Nash equilibrium",
)
ax.set_xlabel(r"P1 exploration rate $\varepsilon_1$")
ax.set_ylabel(r"P2 exploration rate $\varepsilon_2$")
ax.legend(
    loc="upper right",
    fontsize=6,
    borderpad=0.3,
    labelspacing=0.25,
    handlelength=1.5,
    borderaxespad=0.3,
)

# B
axes = [place(B_LEFT + k * (B_WIDTH + B_GAP), B_WIDTH) for k in range(3)]
y_range = Z[:, columns].max() - Z[:, columns].min()
for ax, j in zip(axes, columns):
    add_colored_curve(ax, eps_grid, Z[:, j], Z_d[:, j] / Z[:, j], linewidth=1.5)
    best = np.argmax(Z[:, j])
    ax.scatter(
        eps_grid[best],
        Z[best, j],
        s=18,
        facecolors="white",
        edgecolors="black",
        linewidth=0.9,
        zorder=5,
        clip_on=False,
    )
    ax.set_title(rf"$\varepsilon_1 = {eps_grid[j]:.2g}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(Z[:, columns].min() - 0.05 * y_range, Z[:, columns].max() + 0.05 * y_range)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
    ax.grid(True)
for ax in axes[1:]:
    ax.tick_params(labelleft=False)
axes[1].set_xlabel(r"P2 exploration rate $\varepsilon_2$")
axes[0].set_ylabel("P2 expected reward")
cax = place(B_LEFT + 3 * B_WIDTH + 2 * B_GAP + BAR_GAP, BAR)
fig.colorbar(
    plt.cm.ScalarMappable(plt.Normalize(0, 1), "coolwarm_r"),
    cax=cax,
    ticks=[0, 0.25, 0.5, 0.75, 1],
).set_label(r"Fraction of P2 reward from $d$")

save_figure(fig, "payoff_matrix_and_slices.pdf")
