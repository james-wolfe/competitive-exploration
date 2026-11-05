"""Figure 1: a lone agent's step probabilities and Q-values on the lattice, before and
after it discovers the distant reward R at site d.

Gray cells are Q-values not yet updated: never visited (A) or not visited since R was
found (B). The dashed arrow is the step against the greedy direction.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from competitive_exploration.plotting import PNAS_WIDTHS, save_figure, use_paper_style

use_paper_style(final_size=True)
SMALL = 6

REWARD_FILL, STALE_FILL = "#cfe3d4", "#e4e4e4"
GAP = 0.5  # space between the environment and Q-value rows
SITES = ["0", "1", "2", "3", None, "d-2", "d-1", "d"]
XS = [0, 1, 2, 3, 3.75, 4.5, 5.5, 6.5]  # column centers; the ellipsis column is narrow
REWARDS = ["+1", "", "", "", None, "", "", "+R"]
X_LEFT, X_RIGHT = -1.77, XS[-1] + 0.56
Y_BOTTOM, Y_TOP = -1.0 - GAP - 0.04, 1.62  # the Q-value row to the step probabilities
F = r"\displaystyle\frac{%s}{1-\gamma}"

PANELS = [
    (
        "A",
        "Before discovery",
        3,
        (r"1-\varepsilon/2", r"\varepsilon/2"),
        [F % 1, F % r"\gamma", F % r"\gamma^2", F % r"\gamma^3", None, "0", "0", "0"],
        [1, 1, 1, 1, None, 0, 0, 0],
    ),
    (
        "B",
        "After discovery",
        6,
        (r"\varepsilon/2", r"1-\varepsilon/2"),
        [
            F % 1,
            F % r"\gamma",
            F % r"\gamma^2",
            F % r"\gamma^3",
            None,
            F % r"\gamma^{d-2}",
            F % r"\gamma R",
            F % "R",
        ],
        [0, 0, 0, 0, None, 0, 1, 1],
    ),
]


def draw_panel(ax, label, title, agent, probs, q, updated):
    y_env, y_site, y_q = 0.0, -GAP / 2, -1.0 - GAP
    y_probs = y_env + 1.4
    for x, site, reward, q_value, is_updated in zip(XS, SITES, REWARDS, q, updated):
        if site is None:
            for y in (y_env + 0.5, y_site, y_q + 0.5):
                ax.text(x, y, r"$\cdots$", ha="center", va="center", fontsize=SMALL)
            continue
        ax.add_patch(
            Rectangle(
                (x - 0.5, y_env),
                1,
                1,
                edgecolor="black",
                linewidth=0.5,
                facecolor=REWARD_FILL if reward else "white",
            )
        )
        if reward:
            ax.text(x, y_env + 0.5, f"${reward}$", ha="center", va="center")
        ax.text(x, y_site, f"${site}$", ha="center", va="center", fontsize=SMALL)
        ax.add_patch(
            Rectangle(
                (x - 0.5, y_q),
                1,
                1,
                edgecolor="black",
                linewidth=0.5,
                facecolor="white" if is_updated else STALE_FILL,
            )
        )
        ax.text(x, y_q + 0.5, f"${q_value}$", ha="center", va="center", fontsize=SMALL)

    x_agent = XS[agent]
    ax.plot(x_agent, y_env + 0.5, "o", color="black", markersize=4)
    for dx, p in zip((-1, 1), probs):
        x0, x1, y = x_agent + 0.2 * dx, x_agent + 0.95 * dx, y_env + 1.15
        ax.plot(
            [x0, x1 - 0.08 * dx],
            [y, y],
            color="black",
            linewidth=0.6,
            linestyle="-" if p.startswith("1") else (0, (2.5, 1.5)),
            solid_capstyle="butt",
        )
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x1 - 0.1 * dx, y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "linewidth": 0.6,
                "mutation_scale": 6,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        ax.text(
            x_agent + 0.575 * dx, y_probs, f"${p}$", ha="center", va="center", fontsize=SMALL
        )

    ax.text(-0.62, y_site, "Site", ha="right", va="center")
    ax.text(-0.62, y_q + 0.5, r"$Q$-value", ha="right", va="center")
    # On the line of the step probabilities, which lie to its right.
    ax.text(X_LEFT + 0.03, y_probs, rf"\textbf{{{label}}}\enspace {title}", va="center")
    ax.set_xlim(X_LEFT, X_RIGHT)
    ax.set_ylim(Y_BOTTOM, Y_TOP)
    ax.axis("off")


# One column wide, at the printed size, with equal scales on both axes.
width = PNAS_WIDTHS["single"]
unit = width / (X_RIGHT - X_LEFT)  # inches per cell
panel_height, gap = unit * (Y_TOP - Y_BOTTOM), 0.1
height = 2 * panel_height + gap
fig = plt.figure(figsize=(width, height))
for i, panel in enumerate(PANELS):
    bottom = (1 - i) * (panel_height + gap)
    ax = fig.add_axes([0, bottom / height, 1, panel_height / height])
    draw_panel(ax, *panel)

save_figure(fig, "lattice_schematic.pdf")
