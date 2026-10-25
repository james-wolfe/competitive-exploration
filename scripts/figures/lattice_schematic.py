"""Figure 1: a lone agent's step probabilities and Q-values on the lattice, before and
after it discovers the distant reward R at site d.

Gray cells are Q-values not yet updated: never visited (A) or not visited since R was
found (B). The dashed arrow is the step against the greedy direction.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from competitive_exploration.plotting import save_figure, use_paper_style

use_paper_style()
plt.rcParams.update({"font.size": 8, "savefig.pad_inches": 0.02})
SMALL = 7

REWARD_FILL, STALE_FILL = "#cfe3d4", "#e4e4e4"
GAP = 0.55  # space between the environment and Q-value rows
SITES = ["0", "1", "2", None, "d-2", "d-1", "d"]
XS = [0, 1, 2, 2.75, 3.5, 4.5, 5.5]  # column centers; the ellipsis column is narrow
REWARDS = ["+1", "", "", None, "", "", "+R"]
X_LEFT = -1.45
F = r"\displaystyle\frac{%s}{1-\gamma}"

PANELS = [
    (
        "A",
        "Before discovery",
        1,
        (r"1-\varepsilon/2", r"\varepsilon/2"),
        [F % 1, F % r"\gamma", F % r"\gamma^2", None, "0", "0", "0"],
        [1, 1, 1, None, 0, 0, 0],
    ),
    (
        "B",
        "After discovery",
        5,
        (r"\varepsilon/2", r"1-\varepsilon/2"),
        [
            F % 1,
            F % r"\gamma",
            F % r"\gamma^2",
            None,
            F % r"\gamma^{d-2}",
            F % r"\gamma R",
            F % "R",
        ],
        [0, 0, 0, None, 0, 1, 1],
    ),
]


def draw_panel(ax, label, title, agent, probs, q, updated):
    y_env, y_site, y_q = 0.0, -GAP / 2, -1.0 - GAP
    for x, site, reward, q_value, is_updated in zip(XS, SITES, REWARDS, q, updated):
        if site is None:
            for y in (y_env + 0.5, y_site, y_q + 0.5):
                ax.text(x, y, r"$\cdots$", ha="center", va="center")
            continue
        ax.add_patch(
            Rectangle(
                (x - 0.5, y_env),
                1,
                1,
                edgecolor="black",
                linewidth=0.6,
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
                linewidth=0.6,
                facecolor="white" if is_updated else STALE_FILL,
            )
        )
        ax.text(x, y_q + 0.5, f"${q_value}$", ha="center", va="center")

    x_agent = XS[agent]
    ax.plot(x_agent, y_env + 0.5, "o", color="black", markersize=5)
    for dx, p in zip((-1, 1), probs):
        x0, x1, y = x_agent + 0.2 * dx, x_agent + 0.95 * dx, y_env + 1.2
        ax.plot(
            [x0, x1 - 0.08 * dx],
            [y, y],
            color="black",
            linewidth=0.7,
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
                "linewidth": 0.7,
                "mutation_scale": 7,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        ax.text(
            x_agent + 0.575 * dx,
            y_env + 1.47,
            f"${p}$",
            ha="center",
            va="center",
            fontsize=SMALL,
        )

    ax.text(-0.65, y_site, "Site", ha="right", va="center")
    ax.text(-0.65, y_q + 0.5, r"$Q$-value", ha="right", va="center")
    ax.text(X_LEFT, y_env + 2.0, rf"\textbf{{{label}}}\enspace {title}", va="center")
    ax.set_xlim(X_LEFT, XS[-1] + 0.55)
    ax.set_ylim(y_q - 0.05, y_env + 2.2)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(2, 1, figsize=(4.03, 3.75))
for ax, panel in zip(axes, PANELS):
    draw_panel(ax, *panel)
fig.tight_layout(h_pad=1.2)

save_figure(fig, "lattice_schematic.pdf")
